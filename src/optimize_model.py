#!/usr/bin/env python3
# optimize_model.py - Optimize trained model for mobile deployment

import copy
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms_v2
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Import executorch
try:
    import executorch
    from executorch.exir import Inspector
    EXECUTORCH_AVAILABLE = True
    print("ExecuTorch SDK available.")
except ImportError:
    print("ExecuTorch SDK not available. Install with: pip install executorch-sdk")
    EXECUTORCH_AVAILABLE = False

try:
    from train_phase2local import create_model, EmotionDataset, EMOTION_CLASSES
except ImportError as e:
    print(f"Error importing from train_phase2local: {e}")
    raise ImportError("Critical components from train_phase2local.py could not be imported.") from e

def get_model_size(model_path):
    """Return the size of the model in MB"""
    if not os.path.exists(model_path):
        return 0
    return os.path.getsize(model_path) / (1024 * 1024)

def measure_inference_time(model, input_tensor, device, num_runs=30):
    """Measure inference time over multiple runs"""
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    
    # Timed runs
    latency_ms_list = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latency_ms_list.append(latency_ms)
    
    return {
        'avg_ms': sum(latency_ms_list) / len(latency_ms_list),
        'min_ms': min(latency_ms_list),
        'max_ms': max(latency_ms_list),
        'raw_measurements_ms': latency_ms_list
    }

def get_available_qengine():
    """Get the best available quantization engine"""
    available_engines = torch.backends.quantized.supported_engines
    print(f"Available quantization engines: {available_engines}")
    
    # Prefer FBGEMM for x86 and QNNPACK for ARM
    if 'fbgemm' in available_engines:
        return 'fbgemm'
    elif 'qnnpack' in available_engines:
        return 'qnnpack'
    elif 'onednn' in available_engines:
        return 'onednn'
    else:
        return available_engines[0] if available_engines else None

def quantize_model(model_fp32, dataloader, device):
    """Quantize the model using dynamic int8 quantization"""
    print("Preparing model for quantization...")
    # Create a copy to avoid modifying the original model
    model_copy = copy.deepcopy(model_fp32)
    model_copy = model_copy.to("cpu").eval()
    
    try:
        # Use dynamic quantization which is more widely supported
        print("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear, nn.Conv2d},  # Quantize both linear and conv layers
            dtype=torch.qint8
        )
        print("Successfully quantized model (dynamic INT8)")
        return quantized_model
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        print("Falling back to original model...")
        return model_copy

def evaluate_model(model, dataloader, device, is_quantized=False):
    """Evaluate model and return metrics"""
    model.eval()
    eval_device = 'cpu' if is_quantized else device
    model = model.to(eval_device)
    
    all_preds = []
    all_labels = []
    total_batches = len(dataloader)
    
    print(f"Evaluating model on {total_batches} batches...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"Processing batch {batch_idx}/{total_batches}")
                
            inputs = inputs.to(eval_device)
            labels = labels.to(eval_device)
            try:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Warning: Error during evaluation: {e}")
                continue
    
    if not all_preds:  # If no predictions were made
        return {
            'accuracy': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0
        }
    
    print("Calculating metrics...")
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }

def export_to_torchscript_lite(model, sample_input, output_path):
    """Export model to TorchScript Lite format for Android deployment"""
    print(f"Exporting model to TorchScript Lite format at {output_path}...")
    try:
        # Ensure model is on CPU and in eval mode
        model.eval()
        model = model.to("cpu")
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Save the model
        traced_model.save(output_path)
        
        print(f"Successfully exported model to {output_path}")
        return True, output_path
    except Exception as e:
        print(f"Error exporting to TorchScript Lite: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def verify_torchscript_model(ptl_model_path, sample_input):
    """Verify TorchScript Lite model"""
    if not os.path.exists(ptl_model_path):
        return False, "Model file not found", None, float('inf')
    
    try:
        # Load the model
        model = torch.jit.load(ptl_model_path)
        model.eval()
        
        # Test inference
        with torch.no_grad():
            _ = model(sample_input)
        return True, "Success", None, 0.0
    except Exception as e:
        return False, str(e), None, float('inf')

def load_trained_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    num_classes = checkpoint.get('num_classes', len(EMOTION_CLASSES))
    
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device and ensure all parameters are on the correct device
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    
    model.eval()
    
    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    
    return model, model_name

def run_ablation_study(model_path, data_dir, label_file, output_dir, num_experiments=30):
    """Run ablation study comparing original and quantized models"""
    print(f"Starting ablation study with {num_experiments} experiments...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
       # Load custom dataset
    print("Loading custom dataset...")
    # Use absolute path for custom dataset
    custom_data_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'custom', 'processed')
    custom_label_file = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'custom', 'labels.csv')
    
    print(f"Looking for custom dataset at:")
    print(f"  Data directory: {custom_data_dir}")
    print(f"  Label file: {custom_label_file}")
    
    # Load both datasets
    print("Loading RAF-DB dataset...")
    raf_labels_df = pd.read_csv(label_file)
    val_transform = transforms_v2.Compose([
        transforms_v2.ToDtype(torch.float32),
        transforms_v2.Lambda(lambda x: x / 255.0),
        transforms_v2.Resize((256, 256), antialias=True),
        transforms_v2.CenterCrop(224),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raf_dataset = EmotionDataset(data_dir, raf_labels_df, transform=val_transform)
    
 
    
    if not os.path.exists(custom_data_dir) or not os.path.exists(custom_label_file):
        print("Warning: Custom dataset not found. Proceeding with only RAF-DB dataset.")
        custom_dataset = None
    else:
        custom_labels_df = pd.read_csv(custom_label_file)
        custom_dataset = EmotionDataset(custom_data_dir, custom_labels_df, transform=val_transform)
        print(f"Loaded {len(custom_dataset)} images from custom dataset")
    
    # Load original model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    original_model, _ = load_trained_model(model_path, device)
    
    # Ensure original model is on the correct device
    original_model = original_model.to(device)
    
    # Quantize model once
    print("\nQuantizing model...")
    quantized_model = quantize_model(original_model, None, device)  # No need for dataloader in dynamic quantization
    quantized_model = quantized_model.to('cpu')  # Ensure quantized model is on CPU
    
    # Save quantized model
    quantized_path = os.path.join(output_dir, 'quantized_model.pt')
    torch.save(quantized_model.state_dict(), quantized_path)
    
    # Get model sizes once
    original_size = get_model_size(model_path)
    quantized_size = get_model_size(quantized_path)
    
    # Create dataloaders for both datasets
    raf_dataloader = DataLoader(raf_dataset, batch_size=5, shuffle=True, num_workers=4)
    raf_iter = iter(raf_dataloader)
    
    if custom_dataset:
        custom_dataloader = DataLoader(custom_dataset, batch_size=5, shuffle=True, num_workers=4)
        custom_iter = iter(custom_dataloader)
    
    # Run experiments comparing original and quantized models
    print(f"\nRunning {num_experiments} experiments...")
    results = []
    
    for exp in range(num_experiments):
        print(f"\nExperiment {exp + 1}/{num_experiments}")
        
        # Alternate between RAF-DB and custom dataset
        use_custom = custom_dataset and exp % 2 == 1  # Use custom dataset for odd-numbered experiments
        
        try:
            if use_custom:
                images, labels = next(custom_iter)
                dataset_name = "Custom"
            else:
                images, labels = next(raf_iter)
                dataset_name = "RAF-DB"
        except StopIteration:
            # If we run out of images, create a new iterator
            if use_custom:
                custom_iter = iter(custom_dataloader)
                images, labels = next(custom_iter)
            else:
                raf_iter = iter(raf_dataloader)
                images, labels = next(raf_iter)
        
        # Evaluate both models on this batch of images
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            # Original model predictions (on GPU if available)
            images_gpu = images.to(device)
            original_outputs = original_model(images_gpu)
            original_preds = torch.argmax(original_outputs, dim=1)
            original_correct = (original_preds == labels.to(device)).float()
            
            # Quantized model predictions (on CPU)
            images_cpu = images.cpu()
            quantized_outputs = quantized_model(images_cpu)
            quantized_preds = torch.argmax(quantized_outputs, dim=1)
            quantized_correct = (quantized_preds == labels).float()
        
        # Measure latency for both models using the same batch
        original_latency = measure_inference_time(original_model, images_gpu, device)
        quantized_latency = measure_inference_time(quantized_model, images_cpu, 'cpu')
        
        # Store results for each image in the batch
        batch_results = []
        for i in range(len(images)):
            batch_results.append({
                'image_idx': i,
                'original': {
                    'prediction': original_preds[i].item(),
                    'true_label': labels[i].item(),
                    'correct': original_correct[i].item()
                },
                'quantized': {
                    'prediction': quantized_preds[i].item(),
                    'true_label': labels[i].item(),
                    'correct': quantized_correct[i].item()
                }
            })
        
        results.append({
            'experiment': exp + 1,
            'dataset': dataset_name,
            'original': {
                'size_mb': original_size,
                'latency_ms': original_latency['avg_ms'],
                'accuracy': float(original_correct.mean().item()),  # Average accuracy across batch
                'batch_results': batch_results
            },
            'quantized': {
                'size_mb': quantized_size,
                'latency_ms': quantized_latency['avg_ms'],
                'accuracy': float(quantized_correct.mean().item()),  # Average accuracy across batch
                'batch_results': batch_results
            }
        })
        
        print(f"Experiment {exp + 1} Results (Dataset: {dataset_name}):")
        print(f"Original - Batch Accuracy: {float(original_correct.mean().item()):.4f}, Latency: {original_latency['avg_ms']:.2f}ms")
        print(f"Quantized - Batch Accuracy: {float(quantized_correct.mean().item()):.4f}, Latency: {quantized_latency['avg_ms']:.2f}ms")
        
        # Print individual image results
        for i, img_result in enumerate(batch_results):
            print(f"  Image {i + 1}:")
            print(f"    True Label: {img_result['original']['true_label']}")
            print(f"    Original - Prediction: {img_result['original']['prediction']}, Correct: {img_result['original']['correct']}")
            print(f"    Quantized - Prediction: {img_result['quantized']['prediction']}, Correct: {img_result['quantized']['correct']}")
    
    # Save results
    report_path = os.path.join(output_dir, 'ablation_study.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate overall accuracy and per-dataset accuracy
    all_original_accuracies = [r['original']['accuracy'] for r in results]
    all_quantized_accuracies = [r['quantized']['accuracy'] for r in results]
    
    raf_original_accuracies = [r['original']['accuracy'] for r in results if r['dataset'] == 'RAF-DB']
    raf_quantized_accuracies = [r['quantized']['accuracy'] for r in results if r['dataset'] == 'RAF-DB']
    
    custom_original_accuracies = [r['original']['accuracy'] for r in results if r['dataset'] == 'Custom']
    custom_quantized_accuracies = [r['quantized']['accuracy'] for r in results if r['dataset'] == 'Custom']
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("Original Model:")
    print(f"  Size: {original_size:.2f} MB")
    print(f"  Average Latency: {np.mean([r['original']['latency_ms'] for r in results]):.2f} ms")
    print(f"  Overall Accuracy: {np.mean(all_original_accuracies):.4f}")
    if raf_original_accuracies:
        print(f"  RAF-DB Accuracy: {np.mean(raf_original_accuracies):.4f}")
    if custom_original_accuracies:
        print(f"  Custom Dataset Accuracy: {np.mean(custom_original_accuracies):.4f}")
    
    print("\nQuantized Model:")
    print(f"  Size: {quantized_size:.2f} MB")
    print(f"  Average Latency: {np.mean([r['quantized']['latency_ms'] for r in results]):.2f} ms")
    print(f"  Overall Accuracy: {np.mean(all_quantized_accuracies):.4f}")
    if raf_quantized_accuracies:
        print(f"  RAF-DB Accuracy: {np.mean(raf_quantized_accuracies):.4f}")
    if custom_quantized_accuracies:
        print(f"  Custom Dataset Accuracy: {np.mean(custom_quantized_accuracies):.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Optimize model for mobile deployment')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label file')
    parser.add_argument('--output_dir', type=str, default='./outputs/final', help='Output directory')
    parser.add_argument('--num_experiments', type=int, default=30, help='Number of experiments for ablation study')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.exists(args.label_file):
        raise FileNotFoundError(f"Label file not found: {args.label_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run ablation study
    results = run_ablation_study(
        args.model_path,
        args.data_dir,
        args.label_file,
        args.output_dir,
        args.num_experiments
    )

if __name__ == "__main__":
    main()