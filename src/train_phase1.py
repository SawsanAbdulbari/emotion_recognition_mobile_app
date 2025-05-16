#!/usr/bin/env python3

import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import OmegaConf for config handling
from omegaconf import OmegaConf

# Set up tensorboard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

# Define emotion classes
EMOTION_CLASSES = ['anger', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class EmotionDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images
            labels_file (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        
        # Create label mapping (string to int)
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_CLASSES)}
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Read image
        try:
            image = read_image(img_path)
            # Convert to 3 channels if grayscale
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            image = torch.zeros((3, 224, 224), dtype=torch.uint8)
            
        # Convert label string to index
        label_str = self.labels_df.iloc[idx, 1]
        label = self.label_to_idx.get(label_str, 0)  # Default to 0 if not found
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_model(model_name, num_classes=6, pretrained=True):
    """Create a model with pretrained weights and custom head."""
    
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'mobilevit_xxs':
        # For this model, we need to install timm library
        try:
            import timm
            model = timm.create_model('mobilevit_xxs', pretrained=pretrained)
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes)
        except ImportError:
            print("timm library not installed. Install with: pip install timm")
            print("Falling back to EfficientNet-B0")
            model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        print(f"Model {model_name} not supported. Using EfficientNet-B0")
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, log_dir=None):
    """Train the model and return best model weights."""
    since = time.time()
    
    # Initialize tensorboard writer if available
    writer = None
    if TENSORBOARD_AVAILABLE and log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize best model parameters
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Clear cache to prevent OOM
                torch.cuda.empty_cache()
            
            # Compute epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log to tensorboard
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = model.state_dict()
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        print()
    
    # Training complete
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    return model, history

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Clear cache to prevent OOM
            torch.cuda.empty_cache()
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    # Per-class metrics
    for i, emotion in enumerate(EMOTION_CLASSES):
        class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)[i]
        class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)[i]
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)[i]
        print(f"{emotion}: Precision={class_prec:.4f}, Recall={class_rec:.4f}, F1={class_f1:.4f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'conf_matrix': conf_mat
    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train emotion recognition model.')
    parser.add_argument('--config', type=str, default='configs/phase1.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, 
                        help='Directory containing processed data splits (overrides config)')
    parser.add_argument('--model', type=str,
                        choices=['efficientnet_b0', 'mobilenet_v3_small', 'mobilevit_xxs', 'resnet50', 'densenet121'],
                        help='Model architecture to use (overrides config)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights (overrides config)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights (overrides config)')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (overrides config)')
    parser.add_argument('--weight-decay', type=float,
                        help='Weight decay for optimizer (overrides config)')
    parser.add_argument('--log-dir', type=str,
                        help='Directory to save logs (overrides config)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save model checkpoints (overrides config)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--test', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run in smoke test mode (1 epoch)')
    return parser.parse_args()

def expand_path(path, model_name):
    """Expand path with model name.
    
    Args:
        path (str): Path with variables to expand
        model_name (str): Model name to insert
        
    Returns:
        str: Expanded path
    """
    return path.replace("${model.name}", model_name)

def main():
    args = parse_arguments()
    
    # Log configuration source
    print(f"Loading configuration from: {args.config}")
    
    try:
        # Load config file
        config = OmegaConf.load(args.config)
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    # Override config with command line arguments
    if args.data_dir:
        config.data.dir = args.data_dir
        print(f"Overriding data directory: {config.data.dir}")
    if args.model:
        config.model.name = args.model
        print(f"Overriding model: {config.model.name}")
    if args.pretrained:
        config.model.pretrained = True
        print("Using pretrained weights")
    if args.no_pretrained:
        config.model.pretrained = False
        print("Not using pretrained weights")
    if args.batch_size:
        config.training.batch_size = args.batch_size
        print(f"Overriding batch size: {config.training.batch_size}")
    if args.epochs:
        config.training.epochs = args.epochs
        print(f"Overriding epochs: {config.training.epochs}")
    if args.lr:
        config.training.lr = args.lr
        print(f"Overriding learning rate: {config.training.lr}")
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay
        print(f"Overriding weight decay: {config.training.weight_decay}")
    if args.log_dir:
        config.paths.log_dir = args.log_dir
        print(f"Overriding log directory: {config.paths.log_dir}")
    if args.output_dir:
        config.paths.output_dir = args.output_dir
        print(f"Overriding output directory: {config.paths.output_dir}")
    if args.seed:
        config.training.seed = args.seed
        print(f"Overriding random seed: {config.training.seed}")
    
    # Handle smoke test
    if args.smoke_test:
        config.training.epochs = 1
        print("Running in smoke test mode (1 epoch)")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    print(f"Random seed set to: {config.training.seed}")
    
    # Expand paths with model name
    log_dir = expand_path(config.paths.log_dir, config.model.name)
    output_dir = expand_path(config.paths.output_dir, config.model.name)
    metrics_dir = expand_path(config.paths.metrics_dir, config.model.name)
    checkpoints_dir = expand_path(config.paths.checkpoints_dir, config.model.name)
    
    print(f"Using model: {config.model.name}")
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics directory: {metrics_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    
    # Create output directories
    for directory in [log_dir, output_dir, metrics_dir, checkpoints_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up memory optimization for CUDA
    if torch.cuda.is_available():
        # Enable memory optimization
        torch.cuda.empty_cache()
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Reduced jitter
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # Reduced erasing
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(248), 
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(248),  
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Create datasets
    image_datasets = {
        x: EmotionDataset(
            data_dir=os.path.join(config.data.dir, x),
            labels_file=os.path.join(config.data.dir, f"{x}_labels.csv"),
            transform=data_transforms[x]
        )
        for x in ['train', 'val', 'test']
    }
    
    # Create dataloaders with reduced batch size and workers
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=config.training.batch_size,
            shuffle=(x == 'train'),
            num_workers=2,  
            pin_memory=True  # Use pinned memory for faster transfers
        )
        for x in ['train', 'val', 'test']
    }
    
    # Create model
    model = create_model(config.model.name, num_classes=len(EMOTION_CLASSES), pretrained=config.model.pretrained)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    
    # Create unique log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_dir, f"{timestamp}")
    
    # Train model
    print(f"Starting training for {config.training.epochs} epochs...")
    print(f"Model: {config.model.name}, Pretrained: {config.model.pretrained}")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        device, num_epochs=config.training.epochs, log_dir=run_log_dir
    )
    
    # Save model
    model_path = os.path.join(output_dir, f"{config.model.name}_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'class_to_idx': {label: idx for idx, label in enumerate(EMOTION_CLASSES)},
        'model_name': config.model.name,
        'config': OmegaConf.to_container(config),
        'timestamp': timestamp
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, f"{config.model.name}_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'epoch': config.training.epochs,
        'config': OmegaConf.to_container(config),
        'timestamp': timestamp
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Evaluate on test set if requested
    if args.test:
        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, dataloaders['test'], device)
        
        # Save metrics
        metrics_path = os.path.join(metrics_dir, f"{config.model.name}_{timestamp}.pt")
        torch.save({
            'metrics': metrics,
            'config': OmegaConf.to_container(config),
            'timestamp': timestamp
        }, metrics_path)
        print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
