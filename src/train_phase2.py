#!/usr/bin/env python3
# train_phase2.py - Train emotion recognition model for RAF-DB dataset
# Optimized version based on successful local training runs (84.68% accuracy with EfficientNet-B0)
# Default parameters set to match the successful local configuration:
# - Model: EfficientNet-B0
# - Learning Rate: 0.0005
# - Batch Size: 16
# - Weight Decay: 1e-3
# - Epochs: 30

import json
import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import random
from sklearn.model_selection import StratifiedKFold
import warnings
import sys
import torchvision
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from torchvision.transforms import v2 as transforms_v2
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define nullcontext for environments without CUDA support
class nullcontext:
    """Context manager that does nothing. Used as a fallback for autocast when CUDA is not available."""
    def __enter__(self):
        return None
    
    def __exit__(self, *args):
        pass

# Import OmegaConf for config handling
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("OmegaConf not available. Install with: pip install omegaconf")

# Set up tensorboard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

# Define emotion classes
EMOTION_CLASSES = ['anger', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'disgust']

# RAF-DB uses numbers (1-7) to represent emotions in its label file
# 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
RAFDB_EMOTION_MAPPING = {
    1: 'surprise',
    2: 'fear', 
    3: 'disgust',
    4: 'happy',
    5: 'sad',
    6: 'anger',
    7: 'neutral'
}

class BlurPool(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(BlurPool, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size == 3:
            bk = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
            bk = bk / torch.sum(bk)
        else:  # kernel_size == 2
            bk = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
            bk = bk / torch.sum(bk)
            
        bk = bk.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer('blur_kernel', bk)
        self.pad = nn.ReflectionPad2d(1) if kernel_size == 3 else nn.ReflectionPad2d(0)
        
    def forward(self, x):
        x = self.pad(x)
        return F.conv2d(x, self.blur_kernel, stride=2, padding=0, groups=x.shape[1])

class EmotionDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None, add_identity_check=True):
        """
        Args:
            data_dir (string): Directory with all the images
            labels_df (DataFrame): DataFrame containing image filenames and labels
            transform (callable, optional): Optional transform to be applied on a sample
            add_identity_check (bool): Check if images contain valid face data
        """
        self.data_dir = data_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.add_identity_check = add_identity_check
        
        # Create label mapping (string to int)
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_CLASSES)}
        
        # Check the format of labels_df and adapt if needed
        if 'filename' in self.labels_df.columns and 'label' in self.labels_df.columns:
            # Use the named columns
            self.filename_col = 'filename'
            self.label_col = 'label'
            print("Using named columns: filename and label")
        elif len(self.labels_df.columns) >= 2:
            # Use positional columns
            self.filename_col = self.labels_df.columns[0]
            self.label_col = self.labels_df.columns[1]
            print(f"Using positional columns: {self.filename_col} and {self.label_col}")
        else:
            print("Could not determine label column. Please check your labels file format.")
            print("DataFrame columns:", self.labels_df.columns.tolist())
            raise ValueError("Incompatible labels file format")
        
        # Verify all numeric labels are valid
        try:
            # Try to convert labels to integers if they're not already
            self.labels_df[self.label_col] = self.labels_df[self.label_col].apply(
                lambda x: int(x) if not isinstance(x, int) else x
            )
            
            unique_labels = self.labels_df[self.label_col].unique()
            unknown_labels = set(unique_labels) - set(RAFDB_EMOTION_MAPPING.keys())
            if unknown_labels:
                print(f"Warning: Found unknown numeric labels: {unknown_labels}.")
                print(f"Expected labels: {list(RAFDB_EMOTION_MAPPING.keys())}")
                print("Will attempt to continue with valid labels only.")
                
                # Filter out rows with invalid labels
                valid_label_mask = self.labels_df[self.label_col].isin(RAFDB_EMOTION_MAPPING.keys())
                self.labels_df = self.labels_df[valid_label_mask].reset_index(drop=True)
                print(f"Filtered dataset to {len(self.labels_df)} entries with valid labels")
                
        except Exception as e:
            print(f"Error processing labels: {e}")
            print(f"Labels in the file ({self.label_col}):", self.labels_df[self.label_col].head())
            raise ValueError("Failed to process labels. Ensure they are numeric and match RAF-DB format.")
        
        # Precompute valid image paths and store them (optimize for Puhti)
        self.valid_paths = {}
        self._validate_and_cache_paths()
            
    def _validate_and_cache_paths(self):
        """Pre-validate and cache image paths for faster access"""
        print("Validating and caching image paths...")
        valid_indices = []
        
        # For progress tracking
        total = len(self.labels_df)
        batch_size = 1000  # Process in batches to show progress
        num_batches = (total + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            
            for idx in tqdm(range(start_idx, end_idx), desc=f"Validating batch {batch_idx+1}/{num_batches}"):
                try:
                    img_name = str(self.labels_df.iloc[idx][self.filename_col])
                    # Handle different path formats
                    if '/' in img_name:
                        split = img_name.split('/')[0]
                        img_name = os.path.basename(img_name)
                    else:
                        # If no path separator, assume it's just a filename in the train split
                        split = 'train'
                    
                    # Get the label
                    numeric_label = int(self.labels_df.iloc[idx][self.label_col])
                    if numeric_label not in RAFDB_EMOTION_MAPPING:
                        continue  # Skip invalid labels
                    
                    # Try different path structures
                    img_paths = [
                        os.path.join(self.data_dir, split, str(numeric_label), img_name),
                        os.path.join(self.data_dir, split, img_name),
                        os.path.join(self.data_dir, img_name)
                    ]
                    
                    found = False
                    for img_path in img_paths:
                        if os.path.exists(img_path):
                            self.valid_paths[idx] = img_path
                            valid_indices.append(idx)
                            found = True
                            break
                    
                    if not found and idx < 5:  # Print first few failures for debugging
                        print(f"Warning: Could not find image for entry {idx}: {img_name} (label {numeric_label})")
                        print(f"Tried paths: {img_paths}")
                
                except Exception as e:
                    if idx < 5:  # Print only first few errors
                        print(f"Error processing entry {idx}: {e}")
        
        if len(valid_indices) < len(self.labels_df):
            print(f"WARNING: {len(self.labels_df) - len(valid_indices)} images not found. Using {len(valid_indices)} valid images.")
            if len(valid_indices) == 0:
                raise ValueError("No valid images found. Check your data_dir and label_file paths.")
            self.labels_df = self.labels_df.iloc[valid_indices].reset_index(drop=True)
            
            # Update valid_paths with new indices
            new_valid_paths = {}
            for new_idx, old_idx in enumerate(valid_indices):
                if old_idx in self.valid_paths:
                    new_valid_paths[new_idx] = self.valid_paths[old_idx]
            self.valid_paths = new_valid_paths
        
        print(f"Successfully cached {len(self.valid_paths)} image paths")
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get the cached path if available
        if idx in self.valid_paths:
            img_path = self.valid_paths[idx]
            try:
                image = read_image(img_path)
                # Convert to 3 channels if grayscale
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
            except Exception as e:
                print(f"Error loading cached image {img_path}: {e}")
                image = torch.zeros((3, 224, 224), dtype=torch.uint8)
        else:
            # Fall back to the original path finding logic
            img_name = str(self.labels_df.iloc[idx][self.filename_col])
            # Handle different path formats
            if '/' in img_name:
                split = img_name.split('/')[0]
                img_name = os.path.basename(img_name)
            else:
                # If no path separator, assume it's just a filename in the train split
                split = 'train'
            
            # Get numeric label first
            numeric_label = int(self.labels_df.iloc[idx][self.label_col])
            
            # Try different path structures
            img_paths = [
                os.path.join(self.data_dir, split, str(numeric_label), img_name),
                os.path.join(self.data_dir, split, img_name),
                os.path.join(self.data_dir, img_name)
            ]
            
            # Try to find the image
            image = None
            for img_path in img_paths:
                if os.path.exists(img_path):
                    try:
                        image = read_image(img_path)
                        # Convert to 3 channels if grayscale
                        if image.shape[0] == 1:
                            image = image.repeat(3, 1, 1)
                        # Cache this path for future use
                        self.valid_paths[idx] = img_path
                        break
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
            
            # If no image was found or loaded, return a placeholder
            if image is None:
                print(f"Warning: Could not find or read image for index {idx}, paths tried: {img_paths}")
                image = torch.zeros((3, 224, 224), dtype=torch.uint8)
        
        # Get numeric label for this index
        numeric_label = int(self.labels_df.iloc[idx][self.label_col])
        # Convert numeric label to emotion string and then to index
        emotion_str = RAFDB_EMOTION_MAPPING[numeric_label]
        label = self.label_to_idx[emotion_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EmotionAttentionNetwork(nn.Module):
    def __init__(self, num_classes=7, base_model='efficientnet_b0', pretrained=True):
        super(EmotionAttentionNetwork, self).__init__()
        
        # Load base model
        if base_model == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            # Remove the final classifier
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            feature_dim = 1280
        elif base_model == 'efficientnet_b2':
            self.base_model = models.efficientnet_b2(weights='DEFAULT' if pretrained else None)
            # Remove the final classifier
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            feature_dim = 1408
        elif base_model == 'resnet50':
            self.base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            # Remove the final fully connected layer
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Base model {base_model} not supported")
            
        # Attention mechanisms
        self.cbam = CBAM(feature_dim, reduction=16)
        self.se = SEBlock(feature_dim, reduction=16)
        
        # Classifier
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.features(x)
        
        # Apply attention
        x = self.cbam(x)
        x = self.se(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Classifier
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        
        return x

def create_model(model_name, num_classes=7, pretrained=True):
    """Create model based on specified architecture."""
    if model_name == 'emotion_attention_net':
        base = 'efficientnet_b0'  # Default to EfficientNet-B0 as it performed better
        model = EmotionAttentionNetwork(num_classes=num_classes, base_model=base, pretrained=pretrained)
        print(f"Created EmotionAttentionNetwork with {base} backbone")
    
    elif model_name == 'emotion_attention_net_large':
        base = 'efficientnet_b2'
        model = EmotionAttentionNetwork(num_classes=num_classes, base_model=base, pretrained=pretrained)
        print(f"Created EmotionAttentionNetwork with {base} backbone")
    
    elif model_name == 'emotion_attention_net_resnet':
        base = 'resnet50'
        model = EmotionAttentionNetwork(num_classes=num_classes, base_model=base, pretrained=pretrained)
        print(f"Created EmotionAttentionNetwork with {base} backbone")
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(f"Created EfficientNet-B0 (best performing in local tests)")
    
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    else:
        print(f"Model {model_name} not supported. Using EfficientNet-B0")
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(f"Created EfficientNet-B0 (best performing in local tests)")
    
    return model

def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, log_dir=None, 
                mixup_alpha=0.2, early_stopping_patience=10, grad_clip_value=1.0, label_smoothing=0.1,
                checkpoint_freq=1, checkpoint_dir=None):
    """Train the model with advanced training techniques and checkpointing."""
    since = time.time()
    
    # Initialize tensorboard writer if available
    writer = None
    if TENSORBOARD_AVAILABLE and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize best model parameters
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    best_f1 = 0.0
    
    # Early stopping
    patience = early_stopping_patience
    patience_counter = 0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Initialize criterion with label smoothing
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Use automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Learning rate history
    lr_history = []
    
    # Initialize start epoch (useful for resuming training)
    start_epoch = 0
    
    # Record memory usage throughout training
    memory_usage = []
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # Log current memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage.append((current_memory, peak_memory))
            print(f"Current GPU memory: {current_memory:.2f} MB, Peak: {peak_memory:.2f} MB")
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            # Get the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if phase == 'train':
                lr_history.append(current_lr)
                print(f"Current learning rate: {current_lr}")
                
            # Iterate over data
            total_batches = len(dataloaders[phase])
            progress_interval = max(1, total_batches // 10)  # Show progress 10 times per epoch
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # Report progress
                if batch_idx % progress_interval == 0:
                    print(f"  Processing batch {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)")
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)  # More efficient than zeros
                
                # Forward pass with autocast for mixed precision
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                        # Apply mixup only during training
                        if phase == 'train' and mixup_alpha > 0:
                            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                            outputs = model(inputs)
                            loss = mixup_criterion(criterion_smooth, outputs, labels_a, labels_b, lam)
                            # For accuracy calculation with mixup
                            _, preds = torch.max(outputs, 1)
                            running_corrects += (lam * torch.sum(preds == labels_a.data) + 
                                              (1 - lam) * torch.sum(preds == labels_b.data))
                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion_smooth(outputs, labels)
                            # Collect predictions for metrics
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                            running_corrects += torch.sum(preds == labels.data)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        if scaler:
                            # Mixed precision training
                            scaler.scale(loss).backward()
                            # Gradient clipping
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Clear cache periodically to prevent OOM
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Compute epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Calculate additional metrics for validation
            if phase == 'val' and len(all_preds) > 0:
                epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
                history[f'{phase}_f1'].append(epoch_f1)
                
                # Log confusion matrix
                if writer:
                    cm = confusion_matrix(all_labels, all_preds)
                    cm_fig = plot_confusion_matrix(cm, EMOTION_CLASSES)
                    writer.add_figure(f'Confusion Matrix/Epoch {epoch+1}', cm_fig, epoch)
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Log to tensorboard
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
                if phase == 'val' and len(all_preds) > 0:
                    writer.add_scalar(f'F1/{phase}', epoch_f1, epoch)
                    
                    # Log memory usage
                    if torch.cuda.is_available():
                        writer.add_scalar('Memory/GPU_Allocated_MB', torch.cuda.memory_allocated() / 1024**2, epoch)
                        writer.add_scalar('Memory/GPU_Reserved_MB', torch.cuda.memory_reserved() / 1024**2, epoch)
            
            # Save best model (track both accuracy and F1)
            if phase == 'val':
                # Maintain best model based on F1 score for imbalanced data
                current_metric = epoch_f1 if len(all_preds) > 0 else epoch_acc
                if current_metric > best_f1:
                    best_acc = epoch_acc
                    best_f1 = current_metric if len(all_preds) > 0 else best_f1
                    best_epoch = epoch
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                    
                    # Save checkpoint of the best model
                    if checkpoint_dir:
                        best_model_path = os.path.join(checkpoint_dir, f"best_model.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'loss': epoch_loss,
                            'accuracy': epoch_acc.item(),
                            'f1': epoch_f1 if len(all_preds) > 0 else 0.0,
                            'history': history
                        }, best_model_path)
                        print(f"Saved best model checkpoint to {best_model_path}")
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered! No improvement for {patience} epochs.")
                    print(f'Best val Acc: {best_acc:.4f}, Best F1: {best_f1:.4f} at epoch {best_epoch+1}')
                    model.load_state_dict(best_model_wts)
                    
                    # Add early stopping info to history
                    history['early_stopped'] = True
                    history['best_epoch'] = best_epoch
                    
                    # Close tensorboard writer
                    if writer:
                        writer.close()
                    
                    return model, history
        
        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_freq == 0 and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': history['val_loss'][-1] if history['val_loss'] else 0,
                'accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
                'f1': history['val_f1'][-1] if history['val_f1'] else 0,
                'history': history
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()
            
            # Log learning rate
            if writer:
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Report epoch timing
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print()
    
    # Training complete
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}, Best F1: {best_f1:.4f} at epoch {best_epoch+1}')
    
    # Add best epoch info to history
    history['best_epoch'] = best_epoch
    history['best_acc'] = best_acc
    history['best_f1'] = best_f1
    history['training_time'] = time_elapsed
    history['memory_usage'] = memory_usage
    history['lr_history'] = lr_history
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    return model, history

def plot_confusion_matrix(cm, class_names):
    """
    Generate a matplotlib figure containing the confusion matrix heatmap.
    
    Args:
        cm (ndarray): The confusion matrix
        class_names (list): Names of the classes
        
    Returns:
        matplotlib.figure.Figure: The figure with the confusion matrix plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure and axes
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Add axis ticks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize and add text annotations
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                     horizontalalignment="center", color=color)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return figure

def evaluate_model(model, test_loader, device, log_dir=None):
    """Evaluate model on test set and return metrics. Optimized for Puhti CSC."""
    print("Starting model evaluation...")
    print(f"Using emotion classes: {EMOTION_CLASSES}")
    print(f"Number of emotion classes: {len(EMOTION_CLASSES)}")
    model.eval()
    
    # Create tensorboard writer if log_dir provided
    writer = None
    if TENSORBOARD_AVAILABLE and log_dir:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "evaluation"))
    
    # For efficient memory usage, process in batches
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Track timing for performance analysis
    batch_times = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            batch_start = time.time()
            
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference if available
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels to CPU to free up GPU memory
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Record batch processing time
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Clear cache periodically to prevent OOM
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Report occasional progress for long evaluations
            if batch_idx % 20 == 0 and batch_idx > 0:
                avg_time = sum(batch_times) / len(batch_times)
                remaining = avg_time * (len(test_loader) - batch_idx)
                print(f"Processed {batch_idx}/{len(test_loader)} batches. " 
                      f"Avg time: {avg_time:.4f}s. Est. remaining: {remaining:.1f}s")
    
    # Concatenate probability arrays
    if all_probs:
        all_probs = np.concatenate(all_probs, axis=0)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f}s")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    # Log confusion matrix to tensorboard
    if writer:
        cm_figure = plot_confusion_matrix(conf_mat, EMOTION_CLASSES)
        writer.add_figure('Confusion Matrix/Test', cm_figure)
    
    # Per-class metrics
    class_metrics = {}
    print("\nPer-class metrics:")
    
    # Get the actual classes present in the predictions and labels
    unique_classes = sorted(set(all_labels + all_preds))
    print(f"Unique classes in evaluation data: {unique_classes}")
    
    # Calculate per-class metrics for each emotion class that's present in the data
    per_class_metrics = precision_score(all_labels, all_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    for i, emotion in enumerate(EMOTION_CLASSES):
        if i < len(per_class_metrics):
            class_prec = per_class_metrics[i]
            class_rec = class_recall[i]
            class_f1_score = class_f1[i]
        else:
            print(f"Warning: Class {emotion} (index {i}) not present in evaluation data.")
            class_prec = 0.0
            class_rec = 0.0
            class_f1_score = 0.0
            
        print(f"{emotion}: Precision={class_prec:.4f}, Recall={class_rec:.4f}, F1={class_f1_score:.4f}")
        class_metrics[emotion] = {'precision': class_prec, 'recall': class_rec, 'f1': class_f1_score}
        
        # Log class metrics to tensorboard
        if writer:
            writer.add_scalar(f'Metrics/{emotion}/Precision', class_prec)
            writer.add_scalar(f'Metrics/{emotion}/Recall', class_rec)
            writer.add_scalar(f'Metrics/{emotion}/F1', class_f1_score)
    
    # Calculate ROC and PR curves for multi-class
    if all_probs is not None and all_probs.size > 0:
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            from sklearn.preprocessing import label_binarize
            
            # Binarize labels for ROC curve
            y_bin = label_binarize(all_labels, classes=range(len(EMOTION_CLASSES)))
            
            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i, emotion in enumerate(EMOTION_CLASSES):
                fpr[emotion], tpr[emotion], _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc[emotion] = auc(fpr[emotion], tpr[emotion])
                
                # Log to metrics dict
                class_metrics[emotion]['roc_auc'] = roc_auc[emotion]
                
                # Plot ROC curve if we have a writer
                if writer:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(8, 6))
                    plt.plot(fpr[emotion], tpr[emotion], lw=2, 
                             label=f'ROC curve (area = {roc_auc[emotion]:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {emotion}')
                    plt.legend(loc="lower right")
                    writer.add_figure(f'ROC/{emotion}', fig)
            
            # Log macro average ROC AUC
            if writer:
                writer.add_scalar('Metrics/MacroAvg/ROC_AUC', sum(roc_auc.values()) / len(roc_auc))
                
        except Exception as e:
            print(f"Error generating ROC curves: {e}")
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'conf_matrix': conf_mat,
        'class_metrics': class_metrics,
        'evaluation_time': total_time,
        'batch_times': {
            'mean': np.mean(batch_times),
            'std': np.std(batch_times),
            'min': min(batch_times),
            'max': max(batch_times)
        }
    }

def k_fold_cross_validation(model_name, dataset, k=5, batch_size=32, num_epochs=25, lr=0.001, 
                           weight_decay=1e-4, device='cuda', log_dir=None, seed=42,
                           num_workers=4, pin_memory=True, mixup_alpha=0.2,
                           early_stopping_patience=10, grad_clip_value=1.0):
    """Perform k-fold cross-validation optimized for Puhti CSC."""
    print(f"Starting {k}-fold cross-validation with Puhti CSC optimizations...")
    start_time = time.time()
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize fold results
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_f1': {emotion: [] for emotion in EMOTION_CLASSES},
        'training_time': [],
        'best_epochs': []
    }
    
    # Define k-fold cross validator with stratification
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Get all labels for stratification
    all_labels = np.array([label for _, label in dataset])
    
    # Calculate label distribution for monitoring class balance
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print("Dataset class distribution:")
    for label_idx, count in zip(unique_labels, label_counts):
        print(f"  Class {EMOTION_CLASSES[label_idx]}: {count} samples ({count/len(all_labels)*100:.1f}%)")
    
    # Start k-fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(len(dataset)), all_labels)):
        fold_start_time = time.time()
        print(f"\nFOLD {fold+1}/{k}")
        print("-" * 50)
        
        # Create fold-specific log and checkpoint directories
        fold_log_dir = os.path.join(log_dir, f"fold_{fold+1}") if log_dir else None
        fold_checkpoint_dir = os.path.join(log_dir, f"fold_{fold+1}", "checkpoints") if log_dir else None
        os.makedirs(fold_checkpoint_dir, exist_ok=True) if fold_checkpoint_dir else None
        
        # Sample data for each fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Verify fold class distribution
        train_labels = all_labels[train_ids]
        val_labels = all_labels[val_ids]
        
        print(f"Train set: {len(train_ids)} samples")
        print(f"Validation set: {len(val_ids)} samples")
        
        # Create data loaders with Puhti CSC optimizations
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        dataloaders = {'train': train_loader, 'val': val_loader}
        
        # Create model with memory optimizations
        model = create_model(model_name, num_classes=len(EMOTION_CLASSES), pretrained=True)
        
        # Apply memory optimizations to model
        if hasattr(model, 'cpu'):
            # First create model on CPU
            model.cpu()
            # Clear cache before loading model to GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Now move to GPU
            model = model.to(device)
        else:
            model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Train model with Puhti optimizations
        print(f"Training fold {fold+1}...")
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler,
            device, num_epochs=num_epochs, log_dir=fold_log_dir,
            mixup_alpha=mixup_alpha, early_stopping_patience=early_stopping_patience,
            grad_clip_value=grad_clip_value, checkpoint_dir=fold_checkpoint_dir,
            checkpoint_freq=5  # Save checkpoints every 5 epochs
        )
        
        # Evaluate model on validation set
        print(f"Evaluating fold {fold+1}...")
        metrics = evaluate_model(model, val_loader, device, log_dir=fold_log_dir)
        
        # Store results
        fold_results['accuracy'].append(metrics['accuracy'])
        fold_results['precision'].append(metrics['precision'])
        fold_results['recall'].append(metrics['recall'])
        fold_results['f1'].append(metrics['f1'])
        fold_results['training_time'].append(time.time() - fold_start_time)
        fold_results['best_epochs'].append(history.get('best_epoch', 0))
        
        # Store per-class F1 scores
        for emotion in EMOTION_CLASSES:
            fold_results['per_class_f1'][emotion].append(
                metrics['class_metrics'][emotion]['f1']
            )
        
        # Save fold model
        fold_model_path = os.path.join(log_dir, f"fold_{fold+1}_model.pt") if log_dir else None
        if fold_model_path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
                'metrics': metrics,
                'epoch': history.get('best_epoch', 0)
            }, fold_model_path)
            print(f"Saved fold {fold+1} model to {fold_model_path}")
        
        # Memory cleanup after each fold
        del model, optimizer, scheduler, dataloaders, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold+1} completed in {fold_time//60:.0f}m {fold_time%60:.0f}s")
    
    # Calculate average performance across folds
    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_precision = np.mean(fold_results['precision'])
    avg_recall = np.mean(fold_results['recall'])
    avg_f1 = np.mean(fold_results['f1'])
    avg_time = np.mean(fold_results['training_time'])
    
    total_time = time.time() - start_time
    
    print(f"\nK-Fold Cross-Validation Results (k={k}):")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(fold_results['accuracy']):.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {np.std(fold_results['precision']):.4f}")
    print(f"Average Recall: {avg_recall:.4f} ± {np.std(fold_results['recall']):.4f}")
    print(f"Average F1-Score: {avg_f1:.4f} ± {np.std(fold_results['f1']):.4f}")
    print(f"Average Training Time: {avg_time//60:.0f}m {avg_time%60:.0f}s per fold")
    print(f"Total Time: {total_time//60:.0f}m {total_time%60:.0f}s")
    
    # Per-class average F1 scores
    print("\nPer-class average F1 scores:")
    for emotion in EMOTION_CLASSES:
        avg_class_f1 = np.mean(fold_results['per_class_f1'][emotion])
        std_class_f1 = np.std(fold_results['per_class_f1'][emotion])
        print(f"  {emotion}: {avg_class_f1:.4f} ± {std_class_f1:.4f}")
    
    # Add summary metrics to fold_results
    fold_results['summary'] = {
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'std_accuracy': np.std(fold_results['accuracy']),
        'std_precision': np.std(fold_results['precision']),
        'std_recall': np.std(fold_results['recall']),
        'std_f1': np.std(fold_results['f1']),
        'avg_training_time': avg_time,
        'total_time': total_time
    }
    
    return fold_results

def parse_args():
    """Parse command line arguments with performance optimizations."""
    parser = argparse.ArgumentParser(description='Train emotion recognition model on RAF-DB dataset (Optimized version)')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--data_dir', type=str, help='Path to RAF-DB dataset directory')
    parser.add_argument('--label_file', type=str, help='Path to RAF-DB label file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimization')
    parser.add_argument('--k_fold', type=int, default=0, 
                        help='Number of folds for cross-validation (0 to disable)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, 
                        help='Alpha parameter for mixup augmentation (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_pretrained', action='store_true', 
                        help='Disable use of pretrained weights')
    
    # Performance optimizations
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Use pin_memory for faster data transfer to GPU')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training for faster computation')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value to prevent exploding gradients')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch (if num_workers > 0)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode for reproducibility')
    parser.add_argument('--cache_dataset', action='store_true',
                        help='Cache dataset in memory to speed up training')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor for loss function')
    
    return parser.parse_args()

def main():
    """Main function with performance optimizations."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Print environment information for debugging
    print("\nEnvironment Information:")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Number of CPUs: {os.cpu_count()}")
    
    # Enable deterministic mode if requested
    if args.deterministic:
        print("Enabling deterministic mode for reproducibility (may impact performance)")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
    
    # Load config file if provided
    config = {}
    if args.config and OMEGACONF_AVAILABLE:
        print(f"\nLoading configuration from: {args.config}")
        try:
            config = OmegaConf.load(args.config)
            print("Configuration loaded successfully")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    
    # Override config with command line arguments or use defaults
    data_dir = args.data_dir if args.data_dir else config.get('data', {}).get('dir', None)
    if not data_dir:
        print("Error: No data directory provided. Use --data_dir or specify in config.")
        return
        
    label_file = args.label_file if args.label_file else config.get('data', {}).get('label_file', None)
    if not label_file:
        print("Error: No label file provided. Use --label_file or specify in config.")
        return
        
    model_name = args.model if args.model != 'efficientnet_b0' else config.get('model', {}).get('name', 'efficientnet_b0')
    batch_size = args.batch_size if args.batch_size != 16 else config.get('training', {}).get('batch_size', 16)
    epochs = args.epochs if args.epochs != 30 else config.get('training', {}).get('epochs', 30)
    learning_rate = args.lr if args.lr != 0.0005 else config.get('training', {}).get('lr', 0.0005)
    weight_decay = args.weight_decay if args.weight_decay != 1e-3 else config.get('training', {}).get('weight_decay', 1e-3)
    k_fold = args.k_fold if args.k_fold != 0 else config.get('training', {}).get('k_fold', 0)
    mixup_alpha = args.mixup_alpha if args.mixup_alpha != 0.2 else config.get('training', {}).get('mixup_alpha', 0.2)
    early_stopping = args.early_stopping if args.early_stopping != 10 else config.get('training', {}).get('early_stopping_patience', 10)
    seed = args.seed if args.seed != 42 else config.get('training', {}).get('seed', 42)
    no_pretrained = args.no_pretrained if args.no_pretrained else not config.get('model', {}).get('pretrained', True)
    
    # Puhti CSC specific parameters
    num_workers = args.num_workers if args.num_workers != 4 else config.get('puhti', {}).get('num_workers', 4)
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq != 5 else config.get('puhti', {}).get('checkpoint_freq', 5)
    pin_memory = args.pin_memory if args.pin_memory else config.get('puhti', {}).get('pin_memory', True)
    mixed_precision = args.mixed_precision if args.mixed_precision else config.get('puhti', {}).get('mixed_precision', True)
    grad_clip = args.grad_clip if args.grad_clip != 1.0 else config.get('puhti', {}).get('grad_clip', 1.0)
    label_smoothing = args.label_smoothing if args.label_smoothing != 0.1 else config.get('puhti', {}).get('label_smoothing', 0.1)
    
    # Create output directory path based on config if needed
    output_base = args.output_dir if args.output_dir != './output' else config.get('paths', {}).get('output_dir', './output')
    
    # Display configuration
    print(f"\nTraining Configuration:")
    print(f"  Data Directory: {data_dir}")
    print(f"  Label File: {label_file}")
    print(f"  Model: {model_name}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  K-Fold: {k_fold}")
    print(f"  Mixup Alpha: {mixup_alpha}")
    print(f"  Early Stopping: {early_stopping}")
    print(f"  Seed: {seed}")
    print(f"  Pretrained: {not no_pretrained}")
    print(f"  Label Smoothing: {label_smoothing}")
    print(f"\nPerformance Optimizations:")
    print(f"  Number of Workers: {num_workers}")
    print(f"  Checkpoint Frequency: {checkpoint_freq}")
    print(f"  Pin Memory: {pin_memory}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  Gradient Clipping: {grad_clip}")
    print(f"  Output Directory: {output_base}")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create output directory with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config_summary = {
        'model': model_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'k_fold': k_fold,
        'mixup_alpha': mixup_alpha,
        'early_stopping': early_stopping,
        'seed': seed,
        'pretrained': not no_pretrained,
        'data_dir': data_dir,
        'label_file': label_file,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'mixed_precision': mixed_precision,
        'grad_clip': grad_clip,
        'label_smoothing': label_smoothing,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A",
        'environment': 'optimized'
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_summary, f, indent=4)
    
    # Set device
    if torch.cuda.is_available():
        # Get available GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
        print(f"\nUsing CUDA device with {gpu_mem:.2f} GB memory")
        
        # Adjust batch size if memory is limited
        if gpu_mem < 6 and batch_size > 16:
            old_batch_size = batch_size
            batch_size = 16
            print(f"WARNING: Limited GPU memory detected. Reducing batch size from {old_batch_size} to {batch_size}")
        
        device = torch.device("cuda")
    else:
        print("\nWARNING: CUDA not available, using CPU. This will be much slower!")
        device = torch.device("cpu")
        # Reduce batch size for CPU training
        if batch_size > 8:
            old_batch_size = batch_size
            batch_size = 8
            print(f"Reducing batch size for CPU training from {old_batch_size} to {batch_size}")
    
    # Monitor initial memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Load label data with robust handling of different formats
    print(f"\nLoading label data from {label_file}")
    try:
        # Try different delimiters to handle different file formats
        # First try with header detection
        for delimiter in [',', ' ', '\t']:
            try:
                # Try with header first
                labels_df = pd.read_csv(label_file, sep=delimiter)
                if len(labels_df.columns) >= 2:
                    print(f"Successfully loaded label file with delimiter '{delimiter}' and header row")
                    has_header = True
                    break
            except:
                continue
        else:
            # If loading with header didn't work, try without header
            for delimiter in [',', ' ', '\t']:
                try:
                    labels_df = pd.read_csv(label_file, sep=delimiter, header=None)
                    if len(labels_df.columns) >= 2:
                        print(f"Successfully loaded label file with delimiter '{delimiter}' without header")
                        has_header = False
                        break
                except:
                    continue
            else:
                # Last resort: try pandas auto-detection
                try:
                    labels_df = pd.read_csv(label_file)
                    has_header = True
                    print(f"Successfully loaded label file with auto-detected format")
                except:
                    try:
                        labels_df = pd.read_csv(label_file, header=None)
                        has_header = False
                        print(f"Successfully loaded label file with auto-detected format, no header")
                    except:
                        raise ValueError("Could not parse the label file with any known format")
            
        print(f"Loaded {len(labels_df)} label entries")
        
        # Display the first few rows to help with debugging
        print("First few rows of the label file:")
        print(labels_df.head())
        
        # If the first row contains headers, remove it from the data
        if has_header and any(col.lower() in ['filename', 'image', 'file'] for col in labels_df.columns):
            print("Detected header row, using column names:", labels_df.columns.tolist())
            # Keep the original column names
            column_names = labels_df.columns.tolist()
        else:
            # If no clear header, assume standard format and rename columns
            column_names = ['filename', 'label'] + [f'col{i}' for i in range(2, len(labels_df.columns))]
            labels_df.columns = column_names
            print("No clear header detected, assigned column names:", column_names)
            
        # Check if the first row appears to be a header (e.g., contains 'filename' or 'label')
        first_row = labels_df.iloc[0].astype(str).tolist()
        if any(h.lower() in ['filename', 'label', 'image', 'file'] for h in first_row):
            print("First row appears to be a header, removing it from data")
            labels_df = labels_df.iloc[1:].reset_index(drop=True)
        
    except Exception as e:
        print(f"Error loading label file: {e}")
        print("Please check that your label file exists and has the correct format.")
        return
    
    # Define transformations for RAF-DB dataset
    # Optimized for EfficientNet-B0
    train_transform = transforms.Compose([
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Resize((256, 256)),
        transforms_v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Resize((256, 256)),
        transforms_v2.CenterCrop(224),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("\nInitializing dataset and validating image paths...")
    dataset = EmotionDataset(data_dir, labels_df, transform=None)
    
    # Record memory usage after dataset initialization
    if torch.cuda.is_available():
        print(f"GPU memory after dataset init: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Determine train/test split or use k-fold cross-validation
    if k_fold > 1:
        # Use k-fold cross-validation
        print(f"\nUsing {k_fold}-fold cross-validation")
        dataset.transform = train_transform
        fold_results = k_fold_cross_validation(
            model_name, dataset, 
            k=k_fold, 
            batch_size=batch_size,
            num_epochs=epochs, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            device=device, 
            log_dir=output_dir, 
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mixup_alpha=mixup_alpha,
            early_stopping_patience=early_stopping,
            grad_clip_value=grad_clip
        )
        
        # Save fold results
        result_file = os.path.join(output_dir, "fold_results.json")
        
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for key, value in fold_results.items():
            if key == 'per_class_f1':
                serializable_results[key] = {
                    emotion: [float(v) for v in values] 
                    for emotion, values in value.items()
                }
            elif key == 'summary':
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
        
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # Also save as CSV for easy viewing
        pd.DataFrame({
            'Fold': list(range(1, k_fold + 1)),
            'Accuracy': fold_results['accuracy'],
            'Precision': fold_results['precision'],
            'Recall': fold_results['recall'],
            'F1': fold_results['f1'],
            'Training_Time': fold_results['training_time'],
            'Best_Epoch': fold_results['best_epochs']
        }).to_csv(os.path.join(output_dir, "fold_results.csv"), index=False)
        
        print(f"Saved fold results to {result_file}")
    
    else:
        # Regular train/val split
        print("\nUsing train/val split (80/20)")
        # Split dataset into train and validation sets (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        print(f"Train set: {train_size} samples")
        print(f"Validation set: {val_size} samples")
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders with Puhti optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        dataloaders = {'train': train_loader, 'val': val_loader}
        
        # Clear memory before model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create model
        print(f"\nCreating {model_name} model...")
        model = create_model(
            model_name, 
            num_classes=len(EMOTION_CLASSES), 
            pretrained=not no_pretrained
        )
        model = model.to(device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Train model with optimizations
        print("\nStarting model training...")
        model, history = train_model(
            model, 
            dataloaders, 
            criterion, 
            optimizer, 
            scheduler,
            device, 
            num_epochs=epochs, 
            log_dir=output_dir,
            mixup_alpha=mixup_alpha, 
            early_stopping_patience=early_stopping,
            grad_clip_value=grad_clip,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir,
            label_smoothing=label_smoothing
        )
        
        # Save model
        model_path = os.path.join(output_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'model_name': model_name,
            'num_classes': len(EMOTION_CLASSES),
            'emotion_classes': EMOTION_CLASSES,
            'config': config_summary
        }, model_path)
        print(f"\nSaved model to {model_path}")
        
        # Save a smaller model file with just the weights for deployment
        deployment_model_path = os.path.join(output_dir, "model_deployment.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'num_classes': len(EMOTION_CLASSES),
            'emotion_classes': EMOTION_CLASSES
        }, deployment_model_path)
        print(f"Saved deployment model to {deployment_model_path}")
        
        # Evaluate model on validation set
        print("\nEvaluating model on validation set...")
        metrics = evaluate_model(model, val_loader, device, log_dir=output_dir)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        
        # Convert numpy values to serializable Python types
        serializable_metrics = {}
        for key, value in metrics.items():
            if key == 'conf_matrix':
                serializable_metrics[key] = value.tolist()
            elif key == 'class_metrics':
                serializable_metrics[key] = {
                    emotion: {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics.items()
                    } for emotion, metrics in value.items()
                }
            elif key == 'batch_times':
                serializable_metrics[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items()
                }
            else:
                serializable_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
    
    # Final training time
    total_time = time.time() - start_time
    print(f"\nTotal training completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    
    # Final GPU memory report
    if torch.cuda.is_available():
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print(f"\nModel training completed successfully!")
    print(f"Results saved to: {output_dir}")

# Utility class for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()