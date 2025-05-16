#!/usr/bin/env python3
# train_rafdb.py - Train EfficientNet-B0 model for RAF-DB dataset on Puhti

import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import random
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import v2 as transforms_v2
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define emotion classes
EMOTION_CLASSES = ['anger', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'disgust']

# RAF-DB emotion mapping
RAFDB_EMOTION_MAPPING = {
    1: 'surprise',
    2: 'fear', 
    3: 'disgust',
    4: 'happy',
    5: 'sad',
    6: 'anger',
    7: 'neutral'
}

# Attention modules
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
        else:
            raise ValueError(f"Base model {base_model} not supported")
            
        # Attention mechanisms
        self.cbam = CBAM(feature_dim, reduction=16)
        self.se = SEBlock(feature_dim, reduction=16)
        
        # Classifier
        self.drop1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.4)
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
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
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

def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Create a copy of x to ensure it has proper gradient tracking
    mixed_x = x.clone()
    
    # Ensure lam is a tensor with the right device and dtype
    lam_tensor = torch.tensor(lam, device=device, dtype=torch.float)
    
    # Mix inputs without breaking gradient flow
    mixed_x = lam_tensor * x + (1 - lam_tensor) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam_tensor

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Implement Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()

class EmotionDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images
            labels_df (DataFrame): DataFrame containing image filenames and labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        
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
            raise ValueError("Incompatible labels file format")
        
        # Convert labels to integers if they're not already
        self.labels_df[self.label_col] = self.labels_df[self.label_col].apply(
            lambda x: int(x) if not isinstance(x, int) else x
        )
        
        # Validate image paths
        valid_indices = []
        print("Validating image paths...")
        for idx in tqdm(range(len(self.labels_df))):
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
                
                # Try different path structures
                img_paths = [
                    os.path.join(self.data_dir, split, str(numeric_label), img_name),
                    os.path.join(self.data_dir, split, img_name),
                    os.path.join(self.data_dir, img_name)
                ]
                
                for img_path in img_paths:
                    if os.path.exists(img_path):
                        valid_indices.append(idx)
                        break
            
            except Exception as e:
                if idx < 5:  # Print only first few errors
                    print(f"Error processing entry {idx}: {e}")
        
        if len(valid_indices) < len(self.labels_df):
            print(f"WARNING: {len(self.labels_df) - len(valid_indices)} images not found. Using {len(valid_indices)} valid images.")
            if len(valid_indices) == 0:
                raise ValueError("No valid images found. Check your data_dir and label_file paths.")
            self.labels_df = self.labels_df.iloc[valid_indices].reset_index(drop=True)
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
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
                    break
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        # If no image was found or loaded, return a placeholder
        if image is None:
            print(f"Warning: Could not find or read image for index {idx}")
            image = torch.zeros((3, 224, 224), dtype=torch.uint8)
        
        # Convert numeric label to emotion string and then to index
        emotion_str = RAFDB_EMOTION_MAPPING[numeric_label]
        label = self.label_to_idx[emotion_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass

def test_time_augmentation(model, inputs, device, num_aug=5):
    """Perform test-time augmentation for more robust predictions."""
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        outputs = model(inputs)
        
        try:
            # Horizontal flip
            flipped_inputs = torch.flip(inputs, [3])
            outputs += model(flipped_inputs)
            
            # Center crop
            try:
                crop_size = int(0.9 * min(inputs.shape[2], inputs.shape[3]))
                center_crop = transforms_v2.CenterCrop(crop_size)(inputs)
                resized_crop = transforms_v2.Resize((inputs.shape[2], inputs.shape[3]))(center_crop)
                outputs += model(resized_crop)
            except Exception as e:
                print(f"Warning: Center crop augmentation failed: {e}")
            
            # Slight rotation (±10 degrees)
            try:
                # Different approach for rotation that's more compatible
                rotation_transform = transforms.RandomRotation(degrees=10)
                rotated_plus = rotation_transform(inputs)
                rotation_transform = transforms.RandomRotation(degrees=(-10, -10))
                rotated_minus = rotation_transform(inputs)
                outputs += model(rotated_plus) + model(rotated_minus)
            except Exception as e:
                print(f"Warning: Rotation augmentation failed: {e}")
                
            # Divide by the actual number of successful augmentations + 1 (original)
            return outputs / (num_aug)
        except Exception as e:
            print(f"Warning: Test-time augmentation failed, using original prediction: {e}")
            return outputs

# Progressive resizing transforms
def create_train_transform(size, convert_image_fn):
    return transforms.Compose([
        transforms.Lambda(convert_image_fn),
        transforms_v2.Resize((size+32, size+32), antialias=True),  # Larger initial size for crop
        transforms_v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms_v2.RandomErasing(p=0.2),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_val_transform(size, convert_image_fn):
    return transforms.Compose([
        transforms.Lambda(convert_image_fn),
        transforms_v2.Resize((size+32, size+32), antialias=True),
        transforms_v2.CenterCrop(size),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=30, 
                grad_clip_value=1.0, mixup_alpha=0.2, early_stopping_patience=10, label_smoothing=0.1,
                progressive_resizing=False, use_tta=True, class_weights=None):
    """Train the model with advanced training techniques."""
    since = time.time()
    
    # Initialize best model parameters
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    
    # Early stopping parameters
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
    criterion_smooth = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
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
            all_preds = []
            all_labels = []
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Apply mixup for training phase
                    if phase == 'train' and mixup_alpha > 0:
                        inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                        
                        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                            outputs = model(inputs_mixed)
                            loss = mixup_criterion(criterion_smooth, outputs, labels_a, labels_b, lam)
                            
                            # For accuracy calculation with mixup
                            _, preds = torch.max(outputs, 1)
                            running_corrects += (lam * torch.sum(preds == labels_a.data) + 
                                              (1 - lam) * torch.sum(preds == labels_b.data))
                    else:
                        # Standard training/validation
                        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                            outputs = model(inputs)
                            loss = criterion_smooth(outputs, labels)
                            
                            _, preds = torch.max(outputs, 1)
                            running_corrects += torch.sum(preds == labels.data)
                            
                            # Collect predictions for metrics
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Clear cache to prevent OOM
                torch.cuda.empty_cache()
            
            # Compute epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Calculate additional metrics for validation
            if phase == 'val' and len(all_preds) > 0:
                epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
                history[f'{phase}_f1'].append(epoch_f1)
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Save best model and check early stopping
            if phase == 'val':
                # Use F1 score for tracking best model
                current_metric = epoch_f1 if len(all_preds) > 0 else epoch_acc
                if current_metric > best_f1:
                    best_acc = epoch_acc
                    best_f1 = current_metric
                    best_epoch = epoch
                    best_model_wts = model.state_dict()
                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1
                
                # Check for early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered! No improvement for {patience} epochs.")
                    print(f'Best val Acc: {best_acc:.4f}, Best F1: {best_f1:.4f} at epoch {best_epoch+1}')
                    break
        
        # Check if early stopping triggered
        if patience_counter >= patience:
            break
            
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()
        
        print()
    
    # Training complete
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}, Best F1: {best_f1:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, test_loader, device, use_tta=True):
    """Evaluate model on test set and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Use test-time augmentation if enabled
            if use_tta:
                outputs = test_time_augmentation(model, inputs, device)
            else:
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
    class_metrics = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        class_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)[i]
        class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)[i]
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)[i]
        print(f"{emotion}: Precision={class_prec:.4f}, Recall={class_rec:.4f}, F1={class_f1:.4f}")
        class_metrics[emotion] = {'precision': class_prec, 'recall': class_rec, 'f1': class_f1}
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'conf_matrix': conf_mat.tolist(),
        'class_metrics': class_metrics
    }

# Define manual conversion function for pixel values
def convert_image(img):
    """Convert image from uint8 to float32 and scale from [0, 255] to [0, 1]"""
    return img.float() / 255.0

# Function to gradually unfreeze model layers
def freeze_layers(model, unfreeze_last_n=0):
    """Freeze all layers except the last n layers"""
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Get list of all parameters
    if hasattr(model, 'base_model'):
        # For EmotionAttentionNetwork
        all_params = list(model.parameters())
        start_idx = max(0, len(all_params) - unfreeze_last_n)
        for param in all_params[start_idx:]:
            param.requires_grad = True
    else:
        # For regular EfficientNet
        modules = list(model.children())
        # Always unfreeze the classifier
        for param in modules[-1].parameters():
            param.requires_grad = True
        
        # Unfreeze additional layers if requested
        if unfreeze_last_n > 1:
            additional_layers = modules[-unfreeze_last_n:-1]
            for layer in additional_layers:
                for param in layer.parameters():
                    param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")

def create_model(model_name, num_classes=7, pretrained=True):
    """Create model based on specified architecture."""
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),  # Add dropout for regularization
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        print(f"Model {model_name} not supported. Using EfficientNet-B0")
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),  # Add dropout for regularization
            nn.Linear(num_ftrs, num_classes)
        )
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 model on RAF-DB dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to RAF-DB dataset directory')
    parser.add_argument('--label_file', type=str, required=True, help='Path to RAF-DB label file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for TensorBoard logs')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        choices=['efficientnet_b0'], 
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimization')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--class_balance', action='store_true', help='Use class balancing')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms_v2.ToDtype(torch.float32),  # Convert to float and scale to [0,1]
        transforms_v2.Resize((224, 224), antialias=True),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomRotation(10),
        transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms_v2.ToDtype(torch.float32),  # Convert to float and scale to [0,1]
        transforms_v2.Resize((224, 224), antialias=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = EmotionDataset(args.data_dir, pd.read_csv(args.label_file), transform=None)
    
    # Regular random split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Calculate class weights for balancing if requested
    class_weights = None
    if args.class_balance:
        try:
            # Get class distribution
            class_counts = torch.zeros(len(EMOTION_CLASSES))
            for idx in range(len(dataset)):
                img_name = str(dataset.labels_df.iloc[idx][dataset.filename_col])
                numeric_label = int(dataset.labels_df.iloc[idx][dataset.label_col])
                emotion_str = RAFDB_EMOTION_MAPPING[numeric_label]
                label = dataset.label_to_idx[emotion_str]
                class_counts[label] += 1
            
            # Calculate weights
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(EMOTION_CLASSES)
            class_weights = class_weights.to(device)
            
            print("Class weights:", class_weights)
        except Exception as e:
            print(f"Warning: Error calculating class weights: {e}")
            class_weights = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Create model
    model = create_model(args.model, num_classes=len(EMOTION_CLASSES), pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    
    # Train model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        device, num_epochs=args.epochs, 
        mixup_alpha=args.mixup_alpha,
        early_stopping_patience=args.early_stopping,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'model_name': args.model,
        'num_classes': len(EMOTION_CLASSES)
    }, model_path)
    print(f"Saved model to {model_path}")

if __name__ == '__main__':
    main()