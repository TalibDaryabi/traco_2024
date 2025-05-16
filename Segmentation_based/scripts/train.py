# scripts/train.py
# Example training script.

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import logging
from torchvision import transforms
from segmentation_based.models.resnet_unet import ResNetUNet
from segmentation_based.losses.loss_function import calc_loss
from segmentation_based.utils.metrices import print_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HexbugDataset(Dataset):
    def __init__(self, video_path, annotation_path, transform=None, target_size=(256, 256)):
        """
        Dataset for hexbug tracking.
        
        Args:
            video_path (str): Path to the video file
            annotation_path (str): Path to the annotation CSV file
            transform (callable, optional): Optional transform to be applied on frames
            target_size (tuple): Target size for resizing (width, height)
        """
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.target_size = target_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        self.annotations = pd.read_csv(annotation_path)
        if 'Unnamed: 0' in self.annotations.columns:
            self.annotations = self.annotations.drop(columns=['Unnamed: 0'])
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
            
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create frame to annotation mapping
        self.frame_annotations = {}
        for _, row in self.annotations.iterrows():
            frame_idx = int(row['t'])
            if frame_idx not in self.frame_annotations:
                self.frame_annotations[frame_idx] = []
            # Scale coordinates to target size
            x = float(row['x']) * target_size[0] / self.original_width
            y = float(row['y']) * target_size[1] / self.original_height
            self.frame_annotations[frame_idx].append({
                'hexbug': int(row['hexbug']),
                'x': x,
                'y': y
            })
        
        # Pre-load all frames into memory
        logging.info(f"Pre-loading video: {video_path}")
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        for _ in tqdm(range(self.frame_count), desc="Loading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to target size
            frame = cv2.resize(frame, target_size)
            self.frames.append(frame)
        cap.release()
        logging.info(f"Loaded {len(self.frames)} frames")

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        # Get pre-loaded frame
        frame = self.frames[idx]
        
        # Create target mask
        target_mask = np.zeros(self.target_size[::-1], dtype=np.float32)  # Note: OpenCV uses (height, width)
        
        # Add hexbug positions to mask
        if idx in self.frame_annotations:
            for annotation in self.frame_annotations[idx]:
                x, y = int(annotation['x']), int(annotation['y'])
                # Create Gaussian blob around hexbug position
                cv2.circle(target_mask, (x, y), 3, 1.0, -1)  # Reduced radius for smaller image size
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
            target_mask = torch.from_numpy(target_mask).unsqueeze(0)
        
        return {
            'image': frame,
            'mask': target_mask,
            'frame_idx': idx,
            'original_size': (self.original_width, self.original_height),  # Keep original size for post-processing
            'video_name': self.video_name  # Add video name to output
        }

def create_dataloaders(data_dir, batch_size=8, num_workers=4):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir (str): Directory containing training data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get all video files
    video_files = sorted(Path(data_dir).glob('training[0-9][0-9].mp4'))
    
    if not video_files:
        raise RuntimeError(f"No training videos found in {data_dir}")
    
    logging.info(f"Found {len(video_files)} training videos")
    
    # Split into train and validation (80-20 split)
    split_idx = int(len(video_files) * 0.8)
    train_videos = video_files[:split_idx]
    val_videos = video_files[split_idx:]
    
    logging.info(f"Training on {len(train_videos)} videos, validating on {len(val_videos)} videos")
    
    # Create datasets
    train_dataset = torch.utils.data.ConcatDataset([
        HexbugDataset(str(video), str(video.with_suffix('.csv')))
        for video in train_videos
    ])
    
    val_dataset = torch.utils.data.ConcatDataset([
        HexbugDataset(str(video), str(video.with_suffix('.csv')))
        for video in val_videos
    ])
    
    # Create dataloaders with reduced number of workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, num_workers),  # Limit workers
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, num_workers),  # Limit workers
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to train on
    
    Returns:
        dict: Training metrics
    """
    model.train()
    metrics = {'bce': 0, 'dice': 0, 'loss': 0}
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = calc_loss(outputs, masks, metrics)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']/len(train_loader):.4f}",
            'bce': f"{metrics['bce']/len(train_loader):.4f}",
            'dice': f"{metrics['dice']/len(train_loader):.4f}"
        })
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(train_loader)
    
    return metrics

def validate(model, val_loader, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    metrics = {'bce': 0, 'dice': 0, 'loss': 0}
    
    # Create prediction directory
    pred_dir = os.path.join(project_root, 'data', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    # Track last frame for each video
    last_frames = {}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            calc_loss(outputs, masks, metrics)
            
            # Store last frame for each video
            for i in range(images.size(0)):
                video_name = batch['video_name'][i]
                last_frames[video_name] = {
                    'frame': batch['image'][i].cpu().numpy().transpose(1, 2, 0),
                    'true_mask': batch['mask'][i].cpu().numpy().squeeze(),
                    'pred_mask': torch.sigmoid(outputs[i]).cpu().numpy().squeeze(),
                    'frame_idx': batch['frame_idx'][i].item()
                }
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']/len(val_loader):.4f}",
                'bce': f"{metrics['bce']/len(val_loader):.4f}",
                'dice': f"{metrics['dice']/len(val_loader):.4f}"
            })
    
    # Save only the last frame for each video
    for video_name, data in last_frames.items():
        # Denormalize frame
        frame = ((data['frame'] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        
        # Save visualization
        from utils.visualization import save_prediction_visualization
        save_prediction_visualization(
            frame=frame,
            true_mask=data['true_mask'],
            pred_mask=data['pred_mask'],
            video_name=video_name,
            output_dir=pred_dir,
            frame_idx=data['frame_idx']
        )
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(val_loader)
    
    return metrics

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    save_path='my_solution.pth'
):
    """
    Train the model with validation and model saving.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        save_path (str): Path to save the model
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        logging.info(f"Train metrics:")
        logging.info(f"  Loss: {train_metrics['loss']:.4f}")
        logging.info(f"  BCE: {train_metrics['bce']:.4f}")
        logging.info(f"  Dice: {train_metrics['dice']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        logging.info(f"Validation metrics:")
        logging.info(f"  Loss: {val_metrics['loss']:.4f}")
        logging.info(f"  BCE: {val_metrics['bce']:.4f}")
        logging.info(f"  Dice: {val_metrics['dice']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model with validation loss: {best_val_loss:.4f}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    model = ResNetUNet(n_class=1)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=os.path.join(project_root, 'data', 'training'),  # Use absolute path
        batch_size=8,
        num_workers=4
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        save_path=os.path.join(project_root, 'my_solution.pth')  # Save in project root
    )

if __name__ == '__main__':
    main()