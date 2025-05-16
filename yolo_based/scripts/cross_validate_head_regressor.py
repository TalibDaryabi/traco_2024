"""
Script to perform K-fold cross-validation for the head regression model.
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from models.head_regression.losses import mse_loss, l1_loss, huber_loss, wing_loss, weighted_loss
from scripts.train_head_regressor import HeadRegressionDataset, HeadRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, scaler=None, writer=None, epoch=0):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    val_loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item() * images.size(0)
    val_loss = running_val_loss / len(val_loader.dataset)

    if writer:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    return train_loss, val_loss

def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for head regression model')
    parser.add_argument('--data_dir', type=str, default='data/head_regression', help='Head regression data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','mobilenet_v2','efficientnet_b0'], help='Model architecture')
    parser.add_argument('--loss', type=str, default='mse_loss', help='Loss function module name (in models/head_regression/losses)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size (square)')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none','plateau','step'], help='LR scheduler type')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss (for imbalanced data)')
    parser.add_argument('--data_augmentation', action='store_true', help='Enable data augmentation for training set')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    images_dir = os.path.join(args.data_dir, 'images')
    labels_dir = os.path.join(args.data_dir, 'labels')
    base_transform = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    if args.data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            *base_transform
        ])
        logger.info("Data augmentation enabled: RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation")
    else:
        train_transform = transforms.Compose(base_transform)
    val_transform = transforms.Compose(base_transform)
    dataset = HeadRegressionDataset(images_dir, labels_dir, transform=None)
    indices = np.arange(len(dataset))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        logger.info(f"Fold {fold+1}/{args.folds}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

        # Import loss function
        loss_module = importlib.import_module(f"models.head_regression.losses.{args.loss}")
        if args.weighted_loss:
            criterion = loss_module.get_loss(weight=torch.tensor([1.0, 1.0]))
        else:
            criterion = loss_module.get_loss()
        logger.info(f"Using loss: {args.loss}")

        model = HeadRegressor(arch=args.arch).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if args.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        elif args.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        else:
            scheduler = None
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        writer = SummaryWriter(f'runs/fold_{fold+1}') if args.tensorboard else None

        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, val_loss = train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, scaler, writer, epoch)
            logger.info(f"Fold {fold+1} Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                break
            if scheduler:
                if args.lr_scheduler == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        fold_val_losses.append(best_val_loss)
        if writer:
            writer.close()
        logger.info(f"Fold {fold+1} best val loss: {best_val_loss:.6f}")
    logger.info(f"Cross-validation complete. Mean val loss: {np.mean(fold_val_losses):.6f}, Std: {np.std(fold_val_losses):.6f}")
    for i, loss in enumerate(fold_val_losses):
        logger.info(f"Fold {i+1} val loss: {loss:.6f}")
    # Save mean and std to file
    with open('crossval_results.txt', 'w') as f:
        f.write(f"mean_val_loss: {np.mean(fold_val_losses):.6f}\n")
        f.write(f"std_val_loss: {np.std(fold_val_losses):.6f}\n")
        for i, loss in enumerate(fold_val_losses):
            f.write(f"fold_{i+1}_val_loss: {loss:.6f}\n")
    # Plot and save validation losses
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.bar(range(1, len(fold_val_losses)+1), fold_val_losses, color='skyblue', label='Fold Val Loss')
        plt.axhline(np.mean(fold_val_losses), color='red', linestyle='--', label=f'Mean ({np.mean(fold_val_losses):.4f})')
        plt.fill_between(range(1, len(fold_val_losses)+1),
                         np.mean(fold_val_losses)-np.std(fold_val_losses),
                         np.mean(fold_val_losses)+np.std(fold_val_losses),
                         color='red', alpha=0.2, label='Std Dev')
        plt.xlabel('Fold')
        plt.ylabel('Validation Loss')
        plt.title('Cross-Validation Fold Validation Losses')
        plt.legend()
        plt.tight_layout()
        plt.savefig('crossval_val_losses.png')
        plt.close()
    except ImportError:
        logger.warning('matplotlib not installed, skipping plot of cross-validation losses.')

if __name__ == '__main__':
    main() 