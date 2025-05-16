"""
Script to train a head regression model for hexbug head localization.
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import importlib
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HeadRegressionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.samples = []
        for img_file in sorted(self.images_dir.glob('*.jpg')):
            label_file = self.labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                self.samples.append((img_file, label_file))
        logger.info(f"Loaded {len(self.samples)} samples from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            x, y = map(float, line.split())
        if self.transform:
            image = self.transform(image)
        target = torch.tensor([x, y], dtype=torch.float32)
        return image, target

class HeadRegressor(nn.Module):
    def __init__(self, arch='resnet18'):
        super().__init__()
        if arch == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        elif arch == 'resnet34':
            self.backbone = models.resnet34(weights=None)
        elif arch == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        elif arch == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights=None)
            self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
            return
        elif arch == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=None)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
            return
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        return self.backbone(x)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train head regression model')
    parser.add_argument('--data_dir', type=str, default='data/head_regression', help='Head regression data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='models/head_regression', help='Directory to save model')
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
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Patch transforms for train/val
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

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

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    writer = SummaryWriter(args.save_dir) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(1, args.epochs + 1):
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
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_head_regressor.pt'))
            logger.info(f"Saved best model at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break
        # LR scheduling
        if scheduler:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
    if writer:
        writer.close()
    logger.info("Training complete.")

if __name__ == '__main__':
    main() 