"""
Script to perform hyperparameter tuning for the head regression model using Optuna.
"""
import os
import sys
import logging
from pathlib import Path
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import optuna
from optuna.trial import TrialState

sys.path.append(str(Path(__file__).parent.parent))
from models.head_regression.losses import mse_loss, l1_loss, huber_loss, wing_loss, weighted_loss
from scripts.train_head_regressor import HeadRegressionDataset, HeadRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def objective(trial):
    # Hyperparameter search space
    data_dir = 'data/head_regression'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    img_size = trial.suggest_categorical('img_size', [64, 96, 128])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    arch = trial.suggest_categorical('arch', ['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'])
    loss_name = trial.suggest_categorical('loss', ['mse_loss', 'l1_loss', 'huber_loss', 'wing_loss'])
    data_aug = trial.suggest_categorical('data_augmentation', [False, True])
    epochs = 20
    val_split = 0.1
    early_stop_patience = 5

    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            *base_transform
        ])
    else:
        train_transform = transforms.Compose(base_transform)
    val_transform = transforms.Compose(base_transform)

    dataset = HeadRegressionDataset(images_dir, labels_dir, transform=None)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    loss_module = importlib.import_module(f"models.head_regression.losses.{loss_name}")
    criterion = loss_module.get_loss()
    model = HeadRegressor(arch=arch).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            break
    return best_val_loss

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    logger.info(f"Study statistics: {len(study.trials)} trials, {len(pruned_trials)} pruned, {len(complete_trials)} complete")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    # Save best parameters to file
    with open('best_parameters_optuna.txt', 'w') as f:
        f.write(f"value: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main() 