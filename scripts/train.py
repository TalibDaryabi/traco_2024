# scripts/train.py
# Example training script.

import torch
from models.resnet_unet import ResNetUNet
from losses.functions import calc_loss
from utils.metrics import print_metrics

def train_model(model, dataloader, device, epochs=1):
    """
    Train the model for a specified number of epochs.
    
    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs to train.
    """
    model.train()
    for epoch in range(epochs):
        metrics = {'bce': 0, 'dice': 0, 'loss': 0}
        for batch in dataloader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = calc_loss(outputs, targets, metrics)
            loss.backward()
        print_metrics(metrics, len(dataloader), f"Epoch {epoch}")