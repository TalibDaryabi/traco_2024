# losses/functions.py
# Contains core loss functions for training and evaluation.

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    """
    Compute the Dice loss between predictions and target masks.
    
    Args:
        pred (torch.Tensor): Predicted tensor (batch, channels, height, width).
        target (torch.Tensor): Target tensor (batch, channels, height, width).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        torch.Tensor: Mean Dice loss across the batch.
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / 
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.9):
    """
    Compute combined loss (BCE + Dice) and update metrics.
    
    Args:
        pred (torch.Tensor): Predicted logits (before sigmoid).
        target (torch.Tensor): Target tensor.
        metrics (dict): Dictionary to store running metrics (bce, dice, loss).
        bce_weight (float): Weight for BCE loss in the combined loss.
    
    Returns:
        torch.Tensor: Combined loss.
    
    Raises:
        ValueError: If input shapes do not match.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes do not match: {pred.shape} vs {target.shape}")
    
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    # Update metrics using item() for scalar values to avoid GPU memory issues
    metrics['bce'] += bce.item() * target.size(0)
    metrics['dice'] += dice.item() * target.size(0)
    metrics['loss'] += loss.item() * target.size(0)

    return loss