"""
Script to visualize predictions vs. ground truth for the head regression model.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
import importlib

sys.path.append(str(Path(__file__).parent.parent))
from scripts.train_head_regressor import HeadRegressionDataset, HeadRegressor

def visualize_predictions(model, dataset, device, num_samples=16, img_size=64, save_dir=None):
    model.eval()
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image).cpu().numpy()[0]
        img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # unnormalize
        ax = axes[i]
        ax.imshow(img_np)
        # Denormalize coordinates
        px, py = pred[0] * img_size, pred[1] * img_size
        tx, ty = target[0].item() * img_size, target[1].item() * img_size
        ax.scatter([px], [py], c='r', label='Pred', s=60)
        ax.scatter([tx], [ty], c='g', label='GT', s=60, marker='x')
        ax.set_title(f"Pred: ({px:.1f},{py:.1f})\nGT: ({tx:.1f},{ty:.1f})")
        ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'regression_predictions.png'))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize regression predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pt)')
    parser.add_argument('--data_dir', type=str, default='data/head_regression', help='Head regression data directory')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18','resnet34','resnet50','mobilenet_v2','efficientnet_b0'], help='Model architecture')
    parser.add_argument('--img_size', type=int, default=64, help='Image size (square)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_transform = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    transform = transforms.Compose(base_transform)
    images_dir = os.path.join(args.data_dir, 'images')
    labels_dir = os.path.join(args.data_dir, 'labels')
    dataset = HeadRegressionDataset(images_dir, labels_dir, transform=transform)
    model = HeadRegressor(arch=args.arch).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    visualize_predictions(model, dataset, device, num_samples=args.num_samples, img_size=args.img_size, save_dir=args.save_dir)

if __name__ == '__main__':
    main() 