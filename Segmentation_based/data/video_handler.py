# data/video_handler.py
# Handles video loading, preprocessing, and model inference.

import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.processing import get_densest_numpy_patches

def process_video(video_path, model, device, num_hex=3):
    """
    Process a video to extract positions using the model.
    
    Args:
        video_path (str): Path to the video file.
        model (nn.Module): Trained model for inference.
        device (torch.device): Device to run the model.
        num_hex (int): Number of positions to extract per frame.
    
    Returns:
        tuple: List of unordered positions, frame width, frame height, frame count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fc = 0
    ret = True
    list_max_value_unordered = []
    transform = transforms.Compose([transforms.ToTensor()])
    frame_width, frame_height = 0, 0
    
    while fc < frame_count and ret:
        ret, buf = cap.read()
        if not ret:
            break
        im_rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        im_rgb = cv2.resize(im_rgb, dsize=(256, 256))
        frame_width, frame_height, _ = buf.shape
        im_rgb = transform(im_rgb)
        img = torch.unsqueeze(im_rgb, 0).to(device)
        with torch.no_grad():
            result = model(img)
        result = torch.squeeze(result, 0).squeeze(0).cpu().detach().numpy()
        positions = get_densest_numpy_patches(result, num_hex=num_hex)
        list_max_value_unordered.append(positions)
        fc += 1
    
    cap.release()
    print(f"Processed {fc} frames")
    return list_max_value_unordered, frame_width, frame_height, fc