import os
import cv2
import numpy as np
import torch
from pathlib import Path

def save_prediction_visualization(
    frame,
    true_mask,
    pred_mask,
    video_name,
    output_dir,
    frame_idx=None
):
    """
    Save visualization of true mask and prediction heatmap.
    
    Args:
        frame (np.ndarray): Original frame (RGB)
        true_mask (np.ndarray): Ground truth mask
        pred_mask (np.ndarray): Model prediction mask
        video_name (str): Name of the video file
        output_dir (str): Directory to save visualizations
        frame_idx (int, optional): Frame index for filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert masks to uint8 for visualization
    true_mask = (true_mask * 255).astype(np.uint8)
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    # Create heatmap for prediction
    pred_heatmap = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
    pred_heatmap = cv2.addWeighted(frame, 0.7, pred_heatmap, 0.3, 0)
    
    # Create visualization for true mask
    true_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # Save visualizations
    base_name = Path(video_name).stem
    if frame_idx is not None:
        base_name = f"{base_name}_frame{frame_idx}"
    
    # Save true mask visualization
    true_path = os.path.join(output_dir, f"{base_name}_true_mask.jpg")
    cv2.imwrite(true_path, cv2.cvtColor(true_vis, cv2.COLOR_RGB2BGR))
    
    # Save prediction heatmap
    pred_path = os.path.join(output_dir, f"{base_name}_pred_heatmap.jpg")
    cv2.imwrite(pred_path, cv2.cvtColor(pred_heatmap, cv2.COLOR_RGB2BGR))
    
    return true_path, pred_path 