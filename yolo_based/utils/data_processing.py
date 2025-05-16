"""
Data processing utilities for hexbug detection and tracking.
"""

import os
import cv2
import numpy as np
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from tqdm import tqdm  # type: ignore
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1, video_prefix: str = "") -> List[str]:
    """
    Extract frames from video at specified intervals, with unique naming per video.
    Args:
        video_path (str): Path to video file
        output_dir (str): Directory to save frames
        frame_interval (int): Extract every nth frame
        video_prefix (str): Prefix for frame filenames (e.g., video name)
    Returns:
        List[str]: List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_paths = []
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{video_prefix}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        frame_count += 1
    cap.release()
    return frame_paths

def convert_annotations_to_yolo(
    csv_path: str,
    frame_paths: List[str],
    output_dir: str,
    img_width: int,
    img_height: int
) -> None:
    """
    Convert CSV annotations to YOLO format, matching unique frame names. For frames with no annotation, create an empty .txt file.
    Args:
        csv_path (str): Path to CSV annotation file
        frame_paths (List[str]): List of frame paths
        output_dir (str): Directory to save YOLO annotations
        img_width (int): Image width
        img_height (int): Image height
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    frame_annotations = df.groupby('t')
    # Map frame index to annotation group
    annotation_map = {int(frame_idx): group for frame_idx, group in frame_annotations}
    for frame_path in frame_paths:
        # Extract frame index from filename
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        frame_idx_str = frame_name.split('_frame_')[-1]
        try:
            frame_idx = int(frame_idx_str)
        except Exception:
            continue
        yolo_path = os.path.join(output_dir, f"{frame_name}.txt")
        with open(yolo_path, 'w') as f:
            if frame_idx in annotation_map:
                group = annotation_map[frame_idx]
                for _, row in group.iterrows():
                    x = float(row['x'])
                    y = float(row['y'])
                    width = 20
                    height = 20
                    # Clip box to image boundaries
                    x1 = max(x - width / 2, 0)
                    y1 = max(y - height / 2, 0)
                    x2 = min(x + width / 2, img_width)
                    y2 = min(y + height / 2, img_height)
                    # Recompute center and size after clipping
                    box_w = x2 - x1
                    box_h = y2 - y1
                    if box_w <= 0 or box_h <= 0:
                        logging.warning(f"Skipped invalid box (zero area) at {frame_name} (head: {x},{y})")
                        continue
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    norm_width = box_w / img_width
                    norm_height = box_h / img_height
                    # Only write if all normalized values are in [0, 1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_width <= 1 and 0 < norm_height <= 1):
                        logging.warning(f"Skipped out-of-bounds box at {frame_name} (normalized: {x_center},{y_center},{norm_width},{norm_height})")
                        continue
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            # else: leave file empty for frames with no annotation

def prepare_yolo_dataset(
    video_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    frame_interval: int = 1
) -> None:
    """
    Prepare dataset for YOLO training.
    
    Args:
        video_dir (str): Directory containing videos and annotations
        output_dir (str): Directory to save processed dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        frame_interval (int): Extract every nth frame
    """
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    test_img_dir = os.path.join(output_dir, 'images', 'test')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')
    test_label_dir = os.path.join(output_dir, 'labels', 'test')
    
    for dir_path in [train_img_dir, val_img_dir, test_img_dir,
                    train_label_dir, val_label_dir, test_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Find all video files and match with CSVs
    video_files = list(Path(video_dir).glob('training*.mp4'))
    video_csv_pairs = []
    for video_path in video_files:
        csv_path = video_path.with_suffix('.csv')
        if csv_path.exists():
            # Extract numeric part for sorting
            stem = video_path.stem
            num_part = ''.join(filter(str.isdigit, stem))
            try:
                num = int(num_part)
            except Exception:
                num = -1
            video_csv_pairs.append((num, video_path, csv_path))
        else:
            logging.warning(f"No CSV annotation for video: {video_path}")
    # Sort numerically
    video_csv_pairs.sort(key=lambda x: x[0])
    if not video_csv_pairs:
        raise RuntimeError(f"No valid video/csv pairs found in {video_dir}")

    # Split into train/val/test
    n = len(video_csv_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_pairs = video_csv_pairs[:n_train]
    val_pairs = video_csv_pairs[n_train:n_train + n_val]
    test_pairs = video_csv_pairs[n_train + n_val:]

    # Helper for processing
    def process_split(pairs, img_dir, label_dir, split_name):
        for _, video_path, csv_path in tqdm(pairs, desc=f"Processing {split_name}"):
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            video_prefix = video_path.stem
            frame_paths = extract_frames(str(video_path), img_dir, frame_interval, video_prefix=video_prefix)
            convert_annotations_to_yolo(str(csv_path), frame_paths, label_dir, width, height)

    process_split(train_pairs, train_img_dir, train_label_dir, 'train')
    process_split(val_pairs, val_img_dir, val_label_dir, 'val')
    process_split(test_pairs, test_img_dir, test_label_dir, 'test')

    logging.info(f"Dataset preparation completed. Output directory: {output_dir}")
    logging.info(f"Training videos: {len(train_pairs)}")
    logging.info(f"Validation videos: {len(val_pairs)}")
    logging.info(f"Test videos: {len(test_pairs)}")

def generate_head_regression_dataset(
    video_dir: str,
    output_dir: str,
    crop_size: int = 40,
    frame_interval: int = 1
) -> None:
    """
    Generate a dataset for head regression:
    - For each frame with a head annotation, crop a region centered at the head (using crop_size),
      save the crop, and record the normalized head (x, y) position within the crop.
    Args:
        video_dir (str): Directory containing videos and annotations
        output_dir (str): Directory to save regression dataset (images/ and labels/)
        crop_size (int): Size of the square crop (pixels)
        frame_interval (int): Extract every nth frame
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    video_files = list(Path(video_dir).glob('training*.mp4'))
    for video_path in video_files:
        csv_path = video_path.with_suffix('.csv')
        if not csv_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        frame_annotations = df.groupby('t')
        # Load all frames needed
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                if frame_idx in frame_annotations.groups:
                    group = frame_annotations.get_group(frame_idx)
                    for i, row in group.iterrows():
                        x = float(row['x'])
                        y = float(row['y'])
                        # Crop region centered at (x, y)
                        left = int(max(x - crop_size // 2, 0))
                        top = int(max(y - crop_size // 2, 0))
                        right = int(min(x + crop_size // 2, width))
                        bottom = int(min(y + crop_size // 2, height))
                        crop = frame[top:bottom, left:right]
                        crop_h, crop_w = crop.shape[:2]
                        # Skip zero-sized crops
                        if crop_h == 0 or crop_w == 0:
                            logging.warning(f"Zero-sized crop at video {video_path}, frame {frame_idx}, head ({x},{y})")
                            continue
                        # Normalize head position in crop
                        norm_x = (x - left) / crop_w
                        norm_y = (y - top) / crop_h
                        # Save crop and label
                        crop_name = f"{video_path.stem}_frame_{frame_idx:06d}_bug_{i}.jpg"
                        label_name = f"{video_path.stem}_frame_{frame_idx:06d}_bug_{i}.txt"
                        crop_path = os.path.join(output_dir, 'images', crop_name)
                        label_path = os.path.join(output_dir, 'labels', label_name)
                        Image.fromarray(crop).save(crop_path)
                        with open(label_path, 'w') as f:
                            f.write(f"{norm_x:.6f} {norm_y:.6f}\n")
                        saved_count += 1
            frame_idx += 1
        cap.release()
    logging.info(f"Head regression dataset created at {output_dir}")

if __name__ == '__main__':
    # Example usage
    prepare_yolo_dataset(
        video_dir='../data/training',
        output_dir='../data/processed',
        train_ratio=0.8,
        val_ratio=0.1,
        frame_interval=1
    ) 