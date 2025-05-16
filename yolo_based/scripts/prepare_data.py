"""
Script to prepare data for YOLO training.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.data_processing import prepare_yolo_dataset, generate_head_regression_dataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to prepare the dataset.
    """
    # Define paths
    video_dir = os.path.join(project_root, 'data', 'raw')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataset
    prepare_yolo_dataset(
        video_dir=video_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        frame_interval=1
    )
    
    logging.info("YOLO data preparation completed!")

    # Prepare head regression dataset
    regression_output_dir = os.path.join(project_root, 'data', 'head_regression')
    generate_head_regression_dataset(
        video_dir=video_dir,
        output_dir=regression_output_dir,
        crop_size=40,
        frame_interval=1
    )
    logging.info("Head regression data preparation completed!")

if __name__ == '__main__':
    main() 