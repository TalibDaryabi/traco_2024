"""
Script to train the YOLO detector.
"""

import os
import sys
from pathlib import Path
import logging
import yaml
import argparse
from typing import Dict, Any, Union

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from models.detection.yolo_detector import YOLODetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_structure() -> bool:
    """
    Check if the required data structure exists.
    
    Returns:
        bool: True if data structure is valid, False otherwise
    """
    # Check data directory
    data_dir = project_root / 'data' / 'processed'
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please run the data preparation script first:")
        logger.info(f"cd {project_root}")
        logger.info("python scripts/prepare_data.py")
        return False
    
    # Check train directory
    train_dir = data_dir / 'images' / 'train'
    if not train_dir.exists():
        logger.error(f"Training directory does not exist: {train_dir}")
        return False
    
    # Check val directory
    val_dir = data_dir / 'images' / 'val'
    if not val_dir.exists():
        logger.error(f"Validation directory does not exist: {val_dir}")
        return False
    
    # Check for images
    train_images = list(train_dir.glob('*.jpg'))
    val_images = list(val_dir.glob('*.jpg'))
    
    if not train_images:
        logger.error(f"No training images found in {train_dir}")
        return False
    
    if not val_images:
        logger.error(f"No validation images found in {val_dir}")
        return False
    
    logger.info(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    return True

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (Union[str, Path]): Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    # Convert to Path object if string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    # Make path absolute if relative
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train YOLO detector')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/yolo_config.yaml',
        help='Path to configuration file'
    )
    return parser.parse_args()

def main():
    """
    Main function to train the YOLO detector.
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Check data structure
        if not check_data_structure():
            raise ValueError("Data structure validation failed")
        
        # Load configuration
        config = load_config(args.config)
        
        # Create model directory
        model_dir = project_root / 'models' / 'detection'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        detector = YOLODetector(
            conf_threshold=config.get('conf_threshold', 0.25),
            iou_threshold=config.get('iou_threshold', 0.45)
        )
        
        logger.info(f"Starting training with {args.epochs} epochs")
        logger.info(f"Using device: {config.get('device', 'auto')}")
        
        # Always resolve config path to absolute
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        results = detector.train(
            data_yaml=str(config_path),
            epochs=args.epochs,
            batch_size=config.get('batch', 16),
            img_size=config.get('imgsz', 640),
            save_dir=str(model_dir)
        )
        
        # Save final model
        model_path = model_dir / 'hexbug_detector.pt'
        detector.save(str(model_path))
        
        logger.info(f"Training completed. Model saved to {model_path}")
        
        # Print metrics
        metrics = results.results_dict
        logger.info("\nTraining Metrics:")
        logger.info(f"mAP50: {metrics.get('metrics/mAP50', 'N/A'):.4f}")
        logger.info(f"mAP50-95: {metrics.get('metrics/mAP50-95', 'N/A'):.4f}")
        logger.info(f"Precision: {metrics.get('metrics/precision', 'N/A'):.4f}")
        logger.info(f"Recall: {metrics.get('metrics/recall', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 