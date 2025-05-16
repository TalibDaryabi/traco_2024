"""
YOLOv8 detector implementation for hexbug detection.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class YOLODetector:
    """
    YOLOv8 detector for hexbug detection.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path (str, optional): Path to pretrained model. If None, uses default YOLOv8n.
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            device (str, optional): Device to run model on ('cuda' or 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize model
        if model_path is None:
            self.model = YOLO('yolov8n.pt')
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            self.model = YOLO(model_path)
            
        # Move model to device
        self.model.to(self.device)
        
        logging.info(f"YOLO detector initialized on {self.device}")
        
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the YOLO detector.
        
        Args:
            data_yaml (str): Path to data configuration YAML
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            img_size (int): Input image size
            save_dir (str, optional): Directory to save model
            
        Returns:
            Dict: Training results
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Data config not found at {data_yaml}")
            
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'project': save_dir if save_dir else 'runs/train',
            'name': 'hexbug_detector',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True
        }
        
        # Train model
        results = self.model.train(**train_args)
        
        logging.info("Training completed successfully!")
        return results
        
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect hexbugs in an image.
        
        Args:
            image (np.ndarray): Input image (H, W, C)
            conf_threshold (float, optional): Override default confidence threshold
            iou_threshold (float, optional): Override default IoU threshold
            
        Returns:
            List[Dict]: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] in pixel coordinates
                - confidence: Detection confidence
                - class_id: Class ID (0 for hexbug)
        """
        # Use default thresholds if not specified
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]
        
        # Process results
        detections = []
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id
            })
            
        return detections
        
    def save(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path (str): Path to save model
        """
        self.model.save(path)
        logging.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """
        Load model weights.
        
        Args:
            path (str): Path to saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        self.model = YOLO(path)
        self.model.to(self.device)
        logging.info(f"Model loaded from {path}")

if __name__ == '__main__':
    # Example usage
    detector = YOLODetector()
    
    # Train model
    results = detector.train(
        data_yaml='../config/yolo_config.yaml',
        epochs=100,
        batch_size=16,
        img_size=640,
        save_dir='../models/detection'
    )
    
    # Save model
    detector.save('../models/detection/hexbug_detector.pt') 