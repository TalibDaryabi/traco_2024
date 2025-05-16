"""
SORT (Simple Online Realtime Tracking) implementation for hexbug tracking.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_bbox_to_z(bbox: List[float]) -> np.ndarray:
    """
    Convert bounding box to state vector.
    
    Args:
        bbox: [x1, y1, x2, y2] format
        
    Returns:
        np.ndarray: State vector [x, y, s, r, x', y', s']
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r, 0, 0, 0]).reshape((7, 1))

def convert_x_to_bbox(x: np.ndarray) -> List[float]:
    """
    Convert state vector to bounding box.
    
    Args:
        x: State vector [x, y, s, r, x', y', s']
        
    Returns:
        List[float]: Bounding box [x1, y1, x2, y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return [x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]

class KalmanBoxTracker:
    """
    Kalman filter tracker for a single object.
    """
    
    count = 0
    
    def __init__(self, bbox: List[float]):
        """
        Initialize tracker with bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] format
        """
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        # Initial state
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        # Track ID and age
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox: List[float]) -> None:
        """
        Update tracker with new bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] format
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        
    def predict(self) -> List[float]:
        """
        Predict next state.
        
        Returns:
            List[float]: Predicted bounding box [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self) -> List[float]:
        """
        Get current state.
        
        Returns:
            List[float]: Current bounding box [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)

class SORTTracker:
    """
    SORT tracker for multiple objects.
    """
    
    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize SORT tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep track of lost objects
            min_hits (int): Minimum number of hits to confirm a track
            iou_threshold (float): IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: Detection confidence
                - class_id: Class ID
                
        Returns:
            List[Dict]: List of tracked objects, each containing:
                - bbox: [x1, y1, x2, y2]
                - track_id: Track ID
                - confidence: Detection confidence
                - class_id: Class ID
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]]['bbox'])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'])
            self.trackers.append(trk)
            
        # Get final set of tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append({
                    'bbox': d,
                    'track_id': trk.id,
                    'confidence': 1.0,  # We don't have confidence for tracked objects
                    'class_id': 0  # Assuming all tracked objects are hexbugs
                })
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        return ret

def iou(bb_det: List[float], bb_trk: List[float]) -> float:
    """
    Calculate IoU between detection and tracker.
    
    Args:
        bb_det: Detection bounding box [x1, y1, x2, y2]
        bb_trk: Tracker bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU score
    """
    # Get coordinates
    x1 = max(bb_det[0], bb_trk[0])
    y1 = max(bb_det[1], bb_trk[1])
    x2 = min(bb_det[2], bb_trk[2])
    y2 = min(bb_det[3], bb_trk[3])
    
    # Calculate area
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)
    intersection = w * h
    
    # Calculate union
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    union = area_det + area_trk - intersection
    
    return intersection / union if union > 0 else 0

def associate_detections_to_trackers(
    detections: List[Dict],
    trackers: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate detections to trackers using IoU.
    
    Args:
        detections: List of detections
        trackers: Array of tracker states
        iou_threshold: IoU threshold for association
        
    Returns:
        Tuple containing:
            - List of matched indices (det_idx, trk_idx)
            - List of unmatched detection indices
            - List of unmatched tracker indices
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
        
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det['bbox'], trk)
            
    # Use Hungarian algorithm to find optimal assignment
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)
    
    # Filter out matches with low IoU
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, unmatched_detections, unmatched_trackers

if __name__ == '__main__':
    # Example usage
    tracker = SORTTracker()
    
    # Example detections
    detections = [
        {'bbox': [100, 100, 120, 120], 'confidence': 0.9, 'class_id': 0},
        {'bbox': [200, 200, 220, 220], 'confidence': 0.8, 'class_id': 0}
    ]
    
    # Update tracker
    tracks = tracker.update(detections)
    print(f"Tracks: {tracks}") 