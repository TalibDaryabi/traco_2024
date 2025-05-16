import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import importlib
import csv

sys.path.append(str(Path(__file__).parent.parent))
from models.detection.yolo_detector import YOLODetector
from scripts.train_head_regressor import HeadRegressor
from models.tracking.sort_tracker import SORTTracker

# Helper: crop and preprocess bug region for regressor

def crop_and_preprocess(image, bbox, img_size):
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    crop_pil = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = cv2.resize(crop_pil, (img_size, img_size))
    crop_pil = torch.from_numpy(crop_pil).float().permute(2, 0, 1) / 255.0
    crop_pil = (crop_pil - 0.5) / 0.5  # Normalize to [-1, 1]
    return crop_pil.unsqueeze(0)  # (1, 3, H, W)

def map_head_to_original(bbox, pred_xy, crop_size, orig_shape):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    px = x1 + pred_xy[0] * w
    py = y1 + pred_xy[1] * h
    # Clip to image
    px = np.clip(px, 0, orig_shape[1] - 1)
    py = np.clip(py, 0, orig_shape[0] - 1)
    return float(px), float(py)

def process_video(
    video_path,
    yolo_model_path,
    regressor_model_path,
    regressor_arch,
    regressor_img_size,
    output_csv,
    yolo_conf=0.25,
    yolo_iou=0.45,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    detector = YOLODetector(model_path=yolo_model_path, conf_threshold=yolo_conf, iou_threshold=yolo_iou, device=device)
    regressor = HeadRegressor(arch=regressor_arch).to(device)
    regressor.load_state_dict(torch.load(regressor_model_path, map_location=device))
    regressor.eval()
    tracker = SORTTracker()  # Always use tracking

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    results = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing frames')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.predict(frame)
        bug_results = []
        for det in detections:
            bbox = det['bbox']
            crop = crop_and_preprocess(frame, bbox, regressor_img_size).to(device)
            with torch.no_grad():
                pred_xy = regressor(crop).cpu().numpy()[0]  # normalized in crop
            px, py = map_head_to_original(bbox, pred_xy, regressor_img_size, frame.shape)
            bug_results.append({
                'bbox': bbox,
                'head_x': px,
                'head_y': py,
                'confidence': det['confidence'],
                'class_id': det['class_id']
            })
        # Always use tracking
        tracked = tracker.update([{'bbox': b['bbox'], 'confidence': b['confidence'], 'class_id': b['class_id']} for b in bug_results])
        for i, trk in enumerate(tracked):
            bug_results[i]['track_id'] = trk['track_id']
        for i, bug in enumerate(bug_results):
            results.append({
                'frame': frame_idx,
                'bug_id': bug.get('track_id', i),
                'head_x': bug['head_x'],
                'head_y': bug['head_y'],
                'confidence': bug['confidence'],
                'class_id': bug['class_id']
            })
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    # Save results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'bug_id', 'head_x', 'head_y', 'confidence', 'class_id'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Run full hexbug detection + head localization pipeline (SORT tracking always enabled)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--yolo_model', type=str, required=True, help='Path to trained YOLO model (.pt)')
    parser.add_argument('--regressor_model', type=str, required=True, help='Path to trained head regressor model (.pt)')
    parser.add_argument('--regressor_arch', type=str, default='resnet34', choices=['resnet18','resnet34','resnet50','mobilenet_v2','efficientnet_b0'], help='Head regressor architecture')
    parser.add_argument('--regressor_img_size', type=int, default=64, help='Input size for regressor')
    parser.add_argument('--output_csv', type=str, default='pipeline_results.csv', help='Output CSV file')
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--yolo_iou', type=float, default=0.45, help='YOLO IoU threshold')
    parser.add_argument('--device', type=str, default=None, help='Device to run models on (cpu or cuda)')
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        yolo_model_path=args.yolo_model,
        regressor_model_path=args.regressor_model,
        regressor_arch=args.regressor_arch,
        regressor_img_size=args.regressor_img_size,
        output_csv=args.output_csv,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        device=args.device
    )

if __name__ == '__main__':
    main() 