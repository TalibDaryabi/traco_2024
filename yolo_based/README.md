# Hexbug Detection and Head Localization Pipeline

## Project Status and Roadmap

This repository implements a robust, competition-ready pipeline for detecting and tracking hexbugs in video frames, with precise head localization. The project is modular, reproducible, and supports best practices for training, evaluation, and inference.

### **What Has Been Done**
- Data preparation scripts for YOLO and regression datasets (robust, handles all edge cases)
- YOLOv8 bug detector training and validation
- Head regressor model training (ResNet, MobileNet, EfficientNet, etc.)
- Modular loss functions, data augmentation, early stopping, LR scheduling, TensorBoard
- Hyperparameter tuning (Optuna) and cross-validation (with results/plots saved)
- Full pipeline integration: detection → regression → (SORT) tracking → output
- Output CSV with consistent bug IDs (track IDs) for each frame
- All scripts are CLI-driven and documented
- Project is versioned and clean (processed data ignored)

### **What Remains / Optional Improvements**
- (Optional) More advanced post-processing, ensembling, or error analysis
- (Optional) Automated leaderboard/test submission script
- (Optional) More visualizations or reporting

---

## Workflow Overview

| Step                        | Status      | Notes                                  |
|-----------------------------|-------------|----------------------------------------|
| Data Preparation            | ✅ Complete | Robust, all edge cases handled         |
| YOLO Training               | ✅ Complete | Scripted, documented                   |
| Regression Model Training   | ✅ Complete | Modular, CLI, best practices           |
| Hyperparameter Tuning       | ✅ Complete | Optuna, saves best params              |
| Cross-Validation            | ✅ Complete | Saves results, plots, robust           |
| Final Model Training        | ✅ Complete | Clear workflow, reproducible           |
| Visualization               | ✅ Complete | Scripted, documented                   |
| Full Pipeline Integration   | ✅ Complete | End-to-end script, tracking enabled    |
| Test/Leaderboard Evaluation | 🟡 Pending  | To be run after pipeline integration   |
| Optional Improvements       | 🟡 Optional | As needed for competition/performance  |

---

## Step-by-Step Instructions

### 1. **Data Preparation**
Prepare both YOLO and regression datasets:
```sh
python scripts/prepare_data.py
```
- This will create processed data for YOLO and head regression in `data/processed/` and `data/head_regression/`.

### 2. **YOLO Training**
Train the YOLO bug detector:
```sh
python scripts/train_detector.py --epochs 50 --save_dir models/detection
```
- Edit `config/yolo_config.yaml` for more options.
- Produces: `models/detection/hexbug_detector.pt`

### 3. **Head Regression Model: Training & Selection**

#### a. **Hyperparameter Tuning**
```sh
python scripts/tune_head_regressor_optuna.py
```
- Notes best parameters to `best_parameters_optuna.txt`.

#### b. **Cross-Validation**
```sh
python scripts/cross_validate_head_regressor.py --folds 5 --epochs 30 --arch resnet34 --img_size 64 --batch_size 32 --lr 0.001 --loss huber_loss --data_augmentation
```
- Use the best parameters from tuning.
- Results and plot saved to `crossval_results.txt` and `crossval_val_losses.png`.

#### c. **Final Training**
```sh
python scripts/train_head_regressor.py --epochs 50 --arch resnet34 --img_size 64 --batch_size 32 --lr 0.001 --loss huber_loss --data_augmentation --save_dir models/head_regression
```
- Produces: `models/head_regression/best_head_regressor.pt`

**Important:**
- Always use the same `--arch` and `--img_size` for both training and inference. Save these in a config file for reproducibility.

### 4. **Full Pipeline Inference (Detection + Regression + Tracking)**
Run the full pipeline on a new video:
```sh
python scripts/run_full_pipeline.py \
  --video path/to/new_video.mp4 \
  --yolo_model models/detection/hexbug_detector.pt \
  --regressor_model models/head_regression/best_head_regressor.pt \
  --regressor_arch resnet34 \
  --regressor_img_size 64 \
  --output_csv results/pipeline_results.csv
```
- **SORT tracking is always enabled**: every bug in every frame is assigned a consistent `bug_id`.
- Output CSV columns: `frame`, `bug_id`, `head_x`, `head_y`, `confidence`, `class_id`.
- **You must use the same architecture and image size as in training.**

### 5. **Visualization**
Visualize regression predictions:
```sh
python scripts/visualize_regression_predictions.py \
  --model_path models/head_regression/best_head_regressor.pt \
  --data_dir data/head_regression \
  --arch resnet34 \
  --img_size 64 \
  --num_samples 16 \
  --save_dir results/
```

---

## **Best Practices & Options**
- Always match architecture and image size between training and inference.
- Use cross-validation and Optuna for robust model selection.
- Use TensorBoard (`--tensorboard`) for live monitoring.
- Visualize predictions to check for systematic errors.
- Save your best parameters in a config file for reproducibility.
- Processed data is ignored by git for a clean repo.

---

## **For New Users: Quick Start**
1. Prepare data: `python scripts/prepare_data.py`
2. Train YOLO: `python scripts/train_detector.py --epochs 50 --save_dir models/detection`
3. Tune and train regressor: see steps 3a–3c above
4. Run full pipeline: see step 4 above
5. Visualize and analyze results

---

## **FAQ**
- **Q: Why do I need to specify architecture and image size at inference?**
  - A: The model weights only work with the same architecture and input size as used in training. Always use the same values for both.
- **Q: How do I get consistent bug IDs?**
  - A: SORT tracking is always enabled in the pipeline, so each bug is assigned a unique, consistent ID across frames.
- **Q: Can I process a folder of videos?**
  - A: The current script processes one video at a time. For batch processing, adapt the script or ask for an extension.

---

## **Contact & Contributions**
- For questions, improvements, or contributions, open an issue or pull request.

---

## Overview

This project implements a two-stage pipeline for robust hexbug detection and precise head localization:

1. **YOLO Bug Detection**: Detects the whole bug using bounding boxes generated from head position annotations.
2. **Head Regression Model**: Given a detected bug region, predicts the precise (x, y) head position within the crop.

This approach leverages the strengths of object detection for robust bug localization and regression for high-accuracy head localization.

## Project Structure
```
yolo_based/
├── data/
│   ├── raw/              # Original training videos and annotations
│   │   ├── training01.mp4
│   │   ├── training01.csv
│   │   ├── training02.mp4
│   │   └── training02.csv
│   │   └── ...
│   │
│   ├── training/         # Active training data (copied from raw)
│   │   ├── training01.mp4
│   │   ├── training01.csv
│   │   └── ...
│   │
│   └── processed/        # Processed data for YOLO training
│       ├── images/
│       │   ├── train/    # Training frames
│       │   ├── val/      # Validation frames
│       │   └── test/     # Test frames
│       └── labels/
│           ├── train/    # YOLO format annotations
│           ├── val/      # YOLO format annotations
│           └── test/     # YOLO format annotations
│
├── models/
│   ├── detection/        # YOLOv8 models
│   │   └── yolo_detector.py
│   ├── tracking/         # SORT implementation
│   │   └── sort_tracker.py
│   └── head_regression/  # Head position regression
│
├── utils/
│   ├── data_processing.py  # Data preparation utilities
│   ├── visualization.py    # Visualization tools
│   └── tracking.py         # Tracking utilities
│
├── config/
│   ├── yolo_config.yaml    # YOLOv8 configuration
│   └── model_config.yaml   # Model parameters
│
└── scripts/
    ├── prepare_data.py     # Data preparation script
    ├── train_detector.py   # YOLO training script
    ├── train_head_regressor.py  # Head regression training
    └── train_pipeline.py   # Full pipeline training
```

## Data Formats

### Raw Data
- **Videos**: MP4 format, named as `trainingXX.mp4` (XX: 01-07)
- **Annotations**: CSV format, named as `trainingXX.csv`
  - Columns: `t` (frame number), `x` (x-coordinate), `y` (y-coordinate)

### Processed Data
- **Images**: JPG format, extracted frames from videos
  - Naming: `frame_XXXXXX.jpg` (XXXXXX: frame number)
- **Labels**: TXT format, YOLO annotation format
  - Naming: `frame_XXXXXX.txt` (matching image name)
  - Format: `class_id x_center y_center width height`
  - All values are normalized to [0,1]

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- FilterPy (for SORT tracker)
- SciPy (for Hungarian algorithm)

## Installation
```bash
pip install -r requirements.txt
```

## Data Preparation

Run the following command to prepare both datasets:

```sh
cd yolo_based
python scripts/prepare_data.py
```

This will:
- **Prepare YOLO training data** in `data/processed/`:
  - `images/train`, `images/val`, `images/test`: All frames from all videos, with unique names.
  - `labels/train`, `labels/val`, `labels/test`: YOLO-format bounding box labels, generated as fixed-size boxes centered on the annotated head position.
- **Prepare head regression data** in `data/head_regression/`:
  - `images/`: Cropped bug regions centered on the head.
  - `labels/`: Text files with normalized head (x, y) position within each crop.

## Pipeline Steps

1. **YOLO Training**
   - Train a YOLO model on the generated bounding box data to detect bugs in frames.
2. **Head Regression Training**
   - Train a regression model (e.g., a small CNN) on the cropped bug images to predict the normalized head position.
3. **Inference Pipeline**
   - Run YOLO to detect bugs in new frames.
   - For each detected bug, crop the region and run the regression model to get the head position.
   - Map the predicted (x, y) back to the original image coordinates.

## Next Steps

- Train the YOLO model using `scripts/train_detector.py`.
- Train a regression model on `data/head_regression/` (a training script will be provided).
- Integrate both models for full inference and tracking.

## Notes
- The bounding boxes for YOLO are generated automatically from head positions using a fixed size (default: 40x40 pixels).
- The regression dataset provides precise head localization for each bug instance.
- This two-stage approach is recommended for highest accuracy in head localization tasks.

## Model Architecture

### 1. Detection (YOLOv8-nano)
- Input: 640x640 RGB images
- Output: Bounding boxes with confidence scores
- Features:
  - Efficient backbone
  - Feature pyramid network
  - Multi-scale detection

### 2. Tracking (SORT)
- Input: Detection results
- Output: Tracked objects with IDs
- Features:
  - Kalman filter for motion prediction
  - IoU-based association
  - Track management

### 3. Head Regression (Coming Soon)
- Input: Cropped hexbug regions
- Output: Precise head coordinates
- Features:
  - ResNet backbone
  - Regression head
  - Multi-task learning

## Performance Metrics
- Detection: mAP50 on validation set
- Tracking: MOTA (Multiple Object Tracking Accuracy)
- Head Regression: Mean Euclidean Distance

## Usage
```python
from models.detection.yolo_detector import YOLODetector
from models.tracking.sort_tracker import SORTTracker

# Initialize models
detector = YOLODetector()
tracker = SORTTracker()

# Process video
results = process_video(video_path, detector, tracker)
```

## Development Status
- [x] Data preparation
- [x] YOLO detector implementation
- [x] SORT tracker implementation
- [ ] Head regression model
- [ ] Full pipeline integration
- [ ] Visualization tools
- [ ] Evaluation metrics

## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Usage Guide: Full Pipeline

## 1. Data Preparation
Prepare both YOLO and regression datasets:
```sh
python scripts/prepare_data.py
```
- This will create processed data for YOLO and head regression in `data/processed/` and `data/head_regression/`.

## 2. YOLO Training
Train the YOLO bug detector:
```sh
python scripts/train_detector.py --epochs 50
```
- Edit `config/yolo_config.yaml` for more options.

## 3. Head Regression Model: Full Training & Selection Workflow

To achieve the best possible head localization accuracy, follow these three steps:

### a. Hyperparameter Tuning (Find the Best Parameters)

Use Optuna to automatically search for the best hyperparameters for your head regression model. This will try different architectures, learning rates, batch sizes, loss functions, and augmentation settings.

```sh
python scripts/tune_head_regressor_optuna.py
```
- **What to do:**  
  - Let the script run all trials.
  - At the end, look for the log output:
    ```
    Best trial:
      Value: 0.01234
      Params:
        arch: resnet34
        img_size: 64
        batch_size: 32
        lr: 0.001
        loss: huber_loss
        data_augmentation: True
    ```
  - **Note down the best parameters** (architecture, image size, batch size, learning rate, loss, augmentation, etc.).

### b. Cross-Validation (Estimate True Performance)

Evaluate the model with your chosen parameters using cross-validation. This gives a robust estimate of how well your model will generalize.

```sh
python scripts/cross_validate_head_regressor.py \
  --folds 5 \
  --epochs 30 \
  --arch resnet34 \
  --img_size 64 \
  --batch_size 32 \
  --lr 0.001 \
  --loss huber_loss \
  --data_augmentation
```
- **What to do:**  
  - Use the parameters you found in step (a).
  - At the end, note the **mean and standard deviation of the validation loss** (printed in the log).
  - This tells you how stable and reliable your model is.

### c. Final Training (Train the Best Model for Inference)

Now, train your final model using all the data and the best parameters you found.

1. **(Optional but recommended)**: Save your best parameters in a config file for reproducibility, e.g. `config/model_config.yaml`:
    ```yaml
    arch: resnet34
    img_size: 64
    batch_size: 32
    lr: 0.001
    loss: huber_loss
    data_augmentation: true
    epochs: 50
    ```
2. **Run the final training:**
    ```sh
    python scripts/train_head_regressor.py \
      --epochs 50 \
      --arch resnet34 \
      --img_size 64 \
      --batch_size 32 \
      --lr 0.001 \
      --loss huber_loss \
      --data_augmentation
    ```
    - Adjust `--epochs` as needed.
    - This will save your trained model (e.g., `models/head_regression/best_head_regressor.pt`).

---

**Summary Table:**

| Step | Script | What to Note/Do |
|------|--------|-----------------|
| a | `tune_head_regressor_optuna.py` | Note best params from log |
| b | `cross_validate_head_regressor.py` | Note mean/std val loss |
| c | `train_head_regressor.py` | Use best params, train final model |

**Best Practices:**
- Always use cross-validation to check for overfitting.
- Save your best parameters for reproducibility.
- Use TensorBoard (`--tensorboard`) for live monitoring.
- Visualize predictions with `visualize_regression_predictions.py` to check for systematic errors.

## 4. Visualization of Regression Predictions
Visualize model predictions vs. ground truth:
```sh
python scripts/visualize_regression_predictions.py \
  --model_path models/head_regression/best_head_regressor.pt \
  --data_dir data/head_regression \
  --arch resnet34 \
  --img_size 64 \
  --num_samples 16 \
  --save_dir results/
```
- Shows and saves a grid of images with predicted (red dot) and true (green X) head positions.

## 7. How to Compare Results
- Use validation loss, cross-validation mean/std, and visual inspection.
- Use TensorBoard (`--tensorboard`) for live loss/metric curves.
- Try different architectures, loss functions, and augmentation.
- Use Optuna for automated hyperparameter search.

## 8. Best Practices
- Use data augmentation for better generalization.
- Use early stopping and LR scheduling to avoid overfitting.
- Use cross-validation and Optuna for robust model selection.
- Always visualize predictions to check for systematic errors.
- Clip bounding boxes to avoid corrupt labels (already implemented).
- Use TensorBoard for monitoring.

## Example Order of Execution
1. Prepare data: `python scripts/prepare_data.py`
2. Train YOLO: `python scripts/train_detector.py --epochs 50`
3. Train regression: `python scripts/train_head_regressor.py ...`
4. Cross-validate: `python scripts/cross_validate_head_regressor.py ...`
5. Hyperparameter tuning: `python scripts/tune_head_regressor_optuna.py`
6. Visualize predictions: `python scripts/visualize_regression_predictions.py ...` 