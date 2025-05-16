Overall Project Plan: Hexbug Detection & Head Localization Pipeline
1. Data Preparation
Extract frames from videos
Generate YOLO-format bounding boxes (centered on head, fixed size, clipped to image boundaries)
Create regression dataset (cropped bug images, normalized head positions)
Handle missing frames and empty labels
Status:
✅ Completed
Scripts are robust, handle all edge cases, and are documented in the README.
2. YOLO Bug Detection
Train YOLOv8 model on generated bounding box data
Validate and tune using config YAML
Status:
✅ Completed
Training script is debugged, works with both CPU/GPU, and is documented.
3. Head Regression Model
Train regression model (ResNet, MobileNet, EfficientNet, etc.) on cropped bug images
Support for multiple loss functions, data augmentation, early stopping, LR scheduling, TensorBoard
Status:
✅ Completed
Script is modular, CLI-driven, and supports all best practices.
4. Hyperparameter Tuning
Use Optuna to find best parameters (arch, loss, batch size, etc.)
Save best parameters to a file for reproducibility
Status:
✅ Completed
Script now saves best params to best_parameters_optuna.txt.
5. Cross-Validation
Run k-fold cross-validation with best parameters to estimate generalization
Save mean/std results and generate a plot for easy analysis
Status:
✅ Completed
Script saves results to crossval_results.txt and plot to crossval_val_losses.png.
6. Final Model Training
Train final regression model on all data using best parameters
Status:
✅ Completed
Workflow is clear: use best params, train, and save model for inference.
7. Visualization
Visualize predictions vs. ground truth for qualitative assessment
Status:
✅ Completed
Script is available and documented.
8. Documentation & Best Practices
README updated with full workflow, CLI options, and best practices
.gitignore excludes processed data
Project pushed to GitHub
Status:
✅ Completed
All steps, scripts, and outputs are documented and versioned.
What Still Needs To Be Done?
A. Full Pipeline Integration & Inference
Integrate YOLO detection and regression model for end-to-end inference on new videos
Map predicted head positions back to original video frames
(Optional) Add tracking (SORT) for multi-bug tracking
Status:
🟡 Partially Complete / Needs Final Integration
The individual components are ready, but a script for full pipeline inference (detect → crop → regress → map back) should be finalized if not already present.