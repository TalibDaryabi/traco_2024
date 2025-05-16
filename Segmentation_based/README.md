# Grok - Video Processing and Hexagon Detection System

## Project Overview
This project is designed to process videos and detect hexagonal patterns using a deep learning model (ResNet-UNet architecture). The system analyzes video frames, identifies hexagons, and assigns IDs to them in a specific order.

## Project Structure
```
grok/
├── config.py           # Configuration settings (paths and parameters)
├── main.py            # Main execution script
├── models/            # Neural network model definitions
│   └── resnet_unet.py # ResNet-UNet model architecture
├── data/              # Data handling utilities
│   └── video_handler.py # Video processing functions
├── utils/             # Utility functions
│   ├── processing.py  # ID assignment and processing utilities
│   └── helpers.py     # Helper functions for logging
├── losses/           # Loss functions for model training
├── scripts/          # Additional utility scripts
└── annotations_/     # Directory for annotations/ground truth
```

## Key Components
1. **Model**: Uses a ResNet-UNet architecture for hexagon detection
2. **Video Processing**: Handles frame-by-frame analysis of input videos
3. **ID Assignment**: Assigns unique IDs to detected hexagons
4. **Logging**: Maintains execution logs for debugging and monitoring

## Prerequisites
- Python 3.x
- PyTorch
- OpenCV (cv2)
- CUDA-capable GPU (optional, but recommended for faster processing)

## Configuration
The `config.py` file contains important parameters:
- `MODEL_PATH`: Path to the trained model weights (reference-model.pth)
- `VIDEO_PATH`: Path to the input video file
- `NUM_HEX`: Number of hexagons to detect (default: 3)

## How to Run

### Single Video Processing
1. Ensure the model file (reference-model.pth) is in the correct location
2. Update VIDEO_PATH in config.py to point to your input video
3. Run the main script:
```bash
python main.py
```

### Processing Multiple Videos
To process multiple videos, you would need to modify the code slightly:
1. Create a new script (e.g., `batch_process.py`) that:
   - Takes a directory of videos as input
   - Iterates through each video file
   - Updates VIDEO_PATH for each iteration
   - Calls the main processing function

## Processing Flow
1. **Model Loading**: 
   - Loads the pre-trained ResNet-UNet model
   - Moves model to GPU if available

2. **Video Processing**:
   - Reads video frames
   - Processes each frame through the neural network
   - Detects hexagonal patterns
   - Extracts position information

3. **ID Assignment**:
   - Takes unordered position data
   - Assigns ordered IDs to detected hexagons
   - Logs the results

## Output
The system produces:
- Processed video frames with detected hexagons
- Ordered list of hexagon positions
- Log messages for tracking execution

## Error Handling
The system includes robust error handling for:
- Model loading failures
- Video processing issues
- Invalid configurations

## Notes
- The system is currently configured for processing one video at a time
- For batch processing of multiple videos, code modifications would be needed
- GPU acceleration is automatically used if available
