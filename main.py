# main.py
# Main script to process video, assign IDs, and log results.

import torch
import sys
import os
import logging
from models.resnet_unet import ResNetUNet
from data.video_handler import process_video
from utils.processing import assign_ID
from utils.helpers import log_message, save_predictions
from config import MODEL_PATH, VIDEO_PATH, NUM_HEX
import torch.serialization

def get_device():
    """
    Get the best available device (CUDA GPU if available, otherwise CPU).
    Also returns device properties for logging.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_props = {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
        return device, device_props
    return torch.device("cpu"), {"name": "CPU"}

def load_model(model_path, device):
    """
    Safely load the model with proper device mapping.
    """
    # First create a new model instance
    model = ResNetUNet(n_class=1)
    
    # Add the model class to sys.modules to handle the __main__ module issue
    sys.modules['ResNetUNet'] = type(model)
    
    try:
        # Try loading the full model with weights_only=False since we trust our own model
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different possible formats of the saved model
        if isinstance(state_dict, ResNetUNet):
            # If we loaded the full model
            model = state_dict
        elif hasattr(state_dict, 'state_dict'):
            # If we loaded an object with state_dict method
            model.load_state_dict(state_dict.state_dict())
        elif isinstance(state_dict, dict):
            # If we loaded a state dict directly
            model.load_state_dict(state_dict)
        else:
            raise Exception(f"Unexpected model format: {type(state_dict)}")
            
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    return model

def main():
    # Setup logging
    logging.basicConfig(
        format='LOG: %(message)s',
        level=logging.INFO
    )
    
    # Get the best available device
    device, device_props = get_device()
    if device.type == "cuda":
        log_message(f"Using GPU: {device_props['name']} with {device_props['memory']} memory")
        # Set default CUDA device
        torch.cuda.set_device(device)
        # Empty CUDA cache to free up memory
        torch.cuda.empty_cache()
    else:
        log_message("No GPU found. Using CPU for computation")
    
    try:
        # Load the model using our safe loading function
        model = load_model(MODEL_PATH, device)
        model = model.to(device)
        model.eval()
        log_message(f"Model successfully loaded and moved to {device.type.upper()}")
    except Exception as e:
        log_message(f"Error loading model: {e}")
        return
    
    try:
        # Ensure input tensors are on the same device as the model
        list_max_value_unordered, frame_width, frame_height, frame_count = process_video(
            VIDEO_PATH, model, device, num_hex=NUM_HEX
        )
        
    except Exception as e:
        log_message(f"Error processing video: {e}")
        return
    
    list_max_value_ordered = assign_ID(list_max_value_unordered)
    log_message("Processing complete. Ordered positions computed.")
    
    # Save results in both JSON and CSV formats
    output_dir = os.path.dirname(VIDEO_PATH)
    save_predictions(list_max_value_ordered, frame_width, frame_height, output_dir)

if __name__ == "__main__":
    main()