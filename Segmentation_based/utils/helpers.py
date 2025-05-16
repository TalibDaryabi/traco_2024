# utils/helpers.py
# Additional helper functions.

import logging
import json
import csv
import os
import numpy as np

def log_message(message):
    """
    Log a message to the console.
    
    Args:
        message (str): Message to log.
    """
    logging.info(message)

def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def save_predictions_json(list_max_value_ordered, output_path):
    """
    Save predictions in JSON format compatible with training data format.
    
    Args:
        list_max_value_ordered: List of ordered positions for each frame
        output_path: Path to save the JSON file
    """
    json_predicted = {'rois': []}
    
    for frame, positions in enumerate(list_max_value_ordered):
        for id, pos in enumerate(positions):
            json_predicted['rois'].append({
                'z': int(frame),  # Ensure frame number is regular Python int
                'id': int(id),    # Ensure ID is regular Python int
                'pos': convert_to_serializable(pos)
            })
    
    with open(output_path, 'w') as f:
        json.dump(json_predicted, f, indent=4)
    log_message(f"JSON predictions saved to {output_path}")

def save_predictions_csv(list_max_value_ordered, frame_width, frame_height, output_path):
    """
    Save predictions in CSV format with coordinates scaled to original frame dimensions.
    
    Args:
        list_max_value_ordered: List of ordered positions for each frame
        frame_width: Original frame width
        frame_height: Original frame height
        output_path: Path to save the CSV file
    """
    with open(output_path, mode='w', newline='') as csv_file:
        fieldnames = ['', 't', 'hexbug', 'x', 'y']
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
        
        idx = 0
        for t, frame in enumerate(list_max_value_ordered):
            for hexbug, pos in enumerate(frame):
                # Convert numpy arrays to list if necessary
                pos = convert_to_serializable(pos)
                # Scale coordinates back to original frame dimensions
                pos_x = np.round((pos[0]/256) * frame_height, 2)
                pos_y = np.round((pos[1]/256) * frame_width, 2)
                writer.writerow([idx, t, hexbug, float(pos_x), float(pos_y)])
                idx += 1
    
    log_message(f"CSV predictions saved to {output_path}")

def save_predictions(list_max_value_ordered, frame_width, frame_height, output_dir):
    """
    Save predictions in both JSON and CSV formats.
    
    Args:
        list_max_value_ordered: List of ordered positions for each frame
        frame_width: Original frame width
        frame_height: Original frame height
        output_dir: Directory to save the output files
    """
    json_path = os.path.join(output_dir, 'predictions.json')
    csv_path = os.path.join(output_dir, 'predictions.csv')
    
    save_predictions_json(list_max_value_ordered, json_path)
    save_predictions_csv(list_max_value_ordered, frame_width, frame_height, csv_path)