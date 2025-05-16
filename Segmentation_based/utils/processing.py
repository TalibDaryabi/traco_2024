# utils/processing.py
# Data processing functions for extracting and ordering positions.

import cv2
import numpy as np

def get_densest_numpy_patches(image, num_hex=3, radius=10):
    """
    Extract the positions of the densest patches in the image.
    
    Args:
        image (np.ndarray): Model prediction output (height, width).
        num_hex (int): Number of positions to extract.
        radius (int): Radius around each position to mask.
    
    Returns:
        list: List of [x, y] positions of the densest patches.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    maximum_value_list = []
    for _ in range(num_hex):
        a = np.argmax(image)
        l = image.shape[0]
        c = a % l  # x-coordinate
        r = a // l  # y-coordinate
        if r < 5 or c < 5:
            image = cv2.circle(image, (c, r), 2, 0, -1)
            a = np.argmax(image)
            c = a % l
            r = a // l
        maximum_value_list.append([c, r])
        image = cv2.circle(image, (c, r), radius, 0, -1)
    return maximum_value_list

def assign_ID(list_max_value_unordered):
    """
    Assign consistent IDs to detected positions across frames.
    
    Args:
        list_max_value_unordered (list): List of frames with unordered positions.
    
    Returns:
        list: List of frames with ordered positions.
    """
    if not list_max_value_unordered:
        raise ValueError("Input list is empty.")
    
    pos_sum = len(list_max_value_unordered[0])
    list_max_value_ordered = [list_max_value_unordered[0]]
    
    for i in range(1, len(list_max_value_unordered)):
        pos1 = list_max_value_ordered[i-1]
        pos2 = list_max_value_unordered[i].copy()
        new_frame = [[0, 0] for _ in range(pos_sum)]
        for j in range(pos_sum):
            d_shortest = float('inf')
            index = 0
            for k in range(len(pos2)):
                d = np.sqrt((pos1[j][0] - pos2[k][0])**2 + (pos1[j][1] - pos2[k][1])**2)
                if d < d_shortest:
                    d_shortest = d
                    index = k
            new_frame[j] = pos2[index]
            del pos2[index]
        list_max_value_ordered.append(new_frame)
    return list_max_value_ordered