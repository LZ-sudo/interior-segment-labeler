"""
Visualization
Draw bounding boxes and create comparison views.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import generate_label_colors


def draw_boxes(image, detections):
    """
    Draw segmentation masks and bounding boxes on image.
    
    Args:
        image: RGB image (numpy array)
        detections: List of detection dicts with 'bbox', 'label', 'confidence', and optionally 'mask'
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    # Generate colors using shared function
    unique_labels = sorted(list(set(det['label'] for det in detections)))
    colors = generate_label_colors(unique_labels)
    
    # Draw masks first (so boxes appear on top)
    if config.VIZ_CONFIG.get('show_masks', True):
        img = _draw_masks(img, detections, colors)

    # Draw bounding boxes and labels (only if enabled)
    if config.VIZ_CONFIG.get('show_boxes', True):
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = det['label']
            confidence = det.get('confidence', 1.0)
            color_rgb = colors[label]['rgb']  # Extract RGB tuple

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb,
                         config.VIZ_CONFIG['box_thickness'])

            # Draw label
            if config.VIZ_CONFIG['show_confidence']:
                text = f"{label}: {confidence:.2f}"
            else:
                text = label

            # Label background
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                config.VIZ_CONFIG['font_scale'], 1
            )
            cv2.rectangle(img, (x1, y1-th-5), (x1+tw+5, y1), color_rgb, -1)

            # Label text
            cv2.putText(img, text, (x1+2, y1-2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       config.VIZ_CONFIG['font_scale'],
                       (255, 255, 255), 1)
    
    return img

def expand_boxes(bbox, expansion_factor=0.10, image_shape=None):
    """
    Expand bounding box by a percentage while keeping it centered.
    
    Args:
        bbox: [x1, y1, x2, y2]
        expansion_factor: Percentage to expand (0.10 = 10%)
        image_shape: (height, width) to clip boxes to image bounds
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate current dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion amounts
    expand_w = width * expansion_factor
    expand_h = height * expansion_factor
    
    # Expand symmetrically
    new_x1 = x1 - expand_w / 2
    new_y1 = y1 - expand_h / 2
    new_x2 = x2 + expand_w / 2
    new_y2 = y2 + expand_h / 2
    
    # Clip to image boundaries if provided
    if image_shape is not None:
        height_img, width_img = image_shape
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(width_img, new_x2)
        new_y2 = min(height_img, new_y2)
    
    return [new_x1, new_y1, new_x2, new_y2]

def _draw_masks(image, detections, colors):
    """
    Draw semi-transparent segmentation masks on image.
    """
    overlay = image.copy()
    
    for det in detections:
        if 'mask' not in det:
            continue

        mask = det['mask']
        # Convert mask to boolean if it's not already
        mask = mask.astype(bool)
        label = det['label']
        color_rgb = colors[label]['rgb']  # Extract RGB tuple

        # Apply color to masked regions
        overlay[mask] = overlay[mask] * (1 - config.VIZ_CONFIG['mask_opacity']) + \
                        np.array(color_rgb) * config.VIZ_CONFIG['mask_opacity']
    
    return overlay.astype(np.uint8)