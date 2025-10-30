"""
Visualization
Draw bounding boxes and create comparison views.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


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
    
    # Generate colors for each label
    unique_labels = list(set(det['label'] for det in detections))
    colors = {}
    for i, label in enumerate(unique_labels):
        # Generate distinct colors
        hue = (i * 137.5) % 360  # Golden angle
        color = _hsv_to_rgb(hue, 0.8, 0.9)
        colors[label] = color
    
    # Draw masks first (so boxes appear on top)
    if config.VIZ_CONFIG.get('show_masks', True):
        img = _draw_masks(img, detections, colors)
    
    # Draw bounding boxes and labels
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        label = det['label']
        confidence = det.get('confidence', 1.0)
        color = colors[label]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 
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
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw+5, y1), color, -1)
        
        # Label text
        cv2.putText(img, text, (x1+2, y1-2), 
                   cv2.FONT_HERSHEY_SIMPLEX,
                   config.VIZ_CONFIG['font_scale'], 
                   (255, 255, 255), 1)
    
    return img


def _draw_masks(image, detections, colors):
    """
    Draw semi-transparent segmentation masks on image.
    
    Args:
        image: RGB image (numpy array)
        detections: List of detections with 'mask' field
        colors: Dict mapping labels to RGB colors
        
    Returns:
        Image with masks overlaid
    """
    # Create overlay for all masks
    overlay = image.copy()
    
    for det in detections:
        if 'mask' not in det:
            continue
            
        mask = det['mask']  # Binary mask (H, W)
        label = det['label']
        color = colors[label]
        
        # Apply color to masked regions
        overlay[mask] = overlay[mask] * (1 - config.VIZ_CONFIG['mask_opacity']) + \
                        np.array(color) * config.VIZ_CONFIG['mask_opacity']
    
    return overlay.astype(np.uint8)


def create_comparison(original, annotated, save_path=None):
    """
    Create side-by-side comparison.
    
    Args:
        original: Original image
        annotated: Annotated image  
        save_path: Optional path to save
        
    Returns:
        Comparison image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.imshow(original)
    ax1.set_title('Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(annotated)
    ax2.set_title('Detected Objects', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {save_path}")
        plt.close()
    else:
        plt.show()


def _hsv_to_rgb(h, s, v):
    """Convert HSV to RGB color."""
    h = h / 60
    c = v * s
    x = c * (1 - abs((h % 2) - 1))
    m = v - c
    
    if 0 <= h < 1:
        r, g, b = c, x, 0
    elif 1 <= h < 2:
        r, g, b = x, c, 0
    elif 2 <= h < 3:
        r, g, b = 0, c, x
    elif 3 <= h < 4:
        r, g, b = 0, x, c
    elif 4 <= h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))