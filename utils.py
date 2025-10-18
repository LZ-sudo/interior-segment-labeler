"""
Image Utilities
Load and save images.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """Load image and convert to RGB."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image, output_path):
    """Save RGB image."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)
    print(f"Saved: {output_path}")