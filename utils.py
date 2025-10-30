"""
Image Utilities
Load and save images.
"""

import cv2
import numpy as np
from pathlib import Path
import json


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

def load_prompts(image_path):
    """
    Load custom prompts from JSON file for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of prompts if JSON exists, None otherwise
    """
    image_path = Path(image_path)
    prompt_file = image_path.parent / f"{image_path.stem}_prompt.json"
    
    if not prompt_file.exists():
        return None
    
    try:
        with open(prompt_file, 'r') as f:
            data = json.load(f)
        
        prompts = data.get('prompts', [])
        
        if not prompts:
            print(f"  ⚠ Warning: {prompt_file.name} has no prompts")
            return None
            
        print(f"  ✓ Loaded {len(prompts)} custom prompts from {prompt_file.name}")
        return prompts
        
    except json.JSONDecodeError as e:
        print(f"  ✗ Error reading {prompt_file.name}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Unexpected error loading prompts: {e}")
        return None