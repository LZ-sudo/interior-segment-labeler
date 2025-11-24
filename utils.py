"""
Image Utilities
Load and save images.
"""

import cv2
from pathlib import Path
import json
import webcolors

# image processing functions

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
            print(f"  [!] Warning: {prompt_file.name} has no prompts")
            return None

        print(f"  [OK] Loaded {len(prompts)} custom prompts from {prompt_file.name}")
        return prompts

    except json.JSONDecodeError as e:
        print(f"  [ERROR] Error reading {prompt_file.name}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected error loading prompts: {e}")
        return None

# Colour generation functions

def generate_label_colors(labels):
    """
    Generate distinct colors for each unique label.
    
    Args:
        labels: List of unique label names
        
    Returns:
        Dict mapping label to RGB color tuple and color name
    """
    colors = {}
    for i, label in enumerate(labels):
        hue = (i * 137.5) % 360  # Golden angle
        rgb = _hsv_to_rgb(hue, 0.8, 0.9)
        color_name = _get_color_name(rgb)
        colors[label] = {
            'rgb': rgb,
            'name': color_name
        }
    return colors

def _get_color_name(rgb_tuple):
    """
    Convert RGB tuple to closest color name.

    Args:
        rgb_tuple: (R, G, B) values from 0-255

    Returns:
        Color name string (e.g., 'red', 'steelblue', 'gold')
    """
    try:
        # Try exact match first
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # Find closest CSS3 color name
        min_distance = float('inf')
        closest_name = None

        # Get all CSS3 color names (compatible with webcolors >= 1.12)
        try:
            css3_names = webcolors.names("css3")
        except (AttributeError, TypeError):
            # Fallback for older versions
            css3_names = webcolors.CSS3_NAMES_TO_HEX.keys()

        for name in css3_names:
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - rgb_tuple[0]) ** 2
            gd = (g_c - rgb_tuple[1]) ** 2
            bd = (b_c - rgb_tuple[2]) ** 2
            distance = rd + gd + bd

            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name

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