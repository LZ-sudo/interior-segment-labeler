#!/usr/bin/env python3
"""
Label Changes Script
Uses Florence-2 for zero-shot object detection based on text prompts.
Returns bounding box coordinates in normalized format (0-1) for frontend use.
Automatically generates JSON and annotated image outputs.

Usage:
    python label_changes.py image.jpg                              # Auto-saves JSON + labeled image
    python label_changes.py image.jpg "sofa, table, lamp"          # Custom vocabulary
    python label_changes.py image.jpg --output-dir results/        # Custom output directory
    python label_changes.py image.jpg --no-image                   # Skip image generation
    python label_changes.py image.jpg --no-json                    # Print to console only
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import config
from utils import load_prompts, save_image, generate_label_colors
from florence2_detector import Florence2Detector, filter_detections


# Florence2Detector and filter_detections are now imported from florence2_detector.py
# This allows the detector to be reused across multiple scripts without reloading


def draw_detection_boxes(image_path, detections_list, output_path):
    """
    Draw bounding boxes with labels on the image.

    Args:
        image_path: Path to original image
        detections_list: List of detection dictionaries
        output_path: Path to save annotated image

    Returns:
        Path to saved image
    """
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    # Generate colors for unique labels
    unique_labels = sorted(list(set(det['label'] for det in detections_list)))
    colors = generate_label_colors(unique_labels)

    # Draw each detection
    for det in detections_list:
        label = det['label']
        bbox_pixels = det['bbox_pixels']
        x1, y1, x2, y2 = [int(v) for v in bbox_pixels]

        color_rgb = colors[label]['rgb']

        # Draw rectangle
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color_rgb, 2)

        # Draw label background
        text = label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_np, (x1, y1-th-8), (x1+tw+8, y1), color_rgb, -1)

        # Draw label text
        cv2.putText(image_np, text, (x1+4, y1-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save annotated image
    save_image(image_np, output_path)
    return output_path


def format_output(result, format_type='normalized'):
    """
    Format output based on requested format type.

    Args:
        result: Full detection result dictionary
        format_type: 'normalized' (default) or 'pixels'

    Returns:
        Formatted result dictionary
    """
    if format_type == 'pixels':
        # Simplify to pixel coordinates only
        formatted_detections = []
        for det in result['detections']:
            formatted_detections.append({
                'label': det['label'],
                'confidence': det['confidence'],
                'bbox': det['bbox_pixels'],
                'center': det['center_pixels'],
                'width': det['width_pixels'],
                'height': det['height_pixels']
            })

        return {
            'image_path': result['image_path'],
            'image_size': result['image_size'],
            'vocabulary': result['vocabulary'],
            'detections': formatted_detections,
            'count': result['count'],
            'coordinate_format': 'pixels'
        }

    else:  # normalized (default)
        # Simplify to normalized coordinates
        formatted_detections = []
        for det in result['detections']:
            formatted_detections.append({
                'label': det['label'],
                'confidence': det['confidence'],
                'bbox': det['bbox_normalized'],
                'center': det['center_normalized'],
                'width': det['width_normalized'],
                'height': det['height_normalized']
            })

        return {
            'image_path': result['image_path'],
            'image_size': result['image_size'],
            'vocabulary': result['vocabulary'],
            'detections': formatted_detections,
            'count': result['count'],
            'coordinate_format': 'normalized'
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect objects in images using Florence-2 and return bounding box coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-saves JSON and labeled image
  python label_changes.py room.jpg

  # With custom vocabulary
  python label_changes.py room.jpg "sofa, coffee table, lamp"

  # Save to custom directory
  python label_changes.py room.jpg --output-dir results/

  # Only generate JSON (no image)
  python label_changes.py room.jpg --no-image

  # Only print to console (no files)
  python label_changes.py room.jpg --no-json --no-image

  # Get pixel coordinates instead of normalized
  python label_changes.py room.jpg --format pixels

Output Files (by default):
  - {image_stem}_detections.json  # Bounding box coordinates
  - {image_stem}_labeled.jpg      # Annotated image with boxes

Prompt JSON File:
  Create a file named {image_stem}_prompt.json with:
  {
    "prompts": ["sofa", "coffee table", "lamp"]
  }

Frontend Usage:
  For HTML Canvas or CSS, use normalized coordinates:
    const x_pixels = detection.bbox[0] * imageElement.width;
    const y_pixels = detection.bbox[1] * imageElement.height;
        """
    )

    parser.add_argument('image', help='Path to input image')
    parser.add_argument('vocabulary', nargs='?', default=None,
                       help='Comma-separated list of objects to detect (e.g., "sofa, table, lamp"). Optional if {image}_prompt.json exists.')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: {image_stem}_detections.json)')
    parser.add_argument('--output-dir', help='Output directory for results (default: same as input image)')
    parser.add_argument('--format', '-f', choices=['normalized', 'pixels', 'both'], default='normalized',
                       help='Coordinate format: normalized (0-1), pixels (absolute), or both (default: normalized)')
    parser.add_argument('--no-image', action='store_true',
                       help='Skip generating annotated image with bounding boxes')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip saving JSON output (print to console only)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Skip filtering out action words and duplicate detections')
    parser.add_argument('--iou-threshold', type=float, default=0.7,
                       help='IoU threshold for duplicate filtering (default: 0.7)')
    parser.add_argument('--pretty', '-p', action='store_true',
                       help='Pretty-print JSON output')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device to run model on (auto-detected if not specified)')

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Determine vocabulary source
    vocabulary = None
    vocab_source = None

    # 1. Try loading from JSON prompt file (like segment_image.py)
    custom_prompts = load_prompts(image_path)
    if custom_prompts:
        vocabulary = custom_prompts
        vocab_source = f"{image_path.stem}_prompt.json"

    # 2. Use command-line vocabulary if provided
    if args.vocabulary:
        vocabulary = [item.strip() for item in args.vocabulary.split(',')]
        vocab_source = "command-line argument"

    # 3. Fall back to default vocabulary from config
    if vocabulary is None:
        vocabulary = config.VOCABULARY
        vocab_source = f"config.py (default vocabulary - {len(config.VOCABULARY)} items)"

    print(f"\n{'='*60}")
    print(f"Florence-2 Object Detection")
    print(f"{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"Vocabulary source: {vocab_source}")
    print(f"Vocabulary: {', '.join(vocabulary[:5])}{f'... (+{len(vocabulary)-5} more)' if len(vocabulary) > 5 else ''}")
    print(f"{'='*60}\n")

    # Initialize detector
    detector = Florence2Detector(device=args.device)

    # Run detection
    result = detector.detect(image_path, vocabulary)

    if result['count'] == 0:
        print(f"\n[!] No objects detected in {image_path.name}")
        return

    # Filter detections to remove action words and duplicates (unless disabled)
    if not args.no_filter:
        print(f"Filtering detections...")
        filtered_detections = filter_detections(result['detections'], iou_threshold=args.iou_threshold)
        result['detections'] = filtered_detections
        result['count'] = len(filtered_detections)

        if result['count'] == 0:
            print(f"\n[!] No valid objects after filtering in {image_path.name}")
            return

    # Format output
    if args.format == 'both':
        # Keep full result with both coordinate formats
        output_result = result
    else:
        output_result = format_output(result, args.format)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = image_path.parent

    # Convert to JSON
    json_indent = 2 if args.pretty else 4  # Default to pretty format
    json_output = json.dumps(output_result, indent=json_indent)

    # Save JSON (default behavior)
    if not args.no_json:
        if args.output:
            json_path = Path(args.output)
        else:
            # Default: save as {image_stem}_detections.json
            json_path = output_dir / f"{image_path.stem}_detections.json"

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            f.write(json_output)
        print(f"[OK] JSON saved to: {json_path}")

    # Generate annotated image (default behavior)
    if not args.no_image:
        annotated_path = output_dir / f"{image_path.stem}_labeled.jpg"
        draw_detection_boxes(image_path, result['detections'], annotated_path)
        print(f"[OK] Annotated image saved to: {annotated_path}")

    # Print to console if no JSON was saved
    if args.no_json:
        print(f"\n{'='*60}")
        print("Detection Results")
        print(f"{'='*60}\n")
        print(json_output)

    print(f"\n{'='*60}")
    print(f"[OK] Detection complete! Found {result['count']} objects")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
