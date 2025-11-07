#!/usr/bin/env python3
"""
Main Script
Simple interface to detect objects in interior images.

Usage:
    python main.py image.jpg              # Single image
    python main.py images/                # Directory
    python main.py images/ --output results/
"""

import sys
from pathlib import Path
from models.model import InteriorDetector
from visualization import draw_boxes
# from export_coco import export_coco
from utils import load_image, save_image, load_prompts
import config


def process_image(image_path, detector, output_dir='results'):
    """
    Process a single image.
    
    Args:
        image_path: Path to input image
        detector: InteriorDetector instance (reused for efficiency)
        output_dir: Where to save results
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing: {image_path.name}")
    
    try:
        # 1. Load image
        image = load_image(image_path)
        
        # 2. Check for custom prompts
        custom_prompts = load_prompts(image_path)
        
        # 3. Detect objects (with custom prompts if available)
        if custom_prompts:
            detections = detector.detect(image_path, vocabulary=custom_prompts)
        else:
            print(f"  → Using default vocabulary ({len(config.VOCABULARY)} items)")
            detections = detector.detect(image_path)
        
        if not detections:
            print(f"  ⚠ No objects detected in {image_path.name}")
            return False
        
        print(f"  ✓ Found {len(detections)} objects")
        
        # 4. Create visualization
        annotated = draw_boxes(image, detections)
        
        # Save annotated image
        annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
        save_image(annotated, annotated_path)
        
        # # Create comparison
        # comparison_path = output_dir / f"{image_path.stem}_comparison.jpg"
        # create_comparison(image, annotated, comparison_path)
        
        # # 5. Export COCO
        # coco_path = output_dir / f"{image_path.stem}_coco.json"
        # export_coco(detections, image_path, image.shape[:2], coco_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_path, output_dir='results'):
    """
    Process all images in a directory.
    
    Args:
        input_path: Path to directory with images
        output_dir: Where to save results
    """
    input_path = Path(input_path)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    # Remove duplicates (Windows is case-insensitive)
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"No image files found in {input_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing directory: {input_path}")
    print(f"Found {len(image_files)} images")
    print(f"{'='*60}")
    
    # Initialize detector once (for efficiency)
    detector = InteriorDetector()
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=' ')
        
        if process_image(image_file, detector, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("✓ Batch processing complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(image_files)}")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir}/")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <image_path>           # Single image")
        print("  python main.py <directory>            # All images in directory")
        print("  python main.py <path> --output <dir>  # Custom output directory")
        print("\nExamples:")
        print("  python main.py room.jpg")
        print("  python main.py images/")
        print("  python main.py images/ --output my_results/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Check for custom output directory
    output_dir = 'results'
    if len(sys.argv) > 2 and sys.argv[2] == '--output':
        if len(sys.argv) > 3:
            output_dir = sys.argv[3]
        else:
            print("Error: --output requires a directory path")
            sys.exit(1)
    
    # Check if input exists
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Process directory or single file
    if input_path.is_dir():
        process_directory(input_path, output_dir)
    elif input_path.is_file():
        print(f"\n{'='*60}")
        print(f"Processing single image: {input_path.name}")
        print(f"{'='*60}")
        
        detector = InteriorDetector()
        process_image(input_path, detector, output_dir)
        
        print(f"\n{'='*60}")
        print("✓ Processing complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_dir}/")
        print()
    else:
        print(f"Error: {input_path} is neither a file nor directory")
        sys.exit(1)


if __name__ == '__main__':
    main()