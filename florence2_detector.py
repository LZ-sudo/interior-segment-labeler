#!/usr/bin/env python3
"""
Florence-2 Detector Module
Shared detector class for object detection using Florence-2.
Can be imported and reused across multiple scripts to avoid model reloading overhead.

Usage:
    from florence2_detector import Florence2Detector

    # Initialize once
    detector = Florence2Detector()

    # Reuse for multiple detections
    result1 = detector.detect(image_path1, vocabulary1)
    result2 = detector.detect(image_path2, vocabulary2)
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
import config


class Florence2Detector:
    """Florence-2 based object detector for generating bounding box coordinates."""

    def __init__(self, device=None, verbose=True):
        """
        Initialize Florence-2 model.

        Args:
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
            verbose: Whether to print loading messages (default: True). Set to False for API/Modal deployments.
        """
        self.verbose = verbose

        if self.verbose:
            print("Loading Florence-2 model...")

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.MODEL_CONFIG['florence_model'],
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_CONFIG['florence_model'],
            dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # Move to device
        self.model = self.model.to(self.device)

        if self.verbose:
            print(f"[OK] Florence-2 loaded on {self.device}")

    def detect(self, image_path, vocabulary):
        """
        Detect objects in image based on vocabulary.

        Args:
            image_path: Path to input image
            vocabulary: List of object labels to detect (e.g., ['sofa', 'table', 'lamp'])

        Returns:
            Dictionary containing detections with normalized and pixel coordinates
        """
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_width, image_height = image_pil.size

        if self.verbose:
            print(f"Image size: {image_width}x{image_height}")
            print(f"Detecting: {len(vocabulary)} object types...")

        # Create text prompt
        text_input = ". ".join(vocabulary) + "."
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        prompt = task_prompt + text_input

        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image_pil,
            return_tensors="pt"
        )

        # Move inputs to device and convert to correct dtype
        inputs = {
            k: v.to(self.device).to(self.dtype) if v.dtype == torch.float32 else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=config.MODEL_CONFIG['max_new_tokens'],
                num_beams=config.MODEL_CONFIG['num_beams'],
                do_sample=False,
                use_cache=False
            )

        # Decode and parse
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_width, image_height)
        )

        # Parse detections
        detections = self._parse_detections(
            parsed_answer,
            image_width,
            image_height
        )

        if self.verbose:
            print(f"[OK] Found {len(detections)} objects")

        return {
            'image_path': str(image_path),
            'image_size': {
                'width': image_width,
                'height': image_height
            },
            'vocabulary': vocabulary,
            'detections': detections,
            'count': len(detections)
        }

    def _parse_detections(self, parsed_answer, image_width, image_height):
        """
        Parse Florence-2 output and convert to frontend-friendly format.

        Args:
            parsed_answer: Parsed output from Florence-2
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            List of detection dictionaries with normalized and pixel coordinates
        """
        detections = []

        # Extract results
        result = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
        if not result:
            return detections

        labels = result.get('labels', [])
        bboxes = result.get('bboxes', [])

        # Process each detection
        for label, bbox in zip(labels, bboxes):
            x1, y1, x2, y2 = bbox

            # Calculate normalized coordinates (0-1 range)
            x1_norm = x1 / image_width
            y1_norm = y1 / image_height
            x2_norm = x2 / image_width
            y2_norm = y2 / image_height

            # Calculate center, width, height (normalized)
            width_norm = x2_norm - x1_norm
            height_norm = y2_norm - y1_norm
            center_x_norm = x1_norm + width_norm / 2
            center_y_norm = y1_norm + height_norm / 2

            # Calculate pixel dimensions
            width_px = x2 - x1
            height_px = y2 - y1

            detection = {
                'label': label,
                'confidence': 1.0,  # Florence-2 doesn't provide confidence scores

                # Normalized coordinates (0-1) - RECOMMENDED for frontend
                'bbox_normalized': [x1_norm, y1_norm, x2_norm, y2_norm],
                'center_normalized': [center_x_norm, center_y_norm],
                'width_normalized': width_norm,
                'height_normalized': height_norm,

                # Pixel coordinates (absolute) - for reference
                'bbox_pixels': [int(x1), int(y1), int(x2), int(y2)],
                'center_pixels': [int(x1 + width_px/2), int(y1 + height_px/2)],
                'width_pixels': int(width_px),
                'height_pixels': int(height_px)
            }

            detections.append(detection)

        return detections


def filter_detections(detections, iou_threshold=0.7, min_words=2, verbose=True):
    """
    Filter out invalid detections and high-overlap duplicates.

    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering boxes as duplicates (0-1)
        min_words: Minimum number of words in label to consider valid (filters single action words)
        verbose: Whether to print filtering messages (default: True)

    Returns:
        Filtered list of detections
    """
    if not detections:
        return detections

    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    # Step 1: Filter out single-word action verbs (like "Install", "Add", "Remove", etc.)
    # These are usually from instruction text, not actual objects
    filtered = []
    removed_labels = set()

    for det in detections:
        label = det['label'].strip()
        word_count = len(label.split())

        # Keep if: multi-word label OR single word that's not a verb
        # Common action verbs to filter: Install, Add, Remove, Place, Mount, etc.
        action_verbs = {'install', 'add', 'remove', 'place', 'mount', 'put', 'set',
                       'create', 'build', 'make', 'change', 'replace'}

        if word_count >= min_words or label.lower() not in action_verbs:
            filtered.append(det)
        else:
            removed_labels.add(label)

    if removed_labels and verbose:
        print(f"  > Filtered out action words: {', '.join(removed_labels)}")

    # Step 2: Remove highly overlapping duplicates (keep the one with more descriptive label)
    final_detections = []
    skip_indices = set()

    for i, det1 in enumerate(filtered):
        if i in skip_indices:
            continue

        # Check against all remaining detections
        duplicates = [i]
        for j, det2 in enumerate(filtered[i+1:], start=i+1):
            if j in skip_indices:
                continue

            iou = calculate_iou(det1['bbox_pixels'], det2['bbox_pixels'])

            if iou >= iou_threshold:
                duplicates.append(j)

        # If duplicates found, keep the one with the longest/most descriptive label
        if len(duplicates) > 1:
            best_idx = max(duplicates, key=lambda idx: len(filtered[idx]['label']))
            skip_indices.update(d for d in duplicates if d != best_idx)
            final_detections.append(filtered[best_idx])
        else:
            final_detections.append(det1)

    if len(detections) > len(final_detections) and verbose:
        print(f"  > Cleaned up duplicates: {len(detections)} -> {len(final_detections)} detections")

    return final_detections
