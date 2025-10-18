"""
Model - OWL-ViT v2 + SAM
Detect and segment objects in images.
"""

import torch
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import config


class InteriorDetector:
    """Simple detector using OWL-ViT v2 + SAM."""

    def __init__(self):
        """Initialize models."""
        print("Loading models...")

        # Paths
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        sam_checkpoint = models_dir / config.MODEL_CONFIG['sam_checkpoint']

        # Download SAM if needed
        if not sam_checkpoint.exists():
            print("Downloading SAM model (vit_b)...")
            import urllib.request
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
            urllib.request.urlretrieve(url, sam_checkpoint)

        # Load SAM
        sam = sam_model_registry['vit_b'](checkpoint=str(sam_checkpoint))
        sam.to(config.MODEL_CONFIG['device'])
        self.sam_predictor = SamPredictor(sam)

        # Load OWL-ViT v2 for object detection
        try:
            print("Loading OWL-ViT v2 model...")
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
            self.owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
            self.owl_model.to(config.MODEL_CONFIG['device'])
            self.owl_model.eval()
            self.device = config.MODEL_CONFIG['device']
            print("✓ OWL-ViT v2 loaded successfully")
        except Exception as e:
            self.processor = None
            self.owl_model = None
            print(f"Warning: OWL-ViT v2 not loaded - {e}")
            print("Run: pip install transformers")

        print("✓ Models loaded")
    
    def detect(self, image_path, vocabulary=None):
        """
        Detect objects in image.

        Args:
            image_path: Path to image
            vocabulary: List of objects to detect (uses default if None)

        Returns:
            List of detections with bounding boxes and labels
        """
        if vocabulary is None:
            vocabulary = config.VOCABULARY

        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        print(f"Detecting: {len(vocabulary)} object types...")

        # Use OWL-ViT v2 if available, otherwise use dummy detections
        if self.owl_model is not None:
            detections = self._detect_with_owlvit(image_pil, vocabulary)
        else:
            print("Warning: Using dummy detections (OWL-ViT v2 not loaded)")
            detections = self._dummy_detect(image_np, vocabulary)

        print(f"Found: {len(detections)} objects")

        return detections

    def _detect_with_owlvit(self, image_pil, vocabulary):
        """Use OWL-ViT v2 for real object detection."""
        # Prepare text queries - add "a photo of a" prefix for better results
        text_queries = [[f"a photo of a {label}" for label in vocabulary]]

        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=image_pil,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.owl_model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=config.MODEL_CONFIG['box_threshold'],
            target_sizes=target_sizes
        )[0]

        # Convert to our format
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_id in zip(boxes, scores, labels):
            # Get label from label_id
            label = vocabulary[label_id] if label_id < len(vocabulary) else "unknown"

            detections.append({
                'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                'label': label,
                'confidence': float(score)
            })

        return detections
    
    def _dummy_detect(self, image, vocabulary):
        """Dummy detection for testing - replace with real GroundingDINO."""
        # This is just for testing the pipeline
        # Real implementation uses GroundingDINO
        h, w = image.shape[:2]
        
        return [
            {
                'bbox': [w//4, h//4, w//2, h//2],
                'label': 'sofa',
                'confidence': 0.85
            },
            {
                'bbox': [w//2, h//3, 3*w//4, 2*h//3],
                'label': 'table',
                'confidence': 0.75
            }
        ]