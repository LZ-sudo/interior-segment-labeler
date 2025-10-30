"""
Model - OWL-ViT v2 + SAM
Detect and segment objects in images.
"""

import torch
import numpy as np
from pathlib import Path
from segment_anything_hq import sam_model_registry, SamPredictor
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

        # Download SAM-HQ if needed
        if not sam_checkpoint.exists():
            model_type = config.MODEL_CONFIG['sam_model_type']
            print(f"Downloading SAM-HQ model ({model_type})...")
            import urllib.request
            
            # SAM-HQ checkpoint URLs
            sam_hq_urls = {
                'vit_b': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth',
                'vit_l': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
                'vit_h': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
                'vit_tiny': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth'
            }
            
            url = sam_hq_urls.get(model_type)
            if url is None:
                raise ValueError(f"Unknown model type: {model_type}")
            
            urllib.request.urlretrieve(url, sam_checkpoint)
            print(f"✓ Downloaded SAM-HQ {model_type} checkpoint")

        # Load SAM-HQ
        sam = sam_model_registry[config.MODEL_CONFIG['sam_model_type']](
            checkpoint=str(sam_checkpoint)
        )
        sam.to(config.MODEL_CONFIG['device'])
        self.sam_predictor = SamPredictor(sam)
        print("✓ SAM-HQ loaded")

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
        Detect objects in image and generate segmentation masks.
        
        Args:
            image_path: Path to image
            vocabulary: List of objects to detect (uses default if None)
            
        Returns:
            List of detections with bounding boxes, labels, and segmentation masks
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
        
        # Generate SAM masks for each detection
        if detections:
            print(f"Generating segmentation masks...")
            detections = self._generate_sam_masks(image_np, detections)
        
        return detections

    def _generate_sam_masks(self, image, detections):
        """
        Generate SAM segmentation masks for detected objects.
        
        Args:
            image: RGB image (numpy array)
            detections: List of detections with bounding boxes
            
        Returns:
            Detections with added 'mask' field
        """
        # Set image for SAM
        self.sam_predictor.set_image(image)
        
        # Generate mask for each detection
        for det in detections:
            bbox = det['bbox']
            
            # Convert [x1, y1, x2, y2] to SAM format [x1, y1, x2, y2]
            input_box = np.array(bbox)
            
            # Predict mask
            masks, scores, logits = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False  # Single mask per box
            )
            
            # Add mask to detection (take first mask since multimask_output=False)
            det['mask'] = masks[0]  # Binary mask (H, W)
        
        print(f"✓ Generated {len(detections)} segmentation masks")
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
    
    # def _dummy_detect(self, image, vocabulary):
    #     """Dummy detection for testing - replace with real GroundingDINO."""
    #     # This is just for testing the pipeline
    #     # Real implementation uses GroundingDINO
    #     h, w = image.shape[:2]
        
    #     return [
    #         {
    #             'bbox': [w//4, h//4, w//2, h//2],
    #             'label': 'sofa',
    #             'confidence': 0.85
    #         },
    #         {
    #             'bbox': [w//2, h//3, 3*w//4, 2*h//3],
    #             'label': 'table',
    #             'confidence': 0.75
    #         }
    #     ]