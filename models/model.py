"""
Model - Florence-2 + SAM-HQ
Detect and segment objects in images.
"""

import torch
import numpy as np
from pathlib import Path
from segment_anything_hq import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import config
from visualization import expand_boxes
import urllib.request


class InteriorDetector:
    """Simple detector using Florence-2 + SAM-HQ."""

    def __init__(self):
        """Initialize models."""
        print("Loading models...")

        # Paths - use absolute path relative to this file's location
        # This ensures weights are stored in the submodule's directory even when used as a submodule
        models_dir = Path(__file__).parent
        models_dir.mkdir(exist_ok=True)

        sam_checkpoint = models_dir / config.MODEL_CONFIG['sam_checkpoint']

        # Download SAM-HQ if needed
        if not sam_checkpoint.exists():
            model_type = config.MODEL_CONFIG['sam_model_type']
            print(f"Downloading SAM-HQ model ({model_type})...")
            
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

        # Load Florence-2 for object detection
        try:
            print("Loading Florence-2 model...")
            self.device = config.MODEL_CONFIG['device']
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.processor = AutoProcessor.from_pretrained(
                config.MODEL_CONFIG['florence_model'],
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_CONFIG['florence_model'],
                dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager"  # Fix SDPA compatibility warning
            ).to(self.device)

            print("✓ Florence-2 loaded successfully")
        except Exception as e:
            self.processor = None
            self.model = None
            print(f"Warning: Florence-2 not loaded - {e}")
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
        
        # Use Florence-2 if available, otherwise use dummy detections
        if self.model is not None:
            detections = self._detect_with_florence2(image_pil, vocabulary)
        else:
            print("Error: Florence-2 not loaded")
        
        print(f"Found: {len(detections)} objects")
        
        # Generate SAM masks for each detection
        if detections:
            print(f"Generating segmentation masks...")
            detections = self._generate_sam_masks(image_np, detections)
        
        return detections

    def _detect_with_florence2(self, image_pil, vocabulary):
        """Use Florence-2 for phrase-grounded object detection."""
        
        # Create text prompt from vocabulary
        # Florence-2 expects format: "object1. object2. object3."
        text_input = ". ".join(vocabulary) + "."
        
        # Use CAPTION_TO_PHRASE_GROUNDING task for vocabulary-based detection
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
                use_cache=False  # Fix for past_key_values NoneType error with beam search
            )
        
        # Decode the generated text
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        # Parse the output
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_pil.width, image_pil.height)
        )
        
        # Convert to detection format
        detections = self._parse_florence_output(parsed_answer)
        
        return detections
    
    def _parse_florence_output(self, parsed_answer):
        """
        Parse Florence-2 output into detection format.
        
        Args:
            parsed_answer: Parsed output from Florence-2
            
        Returns:
            List of detections with bbox and label
        """
        detections = []
        
        # Florence-2 returns a dictionary with task-specific keys
        # For CAPTION_TO_PHRASE_GROUNDING, the key is the task name
        result = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
        
        if not result:
            return detections
        
        # Extract labels and bboxes
        labels = result.get('labels', [])
        bboxes = result.get('bboxes', [])
        
        # Create detection for each bbox-label pair
        for label, bbox in zip(labels, bboxes):
            # bbox is in format [x1, y1, x2, y2]
            detections.append({
                'bbox': bbox,
                'label': label,
                'confidence': 1.0  # Florence-2 doesn't provide confidence scores
            })
        
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
            
            expanded_bbox = expand_boxes(bbox, expansion_factor=0.125, 
                                   image_shape=image.shape[:2])

            # Convert to numpy array [x1, y1, x2, y2]
            input_box = np.array(expanded_bbox)
            
            # Predict mask
            masks, scores, logits = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False  # Single mask per box
            )
            
            # Add mask to detection
            det['mask'] = masks[0]  # Binary mask (H, W)
        
        print(f"✓ Generated {len(detections)} segmentation masks")
        return detections