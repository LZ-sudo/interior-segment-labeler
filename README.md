# Interior Segment Labeler

A computer vision pipeline for detecting and segmenting furniture and objects in interior images using state-of-the-art object detection and segmentation models.

## Overview

This tool automatically identifies and labels furniture and objects in indoor scenes using Florence-2 for open-vocabulary object detection combined with SAM-HQ for precise segmentation.

## Pipeline

The detection pipeline consists of two stages:

1. **Object Detection (OWL-ViT v2)**: Identifies furniture and objects in the image using open-vocabulary object detection
2. **Segmentation (SAM-HQ)**: Creates precise pixel-level masks for each detected object

### How It Works
```
Input Image → Florence-2 Phrase Grounding → SAM-HQ Segmentation → Annotated Output + COCO JSON
```

1. **Florence-2** performs phrase-grounded detection based on the configured vocabulary
2. For each detection, a bounding box is generated
3. **SAM-HQ** refines each detection into a precise segmentation mask
4. Results are saved as annotated images and COCO format JSON files

## Models Used

### Florence-2 (Object Detection)
- **Model**: `microsoft/Florence-2-base` (or Florence-2-large)
- **Provider**: Microsoft
- **Paper**: [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)
- **Purpose**: Open-vocabulary object detection using phrase grounding - detects objects based on text descriptions
- **Size**: ~230MB (base) or ~770MB (large)
- **Note**: Florence-2 doesn't provide confidence scores but offers excellent detection quality

### SAM-HQ (Segmentation)
- **Model**: Segment Anything Model - High Quality (ViT-L variant)
- **Provider**: ETH Zurich VIS Group
- **Paper**: [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567)
- **Purpose**: Generates high-quality object masks with improved boundary precision

## Features

- Detect 40+ types of furniture and interior objects
- Process single images or entire directories
- Generate annotated visualizations with bounding boxes and labels
- Export results in COCO JSON format for further analysis
- Support for both CPU and GPU (CUDA) acceleration

## Installation

### Prerequisites
- Python 3.8+
- (Optional) NVIDIA GPU with CUDA for faster inference

### Setup

1. Clone or download this repository

2. Create a virtual environment:
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
# or
source myenv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install CUDA-enabled PyTorch for GPU acceleration:
```bash
# For CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### First Run

On the first run, the models will be automatically downloaded:
- SAM-HQ weights (~375MB) will download to `models/`
- OWL-ViT v2 weights (~1.5GB) will be cached by Hugging Face

## Usage

### Two Scripts Available

This package provides two scripts:
1. **`segment_image.py`** - Full detection + segmentation pipeline (Florence-2 + SAM-HQ)
2. **`label_changes.py`** - Detection only (Florence-2), outputs JSON coordinates for frontend use

### segment_image.py - Full Pipeline

Process images with detection and segmentation:

```bash
# Process a single image
python segment_image.py path/to/image.jpg

# Process a directory
python segment_image.py path/to/images/

# Custom output directory
python segment_image.py images/ --output custom_results/
```

### label_changes.py - Detection API (NEW)

Get bounding box coordinates for frontend visualization with automatic JSON and image outputs:

```bash
# Basic usage - auto-generates JSON + labeled image
python label_changes.py room.jpg

# With custom vocabulary
python label_changes.py room.jpg "sofa, coffee table, lamp"

# Save to custom directory
python label_changes.py room.jpg --output-dir results/

# Skip image generation (JSON only)
python label_changes.py room.jpg --no-image

# Print to console only (no files)
python label_changes.py room.jpg --no-json --no-image

# Get pixel coordinates instead of normalized
python label_changes.py room.jpg --format pixels

# Disable automatic filtering (keep all detections including duplicates)
python label_changes.py room.jpg --no-filter
```

**Automatic Detection Filtering:**

By default, `label_changes.py` automatically cleans up detections to:
- ✅ Remove action verbs (e.g., "Install", "Add", "Remove") that aren't actual objects
- ✅ Remove duplicate overlapping detections (keeps most descriptive label)
- ✅ Use `--no-filter` to disable this behavior
- ✅ Adjust overlap threshold with `--iou-threshold 0.7` (default)

**Default Output Files:**
- `{image_stem}_detections.json` - Bounding box coordinates (normalized by default)
- `{image_stem}_labeled.jpg` - Annotated image with colored bounding boxes and labels

**Vocabulary Priority:**
1. JSON prompt file (`{image_stem}_prompt.json`) - if exists
2. Command-line vocabulary argument - if provided
3. Default vocabulary from `config.py` - fallback

**Creating a Prompt JSON File:**
For an image named `room.jpg`, create `room_prompt.json`:
```json
{
  "prompts": ["sofa", "coffee table", "lamp", "chair"]
}
```

**Why use `label_changes.py`?**
- ✅ Faster (skips SAM segmentation)
- ✅ Lower memory usage (only Florence-2)
- ✅ Auto-generates JSON + visual output
- ✅ Normalized coordinates (perfect for web frontends)
- ✅ Easy integration with React, Canvas, HTML overlays
- ✅ Same JSON prompt workflow as `segment_image.py`

## Configuration

Edit `config.py` to customize:

### Detection Vocabulary
Add or remove object types in the `VOCABULARY` list:
```python
VOCABULARY = [
    'sofa', 'chair', 'table', 'bed', 'lamp', ...
]
```

### Model Settings
```python
MODEL_CONFIG = {
    'device': 'cuda',  # or 'cpu'
    'box_threshold': 0.3,  # Lower = more detections (try 0.2-0.25)
    'text_threshold': 0.25
}
```

### Visualization Settings
```python
VIZ_CONFIG = {
    'box_thickness': 2,
    'font_scale': 0.5,
    'show_confidence': True
}
```

## Output

### segment_image.py Output

For each processed image, the pipeline generates:

1. **`*_annotated.jpg`**: Image with bounding boxes and labels (disabled visually) and SAM-HQ masks

All outputs are saved to the `results/` directory (or custom directory specified with `--output`).

### label_changes.py Output

Returns JSON with bounding box coordinates. Example output:

**Normalized Format (Default - Recommended for Frontend):**
```json
{
  "image_path": "room.jpg",
  "image_size": {"width": 1920, "height": 1080},
  "vocabulary": ["sofa", "table", "lamp"],
  "detections": [
    {
      "label": "sofa",
      "confidence": 1.0,
      "bbox": [0.125, 0.342, 0.487, 0.856],
      "center": [0.306, 0.599],
      "width": 0.362,
      "height": 0.514
    }
  ],
  "count": 1,
  "coordinate_format": "normalized"
}
```

**Coordinate Systems:**
- **Normalized** (0-1 range): Resolution-independent, perfect for responsive web frontends
- **Pixels** (absolute): Direct pixel coordinates relative to original image size
- **Both**: Includes both formats for maximum flexibility

### Frontend Integration Example

**JavaScript/Canvas:**
```javascript
// Load detections
const detections = await fetch('detections.json').then(r => r.json());

// Draw on canvas
function drawBoxes(canvas, image, detections) {
  const ctx = canvas.getContext('2d');
  detections.detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;
    // Convert normalized to pixels
    const x = x1 * image.width;
    const y = y1 * image.height;
    const w = (x2 - x1) * image.width;
    const h = (y2 - y1) * image.height;

    ctx.strokeStyle = 'red';
    ctx.strokeRect(x, y, w, h);
  });
}
```

**React/HTML Overlay:**
```jsx
function BoundingBoxOverlay({ detections }) {
  return detections.detections.map((det, i) => {
    const [x1, y1, x2, y2] = det.bbox;
    return (
      <div key={i} style={{
        position: 'absolute',
        left: `${x1 * 100}%`,
        top: `${y1 * 100}%`,
        width: `${(x2 - x1) * 100}%`,
        height: `${(y2 - y1) * 100}%`,
        border: '3px solid red',
        pointerEvents: 'none'
      }}>
        <span style={{color: 'red', fontWeight: 'bold'}}>
          {det.label}
        </span>
      </div>
    );
  });
}
```

## Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for 5-10x faster processing
- **Batch Processing**: Process directories instead of individual images to amortize model loading time
- **SAM-HQ Model**: The default ViT-B model provides good balance of speed and quality. For better edge precision (at the cost of speed), you can switch to ViT-L or ViT-H in `config.py`

## Detected Object Types

The pipeline can detect 40+ interior object types including:

- **Seating**: sofa, couch, chair, armchair, bench, stool
- **Tables**: table, desk, coffee table, dining table, side table
- **Storage**: cabinet, shelf, bookshelf, drawer, closet, wardrobe
- **Bedroom**: bed, nightstand, dresser, mirror
- **Lighting**: lamp, ceiling light, chandelier
- **Decor**: plant, picture, painting, vase, curtain, rug, carpet
- **Electronics**: tv, television, monitor, computer
- **Kitchen**: refrigerator, oven, microwave, sink
- **Bathroom**: toilet, bathtub, shower
- **Other**: door, window, pillow, cushion

See `config.py` for the complete vocabulary list.

## Acknowledgments

This project uses the following open-source models:

- **Florence-2** by Microsoft - [Paper](https://arxiv.org/abs/2311.06242) | [Model](https://huggingface.co/microsoft/Florence-2-base)
- **Segment Anything in High Quality (SAM-HQ)**, [NeurIPS 2023] - [Paper](https://arxiv.org/abs/2306.01567) | [GitHub](https://github.com/SysCV/sam-hq.git)

Special thanks to:
- Hugging Face for hosting the OWL-ViT v2 model and transformers library
- Meta AI for releasing SAM as open-source
- The open-source computer vision community

## License

This project uses models with the following licenses:
- OWL-ViT v2: Apache 2.0
- SAM-HQ: Apache 2.0

Please refer to the respective model repositories for detailed license information.

## Troubleshooting

**"CUDA not available" error**:
- Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- Or set `device: 'cpu'` in `config.py`

**Out of memory errors**:
- Switch to CPU mode
- Process images one at a time instead of batches
- Use a smaller SAM-HQ model

**Poor detection results**:
- Add more specific object descriptions to `VOCABULARY`
- Ensure images are well-lit and objects are clearly visible

## Project Structure

```
interior-segment-labeler/
├── config.py              # Configuration and vocabulary
├── segment_image.py       # Full pipeline (detection + segmentation)
├── label_changes.py       # Detection-only API (NEW)
├── models/
│   └── model.py          # Model loading and inference
├── visualization.py       # Image annotation utilities
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Comparison: segment_image.py vs label_changes.py

| Feature | segment_image.py | label_changes.py |
|---------|------------------|------------------|
| **Detection** | ✅ Florence-2 | ✅ Florence-2 |
| **Segmentation** | ✅ SAM-HQ masks | ❌ None |
| **Image Output** | Annotated with masks | Annotated with boxes |
| **JSON Output** | Optional | Default (detections) |
| **Coordinates** | Pixels | Normalized + Pixels |
| **Speed** | Slower (~5-10s/image) | Faster (~1-2s/image) |
| **Memory** | Higher (both models) | Lower (Florence-2 only) |
| **Use Case** | Segmentation masks | Bounding box highlights |
| **Best For** | Research, pixel-masks | Web apps, UI overlays |

## Citation

If you use this tool in your research or project, please cite the original models:
```bibtex
@article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}

@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```
