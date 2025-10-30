# Interior Segment Labeler

A computer vision pipeline for detecting and segmenting furniture and objects in interior images using state-of-the-art object detection and segmentation models.

## Overview

This tool automatically identifies and labels furniture and objects in indoor scenes, making it useful for interior design analysis, room layout understanding, and spatial design studies.

## Pipeline

The detection pipeline consists of two stages:

1. **Object Detection (OWL-ViT v2)**: Identifies furniture and objects in the image using open-vocabulary object detection
2. **Segmentation (SAM-HQ)**: Creates precise pixel-level masks for each detected object

### How It Works

```
Input Image → OWL-ViT v2 Detection → SAM-HQ Segmentation → Annotated Output + COCO JSON
```

1. **OWL-ViT v2** scans the image and detects objects based on the configured vocabulary
2. For each detection, a bounding box and confidence score is generated
3. **SAM-HQ (Segment Anything Model High Quality)** refines each detection into a precise segmentation mask
4. Results are saved as annotated images and COCO format JSON files

## Models Used

### OWL-ViT v2 (Object Detection)
- **Model**: `google/owlv2-large-patch14-ensemble`
- **Provider**: Google Research / Hugging Face
- **Paper**: [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
- **Purpose**: Open-vocabulary object detection - can detect objects based on text descriptions without being limited to predefined classes

### SAM-HQ (Segmentation)
- **Model**: Segment Anything Model - High Quality (ViT-L variant)
- **Provider**: ETH Zurich VIS Group
- **Paper**: [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567)
- **Purpose**: Generates high-quality object masks with improved boundary precision, especially for objects with intricate structures

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

### Process a Single Image
```bash
python main.py path/to/image.jpg
```

### Process a Directory
```bash
python main.py path/to/images/
```

### Custom Output Directory
```bash
python main.py images/ --output custom_results/
```

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

For each processed image, the pipeline generates:

1. **`*_annotated.jpg`**: Image with bounding boxes and labels
2. **`*_comparison.jpg`**: Side-by-side comparison of original and annotated
3. **`*_coco.json`**: Detection results in COCO format

All outputs are saved to the `results/` directory (or custom directory specified with `--output`).

## Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for 5-10x faster processing
- **Batch Processing**: Process directories instead of individual images to amortize model loading time
- **Threshold Tuning**: Lower `box_threshold` to detect more objects, raise it to reduce false positives
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

- **OWL-ViT v2** by Google Research - [Paper](https://arxiv.org/abs/2306.09683) | [Model](https://huggingface.co/google/owlv2-large-patch14-ensemble)
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
- Lower the `box_threshold` in `config.py` (try 0.2 or 0.15)
- Add more specific object descriptions to `VOCABULARY`
- Ensure images are well-lit and objects are clearly visible

## Project Structure

```
interior-segment-labeler/
├── config.py              # Configuration and vocabulary
├── main.py                # Main entry point
├── models/
│   └── model.py          # Model loading and inference
├── visualization.py       # Image annotation utilities
├── export_coco.py        # COCO format export
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Citation

If you use this tool in your research or project, please cite the original models:

```bibtex
@article{minderer2023scaling,
  title={Scaling Open-Vocabulary Object Detection},
  author={Minderer, Matthias and Gritsenko, Alexey and Houlsby, Neil},
  journal={arXiv preprint arXiv:2306.09683},
  year={2023}
}

@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```
