"""
COCO Export
Export detections to COCO JSON format.
"""

import json
import datetime
from pathlib import Path


def export_coco(detections, image_path, image_shape, output_path):
    """
    Export detections to COCO format.
    
    Args:
        detections: List of detection dicts
        image_path: Path to image
        image_shape: (height, width)
        output_path: Where to save JSON
    """
    height, width = image_shape
    image_path = Path(image_path)
    
    # Get unique labels
    unique_labels = sorted(list(set(det['label'] for det in detections)))
    
    # Create COCO structure
    coco_data = {
        'info': {
            'description': 'Interior Detection Dataset',
            'version': '1.0',
            'year': datetime.datetime.now().year,
            'date_created': datetime.datetime.now().strftime('%Y/%m/%d')
        },
        'images': [{
            'id': 1,
            'file_name': image_path.name,
            'width': width,
            'height': height
        }],
        'categories': [
            {'id': i+1, 'name': label, 'supercategory': 'interior'}
            for i, label in enumerate(unique_labels)
        ],
        'annotations': []
    }
    
    # Create label to ID mapping
    label_to_id = {label: i+1 for i, label in enumerate(unique_labels)}
    
    # Add annotations
    for ann_id, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det['bbox']
        bbox_coco = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
        
        annotation = {
            'id': ann_id,
            'image_id': 1,
            'category_id': label_to_id[det['label']],
            'bbox': bbox_coco,
            'area': float((x2-x1) * (y2-y1)),
            'iscrowd': 0
        }
        
        coco_data['annotations'].append(annotation)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Exported COCO: {output_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")