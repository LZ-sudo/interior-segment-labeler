"""
Simple Configuration
Basic vocabulary and settings for interior detection.
"""

# Interior vocabulary - common furniture and objects  
VOCABULARY = [
    # Seating
    'sofa', 'couch', 'chair', 'armchair', 'bench', 'stool',
    
    # Tables
    'table', 'desk', 'coffee table', 'dining table', 'side table',
    
    # Storage
    'cabinet', 'shelf', 'bookshelf', 'drawer', 'closet', 'wardrobe',
    
    # Bedroom
    'bed', 'nightstand', 'dresser', 'mirror',
    
    # Lighting
    'lamp', 'ceiling light', 'chandelier',
    
    # Decor
    'plant', 'picture', 'painting', 'vase', 'curtain', 'rug', 'carpet',
    
    # Electronics
    'tv', 'television', 'monitor', 'computer', 'standing fan',
    
    # Kitchen
    'refrigerator', 'oven', 'microwave', 'sink',
    
    # Bathroom
    'toilet', 'bathtub', 'shower',
    
    # Other
    'door', 'window', 'pillow', 'cushion'
]

# Model settings - using lightest model (vit_b)
MODEL_CONFIG = {
    'sam_checkpoint': 'sam_vit_b_01ec64.pth',
    'sam_model_type': 'vit_b',
    'device': 'cuda',  # Change to 'cpu' if no GPU
    'box_threshold': 0.3,  # Minimum confidence
    'text_threshold': 0.25
}

# Visualization settings
VIZ_CONFIG = {
    'box_thickness': 2,
    'font_scale': 0.5,
    'show_confidence': True
}