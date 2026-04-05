import os
import urllib.request
import json
import numpy as np
import cv2
from pathlib import Path

SAM_MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

def check_sam_weights(model_type: str, weights_dir: str):
    """
    Downloads the official SAM weights if they are not already cached.
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    url = SAM_MODELS.get(model_type)
    if not url:
        raise ValueError(f"Unknown SAM model type: {model_type}")
        
    filename = url.split("/")[-1]
    filepath = weights_dir / filename
    
    if not filepath.exists():
        print(f"Downloading SAM {model_type} weights ({filename})...")
        urllib.request.urlretrieve(url, str(filepath))
        print("Download complete.")
        
    return str(filepath)

def load_scene_graph(filepath: str):
    """ Loads the JSON scene graph into memory. """
    with open(filepath, "r") as f:
        return json.load(f)

def overlay_masks(image_bgr: np.ndarray, masks_with_labels: list, alpha: float = 0.5):
    """
    Overlays a list of boolean masks onto a BGR image with distinct colors.
    """
    img_overlay = image_bgr.copy()
    
    # Generate random vibrant colors for each mask
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(len(masks_with_labels), 3), dtype=np.uint8)
    
    for i, (mask, label) in enumerate(masks_with_labels):
        color = colors[i].tolist()
        
        # Ensure mask is boolean and correct shape
        mask_bool = mask.astype(bool)
        
        # Create a colored mask with the same shape as image
        colored_mask = np.zeros_like(image_bgr)
        colored_mask[mask_bool] = color
        
        # Blend it using alpha weighting where the mask is True
        roi = img_overlay[mask_bool]
        blended = cv2.addWeighted(roi, 1 - alpha, np.array(color, dtype=np.uint8) * np.ones_like(roi), alpha, 0)
        img_overlay[mask_bool] = blended
        
    return img_overlay
