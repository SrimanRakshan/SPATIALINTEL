import os
import argparse
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: Could not import segment_anything.")
    print("Please install via: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

from .utils import check_sam_weights, overlay_masks, load_scene_graph

def run_instance_segmentation(images_dir: str, scene_graph_path: str, output_dir: str, model_type: str = "vit_b"):
    """
    Run Segment Anything (SAM) conditioned on YOLOv8 bounding boxes.
    Extracts precise instance masks for 3D projection.
    """
    images_dir = Path(images_dir)
    scene_graph_path = Path(scene_graph_path)
    output_dir = Path(output_dir)
    
    if not images_dir.exists() or not scene_graph_path.exists():
        print(f"Error: Missing inputs. Ensure {images_dir} and {scene_graph_path} exist.")
        exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ensure Model Weights
    weights_path = check_sam_weights(model_type, str(output_dir / "weights"))
    
    # 2. Initialize SAM
    print(f"Loading SAM ({model_type}) from {weights_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=weights_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # 3. Load 2D Scene Graph (YOLO detections)
    scene_graph_2d = load_scene_graph(str(scene_graph_path))
    frames_data = scene_graph_2d.get("frames", {})
    
    # Will store the enriched scene graph with mask references
    enriched_scene_graph = {"frames": {}}
    
    print(f"Running Instance Segmentation extraction on {len(frames_data)} frames...")
    
    for frame_name, detections in tqdm(frames_data.items(), desc="Segmenting frames"):
        img_path = images_dir / frame_name
        if not img_path.exists():
            continue
            
        # Load image
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        predictor.set_image(image_rgb)
        
        frame_results = []
        frame_masks = []
        
        for idx, det in enumerate(detections):
            bbox = np.array(det["bbox"]) # [x1, y1, x2, y2]
            
            # Predict mask using bounding box prompt
            masks, scores, logits = predictor.predict(
                box=bbox,
                multimask_output=False
            )
            
            best_mask = masks[0]
            
            # Save boolean mask to disk using numpy compressed
            mask_filename = f"{img_path.stem}_obj{idx:03d}_{det['class_name']}.npz"
            mask_filepath = masks_dir / mask_filename
            np.savez_compressed(mask_filepath, mask=best_mask)
            
            # Update detection data
            det_enriched = det.copy()
            det_enriched["mask_path"] = str(mask_filepath.relative_to(output_dir))
            det_enriched["mask_confidence"] = float(scores[0])
            frame_results.append(det_enriched)
            
            frame_masks.append((best_mask, det['class_name']))
            
        enriched_scene_graph["frames"][frame_name] = frame_results
        
        # Overlay masks for visualization
        if frame_masks:
            vis_img = overlay_masks(image_bgr, frame_masks)
            cv2.imwrite(str(vis_dir / frame_name), vis_img)
            
    # Save Enriched Scene Graph
    out_json = output_dir / "scene_graph_sam.json"
    with open(out_json, "w") as f:
        json.dump(enriched_scene_graph, f, indent=4)
        
    print(f"\nInstance Segmentation Complete.")
    print(f"Masks saved to {masks_dir}")
    print(f"Enriched JSON saved to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM Instance Segmentation on YOLO detections.")
    parser.add_argument("--images", required=True, help="Input directory containing preprocessed images")
    parser.add_argument("--scene_graph", required=True, help="Path to YOLO 2D scene graph JSON")
    parser.add_argument("--output", required=True, help="Output directory to store masks and enriched graph")
    parser.add_argument("--model", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    
    args = parser.parse_args()
    run_instance_segmentation(args.images, args.scene_graph, args.output, args.model)
