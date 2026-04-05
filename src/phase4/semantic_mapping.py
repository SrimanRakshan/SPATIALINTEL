import os
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def run_semantic_mapping(images_dir: str, output_dir: str):
    """
    Run YOLOv8 object detection on extracted frames to build a 2D scene graph.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLOv8 model - using 'yolov8n.pt' for faster inference initially
    # Can scale up to yolov8x.pt later if required
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt") 
    
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    scene_graph_2d = {
        "frames": {}
    }
    
    print(f"Running semantic mapping on {len(image_paths)} images...")
    
    for ipath in tqdm(image_paths, desc="Detecting objects"):
        img = cv2.imread(str(ipath))
        if img is None:
            continue
            
        # Run inference
        results = model(img, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy().tolist() # [x1, y1, x2, y2]
            c = box.cls.cpu().numpy().tolist()[0]
            conf = box.conf.cpu().numpy().tolist()[0]
            name = model.names[int(c)]
            
            detections.append({
                "class_id": int(c),
                "class_name": name,
                "confidence": float(conf),
                "bbox": [float(x) for x in b]
            })
            
        scene_graph_2d["frames"][ipath.name] = detections
        
        # Save visualization
        res_img = result.plot()
        cv2.imwrite(str(vis_dir / ipath.name), res_img)
        
    # Save 2D Scene Graph
    out_json = output_dir / "scene_graph_2d.json"
    with open(out_json, "w") as f:
        json.dump(scene_graph_2d, f, indent=4)
        
    print(f"\nSemantic Mapping Complete.")
    print(f"Saved 2D Scene Graph to {out_json}")
    print(f"Saved Visualizations to {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D Semantic Mapping using YOLOv8.")
    parser.add_argument("--images", required=True, help="Input directory containing preprocessed images")
    parser.add_argument("--output", required=True, help="Output directory to store scene graph and visualizations")
    
    args = parser.parse_args()
    run_semantic_mapping(args.images, args.output)
