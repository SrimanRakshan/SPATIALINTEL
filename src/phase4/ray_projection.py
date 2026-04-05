import os
import argparse
import json
import numpy as np
from pathlib import Path

def load_transforms(transforms_path: str):
    """ Loads the NeRFstudio colmap transforms.json to get camera extrinsics/intrinsics """
    with open(transforms_path, "r") as f:
        return json.load(f)

def run_ray_projection(scene_graph_sam_path: str, transforms_path: str, output_dir: str):
    """
    Simulates projecting 2D localized instance masks into 3D world space.
    Reads camera transforms (from NeRF/COLMAP) and casts hypothetical rays.
    """
    scene_graph_sam_path = Path(scene_graph_sam_path)
    transforms_path = Path(transforms_path)
    output_dir = Path(output_dir)
    
    if not scene_graph_sam_path.exists():
        print(f"Error: Missing 2D SAM scene graph: {scene_graph_sam_path}")
        exit(1)
        
    if not transforms_path.exists():
        print(f"Error: Missing NeRF transforms: {transforms_path}")
        exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(scene_graph_sam_path, "r") as f:
        scene_graph_2d = json.load(f)
        
    transforms = load_transforms(transforms_path)
    
    # Simple Hash map to aggregate 3D points by class name
    # In a full production Q1 implementation, this would involve ray marching against the trained NeRF density grid
    # For now, we simulate the aggregation to build a queryable 3D Scene Graph
    scene_graph_3d = {
        "objects": []
    }
    
    # We aggregate instances of the same class across frames to compute a rough 3D centroid
    class_aggregates = {}
    
    frames_data = scene_graph_2d.get("frames", {})
    transform_frames = {Path(f["file_path"]).name: f["transform_matrix"] for f in transforms.get("frames", [])}
    
    print("Projecting 2D semantic maps into 3D space...")
    
    for frame_name, detections in frames_data.items():
        if frame_name not in transform_frames:
            continue
            
        c2w = np.array(transform_frames[frame_name])
        
        # Camera center is the translation component of the c2w matrix
        camera_origin = c2w[:3, 3]
        
        for det in detections:
            cls_name = det["class_name"]
            
            # Use center of bounding box to cast a primary ray
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            # Simple heuristic depth estimation for monocular (since we aren't querying the actual NeRF MLP here yet)
            # In Phase 6/7, this integrates with depth-regularized NeRFs
            simulated_depth = 2.0 + np.random.normal(0, 0.5) 
            
            # Direction vector (simplified pinhole camera projection)
            fl_x = transforms.get("fl_x", 1000)
            fl_y = transforms.get("fl_y", 1000)
            cx_img = transforms.get("cx", 500)
            cy_img = transforms.get("cy", 500)
            
            dir_x = (cx - cx_img) / fl_x
            dir_y = (cy - cy_img) / fl_y
            dir_z = 1.0
            
            direction = np.array([dir_x, dir_y, dir_z])
            direction = direction / np.linalg.norm(direction)
            
            # Transform direction to world space
            world_dir = c2w[:3, :3] @ direction
            
            # 3D point = Origin + Direction * Depth
            pt_3d = camera_origin + world_dir * simulated_depth
            
            if cls_name not in class_aggregates:
                class_aggregates[cls_name] = []
                
            class_aggregates[cls_name].append(pt_3d)
            
    # Compute object centroids
    for cls_name, points in class_aggregates.items():
        pts = np.array(points)
        centroid = np.mean(pts, axis=0)
        
        # Simple heuristic to determine "instances" based on point spread
        # Here we just represent it as one major entity for the LLM
        scene_graph_3d["objects"].append({
            "name": cls_name,
            "position": centroid.tolist(),
            "observations": len(points)
        })
        
    out_json = output_dir / "scene_graph_3d.json"
    with open(out_json, "w") as f:
        json.dump(scene_graph_3d, f, indent=4)
        
    print(f"\n3D Ray Projection Complete.")
    print(f"Generated 3D Scene Graph stored in {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 2D SAM masks into 3D using NeRF extrinsics.")
    parser.add_argument("--scene_graph_sam", required=True, help="Path to 2D enriched SAM JSON graph")
    parser.add_argument("--transforms", required=True, help="Path to Nerfstudio transforms.json")
    parser.add_argument("--output", required=True, help="Output directory to store the final 3D scene graph")
    
    args = parser.parse_args()
    run_ray_projection(args.scene_graph_sam, args.transforms, args.output)
