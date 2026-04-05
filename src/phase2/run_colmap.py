import os
import subprocess
import argparse
from pathlib import Path
import json

def run_cmd(cmd: str, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        exit(1)

def run_colmap(images_dir: str, colmap_workspace: str):
    images_dir = Path(images_dir)
    colmap_workspace = Path(colmap_workspace)
    
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))) == 0:
        print(f"Error: No images found at {images_dir}")
        exit(1)
        
    colmap_workspace.mkdir(parents=True, exist_ok=True)
    
    db_path = str(Path(colmap_workspace / "database.db"))
    sparse_dir = str(Path(colmap_workspace / "sparse"))
    undistorted_dir = str(Path(colmap_workspace / "undistorted"))
    img_dir_str = str(images_dir)
    
    if Path(db_path).exists():
        print(f"Warning: {db_path} already exists. Removing it to start fresh.")
        Path(db_path).unlink()
        
    Path(sparse_dir).mkdir(parents=True, exist_ok=True)
    Path(undistorted_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Extraction
    cmd1 = f'colmap feature_extractor --database_path "{db_path}" --image_path "{img_dir_str}" --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1'
    run_cmd(cmd1, "Feature Extraction")
    
    # 2. Feature Matching
    cmd2 = f'colmap exhaustive_matcher --database_path "{db_path}"'
    run_cmd(cmd2, "Feature Matching")
    
    # 3. Mapper (Sparse Reconstruction)
    cmd3 = f'colmap mapper --database_path "{db_path}" --image_path "{img_dir_str}" --output_path "{sparse_dir}"'
    run_cmd(cmd3, "Sparse Reconstruction")
    
    # 4. Image Undistorter
    # Need to check if a model was created (e.g., sparse/0)
    model_dir_path = Path(sparse_dir) / "0"
    model_dir = str(model_dir_path)
    if not model_dir_path.exists():
        print("Error: COLMAP mapper did not produce a valid model at sparse/0.")
        exit(1)
        
    cmd4 = f'colmap image_undistorter --image_path "{img_dir_str}" --input_path "{model_dir}" --output_path "{undistorted_dir}" --output_type COLMAP'
    run_cmd(cmd4, "Image Undistortion")
    
    print("\nCOLMAP Pipeline complete.")
    print(f"Undistorted images and camera poses saved to {undistorted_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP pipeline for sparse 3D reconstruction.")
    parser.add_argument("--images", required=True, help="Input directory containing preprocessed images")
    parser.add_argument("--workspace", required=True, help="Workspace directory to store database, sparse, and undistorted models")
    
    args = parser.parse_args()
    run_colmap(args.images, args.workspace)
