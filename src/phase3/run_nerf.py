import os
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process COLMAP data for Nerfstudio and train Nerfacto model.")
    parser.add_argument("--colmap_undistorted", required=True, help="Path to COLMAP base dir")
    parser.add_argument("--ns_workspace", required=True, help="Path to output Nerfstudio workspace")
    parser.add_argument("--train", action="store_true", help="Launch training immediately after processing data")
    parser.add_argument("--high_res", action="store_true", help="Use splatfacto or high-res parameters for better image quality")
    
    args = parser.parse_args()
    
    colmap_dir = Path(args.colmap_undistorted).resolve()
    ns_workspace = Path(args.ns_workspace).resolve()
    
    ns_workspace.mkdir(parents=True, exist_ok=True)
    
    ns_workspace_str = ns_workspace.as_posix()
    
    # Process Data
    images_preprocessed_str = Path("data/scenes/scene_001/images_preprocessed").as_posix()
    colmap_model_path_str = (colmap_dir / "sparse" / "0").as_posix()
    
    cmd1 = [
        "ns-process-data", "images",
        "--data", images_preprocessed_str,
        "--output-dir", ns_workspace_str,
        "--colmap-model-path", colmap_model_path_str,
        "--skip-colmap"
    ]
    run_cmd(cmd1, "Nerfstudio Data Processing")
    
    if args.train:
        # User requested higher image quality.
        # Splatfacto (3D Gaussian Splatting) generally trains faster and yields sharper edges than nerfacto.
        model_type = "splatfacto" if args.high_res else "nerfacto"
        
        cmd2 = [
            "ns-train", model_type, 
            "--data", ns_workspace_str, 
            "--vis", "viewer"
        ]
        
        if not args.high_res:
            # Low-VRAM safety defaults for standard nerfacto
            cmd2.extend([
                "--pipeline.datamanager.train-num-rays-per-batch", "2048",
                "--pipeline.model.eval-num-rays-per-chunk", "2048"
            ])
            
        print(f"Starting {model_type} training. Press Ctrl+C to stop.")
        subprocess.run(cmd2)

if __name__ == "__main__":
    main()
