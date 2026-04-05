import argparse
import subprocess
import os
import sys

def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    
    # Use the same monkeypatch wrapper we used for the video render
    # to avoid compatibility issues with PyTorch weights and OS environment bugs.
    runner = (
        "import sys, torch, warnings, os, subprocess; "
        "old_popen = subprocess.Popen; "
        "subprocess.Popen = lambda *a, **k: old_popen(*a, **{**k, 'env': None} if k.get('env') == {} else k); "
        "os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'; "
        "sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None; "
        "warnings.filterwarnings('ignore', category=FutureWarning); "
        "_old_load = torch.load; "
        "torch.load = lambda *a, **k: _old_load(*a, **{**k, 'weights_only': False}); "
        "from nerfstudio.scripts.exporter import entrypoint; "
        "sys.argv=sys.argv[1:]; "
        "entrypoint()"
    )
    
    wrapper_cmd = ["python", "-c", runner] + cmd
    
    try:
        subprocess.run(wrapper_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        exit(1)

def export_3d_mesh(config_path: str, output_dir: str, method: str = "tsdf"):
    """
    Exports the trained NeRF into a standard 3D Mesh geometry (.ply / .obj).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "ns-export", method,
        "--load-config", config_path,
        "--output-dir", output_dir
    ]
    
    run_cmd(cmd, f"3D Mesh Export ({method})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a 3D mesh from a trained NeRF model.")
    parser.add_argument("--config", required=True, help="Path to the trained model's config.yml")
    parser.add_argument("--output_dir", required=True, help="Directory to save the exported mesh")
    parser.add_argument("--method", type=str, default="tsdf", choices=["tsdf", "poisson"], help="Meshing algorithm to use")
    
    args = parser.parse_args()
    export_3d_mesh(args.config, args.output_dir, args.method)
