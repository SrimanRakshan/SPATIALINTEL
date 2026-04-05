import os
import argparse
import subprocess
from pathlib import Path

def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    
def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    
    runner = (
        "import sys, torch, warnings, os, subprocess; "
        "old_popen = subprocess.Popen; "
        "subprocess.Popen = lambda *a, **k: old_popen(*a, **{**k, 'env': None} if k.get('env') == {} else k); "
        "os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'; "
        "sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None; "
        "warnings.filterwarnings('ignore', category=FutureWarning); "
        "_old_load = torch.load; "
        "torch.load = lambda *a, **k: _old_load(*a, **{**k, 'weights_only': False}); "
        "import mediapy; mediapy.set_ffmpeg(r'C:\\ffmpeg\\bin\\ffmpeg.exe'); "
        "from nerfstudio.scripts.render import entrypoint; "
        "sys.argv=sys.argv[1:]; "
        "entrypoint()"
    )
    
    wrapper_cmd = ["python", "-c", runner] + cmd
    
    try:
        subprocess.run(wrapper_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        exit(1)

def export_video(config_path: str, output_path: str, trajectory_type: str = "interpolate"):
    """
    Renders a 360-degree high-resolution video of the reconstructed scene.
    By default, computes a smooth spiral or interpolation path around the object.
    """
    config_path = Path(config_path)
    output_path = Path(output_path)
    
    if not config_path.exists():
        print(f"Error: Could not find training config at {config_path}")
        exit(1)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nRendering high-resolution 360-degree video...")
    print("This will compute novel views along the camera trajectory.")
    
    cmd1 = [
        "ns-render", trajectory_type,
        "--load-config", str(config_path),
        "--output-path", str(output_path)
    ]
    
    # Depending on NeRF memory, 1080p rendering may OOM directly. 
    # If so, ns-render internally batches the rays.
    run_cmd(cmd1, "Video Rendering")
    
    print(f"\nVideo successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 360 trajectory video from a trained NeRF model.")
    parser.add_argument("--config", required=True, help="Path to the trained model's config.yml")
    parser.add_argument("--output", required=True, help="Path to the output MP4 video file")
    parser.add_argument("--trajectory", type=str, default="interpolate", choices=["spiral", "interpolate"], help="Type of camera path")
    
    args = parser.parse_args()
    export_video(args.config, args.output, args.trajectory)
