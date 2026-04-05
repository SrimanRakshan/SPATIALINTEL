import os
import argparse
import subprocess
from pathlib import Path
import json

def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    
    # PyTorch 2.4+ enforce weights_only=True during torch.load, breaking older checkpoint setups.
    # We wrap the entrypoint to monkeypatch torch.load and forcibly disable this.
    runner = (
        "import sys, torch, warnings; "
        "warnings.filterwarnings('ignore', category=FutureWarning); "
        "_old_load = torch.load; "
        "torch.load = lambda *a, **k: _old_load(*a, **{**k, 'weights_only': False}); "
        "from nerfstudio.scripts.eval import entrypoint; "
        "sys.argv=sys.argv[1:]; "
        "entrypoint()"
    )
    
    wrapper_cmd = ["python", "-c", runner] + cmd
    
    try:
        subprocess.run(wrapper_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        exit(1)

def evaluate_nerf(config_path: str, output_dir: str):
    """
    Evaluates a trained NeRF model, computing PSNR, SSIM, and LPIPS metrics.
    Requires the path to the config.yml generated during training.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    
    if not config_path.exists():
        print(f"Error: Could not find training config at {config_path}")
        print("Please ensure your NeRF model finished training and produced a config.yml file.")
        exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = output_dir / "evaluation_metrics.json"
    
    # Run Nerfstudio Evaluation script
    cmd1 = [
        "ns-eval",
        "--load-config", str(config_path),
        "--output-path", str(metrics_out)
    ]
    
    print("\nStarting Quantitative Evaluation (PSNR, SSIM, LPIPS)...")
    print("This will process the held-out validation images from the dataset.")
    run_cmd(cmd1, "NeRF Evaluation")
    
    if metrics_out.exists():
        with open(metrics_out, "r") as f:
            metrics = json.load(f)
            
        print("\n=== EVALUATION RESULTS ===")
        print(f"PSNR (Peak Signal-to-Noise Ratio):     {metrics['results']['psnr']:.2f} dB (Higher is better)")
        print(f"SSIM (Structural Similarity Index):    {metrics['results']['ssim']:.4f} (Closer to 1.0 is better)")
        print(f"LPIPS (Learned Perceptual Patch Sim):  {metrics['results']['lpips']:.4f} (Lower is better)")
        print("==========================")
        print(f"Full metrics saved to: {metrics_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NeRF model for quantitative metrics.")
    parser.add_argument("--config", required=True, help="Path to the trained model's config.yml (e.g., outputs/scene_001/nerfacto/.../config.yml)")
    parser.add_argument("--output", required=True, help="Output directory to save the JSON metrics")
    
    args = parser.parse_args()
    evaluate_nerf(args.config, args.output)
