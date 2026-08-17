"""
Phase 3 — NeRF Training via Nerfstudio CLI.

Execution guarantees:
  [IDEMPOTENCY]   Skips ns-train if checkpoint already exists (unless --force_recompute).
  [OBSERVABILITY] Structured logger + phase_timer; GPU env vars logged.
  [RESOURCE]      Subprocess stdout streamed in real-time via run_cmd.
  [SAFETY]        find_executable() validates ns-process-data / ns-train on PATH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.subprocess_runner import find_executable, run_cmd
from utils.logger import get_logger, phase_timer

logger = get_logger("phase3.run_nerf")


def find_existing_checkpoint(ns_workspace: Path, model_type: str) -> Optional[Path]:
    """Returns the most recent config.yml for the given model type, or None."""
    model_dir = ns_workspace / model_type
    if not model_dir.exists():
        return None
    configs = sorted(model_dir.rglob("config.yml"))
    return configs[-1] if configs else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process COLMAP data for Nerfstudio and train Nerfacto/Splatfacto."
    )
    parser.add_argument("--colmap_undistorted", required=True)
    parser.add_argument("--ns_workspace", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--high_res", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force_recompute", action="store_true",
                        help="Re-train even if a checkpoint already exists.")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap_undistorted).resolve()
    ns_workspace = Path(args.ns_workspace).resolve()
    ns_workspace.mkdir(parents=True, exist_ok=True)

    # [CROSS-PLATFORM] Locate nerfstudio CLIs via shutil.which
    try:
        ns_process = find_executable("ns-process-data")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Hint: Install Nerfstudio: pip install nerfstudio")
        raise SystemExit(1)

    images_preprocessed_str = Path("data/scenes/scene_001/images_preprocessed").as_posix()
    colmap_model_path_str = (colmap_dir / "sparse" / "0").as_posix()

    cmd1 = [
        ns_process, "images",
        "--data", images_preprocessed_str,
        "--output-dir", ns_workspace.as_posix(),
        "--colmap-model-path", colmap_model_path_str,
        "--skip-colmap",
    ]

    with phase_timer(logger, "Nerfstudio Data Processing"):
        run_cmd(cmd1, "Nerfstudio Data Processing")

    if args.train:
        model_type = "splatfacto" if args.high_res else "nerfacto"

        # [IDEMPOTENCY] Skip re-training if checkpoint exists
        if not args.force_recompute:
            existing = find_existing_checkpoint(ns_workspace, model_type)
            if existing:
                logger.info(f"Checkpoint found: {existing}. Skipping ns-train.")
                logger.info("Use --force_recompute to retrain from scratch.")
                return

        try:
            ns_train = find_executable("ns-train")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise SystemExit(1)

        cmd2 = [
            ns_train, model_type,
            "--data", ns_workspace.as_posix(),
            "--vis", "viewer",
        ]
        if not args.high_res:
            cmd2.extend([
                "--pipeline.datamanager.train-num-rays-per-batch", "2048",
                "--pipeline.model.eval-num-rays-per-chunk", "2048",
            ])

        gpu_env = {
            "CUDA_VISIBLE_DEVICES": str(args.device),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        }
        logger.info(f"Starting {model_type} training. CUDA_VISIBLE_DEVICES={args.device}")

        with phase_timer(logger, f"{model_type} Training"):
            run_cmd(cmd2, f"{model_type} Training", env=gpu_env)


if __name__ == "__main__":
    main()
