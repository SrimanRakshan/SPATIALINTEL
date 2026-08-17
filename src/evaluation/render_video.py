"""
Evaluation — NeRF Video Rendering.

Execution guarantees:
  [IDEMPOTENCY]  Skips if output video already exists unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer.
  [SAFETY]       Windows patches preserved via build_nerf_runner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.subprocess_runner import build_nerf_runner, run_cmd
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("evaluation.render_video")


def export_video(
    config_path: str,
    output_path: str,
    trajectory_type: str = "interpolate",
    force_recompute: bool = False,
) -> None:
    """
    Renders a 360-degree NeRF video. Public signature UNCHANGED.

    Windows patches: WinError 87 + PyTorch 2.6 weights_only preserved
    inside build_nerf_runner() → src/utils/subprocess_runner.py.
    """
    config_path_p = Path(config_path)
    output_path_p = Path(output_path)

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(output_path_p):
        logger.info(f"Output already exists: {output_path_p}. Skipping.")
        return

    if not config_path_p.exists():
        logger.error(f"Config not found: {config_path_p}")
        logger.error("Hint: Run Phase 3 (run_nerf.py --train) first.")
        raise SystemExit(1)

    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rendering {trajectory_type} trajectory → {output_path_p}")

    runner = build_nerf_runner(
        "from nerfstudio.scripts.render import entrypoint",
        include_mediapy=True,
    )
    cmd = [
        "python", "-c", runner,
        "ns-render", trajectory_type,
        "--load-config", str(config_path_p),
        "--output-path", str(output_path_p),
    ]

    with phase_timer(logger, "Video Rendering"):
        run_cmd(cmd, "Video Rendering")

    logger.info(f"✅ Video saved → {output_path_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 360 trajectory video from a trained NeRF.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory", type=str, default="interpolate", choices=["spiral", "interpolate"])
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    export_video(args.config, args.output, args.trajectory, args.force_recompute)
