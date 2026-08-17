"""
Evaluation — 3D Mesh Export.

Execution guarantees:
  [IDEMPOTENCY]  Skips if output_dir already has .ply unless --force_recompute.
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

logger = get_logger("evaluation.export_mesh")


def export_3d_mesh(
    config_path: str,
    output_dir: str,
    method: str = "tsdf",
    force_recompute: bool = False,
) -> None:
    """
    Exports the trained NeRF into a 3D mesh. Public signature UNCHANGED.

    Windows patches: WinError 87 + PyTorch 2.6 preserved via build_nerf_runner.
    """
    output_dir_p = Path(output_dir)

    # [IDEMPOTENCY] Check for any .ply in output dir
    if not force_recompute and output_dir_p.exists():
        existing_ply = list(output_dir_p.glob("*.ply"))
        if existing_ply:
            logger.info(f"Mesh already exists: {existing_ply[0]}. Skipping.")
            return

    output_dir_p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting 3D mesh ({method}) → {output_dir_p}")

    runner = build_nerf_runner(
        "from nerfstudio.scripts.exporter import entrypoint",
        include_mediapy=False,
    )
    cmd = [
        "python", "-c", runner,
        "ns-export", method,
        "--load-config", config_path,
        "--output-dir", output_dir,
    ]

    with phase_timer(logger, f"3D Mesh Export ({method})"):
        run_cmd(cmd, f"3D Mesh Export ({method})")

    logger.info(f"✅ Mesh exported → {output_dir_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a 3D mesh from a trained NeRF.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", type=str, default="tsdf", choices=["tsdf", "poisson"])
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    export_3d_mesh(args.config, args.output_dir, args.method, args.force_recompute)
