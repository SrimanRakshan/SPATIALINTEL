"""
Evaluation — NeRF Quantitative Metrics (PSNR, SSIM, LPIPS).

Execution guarantees:
  [IDEMPOTENCY]  Skips if evaluation_metrics.json exists unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer.
  [SAFETY]       .get() guards on metric keys; PyTorch 2.6 patch preserved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import load_json
from utils.subprocess_runner import build_nerf_runner, run_cmd
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("evaluation.evaluate_nerf")


def evaluate_nerf(
    config_path: str,
    output_dir: str,
    force_recompute: bool = False,
) -> None:
    """
    Evaluates a trained NeRF model. Public signature UNCHANGED.
    """
    config_path_p = Path(config_path)
    output_dir_p = Path(output_dir)
    metrics_out = output_dir_p / "evaluation_metrics.json"

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(metrics_out):
        logger.info(f"Metrics already exist: {metrics_out}. Skipping.")
        _print_metrics(metrics_out)
        return

    if not config_path_p.exists():
        logger.error(f"Config not found: {config_path_p}")
        logger.error("Hint: Run Phase 3 (run_nerf.py --train) first.")
        raise SystemExit(1)

    output_dir_p.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Quantitative Evaluation (PSNR, SSIM, LPIPS)...")

    runner = build_nerf_runner(
        "from nerfstudio.scripts.eval import entrypoint",
        include_mediapy=False,
    )
    cmd = [
        "python", "-c", runner,
        "ns-eval",
        "--load-config", str(config_path_p),
        "--output-path", str(metrics_out),
    ]

    with phase_timer(logger, "NeRF Evaluation"):
        run_cmd(cmd, "NeRF Evaluation")

    if metrics_out.exists():
        _print_metrics(metrics_out)


def _print_metrics(metrics_path: Path) -> None:
    try:
        metrics = load_json(metrics_path)
        results = metrics.get("results", {})
        psnr = results.get("psnr", float("nan"))
        ssim_val = results.get("ssim", float("nan"))
        lpips = results.get("lpips", float("nan"))
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"PSNR  : {psnr:.2f} dB  (higher is better)")
        logger.info(f"SSIM  : {ssim_val:.4f} (closer to 1.0 is better)")
        logger.info(f"LPIPS : {lpips:.4f} (lower is better)")
        logger.info(f"Full metrics → {metrics_path}")
    except Exception as e:
        logger.warning(f"Could not parse metrics file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NeRF model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    evaluate_nerf(args.config, args.output, args.force_recompute)
