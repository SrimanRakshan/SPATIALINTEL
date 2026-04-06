"""
Phase 2 — COLMAP Sparse 3D Reconstruction.

Execution guarantees:
  [IDEMPOTENCY]   Skips if undistorted/ output already exists unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer per stage.
  [CROSS-PLATFORM] colmap binary detected via find_executable (shutil.which).
  [SAFETY]        List-based subprocess; no shell=True.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.subprocess_runner import find_executable, run_cmd
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("phase2.run_colmap")


def run_colmap(
    images_dir: str,
    colmap_workspace: str,
    force_recompute: bool = False,
) -> None:
    """
    Run full COLMAP pipeline.
    Public signature UNCHANGED (force_recompute is a new optional kwarg).
    """
    images_dir_p = Path(images_dir)
    workspace_p = Path(colmap_workspace)
    undistorted_dir = workspace_p / "undistorted"

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(undistorted_dir):
        logger.info(
            f"Undistorted output already exists: {undistorted_dir}. Skipping. "
            "Use --force_recompute to re-run."
        )
        return

    image_files = list(images_dir_p.glob("*.jpg")) + list(images_dir_p.glob("*.png"))
    if not images_dir_p.exists() or not image_files:
        logger.error(f"No images found at {images_dir_p}")
        logger.error("Hint: Run Phase 1 (extract_frames + subsample + preprocess) first.")
        raise SystemExit(1)

    # [CROSS-PLATFORM] Locate colmap via shutil.which
    try:
        colmap_bin = find_executable("colmap")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Hint: Install COLMAP — https://colmap.github.io/install.html")
        raise SystemExit(1)

    logger.info(f"Using COLMAP: {colmap_bin}")

    workspace_p.mkdir(parents=True, exist_ok=True)

    db_path = str(workspace_p / "database.db")
    sparse_dir = str(workspace_p / "sparse")
    img_dir_str = str(images_dir_p)

    if (workspace_p / "database.db").exists():
        logger.warning(f"Removing existing database: {db_path}")
        (workspace_p / "database.db").unlink()

    (workspace_p / "sparse").mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    with phase_timer(logger, "Feature Extraction"):
        run_cmd([
            colmap_bin, "feature_extractor",
            "--database_path", db_path,
            "--image_path", img_dir_str,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
        ], "Feature Extraction")

    with phase_timer(logger, "Feature Matching"):
        run_cmd([
            colmap_bin, "exhaustive_matcher",
            "--database_path", db_path,
        ], "Feature Matching")

    with phase_timer(logger, "Sparse Reconstruction"):
        run_cmd([
            colmap_bin, "mapper",
            "--database_path", db_path,
            "--image_path", img_dir_str,
            "--output_path", sparse_dir,
        ], "Sparse Reconstruction")

    model_dir = workspace_p / "sparse" / "0"
    if not model_dir.exists():
        logger.error("COLMAP mapper did not produce a valid model at sparse/0.")
        logger.error("Hint: Check that images have sufficient overlap for feature matching.")
        raise SystemExit(1)

    with phase_timer(logger, "Image Undistortion"):
        run_cmd([
            colmap_bin, "image_undistorter",
            "--image_path", img_dir_str,
            "--input_path", str(model_dir),
            "--output_path", str(undistorted_dir),
            "--output_type", "COLMAP",
        ], "Image Undistortion")

    logger.info(f"COLMAP complete. Undistorted output → {undistorted_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP pipeline for sparse 3D reconstruction.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    run_colmap(args.images, args.workspace, args.force_recompute)
