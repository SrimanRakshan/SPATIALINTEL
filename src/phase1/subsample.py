"""
Phase 1 — Frame Subsampling.

Execution guarantees:
  [REPRODUCIBILITY] set_global_seed(42) for any stochastic operations.
  [IDEMPOTENCY]     Skips if output already has frames, unless --force_recompute.
  [OBSERVABILITY]   Structured logger + phase_timer; per-stage counts logged.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir, glob_images
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists, set_global_seed

logger = get_logger("phase1.subsample")


def _blur_score(ipath: Path) -> Tuple[Path, float]:
    img = cv2.imread(str(ipath))
    if img is None:
        return ipath, -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ipath, cv2.Laplacian(gray, cv2.CV_64F).var()


def _check_ssim(img1: np.ndarray, img2: np.ndarray, threshold: float) -> bool:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold


def subsample_frames(
    input_dir: str,
    output_dir: str,
    blur_threshold: float = 100.0,
    ssim_threshold: float = 0.95,
    num_workers: int = 4,
    force_recompute: bool = False,
) -> None:
    """
    Remove blurry and redundant (near-duplicate) frames.
    Public signature UNCHANGED.
    """
    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)

    if not input_dir_p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir_p}")

    ensure_dir(output_dir_p)

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(output_dir_p):
        existing = list(output_dir_p.glob("frame_*.jpg"))
        if existing:
            logger.info(
                f"Found {len(existing)} frames in {output_dir_p}. "
                "Skipping. Use --force_recompute to re-run."
            )
            return

    # [REPRODUCIBILITY]
    set_global_seed(42)

    image_paths = glob_images(input_dir_p)
    if not image_paths:
        logger.error(f"No images found in {input_dir_p}")
        raise SystemExit(1)

    logger.info(f"Found {len(image_paths)} images. Blur-scoring with {num_workers} workers...")

    # Parallel blur scoring
    blur_scores: dict[Path, float] = {}
    with phase_timer(logger, "Blur Scoring"):
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_blur_score, p): p for p in image_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring blur"):
                path, score = fut.result()
                blur_scores[path] = score

    candidates: List[Path] = [
        p for p in image_paths if blur_scores.get(p, -1.0) >= blur_threshold
    ]
    blurry_count = len(image_paths) - len(candidates)
    logger.info(f"Blur filter: removed {blurry_count}, kept {len(candidates)}")

    prev_image: Optional[np.ndarray] = None
    saved_idx = 0
    duplicate_count = 0

    with phase_timer(logger, "SSIM Deduplication"):
        for ipath in tqdm(candidates, desc="Deduplicating frames"):
            image = cv2.imread(str(ipath))
            if image is None:
                continue
            if prev_image is not None and _check_ssim(prev_image, image, ssim_threshold):
                duplicate_count += 1
                continue
            out_path = output_dir_p / f"frame_{saved_idx:05d}.jpg"
            cv2.imwrite(str(out_path), image)
            prev_image = image
            saved_idx += 1

    logger.info(
        f"Subsampling complete. "
        f"Blurry={blurry_count} | Duplicates={duplicate_count} | Kept={saved_idx}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample frames by removing blurry and redundant images.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--blur_threshold", type=float, default=100.0)
    parser.add_argument("--ssim_threshold", type=float, default=0.90)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    subsample_frames(
        args.input, args.output,
        args.blur_threshold, args.ssim_threshold,
        args.num_workers, args.force_recompute,
    )
