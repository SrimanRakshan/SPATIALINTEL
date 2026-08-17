"""
Phase 1 — Image Preprocessing.

Execution guarantees:
  [IDEMPOTENCY]   Skips if output already has images unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir, glob_images
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("phase1.preprocess")


def _process_one(args: Tuple[Path, Path, int, int, bool]) -> None:
    ipath, output_dir, target_width, target_height, equalize = args
    image = cv2.imread(str(ipath))
    if image is None:
        return
    if target_width > 0 and target_height > 0:
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    elif target_width > 0:
        h, w = image.shape[:2]
        scale = target_width / float(w)
        image = cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    if equalize:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    out_path = output_dir / f"{ipath.stem}.jpg"
    cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def preprocess_images(
    input_dir: str,
    output_dir: str,
    target_width: int = -1,
    target_height: int = -1,
    equalize: bool = False,
    num_workers: int = 4,
    force_recompute: bool = False,
) -> None:
    """
    Preprocess images: resize and optional CLAHE histogram equalisation.
    Public signature UNCHANGED.
    """
    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)

    if not input_dir_p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir_p}")

    ensure_dir(output_dir_p)

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(output_dir_p):
        existing = glob_images(output_dir_p)
        if existing:
            logger.info(
                f"Found {len(existing)} preprocessed images in {output_dir_p}. "
                "Skipping. Use --force_recompute to re-run."
            )
            return

    image_paths = glob_images(input_dir_p)
    if not image_paths:
        logger.error(f"No images in {input_dir_p}")
        raise SystemExit(1)

    logger.info(f"Preprocessing {len(image_paths)} images with {num_workers} workers...")

    task_args = [
        (p, output_dir_p, target_width, target_height, equalize) for p in image_paths
    ]

    with phase_timer(logger, "Image Preprocessing"):
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process_one, a) for a in task_args]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                pass

    logger.info(f"Preprocessed {len(image_paths)} images → {output_dir_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images (resize, optional CLAHE).")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--equalize", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    preprocess_images(
        args.input, args.output, args.width, args.height,
        args.equalize, args.num_workers, args.force_recompute,
    )
