"""
Phase 1 — Frame Extraction.

Execution guarantees:
  [IDEMPOTENCY]  Skips if output_dir already has frames unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer.
  [RESOURCE]     VideoCapture explicitly released in finally block.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("phase1.extract_frames")


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 5,
    force_recompute: bool = False,
) -> None:
    """
    Extract frames from a monocular video at a fixed interval.
    Public signature UNCHANGED.
    """
    video_path_p = Path(video_path)
    output_dir_p = Path(output_dir)

    if not video_path_p.exists():
        raise FileNotFoundError(f"Video file not found: {video_path_p}")

    ensure_dir(output_dir_p)

    # [IDEMPOTENCY] Skip if frames already extracted
    if not force_recompute and check_output_exists(output_dir_p):
        existing = list(output_dir_p.glob("frame_*.jpg"))
        if existing:
            logger.info(
                f"Found {len(existing)} frames in {output_dir_p}. Skipping. "
                "Use --force_recompute to re-extract."
            )
            return

    cap = cv2.VideoCapture(str(video_path_p))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video: {video_path_p.name} | Total: {total_frames} frames | FPS: {fps:.2f}")
        logger.info(f"Extracting every {interval} frames → ~{total_frames // interval} output frames")

        target_indices = list(range(0, total_frames, interval))
        saved_idx = 0

        with phase_timer(logger, "Frame Extraction"):
            for frame_idx in tqdm(target_indices, desc="Extracting frames"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                out_path = output_dir_p / f"frame_{saved_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_idx += 1

        logger.info(f"Extracted {saved_idx} frames → {output_dir_p}")
    finally:
        cap.release()  # [RESOURCE] always release, even on exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval, args.force_recompute)
