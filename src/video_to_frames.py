"""
SPATIALINTEL – Advanced Video Preprocessing
-------------------------------------------

Step 1 of full pipeline:
  - Take a smartphone video as input.
  - Extract frames at a fixed interval OR up to a max frame count.
  - (Optionally) center-crop and resize frames.
  - Save frames in a clean scene structure:
        data/scenes/<scene_name>/images/*.png
  - Save metadata (fps, total frames, timestamps) to metadata.json

This output will be used by:
  - COLMAP for Structure-from-Motion
  - NeRF (Nerfstudio / Instant-NGP) for 3D reconstruction
"""

from __future__ import annotations
import cv2
import os
import json
import argparse
from typing import Optional, Tuple, Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def center_crop(frame, target_aspect: float):
    """
    Center-crop the frame to the desired aspect ratio.

    Args:
        frame: HxWxC (BGR)
        target_aspect: width / height

    Returns:
        Cropped frame (BGR)
    """
    h, w = frame.shape[:2]
    current_aspect = w / h

    if abs(current_aspect - target_aspect) < 1e-3:
        return frame  # already close

    if current_aspect > target_aspect:
        # too wide -> crop horizontally
        new_w = int(h * target_aspect)
        x1 = (w - new_w) // 2
        x2 = x1 + new_w
        return frame[:, x1:x2]
    else:
        # too tall -> crop vertically
        new_h = int(w / target_aspect)
        y1 = (h - new_h) // 2
        y2 = y1 + new_h
        return frame[y1:y2, :]


def extract_frames(
    video_path: str,
    scene_name: str,
    output_root: str = "data/scenes",
    frame_step: int = 5,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    crop_aspect: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract frames from video and save metadata.

    Args:
        video_path: path to input video file
        scene_name: name for the scene folder (e.g. "scene1")
        output_root: root folder to store scenes
        frame_step: save every N-th frame
        max_frames: optional cap on total saved frames
        resize: optional (width, height) for resizing
        crop_aspect: optional aspect ratio (w/h) for center cropping

    Returns:
        metadata dict with basic info and list of saved frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Scene folder layout
    scene_root = os.path.join(output_root, scene_name)
    images_dir = os.path.join(scene_root, "images")
    raw_dir = os.path.join(scene_root, "raw")
    ensure_dir(images_dir)
    ensure_dir(raw_dir)

    # Optionally copy video to scene raw/ (you can also skip this and keep path)
    # For now, we just store relative path in metadata.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Scene: {scene_name}")
    print(f"[INFO] Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"[INFO] Extracting every {frame_step} frame(s)")
    if resize is not None:
        print(f"[INFO] Resizing to: {resize[0]}x{resize[1]}")
    if crop_aspect is not None:
        print(f"[INFO] Center-cropping to aspect ratio: {crop_aspect:.3f}")
    if max_frames is not None:
        print(f"[INFO] Max frames to save: {max_frames}")

    saved_frames_info = []

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Compute timestamp in seconds
            timestamp_sec = frame_idx / fps

            # Optional crop
            if crop_aspect is not None:
                frame = center_crop(frame, crop_aspect)

            # Optional resize
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            filename = f"frame_{saved_idx:04d}.png"
            out_path = os.path.join(images_dir, filename)
            cv2.imwrite(out_path, frame)

            saved_frames_info.append({
                "index": saved_idx,
                "orig_frame_index": frame_idx,
                "timestamp_sec": float(timestamp_sec),
                "filename": filename,
            })

            if saved_idx % 10 == 0:
                print(f"[INFO] Saved frame {saved_idx} -> {out_path}")

            saved_idx += 1

            if max_frames is not None and saved_idx >= max_frames:
                print("[INFO] Reached max_frames limit, stopping extraction.")
                break

        frame_idx += 1

    cap.release()
    print(f"[DONE] Extracted {saved_idx} frames to: {images_dir}")

    metadata = {
        "scene_name": scene_name,
        "video_path": os.path.abspath(video_path),
        "scene_root": os.path.abspath(scene_root),
        "images_dir": os.path.abspath(images_dir),
        "raw_dir": os.path.abspath(raw_dir),
        "total_video_frames": total_frames,
        "fps": float(fps),
        "frame_step": frame_step,
        "max_frames": max_frames,
        "resize": {"width": resize[0], "height": resize[1]} if resize else None,
        "crop_aspect": crop_aspect,
        "saved_frames": saved_frames_info,
    }

    # Save metadata
    metadata_path = os.path.join(scene_root, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to: {metadata_path}")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from smartphone video for COLMAP + NeRF"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (e.g., data/raw/scene1/scene1.mp4)",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Name of the scene (e.g., scene1, desk1)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/scenes",
        help="Root directory to store all scenes (default: data/scenes)",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Save every N-th frame (default: 5)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to save",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional width to resize frames to",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Optional height to resize frames to",
    )
    parser.add_argument(
        "--crop_aspect",
        type=float,
        default=None,
        help="Optional aspect ratio (w/h) for center cropping (e.g. 16/9 = 1.777)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    resize = None
    if args.resize_width is not None and args.resize_height is not None:
        resize = (args.resize_width, args.resize_height)

    extract_frames(
        video_path=args.video,
        scene_name=args.scene_name,
        output_root=args.output_root,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        resize=resize,
        crop_aspect=args.crop_aspect,
    )


if __name__ == "__main__":
    main()
