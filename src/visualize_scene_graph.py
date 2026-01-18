"""
SPATIALINTEL – Video → Frames Extraction (Demo Version)
-------------------------------------------------------

This script:
  - Takes your smartphone video:
        C:\\Users\\Sriman Rakshan N\\Downloads\\demo.mp4
  - Extracts frames every N frames
  - Optionally center-crops + resizes them
  - Saves them under:
        data/scenes/scene1/images/
  - Stores metadata in:
        data/scenes/scene1/metadata.json

These frames and metadata will be used next by:
  - COLMAP (for camera poses + sparse reconstruction)
  - NeRF (for 3D scene reconstruction)
"""

from __future__ import annotations
import cv2
import os
import json
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


def main():
    # 👇 Your exact video path (Windows, use raw string to avoid backslash issues)
    video_path = r"C:\Users\Sriman Rakshan N\Downloads\demo.mp4"

    # Scene configuration
    scene_name = "scene1"
    output_root = "data/scenes"

    # Frame extraction settings
    frame_step = 5         # take every 5th frame
    max_frames = 200       # stop after saving 200 frames (set None for no limit)
    resize = (800, 450)    # resize to 800x450 (good for NeRF & COLMAP)
    crop_aspect = 16 / 9   # center-crop to 16:9 aspect ratio

    extract_frames(
        video_path=video_path,
        scene_name=scene_name,
        output_root=output_root,
        frame_step=frame_step,
        max_frames=max_frames,
        resize=resize,
        crop_aspect=crop_aspect,
    )


if __name__ == "__main__":
    main()
