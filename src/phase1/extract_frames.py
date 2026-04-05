import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path: str, output_dir: str, interval: int = 5):
    """
    Extract frames from a monocular video at a fixed interval.
    Args:
        video_path: Path to the input video.
        output_dir: Directory to save the extracted frames.
        interval: Extract 1 frame every `interval` frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path.name}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.2f}")
    
    frame_idx = 0
    saved_idx = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % interval == 0:
                out_path = output_dir / f"frame_{saved_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_idx += 1
                
            frame_idx += 1
            pbar.update(1)
            
    cap.release()
    print(f"Extracted {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--interval", type=int, default=5, help="Extract 1 frame every interval frames")
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.interval)
