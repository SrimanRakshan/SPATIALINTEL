import cv2
import os
import argparse
import shutil
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_ssim(img1, img2, threshold=0.95):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

def subsample_frames(input_dir: str, output_dir: str, blur_threshold: float = 100.0, ssim_threshold: float = 0.95):
    """
    Remove blurry and redundant (near-duplicate) frames.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_paths:
        print("No images found in input directory.")
        return
        
    print(f"Found {len(image_paths)} images.")
    
    prev_image = None
    saved_idx = 0
    blurry_count = 0
    duplicate_count = 0
    
    for ipath in tqdm(image_paths, desc="Subsampling frames"):
        image = cv2.imread(str(ipath))
        if image is None:
            continue
            
        # 1. Blur detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < blur_threshold:
            blurry_count += 1
            continue
            
        # 2. Redundancy check (SSIM)
        if prev_image is not None:
            if check_ssim(prev_image, image, threshold=ssim_threshold):
                duplicate_count += 1
                continue
                
        # Keep frame
        out_path = output_dir / f"frame_{saved_idx:05d}.jpg"
        cv2.imwrite(str(out_path), image)
        prev_image = image
        saved_idx += 1
        
    print(f"Subsampling complete.")
    print(f"Removed {blurry_count} blurry frames.")
    print(f"Removed {duplicate_count} near-duplicate frames.")
    print(f"Kept {saved_idx} frames. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample frames by removing blurry and redundant images.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory for subsampled frames")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Variance of Laplacian threshold for blur. Lower = more blurry allowed.")
    parser.add_argument("--ssim_threshold", type=float, default=0.90, help="SSIM threshold for redundancy. Higher = more strict duplicate check.")
    args = parser.parse_args()
    
    subsample_frames(args.input, args.output, args.blur_threshold, args.ssim_threshold)
