import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def preprocess_images(input_dir: str, output_dir: str, target_width: int = -1, target_height: int = -1, equalize: bool = False):
    """
    Preprocess images: resolution normalization and optional histogram equalization.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    
    for ipath in tqdm(image_paths, desc="Preprocessing images"):
        image = cv2.imread(str(ipath))
        if image is None:
            continue
            
        # Resize
        if target_width > 0 and target_height > 0:
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        elif target_width > 0:
            # scale based on width
            h, w = image.shape[:2]
            scale = target_width / float(w)
            image = cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            
        # Histogram Equalization (CLAHE on L channel of LAB color space)
        if equalize:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl,a,b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
        # Save as JPG for consistency
        out_path = output_dir / f"{ipath.stem}.jpg"
        cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
    print(f"Preprocessed {len(image_paths)} images. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images (resize, standardize formats, optional EQ).")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save preprocessed frames")
    parser.add_argument("--width", type=int, default=-1, help="Target width. If only width is set, aspect ratio is maintained.")
    parser.add_argument("--height", type=int, default=-1, help="Target height.")
    parser.add_argument("--equalize", action="store_true", help="Apply CLAHE histogram equalization.")
    
    args = parser.parse_args()
    
    preprocess_images(args.input, args.output, args.width, args.height, args.equalize)
