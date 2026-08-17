"""
Phase 4 — Semantic Mapping: YOLOv8 object detection → 2D scene graph.

Execution guarantees:
  [REPRODUCIBILITY] Deterministic detection ordering (sorted image paths).
  [IDEMPOTENCY]     Skips if scene_graph_2d.json exists unless --force_recompute.
  [OBSERVABILITY]   Structured logger + phase_timer; batch progress logged.
  [RESOURCE]        torch.cuda.empty_cache() after inference loop.
  [SAFETY]          Validates output before writing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir, glob_images, save_json
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists, validate_scene_graph_2d

logger = get_logger("phase4.semantic_mapping")


def run_semantic_mapping(
    images_dir: str,
    output_dir: str,
    batch_size: int = 8,
    yolo_model: str = "yolov8n.pt",
    confidence: float = 0.25,
    force_recompute: bool = False,
) -> None:
    """
    Run YOLOv8 object detection on extracted frames to build a 2D scene graph.
    Public signature UNCHANGED (new kwargs are optional with defaults).
    Output file scene_graph_2d.json schema is UNCHANGED.
    """
    images_dir_p = Path(images_dir)
    output_dir_p = Path(output_dir)
    out_json = output_dir_p / "scene_graph_2d.json"

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(out_json):
        logger.info(f"Output already exists: {out_json}. Skipping. Use --force_recompute to override.")
        return

    if not images_dir_p.exists():
        logger.error(f"Images directory not found: {images_dir_p}")
        logger.error("Hint: Run Phase 1 (extract_frames.py + preprocess.py) first.")
        raise SystemExit(1)

    ensure_dir(output_dir_p)
    vis_dir = ensure_dir(output_dir_p / "visualizations")

    logger.info(f"Loading YOLOv8 model ({yolo_model})...")
    model = YOLO(yolo_model)

    use_fp16: bool = torch.cuda.is_available()
    logger.info(f"CUDA available: {torch.cuda.is_available()} | fp16: {use_fp16}")

    image_paths = glob_images(images_dir_p)
    if not image_paths:
        logger.error(f"No images found in {images_dir_p}")
        raise SystemExit(1)

    logger.info(f"Running semantic mapping on {len(image_paths)} images (batch_size={batch_size})...")

    scene_graph_2d: dict = {"frames": {}}

    with phase_timer(logger, "YOLOv8 Semantic Mapping"):
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Detecting objects"):
            batch_paths: List[Path] = image_paths[batch_start : batch_start + batch_size]
            loaded = [(p, cv2.imread(str(p))) for p in batch_paths]
            valid = [(p, img) for p, img in loaded if img is not None]
            if not valid:
                continue

            valid_paths, valid_imgs = zip(*valid)
            results = model(list(valid_imgs), verbose=False, half=use_fp16, conf=confidence)

            for ipath, result in zip(valid_paths, results):
                detections = []
                for box in result.boxes:
                    b = box.xyxy[0].cpu().numpy().tolist()
                    c = int(box.cls.cpu().numpy()[0])
                    conf_val = float(box.conf.cpu().numpy()[0])
                    name = model.names[c]
                    detections.append({
                        "class_id": c,
                        "class_name": name,
                        "confidence": conf_val,
                        "bbox": [float(x) for x in b],
                    })
                scene_graph_2d["frames"][ipath.name] = detections
                res_img = result.plot()
                cv2.imwrite(str(vis_dir / ipath.name), res_img)

    # [RESOURCE] Release GPU memory after inference loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")

    # [VALIDATION] Validate before writing
    validate_scene_graph_2d(scene_graph_2d, path=out_json)
    save_json(scene_graph_2d, out_json)

    total_detections = sum(len(v) for v in scene_graph_2d["frames"].values())
    logger.info(f"Semantic Mapping complete. {total_detections} detections across {len(image_paths)} frames.")
    logger.info(f"Saved scene_graph_2d.json to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D Semantic Mapping using YOLOv8.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    run_semantic_mapping(args.images, args.output, args.batch_size, args.model, args.confidence, args.force_recompute)
