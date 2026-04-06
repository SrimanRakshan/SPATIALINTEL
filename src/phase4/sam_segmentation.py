"""
Phase 4 — SAM Instance Segmentation.

Execution guarantees:
  [IDEMPOTENCY]  Skips if scene_graph_sam.json exists unless --force_recompute.
  [OBSERVABILITY] Structured logger + phase_timer.
  [RESOURCE]     torch.cuda.empty_cache() after inference; file handles via ctx mgr.
  [SAFETY]       Frames with no detections written as empty lists (not skipped).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir, load_json, save_json
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists

logger = get_logger("phase4.sam_segmentation")

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    logger.error("segment_anything not installed.")
    logger.error("Install: pip install git+https://github.com/facebookresearch/segment-anything.git")
    raise SystemExit(1)

from .utils import check_sam_weights, overlay_masks


def run_instance_segmentation(
    images_dir: str,
    scene_graph_path: str,
    output_dir: str,
    model_type: str = "vit_b",
    force_recompute: bool = False,
) -> None:
    """
    Run SAM conditioned on YOLOv8 bounding boxes.
    Public signature UNCHANGED.
    Output schema of scene_graph_sam.json is UNCHANGED.
    """
    images_dir_p = Path(images_dir)
    scene_graph_path_p = Path(scene_graph_path)
    output_dir_p = Path(output_dir)
    out_json = output_dir_p / "scene_graph_sam.json"

    # [IDEMPOTENCY]
    if not force_recompute and check_output_exists(out_json):
        logger.info(f"Output already exists: {out_json}. Skipping. Use --force_recompute to override.")
        return

    if not images_dir_p.exists() or not scene_graph_path_p.exists():
        logger.error(f"Missing inputs: {images_dir_p} or {scene_graph_path_p}")
        logger.error("Hint: Run Phase 4 semantic_mapping.py before sam_segmentation.py.")
        raise SystemExit(1)

    ensure_dir(output_dir_p)
    masks_dir = ensure_dir(output_dir_p / "masks")
    vis_dir = ensure_dir(output_dir_p / "visualizations")

    weights_path = check_sam_weights(model_type, str(output_dir_p / "weights"))

    logger.info(f"Loading SAM ({model_type}) from {weights_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"SAM device: {device}")
    sam = sam_model_registry[model_type](checkpoint=weights_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    scene_graph_2d = load_json(scene_graph_path_p)
    frames_data: dict = scene_graph_2d.get("frames", {})
    enriched_scene_graph: dict = {"frames": {}}

    logger.info(f"Running instance segmentation on {len(frames_data)} frames...")

    from tqdm import tqdm

    with phase_timer(logger, "SAM Instance Segmentation"):
        for frame_name, detections in tqdm(frames_data.items(), desc="Segmenting frames"):
            img_path = images_dir_p / frame_name
            if not img_path.exists() or not detections:
                enriched_scene_graph["frames"][frame_name] = []
                continue

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                enriched_scene_graph["frames"][frame_name] = []
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            boxes_np = np.array([det["bbox"] for det in detections], dtype=np.float32)
            boxes_torch = torch.from_numpy(boxes_np).to(device)
            boxes_transformed = predictor.transform.apply_boxes_torch(
                boxes_torch, image_rgb.shape[:2]
            )

            # [BREAKING RISK - MINOR] predict_torch returns (N,1,H,W); iterated below.
            masks_batch, scores_batch, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_transformed,
                multimask_output=False,
            )

            frame_results = []
            frame_masks: List[Tuple[np.ndarray, str]] = []

            for idx, (det, mask_1hw, score_1) in enumerate(
                zip(detections, masks_batch, scores_batch)
            ):
                best_mask: np.ndarray = mask_1hw[0].cpu().numpy()
                score: float = float(score_1[0].cpu().numpy())
                mask_filename = f"{img_path.stem}_obj{idx:03d}_{det['class_name']}.npz"
                mask_filepath = masks_dir / mask_filename
                np.savez_compressed(mask_filepath, mask=best_mask)

                det_enriched = det.copy()
                det_enriched["mask_path"] = str(mask_filepath.relative_to(output_dir_p))
                det_enriched["mask_confidence"] = score
                frame_results.append(det_enriched)
                frame_masks.append((best_mask, det["class_name"]))

            enriched_scene_graph["frames"][frame_name] = frame_results

            if frame_masks:
                vis_img = overlay_masks(image_bgr, frame_masks)
                cv2.imwrite(str(vis_dir / frame_name), vis_img)

    # [RESOURCE] Release VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")

    save_json(enriched_scene_graph, out_json)
    logger.info(f"Instance Segmentation complete. Masks → {masks_dir}")
    logger.info(f"Enriched scene graph → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM Instance Segmentation on YOLO detections.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--scene_graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    run_instance_segmentation(args.images, args.scene_graph, args.output, args.model, args.force_recompute)
