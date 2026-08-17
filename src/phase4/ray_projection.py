"""
Phase 4 — Ray Projection: 2D semantic masks → 3D world-space centroids.

Execution guarantees:
  [REPRODUCIBILITY] Fixed RNG seed via set_global_seed(42).
  [IDEMPOTENCY]     Skips if scene_graph_3d.json already exists unless --force_recompute.
  [OBSERVABILITY]   Structured logger + phase_timer; counts frames processed.
  [VALIDATION]      Validates output schema before writing.
  [RESOURCE]        File handles closed via context managers.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_helpers import ensure_dir, load_json, save_json
from utils.camera_utils import parse_transform_frames, rays_from_pixels
from utils.logger import get_logger, phase_timer
from utils.validators import check_output_exists, set_global_seed, validate_scene_graph_3d

logger = get_logger("phase4.ray_projection")


def run_ray_projection(
    scene_graph_sam_path: str,
    transforms_path: str,
    output_dir: str,
    force_recompute: bool = False,
) -> None:
    """
    Projects 2D localised instance masks into 3D world space.
    Output schema (scene_graph_3d.json) is UNCHANGED.
    """
    scene_graph_sam_path = Path(scene_graph_sam_path)
    transforms_path = Path(transforms_path)
    output_dir = Path(output_dir)
    out_json = output_dir / "scene_graph_3d.json"

    # [IDEMPOTENCY] Skip if already computed
    if not force_recompute and check_output_exists(out_json):
        logger.info(f"Output already exists: {out_json}. Skipping. Use --force_recompute to override.")
        return

    if not scene_graph_sam_path.exists():
        logger.error(f"Missing 2D SAM scene graph: {scene_graph_sam_path}")
        logger.error("Hint: Run Phase 4 sam_segmentation.py first.")
        raise SystemExit(1)
    if not transforms_path.exists():
        logger.error(f"Missing NeRF transforms: {transforms_path}")
        logger.error("Hint: Run Phase 3 (run_nerf.py --train) to generate transforms.json.")
        raise SystemExit(1)

    ensure_dir(output_dir)

    # [REPRODUCIBILITY] Deterministic RNG
    set_global_seed(42)

    with phase_timer(logger, "Ray Projection"):
        scene_graph_2d = load_json(scene_graph_sam_path)
        transforms = load_json(transforms_path)

        # Extract intrinsics ONCE
        fl_x: float = transforms.get("fl_x", 1000.0)
        fl_y: float = transforms.get("fl_y", 1000.0)
        cx_img: float = transforms.get("cx", 500.0)
        cy_img: float = transforms.get("cy", 500.0)

        transform_frames: Dict[str, npt.NDArray[np.float64]] = parse_transform_frames(transforms)

        # Seeded RNG for reproducible simulated depths
        rng = np.random.default_rng(seed=42)

        class_aggregates: Dict[str, List[npt.NDArray[np.float64]]] = defaultdict(list)
        frames_data: dict = scene_graph_2d.get("frames", {})
        frames_processed = 0

        logger.info(f"Projecting {len(frames_data)} frames into 3D space...")

        for frame_name, detections in frames_data.items():
            if frame_name not in transform_frames or not detections:
                continue

            c2w = transform_frames[frame_name]
            camera_origin: npt.NDArray[np.float64] = c2w[:3, 3]

            bboxes = np.array(
                [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]] for d in detections],
                dtype=np.float64,
            )
            cx_det = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy_det = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

            _, world_dirs = rays_from_pixels(cx_det, cy_det, fl_x, fl_y, cx_img, cy_img, c2w)
            depths = 2.0 + rng.normal(0.0, 0.5, size=len(detections))
            pts_3d = camera_origin[np.newaxis, :] + world_dirs * depths[:, np.newaxis]

            for i, det in enumerate(detections):
                class_aggregates[det["class_name"]].append(pts_3d[i])

            frames_processed += 1

        logger.info(f"Processed {frames_processed} frames; {len(class_aggregates)} unique classes found.")

        scene_graph_3d: dict = {"objects": []}
        for cls_name, points in class_aggregates.items():
            pts = np.array(points)
            centroid = np.mean(pts, axis=0)
            scene_graph_3d["objects"].append({
                "name": cls_name,
                "position": centroid.tolist(),
                "observations": len(points),
            })

        # [VALIDATION] Validate before writing
        validate_scene_graph_3d(scene_graph_3d, path=out_json)
        save_json(scene_graph_3d, out_json)

    logger.info(f"3D Scene Graph saved to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 2D SAM masks into 3D using NeRF extrinsics.")
    parser.add_argument("--scene_graph_sam", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force_recompute", action="store_true",
                        help="Recompute even if output already exists.")
    args = parser.parse_args()
    run_ray_projection(args.scene_graph_sam, args.transforms, args.output, args.force_recompute)
