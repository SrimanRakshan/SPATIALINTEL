"""
SPATIALINTEL – COLMAP SfM Pipeline
----------------------------------

Step 2 of the pipeline:
    - Take extracted frames for a scene:
          data/scenes/<scene_name>/images/*.png
    - Run COLMAP to:
          * Extract features
          * Match features
          * Build sparse reconstruction (SfM)
          * Undistort images + write cameras/images/points3D files

Outputs (under scene_root/colmap):
    data/scenes/<scene_name>/colmap/
      database.db
      sparse/0/
        cameras.bin / images.bin / points3D.bin
      undistorted/
        images/         # undistorted images
        cameras.txt
        images.txt
        points3D.txt

These undistorted outputs will be used next by NeRF (Nerfstudio / Instant-NGP).
"""

from __future__ import annotations
import os
import subprocess
import argparse


def run_cmd(cmd, cwd=None):
    print(f"\n[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def colmap_feature_extractor(
    database_path: str,
    image_path: str,
    camera_model: str = "PINHOLE",
    single_camera: bool = True,
):
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.camera_model", camera_model,
    ]
    if single_camera:
        cmd += ["--ImageReader.single_camera", "1"]

    run_cmd(cmd)


def colmap_matcher(database_path: str, mode: str = "sequential"):
    """
    Run feature matching.

    mode:
      - 'sequential' (good for video-style input)
      - 'exhaustive' (all pairs, slower)
    """
    if mode == "sequential":
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", database_path,
        ]
    elif mode == "exhaustive":
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
        ]
    else:
        raise ValueError(f"Unknown matcher mode: {mode}")

    run_cmd(cmd)


def colmap_mapper(database_path: str, image_path: str, output_path: str):
    cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", output_path,
    ]
    run_cmd(cmd)


def colmap_image_undistorter(
    image_path: str,
    input_path: str,
    output_path: str,
    output_type: str = "COLMAP",
):
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", output_type,
    ]
    run_cmd(cmd)


def run_colmap_pipeline(
    scene_root: str,
    images_subdir: str = "images",
    colmap_subdir: str = "colmap",
    matcher_mode: str = "sequential",
):
    """
    Run the full COLMAP SfM pipeline for a given scene.

    scene_root example:
        data/scenes/scene1
    """
    scene_root = os.path.abspath(scene_root)
    images_path = os.path.join(scene_root, images_subdir)

    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    colmap_root = os.path.join(scene_root, colmap_subdir)
    sparse_dir = os.path.join(colmap_root, "sparse")
    undistorted_dir = os.path.join(colmap_root, "undistorted")

    os.makedirs(colmap_root, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(undistorted_dir, exist_ok=True)

    database_path = os.path.join(colmap_root, "database.db")

    print(f"[INFO] Scene root: {scene_root}")
    print(f"[INFO] Images path: {images_path}")
    print(f"[INFO] COLMAP root: {colmap_root}")

    # 1) Feature extraction
    print("\n[STEP 1] Feature extraction")
    colmap_feature_extractor(
        database_path=database_path,
        image_path=images_path,
        camera_model="PINHOLE",
        single_camera=True,
    )

    # 2) Feature matching
    print("\n[STEP 2] Feature matching")
    colmap_matcher(database_path=database_path, mode=matcher_mode)

    # 3) Sparse reconstruction (mapper)
    print("\n[STEP 3] Sparse reconstruction (mapper)")
    colmap_mapper(
        database_path=database_path,
        image_path=images_path,
        output_path=sparse_dir,
    )

    # We assume model 0 is the main one
    sparse_model_path = os.path.join(sparse_dir, "0")
    if not os.path.isdir(sparse_model_path):
        raise RuntimeError(
            f"No sparse model found at {sparse_model_path}. "
            "COLMAP may have failed to reconstruct."
        )

    # 4) Undistort images for NeRF
    print("\n[STEP 4] Undistorting images for NeRF")
    colmap_image_undistorter(
        image_path=images_path,
        input_path=sparse_model_path,
        output_path=undistorted_dir,
        output_type="COLMAP",  # produces cameras.txt, images.txt, points3D.txt
    )

    print("\n[DONE] COLMAP pipeline completed.")
    print(f"[INFO] Sparse model directory: {sparse_model_path}")
    print(f"[INFO] Undistorted directory: {undistorted_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM pipeline on extracted frames for a scene."
    )
    parser.add_argument(
        "--scene_root",
        type=str,
        required=True,
        help="Path to scene root directory (e.g., data/scenes/scene1)",
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="images",
        help="Subdirectory under scene_root with input images (default: images)",
    )
    parser.add_argument(
        "--colmap_subdir",
        type=str,
        default="colmap",
        help="Subdirectory under scene_root to store COLMAP outputs (default: colmap)",
    )
    parser.add_argument(
        "--matcher_mode",
        type=str,
        default="sequential",
        choices=["sequential", "exhaustive"],
        help="Matcher type: sequential (video) or exhaustive (all pairs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_colmap_pipeline(
        scene_root=args.scene_root,
        images_subdir=args.images_subdir,
        colmap_subdir=args.colmap_subdir,
        matcher_mode=args.matcher_mode,
    )


if __name__ == "__main__":
    main()
