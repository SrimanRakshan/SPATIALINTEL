

# 🧠 Spatial Intelligence from Monocular Video

## Neural Radiance Field Reconstruction with Geometric and Semantic Reasoning

---

# 0. PROJECT OBJECTIVE

Build a complete pipeline that:

1. Takes a monocular smartphone video as input
2. Reconstructs a 3D scene using Structure-from-Motion
3. Trains a Neural Radiance Field (NeRF)
4. Enables interactive novel view synthesis
5. Adds semantic scene understanding
6. Enables LLM-based spatial reasoning
7. Performs quantitative evaluation suitable for Q1 publication

---

# 1. SYSTEM OVERVIEW

Pipeline:

```
Monocular Video
    ↓
Frame Extraction
    ↓
Image Preprocessing
    ↓
Structure-from-Motion (COLMAP)
    ↓
Sparse 3D Reconstruction
    ↓
NeRF Dataset Conversion
    ↓
NeRF Training
    ↓
Interactive 3D Visualization
    ↓
Semantic Mapping
    ↓
LLM-based Spatial Reasoning
    ↓
Evaluation & Benchmarking
```

---

# 2. PROJECT STRUCTURE (MANDATORY)

```
SPATIALINTEL/
│
├── data/
│   ├── raw_videos/
│   ├── scenes/
│   │   └── scene_001/
│   │       ├── images/
│   │       ├── colmap/
│   │       └── nerfstudio/
│
├── src/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── evaluation/
│
├── outputs/
├── notebooks/
├── configs/
└── MASTER_ROADMAP.md
```

---

# 3. ENVIRONMENT SETUP (REPRODUCIBLE)

## 3.1 System Requirements

* Python 3.9+
* CUDA-compatible GPU
* COLMAP (v3.8+)
* Nerfstudio
* PyTorch (CUDA-enabled)
* FFmpeg
* OpenCV
* Ultralytics YOLO
* SAM (Segment Anything)
* OpenAI API or local LLM

---

## 3.2 Install Core Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install nerfstudio
pip install ultralytics
pip install open3d
pip install matplotlib
pip install scikit-image
```

---

# 4. PHASE 1 — VIDEO TO MULTI-VIEW DATASET

## Objective

Convert monocular video into structured multi-view dataset.

---

## Step 1: Capture Video

Constraints:

* Static scene
* Smooth motion
* 360° coverage
* Constant lighting
* Avoid motion blur

Store video in:

```
data/raw_videos/scene_001.mp4
```

---

## Step 2: Extract Frames

Algorithm:

* Read video
* Extract frames at fixed interval
* Save as RGB images

Output:

```
data/scenes/scene_001/images/
```

---

## Step 3: Frame Subsampling

Remove:

* Redundant frames
* Blurry frames
* Near-duplicate frames

Criteria:

* SSIM threshold filtering
* Laplacian variance blur detection

---

## Step 4: Image Preprocessing

Apply:

* Resolution normalization
* RGB consistency
* Format unification
* Optional histogram equalization

Output:
Clean multi-view dataset.

---

# 5. PHASE 2 — STRUCTURE FROM MOTION (COLMAP)

## Objective

Recover camera intrinsics, extrinsics, and sparse 3D geometry.

---

## Step 1: Feature Extraction

```bash
colmap feature_extractor \
  --database_path database.db \
  --image_path images/ \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1
```

---

## Step 2: Feature Matching

```bash
colmap exhaustive_matcher \
  --database_path database.db
```

---

## Step 3: Sparse Reconstruction

```bash
colmap mapper \
  --database_path database.db \
  --image_path images/ \
  --output_path sparse/
```

---

## Step 4: Image Undistortion

```bash
colmap image_undistorter \
  --image_path images/ \
  --input_path sparse/0 \
  --output_path undistorted/ \
  --output_type COLMAP
```

---

## Required Metrics

Extract:

* Registered images
* Mean reprojection error
* Mean track length
* Sparse points count

Store metrics in:

```
outputs/geometry_metrics.json
```

---

# 6. PHASE 3 — NERF DATASET & TRAINING

## Objective

Train neural volumetric representation of scene.

---

## Step 1: Convert to Nerfstudio Format

```bash
ns-process-data images \
  --data undistorted/images \
  --output-dir nerfstudio/
```

---

## Step 2: Train Nerfacto

```bash
ns-train nerfacto \
  --data nerfstudio \
  --vis viewer
```

---

## Step 3: Save Checkpoint

Ensure:

```
outputs/nerfstudio/nerfacto/<timestamp>/
```

---

## Step 4: Interactive Visualization

Open:

```
http://localhost:7007
```

Validate:

* Smooth novel views
* No floating artifacts
* Stable geometry

---

# 7. PHASE 4 — SEMANTIC SCENE UNDERSTANDING

## Objective

Add object-level understanding to NeRF.

---

## Step 1: 2D Object Detection

Use YOLOv8:

```python
from ultralytics import YOLO
model = YOLO("yolov8x.pt")
results = model(image)
```

---

## Step 2: Segmentation (SAM)

Use SAM to extract masks.

---

## Step 3: 3D Object Localization

Algorithm:

* Cast rays using camera parameters
* Intersect rays with NeRF density
* Estimate 3D bounding boxes

Output:
3D Scene Graph.

---

# 8. PHASE 5 — LLM-BASED SPATIAL REASONING

## Objective

Enable natural language interaction.

---

## Input

Structured Scene Graph:

```json
{
  "objects": [
    {"name": "chair", "position": [x,y,z]},
    {"name": "table", "position": [x,y,z]}
  ]
}
```

---

## Tasks

* Relative position queries
* Object counting
* Scene description
* Spatial reasoning

Example query:

> “What is to the left of the table?”

---

# 9. EVALUATION (MANDATORY FOR PUBLICATION)

---

## 9.1 Geometric Evaluation

Metrics:

* Mean reprojection error
* Registered image ratio
* Track length

---

## 9.2 NeRF Evaluation

Split data:

* 90% train
* 10% validation

Compute:

* PSNR
* SSIM
* LPIPS

---

## 9.3 Baseline Comparison

Compare against:

* Vanilla NeRF
* Instant-NGP
* Mip-NeRF (optional)

---

## 9.4 Performance Evaluation

Measure:

* Training time
* Rays per second
* GPU memory usage
* Inference FPS

---

# 10. RESEARCH CONTRIBUTION OPPORTUNITIES

To achieve Q1-level novelty:

* Joint optimization of camera pose during NeRF training
* Depth-regularized NeRF
* Semantic-aware NeRF
* Real-time streaming NeRF from video
* Hybrid sparse + neural reconstruction

---

# 11. FINAL DELIVERABLES

* Trained NeRF model
* Quantitative evaluation metrics
* Scene graph
* LLM-based reasoning demo
* Comparison table vs baselines
* Ablation study
* Publication-ready figures

---

# 12. SUCCESS CRITERIA

The project is considered complete when:

✔ End-to-end pipeline runs from video → 3D
✔ NeRF produces high-quality novel views
✔ PSNR > baseline NeRF
✔ Semantic objects mapped in 3D
✔ LLM answers spatial queries correctly
✔ All metrics reproducible

---

# 13. LONG-TERM EXTENSIONS

* AR integration
* Real-time capture + reconstruction
* Multi-scene generalization
* Mobile deployment
* Edge optimization

---

# 14. FINAL STATEMENT

This document defines a complete reproducible pipeline for:

Monocular Video → Geometric Reconstruction → Neural Radiance Field → Semantic Mapping → Spatial Intelligence → Quantitative Evaluation

It is structured to support:

* Academic research
* Reproducibility
* Extension toward publication
* AI-agent-driven execution

---

