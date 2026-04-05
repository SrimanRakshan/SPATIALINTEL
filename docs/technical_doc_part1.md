# Spatial Intelligence: Technical Documentation
## An End-to-End Pipeline from Monocular Video to LLM-Reasoned 3D Scene Understanding

*Research-paper-quality technical reference for the Spatialintel_new codebase.*

---

## 1. Abstract

This document describes **Spatial Intelligence**, a nine-phase, end-to-end computational pipeline that transforms a single monocular RGB video into a semantically annotated, interactively queryable 3D scene representation. The system fuses classical photogrammetry, neural volumetric rendering, vision foundation models, and large language model (LLM) spatial reasoning into a unified inference-time architecture.

The pipeline begins with temporal sub-sampling and quality filtering of raw video frames, applies Structure-from-Motion (SfM) via COLMAP to recover sparse 3D geometry and per-frame camera extrinsics, and then trains a Neural Radiance Field (NeRF)—either `nerfacto` (implicit MLP) or `splatfacto` (3D Gaussian Splatting)—over the recovered camera manifold to produce a dense, photorealistic volumetric scene representation. Concurrently, a two-stage semantic detector chain—YOLOv8 for 2D bounding-box detection followed by the Segment Anything Model (SAM ViT-B/L/H) for pixel-precise instance segmentation—annotates every frame with per-object masks. A novel **ray-projection aggregation** module then lifts these 2D detections into 3D world coordinates by casting pinhole-camera rays through bounding-box centroids, rotating them into world frame via COLMAP extrinsic matrices, and accumulating per-class 3D point estimates across all frames to compute robust geometric centroids. The resulting `scene_graph_3d.json` is consumed by a provider-agnostic LLM spatial reasoning agent (supporting Google Gemini, OpenAI GPT-4o, and HuggingFace Mistral-7B) that answers free-form geometric queries over the reconstructed scene purely from serialized centroid coordinates.

Downstream phases produce an Open3D semantic point cloud, a NetworkX proximity graph, and a Streamlit dashboard unifying all outputs—interactive Plotly 3D viewer, topology map, live LLM chat, and an embedded `ns-viewer` subprocess—into a single presentation layer. The system also resolves several non-trivial engineering constraints: Windows-specific `WinError 87` subprocess bugs in `mediapy`, PyTorch 2.6 `weights_only` deprecation, and WebGL memory limits for large point clouds.

---

## 2. System Architecture

### 2.1 End-to-End Pipeline Overview

The pipeline is organized as a directed acyclic graph (DAG) of sequential phases, each consuming artifacts from its predecessor and producing new artifacts consumed downstream.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     SPATIAL INTELLIGENCE PIPELINE                      │
│                                                                        │
│  [input.mp4]                                                           │
│      │                                                                 │
│      ▼                                                                 │
│  ┌──────────┐   frame_{n}.jpg    ┌────────────┐  frame_{n}.jpg         │
│  │ Phase 1  │ ─────────────────► │  Phase 2   │ (undistorted)          │
│  │ Video →  │                   │  COLMAP    │                         │
│  │ Frames   │                   │  SfM       │                         │
│  └──────────┘                   └─────┬──────┘                         │
│                                        │ sparse/0/ + transforms.json   │
│                                        ▼                               │
│                               ┌────────────────┐                       │
│                               │   Phase 3      │                       │
│                               │  Nerfstudio    │                       │
│                               │ nerfacto /     │                       │
│                               │  splatfacto    │                       │
│                               └──────┬─────────┘                       │
│                                      │ config.yml + checkpoints        │
│              ┌───────────────────────┼────────────────────────┐        │
│              │                       │                        │        │
│              ▼                       ▼                        ▼        │
│   ┌──────────────────┐   ┌──────────────────────┐  ┌───────────────┐  │
│   │    Phase 4a      │   │      Phase 6          │  │   Phase 6b/c │  │
│   │  YOLOv8 +SAM     │   │  ns-eval (PSNR/SSIM/ │  │  ns-export   │  │
│   │  semantic_mapping│   │  LPIPS) + ns-render   │  │  TSDF mesh   │  │
│   │  + ray_project   │   │  (MP4 flythrough)     │  │  (.ply)      │  │
│   └────────┬─────────┘   └──────────────────────┘  └───────────────┘  │
│            │ scene_graph_3d.json                                       │
│   ┌────────┴──────────────────────────────────────┐                    │
│   │      Phase 5: LLM Spatial Reasoning Agent     │                    │
│   │      (Gemini / GPT-4o / HuggingFace)          │                    │
│   └────────┬──────────────────────────────────────┘                    │
│            │                                                            │
│   ┌────────┴──────────┐  ┌────────────────────────┐                    │
│   │     Phase 7       │  │       Phase 8           │                   │
│   │  Open3D Semantic  │  │  NetworkX Topology      │                   │
│   │  PCD spheres      │  │  Proximity Graph        │                   │
│   │  (semantic_pcd    │  │  (topology_graph.png)   │                   │
│   │   .ply)           │  └───────────┬─────────────┘                   │
│   └───────────────────┘              │                                  │
│                          ┌───────────┴──────────────────────────────┐  │
│                          │        Phase 9: Streamlit Dashboard       │  │
│                          │  Tab1: Plotly 3D | Tab2: Topology |       │  │
│                          │  Tab3: LLM Chat  | Tab4: ns-viewer        │  │
│                          └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Central Data Artifacts

| Artifact | Format | Producer | Consumer(s) |
|---|---|---|---|
| `frame_{n}.jpg` | JPEG, BGR | Phase 1 | Phase 2, Phase 4 |
| `database.db` | SQLite (COLMAP) | Phase 2 | Phase 2 (internal) |
| `sparse/0/{cameras,images,points3D}.bin` | COLMAP binary | Phase 2 | Phase 2→undistort |
| `undistorted/` images | JPEG | Phase 2 | Phase 3 |
| `transforms.json` | Nerfstudio JSON | Phase 3 (`ns-process-data`) | Phase 3 train, Phase 4 |
| `config.yml` | YAML (Nerfstudio) | Phase 3 training | Phase 6 eval/render/export |
| `scene_graph_2d.json` | JSON | Phase 4a (YOLO) | Phase 4b (SAM) |
| `scene_graph_sam.json` | JSON + `.npz` mask refs | Phase 4b (SAM) | Phase 4c (ray proj.) |
| **`scene_graph_3d.json`** | JSON | Phase 4c (ray proj.) | Phase 5, 7, 8, 9 |
| `evaluation_metrics.json` | JSON | Phase 6 (ns-eval) | Phase 9 display |
| `tsdf_mesh.ply` | PLY mesh | Phase 6 (ns-export) | Phase 9 download |
| `*_interpolate.mp4` | H.264 MP4 | Phase 6 (ns-render) | Phase 9 display |
| [semantic_pcd.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/semantic_pcd.ply) | PLY point cloud | Phase 7 | Phase 9 (download) |
| [topology_graph.png](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/topology_graph.png) | PNG 300dpi | Phase 8 | Phase 9 Tab 2 |

### 2.3 Separation of Concerns and Implicit Path Contracts

Each source subdirectory (`src/phaseN/`) is nominally a self-contained CLI module accepting file paths as arguments. However, the decoupling breaks at the boundary between Phase 4c and Phase 9. [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) writes its output to `args.output / "scene_graph_3d.json"` — a caller-specified directory. [app.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py) hardcodes its read path as:
```python
SCENE_GRAPH_PATH = Path("data/scenes/scene_001/semantic_3d/scene_graph_3d.json")
```
This means Phase 4c **must** be invoked with `--output data/scenes/scene_001/semantic_3d` or [app.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py) will fail to find the 3D scene graph. This implicit contract is not documented in either script. A similar implicit contract exists for the topology graph ([outputs/topology_graph.png](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/topology_graph.png)), semantic PCD ([outputs/semantic_pcd.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/semantic_pcd.ply)), rendered video ([outputs/scene_001_final_render_interpolate.mp4](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/scene_001_final_render_interpolate.mp4)), and mesh ([outputs/mesh/tsdf_mesh.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/mesh/tsdf_mesh.ply)) — all hardcoded paths in [app.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py) that the CLI scripts must be directed to match.

---

## 3. Phase-by-Phase Technical Breakdown

### Phase 1: Video Ingestion and Frame Dataset Construction

**Objective:** Convert a raw monocular video into a curated, geometrically useful set of individual frames suitable for photogrammetric reconstruction and semantic detection.

**Input Artifacts:** `input.mp4` (any OpenCV-readable video format)

**Core Logic — Three-stage pipeline:**

#### Stage 1a: Temporal Subsampling ([extract_frames.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase1/extract_frames.py))

OpenCV's `VideoCapture` decodes the video stream sequentially. The extractor reads `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` at initialization, then iterates frame-by-frame: for every frame at index `i`, it is written to disk only when `i % interval == 0`. The default interval of 5 yields 1 frame per 5 decoded, i.e., at 30 fps source video this produces 6 images/second. Frames are zero-padded and named `frame_{i:05d}.jpg`.

**Design choice:** Uniform temporal sampling is a deliberate simplification over adaptive keyframe selection. It guarantees predictable output cardinality and avoids scene-dependent keyframe detection overhead. The tradeoff is redundancy in static shots.

#### Stage 1b: Preprocessing ([preprocess.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase1/preprocess.py))

Each extracted frame undergoes optional resolution normalization and histogram equalization:

- **Resize:** If `target_width > 0 and target_height > 0`, both dimensions are explicitly set (`INTER_AREA` interpolation, which is anti-aliasing optimal for downsampling). If only `target_width` is given, aspect ratio is preserved: `scale = target_width / w; new_h = int(h * scale)`.

- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Operates in LAB color space, not BGR. The image is converted to LAB, the L (luminance) channel is extracted, and `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` is applied exclusively to L. This avoids color shift artifacts that would occur if equalization were applied to individual BGR channels. The clip limit of 2.0 prevents over-amplification of noise in low-texture regions. Reconversion to BGR preserves chrominance (A, B channels) unchanged.

**Why LAB for equalization:** CLAHE on the L channel of LAB space gives perceptually uniform brightness enhancement without introducing color artifacts — critical because downstream SfM descriptor computation (SIFT features) is heavily affected by illumination consistency.

#### Stage 1c: Quality Filtering ([subsample.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase1/subsample.py))

Two sequential rejection criteria are applied per frame:

**Blur Detection via Variance of Laplacian:**
```
f(I) = Var(∇²I) = Var(Laplacian(I_gray, CV_64F))
```
The discrete Laplacian operator approximates the second derivative of image intensity. High-frequency detail (sharp edges) produces large second-derivative responses; blurry images suppress high frequencies, yielding near-zero Laplacian responses. The variance of the Laplacian map measures the spread of these responses: a sharp image has high variance (both large positive and negative second-derivative values), a blurry image has near-zero variance throughout. Default threshold: 100.0. Frames with [f(I) < 100.0](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/evaluate_nerf.py#31-68) are discarded.

**Near-Duplicate Rejection via SSIM:**
```
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
```
where `l` is luminance comparison, `c` is contrast comparison, and `s` is structure comparison (cross-covariance normalized by product of standard deviations). The `skimage.metrics.structural_similarity` implementation uses the default equal exponents (α=β=γ=1) and computes SSIM over local windows. Grayscale conversion precedes comparison. If `SSIM(prev_frame, current_frame) > 0.95`, the current frame is discarded as a near-duplicate. This guards against static shots or slow camera motion producing redundant viewpoints that would bloat the COLMAP database without improving pose diversity.

**Output Artifacts:** `data/scenes/scene_001/images_preprocessed/frame_{n:05d}.jpg`

**Dependencies & Assumptions:** Input video must be readable by OpenCV. The blur threshold (100.0) and SSIM threshold (0.90–0.95 — note discrepancy: argparse default is 0.90 but function signature default is 0.95) are hyperparameters sensitive to scene content. Indoor, uniformly lit scenes with slow motion may trigger aggressive SSIM rejection.

---

### Phase 2: Structure from Motion via COLMAP

**Objective:** Recover the sparse 3D structure of the scene and the 6-DoF camera pose (position + orientation) for every retained frame. This establishes the metric coordinate frame that all downstream 3D reasoning operates within.

**Input Artifacts:** Preprocessed JPEG frames from Phase 1.

**Core Logic ([run_colmap.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase2/run_colmap.py)):** A Python script wrapping four sequential COLMAP CLI subprocesses via `subprocess.run(cmd, shell=True, check=True)`. Each COLMAP stage is invoked as an OS-level process; `check=True` exits the pipeline immediately on non-zero return code.

#### COLMAP Stage 1: Feature Extraction

```
colmap feature_extractor
  --database_path database.db
  --image_path images_dir
  --ImageReader.camera_model PINHOLE
  --ImageReader.single_camera 1
```

COLMAP extracts SIFT (Scale-Invariant Feature Transform) keypoints and 128-dimensional descriptors from each image. The `PINHOLE` camera model is explicitly specified — this is a 4-parameter model (fx, fy, cx, cy) with no distortion coefficients. `single_camera 1` forces COLMAP to assume all images were captured by a single camera with shared intrinsics — appropriate for video, where focal length and sensor are constant. All keypoints and descriptors are stored in `database.db` (SQLite format).

**Mathematical detail:** SIFT detects features at scale-space extrema of the Difference-of-Gaussian (DoG) pyramid. For each detected keypoint at scale σ, a 128-bin gradient orientation histogram descriptor is computed over a 16×16 pixel neighborhood. SIFT descriptors are robust to rotation, scale, and moderate illumination changes.

#### COLMAP Stage 2: Feature Matching

```
colmap exhaustive_matcher --database_path database.db
```

Exhaustive matching compares every image pair (O(N²) complexity). For each pair, COLMAP applies ratio-test nearest-neighbor matching in 128-D descriptor space, then estimates the fundamental matrix F using RANSAC to reject geometric outliers. Only inlier matches are retained in the database. For N frames, this produces at most N(N-1)/2 candidate pairs with verified correspondences.

**Why exhaustive over sequential/vocabulary-tree matching:** Exhaustive matching guarantees no valid image pairs are missed regardless of capture order. For small-to-medium frame sets (hundreds of frames), it is computationally tractable. Sequential matching would miss pairs from looping camera trajectories; vocabulary-tree matching requires pre-built tree structures.

#### COLMAP Stage 3: Sparse Reconstruction (Bundle Adjustment)

```
colmap mapper
  --database_path database.db
  --image_path images_dir
  --output_path sparse/
```

COLMAP's incremental mapper reconstructs the sparse 3D point cloud and camera poses simultaneously through bundle adjustment (BA). Starting from the image pair with the highest number of verified correspondences, the mapper:
1. Estimates the essential matrix E from matched keypoints (using 5-point algorithm), recovers relative rotation R and translation t (up to scale).
2. Triangulates matched keypoints into 3D points via DLT (Direct Linear Transform).
3. Registers subsequent images via PnP (Perspective-n-Point) — solves for each new camera pose given 3D–2D correspondences.
4. Performs global bundle adjustment iteratively, minimizing the reprojection error:

```
E_BA = Σ_{i,j} ρ( ||π(P_j, C_i) - x_{ij}||² )
```

where `π(P_j, C_i)` projects 3D point `P_j` through camera `C_i`'s intrinsic/extrinsic matrix, `x_{ij}` is the observed 2D keypoint, and `ρ` is a robust Huber loss function dampening outlier influence.

Output: `sparse/0/` containing `cameras.bin` (intrinsics per camera), `images.bin` (extrinsics — rotation quaternions and translation vectors — per image), `points3D.bin` (3D point positions with color from contributing images).

#### COLMAP Stage 4: Image Undistortion

```
colmap image_undistorter
  --image_path images_dir
  --input_path sparse/0
  --output_path undistorted/
  --output_type COLMAP
```

Although PINHOLE has no distortion model, this step produces a normalized output directory structure expected by `ns-process-data`. The undistorter also writes `cameras.txt`, `images.txt`, and `points3D.txt` in COLMAP text format, which `ns-process-data` parses to produce `transforms.json`.

**Output Artifacts:** `outputs/colmap/undistorted/` with images, `cameras.txt`, `images.txt`, `points3D.txt`, and the SQLite `database.db`.

**Implicit coordinate system:** COLMAP operates in an arbitrary metric coordinate frame aligned with the first registered image pair. The scale is metric only if the scene has a known reference; otherwise it is up to a global scale factor. Nerfstudio's `ns-process-data` reads this coordinate system directly, so all downstream 3D coordinates inherit COLMAP's frame.

---

### Phase 3: Neural Radiance Field Training

**Objective:** Fit a continuous volumetric scene representation—either an implicit MLP (nerfacto) or explicit 3D Gaussian splatting (splatfacto)—over the COLMAP-calibrated image set. The trained model provides: (a) photorealistic novel-view synthesis, (b) a queryable density field for depth estimation, and (c) the `transforms.json` camera model used by the semantic-geometric bridge.

**Input Artifacts:** `undistorted/` images from Phase 2, `sparse/0/` COLMAP model.

**Core Logic ([run_nerf.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase3/run_nerf.py)):** Two sequential subprocess calls to the Nerfstudio CLI.

#### Stage 3a: Data Processing (`ns-process-data`)

```python
# run_nerf.py lines 32-41
images_preprocessed_str = Path("data/scenes/scene_001/images_preprocessed").as_posix()
colmap_model_path_str = (colmap_dir / "sparse" / "0").as_posix()

cmd1 = [
    "ns-process-data", "images",
    "--data", images_preprocessed_str,
    "--output-dir", ns_workspace_str,
    "--colmap-model-path", colmap_model_path_str,
    "--skip-colmap"
]
```

**Critical design flaw:** `images_preprocessed_str` is **hardcoded** to `data/scenes/scene_001/images_preprocessed` on line 32, ignoring the `--colmap_undistorted` argument that the user passes. The `--colmap_undistorted` argument correctly influences `colmap_dir`, but the images path for `ns-process-data` is always `scene_001`, making this script non-reusable for any other scene without source modification. This is inconsistent with the argument-driven design of all other scripts in the pipeline.

`--skip-colmap` instructs Nerfstudio to read the pre-existing COLMAP model at `sparse/0/` and convert it to `transforms.json` without re-running feature extraction. The converter reads `cameras.bin`/`images.bin` and constructs a `transforms.json` with top-level intrinsics and a per-frame list of 4×4 camera-to-world matrices. See §4.2 for the full schema.

The `transform_matrix` is the **camera-to-world** (c2w) matrix: a 4×4 homogeneous rigid transform with the bottom row [0,0,0,1]. The rotation block R ∈ SO(3) encodes camera orientation; the translation column t encodes camera position in world coordinates. This matrix is what `ray_projection.py` reads and applies via `c2w[:3, 3]` (origin) and `c2w[:3, :3]` (rotation).

#### Stage 3b: NeRF Training (`ns-train`)

```python
model_type = "splatfacto" if args.high_res else "nerfacto"
cmd2 = ["ns-train", model_type, "--data", ns_workspace_str, "--vis", "viewer"]
```

**Nerfacto (default, low-VRAM mode):** An implicit volumetric representation based on the original NeRF formulation. A trained MLP `F_θ: (x, d) → (c, σ)` maps 3D position x and viewing direction d to color c and volume density σ. Novel views are rendered via the volume rendering integral:

```
C(r) = ∫_{t_n}^{t_f} T(t) · σ(r(t)) · c(r(t), d) dt

where T(t) = exp(-∫_{t_n}^{t} σ(r(s)) ds)
```

`T(t)` is transmittance — the probability of a ray surviving to depth t without hitting opaque matter. The discrete approximation sums over sampled depths `{t_i}`:

```
Ĉ(r) = Σ_i T_i (1 - exp(-σ_i δ_i)) c_i
where T_i = exp(-Σ_{j<i} σ_j δ_j), δ_i = t_{i+1} - t_i
```

Training minimizes photometric L2 loss between rendered `Ĉ(r)` and observed pixel colors over all training rays. Low-VRAM defaults are applied: `--pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.eval-num-rays-per-chunk 2048`.

**Splatfacto (high-res mode — 3D Gaussian Splatting):** Rather than integrating a continuous density field, splatfacto represents the scene as a set of explicit 3D Gaussians `{μ_k, Σ_k, α_k, SH_k}` where μ_k is centroid, Σ_k is the 3D covariance (parameterized via scale s_k and rotation quaternion q_k), α_k is opacity, and SH_k are spherical harmonic coefficients encoding view-dependent color. Rendering is done by projecting ("splatting") each 3D Gaussian onto the image plane and performing alpha-compositing in depth-sorted order:

```
Σ_2D = J · W · Σ_3D · W^T · J^T
```

where J is the Jacobian of the projective mapping and W is the world-to-camera rotation. This rasterization-based approach is 10–100× faster than ray marching and produces shaper high-frequency textures, at the cost of requiring initialization from COLMAP sparse points. The training objective is the same photometric L2 loss plus an optional SSIM regularization.

**Code comment annotation:** The inline comment `# Splatfacto (3D Gaussian Splatting) generally trains faster and yields sharper edges than nerfacto` correctly summarizes the architectural tradeoff — splatfacto's explicit rasterization bypasses the MLP forward pass bottleneck of nerfacto.

**Output Artifacts:** `outputs/nerfstudio/{model_type}/{timestamp}/config.yml` and model checkpoints. The `--vis viewer` flag spawns the Nerfstudio WebSocket viewer during training (port 7007 by default).

---

### Phase 4: Semantic Scene Understanding

Phase 4 is implemented as three sequential sub-modules that collectively lift 2D visual detections into 3D world-space semantic entities.

#### Phase 4a: 2D Object Detection — YOLOv8 (`semantic_mapping.py`)

**Objective:** Detect all recognizable objects in every retained frame and record their bounding boxes and class labels.

**Model:** YOLOv8-nano (`yolov8n.pt`, 6.5 MB), loaded via the `ultralytics` package. The nano variant is deliberately chosen — the inline comment notes upgrade to `yolov8x.pt` is possible for higher recall. YOLOv8n uses a CSP (Cross-Stage Partial) backbone with a decoupled detection head and performs single-pass inference at ~hundreds of FPS on GPU.

**Inference logic:** For each frame, `model(img, verbose=False)` returns a `Results` object. Per detected bounding box, the code extracts:
- `box.xyxy[0]` — coordinates in [x1, y1, x2, y2] format (pixel space, float)
- `box.cls` — integer class index, mapped to string name via `model.names`
- `box.conf` — scalar confidence score

**Output schema per frame entry:**
```json
{
  "class_id": 57,
  "class_name": "chair",
  "confidence": 0.874,
  "bbox": [142.3, 219.7, 388.0, 541.2]
}
```

The 2D scene graph is accumulated as `{"frames": {"frame_00001.jpg": [det1, det2, ...], ...}}` and written to `scene_graph_2d.json`.

**Design note:** The model is loaded once and reused across all frames — critical for throughput since YOLOv8 model initialization involves GPU memory allocation and weight loading.

#### Phase 4b: Instance Segmentation — SAM (`sam_segmentation.py`)

**Objective:** For each YOLO-detected bounding box, extract a pixel-precise binary segmentation mask isolating the object from background.

**Model:** Segment Anything Model (SAM), available in three capacities: ViT-B (91M params), ViT-L (308M params), ViT-H (636M params). Weights are downloaded on demand from Meta's CDN via `urllib.request.urlretrieve`, cached at `output_dir/weights/`. The model registry pattern `sam_model_registry[model_type](checkpoint=weights_path)` uses SAM's internal factory to instantiate the appropriate ViT encoder variant.

**Inference pattern:** SAM operates in **prompted mode** — the predictor encodes the full image into a feature embedding once per frame (`predictor.set_image(image_rgb)`), then accepts prompts to generate masks. Here, YOLO bounding boxes serve as **box prompts**:

```python
masks, scores, logits = predictor.predict(
    box=bbox,           # np.array [x1, y1, x2, y2]
    multimask_output=False
)
```

`multimask_output=False` requests a single best mask rather than three candidate masks at different granularities. The returned `masks[0]` is a boolean numpy array of shape `(H, W)`. Confidence is recorded as `scores[0]` (SAM's internal IoU prediction head score).

**Internal SAM mechanism:** The ViT image encoder applies 14×14 patch embedding with absolute positional encodings, producing a 64×64 feature grid. The prompt encoder converts box coordinates into sparse embeddings. The mask decoder applies two-way cross-attention between prompt tokens and image features, producing three candidate masks and IoU scores. When `multimask_output=False`, the highest-scoring mask is selected.

**Storage:** Each mask is saved as a compressed `.npz` file via `np.savez_compressed(filepath, mask=best_mask)`. The enriched scene graph stores a relative path reference (`mask_path`) rather than inlining the binary data — a correct design decision since a (1920×1080) boolean mask is ~2MB uncompressed but ~50KB compressed.

**Visualization:** The `overlay_masks()` utility (in `utils.py`) blends each mask over the original frame with alpha=0.5 using `cv2.addWeighted`. The color generation logic contains a subtle behavioral detail:
```python
np.random.seed(42)  # For consistent colors
colors = np.random.randint(0, 255, size=(len(masks_with_labels), 3), dtype=np.uint8)
```
`np.random.seed(42)` is called **inside `overlay_masks` on every invocation** — i.e., it resets the random state on every frame. This means color assignments are consistent within a frame across runs (deterministic), but the color-to-object mapping is determined by each frame's local detection index `i`, not by a globally consistent class identity. Two frames each containing a chair and a table will assign the same two colors, but which color maps to which class depends on detection order within that frame. Class identity is conveyed by the label text overlay via `sam_segmentation.py`'s call: `frame_masks.append((best_mask, det['class_name']))`.

**Output Artifacts:** `scene_graph_sam.json` (enriched graph with `mask_path` and `mask_confidence` fields), `masks/` directory with `.npz` files, `visualizations/` with annotated frames.

#### Phase 4c: 3D Ray Projection and Centroid Aggregation (`ray_projection.py`)

**Objective:** Convert 2D bounding-box centroids, observed across multiple frames with known camera poses, into 3D world-space position estimates for each detected object class.

This is the **core novel technical contribution** of the pipeline. See Section 5 for a full mathematical deep-dive. Summary:

1. For each frame, retrieve the 4×4 c2w matrix from `transforms.json` using filename-based lookup.
2. For each detection, compute 2D bbox center `(cx, cy)` and unproject through the pinhole model to get a normalized 3D ray direction in camera space.
3. Rotate the direction into world space via `c2w[:3,:3] @ direction`.
4. Cast the ray to a simulated depth (stochastic heuristic in current implementation) to get a 3D point estimate.
5. Accumulate per-class estimates, compute mean centroid.

**Output Artifacts:** `scene_graph_3d.json`:
```json
{
  "objects": [
    {
      "name": "chair",
      "position": [1.23, -0.45, 2.87],
      "observations": 14
    },
    ...
  ]
}
```

This JSON is the **central semantic artifact** consumed by Phases 5, 7, 8, and 9.
