# Spatial Intelligence: Technical Documentation — Part 2
## Phases 5–9, Data Schemas, Deep Dives, Evaluation, Engineering Concerns, Tradeoffs

---

## 3 (continued). Phase-by-Phase Technical Breakdown

### Phase 5: LLM-Based Spatial Reasoning Agent

**Objective:** Enable free-form natural language queries over the 3D scene graph by injecting geometric context into an LLM prompt. See Section 6 for the full architecture deep-dive.

**Input Artifacts:** `scene_graph_3d.json`

**Core Logic ([spatial_agent.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py)):**

The agent operates in two modes: single-query (via `--query` argument) and interactive REPL loop. Both modes share the same context serialization and provider dispatch logic.

**Context serialization ([format_scene_context](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#11-24)):**
The 3D scene graph is linearized into a natural-language string:
```
"The following objects were detected in the 3D scene, along with their 3D coordinates (X, Y, Z):
- A chair located at position [X: 1.23, Y: -0.45, Z: 2.87] (Aggregated from 14 views)
- A tv located at position [X: 3.10, Y: 0.12, Z: 1.50] (Aggregated from 8 views)
..."
```
Each object's position is rounded to 2 decimal places via `round(pos[i], 2)` before injection. The `observations` count is included in the context string as a proxy for detection confidence — a higher observation count implies the centroid estimate is more reliable.

**Provider dispatch:** A simple conditional chain (`if provider == "huggingface": ... elif provider == "gemini": ... else: openai`) implements runtime selection. Each branch calls an independent function with its own API client initialization. No common interface or base class is defined — this is a flat strategy pattern implemented with top-level functions.

**Output:** Natural language response string, printed to stdout (CLI mode) or returned to the Streamlit session state (GUI mode).

---

### Phase 6: Evaluation, Mesh Export, and Video Rendering

**Objective:** Quantitatively assess NeRF reconstruction quality, export a polygonal mesh for downstream DCC (Digital Content Creation) use, and produce a rendered flythrough video for visual validation.

**Input Artifacts:** `config.yml` from Phase 3 training output.

#### Phase 6a: Quantitative Evaluation ([evaluate_nerf.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/evaluate_nerf.py))

Invokes `ns-eval` via the PyTorch monkeypatch wrapper (see Section 9). The Nerfstudio `ns-eval` script holds out a subset of training images as a validation split, renders novel views at those exact camera poses, and computes pixel-level metrics against ground truth:

- **PSNR:** `PSNR = 10 · log₁₀(MAX²/MSE)` where MAX=1.0 for normalized images. Higher is better; typically NeRFs achieve 25–31 dB.
- **SSIM:** Structural Similarity Index (same as Phase 1 subsample computation but applied between rendered and GT images). Range [0,1], higher is better.
- **LPIPS:** Learned Perceptual Image Patch Similarity. Uses VGG/AlexNet feature maps to measure perceptual distance. Lower is better.

Results are written to [outputs/metrics/evaluation_metrics.json](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/metrics/evaluation_metrics.json):
```json
{"results": {"psnr": 28.47, "ssim": 0.843, "lpips": 0.187}}
```

#### Phase 6b: 360° Video Rendering ([render_video.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py))

```
ns-render interpolate
  --load-config config.yml
  --output-path scene_001_final_render_interpolate.mp4
```

The `interpolate` trajectory type instructs Nerfstudio to compute a smooth camera path by interpolating between all training camera poses using spline interpolation. This produces a continuous flythrough that visits the coverage region of all training cameras. The code also registers `spiral` as an alternative trajectory (helical path around scene center) via the `--trajectory` argparse argument.

**Dead code — double [run_cmd](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#6-9) definition:** [render_video.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py) contains a critical code artifact: [run_cmd](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#6-9) is defined **twice** in the same module. Lines 6–9 define a stub that only prints the command:
```python
def run_cmd(cmd: list, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
```
This is immediately overwritten by the full 25-line definition at lines 10–35 containing the actual monkeypatch `runner` string and `subprocess.run(wrapper_cmd, check=True)`. The first definition is unreachable dead code — Python binds [run_cmd](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#6-9) to the second definition at module import time. The stub appears to be a development artifact from an earlier implementation that was never removed.

**Windows WinError 87 patch:** The surviving [run_cmd](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#6-9) (lines 10–35) injects two monkeypatches into the `ns-render` subprocess. The root cause of WinError 87 (ERROR_INVALID_PARAMETER) is `mediapy`'s internal call to `subprocess.Popen` with `env={}` (empty dict). On Windows, the `CreateProcess` Win32 API rejects an empty environment block. The patch:
```python
"old_popen = subprocess.Popen; "
"subprocess.Popen = lambda *a, **k: old_popen(*a, "
"**{**k, 'env': None} if k.get('env') == {} else k); "
```
detects `env={}` and replaces it with `env=None` (parent environment inheritance). Additionally, the FFmpeg path is hardcoded:
```python
"import mediapy; mediapy.set_ffmpeg(r'C:\\ffmpeg\\bin\\ffmpeg.exe'); "
```
This machine-specific absolute path ([C:\ffmpeg\bin\ffmpeg.exe](file:///ffmpeg/bin/ffmpeg.exe)) will fail on any system where FFmpeg is installed elsewhere. A more robust implementation would locate FFmpeg via `shutil.which('ffmpeg')`.

#### Phase 6c: TSDF Mesh Export ([export_mesh.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/export_mesh.py))

```
ns-export tsdf
  --load-config config.yml
  --output-dir outputs/mesh/
```

Nerfstudio's TSDF exporter samples the trained NeRF density field on a volumetric grid, computes the **Truncated Signed Distance Function** (TSDF), and runs Marching Cubes to extract the isosurface.

**TSDF mathematical basis:** For each voxel center p in a 3D grid, the TSDF value is computed as:
```
D(p) = trunc(d(p) / τ) · sign(d(p))
```
where [d(p)](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#6-9) is the signed distance from p to the nearest surface (positive outside, negative inside), τ is the truncation distance, and `trunc` clips values to [-1, +1]. TSDF values are accumulated across multiple viewpoints via weighted averaging. Marching Cubes then extracts the zero-level isosurface of D(p), producing a triangulated mesh.

**Why TSDF over Poisson:** TSDF is the default export method (`method="tsdf"`). Poisson surface reconstruction (the alternative) requires oriented point normals and fails on incomplete data; TSDF handles partial observations more gracefully by integrating over volumetric grid uncertainty.

**Output Artifacts:** [outputs/mesh/tsdf_mesh.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/mesh/tsdf_mesh.ply) (PLY format, triangulated mesh with vertex colors derived from NeRF color field at mesh vertices).

---

### Phase 7: Semantic 3D Point Cloud Generation

**Objective:** Produce an Open3D [.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/semantic_pcd.ply) point cloud where each detected semantic class is represented as a volumetric sphere of colored points centered at its 3D centroid, with sphere size and density proportional to observation count.

**Input Artifacts:** `scene_graph_3d.json`

**Core Logic ([semantic_pointcloud.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase7/semantic_pointcloud.py)):**

For each object entry, a sphere of `N_points` is synthesized using **uniform spherical sampling**:

```
φ ~ Uniform(0, 2π)
cos(θ) ~ Uniform(-1, 1)  →  θ = arccos(cos θ)
u ~ Uniform(0, 1)         →  r = R · u^(1/3)

x = cx + r · sin(θ) · cos(φ)
y = cy + r · sin(θ) · sin(φ)
z = cz + r · cos(θ)
```

The cube-root transform on `u` enforces **uniform density within the sphere volume** — without it, the distribution `r ~ Uniform(0, R)` would concentrate points near the center since the volume element `dV = r² dr sin(θ) dθ dφ` grows with r². Using `r = R · u^(1/3)` inverts this bias.

**Adaptive sizing formulas:**
```python
num_points = min(5000, 500 + (obs_count * 100))
radius = min(1.0, 0.2 + (np.log1p(obs_count) * 0.1))
```
The logarithmic radius growth (`log1p` = log(1 + obs_count)) ensures diminishing returns — an object seen 100 times has radius 0.2 + 0.46 ≈ 0.66, not 10× larger than an object seen 10 times.

**Class color assignment:** `np.random.seed(42)` then `np.random.rand(3)` per unique class name. The seed ensures identical colors across runs for the same set of class names. Colors are in [0,1] float RGB, later converted to Open3D's `Vector3dVector` format.

**Normal estimation:** `pcd.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=2*sphere_radius, max_nn=30))` computes per-point normals via PCA over the local neighborhood. This is not required for point cloud display but enables correct shading in Meshlab and other 3D viewers that compute per-point lighting.

**Output Artifacts:** [outputs/semantic_pcd.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/semantic_pcd.ply) — PLY with XYZ + RGB + Normal fields.

---

### Phase 8: Topological Proximity Graph

**Objective:** Reduce the 3D scene graph to a 2D relational network where edges encode spatial proximity between object centroids, enabling at-a-glance comprehension of room layout topology.

**Input Artifacts:** `scene_graph_3d.json`

**Core Logic ([graph_visualizer.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase8/graph_visualizer.py)):**

**Distance matrix computation:**
```python
dist_matrix = squareform(pdist(positions))
```
`scipy.spatial.distance.pdist` computes pairwise Euclidean distances between all N object centroids in O(N²) time:
```
d(i,j) = ||p_i - p_j||₂ = √((x_i-x_j)² + (y_i-y_j)² + (z_i-z_j)²)
```
`squareform` converts the condensed distance vector (upper triangle) into a full N×N symmetric matrix.

**Graph construction:**
```python
if dist < distance_threshold:  # default 7.0 meters
    weight = max(0.5, 3.0 - (dist / (distance_threshold / 3.0)))
    G.add_edge(i, j, weight=weight, distance=dist)
```
Edges are added only for object pairs within 7.0 meters. Edge weight is a linear inverse-distance function: objects 0m apart have weight 3.0; objects at 7m have weight 3.0 - (7/(7/3)) = 3.0 - 3.0 = 0, floored to 0.5. This causes spatially close objects to render with visually thicker edges.

**Node sizing:**
```python
size = 800 + (np.log1p(observations[i]) * 400)
```
Observation count drives node size logarithmically — a heavily-detected object (e.g., large furniture in frame most of the time) renders as a visually larger node than a rarely-detected small object.

**Layout algorithm:** Kamada-Kawai (`nx.kamada_kawai_layout(G)`) minimizes the sum of squared differences between graph-theoretic distances and Euclidean layout distances:
```
E = Σ_{i<j} k_{ij}(d_{ij} - ||p_i - p_j||)²
```
This produces a layout where graph distance (hop count) correlates with geometric separation, which for proximity graphs reflects real spatial relationships. Kamada-Kawai is preferred over spring (`fruchterman_reingold`) because it is deterministic and preserves metric structure better for sparse graphs.

**Edge label filtering:** Distance labels are only drawn for edges where `dist < distance_threshold * 0.4 = 2.8m` to avoid cluttering the chart. Labels show distance in meters formatted as `f"{dist:.1f}m"`.

**Output Artifacts:** [outputs/topology_graph.png](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/topology_graph.png) at 300 DPI.

---

### Phase 9: Streamlit Presentation Dashboard

**Objective:** Unify all pipeline outputs into a single interactive web application that is computable on any machine that ran the prior phases, requiring no additional configuration beyond the streaming Gemini API key.

**Input Artifacts:** All phase outputs via hardcoded relative paths from the project root.

**Core Logic ([app.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py)):**

**Path registration (module-level constants):**
```python
BASE_DIR = Path("outputs")
SCENE_GRAPH_PATH = Path("data/scenes/scene_001/semantic_3d/scene_graph_3d.json")
TOPOLOGY_GRAPH_PATH = BASE_DIR / "topology_graph.png"
PCD_PATH = BASE_DIR / "semantic_pcd.ply"
VIDEO_PATH = BASE_DIR / "scene_001_final_render_interpolate.mp4"
MESH_PATH = BASE_DIR / "mesh" / "tsdf_mesh.ply"
```
All paths are relative to the Streamlit working directory (project root). This assumes `streamlit run app.py` is executed from the project root. There is no dynamic path discovery — a missing file causes `st.error()` or `st.info()` graceful degradation rather than a crash.

**Tab 1 — 3D Semantic Viewer:**

[load_plotly_pcd()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py#67-136) is decorated with `@st.cache_data`, meaning it executes once per unique `scene_graph_file` argument and returns a cached `go.Figure` on subsequent rerenders. This prevents expensive sphere-point recomputation on every Streamlit widget interaction.

The function re-implements Phase 7's sphere generation (uniform spherical sampling via cube-root transform) but with **dramatically reduced point counts**:
```python
num_points = min(800, 200 + (obs_count * 20))  # vs Phase 7's 5000
```
This is the **WebGL memory budget decision**: a full-density point cloud with 5000 points per class × multiple classes would exceed WebGL's 16MB per-buffer limit in browsers, causing tab crashes. At 800 points/class, the entire scene renders comfortably. Each class is a separate `go.Scatter3d` trace with its own color. An additional text label trace is added per object at position `[cx, cy, cz + radius + 0.1]` — slightly above the sphere centroid so it remains visible.

The `go.Layout` uses a dark background `bgcolor="rgb(20, 24, 30)"` with transparent paper background `paper_bgcolor="rgba(0,0,0,0)"` — a deliberate aesthetic choice for dashboard presentation.

**Tab 2 — Topology Map:** A static image served via `st.image(Image.open(...))`. No interactivity beyond Streamlit's native image zoom.

**Tab 3 — LLM Chat:**

Uses Streamlit's `st.session_state.messages` list (a list of `{"role": str, "content": str}` dicts) as the conversation history buffer. On each `st.chat_input()` submission, the scene graph is freshly loaded ([load_scene_graph_3d()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#7-10)), context is re-serialized ([format_scene_context()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#11-24)), and the query is sent to the selected provider. **Note:** the context is reconstructed from scratch on every query — there is no stateful message history passed to the LLM. Each LLM call is independent; the response does not accumulate conversational context across turns. Only the Gemini path is fully wired — other providers return a placeholder string noting upcoming support.

**Tab 4 — Interactive NeRF Viewer ([start_nerf_viewer](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py#223-249)):**

This feature has two significant hardcoded values that deserve explicit documentation:

**Hardcoded API key (security issue):** [app.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py) line 50 reads:
```python
api_key = st.text_input("Enter API Key (Google AI Studio / OpenAI)", type="password",
    value="AIzaSyAqjW8pEIqdYgzrFGwopNg7X2ceEXV3QIw")
```
A live Gemini API key is embedded as the `value=` default. While `type="password"` masks it in the UI, the key is fully visible in the source file and any version control history. This is a hardcoded secret that should be replaced by an environment variable lookup (`os.getenv("GEMINI_API_KEY", "")`).

**Hardcoded training timestamp path:** The `runner` string passed to `subprocess.Popen` contains:
```python
"sys.argv=['ns-viewer', '--load-config', "
"'outputs/nerfstudio/nerfacto/2026-03-04_021356/config.yml', "
"'--viewer.websocket-port', '7007']; "
```
The config path embeds the training timestamp `2026-03-04_021356` as a literal string. Any re-invocation of `ns-train` produces a new timestamp directory, silently breaking Tab 4. No glob or discovery logic is applied to find the latest checkpoint.

On button click, [start_nerf_viewer()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py#223-249) checks `st.session_state.viewer_proc` — if absent, spawns the background process. A simulated progress bar (`time.sleep(0.15)` × 100 iterations = ~15 seconds) provides user feedback, since `subprocess.Popen` is non-blocking and no IPC mechanism signals actual model readiness. Once running, `components.iframe("http://localhost:7007", height=800)` embeds the Nerfstudio WebSocket viewer: a React/WebGL frontend that streams rendered frames from the PyTorch server over WebSocket, allowing real-time camera control.

---

## 4. Central Data Structures and Schemas

### 4.1 `scene_graph_3d.json` — Full Schema

The central cross-phase artifact, produced by [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) and consumed by Phases 5, 7, 8, and 9.

```json
{
  "objects": [
    {
      "name": "chair",        // string — YOLOv8 COCO class name
      "position": [           // list[float, float, float] — 3D world coord (X, Y, Z)
        1.230456,             //   X: lateral axis (COLMAP frame)
        -0.451230,            //   Y: vertical axis
        2.874512              //   Z: depth/forward axis
      ],
      "observations": 14      // int — number of per-frame detection samples aggregated
    }
  ]
}
```

**Coordinate frame:** COLMAP world coordinates, inherited from COLMAP's arbitrary bundle-adjustment frame. No canonical up-axis is enforced — the Y-axis direction depends on the initial camera orientation in the first registered frame pair.

**One entry per unique class, not per instance:** Multiple instances of the same class (e.g., two chairs) are **merged into a single centroid** — the mean of all per-frame ray-projected points for that class. This is a significant structural limitation documented in Section 11.

### 4.2 `transforms.json` — Camera Model Schema

Produced by `ns-process-data --skip-colmap`, consumed by [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) and Phase 3 training. Since Phase 2 forces `--ImageReader.camera_model PINHOLE`, the output `transforms.json` reflects a 4-parameter pinhole model with no distortion fields (`k1`, `k2`, `p1`, `p2` are absent):

```json
{
  "fl_x": 1234.56,        // focal length in pixels, x-axis (from COLMAP cameras.bin)
  "fl_y": 1234.56,        // focal length in pixels, y-axis
  "cx": 960.0,            // principal point x (typically near W/2)
  "cy": 540.0,            // principal point y (typically near H/2)
  "w": 1920,              // image width in pixels
  "h": 1080,              // image height in pixels
  // No distortion coefficients — PINHOLE model has none
  "frames": [
    {
      "file_path": "./images/frame_00001.jpg",   // relative path from ns_workspace
      "transform_matrix": [                       // 4×4 float, row-major, c2w convention
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

[ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) reads this file and accesses these exact top-level keys with fallback defaults: `transforms.get("fl_x", 1000)`, `transforms.get("cx", 500)`. The fallback values of 1000 and 500 are only exercised if `ns-process-data` fails to populate the file — in correct pipeline execution they are always overridden by the COLMAP-estimated values. The `transform_matrix` uses the **camera-to-world (c2w)** convention: column 3 `[tx, ty, tz]` is camera position in world space, extracted by [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) as `c2w[:3, 3]`, and the R subblock as `c2w[:3, :3]`.

### 4.3 `scene_graph_2d.json` and `scene_graph_sam.json`

```json
// scene_graph_2d.json (YOLO output)
{
  "frames": {
    "frame_00001.jpg": [
      {
        "class_id": 57, "class_name": "chair",
        "confidence": 0.874,
        "bbox": [142.3, 219.7, 388.0, 541.2]  // [x1, y1, x2, y2] pixels
      }
    ]
  }
}

// scene_graph_sam.json (SAM-enriched)
{
  "frames": {
    "frame_00001.jpg": [
      {
        "class_id": 57, "class_name": "chair",
        "confidence": 0.874,
        "bbox": [142.3, 219.7, 388.0, 541.2],
        "mask_path": "masks/frame_00001_obj000_chair.npz",  // relative to SAM output dir
        "mask_confidence": 0.931   // SAM IoU prediction score
      }
    ]
  }
}
```

### 4.4 `tsdf_mesh.ply` — Mesh Format

Binary PLY format with the following element/property structure (Nerfstudio standard output):
- **vertex** element: `x y z` (float32 XYZ position) + `red green blue` (uint8 RGB) + `nx ny nz` (float32 normals)
- **face** element: vertex_indices list (typically `uchar N` + `int v1 v2 v3` for triangles)

The mesh is in COLMAP world coordinates, matching the NeRF volume coordinate frame.

---

## 5. The Semantic-Geometric Bridge — Phase 4 Deep Dive

Phase 4c ([ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py)) is the system's most technically novel module. It performs **multi-view geometric lifting** of 2D semantic detections into 3D world coordinates without running actual NeRF inference at query time.

### 5.1 Mathematical Pipeline: 2D Pixel → 3D World Coordinate

**Step 1: Bounding Box Centroid Extraction**

For detection with bbox `[x1, y1, x2, y2]` in pixel space:
```
cx = (x1 + x2) / 2.0
cy = (y1 + y2) / 2.0
```
This represents the 2D image-plane centroid of the detected object's bounding box — a heuristic proxy for the object's image-plane projection center. It is less accurate than using the SAM mask centroid (centroid of the binary mask's True pixels), but mask centroids were not plumbed through to [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) in the current implementation — the code reads from `scene_graph_sam.json` which contains the bbox. This is a documented approximation.

**Step 2: Pinhole Camera Unprojection (Camera Space Ray)**

The standard pinhole camera model maps a 3D camera-space point (X_c, Y_c, Z_c) to image pixel (u, v):
```
u = fl_x · (X_c / Z_c) + cx_img
v = fl_y · (Y_c / Z_c) + cy_img
```

Inverting for a unit-depth ray direction d in camera space:
```
d_x = (cx - cx_img) / fl_x
d_y = (cy - cy_img) / fl_y
d_z = 1.0
```

The code reads intrinsics from `transforms.json`:
```python
fl_x = transforms.get("fl_x", 1000)   # fallback 1000 if missing
fl_y = transforms.get("fl_y", 1000)
cx_img = transforms.get("cx", 500)
cy_img = transforms.get("cy", 500)
```
The fallback values (fl_x=1000, cx=500) correspond to a rough approximation of a 1920×1080 camera. If `transforms.json` is correctly generated by `ns-process-data`, these will be populated with actual COLMAP-estimated values.

**Step 3: Ray Normalization**

```
d̂ = d / ||d||
```
where `||d|| = √(d_x² + d_y² + 1.0)`. This produces a unit direction vector in camera space.

**Step 4: World-Frame Ray Direction**

```
d_world = R_{c2w} · d̂
```
where `R_{c2w} = c2w[:3, :3]` is the rotation submatrix of the camera-to-world transform. This rotates the camera-space direction into the world coordinate frame.

**Step 5: Depth Estimation (Heuristic)**

```python
simulated_depth = 2.0 + np.random.normal(0, 0.5)
```

This is a **stochastic heuristic** — the most significant approximation in the pipeline. Rather than querying the NeRF MLP's density field to find the actual surface intersection along the ray, a Gaussian-distributed depth centered at 2.0 meters is sampled. The code comment explicitly acknowledges this: *"In a full production implementation, this would involve ray marching against the trained NeRF density grid."*

The implication: 3D centroid positions carry a systematic bias based on the assumed depth. Objects consistently underestimated (actual depth > 2.0m) will have their centroids pushed too close to the camera; objects at < 2.0m depth will have centroids displaced forward. The random jitter (σ=0.5m) adds stochastic variance on top of this systematic bias.

**Step 6: 3D Point Computation**

```
P_3D = camera_origin + d_world · depth
     = c2w[:3,3] + (c2w[:3,:3] · d̂) · simulated_depth
```

This is the standard ray equation: origin + direction × depth.

**Step 7: Per-Class Centroid Aggregation**

All 3D point estimates `{P_3D^(f)}` for a given class across all frames `f` where that class was detected are accumulated:
```
centroid_{class} = (1/N) · Σ_{f=1}^{N} P_3D^(f)
```

The mean is computed via `np.mean(pts, axis=0)` over the stacked Nx3 array. This **averaging** across multiple viewpoints suppresses individual-frame estimation errors:
- Systematic depth error is partially cancelled if the object is observed from cameras at varying distances.
- Random noise has standard deviation reduced by 1/√N for N observations.
- Camera pose errors from COLMAP bundle adjustment also contribute non-zero variance that averaging partially mitigates.

### 5.2 Accuracy Considerations and Known Limitations

1. **Depth heuristic invalidates metric accuracy:** Without actual NeRF depth queries, centroid positions are offset from ground truth by `E[actual_depth] - 2.0` meters in the ray direction. For objects near walls (3–5m depth in indoor rooms), all centroids are systematically shifted ~1–3m toward the camera.

2. **Bounding-box centroid ≠ object centroid:** For asymmetric objects (chairs, people) or partially occluded objects, the bbox centroid may be far from the object's 3D center of mass.

3. **Single-class aggregation collapses instances:** All chairs are merged. If two chairs are on opposite sides of a room, their centroid lands in empty space between them.

4. **No depth-from-NeRF integration:** The comment "In Phase 6/7, this integrates with depth-regularized NeRFs" is forward-looking aspirational text — no such integration exists in the current codebase. Depth-regularized NeRFs (e.g., DS-NeRF, Depth-supervised NeRF) would provide per-ray expected depth values from the trained density field, replacing the heuristic with actual learned geometry.

5. **Frame-transform mapping relies on filename exact match:** `transform_frames = {Path(f["file_path"]).name: f["transform_matrix"] ...}` strips the path to just the filename. Frames that were extracted and preprocessed but not registered by COLMAP (and thus absent from `transforms.json`) are silently skipped via the `if frame_name not in transform_frames: continue` guard.

---

## 6. LLM Spatial Reasoning Architecture — Phase 5 Deep Dive

### 6.1 Prompt Engineering Strategy

The prompt construction in [format_scene_context()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#11-24) follows a **structured enumeration** strategy: each detected object is rendered as a bullet point with its coordinates and confidence proxy (observations count). The format is compact and unambiguous — no prose descriptions, no inferred relationships, no spatial preprocessing.

**Full prompt template (Gemini/OpenAI):**
```
System: "You are an intelligent 3D spatial reasoning agent analyzing a 3D scene
reconstructed from a monocular video. You are provided with a semantic scene graph."

User:
"Context about the room:
The following objects were detected in the 3D scene, along with their 3D coordinates (X, Y, Z):
- A chair located at position [X: 1.23, Y: -0.45, Z: 2.87] (Aggregated from 14 views)
- A tv located at position [X: 3.10, Y: 0.12, Z: 1.50] (Aggregated from 8 views)
...

Question: Where is the TV relative to the chair?"
```

**What geometric context is injected:** Raw XYZ world coordinates and a per-object confidence proxy (observation count). The LLM must internally compute:
- Euclidean distances between object pairs
- Directional relationships (left/right/front/behind based on X/Y/Z sign conventions)
- Relative scales

**What is NOT injected:** Object sizes, bounding box extents, room boundaries, orientation of objects, occlusion relationships, or the coordinate axis conventions (e.g., which axis is vertical). This places significant burden on the LLM to reason correctly about the coordinate frame.

**HuggingFace prompt (Mistral-7B Instruct format):**
```
<s>[INST] You are an intelligent 3D spatial reasoning agent...
Context about the room:
{context}

Question: {user_query} [/INST]
```
The response is extracted by splitting on `[/INST]` and taking the last part: `result.split("[/INST]")[-1].strip()`. This strips the injected prompt from the model's output, which Mistral-7B-Instruct repeats before generating the response.

### 6.2 Provider Abstraction Design

The abstraction is a **flat strategy pattern** — three independent functions ([query_huggingface](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#25-40), [query_gemini](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#41-74), [query_openai](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#75-96)) with identical signatures [(context: str, user_query: str, [api_key: str]) -> str](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase3/run_nerf.py#15-64). No base class, interface, or ABC is defined. Selection is via a string-keyed conditional in [interactive_agent()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#97-132) and the Streamlit `if provider == "Gemini"` block.

**Gemini-specific behavior:** Model selection is fully dynamic — no version is hardcoded:
```python
models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
target_model = next((m for m in models if "flash" in m and "exp" not in m), models[0])
```
The selector filters to models where the string `"flash"` appears in the model name AND `"exp"` does not (to exclude experimental preview builds). At time of development the selected model was `gemini-1.5-flash`; after Gemini 2.0 Flash became generally available, the same code would select `gemini-2.0-flash` without any code change. The [exp](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py#37-65) exclusion prevents latching onto unstable models like `gemini-2.0-flash-exp`. The fallback `models[0]` activates only if no flash model exists in the account's available model list, selecting an arbitrary generative model — a potential silent quality regression.

**Temperature setting:** Gemini and HuggingFace use `temperature=0.2`; OpenAI uses `temperature=0.2` via `client.chat.completions.create`. Low temperature reduces hallucinated spatial relationships by making the model more deterministic — critical for coordinate-based reasoning where incorrect directional answers (e.g., "the TV is to the left" when it is actually to the right) represent factual errors.

### 6.3 Spatial Relationship Inference from Raw XYZ

The LLM is expected to compute spatial relationships from raw coordinates without any preprocessing. For a query "Where is the TV relative to the chair?", the model must:
1. Extract chair position [1.23, -0.45, 2.87] and TV position [3.10, 0.12, 1.50]
2. Compute displacement vector [1.87, 0.57, -1.37]
3. Map this to natural language: "The TV is to the right of and slightly closer than the chair" (assuming X is right, Z is depth)

Modern LLMs (Gemini 1.5 Flash, GPT-4o) perform this arithmetic reliably in-context. Mistral-7B-Instruct (HuggingFace free tier) is significantly less reliable on coordinate arithmetic, particularly for multi-object comparison queries.

---

## 7. Evaluation Framework

### 7.1 Metrics: PSNR, SSIM, LPIPS

`ns-eval` operates by splitting the training dataset into train and validation subsets, rendering the trained NeRF at validation camera poses, and computing image-level metrics:

**PSNR (Peak Signal-to-Noise Ratio):**
```
MSE = (1/N) · Σ (I_rendered - I_gt)²
PSNR = 10 · log₁₀(1.0 / MSE)   [for images normalized to [0,1]]
```
PSNR is a purely pixel-level metric. It does not account for perceptual quality or structural similarity. Values of 25–32 dB are typical for indoor NeRF scenes.

**SSIM (Structural Similarity Index):**
```
SSIM(x,y) = (2μ_xμ_y + C₁)(2σ_xy + C₂) / (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)
```
Computed over local patches (default 11×11 window). C₁ and C₂ are stabilization constants. Range [0,1]. SSIM captures luminance, contrast, and structure simultaneously.

**LPIPS (Learned Perceptual Image Patch Similarity):**
Computes feature-space distance using pretrained VGG or AlexNet activations:
```
LPIPS(x,y) = Σ_l w_l · ||φ_l(x) - φ_l(y)||²₂
```
where `φ_l` are normalized activations at layer l and `w_l` are learned linear weights. Lower is better. LPIPS best correlates with human perceptual quality judgments for NeRF artifacts like blurring and floaters.

### 7.2 PyTorch 2.6 `weights_only` Monkeypatch

Both [evaluate_nerf.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/evaluate_nerf.py) and [export_mesh.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/export_mesh.py) apply the same runtime monkeypatch:
```python
runner = (
    "import sys, torch, warnings; "
    "warnings.filterwarnings('ignore', category=FutureWarning); "
    "_old_load = torch.load; "
    "torch.load = lambda *a, **k: _old_load(*a, **{**k, 'weights_only': False}); "
    "from nerfstudio.scripts.eval import entrypoint; "
    "sys.argv=sys.argv[1:]; "
    "entrypoint()"
)
wrapper_cmd = ["python", "-c", runner] + cmd
```

**Root cause:** PyTorch 2.4+ changed the default for `torch.load(..., weights_only=...)` from `False` to `True` as a security measure (preventing arbitrary code execution via pickled Python objects in checkpoint files). Nerfstudio checkpoints saved under earlier PyTorch versions contain arbitrary Python objects (optimizer states, custom schedulers) that cannot be loaded with `weights_only=True`. The monkeypatch intercepts every `torch.load` call in the process and forces `weights_only=False`.

**Implementation pattern:** Rather than modifying Nerfstudio source code or using the `TORCH_FORCE_WEIGHTS_ONLY_LOAD=0` env variable alone (which is Nerfstudio-specific), the solution injects the monkeypatch at process startup via `-c` inline script. The `sys.argv = sys.argv[1:]` strips the wrapper's own `python -c ...` from argv before invoking the entrypoint, so Nerfstudio's argparse sees the original `ns-eval` / `ns-export` / `ns-render` arguments as if invoked directly. This is a surgical runtime interception pattern that requires no source-level changes to Nerfstudio.

---

## 8. Visualization and Presentation Layer

### 8.1 Phase 7 — Why Colored Spheres, Not Raw Point Clouds

The NeRF's underlying geometry is represented as a continuous density field, not a point cloud. Extracting a raw dense point cloud would require sampling millions of NeRF query points — producing files that are 1–4 GB uncompressed. The Phase 7 design sidesteps this by constructing **synthetic sphere point clouds** from the 3D centroid data, with three key properties:

1. **Bounded size:** [min(5000, 500 + obs_count * 100)](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py#41-74) points per class — total file size remains tens of MB.
2. **Semantic color coding:** Each class gets a deterministic random RGB color (`np.random.seed(42)`), making the semantic map visually unambiguous.
3. **Confidence-proportional**: Observation count controls both density (more points = denser sphere) and radius (logarithmic scaling), making frequently-detected objects visually more prominent.

The tradeoff is that Phase 7 represents **class-level positions, not individual object geometry** — it sacrifices geometric fidelity for semantic interpretability.

### 8.2 Phase 8 — Topology Graph Design

The Kamada-Kawai layout is specifically chosen (over spring/spectral) because it minimizes edge-crossing while respecting graph-theoretic distances. For a proximity graph (where edges represent "near" relationships), this means spatially proximate objects tend to be geometrically close in the 2D layout — creating visual correspondence between the rendered graph and the actual room topology.

The `arc3,rad=0.1` curved edge style prevents overlapping edges when multiple connections share the same pair of nodes, and adds visual distinction between edge directions in the rendered graph.

Distance labels are filtered to only show for edges with `dist < 2.8m` (40% of the 7.0m threshold). This prevents the diagram from becoming unreadable when many medium-distance objects are present, while still highlighting the closest spatial relationships.

### 8.3 Streamlit App Architecture

**Data loading lifecycle:**
- [load_plotly_pcd()](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py#67-136) is called once per scene graph file and cached by `@st.cache_data`.
- Topology image is loaded via `PIL.Image.open()` on each app render (no caching — image files are small).
- Scene graph for LLM is loaded fresh on each chat query.
- NeRF viewer process is stored in `st.session_state.viewer_proc` — `session_state` persists across rerenders within a user session but is reset on page refresh.

**Subprocess lifecycle management:** The ns-viewer process (`subprocess.Popen`) is spawned but never explicitly terminated. If the user refreshes the browser tab, `st.session_state` is reset, `viewer_proc` is lost, but the background Python process continues running bound to port 7007. A second click of "Launch 3D Engine" would spawn a second process that fails to bind the already-occupied port. This is an acknowledged limitation — no graceful shutdown handler is implemented.

---

## 9. Cross-Cutting Engineering Concerns

### 9.1 Coordinate System Conventions

| System | Frame | Right | Up | Forward |
|---|---|---|---|---|
| COLMAP | Arbitrary, set by first image pair | +X | -Y (typical) | +Z |
| Nerfstudio c2w | COLMAP-inherited | +X | +Y | -Z (OpenGL convention) |
| OpenCV / NeRF camera | Camera-local | +X (right) | -Y (down) | +Z (into scene) |

[ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) uses Nerfstudio's c2w convention directly: it extracts `c2w[:3, 3]` as origin (camera position in world frame) and `c2w[:3, :3]` as the rotation matrix. The ray direction in camera space is constructed as `[dir_x, dir_y, 1.0]` where `dir_z = 1.0` represents pointing along the camera's +Z axis. This is consistent with OpenCV's camera convention (looking down +Z).

Nerfstudio internally converts COLMAP's coordinate convention to OpenGL convention during `ns-process-data`, so the `transforms.json` c2w matrices use OpenGL's conventions (Y-up, Z-backward). The [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) code assumes the convention of the `transforms.json` as-is, which is correct for Nerfstudio outputs.

### 9.2 Memory and Performance Management

**WebGL budget (Phase 9 / App):** The primary memory constraint is browser WebGL. The [load_plotly_pcd](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/app.py#67-136) function explicitly caps per-class sphere points at 800 (vs Phase 7's 5000) to keep Plotly's WebGL renderer within browser limits. The function comment: *"preventing WebGL browser crashes from the 1.5GB NeRF PCD!"*

**GPU memory management (Phase 3):** For nerfacto (non-high-res), `train-num-rays-per-batch=2048` and `eval-num-rays-per-chunk=2048` are explicit low-VRAM safety defaults. Standard Nerfstudio uses 4096–16384 rays per batch; 2048 approximately halves peak VRAM consumption.

**No parallel processing across phases:** All phases are single-threaded Python processes. Parallelization (e.g., multi-worker COLMAP feature extraction, batched SAM inference) is not implemented.

### 9.3 Windows-Specific Patches

**Three distinct patches are applied:**

1. **WinError 87 in `mediapy`** ([render_video.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py), [export_mesh.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/export_mesh.py)): Intercepts `subprocess.Popen` calls with `env={}` and replaces with `env=None`. Root cause: Windows `CreateProcess` API rejects empty environment blocks. Fix: detect `k.get('env') == {}` predicate at point of call.

2. **PyTorch 2.6 `weights_only`** (all evaluation scripts): Monkeypatches `torch.load` to force `weights_only=False`. Applied via inline `-c` bootstrap string to avoid modifying Nerfstudio source.

3. **Stdout encoding** ([render_video.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/render_video.py), [export_mesh.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/evaluation/export_mesh.py)): `sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None` — ensures UTF-8 encoding on Windows console, where the default `cp1252` encoding causes UnicodeDecodeError when Nerfstudio prints progress bars with Unicode box-drawing characters (e.g., `█`, `─`).

### 9.4 LLM Provider Agnosticism — Design Rationale

The flat function-per-provider design allows independent evolution of each provider's client code (different API versions, auth mechanisms, response schemas) without breaking the others. The cost is code duplication in the system prompt text. An alternative — a base class with [generate(context, query)](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase7/semantic_pointcloud.py#6-95) abstract method — would reduce duplication but add indirection. The flat design was chosen for simplicity and debuggability, appropriate for a research prototype.

Gemini was adopted as the default provider for cost: the dynamically-selected Flash model (`gemini-1.5-flash` or `gemini-2.0-flash` depending on availability) has a free tier with generous rate limits; GPT-4o requires paid credits. The HuggingFace path (hardcoded to `mistralai/Mistral-7B-Instruct-v0.2` via the API URL `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2`) requires no API key, making it the fallback for zero-cost operation at the cost of lower arithmetic reasoning reliability.

---

## 10. Design Decisions and Tradeoffs

### 10.1 Splatfacto vs. Nerfacto

| Dimension | Nerfacto | Splatfacto (3DGS) |
|---|---|---|
| Primitive | Continuous density MLP | Explicit 3D Gaussians |
| Rendering | Ray marching (volume rendering integral) | Differentiable rasterization (alpha compositing) |
| Training speed | ~2–6 hours (GPU) | ~20–40 min (GPU) |
| Inference speed | Seconds per frame | 30–120 FPS (real-time) |
| High-freq detail | Blurry (oversmoothed) | Sharp edges, better texture |
| Memory | Lower (MLP params ~100MB) | Higher (millions of Gaussians) |
| COLMAP dependency | Dense point cloud optional | Required — Gaussians initialized from sparse points |

[run_nerf.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase3/run_nerf.py) selects splatfacto when `--high_res` is passed, nerfacto otherwise. The system defaults to nerfacto for robustness (COLMAP failures produce fewer points, harming Gaussian initialization).

### 10.2 Centroid vs. Volumetric Object Representation

Centroid-based representation was chosen for: (a) simplicity — it collapses the high-dimensional mask geometry to a single 3-vector; (b) LLM compatibility — coordinate-based spatial relationships are directly expressible in text; (c) computational tractability — no volumetric mask aggregation is needed. The primary cost is loss of shape, size, orientation, and multi-instance discriminability (see Section 11).

### 10.3 Streamlit vs. Custom Web Framework

Streamlit provides interactive chat (`st.chat_input`), caching (`@st.cache_data`), file download buttons, image display, and subprocess management with ~300 lines of Python. A custom React/Flask stack would achieve better performance (e.g., incremental streaming of LLM responses, WebSocket-native NeRF viewer embedding) but at 10–50× implementation cost. For a research demonstration prototype, Streamlit's rapid iteration advantage outweighs its performance constraints.

---

## 11. Limitations and Open Problems

### 11.1 Ray Projection Accuracy

The heuristic depth `simulated_depth = 2.0 + N(0, 0.5)` introduces systematic centroid displacement that scales with `|actual_depth - 2.0|` meters. In a typical indoor scene with object depths ranging from 1m to 5m, centroid errors of 0.5–3.0m are expected. This makes absolute position queries ("Is the chair within 1 meter of the door?") unreliable; relative directional queries ("Is the TV to the left of the sofa?") are more robust providing the bias is consistent across objects at similar depths.

### 11.2 Single-Instance-Per-Class Limitation

`class_aggregates[cls_name].append(pt_3d)` accumulates all detections of a class into a single list. Two chairs on opposite sides of a room produce a centroid in the empty middle of the room. The LLM cannot distinguish between one chair and two chairs from the current schema — `observations: 22` conflates viewing the same chair 22 times with viewing two chairs 11 times each.

### 11.3 LLM Reasoning Bounds

The LLM receives raw coordinates and is asked to perform: (1) arithmetic subtraction, (2) direction inference, (3) proximity estimation. GPT-4o and Gemini 1.5 Flash perform this reliably for simple pairwise queries. However, for complex multi-object queries ("What is the most crowded corner of the room?"), the model must reason over all pairwise distances simultaneously — a quadratic comparison that degrades for larger scene graphs. Additionally, the model has no knowledge of coordinate axis semantics (which is X, Y, Z in room terms), which can produce semantically incorrect directional descriptions.

### 11.4 NeRF Quality Dependencies

NeRF training quality degrades with: insufficient frame diversity (limited camera baseline → poor depth estimation), ambiguous lighting (specular reflections cause view-dependent artifacts), textureless regions (blank walls → ill-conditioned photometric loss), and small training sets (<100 frames for large rooms). COLMAP may fail to register all frames in scenes with poor texture, producing gaps in camera coverage that manifest as blurry or missing scene regions.

---

## 12. Summary and Key Contributions

### 12.1 Technical Contributions

1. **End-to-end monocular video → 3D semantic scene graph pipeline:** A nine-phase system that unifies classical SfM (COLMAP), neural rendering (Nerfstudio/3DGS), vision foundation models (YOLOv8, SAM), and LLM spatial reasoning in a single codebase, without requiring multi-camera rigs or depth sensors.

2. **Multi-view semantic-geometric lifting:** The ray projection module in [ray_projection.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase4/ray_projection.py) implements a novel aggregation strategy: for each class, per-frame camera-space rays (derived from COLMAP-estimated extrinsics and YOLO/SAM bounding box centroids) are cast to a simulated depth, then averaged across all views. Despite the heuristic depth approximation, multi-view averaging provides meaningful centroid estimates for qualitative spatial reasoning.

3. **Provider-agnostic LLM spatial reasoning:** [spatial_agent.py](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/src/phase5/spatial_agent.py) demonstrates that raw centroid coordinates can serve as sufficient context for LLMs to answer qualitative spatial relationship queries ("to the left of," "closer to the door"), without requiring explicit spatial preprocessing or graph embedding.

4. **Windows-compatible NeRF toolchain:** Three surgical runtime patches (WinError 87 env monkeypatch, PyTorch `weights_only=False` override, stdout UTF-8 reconfiguration) make the full Nerfstudio evaluation and rendering pipeline operational on Windows — an environment with historically poor support in the NeRF community.

5. **Memory-efficient interactive 3D dashboard:** The Streamlit app's `@st.cache_data` + point-count reduction strategy (800 vs 5000 Plotly sphere points) enables browser-friendly interactive 3D visualization of semantic scene graphs without WebGL memory crashes, while preserving the semantic point cloud at full density in the exported [.ply](file:///c:/Users/Sriman%20Rakshan%20N/Documents/Amrita/Project%20%28Self%29/Spatialintel_new/outputs/semantic_pcd.ply) file.

### 12.2 Differentiation from Prior Work

Prior NeRF + semantic labeling systems (SemanticNeRF, Panoptic Lifting, LERF) typically require: (a) real-time feature distillation within the NeRF training loop, (b) per-point semantic labels rather than class-level centroids, or (c) multi-view consistency enforcement for masks. This system instead performs **post-hoc 2D-to-3D lifting** using frozen pre-trained detectors and a pre-trained NeRF — the semantic pipeline runs independently of training. This decoupling enables: faster iteration on the semantic representation without retraining the NeRF; freedom to swap the detection model (YOLOv8n → YOLOv8x, or Grounding DINO) without modifying Phase 3; and direct compatibility with any COLMAP-based NeRF system. The tradeoff is lower 3D localization accuracy compared to systems that distill feature fields directly into the NeRF volume.
