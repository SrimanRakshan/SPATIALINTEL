# 🌌 Spatial Intelligence: End-to-End Pipeline Overview

This document serves as a comprehensive summary of the Spatial Intelligence project. It outlines the complete workflow we have built to transform raw monocular video into an interactive, 3D semantic scene equipped with Large Language Model (LLM) spatial reasoning capabilities.

## 🎯 Project Objective
The primary goal of this project was to establish an automated pipeline capable of understanding physical spaces from a simple video sweep. By combining cutting-edge NeRF (Neural Radiance Fields) technology with Foundation Models for computer vision (YOLOv8, Segment Anything) and generative AI (Gemini/GPT-4), we created a system that not only reconstructs a room in 3D but actually *understands* the objects within it, their geometric boundaries, and their spatial relationships.

---

## 🏗️ Pipeline Phases & Accomplishments

### Phase 1: Video to Multi-View Dataset (`src/phase1/`)
- **Action:** Processed a raw input video (`input.mp4`) of a scene.
- **Details:** 
  - `extract_frames.py`: Extracted individual high-quality frames from the video.
  - `subsample.py`: Filtered out blurry or redundant frames using Laplacian variance and Structural Similarity Index (SSIM), producing an optimized set of sharp images.
  - `preprocess.py`: Standardized the resolution and formats to prepare the imagery for geometric reconstruction.

### Phase 2: Structure From Motion (`src/phase2/`)
- **Action:** Extracted sparse 3D geometry and camera poses.
- **Details:** 
  - `run_colmap.py`: Automated the execution of COLMAP. This extracted feature points across all images, matched them, and performed bundle adjustment to calculate exactly where the camera was located in 3D space for every single frame.

### Phase 3: Neural Radiance Fields (NeRF) Training (`src/phase3/`)
- **Action:** Trained a dense, photorealistic 3D representation of the room.
- **Details:** 
  - Formatted the COLMAP outputs into a structure readable by Nerfstudio (`ns-process-data`).
  - Added support for high-resolution 3D Gaussian Splatting (`splatfacto`) for superior visual fidelity over standard `nerfacto`.
  - Automated the GPU-accelerated training process natively through our `run_nerf.py` wrapper, overcoming various Windows memory and PyTorch multiprocessing hurdles.

### Phase 4: Semantic Scene Understanding (`src/phase4/`)
- **Action:** Identified and localized objects within the 3D space.
- **Details:** 
  - **YOLOv8** (`semantic_mapping.py`): Scanned the original video frames to detect common objects (e.g., bed, refrigerator, person, cup) as 2D bounding boxes.
  - **Segment Anything Model (SAM)** (`sam_segmentation.py`): Used Meta's Vision Transformer to extract pixel-perfect segmentation masks for every YOLO bounding box, isolating the exact shapes of the objects.
  - **3D Ray Projection** (`ray_projection.py`): Fired mathematical rays from the COLMAP camera origins through the 2D semantic masks intersecting with the NeRF depth fields. This calculated the absolute 3D Cartesian coordinates (Centroids) for every recognized object, compiling them into a `scene_graph_3d.json`.

### Phase 5: LLM-Based Spatial Reasoning (`src/phase5/`)
- **Action:** Gave the pipeline a "brain" to reason about the scene geometry.
- **Details:** 
  - `spatial_agent.py`: Developed a provider-agnostic LLM interface (supporting Google Gemini, OpenAI GPT-4, and HuggingFace).
  - The script parses the 3D Scene Graph mapping out the X, Y, Z coordinates and observation frequencies of all objects, and constructs a geometric context prompt.
  - Users can ask complex spatial questions like *"Where is the TV relative to the bed?"* and the LLM accurately answers based on the mathematically inferred relationships.

### Phase 6: Evaluation & Export (`src/evaluation/`)
- **Action:** Validated the model quality and exported native 3D assets.
- **Details:** 
  - `evaluate_nerf.py`: Automatically calculates academic visual metrics (PSNR, SSIM, LPIPS) for the trained NeRF.
  - `render_video.py`: Overcame Windows-specific `mediapy` and `ffmpeg` `WinError 87` bugs using surgical subprocess patches to render a beautiful, interpolated 360° camera flythrough video (`scene_001_final_render_interpolate.mp4`).
  - `export_mesh.py`: Converts the neural density fields into a standard, dense 3D TSDF geometry mesh (`tsdf_mesh.ply`), allowing the room to be imported into traditional 3D software like Blender.

### Phase 7 & 8: Academic Visualizations
- **Action:** Created distinct, specialized visualization assets for papers/presentations.
- **Details:** 
  - **Phase 7** (`semantic_pointcloud.py`): Generated a synthetic 3D point cloud (`semantic_pcd.ply`) using Open3D that represents each detected class as a colored sphere at its inferred 3D centroid geometry.
  - **Phase 8** (`graph_visualizer.py`): Used NetworkX to calculate the Euclidean matrices between all objects, collapsing the 3D space into a 2D Topological Network Feature Map (`topology_graph.png`) showing spatial adjacency.

### Phase 9: Interactive Presentation Web App (`app.py`)
- **Action:** Unified everything into a beautiful, interactive Streamlit Dashboard.
- **Details:** 
  - Developed a multi-tab web application combining the pipeline's outputs into a single presentation layer.
  - **Tab 1:** Hosts an interactive, browser-friendly WebGL plotting of the Semantic Point Cloud alongside the Interpolated 360° NeRF video and a button to download the dense TSDF `.ply` mesh.
  - **Tab 2:** Displays the 2D Topological Proximity Graph.
  - **Tab 3:** Provides a live chatbox directly connected to the Google Gemini agent, allowing live spatial Q&A queries against the environment.
  - **Tab 4:** Features a massive integration—a button that dynamically spawns the original PyTorch `ns-viewer` and embeds the GPU-computed live rendering engine directly into the website for real-time 3D flight!

---

## 🧰 Tools & Technologies
The pipeline leverages a robust stack of open-source libraries and advanced machine learning models:
- **3D Reconstruction & Photogrammetry:** COLMAP (Sparse Reconstruction), Nerfstudio (Dense NeRF/Splatfacto training).
- **Vision Foundation Models:** YOLOv8 (2D Object Detection), Meta Segment Anything Model (SAM) (Pixel-perfect Instance Segmentation).
- **Large Language Models (LLMs):** Google Gemini 1.5 Pro/Flash, OpenAI GPT-4, and HuggingFace Inference APIs for spatial reasoning.
- **Data Processing & Visualization:** OpenCV, NumPy, NetworkX (Topological graphing), Plotly (Interactive 3D Point Clouds), Open3D (Point cloud manipulation), and Streamlit (Interactive presentation dashboard).
- **Evaluation & Automation:** FFmpeg and mediapy (Video rendering), PyTorch (Tensor operations and neural field processing).

## 📊 Evaluation & Metrics
To ensure academic rigor and high-quality outputs, the pipeline performs comprehensive evaluations:
- **Quantitative Metrics (`evaluate_nerf.py`):** 
  - **PSNR (Peak Signal-to-Noise Ratio):** Measures the raw pixel-level reconstruction quality.
  - **SSIM (Structural Similarity Index):** Assesses the perceptual structural integrity of the rendered novel views compared to ground truth.
  - **LPIPS (Learned Perceptual Image Patch Similarity):** Uses deep neural network feature maps to measure continuous perceptual quality exactly how human visual cortex perceives it.
- **Qualitative Video Rendering (`render_video.py`):** Generates high-resolution continuous `interpolate` camera trajectory videos to verify temporal consistency and ensure the 3D scene lacks popping artifacts or "floaters".
- **Geometric Exporting (`export_mesh.py`):** Reconstructs the neural density fields into dense TSDF (Truncated Signed Distance Function) and Poisson surface meshes, verifying that the underlying 3D structures are manifold, dense, and structurally sound for downstream manipulation.

---

## 🛠️ Key Technical Challenges Solved
1. **Windows Subprocess Bugs:** Diagnosed and patched deep OS-level `CreateProcess` `WinError 87` bugs in the Python `mediapy` library regarding empty environment payload blocks to allow `ffmpeg` video rendering.
2. **Library Compatibility:** Handled deprecations across PyTorch 2.6 (`weights_only=True` monkeypatching) to allow loading of `nerfstudio` models effortlessly.
3. **API Constraints:** Swapped hard-coded OpenAI implementations for dynamic Google Gemini integration to provide a fully functional LLM reasoner on a free tier.
4. **WebGL Limits:** Built data-reduction visualizers (Phase 7) to prevent browser crashes when attempting to render massive 4GB NeRF point clouds natively in Streamlit, keeping the app fast and responsive.

## 🏁 Conclusion
The Spatial Intelligence repository is now a fully functional, end-to-end framework. It proves that by combining classical photogrammetry, modern neural radiance fields, and semantic foundation models, we can grant AI systems a true, geometric understanding of our physical world.
