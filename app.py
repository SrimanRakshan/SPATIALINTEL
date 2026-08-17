import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# --- shared utils ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.append("src/phase5")

from spatial_agent import SpatialAgent, format_scene_context, load_scene_graph_3d, query_gemini
from utils.subprocess_runner import build_nerf_runner
from utils.logger import get_logger
from utils.validators import validate_scene_graph_3d, ValidationError

logger = get_logger("app")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spatial Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("outputs")
SCENE_GRAPH_PATH = Path("data/scenes/scene_001/semantic_3d/scene_graph_3d.json")
TOPOLOGY_GRAPH_PATH = BASE_DIR / "topology_graph.png"
PCD_PATH = BASE_DIR / "semantic_pcd.ply"
VIDEO_PATH = BASE_DIR / "scene_001_final_render_interpolate.mp4"
MESH_PATH = BASE_DIR / "mesh" / "tsdf_mesh.ply"
MAX_POINTS = 50_000

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load_scene_graph(path: str) -> dict:
    """Parse + validate scene_graph_3d.json once; cached across all reruns."""
    data = load_scene_graph_3d(path)
    try:
        validate_scene_graph_3d(data, path=path)
    except ValidationError as e:
        st.warning(f"Scene graph schema warning: {e}")
    return data


@st.cache_data(show_spinner=False)
def cached_scene_context(path: str) -> str:
    """Pre-build LLM context string once; cached."""
    graph = cached_load_scene_graph(path)
    return format_scene_context(graph, top_k=50)


@st.cache_resource(show_spinner=False)
def cached_gemini_agent(api_key: str) -> "SpatialAgent | None":
    """Initialise SpatialAgent once per session (heavy: loads model + scene graph)."""
    if not SCENE_GRAPH_PATH.exists():
        return None
    try:
        return SpatialAgent(str(SCENE_GRAPH_PATH), provider="gemini", api_key=api_key)
    except Exception as e:
        logger.warning(f"Could not init SpatialAgent: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_plotly_pcd(scene_graph_file: str) -> "go.Figure | None":
    """
    Reconstructs a browser-friendly semantic sphere visualisation.
    Downsampled to MAX_POINTS to prevent WebGL OOM.
    """
    import json
    with open(scene_graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    if not objects:
        return None

    traces = []
    np.random.seed(42)
    pts_per_obj = max(50, MAX_POINTS // len(objects))

    for obj in objects:
        name = obj["name"]
        centroid = np.array(obj["position"])
        obs_count = obj["observations"]
        color = "rgb({},{},{})".format(*np.random.randint(50, 255, 3))
        num_points = min(pts_per_obj, 200 + (obs_count * 20))
        radius = min(1.0, 0.2 + (np.log1p(obs_count) * 0.1))

        phi = np.random.uniform(0, 2 * np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        u = np.random.uniform(0, 1, num_points)
        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = centroid[0] + r * np.sin(theta) * np.cos(phi)
        y = centroid[1] + r * np.sin(theta) * np.sin(phi)
        z = centroid[2] + r * np.cos(theta)

        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=2, color=color, opacity=0.8),
            name=f"{name} (Obs: {obs_count})",
        ))
        traces.append(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2] + radius + 0.1],
            mode="text", text=[name], textposition="top center",
            textfont=dict(size=14, color="white"), showlegend=False,
        ))

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Depth (Z)",
            bgcolor="rgb(20, 24, 30)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0, y=1, font=dict(color="white")),
    )
    return go.Figure(data=traces, layout=layout)


# ---------------------------------------------------------------------------
# Viewer lifecycle
# ---------------------------------------------------------------------------

def _kill_viewer() -> None:
    proc = st.session_state.get("viewer_proc")
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        del st.session_state["viewer_proc"]


def start_nerf_viewer() -> None:
    # Clean up zombie from previous restart
    proc = st.session_state.get("viewer_proc")
    if proc is not None and proc.poll() is not None:
        del st.session_state["viewer_proc"]

    if "viewer_proc" not in st.session_state:
        env = os.environ.copy()
        env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

        runner = build_nerf_runner(
            "from nerfstudio.scripts.viewer.run_viewer import entrypoint",
            include_mediapy=False,
        )
        runner = runner.replace(
            "sys.argv=sys.argv[1:]; entrypoint()",
            (
                "sys.argv=['ns-viewer', "
                "'--load-config', 'outputs/nerfstudio/nerfacto/2026-03-04_021356/config.yml', "
                "'--viewer.websocket-port', '7007']; "
                "entrypoint()"
            ),
        )

        proc = subprocess.Popen(["python", "-c", runner], env=env)
        st.session_state.viewer_proc = proc

        progress_text = "Booting PyTorch Neural Engine. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        for pct in range(100):
            time.sleep(0.15)
            my_bar.progress(pct + 1, text=progress_text)
        my_bar.empty()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌌 Spatial Intelligence Dashboard")
st.markdown(
    "Welcome to the end-to-end Monocular Video 3D Scene Analysis platform. "
    "This dashboard visualizes the processed results spanning from COLMAP NeRF "
    "reconstruction and SAM Semantic Mapping to GPT-4/Gemini level Spatial Reasoning."
)

with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    st.markdown("**Active Scene:** `scene_001`")
    provider = st.selectbox("LLM Reasoning Provider", ("Gemini", "OpenAI", "HuggingFace (Free)"))
    api_key = st.text_input(
        "Enter API Key (Google AI Studio / OpenAI)", type="password",
        value="AIzaSyAqjW8pEIqdYgzrFGwopNg7X2ceEXV3QIw",
    )
    st.markdown("---")
    st.markdown("### Process Status")
    st.success("✅ Frame Extraction & NeRF Training")
    st.success("✅ YOLOv8 + Segment Anything (SAM)")
    st.success("✅ 3D Ray Projection / Centroids")
    st.success("✅ Relational Topology & PCD Export")

# Tab structure UNCHANGED — 4 tabs, same names, same order
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 3D Semantic Point Cloud",
    "🕸️ Topological Network Layout",
    "💬 LLM Spatial Chat Agent",
    "🔭 Interactive NeRF Viewer",
])

# --- Tab 1 ---
with tab1:
    st.header("Interactive 3D Semantic Scene")
    st.markdown("Pan, zoom, and orbit the 3D space. Objects are positioned at their exactly inferred geometric bounds.")
    with st.spinner("Rendering Browser-Friendly 3D Visualizer..."):
        if SCENE_GRAPH_PATH.exists():
            fig = load_plotly_pcd(str(SCENE_GRAPH_PATH))
            if fig:
                st.plotly_chart(fig, use_container_width=True, height=600)
            else:
                st.info("No objects detected in the scene graph to render.")
        else:
            st.error(f"Scene graph not found at {SCENE_GRAPH_PATH}. Ensure Phase 4 & 5 ran successfully.")

    st.markdown("---")
    st.subheader("360° NeRF Rendering (Interpolated Trajectory)")
    if VIDEO_PATH.exists():
        st.video(str(VIDEO_PATH))
    else:
        st.info("FFMPEG was not installed during NeRF render phase, so the final .mp4 video is unavailable.")

    st.markdown("---")
    st.subheader("Download Dense 3D Mesh")
    st.markdown("Automatically generated **TSDF 3D Mesh** from the trained Neural Radiance Fields. Import into Blender, Unity, or Maya.")
    if MESH_PATH.exists():
        with open(MESH_PATH, "rb") as file:
            st.download_button(
                label="📦 Download 3D Mesh (.ply format)",
                data=file, file_name="tsdf_mesh.ply",
                mime="application/octet-stream", use_container_width=True,
            )
    else:
        st.info("3D Mesh not found. Wait for Phase 6 TSDF exporter to finish.")

# --- Tab 2 ---
with tab2:
    st.header("Topological Inference Map")
    st.markdown("A 2D mathematical reduction of the Scene Graph demonstrating Euclidean object cluster proximities. Edges define distances in meters.")
    if TOPOLOGY_GRAPH_PATH.exists():
        img = Image.open(TOPOLOGY_GRAPH_PATH)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Feature Map {TOPOLOGY_GRAPH_PATH} is missing. Please run Phase 8 first.")

# --- Tab 3 ---
with tab3:
    st.header(f"💬 Chat with the Environment ({provider})")
    st.markdown("Ask the multimodal agent complex spatial questions about the reconstructed scene.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Where is the TV located relative to the sofa?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing Spatial Geometry via {provider}..."):
                if not SCENE_GRAPH_PATH.exists():
                    st.error("Scene Graph Missing.")
                    st.stop()

                # [PERF] Cached — not re-parsed per message
                context = cached_scene_context(str(SCENE_GRAPH_PATH))

                if provider == "Gemini":
                    response = query_gemini(context, prompt, api_key)
                else:
                    response = (
                        "API Support for {} via Streamlit is incoming. "
                        "For now, please select 'Gemini'!"
                    ).format(provider)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Tab 4 ---
with tab4:
    st.header("🔭 Interactive NeRF Web Viewer")
    st.markdown("Launch the GPU-accelerated PyTorch rendering engine. **(Requires ~4GB VRAM)**")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🚀 Launch 3D Engine", use_container_width=True, key="btn_launch"):
            start_nerf_viewer()
    with col2:
        if st.button("⛔ Kill Viewer", use_container_width=True, key="btn_kill"):
            _kill_viewer()
            st.success("Viewer stopped.")

    proc = st.session_state.get("viewer_proc")
    if proc is not None:
        if proc.poll() is None:
            st.success("Neural Engine is running! If blank, wait a few more seconds for weight loading.")
            components.iframe("http://localhost:7007", height=800, scrolling=False)
        else:
            st.warning("Viewer process exited unexpectedly. Click Launch to restart.")
            del st.session_state["viewer_proc"]
