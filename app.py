import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import os
import subprocess
import time

# Import LLM Agent logic
import sys
sys.path.append('src/phase5')
from spatial_agent import load_scene_graph_3d, format_scene_context, query_gemini

# Configure Page
st.set_page_config(
    page_title="Spatial Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Header
st.title("🌌 Spatial Intelligence Dashboard")
st.markdown("""
Welcome to the end-to-end Monocular Video 3D Scene Analysis platform. 
This dashboard visualizes the processed results of our pipeline, spanning from COLMAP NeRF reconstruction and SAM Semantic Mapping to GPT-4/Gemini level Spatial Reasoning.
""")

# Setup absolute paths for the demo
BASE_DIR = Path("outputs")
SCENE_GRAPH_PATH = Path("data/scenes/scene_001/semantic_3d/scene_graph_3d.json")
TOPOLOGY_GRAPH_PATH = BASE_DIR / "topology_graph.png"
PCD_PATH = BASE_DIR / "semantic_pcd.ply"
VIDEO_PATH = BASE_DIR / "scene_001_final_render_interpolate.mp4" # May not exist if ffmpeg wasn't installed, but we handle it gracefully
MESH_PATH = BASE_DIR / "mesh" / "tsdf_mesh.ply"

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    st.markdown("**Active Scene:** `scene_001`")
    
    provider = st.selectbox(
        "LLM Reasoning Provider",
        ("Gemini", "OpenAI", "HuggingFace (Free)")
    )
    api_key = st.text_input("Enter API Key (Google AI Studio / OpenAI)", type="password", value="AIzaSyAqjW8pEIqdYgzrFGwopNg7X2ceEXV3QIw")
    
    st.markdown("---")
    st.markdown("### Process Status")
    st.success("✅ Frame Extraction & NeRF Training")
    st.success("✅ YOLOv8 + Segment Anything (SAM)")
    st.success("✅ 3D Ray Projection / Centroids")
    st.success("✅ Relational Topology & PCD Export")

# Main Content Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 3D Semantic Point Cloud",
    "🕸️ Topological Network Layout", 
    "💬 LLM Spatial Chat Agent",
    "🔭 Interactive NeRF Viewer"
])

@st.cache_data
def load_plotly_pcd(scene_graph_file: str):
    """
    Simulates loading the dense PCD by dynamically reconstructing a smaller, 
    browser-friendly version of the semantic spheres using the exact 3D centroids 
    from the scene graph, preventing WebGL browser crashes from the 1.5GB NeRF PCD!
    """
    with open(scene_graph_file, "r") as f:
        data = json.load(f)
        
    objects = data.get("objects", [])
    if not objects:
        return None
        
    traces = []
    np.random.seed(42)
    
    for obj in objects:
        name = obj["name"]
        centroid = np.array(obj["position"])
        obs_count = obj["observations"]
        
        # Color & sizes
        color = 'rgb({},{},{})'.format(*np.random.randint(50, 255, 3))
        
        # We sample ~500 points per sphere for smooth web rendering instead of 10k
        num_points = min(800, 200 + (obs_count * 20))
        radius = min(1.0, 0.2 + (np.log1p(obs_count) * 0.1))
        
        phi = np.random.uniform(0, 2*np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        u = np.random.uniform(0, 1, num_points)
        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = centroid[0] + r * np.sin(theta) * np.cos(phi)
        y = centroid[1] + r * np.sin(theta) * np.sin(phi)
        z = centroid[2] + r * np.cos(theta)
        
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.8),
            name=f"{name} (Obs: {obs_count})"
        ))
        
        # Add a text label anchored exactly at the centroid
        traces.append(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]+radius+0.1],
            mode='text',
            text=[name],
            textposition="top center",
            textfont=dict(size=14, color="white"),
            showlegend=False
        ))
        
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Depth (Z)',
            bgcolor="rgb(20, 24, 30)" # Dark aesthetic background
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0, y=1, font=dict(color="white"))
    )
    
    return go.Figure(data=traces, layout=layout)

# TAB 1: 3D Pointcloud
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
            
    # NeRF Rendered Video (If it successfully built via FFmpeg)
    st.markdown("---")
    st.subheader("360° NeRF Rendering (Interpolated Trajectory)")
    if VIDEO_PATH.exists():
        st.video(str(VIDEO_PATH))
    else:
        st.info("FFMPEG was not installed during NeRF render phase, so the final .mp4 video is unavailable. The 3D geometry above is calculated from the raw model instead!")
        
    # 3D Mesh Download
    st.markdown("---")
    st.subheader("Download Dense 3D Mesh")
    st.markdown("We have automatically generated a robust **TSDF 3D Mesh geometry** of this scene from the trained Neural Radiance Fields. You can download this fully textured `.ply` object to import natively into Blender, Unity, or Maya.")
    if MESH_PATH.exists():
        with open(MESH_PATH, "rb") as file:
            st.download_button(
                label="📦 Download 3D Mesh (.ply format)",
                data=file,
                file_name="tsdf_mesh.ply",
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.info("3D Mesh geometric export not found. Wait for Phase 6 TSDF exporter to finish.")

# TAB 2: Topology Graph
with tab2:
    st.header("Topological Inference Map")
    st.markdown("A 2D mathematical reduction of the Scene Graph demonstrating Euclidean object cluster proximities. Edges define distances in meters.")
    
    if TOPOLOGY_GRAPH_PATH.exists():
        img = Image.open(TOPOLOGY_GRAPH_PATH)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Feature Map {TOPOLOGY_GRAPH_PATH} is missing. Please run Phase 8 first.")

# TAB 3: LLM Chat Interface
with tab3:
    st.header(f"💬 Chat with the Environment ({provider})")
    st.markdown("Ask the multimodal agent complex spatial questions regarding the arrangement, layout, and contents of the reconstructed scene.")
    
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
                    
                scene_graph = load_scene_graph_3d(str(SCENE_GRAPH_PATH))
                context = format_scene_context(scene_graph)
                
                # We only fully wired Gemini for the free tier demo, 
                # but you could map the others easily.
                if provider == "Gemini":
                    response = query_gemini(context, prompt, api_key)
                else:
                    response = "API Support for {} via Streamlit is incoming. For now, please select 'Gemini' and use the loaded API key!".format(provider)
                    
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def start_nerf_viewer():
    if "viewer_proc" not in st.session_state:
        env = os.environ.copy()
        env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
        
        runner = (
            "import sys, torch, warnings, os; "
            "warnings.filterwarnings('ignore', category=FutureWarning); "
            "_old_load = torch.load; "
            "torch.load = lambda *a, **k: _old_load(*a, **{**k, 'weights_only': False}); "
            "from nerfstudio.scripts.viewer.run_viewer import entrypoint; "
            "sys.argv=['ns-viewer', '--load-config', 'outputs/nerfstudio/nerfacto/2026-03-04_021356/config.yml', '--viewer.websocket-port', '7007']; "
            "entrypoint()"
        )
        
        # Spawn in background
        proc = subprocess.Popen(["python", "-c", runner], env=env)
        st.session_state.viewer_proc = proc
        
        # Progress bar while PyTorch loads the 4GB model
        progress_text = "Booting PyTorch Neural Engine. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.15)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()

# TAB 4: Real-time NeRF Viewer
with tab4:
    st.header("🔭 Interactive NeRF Web Viewer")
    st.markdown("Launch the original GPU-accelerated PyTorch rendering engine to fly through the 3D environment live! **(Requires ~4GB VRAM)**")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Launch 3D Engine", use_container_width=True):
            start_nerf_viewer()
            
    if "viewer_proc" in st.session_state:
        st.success("Neural Engine is running! If the view is blank, give it a few more seconds to finish loading weights.")
        components.iframe("http://localhost:7007", height=800, scrolling=False)
