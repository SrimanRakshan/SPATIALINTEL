import argparse
import json
import numpy as np
from pathlib import Path

def generate_semantic_pcd(scene_graph_path: str, output_path: str):
    """
    Generates an interactive 3D Semantic Point Cloud from our computed Ray Projections.
    It takes the 3D centroids of every detected object and expands them into semantic 
    point cloud spheres, color-coded by class, for beautiful interactive viewing.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: Open3D is required. Please install via: pip install open3d")
        exit(1)
        
    scene_graph_path = Path(scene_graph_path)
    output_path = Path(output_path)
    
    if not scene_graph_path.exists():
        print(f"Error: Missing 3D Scene Graph at {scene_graph_path}")
        exit(1)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(scene_graph_path, 'r') as f:
        data = json.load(f)
        
    objects = data.get("objects", [])
    if not objects:
        print("Empty scene graph.")
        return
        
    print(f"\nPhase 7: Generating Semantic 3D Point Cloud for {len(objects)} detected objects...")
    
    # Store aggregated points and colors
    all_points = []
    all_colors = []
    
    # Generate deterministic vibrant colors for each class
    np.random.seed(42)
    class_colors = {}
    
    for obj in objects:
        name = obj["name"]
        centroid = np.array(obj["position"])
        obs_count = obj["observations"]
        
        if name not in class_colors:
            class_colors[name] = np.random.rand(3)
            
        color = class_colors[name]
        
        # Scale the size/density of the semantic sphere based on how many times
        # the object was observed (higher confidence = more dense point cloud)
        num_points_in_sphere = min(5000, 500 + (obs_count * 100))
        radius = min(1.0, 0.2 + (np.log1p(obs_count) * 0.1))
        
        # Generate points uniformly within a sphere
        phi = np.random.uniform(0, 2*np.pi, num_points_in_sphere)
        costheta = np.random.uniform(-1, 1, num_points_in_sphere)
        u = np.random.uniform(0, 1, num_points_in_sphere)

        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = centroid[0] + r * np.sin(theta) * np.cos(phi)
        y = centroid[1] + r * np.sin(theta) * np.sin(phi)
        z = centroid[2] + r * np.cos(theta)
        
        sphere_points = np.vstack((x, y, z)).T
        sphere_colors = np.tile(color, (num_points_in_sphere, 1))
        
        all_points.append(sphere_points)
        all_colors.append(sphere_colors)
        
    # Combine all geometries into a single Point Cloud
    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    # Estimate normals for better shading when viewed in Meshlab/Open3D
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=30))
    
    o3d.io.write_point_cloud(str(output_path), pcd)
    
    print("\n✅ Semantic Point Cloud Successfully Generated!")
    print(f"Total semantic vertices: {len(final_points)}")
    print(f"Saved to: {output_path}")
    print("Open this .ply file in Meshlab or Open3D to explore the explicit 3D semantic layout of your scene!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an interactive Semantic Point Cloud from a 3D Scene Graph.")
    parser.add_argument("--scene_graph", required=True, help="Path to scene_graph_3d.json")
    parser.add_argument("--output", required=True, help="Path to save the output PLY file")
    
    args = parser.parse_args()
    generate_semantic_pcd(args.scene_graph, args.output)
