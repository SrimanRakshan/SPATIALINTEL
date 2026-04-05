import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def generate_topology_graph(scene_graph_path, output_path, distance_threshold=7.0):
    """
    Renders an academic-quality Network Topology graph linking 3D semantic objects
    based on their Euclidean proximity inside the scene mapping.
    """
    with open(scene_graph_path, 'r') as f:
        data = json.load(f)
        
    objects = data.get("objects", [])
    if not objects:
        print("Empty scene graph. Nothing to visualize.")
        return
        
    names = [obj['name'] for obj in objects]
    positions = np.array([obj['position'] for obj in objects])
    observations = [obj['observations'] for obj in objects]
    
    # Calculate all pair-wise Euclidean distances between all objects
    dist_matrix = squareform(pdist(positions))
    
    G = nx.Graph()
    
    # Base styling
    for i, name in enumerate(names):
        # Scale node size logarithmically with observations to signify structural confidence
        size = 800 + (np.log1p(observations[i]) * 400)
        G.add_node(i, label=name, size=size)
        
    # Draw connections
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            dist = dist_matrix[i, j]
            if dist < distance_threshold:
                # Stronger visual weight for spatially closer objects
                weight = max(0.5, 3.0 - (dist / (distance_threshold / 3.0)))
                G.add_edge(i, j, weight=weight, distance=dist)
                
    plt.figure(figsize=(12, 10), facecolor='white')
    
    # Kamada-Kawai layout mimics real-world spatial geometries extremely well for networks
    pos = nx.kamada_kawai_layout(G)
    
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    
    # Color palette
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color='#3498db', 
        alpha=0.8, 
        edgecolors='#2980b9', 
        linewidths=2.5
    )
    
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2.5 for u, v in edges]
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges, 
        width=weights, 
        edge_color='#bdc3c7', 
        alpha=0.7,
        connectionstyle="arc3,rad=0.1" # slight curve adds organic feel
    )
    
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=11, 
        font_family='sans-serif', 
        font_weight='bold',
        font_color='#2c3e50'
    )
    
    # Overlay distance metrics selectively to maintain cleanly structured paper layout
    edge_labels = {(u, v): f"{G[u][v]['distance']:.1f}m" for u, v in edges if G[u][v]['distance'] < (distance_threshold * 0.4)}
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_size=9, 
        font_color='#e74c3c',
        label_pos=0.45,
        bbox=dict(boxstyle="round4,pad=0.3", fc="white", alpha=0.9, ec="white")
    )
    
    plt.title("Phase 8: 3D Topological Scene Inference Network", fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Publication-quality topological graph saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D Topological Relational Graph from 3D Scene Map")
    parser.add_argument("--scene_graph", required=True, help="Path to scene_graph_3d.json")
    parser.add_argument("--output", required=True, help="Output path for the PNG image (e.g. outputs/topology_graph.png)")
    
    args = parser.parse_args()
    generate_topology_graph(args.scene_graph, args.output)
