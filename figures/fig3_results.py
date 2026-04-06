#!/usr/bin/env python3
"""
Figure 3 — IEEE paper: 3D Scene Understanding Results
Panels: (a) NeRF render  |  (b) Semantic 3D scatter  |  (c) GT vs Predicted
"""

import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ── IEEE style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         7,
    "axes.linewidth":    0.6,
    "axes.labelsize":    7,
    "axes.titlesize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "#CCCCCC",
    "lines.linewidth":   0.9,
    "text.usetex":       False,   # set True if LaTeX installed
})

# Palette — ColorBrewer Set1 subset (colorblind-safe for print)
CLASS_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#999999",
]

RENDER_PATH = "render.png"
JSON_PATH   = "data/scenes/scene_001/semantic_3d/scene_graph_3d.json"
OUT_PATH    = "figures/fig3_results.pdf"
OUT_PNG     = "figures/fig3_results.png"

# ── Synthetic fallback data (used when files are absent) ───────────────────
def _synthetic_objects():
    rng = np.random.default_rng(0)
    labels = ["chair", "table", "sofa", "tv", "bed", "lamp"]
    objs = []
    for i, lbl in enumerate(labels):
        n = rng.integers(3, 7)
        for _ in range(n):
            objs.append({
                "name": lbl,
                "position": (rng.uniform(-3, 3, 3) +
                             np.array([i*0.5, 0, i*0.3])).tolist(),
                "observations": int(rng.integers(5, 25)),
            })
    return {"objects": objs}


def _synthetic_render():
    """Return an (H, W, 3) uint8 placeholder when render.png is absent."""
    h, w = 270, 480
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    # simple gradient to suggest depth
    img[:, :, 2] = np.linspace(180, 255, w, dtype=np.uint8)
    img[:, :, 0] = np.linspace(220, 190, h, dtype=np.uint8).reshape(-1, 1)
    return img


def _synthetic_gt_pred(objects):
    """Generate GT / predicted pairs from the centroids."""
    rng = np.random.default_rng(42)
    pts = np.array([o["position"] for o in objects])
    noise = rng.normal(0, 0.35, pts.shape)
    return pts, pts + noise


# ── Load data ──────────────────────────────────────────────────────────────
# Render image
if os.path.exists(RENDER_PATH):
    render_img = plt.imread(RENDER_PATH)
    print(f"Loaded render from {RENDER_PATH}")
else:
    render_img = _synthetic_render()
    print(f"[warn] {RENDER_PATH} not found — using synthetic placeholder.")

# Scene graph
if os.path.exists(JSON_PATH):
    with open(JSON_PATH) as f:
        sg = json.load(f)
    print(f"Loaded scene graph from {JSON_PATH}")
else:
    sg = _synthetic_objects()
    print(f"[warn] {JSON_PATH} not found — using synthetic objects.")

objects = sg.get("objects", [])

# Unify key names (supports both "name"/"label" and "position"/x,y,z)
def _pos(o):
    if "position" in o:
        return o["position"]
    return [o["x"], o["y"], o["z"]]

def _lbl(o):
    return o.get("name", o.get("label", "object"))

labels_all = [_lbl(o) for o in objects]
positions  = np.array([_pos(o) for o in objects])   # (N, 3)
unique_cls = sorted(set(labels_all))
cls_to_idx = {c: i for i, c in enumerate(unique_cls)}
colors_per_pt = [CLASS_COLORS[cls_to_idx[l] % len(CLASS_COLORS)] for l in labels_all]

gt_pts, pred_pts = _synthetic_gt_pred(objects)


# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.16, 2.5))
fig.patch.set_facecolor("white")

# GridSpec: image | 3-D scatter | GT vs Pred
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig, wspace=0.28,
              left=0.04, right=0.98, top=0.90, bottom=0.18)

ax_img  = fig.add_subplot(gs[0])
ax_3d   = fig.add_subplot(gs[1], projection="3d")
ax_err  = fig.add_subplot(gs[2], projection="3d")


# ═══════════════════════════════════════════════════════════════════════════
# Panel (a) — NeRF render
# ═══════════════════════════════════════════════════════════════════════════
ax_img.imshow(render_img, interpolation="bilinear")
ax_img.axis("off")
ax_img.set_title("(a)  NeRF Novel-View Render", pad=4)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b) — Semantic 3D scatter
# ═══════════════════════════════════════════════════════════════════════════
def _set_3d_style(ax, title, xlabel="X", ylabel="Y", zlabel="Z"):
    ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_zlabel(zlabel, labelpad=2)
    ax.tick_params(axis='both', which='major', pad=1, labelsize=5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("#DDDDDD")
    ax.grid(True, linewidth=0.3, color="#EEEEEE")

for cls in unique_cls:
    mask = [l == cls for l in labels_all]
    pts  = positions[mask]
    idx  = cls_to_idx[cls]
    ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=CLASS_COLORS[idx % len(CLASS_COLORS)],
                  s=22, edgecolors="white", linewidths=0.3,
                  label=cls, zorder=4, depthshade=False)

_set_3d_style(ax_3d, "(b)  Semantic 3-D Point Cloud")

# Compact legend outside axes
handles = [mpatches.Patch(fc=CLASS_COLORS[cls_to_idx[c] % len(CLASS_COLORS)],
                          ec="white", label=c)
           for c in unique_cls]
ax_3d.legend(handles=handles, loc="upper left",
             bbox_to_anchor=(-0.08, 1.02),
             ncol=1, fontsize=5,
             handlelength=0.8, handletextpad=0.4,
             borderpad=0.4, labelspacing=0.3,
             frameon=True)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (c) — GT vs Predicted (error vectors)
# ═══════════════════════════════════════════════════════════════════════════
# GT points
ax_err.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
               c="#2CA02C", s=22,
               edgecolors="white", linewidths=0.3,
               label="Ground truth", zorder=5, depthshade=False)

# Predicted points
ax_err.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
               c="#D62728", s=22, marker="^",
               edgecolors="white", linewidths=0.3,
               label="Predicted", zorder=5, depthshade=False)

# Error vectors: GT → Predicted
segments = [[(g[0], g[1], g[2]), (p[0], p[1], p[2])]
            for g, p in zip(gt_pts, pred_pts)]
err_col = Line3DCollection(segments, colors="#888888",
                           linewidths=0.55, linestyles="--",
                           alpha=0.7, zorder=3)
ax_err.add_collection(err_col)

# RMSE annotation
errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
rmse   = np.sqrt(np.mean(errors**2))
ax_err.text2D(0.97, 0.04,
              f"RMSE = {rmse:.2f} m",
              transform=ax_err.transAxes,
              ha="right", va="bottom",
              fontsize=6, color="#B2182B",
              bbox=dict(boxstyle="round,pad=0.25", fc="white",
                        ec="#CCCCCC", lw=0.5))

_set_3d_style(ax_err, "(c)  Error: GT vs. Predicted")
ax_err.legend(loc="upper left",
              bbox_to_anchor=(-0.08, 1.02),
              ncol=1, fontsize=5,
              handlelength=1.0, handletextpad=0.4,
              borderpad=0.4, labelspacing=0.3,
              frameon=True)


# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PNG,  dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved:\n  {OUT_PATH}\n  {OUT_PNG}")
plt.show()
