#!/usr/bin/env python3
"""
Publication-quality IEEE figure: Ray Projection for 3D Semantic Lifting
Three panels:
  (a) 2D Detection -> Ray Initialization
  (b) Depth Ambiguity (Single View)
  (c) Multi-view Fusion
Output: ray_projection_figure.pdf  (vector) + .png (preview)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.lines import Line2D

# ── IEEE style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      7,
    "axes.linewidth": 0.5,
    "text.usetex":    False,   # flip True if LaTeX is installed
})

# Color palette (minimal, color-blind friendly)
C_RAY  = "#2166AC"   # blue   – rays
C_OBJ  = "#B2182B"   # red    – 3-D point
C_CAM  = "#4D4D4D"   # gray   – cameras
C_CONE = "#DEEBF7"   # pale blue – uncertainty
C_CONV = "#FDDBC7"   # pale red  – convergence zone

LW  = 1.0    # primary line width
LWS = 0.55   # secondary
MS  = 4.5    # marker size

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.22, wspace=0.14)

for ax in axes:
    ax.set_facecolor("white")
    ax.axis("off")


# ── Shared helper: cone polygon ───────────────────────────────────────────────
def cone_polygon(ox, oy, angle_rad, half_angle_rad, t_max, n=100):
    t = np.linspace(0, t_max, n)
    tx = ox + t * np.cos(angle_rad + half_angle_rad)
    ty = oy + t * np.sin(angle_rad + half_angle_rad)
    bx = ox + t * np.cos(angle_rad - half_angle_rad)
    by = oy + t * np.sin(angle_rad - half_angle_rad)
    verts = np.c_[np.concatenate([tx, bx[::-1]]),
                  np.concatenate([ty, by[::-1]])]
    return Polygon(verts, closed=True,
                   facecolor=C_CONE, edgecolor="none", alpha=0.80, zorder=2)


# ── Shared helper: camera icon ────────────────────────────────────────────────
def camera(ax, x, y, s=0.28):
    ax.add_patch(Rectangle((x - s/2, y - s/2), s, s,
                            fc=C_CAM, ec="white", lw=0.4, zorder=7))


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): 2D image plane
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(-2.8, 10)

# Image frame
ax.add_patch(Rectangle((0.8, 2.0), 8.4, 7.2,
                        lw=LWS, ec="#222222", fc="#F5F5F5", zorder=1))
ax.text(5.0, 9.5, "image plane", ha="center", va="bottom",
        fontsize=6, color="#333333")

# Bounding box (dashed)
ax.add_patch(Rectangle((3.0, 4.2), 4.0, 3.4,
                        lw=LWS, ec=C_CAM, fc="none",
                        linestyle="--", zorder=3))

# Centroid
cx, cy = 5.0, 5.9
ax.plot(cx, cy, "o", color=C_OBJ, ms=MS, zorder=6)
for dx, dy in [(-0.28, 0), (0.28, 0), (0, -0.28), (0, 0.28)]:
    ax.plot([cx, cx+dx], [cy, cy+dy], "-", color=C_OBJ, lw=0.5, zorder=5)
ax.text(cx + 0.45, cy + 0.45, r"$(c_x,\, c_y)$",
        fontsize=6.5, color=C_OBJ, va="bottom")

# Camera center
ox, oy = 3.5, 0.2
camera(ax, ox, oy)
ax.text(ox, oy - 0.55, "$O$", ha="center", va="top",
        fontsize=7, color=C_CAM, style="italic")

# Faint frustum guides
for ix in [0.8, 9.2]:
    ax.plot([ox, ix], [oy, 2.0], ":", color=C_CAM, lw=0.45, alpha=0.4, zorder=1)

# Ray through centroid (extended)
dx = cx - ox; dy = cy - oy
r = np.hypot(dx, dy); dx /= r; dy /= r
t_end = 11.0
ax.annotate("", xy=(ox + t_end*dx, oy + t_end*dy), xytext=(ox, oy),
            arrowprops=dict(arrowstyle="->", color=C_RAY, lw=LW,
                            mutation_scale=8, shrinkA=5, shrinkB=0), zorder=4)

ax.text(5.0, -2.5, "(a)  2D Detection \u2192 Ray Initialization",
        ha="center", va="bottom", fontsize=7, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Single-view depth ambiguity (side view)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(-2.8, 10)

ob = (0.8, 4.5)
camera(ax, *ob)
ax.text(ob[0], ob[1] - 0.55, "$O$", ha="center", va="top",
        fontsize=7, color=C_CAM, style="italic")

# Image plane (vertical line)
ax.plot([2.7, 2.7], [2.2, 6.8], color="#333333", lw=LWS, zorder=2)
ax.text(2.7, 7.1, "image\nplane", ha="center", va="bottom",
        fontsize=5.5, color="#333333")

# Outer frustum
for ey in [9.0, 0.1]:
    ax.plot([ob[0], 9.6], [ob[1], ey], "-", color=C_CAM, lw=0.4, alpha=0.3, zorder=1)

# Uncertainty cone
ang = np.radians(2.5)
ha  = np.radians(7.5)
ax.add_patch(cone_polygon(*ob, ang, ha, t_max=9.0))

t_vals = np.linspace(0, 9.0, 120)
for sign in [+1, -1]:
    ax.plot(ob[0] + t_vals * np.cos(ang + sign*ha),
            ob[1] + t_vals * np.sin(ang + sign*ha),
            "--", color=C_RAY, lw=0.5, alpha=0.7, zorder=3)

# Primary ray
dr = (np.cos(ang), np.sin(ang))
ax.annotate("", xy=(ob[0]+9.0*dr[0], ob[1]+9.0*dr[1]), xytext=ob,
            arrowprops=dict(arrowstyle="->", color=C_RAY, lw=LW,
                            mutation_scale=8, shrinkA=5, shrinkB=0), zorder=4)

# t* point
t_star = 5.7
pt_b = (ob[0] + t_star*dr[0], ob[1] + t_star*dr[1])
ax.plot(*pt_b, "o", color=C_OBJ, ms=MS, zorder=7)
ax.annotate(r"$P(t^*)$", xy=pt_b, xytext=(pt_b[0]-1.1, pt_b[1]+1.6),
            fontsize=6.5, color=C_OBJ,
            arrowprops=dict(arrowstyle="->", color=C_OBJ, lw=0.5,
                            mutation_scale=6), zorder=8)

# Depth bracket
bry = 1.2
ax.annotate("", xy=(pt_b[0], bry), xytext=(ob[0], bry),
            arrowprops=dict(arrowstyle="<->", color=C_CAM,
                            lw=0.6, mutation_scale=6))
ax.text((ob[0]+pt_b[0])/2, bry - 0.55, r"$t^*$",
        ha="center", va="top", fontsize=7, color=C_CAM)

# Uncertainty label (inside cone, far end)
ux = ob[0] + 7.5*np.cos(ang + ha)
uy = ob[1] + 7.5*np.sin(ang + ha)
ax.text(ux - 0.2, uy + 0.5, "uncertainty\nregion",
        ha="center", va="bottom", fontsize=5.5, color=C_RAY)

ax.text(5.0, -2.5, "(b)  Depth Ambiguity (Single View)",
        ha="center", va="bottom", fontsize=7, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Panel (c): Multi-view fusion
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(-2.8, 10)

pt_c = (7.0, 4.5)
cams_c = [(0.8, 0.7), (0.8, 4.5), (0.8, 8.3)]
labels = [r"$C_1$", r"$C_2$", r"$C_3$"]

for (cx_, cy_), lbl in zip(cams_c, labels):
    camera(ax, cx_, cy_)
    ax.text(cx_ - 0.55, cy_, lbl, ha="right", va="center",
            fontsize=6.5, color=C_CAM)

    dx_ = pt_c[0]-cx_; dy_ = pt_c[1]-cy_
    ang_ = np.arctan2(dy_, dx_)
    lgt_ = np.hypot(dx_, dy_)

    # per-camera cone (lighter alpha for readability)
    t_c = np.linspace(0, lgt_, 80)
    ca  = np.radians(5)
    tx  = cx_ + t_c*np.cos(ang_+ca);  ty  = cy_ + t_c*np.sin(ang_+ca)
    bx  = cx_ + t_c*np.cos(ang_-ca);  by_ = cy_ + t_c*np.sin(ang_-ca)
    verts = np.c_[np.concatenate([tx, bx[::-1]]),
                  np.concatenate([ty, by_[::-1]])]
    ax.add_patch(Polygon(verts, closed=True,
                         fc=C_CONE, ec="none", alpha=0.30, zorder=2))

    ax.annotate("", xy=pt_c, xytext=(cx_, cy_),
                arrowprops=dict(arrowstyle="->", color=C_RAY, lw=LW,
                                mutation_scale=8,
                                shrinkA=5, shrinkB=9), zorder=4)

# Convergence zone
ax.add_patch(Ellipse(pt_c, 0.90, 0.90,
                     fc=C_CONV, ec=C_OBJ, lw=LWS, zorder=5))
ax.plot(*pt_c, "o", color=C_OBJ, ms=MS+0.5, zorder=8)
ax.text(pt_c[0]+0.6, pt_c[1]+0.5, r"$\bar{P}$",
        fontsize=8, color=C_OBJ, fontweight="bold", va="bottom")

# Error annotation
ax.annotate(r"$\sigma \propto 1/\!\sqrt{N}$",
            xy=(pt_c[0]+0.45, pt_c[1]-0.5),
            xytext=(pt_c[0]+1.3, pt_c[1]-2.1),
            fontsize=7, color=C_OBJ,
            arrowprops=dict(arrowstyle="->", color=C_OBJ,
                            lw=0.5, mutation_scale=6))

ax.text(5.0, -2.5, r"(c)  Multi-view Fusion ($N$ cameras)",
        ha="center", va="bottom", fontsize=7, style="italic")


# ── Global legend ─────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], color=C_RAY, lw=LW,        label="projection ray"),
    mpatches.Patch(fc=C_CONE, ec="none",        label="depth uncertainty"),
    Line2D([0],[0], color=C_OBJ, marker="o",
           ls="none", ms=MS,                    label="3-D point estimate"),
    Line2D([0],[0], color=C_CAM, marker="s",
           ls="none", ms=MS,                    label="camera center"),
]
fig.legend(handles=legend_elements,
           loc="lower center", ncol=4,
           fontsize=6, frameon=True, framealpha=1.0,
           edgecolor="#CCCCCC",
           bbox_to_anchor=(0.50, 0.01),
           handlelength=1.5, handletextpad=0.4, columnspacing=1.2)


# ── Save ──────────────────────────────────────────────────────────────────────
import os
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "ray_projection_figure.pdf")
png_path = os.path.join(out_dir, "ray_projection_figure.png")

plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
