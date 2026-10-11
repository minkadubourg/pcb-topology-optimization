"""Matplotlib rendering of element fields on 2D meshes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import VOL  # noqa: E402

COPPER = matplotlib.colors.LinearSegmentedColormap.from_list("copper_board", ["#f4f1ea", "#c06a2b", "#6b2f0b"])


def triangulation(mesh):
    pts = np.array([v.point for v in mesh.vertices])
    tris = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])
    return mtri.Triangulation(pts[:, 0], pts[:, 1], tris)


def element_values(mesh, cf):
    """Value of a coefficient function at each element centroid."""
    pts = np.array([v.point for v in mesh.vertices])
    out = []
    for el in mesh.Elements(VOL):
        c = pts[[v.nr for v in el.vertices]].mean(axis=0)
        val = cf(mesh(*c))
        out.append(np.linalg.norm(val) if np.ndim(val) else val)
    return np.array(out)


def draw_density(ax, model, rho_full, title=None, pads=True):
    tri = triangulation(model.mesh)
    mats = np.array([el.mat for el in model.mesh.Elements(VOL)])
    vals = np.where(model.conducting, rho_full, np.nan)
    ax.tripcolor(tri, vals, cmap=COPPER, vmin=0, vmax=1, shading="flat")
    if pads:
        port_names = {p.name for p in model.ports}
        pad = np.isin(mats, list(port_names))
        if pad.any():
            ax.tripcolor(mtri.Triangulation(tri.x, tri.y, tri.triangles[pad]), np.ones(pad.sum()),
                         cmap=matplotlib.colors.ListedColormap(["#2b3a55"]), shading="flat")
            pts = np.array([v.point for v in model.mesh.vertices])
            for name in port_names:
                sel = tri.triangles[mats == name].ravel()
                cx, cy = pts[sel].mean(axis=0)
                ax.annotate(name, (cx, cy), color="white", ha="center", va="center", fontsize=7, weight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def draw_field(ax, mesh, values, title=None, cmap="viridis", log=False):
    tri = triangulation(mesh)
    v = np.log10(np.maximum(values, 1e-12)) if log else values
    im = ax.tripcolor(tri, v, cmap=cmap, shading="flat")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im
