"""Compute + plot helpers for the flux_trajectoid Space app (local-first)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Package path: repo src/ when running locally
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flux_trajectoid import PhotonSeedAsteroid, oam_flux_backend  # noqa: E402

GITHUB = "https://github.com/kinaar8340/flux_trajectoid"
HF_SPACE = "https://huggingface.co/spaces/kinaar111/flux_trajectoid"
ASSETS = Path(__file__).resolve().parent / "assets"


def asset_path(name: str) -> str | None:
    p = ASSETS / name
    return str(p) if p.is_file() else None


def _fig_to_rgb(fig, *, size: tuple[int, int] | None = None) -> np.ndarray:
    """Rasterize figure to RGB; optionally force exact (H, W) for GIF stability."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3].copy()
    except Exception:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb = buf.reshape(h, w, 3).copy()
    plt.close(fig)
    if size is not None:
        th, tw = size
        if rgb.shape[0] != th or rgb.shape[1] != tw:
            from PIL import Image

            rgb = np.asarray(
                Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR)
            )
    return rgb


def _slice_axis_and_value(
    shell,
    slice_frac: float = 0.5,
    plane: str = "z",
) -> tuple[int, int, int, float, np.ndarray, np.ndarray] | None:
    """Return (axis, a0, a1, plane_value, lo, hi) for the green scan frame."""
    if shell.surface is None and shell.mesh_vertices is None:
        return None
    if shell.surface is not None:
        pts = np.asarray(shell.surface, dtype=float).reshape(-1, 3)
    else:
        pts = np.asarray(shell.mesh_vertices, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = 0.03 * (hi - lo + 1e-9)
    lo = lo - pad
    hi = hi + pad
    axis = {"x": 0, "y": 1, "z": 2}.get(str(plane).lower(), 2)
    a0, a1 = [i for i in range(3) if i != axis]
    c = float(lo[axis] + np.clip(float(slice_frac), 0.0, 1.0) * (hi[axis] - lo[axis]))
    return axis, a0, a1, c, lo, hi


def _intersection_curve_on_plane(
    shell,
    *,
    axis: int,
    a0: int,
    a1: int,
    c: float,
) -> np.ndarray | None:
    """
    Sample the curve where the scan plane hits the 3D shell / sphere.

    Prefer UV-surface edge crossings (true shell silhouette on the plane).
    Fall back to sphere∩plane circle using rolling_radius / surface mean.
    Returns (N, 3) polyline ordered by angle (closed if possible).
    """
    crossings: list[np.ndarray] = []

    if shell.surface is not None:
        s = np.asarray(shell.surface, dtype=float)
        if s.ndim == 3 and s.shape[-1] == 3:
            n_lat, n_lon, _ = s.shape
            # Cross along meridians (varying latitude)
            for j in range(n_lon):
                col = s[:, j, :]
                vals = col[:, axis]
                for i in range(n_lat - 1):
                    v0, v1 = vals[i], vals[i + 1]
                    if (v0 - c) * (v1 - c) <= 0.0 and abs(v1 - v0) > 1e-14:
                        t = (c - v0) / (v1 - v0)
                        crossings.append(col[i] + t * (col[i + 1] - col[i]))
            # Cross along parallels (varying longitude) for denser arc
            for i in range(n_lat):
                row = s[i, :, :]
                vals = row[:, axis]
                for j in range(n_lon - 1):
                    v0, v1 = vals[j], vals[j + 1]
                    if (v0 - c) * (v1 - c) <= 0.0 and abs(v1 - v0) > 1e-14:
                        t = (c - v0) / (v1 - v0)
                        crossings.append(row[j] + t * (row[j + 1] - row[j]))

    if len(crossings) < 6 and shell.mesh_vertices is not None and shell.mesh_faces is not None:
        # Mesh edge crossings as secondary sample
        V = np.asarray(shell.mesh_vertices, dtype=float)
        F = np.asarray(shell.mesh_faces, dtype=int)
        for tri in F:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                p0, p1 = V[tri[a]], V[tri[b]]
                v0, v1 = p0[axis], p1[axis]
                if (v0 - c) * (v1 - c) <= 0.0 and abs(v1 - v0) > 1e-14:
                    t = (c - v0) / (v1 - v0)
                    crossings.append(p0 + t * (p1 - p0))

    if len(crossings) >= 6:
        pts = np.asarray(crossings, dtype=float)
        # Project onto plane (numerical snap)
        pts[:, axis] = c
        mid = pts.mean(axis=0)
        u = pts[:, a0] - mid[a0]
        v = pts[:, a1] - mid[a1]
        ang = np.arctan2(v, u)
        order = np.argsort(ang)
        pts = pts[order]
        ang = ang[order]
        # Angular bin average for a smooth single-valued arc
        n_bins = min(180, max(48, len(pts) // 2))
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        smooth: list[np.ndarray] = []
        for k in range(n_bins):
            mask = (ang >= edges[k]) & (ang < edges[k + 1])
            if not np.any(mask):
                continue
            smooth.append(pts[mask].mean(axis=0))
        if len(smooth) >= 6:
            out = np.asarray(smooth, dtype=float)
            out = np.vstack([out, out[:1]])  # close loop
            return out

    # Sphere ∩ plane fallback (idealised rolling sphere)
    if shell.surface is not None:
        pts_all = np.asarray(shell.surface, dtype=float).reshape(-1, 3)
    elif shell.mesh_vertices is not None:
        pts_all = np.asarray(shell.mesh_vertices, dtype=float)
    else:
        return None
    center = pts_all.mean(axis=0)
    R = float(getattr(shell, "rolling_radius", 0.0) or 0.0)
    if R <= 1e-9:
        R = float(np.median(np.linalg.norm(pts_all - center, axis=1)))
    delta = float(c - center[axis])
    if abs(delta) >= R * 0.999:
        return None
    r_c = float(np.sqrt(max(R * R - delta * delta, 0.0)))
    mid = center.copy()
    mid[axis] = c
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    circle = np.zeros((theta.size, 3), dtype=float)
    circle[:, axis] = c
    circle[:, a0] = mid[a0] + r_c * np.cos(theta)
    circle[:, a1] = mid[a1] + r_c * np.sin(theta)
    return circle


def _draw_intersection_arc(
    ax,
    shell,
    *,
    axis: int,
    a0: int,
    a1: int,
    c: float,
) -> None:
    """Green #00FF00 arc/circle: scan plane ∩ sphere/shell, coplanar with frame.

    Core stroke half of prior (2.6 → 1.3) with multi-layer glow.
    """
    curve = _intersection_curve_on_plane(shell, axis=axis, a0=a0, a1=a1, c=c)
    if curve is None or len(curve) < 4:
        return
    xs, ys, zs = curve[:, 0], curve[:, 1], curve[:, 2]
    # Glow halo scaled under core opacity 0.3
    for lw, alpha, z in (
        (6.0, 0.04, 12),
        (3.5, 0.08, 13),
        (2.2, 0.14, 14),
    ):
        ax.plot(
            xs, ys, zs,
            color="#00FF00",
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )
    # Core intersection oval — opacity 0.3
    ax.plot(
        xs, ys, zs,
        color="#00FF00",
        lw=1.3,
        alpha=0.3,
        solid_capstyle="round",
        zorder=15,
    )


def _draw_green_slice(
    ax,
    shell,
    slice_frac: float = 0.5,
    plane: str = "z",
) -> None:
    """
    Matrix-green outlined rectangular slice through the shell,
    plus coplanar intersection arc where the plane hits the shell/sphere.

    Edge: #00FF00 solid · face fill: fully transparent · hit arc: #00FF00
    """
    meta = _slice_axis_and_value(shell, slice_frac=slice_frac, plane=plane)
    if meta is None:
        return
    axis, a0, a1, c, lo, hi = meta

    # Outline rectangle only (fill opacity 0 — no face paint)
    corners_uv = np.array(
        [
            [lo[a0], lo[a1]],
            [hi[a0], lo[a1]],
            [hi[a0], hi[a1]],
            [lo[a0], hi[a1]],
            [lo[a0], lo[a1]],
        ]
    )
    outline = np.zeros((5, 3))
    outline[:, a0] = corners_uv[:, 0]
    outline[:, a1] = corners_uv[:, 1]
    outline[:, axis] = c
    ox, oy, oz = outline[:, 0], outline[:, 1], outline[:, 2]
    # Frame glow scaled under core opacity 0.3
    for lw, alpha, z in (
        (4.5, 0.04, 8),
        (2.6, 0.08, 9),
        (1.6, 0.14, 10),
    ):
        ax.plot(
            ox, oy, oz,
            color="#00FF00",
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )
    # Frame outer edge core — opacity 0.3
    ax.plot(
        ox, oy, oz,
        color="#00FF00",
        lw=0.9,
        alpha=0.3,
        solid_capstyle="round",
        zorder=11,
    )

    # Arc on the same plane: plane ∩ shell/sphere
    _draw_intersection_arc(ax, shell, axis=axis, a0=a0, a1=a1, c=c)


# Fixed raster sizes for GIF frames (prevents loop glitch from layout jitter)
_SHELL_SIZE = (360, 420)  # H, W
_RADIAL_SIZE = (320, 420)
_PATH_SIZE = (550, 320)


def plot_shell_3d(
    shell,
    *,
    slice_frac: float = 0.5,
    slice_plane: str = "z",
    show_slice: bool = True,
) -> np.ndarray:
    fig = plt.figure(figsize=(4.2, 3.6), facecolor="#0b1220", dpi=100)
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b1220")
    ax.tick_params(colors="#8aa0c0", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color("#334155")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Lock camera so frames don't jump between slices
    ax.view_init(elev=22, azim=-55)

    if getattr(shell, "is_3d", False) and shell.surface is not None:
        s = shell.surface
        stride = max(1, s.shape[0] // 24)
        ax.plot_surface(
            s[::stride, ::stride, 0],
            s[::stride, ::stride, 1],
            s[::stride, ::stride, 2],
            rstride=1,
            cstride=1,
            color="#3b82f6",
            alpha=0.55,
            linewidth=0,
            antialiased=True,
        )
        if shell.path_on_body is not None:
            p = shell.path_on_body
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#f87171", lw=1.8, alpha=0.95)
        if show_slice:
            _draw_green_slice(ax, shell, slice_frac=slice_frac, plane=slice_plane)
        # Fixed axis limits from full surface
        pts = s.reshape(-1, 3)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        mid = 0.5 * (lo + hi)
        span = float(np.max(hi - lo) * 0.55 + 1e-9)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
    else:
        v = shell.vertices
        ax.plot(
            v[:, 0],
            v[:, 1],
            v[:, 2] if v.shape[1] > 2 else np.zeros(len(v)),
            color="#60a5fa",
        )
        if show_slice:
            _draw_green_slice(ax, shell, slice_frac=slice_frac, plane=slice_plane)

    # Fixed-length title (no layout thrash from changing digit strings)
    ax.set_title("3D Trajectoid Shell · matrix scan", color="#e2e8f0", fontsize=9, pad=2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    return _fig_to_rgb(fig, size=_SHELL_SIZE)


def plot_radial_map(
    shell,
    *,
    slice_frac: float | None = None,
    slice_plane: str = "z",
) -> np.ndarray:
    """Radial / trench map; optional matrix-green band locked to scan frac."""
    fig = plt.figure(figsize=(4.2, 3.2), facecolor="#0b1220", dpi=100)
    # Fixed axes box so colorbar/layout never shifts frame-to-frame
    ax = fig.add_axes([0.08, 0.10, 0.72, 0.78])
    cax = fig.add_axes([0.84, 0.10, 0.03, 0.78])
    ax.set_facecolor("#0b1220")
    plane = str(slice_plane or "z").lower()
    if plane not in ("x", "y", "z"):
        plane = "z"
    f = None if slice_frac is None else float(np.clip(slice_frac, 0.0, 1.0))

    if shell.radial_map is not None:
        rmap = np.asarray(shell.radial_map, dtype=float)
        n_lat, n_lon = rmap.shape
        vmin = float(np.nanmin(rmap))
        vmax = float(np.nanmax(rmap))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(
            rmap,
            origin="lower",
            cmap="magma",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        cb = fig.colorbar(im, cax=cax)
        cb.ax.yaxis.set_tick_params(color="#94a3b8", labelsize=7)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#94a3b8", fontsize=7)
        ax.set_xlim(-0.5, n_lon - 0.5)
        ax.set_ylim(-0.5, n_lat - 0.5)
        if f is not None:
            # UV map: rows ~ polar (z), cols ~ azimuth (x/y scan proxy)
            if plane == "z":
                i = f * max(n_lat - 1, 1)  # continuous position (no integer snap jitter)
                ax.axhline(i, color="#00FF00", lw=2.0, alpha=0.95, zorder=5)
                ax.axhspan(i - 0.85, i + 0.85, color="#00FF00", alpha=0.12, zorder=4)
            else:
                j = f * max(n_lon - 1, 1)
                ax.axvline(j, color="#00FF00", lw=2.0, alpha=0.95, zorder=5)
                ax.axvspan(j - 0.85, j + 0.85, color="#00FF00", alpha=0.12, zorder=4)
        ax.set_title("Radial map · trench / shave", color="#e2e8f0", fontsize=10)
    else:
        cax.set_visible(False)
        curv = shell.curvature_signal
        if curv is not None and len(curv):
            curv = np.asarray(curv, dtype=float)
            ax.plot(curv, color="#38bdf8", lw=1.4, alpha=0.85)
            ax.set_xlim(0, max(len(curv) - 1, 1))
            ymin, ymax = float(np.nanmin(curv)), float(np.nanmax(curv))
            pad = 0.05 * (ymax - ymin + 1e-9)
            ax.set_ylim(ymin - pad, ymax + pad)
            if f is not None:
                i = f * max(len(curv) - 1, 1)
                ax.axvline(i, color="#00FF00", lw=2.0, alpha=0.95)
                ii = int(round(i))
                ax.scatter([ii], [curv[ii]], c="#00FF00", s=36, zorder=5)
            ax.set_title("Curvature signal", color="#e2e8f0", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no radial map", ha="center", color="#64748b")
            ax.set_axis_off()
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    return _fig_to_rgb(fig, size=_RADIAL_SIZE)


def plot_protected_field(flux) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    if flux is not None and flux.protected_field is not None:
        I = np.abs(flux.protected_field) ** 2
        ax.imshow(I, origin="lower", cmap="inferno")
        ax.set_title("|protected field|² · shell trench", color="#e2e8f0", fontsize=10)
    else:
        ax.text(0.5, 0.5, "no field", ha="center", va="center", color="#94a3b8")
        ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#334155")
    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig)


def plot_path_panel(
    shell,
    style: str = "theory",
    *,
    progress: float | None = None,
) -> np.ndarray:
    """Path viewport inspired by Nature trajectoid path columns.

    If ``progress`` ∈ [0, 1] is set, draw a scan-head trail synced to matrix scan.
    """
    fig = plt.figure(figsize=(3.2, 5.5), facecolor="#0b1220", dpi=100)
    ax = fig.add_axes([0.08, 0.04, 0.84, 0.90])
    ax.set_facecolor("#0b1220")
    path = shell.vertices[:, :2]
    # Normalize and stretch vertically like the figure columns
    p = path - path.mean(axis=0)
    p = p / (np.max(np.abs(p)) + 1e-12)
    # Unroll as vertical wand: x small, y = arc progress + lateral
    s = np.linspace(0, 1, len(p))
    x = p[:, 0] * 0.35
    y = s * 4.0 + p[:, 1] * 0.25
    color = "#38bdf8" if style == "theory" else "#f472b6"

    # Ghost full path (always full length — stable composition)
    ax.plot(x, y, color="#64748b", lw=1.2, alpha=0.35, zorder=1)
    ax.plot(x * 0.92 + 0.08, y, color="#475569", lw=0.9, alpha=0.25, zorder=0)

    if progress is None:
        ax.plot(x, y, color=color, lw=1.6, alpha=0.95, zorder=2)
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], c="#f8fafc", s=18, zorder=3)
    else:
        f = float(np.clip(progress, 0.0, 1.0))
        # Continuous head position (lerp) reduces discrete snap
        t = f * max(len(x) - 1, 1)
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, len(x) - 1)
        a = t - i0
        hx = (1 - a) * x[i0] + a * x[i1]
        hy = (1 - a) * y[i0] + a * y[i1]
        idx = i1
        ax.plot(x[: idx + 1], y[: idx + 1], color=color, lw=2.0, alpha=0.95, zorder=2)
        ax.scatter([x[0]], [y[0]], c="#f8fafc", s=16, zorder=3)
        ax.scatter(
            [hx],
            [hy],
            c="#00FF00",
            s=48,
            zorder=5,
            edgecolors="#bbf7d0",
            linewidths=0.6,
        )
        ax.axhline(hy, color="#00FF00", lw=0.8, alpha=0.35, zorder=1)

    ax.set_title("Rolling path · matrix scan", color="#e2e8f0", fontsize=10)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.15, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return _fig_to_rgb(fig, size=_PATH_SIZE)


def plot_metrics_bars(metrics) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.5, 3.0), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    if metrics is None:
        ax.text(0.5, 0.5, "Run channel first", ha="center", color="#94a3b8")
        ax.set_axis_off()
        return _fig_to_rgb(fig)

    labels = ["F", "Icorr", "OAMf", "Strehl", "P"]
    vals = [
        metrics.overlap_fidelity,
        max(0.0, metrics.intensity_correlation),
        metrics.oam_fidelity,
        min(1.0, metrics.strehl_proxy),
        min(1.0, metrics.power_retention),
    ]
    colors = ["#3b82f6", "#22d3ee", "#a78bfa", "#f59e0b", "#34d399"]
    bars = ax.barh(labels, vals, color=colors, height=0.55, alpha=0.85)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.set_title("Turbulence scorecard", color="#e2e8f0", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color="#cbd5e1", fontsize=8)
    fig.tight_layout(pad=0.4)
    return _fig_to_rgb(fig)


def plot_fidelity_trace(prop) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.5, 2.4), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    if prop is None or not prop.fidelity_trace:
        ax.text(0.5, 0.5, "no trace", ha="center", color="#64748b")
        ax.set_axis_off()
        return _fig_to_rgb(fig)
    t = np.arange(len(prop.fidelity_trace))
    ax.plot(t, prop.fidelity_trace, color="#60a5fa", lw=1.8)
    ax.fill_between(t, prop.fidelity_trace, alpha=0.25, color="#3b82f6")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("step", color="#64748b", fontsize=8)
    ax.set_ylabel("F", color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.set_title("Fidelity vs propagation step", color="#e2e8f0", fontsize=10)
    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig)


def run_pipeline(
    payload: str,
    seed: int,
    turbulence: float,
    n_steps: int,
    use_tpt: bool,
    build_3d: bool,
    force_stub: bool,
    slice_frac: float = 0.5,
    slice_plane: str = "z",
    show_slice: bool = True,
) -> dict[str, Any]:
    """Build → propagate → recover; return images + text metrics."""
    payload = (payload or "Photon Seed Asteroid").strip()[:80]
    seed = int(seed)
    turbulence = float(np.clip(turbulence, 0.0, 1.0))
    n_steps = int(np.clip(n_steps, 4, 48))
    slice_frac = float(np.clip(slice_frac, 0.0, 1.0))
    slice_plane = str(slice_plane or "z").lower()
    if slice_plane not in ("x", "y", "z"):
        slice_plane = "z"

    # Fast defaults for interactive UI
    ast = PhotonSeedAsteroid(payload, seed=seed).build(
        use_tpt=use_tpt,
        build_3d=build_3d,
        force_stub_flux=force_stub,
        lattice_nx=10,
        n_coupling_steps=4,
        n_shards=4,
        n_points=96,
        scale_grid=3,
        scale_max_iter=4,
        n_lat=28 if build_3d else 16,
        n_lon=56 if build_3d else 32,
    )
    prop = ast.propagate(
        turbulence_level=turbulence,
        n_steps=n_steps,
        seed=seed,
        apply_bmgl=True,
    )
    rec_h = ast.recover(mode="hybrid")
    rec_p = ast.recover(mode="photonic")

    shell = ast.shell
    assert shell is not None

    img_shell = plot_shell_3d(
        shell,
        slice_frac=slice_frac,
        slice_plane=slice_plane,
        show_slice=bool(show_slice),
    )
    img_radial = plot_radial_map(shell)
    img_field = plot_protected_field(ast.flux_state)
    img_path = plot_path_panel(shell)
    img_metrics = plot_metrics_bars(prop.metrics)
    img_trace = plot_fidelity_trace(prop)

    m = prop.metrics
    flux_backend = (
        ast.flux_state.backend if ast.flux_state is not None else oam_flux_backend()
    )
    summary = {
        "backend": flux_backend,
        "is_3d": shell.is_3d,
        "mismatch_deg": round(shell.mismatch_deg, 3),
        "tilt_deg": round(shell.tilt_deg, 3),
        "kx": round(shell.kx, 3),
        "ky": round(shell.ky, 3),
        "volume_proxy": round(shell.volume_proxy, 4) if shell.is_3d else None,
        "mesh_V": int(shell.mesh_vertices.shape[0]) if shell.mesh_vertices is not None else 0,
        "mesh_F": int(shell.mesh_faces.shape[0]) if shell.mesh_faces is not None else 0,
        "contact_coverage": shell.metadata.get("contact_coverage"),
        "fidelity": round(prop.fidelity_proxy, 4),
        "oam_fidelity": round(m.oam_fidelity, 4) if m else None,
        "phase_rmse": round(m.phase_rmse_rad, 4) if m else None,
        "strehl": round(m.strehl_proxy, 4) if m else None,
        "payload_hybrid": rec_h.payload_text or rec_h.payload_hat[:32].hex(),
        "crc_ok": rec_h.crc_ok,
        "photonic_ber": rec_p.byte_error_rate,
        "chordal": rec_p.chordal_error_mean,
        "metrics_line": m.summary_line() if m else "",
    }

    md = _format_status_md(summary, payload, turbulence, n_steps)
    return {
        "shell": shell,
        "img_shell": img_shell,
        "img_radial": img_radial,
        "img_field": img_field,
        "img_path": img_path,
        "img_metrics": img_metrics,
        "img_trace": img_trace,
        "status_md": md,
        "summary": summary,
    }


def _format_status_md(s: dict, payload: str, turb: float, steps: int) -> str:
    return f"""### Seed status
| | |
|---|---|
| **Payload** | `{payload}` |
| **Backend** | `{s['backend']}` |
| **3D shell** | {s['is_3d']} · V={s['mesh_V']} F={s['mesh_F']} |
| **Mismatch** | {s['mismatch_deg']}° · tilt {s['tilt_deg']}° |
| **Scale** | kx={s['kx']} ky={s['ky']} |
| **Volume** | {s['volume_proxy']} · coverage {s['contact_coverage']} |
| **Channel** | turb={turb:.2f} · steps={steps} |
| **F / OAM** | {s['fidelity']} / {s['oam_fidelity']} |
| **φrms / Strehl** | {s['phase_rmse']} / {s['strehl']} |
| **Hybrid** | `{s['payload_hybrid']}` · CRC={s['crc_ok']} |
| **Photonic BER** | {s['photonic_ber']} · chordal={s['chordal']} |

`{s['metrics_line']}`
"""


def blank_rgb(h: int = 240, w: int = 320) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (11, 18, 32)
    return img


def replot_shell_only(
    shell,
    slice_frac: float = 0.5,
    slice_plane: str = "z",
    show_slice: bool = True,
) -> np.ndarray:
    """Fast re-render of 3D shell for live matrix-slice controls (no rebuild)."""
    if shell is None:
        return blank_rgb(300, 360)
    plane = str(slice_plane or "z").lower()
    if plane not in ("x", "y", "z"):
        plane = "z"
    try:
        return plot_shell_3d(
            shell,
            slice_frac=float(np.clip(slice_frac, 0.0, 1.0)),
            slice_plane=plane,
            show_slice=bool(show_slice),
        )
    except Exception:
        return blank_rgb(300, 360)


def _write_gif(frames_rgb: list[np.ndarray], path: Path, duration_ms: int) -> None:
    from PIL import Image

    if not frames_rgb:
        raise ValueError("no frames")
    # Normalize every frame to identical size (kills loop glitch)
    h0, w0 = frames_rgb[0].shape[:2]
    pil_frames = []
    for rgb in frames_rgb:
        im = Image.fromarray(rgb)
        if im.size != (w0, h0):
            im = im.resize((w0, h0), Image.Resampling.BILINEAR)
        # Flatten to palette-friendly RGB without per-frame dither variance
        pil_frames.append(im.convert("RGB"))
    # disposal=2 + fixed duration: clean browser loops
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def animate_matrix_scan(
    shell,
    *,
    slice_plane: str = "z",
    n_frames: int = 14,
    ping_pong: bool = False,
    duration_ms: int = 90,
    out_dir: Path | None = None,
) -> tuple[str | None, str | None, str | None, str]:
    """
    Synced axial scan suite (same timeline for all three viewports):

    1. 3D shell + green matrix plane
    2. Radial map green band
    3. Rolling path progress head

    Returns (gif_shell, gif_radial, gif_path, status_md).
    Default one-way 14-frame scan (no ping-pong bounce).
    """
    if shell is None:
        empty = "### Scan\n_No shell yet — run **Build** first._"
        return None, None, None, empty

    plane = str(slice_plane or "z").lower()
    if plane not in ("x", "y", "z"):
        plane = "z"
    # Balanced demo defaults: 8–24, preferred 14
    n_frames = int(np.clip(n_frames, 8, 24))
    duration_ms = int(np.clip(duration_ms, 50, 150))

    # One-way scan by default (ping-pong causes visible bounce/glitch)
    fracs = np.linspace(0.0, 1.0, n_frames)
    if ping_pong:
        # Smooth turnaround: hold end frame once, then reverse interior
        fracs = np.concatenate([fracs, fracs[-1:], fracs[-2:0:-1]])

    shells: list[np.ndarray] = []
    radials: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    for f in fracs:
        ff = float(f)
        shells.append(
            plot_shell_3d(
                shell,
                slice_frac=ff,
                slice_plane=plane,
                show_slice=True,
            )
        )
        radials.append(
            plot_radial_map(shell, slice_frac=ff, slice_plane=plane)
        )
        paths.append(plot_path_panel(shell, progress=ff))

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return (
            None,
            None,
            None,
            "### Scan\n_Pillow required for GIF export (`pip install pillow`)._",
        )

    out_dir = out_dir or (ROOT / "outputs" / "space_anim")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Unique names avoid browser/Gradio caching old glitchy GIFs
    stamp = int(np.random.randint(0, 1_000_000))
    p_shell = out_dir / f"matrix_scan_{plane}_shell_{stamp}.gif"
    p_radial = out_dir / f"matrix_scan_{plane}_radial_{stamp}.gif"
    p_path = out_dir / f"matrix_scan_{plane}_path_{stamp}.gif"

    _write_gif(shells, p_shell, duration_ms)
    _write_gif(radials, p_radial, duration_ms)
    _write_gif(paths, p_path, duration_ms)

    mode = "ping-pong" if ping_pong else "one-way (smooth loop)"
    msg = (
        f"### Matrix scan (synced)\n"
        f"Plane **{plane}** · {len(fracs)} frames · {duration_ms} ms/frame · {mode}\n\n"
        f"| Viewport | File |\n|---|---|\n"
        f"| 3D shell | `{p_shell.name}` |\n"
        f"| Radial map | `{p_radial.name}` |\n"
        f"| Rolling path | `{p_path.name}` |\n"
    )
    return str(p_shell), str(p_radial), str(p_path), msg
