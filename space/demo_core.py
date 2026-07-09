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


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # matplotlib 3.8+ uses buffer_rgba
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3].copy()
    except Exception:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb = buf.reshape(h, w, 3).copy()
    plt.close(fig)
    return rgb


def plot_shell_3d(shell) -> np.ndarray:
    fig = plt.figure(figsize=(4.2, 3.6), facecolor="#0b1220")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0b1220")
    ax.tick_params(colors="#8aa0c0", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color("#334155")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

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
    else:
        v = shell.vertices
        ax.plot(v[:, 0], v[:, 1], v[:, 2] if v.shape[1] > 2 else np.zeros(len(v)), color="#60a5fa")

    ax.set_title("3D Trajectoid Shell", color="#e2e8f0", fontsize=10, pad=2)
    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig)


def plot_radial_map(shell) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    if shell.radial_map is not None:
        im = ax.imshow(shell.radial_map, origin="lower", cmap="magma", aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#94a3b8", fontsize=7)
        ax.set_title("Radial map · trench / shave", color="#e2e8f0", fontsize=10)
    else:
        ax.plot(shell.curvature_signal or [], color="#38bdf8")
        ax.set_title("Curvature signal", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    fig.tight_layout(pad=0.4)
    return _fig_to_rgb(fig)


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


def plot_path_panel(shell, style: str = "theory") -> np.ndarray:
    """Path viewport inspired by Nature trajectoid path columns."""
    fig, ax = plt.subplots(figsize=(3.2, 5.5), facecolor="#0b1220")
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
    ax.plot(x, y, color=color, lw=1.6, alpha=0.95, label=style)
    ax.scatter([x[0], x[-1]], [y[0], y[-1]], c="#f8fafc", s=18, zorder=3)
    # Soft ghost second path (TPT echo)
    ax.plot(x * 0.92 + 0.08, y, color="#64748b", lw=1.0, alpha=0.35)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.15, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rolling path", color="#e2e8f0", fontsize=10)
    fig.tight_layout(pad=0.2)
    return _fig_to_rgb(fig)


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
) -> dict[str, Any]:
    """Build → propagate → recover; return images + text metrics."""
    payload = (payload or "Photon Seed Asteroid").strip()[:80]
    seed = int(seed)
    turbulence = float(np.clip(turbulence, 0.0, 1.0))
    n_steps = int(np.clip(n_steps, 4, 48))

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

    img_shell = plot_shell_3d(shell)
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
