"""Shell geometry → phase mask / constraint on inner OAM modes.

Implements a trajectoid "potential trench" analogy: the outer shell
geometry modulates and protects the inner Laguerre–Gaussian / OAM
modes by imposing a spatially varying phase and amplitude envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import ShellGeometry


@dataclass
class ShellModulation:
    """Phase mask and amplitude constraint derived from shell geometry."""

    phase_mask: np.ndarray  # (H, W) radians
    amplitude_envelope: np.ndarray  # (H, W) in [0, 1]
    potential_trench: np.ndarray  # (H, W) soft wall potential
    oam_phase_bias: dict[int, float]  # per-ℓ phase offset from shell fingerprint
    grid_extent: float = 2.0


def _rasterize_shell_silhouette(
    vertices: np.ndarray,
    *,
    grid_size: int = 64,
    extent: float = 2.0,
) -> np.ndarray:
    """Binary-ish mask of the 2D projection of the shell boundary (filled)."""
    from matplotlib.path import Path

    xy = vertices[:, :2]
    # Normalize to extent
    center = xy.mean(axis=0)
    xy_c = xy - center
    scale = np.max(np.abs(xy_c)) + 1e-12
    xy_n = xy_c / scale * (0.85 * extent)

    xs = np.linspace(-extent, extent, grid_size)
    ys = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    path = Path(xy_n)
    inside = path.contains_points(pts).reshape(grid_size, grid_size)
    return inside.astype(float)


def _radial_map_modulation(
    shell: ShellGeometry,
    *,
    grid_size: int,
    extent: float,
    trench_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D radial_map → 2D phase / envelope / trench."""
    from scipy.ndimage import gaussian_filter, zoom

    rmap = shell.radial_map
    assert rmap is not None
    # Orthographic: use equatorial belt (mid latitudes) average as silhouette strength
    n_lat, n_lon = rmap.shape
    mid = slice(n_lat // 4, 3 * n_lat // 4)
    belt = rmap[mid, :]
    # Map longitude → polar image
    r0 = float(shell.rolling_radius) if shell.rolling_radius else float(np.mean(rmap))
    # Radius deficit = trench proxy
    deficit = np.clip(r0 - rmap, 0.0, None)
    deficit = deficit / (deficit.max() + 1e-12)

    # Build polar field on Cartesian grid from equatorial radius profile
    lon_profile = belt.mean(axis=0)  # (n_lon,)
    lon_def = deficit.mean(axis=0)

    xs = np.linspace(-extent, extent, grid_size)
    ys = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    # Normalize lon_profile to max radius in extent units
    r_eq = lon_profile / (lon_profile.max() + 1e-12) * (0.85 * extent)
    idx = ((phi + np.pi) / (2 * np.pi) * (len(r_eq) - 1)).astype(int)
    idx = np.clip(idx, 0, len(r_eq) - 1)
    r_bound = r_eq[idx]
    mask = (rho <= r_bound).astype(float)
    mask = gaussian_filter(mask, sigma=1.0)
    mask = mask / (mask.max() + 1e-12)

    trench_1d = lon_def[idx]
    trench = trench_depth * trench_1d * np.exp(-((rho - r_bound) ** 2) / (2 * (0.12 * extent) ** 2))
    # Interior contact-trench from full deficit map (zoom to square)
    def_img = zoom(deficit, (grid_size / n_lat, grid_size / n_lon), order=1)
    if def_img.shape != (grid_size, grid_size):
        def_img = np.resize(def_img, (grid_size, grid_size))
    trench = trench + 0.45 * trench_depth * def_img * mask

    # Phase from curvature + deficit
    curv = shell.curvature_signal
    if curv is not None and len(curv) > 0:
        cidx = ((phi + np.pi) / (2 * np.pi) * (len(curv) - 1)).astype(int)
        cidx = np.clip(cidx, 0, len(curv) - 1)
        phase = 0.5 * curv[cidx] + 0.35 * def_img
    else:
        phase = 0.35 * def_img
    phase = phase - phase.mean()
    return phase, mask, trench


def shell_to_phase_mask(
    shell: ShellGeometry,
    *,
    grid_size: int = 64,
    extent: float = 2.0,
    trench_depth: float = 2.5,
    trench_width: float = 0.12,
) -> ShellModulation:
    """
    Map shell geometry to an OAM protection modulation.

    - Interior: free propagation (envelope ~ 1)
    - Boundary trench: high potential (suppresses mode leakage)
    - Phase: unrolled curvature signal mapped azimuthally
    - Per-ℓ bias: from Fourier fingerprint
    - If ``shell.is_3d``: use radial_map contact trench from the shaved sphere
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if getattr(shell, "is_3d", False) and shell.radial_map is not None:
        phase, envelope, trench = _radial_map_modulation(
            shell, grid_size=grid_size, extent=extent, trench_depth=trench_depth
        )
    else:
        mask = _rasterize_shell_silhouette(shell.vertices, grid_size=grid_size, extent=extent)
        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(1.0 - mask)
        signed = dist_in - dist_out
        trench = trench_depth * np.exp(-(signed**2) / (2.0 * trench_width**2))
        envelope = gaussian_filter(mask, sigma=1.0)
        envelope = envelope / (envelope.max() + 1e-12)
        # Azimuthal phase from curvature signal
        xs = np.linspace(-extent, extent, grid_size)
        ys = np.linspace(-extent, extent, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        phi = np.arctan2(yy, xx)
        curv = shell.curvature_signal
        if curv is None or len(curv) == 0:
            phase = np.zeros_like(mask)
        else:
            idx = ((phi + np.pi) / (2 * np.pi) * (len(curv) - 1)).astype(int)
            idx = np.clip(idx, 0, len(curv) - 1)
            phase = 0.5 * curv[idx]
            phase = phase - phase.mean()

    # Per-ℓ phase bias from fingerprint
    fp = shell.fourier_fingerprint
    if fp is None:
        fp = np.zeros(8)
    oam_ells = [0, 1, -1, 2, 3]
    oam_phase_bias = {}
    for i, ell in enumerate(oam_ells):
        oam_phase_bias[ell] = float(2.0 * np.pi * fp[i % len(fp)])

    return ShellModulation(
        phase_mask=phase,
        amplitude_envelope=envelope,
        potential_trench=trench,
        oam_phase_bias=oam_phase_bias,
        grid_extent=extent,
    )


def apply_modulation(
    field: np.ndarray,
    modulation: ShellModulation,
    *,
    ell: int | None = None,
) -> np.ndarray:
    """Apply shell phase/amplitude protection to a complex field."""
    # Resize modulation if needed
    h, w = field.shape[-2], field.shape[-1]
    phase = modulation.phase_mask
    env = modulation.amplitude_envelope
    if phase.shape != (h, w):
        from scipy.ndimage import zoom

        zy, zx = h / phase.shape[0], w / phase.shape[1]
        phase = zoom(phase, (zy, zx), order=1)
        env = zoom(env, (zy, zx), order=1)

    bias = 0.0
    if ell is not None and ell in modulation.oam_phase_bias:
        bias = modulation.oam_phase_bias[ell]

    return field * env * np.exp(1j * (phase + bias))
