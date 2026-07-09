"""Fourier descriptors and arc-length unrolling for closed shell boundaries."""

from __future__ import annotations

import numpy as np


def _ensure_closed(curve: np.ndarray) -> np.ndarray:
    """Ensure first and last points coincide for a closed 2D/3D polyline."""
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 3:
        raise ValueError("curve must be (N, D) with N >= 3")
    if not np.allclose(curve[0], curve[-1], atol=1e-9):
        curve = np.vstack([curve, curve[0]])
    return curve


def arc_length_parameterize(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cumulative arc length s and unit-speed reparameterization params.

    Parameters
    ----------
    curve : (N, D) closed or open polyline

    Returns
    -------
    s : (N,) cumulative arc length
    total : float total length
    """
    curve = np.asarray(curve, dtype=float)
    diffs = np.diff(curve, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, float(s[-1])


def unroll_curve(curve: np.ndarray, n_samples: int | None = None) -> dict:
    """
    Arc-length unroll a closed curve into a 1D metadata path.

    Returns dict with:
      - s: arc-length samples
      - path: positions along curve at uniform s
      - curvature_signal: approximate curvature proxy (for metadata encoding)
      - total_length: perimeter
    """
    curve = _ensure_closed(curve)
    s, total = arc_length_parameterize(curve)
    if total < 1e-12:
        raise ValueError("degenerate curve (zero length)")

    n = n_samples or (len(curve) - 1)
    s_uniform = np.linspace(0.0, total, n, endpoint=False)

    # Interpolate each coordinate vs arc length
    path = np.column_stack(
        [np.interp(s_uniform, s, curve[:, d], period=None) for d in range(curve.shape[1])]
    )

    # Finite-difference curvature proxy on unrolled path
    d1 = np.gradient(path, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.linalg.norm(d1, axis=1) + 1e-12
    if path.shape[1] == 2:
        curvature = np.abs(_cross2d(d1, d2)) / (speed**3)
    else:
        cross = np.cross(d1, d2)
        curvature = np.linalg.norm(cross, axis=-1) / (speed**3)

    return {
        "s": s_uniform,
        "path": path,
        "curvature_signal": curvature,
        "total_length": total,
    }


def _cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D cross product magnitude (signed)."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def compute_fourier_descriptors(
    curve: np.ndarray,
    n_harmonics: int = 12,
) -> dict:
    """
    Complex Fourier descriptors of a closed 2D boundary.

    Uses the complex plane representation z = x + i y, DFT, and returns
    the lowest ``n_harmonics`` positive frequencies (plus DC and negative
    for reconstruction).

    Returns
    -------
    dict with keys:
      - coeffs: complex DFT coefficients (full)
      - descriptors: truncated normalized descriptors (translation/scale invariant)
      - n_harmonics: int
      - reconstruction: (M, 2) curve from truncated coeffs
    """
    curve = _ensure_closed(curve)
    # Work in 2D projection if 3D given
    if curve.shape[1] >= 3:
        xy = curve[:-1, :2]
    else:
        xy = curve[:-1]

    z = xy[:, 0] + 1j * xy[:, 1]
    # Center (translation invariance)
    z = z - z.mean()
    coeffs = np.fft.fft(z)

    # Scale invariance: normalize by first non-zero harmonic magnitude
    scale = np.abs(coeffs[1]) if np.abs(coeffs[1]) > 1e-12 else (np.abs(coeffs).max() + 1e-12)
    descriptors = coeffs.copy() / scale
    # Truncate: keep DC + ±1..±n_harmonics
    n = len(descriptors)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    for k in range(1, n_harmonics + 1):
        mask[k % n] = True
        mask[(-k) % n] = True
    truncated = np.where(mask, descriptors, 0.0)

    recon_z = np.fft.ifft(truncated * scale)
    reconstruction = np.column_stack([recon_z.real, recon_z.imag])

    return {
        "coeffs": coeffs,
        "descriptors": descriptors[mask],
        "descriptor_indices": np.where(mask)[0],
        "n_harmonics": n_harmonics,
        "scale": float(scale),
        "reconstruction": reconstruction,
        "fingerprint": _fingerprint_from_descriptors(descriptors, n_harmonics),
    }


def _fingerprint_from_descriptors(descriptors: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Compact real fingerprint: magnitudes of first n_harmonics modes."""
    mags = []
    n = len(descriptors)
    for k in range(1, n_harmonics + 1):
        mags.append(float(np.abs(descriptors[k % n])))
    arr = np.asarray(mags, dtype=float)
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm


def match_fingerprints(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Cosine similarity between two Fourier fingerprints in [0, 1]."""
    a = np.asarray(fp_a, dtype=float).ravel()
    b = np.asarray(fp_b, dtype=float).ravel()
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))
