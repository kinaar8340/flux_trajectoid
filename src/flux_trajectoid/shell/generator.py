"""Trajectoid-inspired outer shell generation + Fourier + arc-length unroll.

The shell is a hard, uniquely shaped protective geometry derived from a
payload hash and seed. It provides identification (Fourier fingerprint),
metadata (unrolled curvature signal), and a physical/optical boundary
that later modulates inner OAM modes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.fourier_descriptors import compute_fourier_descriptors, unroll_curve


@dataclass
class ShellGeometry:
    """Closed trajectoid-style shell with identification descriptors."""

    vertices: np.ndarray  # (N, 3) closed curve / silhouette
    surface: np.ndarray | None = None  # optional (nu, nv, 3) parametric surface
    fourier_coeffs: np.ndarray | None = None
    fourier_fingerprint: np.ndarray | None = None
    unrolled_path: np.ndarray | None = None
    curvature_signal: np.ndarray | None = None
    total_length: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vertices": self.vertices,
            "surface": self.surface,
            "fourier_coeffs": self.fourier_coeffs,
            "fourier_fingerprint": self.fourier_fingerprint,
            "unrolled_path": self.unrolled_path,
            "curvature_signal": self.curvature_signal,
            "total_length": self.total_length,
            "metadata": self.metadata,
        }


def _payload_digest(payload: str | bytes, seed: int) -> np.ndarray:
    """Deterministic float stream from payload + seed."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256(payload + seed.to_bytes(8, "little", signed=False)).digest()
    # Expand via repeated hashing for enough coefficients
    buf = bytearray(h)
    while len(buf) < 64:
        buf.extend(hashlib.sha256(bytes(buf[-32:])).digest())
    arr = np.frombuffer(bytes(buf[:64]), dtype=np.uint8).astype(float) / 255.0
    return arr


def _trajectoid_closed_curve(
    digest: np.ndarray,
    *,
    n_points: int = 256,
    n_harmonics: int = 12,
    periods: int = 2,
    base_radius: float = 1.0,
    amplitude_scale: float = 0.35,
) -> np.ndarray:
    """
    Build a unique closed parametric curve (trajectoid-inspired).

    Simplified two-period construction: a near-circular base with harmonic
    radial and vertical modulation so that a 2π sweep of the rolling
    parameter closes after ``periods`` full rotations (toy model of a
    trajectoid's developable rolling constraint).
    """
    t = np.linspace(0.0, 2.0 * np.pi * periods, n_points, endpoint=False)

    # Harmonic coefficients from digest
    a = amplitude_scale * (2.0 * digest[:n_harmonics] - 1.0)
    b = amplitude_scale * (2.0 * digest[n_harmonics : 2 * n_harmonics] - 1.0)
    c = 0.5 * amplitude_scale * (2.0 * digest[2 * n_harmonics : 3 * n_harmonics] - 1.0)

    r = np.full_like(t, base_radius)
    z = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        # Two-period locking: frequencies are integer multiples of 1/periods
        omega = k / float(periods)
        r = r + a[k - 1] * np.cos(omega * t) + b[k - 1] * np.sin(omega * t)
        z = z + c[k - 1] * np.sin(omega * t + digest[k % len(digest)] * np.pi)

    r = np.clip(r, 0.15 * base_radius, None)
    # Map to single closed loop in xy by folding period into one turn
    phi = t / float(periods)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    curve = np.column_stack([x, y, z])
    # Close explicitly
    return np.vstack([curve, curve[0]])


def _parametric_surface(
    curve: np.ndarray,
    *,
    n_meridian: int = 32,
) -> np.ndarray:
    """
    Sweep a soft radial profile along the closed silhouette to form a
    macadamia-like shell surface (toy solid of revolution around path).
    """
    # Use equatorial (x,y) radius and z of silhouette as generating curve
    xy = curve[:-1, :2]
    z = curve[:-1, 2]
    r_eq = np.linalg.norm(xy, axis=1)
    n_theta = len(r_eq)
    theta = np.linspace(0.0, 2.0 * np.pi, n_meridian, endpoint=False)

    surface = np.zeros((n_theta, n_meridian, 3))
    for i in range(n_theta):
        # Ellipsoidal cross-section scaled by local r_eq
        a = r_eq[i]
        b = 0.55 * r_eq[i] + 0.1 * abs(z[i])
        surface[i, :, 0] = a * np.cos(theta)
        surface[i, :, 1] = a * np.sin(theta) * 0.0 + xy[i, 1] * 0.0  # placeholder
        # Better: rotate local ellipse about z, centered on silhouette point
        cx, cy = xy[i]
        surface[i, :, 0] = cx + (a * 0.25) * np.cos(theta)
        surface[i, :, 1] = cy + (a * 0.25) * np.sin(theta)
        surface[i, :, 2] = z[i] + b * np.sin(theta) * 0.3
    return surface


def generate_shell(
    payload: str | bytes,
    seed: int = 42,
    *,
    n_points: int = 256,
    n_harmonics: int = 12,
    trajectoid_periods: int = 2,
    base_radius: float = 1.0,
    amplitude_scale: float = 0.35,
) -> ShellGeometry:
    """
    Create a unique closed shell from payload hash + seed.

    Steps
    -----
    1. Hash payload → harmonic coefficients
    2. Trajectoid-style two-period closed curve
    3. Fourier descriptors (boundary → coefficients / fingerprint)
    4. Arc-length unrolling → 1D metadata signal
    """
    digest = _payload_digest(payload, seed)
    vertices = _trajectoid_closed_curve(
        digest,
        n_points=n_points,
        n_harmonics=n_harmonics,
        periods=trajectoid_periods,
        base_radius=base_radius,
        amplitude_scale=amplitude_scale,
    )
    surface = _parametric_surface(vertices)

    fd = compute_fourier_descriptors(vertices, n_harmonics=n_harmonics)
    unrolled = unroll_curve(vertices, n_samples=n_points)

    meta = {
        "seed": seed,
        "n_points": n_points,
        "n_harmonics": n_harmonics,
        "trajectoid_periods": trajectoid_periods,
        "payload_hash": hashlib.sha256(
            (payload.encode("utf-8") if isinstance(payload, str) else payload)
            + seed.to_bytes(8, "little", signed=False)
        ).hexdigest()[:16],
    }

    return ShellGeometry(
        vertices=vertices,
        surface=surface,
        fourier_coeffs=fd["coeffs"],
        fourier_fingerprint=fd["fingerprint"],
        unrolled_path=unrolled["path"],
        curvature_signal=unrolled["curvature_signal"],
        total_length=unrolled["total_length"],
        metadata=meta,
    )
