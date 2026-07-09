"""Trajectoid outer shell generation with real rolling constraints.

Implements Nature-style trajectoid machinery (simplified but rigorous):

- Path scaling (kx, ky) to minimize final SO(3) orientation mismatch
- Cumulative rolling rotation matrices along the path
- Two-Period Trajectoid (TPT) closure
- Arc-length parametrization + curvature signal
- Fourier fingerprint + 1D phase/trench mask for the modulation layer

Reference idea: trajectoids — 3D bodies that roll along a prescribed planar
path and return with controlled orientation (Sobolev et al., Nature 2023).

Important: unconstrained minimization of ||R_final|| is degenerate (paths
shrink to zero). We therefore search **anisotropic** scales with
**perimeter normalization**, optionally followed by a mild global scale
band around the design rolling radius.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.fourier_descriptors import compute_fourier_descriptors, unroll_curve


@dataclass
class ShellGeometry:
    """Closed trajectoid-style shell with identification descriptors."""

    vertices: np.ndarray  # (N, 2) or (N, 3) closed curve / silhouette
    surface: np.ndarray | None = None  # optional (nu, nv, 3) parametric surface
    fourier_coeffs: np.ndarray | None = None
    fourier_fingerprint: np.ndarray | None = None
    unrolled_path: np.ndarray | None = None
    curvature_signal: np.ndarray | None = None
    total_length: float = 0.0
    # Trajectoid rolling state
    kx: float = 1.0
    ky: float = 1.0
    global_scale: float = 1.0
    mismatch_deg: float = 0.0
    tilt_deg: float = 0.0
    rolling_radius: float = 1.0
    use_tpt: bool = True
    arc_length: np.ndarray | None = None
    phase_trench_mask: np.ndarray | None = None
    rotation_matrices: np.ndarray | None = None  # (M, 3, 3) cumulative SO(3)
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
            "kx": self.kx,
            "ky": self.ky,
            "global_scale": self.global_scale,
            "mismatch_deg": self.mismatch_deg,
            "tilt_deg": self.tilt_deg,
            "rolling_radius": self.rolling_radius,
            "use_tpt": self.use_tpt,
            "arc_length": self.arc_length,
            "phase_trench_mask": self.phase_trench_mask,
            "rotation_matrices": self.rotation_matrices,
            "metadata": self.metadata,
        }


def _payload_digest(payload: str | bytes, seed: int) -> np.ndarray:
    """Deterministic float stream from payload + seed."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256(payload + seed.to_bytes(8, "little", signed=False)).digest()
    buf = bytearray(h)
    while len(buf) < 64:
        buf.extend(hashlib.sha256(bytes(buf[-32:])).digest())
    return np.frombuffer(bytes(buf[:64]), dtype=np.uint8).astype(float) / 255.0


def _payload_hash16(payload: str | bytes, seed: int) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = payload
    return hashlib.sha256(raw + seed.to_bytes(8, "little", signed=False)).hexdigest()[:16]


def _as_xy(path: np.ndarray) -> np.ndarray:
    """Ensure (N, 2) planar path for rolling."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3:
        raise ValueError("path must be (N, D) with N >= 3")
    return path[:, :2].copy()


def _ensure_closed_2d(path: np.ndarray) -> np.ndarray:
    path = _as_xy(path)
    if not np.allclose(path[0], path[-1], atol=1e-9):
        path = np.vstack([path, path[0]])
    return path


def path_perimeter(path: np.ndarray) -> float:
    path = _as_xy(path)
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def renormalize_perimeter(path: np.ndarray, target: float) -> np.ndarray:
    """Uniformly scale a closed/open polyline so its perimeter matches ``target``."""
    path = _as_xy(path)
    closed = np.allclose(path[0], path[-1], atol=1e-9)
    body = path[:-1] if closed else path
    probe = np.vstack([body, body[0]])
    L = path_perimeter(probe)
    if L < 1e-12:
        return _ensure_closed_2d(path)
    body = body * (target / L)
    return np.vstack([body, body[0]])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """3×3 rotation matrix for right-handed rotation by ``angle`` about unit ``axis``."""
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _so3_angle_deg(matrix: np.ndarray) -> float:
    """Geodesic angle of an SO(3) matrix from identity, in degrees ∈ [0, 180]."""
    tr = float(np.trace(matrix))
    cos_t = 0.5 * (tr - 1.0)
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _tilt_deg(matrix: np.ndarray) -> float:
    """How far body +z is from space +z after applying ``matrix`` (degrees)."""
    z = matrix @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def compute_cumulative_rotations(
    path: np.ndarray,
    r: float = 1.0,
) -> tuple[list[R], float, np.ndarray]:
    """
    Compute rolling rotations along a planar path (trajectoid-style).

    At each segment the body rolls by angle ``ds / r`` about the in-plane
    axis perpendicular to the tangent (``e_z × direction``). Rotations are
    composed in the space-fixed convention: ``R ← R_step @ R``.

    Returns
    -------
    rotations
        List of cumulative ``Rotation`` objects (including identity at start).
    mismatch_deg
        SO(3) geodesic from final orientation to identity (degrees).
    matrices
        Stacked rotation matrices ``(M, 3, 3)``.
    """
    path = _as_xy(path)
    r = max(float(r), 1e-9)
    total = np.eye(3)
    matrices = [total.copy()]
    rotations: list[R] = [R.identity()]

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        ds = float(np.linalg.norm(p1 - p0))
        if ds < 1e-10:
            matrices.append(total.copy())
            rotations.append(R.from_matrix(total))
            continue
        direction = (p1 - p0) / ds
        # Space-fixed horizontal axis: e_z × direction
        axis = np.array([-direction[1], direction[0], 0.0], dtype=float)
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-12:
            matrices.append(total.copy())
            rotations.append(R.from_matrix(total))
            continue
        axis /= nrm
        angle = ds / r
        step = _rodrigues(axis, angle)
        total = step @ total
        # Re-orthonormalize occasionally for long paths
        if (i + 1) % 64 == 0:
            u, _, vt = np.linalg.svd(total)
            total = u @ vt
            if np.linalg.det(total) < 0:
                u[:, -1] *= -1
                total = u @ vt
        matrices.append(total.copy())
        rotations.append(R.from_matrix(total))

    mismatch_deg = _so3_angle_deg(total)
    return rotations, mismatch_deg, np.stack(matrices, axis=0)


def _roll_mismatch_fast(path: np.ndarray, r: float) -> tuple[float, float]:
    """Fast mismatch + tilt without storing intermediate frames."""
    path = _as_xy(path)
    r = max(float(r), 1e-9)
    total = np.eye(3)
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        ds = float(np.linalg.norm(p1 - p0))
        if ds < 1e-10:
            continue
        direction = (p1 - p0) / ds
        axis = np.array([-direction[1], direction[0], 0.0], dtype=float)
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-12:
            continue
        axis /= nrm
        total = _rodrigues(axis, ds / r) @ total
    # Final polar re-orthonormalization
    u, _, vt = np.linalg.svd(total)
    total = u @ vt
    if np.linalg.det(total) < 0:
        u[:, -1] *= -1
        total = u @ vt
    return _so3_angle_deg(total), _tilt_deg(total)


def scale_path_for_closure(
    path: np.ndarray,
    *,
    rolling_radius: float = 1.0,
    max_iter: int = 20,
    rng: np.random.Generator | None = None,
    kx_bounds: tuple[float, float] = (0.45, 2.2),
    ky_bounds: tuple[float, float] = (0.45, 2.2),
    grid: int = 7,
    preserve_perimeter: bool = True,
    global_scale_bounds: tuple[float, float] = (0.75, 1.35),
    global_scale_steps: int = 7,
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Find anisotropic scales (kx, ky) minimizing final orientation mismatch.

    Steps
    -----
    1. Sample (kx, ky) on a grid + seeded random draws.
    2. Apply anisotropic scale; **renormalize perimeter** to the original
       length (avoids the trivial shrink-to-zero minimum).
    3. Mild global scale band around 1 (effective path/radius ratio).
    4. Local coordinate descent polish.

    Returns
    -------
    best_path, kx, ky, global_scale, mismatch_deg
    """
    path = _ensure_closed_2d(path)
    if rng is None:
        rng = np.random.default_rng(0)

    L0 = path_perimeter(path)
    r = rolling_radius

    candidates: list[tuple[float, float]] = [(1.0, 1.0)]
    for kx in np.linspace(kx_bounds[0], kx_bounds[1], grid):
        for ky in np.linspace(ky_bounds[0], ky_bounds[1], grid):
            candidates.append((float(kx), float(ky)))
    for _ in range(max_iter):
        candidates.append(
            (float(rng.uniform(*kx_bounds)), float(rng.uniform(*ky_bounds)))
        )

    g_scales = np.linspace(global_scale_bounds[0], global_scale_bounds[1], global_scale_steps)

    best_mismatch = float("inf")
    best_kx, best_ky, best_g = 1.0, 1.0, 1.0
    best_path = path.copy()

    def evaluate(kx: float, ky: float, g: float) -> tuple[float, np.ndarray]:
        body = path[:-1] * np.array([kx, ky], dtype=float)
        scaled = np.vstack([body, body[0]])
        if preserve_perimeter:
            scaled = renormalize_perimeter(scaled, L0)
        scaled = scaled * g
        m, _ = _roll_mismatch_fast(scaled, r)
        return m, scaled

    for kx, ky in candidates:
        for g in g_scales:
            m, scaled = evaluate(kx, ky, float(g))
            if m < best_mismatch:
                best_mismatch = m
                best_kx, best_ky, best_g = kx, ky, float(g)
                best_path = scaled

    # Coordinate descent polish
    for step in (0.08, 0.03, 0.01):
        improved = True
        while improved:
            improved = False
            for dkx, dky, dg in (
                (step, 0, 0),
                (-step, 0, 0),
                (0, step, 0),
                (0, -step, 0),
                (0, 0, step * 0.5),
                (0, 0, -step * 0.5),
                (step, step, 0),
                (-step, step, 0),
                (step, -step, 0),
                (-step, -step, 0),
            ):
                kx = float(np.clip(best_kx + dkx, *kx_bounds))
                ky = float(np.clip(best_ky + dky, *ky_bounds))
                g = float(np.clip(best_g + dg, *global_scale_bounds))
                m, scaled = evaluate(kx, ky, g)
                if m + 1e-9 < best_mismatch:
                    best_mismatch = m
                    best_kx, best_ky, best_g = kx, ky, g
                    best_path = scaled
                    improved = True

    return best_path, best_kx, best_ky, best_g, best_mismatch


def two_period_trajectoid_closure(
    path: np.ndarray,
    *,
    n_bridge: int = 10,
) -> np.ndarray:
    """
    Simple TPT-style closure: first period + 180° planar inversion + bridge.

    The second period is the path inverted through its centroid (planar 180°
    about z), joined by short linear bridges. Result is re-closed to the
    first point for Fourier / rolling analysis.
    """
    path = _as_xy(path)
    if np.allclose(path[0], path[-1], atol=1e-9):
        p1 = path[:-1].copy()
    else:
        p1 = path.copy()

    center = p1.mean(axis=0)
    p2 = 2.0 * center - p1  # 180° point reflection in the plane

    bridge = np.linspace(p1[-1], p2[0], n_bridge, endpoint=False)[1:]
    bridge_back = np.linspace(p2[-1], p1[0], n_bridge, endpoint=False)[1:]

    return np.vstack([p1, bridge, p2, bridge_back, p1[0:1]])


def _base_path_from_payload(
    payload: str | bytes,
    seed: int,
    *,
    n_points: int = 256,
    n_harmonics: int = 12,
    base_radius: float = 1.0,
    amplitude_scale: float = 0.35,
) -> np.ndarray:
    """Unique closed planar curve from payload hash (harmonic radial series)."""
    digest = _payload_digest(payload, seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    phase0 = float(digest[0]) * 2.0 * np.pi

    r = np.full_like(t, base_radius)
    for k in range(1, n_harmonics + 1):
        a = amplitude_scale * (2.0 * digest[k % len(digest)] - 1.0) / (1.0 + 0.15 * k)
        b = amplitude_scale * (2.0 * digest[(k + 17) % len(digest)] - 1.0) / (1.0 + 0.15 * k)
        r = r + a * np.cos(k * t + phase0) + b * np.sin(k * t + 0.5 * phase0)

    r = np.clip(r, 0.2 * base_radius, None)
    path = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return np.vstack([path, path[0]])


def _arc_length_and_curvature(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Arc-length samples (N,) and curvature proxy (N,) for a polyline."""
    path = _as_xy(path)
    diffs = np.diff(path, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(arc[-1]) if arc[-1] > 0 else 1.0

    heading = np.arctan2(diffs[:, 1], diffs[:, 0])
    heading_u = np.unwrap(heading)
    dheading = np.gradient(heading_u)
    ds = np.clip(seglen, 1e-12, None)
    kappa_seg = dheading / ds
    kappa = np.concatenate([kappa_seg, kappa_seg[-1:]])
    if len(kappa) != len(path):
        kappa = np.interp(
            np.linspace(0, 1, len(path)),
            np.linspace(0, 1, len(kappa)),
            kappa,
        )
    return arc, kappa, total


def _phase_trench_mask(arc_lengths: np.ndarray, mismatch_deg: float) -> np.ndarray:
    """1D phase / trench signal along the path for the modulation layer."""
    total = float(arc_lengths[-1]) if len(arc_lengths) and arc_lengths[-1] > 0 else 1.0
    atten = float(np.clip(1.0 - mismatch_deg / 180.0, 0.05, 1.0))
    return np.sin(arc_lengths * 2.0 * np.pi / total) * atten


def _embed_xyz(path2d: np.ndarray, curvature: np.ndarray | None = None) -> np.ndarray:
    """Lift planar path to 3D with mild z from curvature (viz / surface)."""
    xy = _as_xy(path2d)
    if curvature is not None and len(curvature) == len(xy):
        z = 0.08 * np.tanh(curvature)
    else:
        z = np.zeros(len(xy))
    return np.column_stack([xy[:, 0], xy[:, 1], z])


def _parametric_surface(curve: np.ndarray, *, n_meridian: int = 32) -> np.ndarray:
    """Soft macadamia-like surface around the silhouette."""
    if curve.shape[1] == 2:
        curve = np.column_stack([curve, np.zeros(len(curve))])
    xy = curve[:-1, :2] if np.allclose(curve[0], curve[-1], atol=1e-9) else curve[:, :2]
    z = curve[: len(xy), 2] if curve.shape[1] > 2 else np.zeros(len(xy))
    r_eq = np.linalg.norm(xy, axis=1)
    n_theta = len(r_eq)
    theta = np.linspace(0.0, 2.0 * np.pi, n_meridian, endpoint=False)
    surface = np.zeros((n_theta, n_meridian, 3))
    for i in range(n_theta):
        a = float(r_eq[i])
        b = 0.55 * a + 0.1 * abs(float(z[i]))
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
    use_tpt: bool = True,
    rolling_radius: float = 1.0,
    scale_max_iter: int = 20,
    scale_grid: int = 7,
) -> ShellGeometry:
    """
    Create a unique shell with real trajectoid rolling constraints.

    Parameters
    ----------
    payload, seed
        Identity of the seed (hash → base curve; seed → scale search RNG).
    use_tpt
        If True, apply two-period trajectoid closure before final rolling analysis.
    rolling_radius
        Effective sphere/body radius ``r`` in ``angle = ds / r``.
    """
    rng = np.random.default_rng(seed)

    # 1. Unique closed planar base from payload
    base_path = _base_path_from_payload(
        payload,
        seed,
        n_points=n_points,
        n_harmonics=n_harmonics,
        base_radius=base_radius,
        amplitude_scale=amplitude_scale,
    )
    _, mismatch_base, _ = compute_cumulative_rotations(base_path, r=rolling_radius)

    # 2. Path scaling for orientation closure (trajectoid trick)
    scaled_path, kx, ky, gscale, mismatch_scaled = scale_path_for_closure(
        base_path,
        rolling_radius=rolling_radius,
        max_iter=scale_max_iter,
        rng=rng,
        grid=scale_grid,
    )

    # 3. Optional TPT closure (two periods)
    apply_tpt = bool(use_tpt or trajectoid_periods >= 2)
    if apply_tpt:
        final_path = two_period_trajectoid_closure(scaled_path)
    else:
        final_path = _ensure_closed_2d(scaled_path)

    # 4. Rolling rotations + mismatch on the final path
    _rots, mismatch, matrices = compute_cumulative_rotations(
        final_path, r=rolling_radius
    )
    tilt = _tilt_deg(matrices[-1])

    # 5. Arc-length + curvature
    arc_lengths, curvature, total_length = _arc_length_and_curvature(final_path)
    phase_mask = _phase_trench_mask(arc_lengths, mismatch)

    # 6. Embed to 3D for surface / viz; Fourier on planar projection
    vertices = _embed_xyz(final_path, curvature)
    surface = _parametric_surface(vertices)

    fd = compute_fourier_descriptors(vertices, n_harmonics=n_harmonics)
    unrolled = unroll_curve(
        vertices, n_samples=min(n_points, max(len(vertices) - 1, 8))
    )

    meta = {
        "seed": seed,
        "n_points": n_points,
        "n_harmonics": n_harmonics,
        "trajectoid_periods": 2 if apply_tpt else 1,
        "payload_hash": _payload_hash16(payload, seed),
        "kx": kx,
        "ky": ky,
        "global_scale": gscale,
        "mismatch_deg": mismatch,
        "tilt_deg": tilt,
        "mismatch_base_deg": mismatch_base,
        "mismatch_scaled_deg": mismatch_scaled,
        "rolling_radius": rolling_radius,
        "use_tpt": apply_tpt,
        "n_rotations": len(matrices),
        "perimeter": total_length,
    }

    return ShellGeometry(
        vertices=vertices,
        surface=surface,
        fourier_coeffs=fd["coeffs"],
        fourier_fingerprint=fd["fingerprint"],
        unrolled_path=unrolled["path"],
        curvature_signal=curvature,
        total_length=total_length if total_length > 0 else unrolled["total_length"],
        kx=kx,
        ky=ky,
        global_scale=gscale,
        mismatch_deg=mismatch,
        tilt_deg=tilt,
        rolling_radius=rolling_radius,
        use_tpt=apply_tpt,
        arc_length=arc_lengths,
        phase_trench_mask=phase_mask,
        rotation_matrices=matrices,
        metadata=meta,
    )
