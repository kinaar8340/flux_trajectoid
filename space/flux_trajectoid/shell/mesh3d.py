"""3D trajectoid body construction (sphere + rolling shave + contact trench).

Builds a macadamia / asteroid–like solid from:

1. A **sphere** of design radius ``r`` (rolling radius)
2. The **planar rolling path** + cumulative SO(3) orientations
3. A **contact curve** on the sphere (body-frame locus of the contact point)
4. **Oriented cutting planes** (shaving procedure) that trim the sphere so
   the body stays above the ground plane in every pose along the path
5. A **potential trench** groove along the contact curve (optical / ID channel)

This is a practical, numpy-only approximation of the Nature (2023) trajectoid
construction — enough to give a true 3D shell mesh, rolling-consistent contact
geometry, and a richer modulation surface than a 2D silhouette alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoidMesh3D:
    """Triangle mesh + UV surface of a 3D trajectoid shell."""

    vertices: np.ndarray  # (Nv, 3) mesh vertices
    faces: np.ndarray  # (Nf, 3) triangle indices
    surface: np.ndarray  # (n_lat, n_lon, 3) UV sphere surface after shaving
    radial_map: np.ndarray  # (n_lat, n_lon) radius field
    path_on_body: np.ndarray  # (Np, 3) contact curve in body frame
    cutting_normals: np.ndarray  # (Nc, 3) plane normals (body frame)
    sphere_radius: float
    volume_proxy: float
    mean_radius: float
    contact_coverage: float  # fraction of sphere near contact trench

    def as_dict(self) -> dict:
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "surface": self.surface,
            "radial_map": self.radial_map,
            "path_on_body": self.path_on_body,
            "cutting_normals": self.cutting_normals,
            "sphere_radius": self.sphere_radius,
            "volume_proxy": self.volume_proxy,
            "mean_radius": self.mean_radius,
            "contact_coverage": self.contact_coverage,
        }


def uv_sphere_grid(
    n_lat: int = 48,
    n_lon: int = 96,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Regular UV sphere.

    Returns
    -------
    surface : (n_lat, n_lon, 3)
    theta : (n_lat,) polar angle [0, π]
    phi : (n_lon,) azimuth [0, 2π)
    directions : (n_lat, n_lon, 3) unit vectors
    """
    theta = np.linspace(0.0, np.pi, n_lat)
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    directions = np.stack([x, y, z], axis=-1)
    surface = radius * directions
    return surface, theta, phi, directions


def grid_to_mesh(
    surface: np.ndarray,
    *,
    close_lon: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (n_lat, n_lon, 3) grid to (vertices, faces) triangle mesh."""
    n_lat, n_lon, _ = surface.shape
    # Duplicate seam if closed
    if close_lon:
        surface = np.concatenate([surface, surface[:, :1, :]], axis=1)
        n_lon = surface.shape[1]

    vertices = surface.reshape(-1, 3)
    faces: list[list[int]] = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            a = i * n_lon + j
            b = i * n_lon + (j + 1)
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1)
            # Skip degenerate polar triangles
            if np.linalg.norm(vertices[a] - vertices[b]) < 1e-14:
                faces.append([a, c, d])
            elif np.linalg.norm(vertices[c] - vertices[d]) < 1e-14:
                faces.append([a, b, c])
            else:
                faces.append([a, b, d])
                faces.append([a, d, c])
    return vertices, np.asarray(faces, dtype=np.int64)


def contact_curve_body_frame(
    rotation_matrices: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Body-frame contact points along the rolling path.

    Convention: ``R`` maps body → space (space-fixed composition used in
    ``compute_cumulative_rotations``). Contact in space is along −e_z relative
    to the sphere center, so body contact is ``Rᵀ (−r ê_z)``.
    """
    mats = np.asarray(rotation_matrices, dtype=float)
    if mats.ndim != 3 or mats.shape[1:] != (3, 3):
        raise ValueError("rotation_matrices must be (N, 3, 3)")
    down = np.array([0.0, 0.0, -float(radius)])
    # p_body = R.T @ down_space
    pts = np.einsum("nij,j->ni", mats.transpose(0, 2, 1), down)
    # Normalize onto sphere of radius r (numerical drift)
    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return pts * (radius / norms)


def cutting_plane_normals(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Oriented cutting-plane normals in the body frame.

    Each pose contributes the body-frame image of the ground normal ê_z
    (plane that supports the body from below).
    """
    mats = np.asarray(rotation_matrices, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    # n_body = R.T @ ê_z
    normals = np.einsum("nij,j->ni", mats.transpose(0, 2, 1), up)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / norms


def spherical_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance between unit vectors (radians)."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    dots = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dots)


def min_distance_to_curve(
    directions: np.ndarray,
    curve_unit: np.ndarray,
) -> np.ndarray:
    """Min spherical distance from each direction to a curve of unit vectors."""
    # directions: (..., 3), curve: (M, 3)
    d = directions.reshape(-1, 3)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    c = curve_unit / (np.linalg.norm(curve_unit, axis=1, keepdims=True) + 1e-12)
    # (N, M)
    dots = np.clip(d @ c.T, -1.0, 1.0)
    dist = np.arccos(dots).min(axis=1)
    return dist.reshape(directions.shape[:-1])


def shave_radial_field(
    directions: np.ndarray,
    *,
    radius: float,
    path_on_body: np.ndarray,
    cutting_normals: np.ndarray,
    trench_depth: float = 0.08,
    trench_width_rad: float = 0.18,
    cut_margin: float = 0.02,
    max_cut_fraction: float = 0.22,
    curvature_signal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute radius field after contact trench + oriented half-space shaves.

    Parameters
    ----------
    directions
        (..., 3) unit directions on the sphere.
    path_on_body
        (N, 3) contact curve in body frame (length ≈ radius).
    cutting_normals
        (N, 3) body-frame ground normals along the path.
    """
    r0 = float(radius)
    shape = directions.shape[:-1]
    dirs = directions.reshape(-1, 3)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    curve = path_on_body / (np.linalg.norm(path_on_body, axis=1, keepdims=True) + 1e-12)
    dist = min_distance_to_curve(dirs.reshape(shape + (3,)), curve).ravel()

    # Contact trench (potential groove)
    trench = trench_depth * np.exp(-(dist**2) / (2.0 * trench_width_rad**2))

    # Optional curvature-modulated trench depth along nearest path index
    if curvature_signal is not None and len(curvature_signal) == len(curve):
        dots = np.clip(dirs @ curve.T, -1.0, 1.0)
        nn = np.argmax(dots, axis=1)
        kappa = np.asarray(curvature_signal, dtype=float)
        k = np.abs(kappa[nn])
        k = k / (k.max() + 1e-12)
        trench = trench * (0.65 + 0.7 * k)

    # Oriented cutting boxes / half-spaces:
    # A supporting plane in pose i: n·x >= -r + margin in body frame for
    # points on the body. Sphere point p = ρ û must satisfy
    # ρ (n·û) >= -r + margin  for all n with n·û < 0 (lower hemisphere cuts).
    # ⇒ ρ <= (-r + margin) / (n·û) when n·û < 0.
    radial = np.full(dirs.shape[0], r0, dtype=float)
    max_cut = r0 * max_cut_fraction
    nrm = cutting_normals / (np.linalg.norm(cutting_normals, axis=1, keepdims=True) + 1e-12)

    # Subsample normals for speed (keep ~48)
    step = max(1, len(nrm) // 48)
    nrm_s = nrm[::step]
    for n in nrm_s:
        proj = dirs @ n  # (N,)
        mask = proj < -0.05
        if not np.any(mask):
            continue
        # Plane limit: center is at 0, ground support at -r along n when n aligns with contact
        limit = (-r0 + cut_margin) / (proj[mask] + 1e-12)
        # Only shave (never grow beyond sphere)
        limit = np.clip(limit, r0 - max_cut, r0)
        radial[mask] = np.minimum(radial[mask], limit)

    radial = radial - trench
    radial = np.clip(radial, r0 * (1.0 - max_cut_fraction - 0.05), r0 * 1.02)
    return radial.reshape(shape)


def build_trajectoid_mesh3d(
    rotation_matrices: np.ndarray,
    *,
    radius: float = 1.0,
    n_lat: int = 40,
    n_lon: int = 80,
    trench_depth: float = 0.08,
    trench_width_rad: float = 0.18,
    curvature_signal: np.ndarray | None = None,
) -> TrajectoidMesh3D:
    """
    Full 3D trajectoid mesh from rolling orientations.

    Steps: UV sphere → contact curve → oriented shaves + trench → triangle mesh.
    """
    r = float(radius)
    surface0, _theta, _phi, directions = uv_sphere_grid(n_lat, n_lon, radius=r)
    path_body = contact_curve_body_frame(rotation_matrices, radius=r)
    normals = cutting_plane_normals(rotation_matrices)

    radial = shave_radial_field(
        directions,
        radius=r,
        path_on_body=path_body,
        cutting_normals=normals,
        trench_depth=trench_depth,
        trench_width_rad=trench_width_rad,
        curvature_signal=curvature_signal,
    )
    surface = radial[..., None] * directions
    vertices, faces = grid_to_mesh(surface, close_lon=True)

    # Coverage: fraction of surface points significantly trenched
    trench_mask = radial < (r * 0.97)
    coverage = float(np.mean(trench_mask))

    # Volume proxy via mean radius^3
    mean_r = float(np.mean(radial))
    volume_proxy = float((4.0 / 3.0) * np.pi * mean_r**3)

    return TrajectoidMesh3D(
        vertices=vertices,
        faces=faces,
        surface=surface,
        radial_map=radial,
        path_on_body=path_body,
        cutting_normals=normals,
        sphere_radius=r,
        volume_proxy=volume_proxy,
        mean_radius=mean_r,
        contact_coverage=coverage,
    )


def orthographic_silhouette(
    mesh_vertices: np.ndarray,
    *,
    axis: str = "z",
    n_angles: int = 256,
) -> np.ndarray:
    """
    Extract a closed 2D silhouette polygon from a 3D mesh via radial max.

    Projects out ``axis``, then takes the outer envelope in polar angle.
    """
    v = np.asarray(mesh_vertices, dtype=float)
    if axis == "z":
        xy = v[:, :2]
    elif axis == "y":
        xy = v[:, [0, 2]]
    else:
        xy = v[:, [1, 2]]
    center = xy.mean(axis=0)
    xy_c = xy - center
    ang = np.arctan2(xy_c[:, 1], xy_c[:, 0])
    rad = np.linalg.norm(xy_c, axis=1)
    bins = np.linspace(-np.pi, np.pi, n_angles + 1)
    silhouette = []
    for i in range(n_angles):
        m = (ang >= bins[i]) & (ang < bins[i + 1])
        if not np.any(m):
            # empty bin — interpolate later
            silhouette.append([np.nan, np.nan])
            continue
        j = np.argmax(rad[m])
        idx = np.where(m)[0][j]
        silhouette.append(xy[idx])
    sil = np.asarray(silhouette, dtype=float)
    # Fill NaNs by circular interpolation
    for dim in range(2):
        col = sil[:, dim]
        nans = np.isnan(col)
        if np.any(nans) and np.any(~nans):
            x = np.arange(len(col))
            col[nans] = np.interp(x[nans], x[~nans], col[~nans], period=len(col))
            sil[:, dim] = col
    if not np.allclose(sil[0], sil[-1]):
        sil = np.vstack([sil, sil[0]])
    return sil
