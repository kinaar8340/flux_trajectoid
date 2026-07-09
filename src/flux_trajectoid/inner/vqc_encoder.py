"""VQC quaternion + OAM shard encoding (adapted from vqc_proto orbital_braille).

Maps payload → quaternion shards → Laguerre–Gaussian OAM mode weights
(Orbital Braille imprint on {0, 1, −1, 2, 3}).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import genlaguerre

from ..utils.quaternion_utils import (
    Quaternion,
    bytes_to_quaternion_shards,
    quaternion_shards_to_bytes,
)

# Match vqc_proto quaternion_oam conventions
PHI_SCALE = 0.3
IMPRINT_SCALE = 0.12
CARRIER_ELL = 1
OAM_QUAT_ELLS = (0, 1, -1, 2)  # x, w-carrier, y, z


@dataclass
class QuaternionEncoding:
    """Encoded inner payload as quaternion shards + OAM spectrum."""

    payload_bytes: bytes
    shards: list[Quaternion]
    oam_weights: list[dict[int, complex]]  # per shard
    composite_weights: dict[int, complex]
    fields: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_quaternion(self) -> Quaternion:
        """First shard as representative quaternion."""
        return self.shards[0] if self.shards else Quaternion()


def _lg_mode(ell: int, rho: np.ndarray, phi: np.ndarray, w0: float = 1.0) -> np.ndarray:
    """p=0 Laguerre–Gaussian complex field on polar grid."""
    L = abs(int(ell))
    try:
        norm = np.sqrt(2.0 / (np.pi * math.factorial(L))) / w0
    except ValueError:
        norm = 1.0 / w0
    rw = rho / w0
    x = 2.0 * rw**2
    lag = genlaguerre(0, L)(x)
    radial = norm * (np.sqrt(2.0) ** L) * (rw**L) * np.exp(-(rw**2)) * lag
    return radial * np.exp(1j * ell * phi)


def quaternion_to_oam_weights(q: Quaternion, *, imprint_scale: float = IMPRINT_SCALE) -> dict[int, complex]:
    """
    Imprint quaternion on low-order OAM modes (vqc_proto style).

    ΔE ⊃ imprint · (q.x·LG₀ + i·q.y·LG₋₁ + q.z·LG₂) + carrier LG₁ with
    Rodrigues phase from q.w.
    """
    phi_q = PHI_SCALE * np.cos(q.w * np.pi / 2.0)
    weights: dict[int, complex] = {
        0: imprint_scale * q.x,
        CARRIER_ELL: np.exp(1j * phi_q),
        -1: 1j * imprint_scale * q.y,
        2: imprint_scale * q.z,
    }
    return weights


def oam_weights_to_field(
    weights: dict[int, complex],
    *,
    grid_size: int = 64,
    extent: float = 2.0,
    w0: float = 0.6,
) -> np.ndarray:
    """Synthesize complex field from OAM spectrum."""
    xs = np.linspace(-extent, extent, grid_size)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    field = np.zeros((grid_size, grid_size), dtype=complex)
    for ell, amp in weights.items():
        field = field + amp * _lg_mode(int(ell), rho, phi, w0=w0)
    return field


def encode_to_quaternion(
    payload: str | bytes,
    *,
    n_shards: int = 8,
    grid_size: int = 64,
    build_fields: bool = True,
) -> QuaternionEncoding:
    """
    Encode payload into quaternion shards + OAM weights (and optional fields).

    Parameters
    ----------
    payload
        Text or raw bytes to pack into the nut / inner carrier.
    n_shards
        Number of quaternion shards (DNA-like packing density).
    """
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = bytes(payload)

    shards = bytes_to_quaternion_shards(raw, n_shards=n_shards)
    oam_weights = [quaternion_to_oam_weights(q) for q in shards]

    # Composite: coherent sum of shard weights (normalized)
    composite: dict[int, complex] = {}
    for w in oam_weights:
        for ell, amp in w.items():
            composite[ell] = composite.get(ell, 0.0) + amp
    scale = sum(abs(v) ** 2 for v in composite.values()) ** 0.5 + 1e-12
    composite = {k: v / scale for k, v in composite.items()}

    fields: list[np.ndarray] = []
    if build_fields:
        fields = [oam_weights_to_field(w, grid_size=grid_size) for w in oam_weights]

    return QuaternionEncoding(
        payload_bytes=raw,
        shards=shards,
        oam_weights=oam_weights,
        composite_weights=composite,
        fields=fields,
        metadata={
            "n_shards": n_shards,
            "n_bytes": len(raw),
            "ells": list(composite.keys()),
            "phi_scale": PHI_SCALE,
            "imprint_scale": IMPRINT_SCALE,
        },
    )


def decode_quaternion_approx(
    encoding: QuaternionEncoding,
    n_bytes: int | None = None,
) -> bytes:
    """Approximate byte recovery from stored quaternion shards."""
    n = n_bytes if n_bytes is not None else encoding.metadata.get("n_bytes")
    return quaternion_shards_to_bytes(encoding.shards, n_bytes=n)
