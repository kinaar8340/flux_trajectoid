"""VQC quaternion + OAM shard encoding (adapted from vqc_proto orbital_braille).

Maps payload → ShardPack (unit q + scale) → Laguerre–Gaussian OAM weights.
Scale is imprinted on the carrier amplitude so photonic recovery can invert
the 4-byte blocks with far less loss than unit-only quaternions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import genlaguerre

from ..utils.quaternion_utils import (
    Quaternion,
    ShardPack,
    crc8,
    pack_payload,
    scale_to_carrier_amp,
    unpack_payload,
)

# Match vqc_proto quaternion_oam conventions
PHI_SCALE = 0.3
IMPRINT_SCALE = 0.12
CARRIER_ELL = 1
OAM_QUAT_ELLS = (0, 1, -1, 2)  # x, w-carrier, y, z
SCALE_MAX = 2.0


@dataclass
class QuaternionEncoding:
    """Encoded inner payload as quaternion shards + OAM spectrum."""

    payload_bytes: bytes
    shards: list[Quaternion]
    packs: list[ShardPack]
    oam_weights: list[dict[int, complex]]  # per shard
    composite_weights: dict[int, complex]
    fields: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_quaternion(self) -> Quaternion:
        return self.shards[0] if self.shards else Quaternion()

    @property
    def scales(self) -> list[float]:
        return [p.scale for p in self.packs]


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


def quaternion_to_oam_weights(
    q: Quaternion,
    *,
    scale: float = 1.0,
    imprint_scale: float = IMPRINT_SCALE,
    scale_max: float = SCALE_MAX,
) -> dict[int, complex]:
    """
    Imprint quaternion + scale on low-order OAM modes.

    - Carrier LG₁: amplitude encodes ``scale``, phase encodes q.w (Rodrigues)
    - LG₀ / LG₋₁ / LG₂: imprint q.x, q.y, q.z
    """
    # Odd map in w (sin) so sign(w) is recoverable — cos(w·π/2) is even and
    # loses the hemisphere, which breaks invertible byte packing on S³.
    phi_q = PHI_SCALE * np.sin(q.w * np.pi / 2.0)
    amp = scale_to_carrier_amp(scale, scale_max=scale_max)
    weights: dict[int, complex] = {
        0: imprint_scale * q.x,
        CARRIER_ELL: amp * np.exp(1j * phi_q),
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
    redundancy: int = 1,
) -> QuaternionEncoding:
    """
    Encode payload into ShardPacks + OAM weights (and optional fields).

    Parameters
    ----------
    redundancy
        If >1, each logical 4-byte chunk is repeated across that many shards
        (majority-vote friendly).
    """
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = bytes(payload)

    packs = pack_payload(raw, n_shards=n_shards, redundancy=redundancy)
    shards = [p.q for p in packs]
    oam_weights = [
        quaternion_to_oam_weights(p.q, scale=p.scale) for p in packs
    ]

    # Composite: coherent sum of shard weights (normalized for lattice coupling)
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
        packs=packs,
        oam_weights=oam_weights,
        composite_weights=composite,
        fields=fields,
        metadata={
            "n_shards": n_shards,
            "n_bytes": len(raw),
            "ells": list(composite.keys()),
            "phi_scale": PHI_SCALE,
            "imprint_scale": IMPRINT_SCALE,
            "scale_max": SCALE_MAX,
            "redundancy": redundancy,
            "crc8": crc8(raw),
            "scales": [p.scale for p in packs],
            "invertible": True,
        },
    )


def decode_quaternion_digital(
    encoding: QuaternionEncoding,
    n_bytes: int | None = None,
) -> bytes:
    """Lossless digital recovery from stored ShardPack blocks."""
    n = n_bytes if n_bytes is not None else encoding.metadata.get("n_bytes")
    red = int(encoding.metadata.get("redundancy", 1))
    return unpack_payload(encoding.packs, n_bytes=n, redundancy=red, use_stored_blocks=True)


def decode_quaternion_approx(
    encoding: QuaternionEncoding,
    n_bytes: int | None = None,
) -> bytes:
    """Recover via q×scale (uses stored scales; still near-lossless digitally)."""
    n = n_bytes if n_bytes is not None else encoding.metadata.get("n_bytes")
    red = int(encoding.metadata.get("redundancy", 1))
    return unpack_payload(encoding.packs, n_bytes=n, redundancy=red, use_stored_blocks=False)
