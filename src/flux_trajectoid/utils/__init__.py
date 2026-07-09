"""Utility helpers for Fourier descriptors and quaternions."""

from .fourier_descriptors import (
    compute_fourier_descriptors,
    match_fingerprints,
    unroll_curve,
)
from .quaternion_utils import (
    Quaternion,
    ShardPack,
    bit_error_rate,
    byte_error_rate,
    bytes_to_quaternion_shards,
    crc8,
    pack_payload,
    quaternion_shards_to_bytes,
    unpack_payload,
)

__all__ = [
    "Quaternion",
    "ShardPack",
    "bytes_to_quaternion_shards",
    "quaternion_shards_to_bytes",
    "pack_payload",
    "unpack_payload",
    "crc8",
    "byte_error_rate",
    "bit_error_rate",
    "compute_fourier_descriptors",
    "unroll_curve",
    "match_fingerprints",
]
