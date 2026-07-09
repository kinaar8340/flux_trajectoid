"""Quaternion utilities adapted from vqc_proto orbital_braille.quaternion_codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Quaternion:
    """Unit quaternion (w, x, y, z) for VQC shard encoding."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def norm(self) -> float:
        return float(np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2))

    def normalize(self) -> Quaternion:
        n = self.norm() + 1e-12
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> Quaternion:
        n2 = self.norm() ** 2 + 1e-12
        return Quaternion(self.w / n2, -self.x / n2, -self.y / n2, -self.z / n2)

    def multiply(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Quaternion:
        a = np.asarray(arr, dtype=float).ravel()
        if a.size < 4:
            a = np.pad(a, (0, 4 - a.size))
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3])).normalize()

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, theta: float) -> Quaternion:
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        half = theta / 2.0
        s = np.sin(half)
        return cls(float(np.cos(half)), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))


def rodrigues_rotation(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v around unit axis k by angle theta."""
    k = k / (np.linalg.norm(k) + 1e-12)
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1.0 - np.cos(theta))
    )


def bytes_to_quaternion_shards(payload: bytes, n_shards: int = 8) -> list[Quaternion]:
    """Pack payload bytes into unit-quaternion shards (DNA-like dense packing)."""
    if not payload:
        payload = b"\x00"
    # Pad to multiple of 4 bytes per shard
    chunk = 4
    need = n_shards * chunk
    data = payload[:need].ljust(need, b"\x00")
    shards: list[Quaternion] = []
    for i in range(n_shards):
        block = np.frombuffer(data[i * chunk : (i + 1) * chunk], dtype=np.uint8).astype(float)
        # Map [0,255] → roughly centered, then normalize to S³
        vec = (block / 127.5) - 1.0
        if np.linalg.norm(vec) < 1e-12:
            vec = np.array([1.0, 0.0, 0.0, 0.0])
        shards.append(Quaternion.from_array(vec))
    return shards


def quaternion_shards_to_bytes(shards: list[Quaternion], n_bytes: int | None = None) -> bytes:
    """Approximate inverse of bytes_to_quaternion_shards."""
    out = bytearray()
    for q in shards:
        raw = q.as_array()
        scaled = ((raw + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        out.extend(scaled.tobytes())
    if n_bytes is not None:
        return bytes(out[:n_bytes])
    return bytes(out)
