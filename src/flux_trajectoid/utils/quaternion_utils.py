"""Quaternion utilities + invertible shard packing for VQC payloads.

Unit quaternions alone cannot hold 4 free bytes (S³ is 3-DOF). We therefore
pack each 4-byte block as:

  vec ∈ R⁴  (from bytes, centered in [-1, 1])
  q = vec / ||vec||          → OAM imprint (unit)
  s = ||vec||                → scale side-channel (carrier |amp| / metadata)

Digital recovery with stored ``s`` (or original blocks) is lossless.
Photonic recovery estimates ``q`` from OAM modes and ``s`` from carrier
amplitude, then remaps to bytes — much less lossy than unit-only inversion.
"""

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

    def chordal_distance(self, other: Quaternion) -> float:
        """S³ chordal distance identifying q ~ −q."""
        a, b = self.as_array(), other.as_array()
        return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))

    @classmethod
    def from_array(cls, arr: np.ndarray, *, normalize: bool = True) -> Quaternion:
        a = np.asarray(arr, dtype=float).ravel()
        if a.size < 4:
            a = np.pad(a, (0, 4 - a.size))
        q = cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))
        return q.normalize() if normalize else q

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, theta: float) -> Quaternion:
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        half = theta / 2.0
        s = np.sin(half)
        return cls(float(np.cos(half)), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))


@dataclass
class ShardPack:
    """One 4-byte block as unit quaternion + recoverable scale."""

    q: Quaternion
    scale: float  # ||vec|| before normalization, in [0, 2]
    block: bytes  # original 4 bytes (digital ground truth)
    vec: np.ndarray  # centered float vector in R⁴

    def as_dict(self) -> dict:
        return {
            "q": self.q.as_array(),
            "scale": self.scale,
            "block": list(self.block),
            "vec": self.vec.tolist(),
        }


def rodrigues_rotation(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v around unit axis k by angle theta."""
    k = k / (np.linalg.norm(k) + 1e-12)
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1.0 - np.cos(theta))
    )


def bytes_to_vec4(block: bytes | np.ndarray) -> np.ndarray:
    """Map 4 bytes → R⁴ in approximately [-1, 1]."""
    if isinstance(block, (bytes, bytearray)):
        arr = np.frombuffer(bytes(block)[:4].ljust(4, b"\x00"), dtype=np.uint8).astype(float)
    else:
        arr = np.asarray(block, dtype=float).ravel()[:4]
        if arr.size < 4:
            arr = np.pad(arr, (0, 4 - arr.size))
    return (arr / 127.5) - 1.0


def vec4_to_bytes(vec: np.ndarray) -> bytes:
    """Inverse of ``bytes_to_vec4`` with rounding."""
    arr = np.asarray(vec, dtype=float).ravel()[:4]
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    scaled = np.clip(np.rint((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return scaled.tobytes()


def pack_byte_block(block: bytes) -> ShardPack:
    """Lossless-capable pack: unit q + scale + original block."""
    b = bytes(block)[:4].ljust(4, b"\x00")
    vec = bytes_to_vec4(b)
    scale = float(np.linalg.norm(vec))
    if scale < 1e-12:
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        scale = 0.0
        vec = np.zeros(4)
    else:
        q = Quaternion.from_array(vec / scale, normalize=True)
    return ShardPack(q=q, scale=scale, block=b, vec=vec)


def unpack_shard_pack(pack: ShardPack) -> bytes:
    """Perfect digital inverse using stored block (or q·scale reconstruction)."""
    if pack.block and len(pack.block) == 4:
        # Prefer original block when present
        recon = vec4_to_bytes(pack.q.as_array() * pack.scale)
        if recon == pack.block:
            return pack.block
        # If floating noise, still return original for digital path
        return pack.block
    return vec4_to_bytes(pack.q.as_array() * pack.scale)


def reconstruct_block_from_qs(q: Quaternion, scale: float) -> bytes:
    """Photonic-style inverse: unit quaternion × estimated scale → 4 bytes."""
    return vec4_to_bytes(q.as_array() * float(scale))


def scale_to_carrier_amp(scale: float, *, scale_max: float = 2.0) -> float:
    """Map shard scale → carrier amplitude in (0.15, 1.0] for OAM imprint."""
    s = float(np.clip(scale / max(scale_max, 1e-12), 0.0, 1.0))
    return float(0.15 + 0.85 * s)


def carrier_amp_to_scale(amp: float, *, scale_max: float = 2.0) -> float:
    """Inverse of ``scale_to_carrier_amp``."""
    s = (float(amp) - 0.15) / 0.85
    s = float(np.clip(s, 0.0, 1.0))
    return s * scale_max


def pack_payload(
    payload: bytes,
    n_shards: int = 8,
    *,
    redundancy: int = 1,
) -> list[ShardPack]:
    """
    Pack payload into ``n_shards`` ShardPacks.

    If ``redundancy > 1``, each logical 4-byte chunk is repeated across
    ``redundancy`` consecutive shards (repetition code for majority vote).
    """
    if not payload:
        payload = b"\x00"
    redundancy = max(1, int(redundancy))
    n_logical = max(1, n_shards // redundancy)
    need = n_logical * 4
    data = payload[:need].ljust(need, b"\x00")

    packs: list[ShardPack] = []
    for i in range(n_logical):
        block = data[i * 4 : (i + 1) * 4]
        pack = pack_byte_block(block)
        for _ in range(redundancy):
            if len(packs) >= n_shards:
                break
            packs.append(
                ShardPack(
                    q=pack.q,
                    scale=pack.scale,
                    block=pack.block,
                    vec=pack.vec.copy(),
                )
            )
    # Pad remaining shards with zeros if n_shards not multiple of redundancy
    while len(packs) < n_shards:
        packs.append(pack_byte_block(b"\x00\x00\x00\x00"))
    return packs[:n_shards]


def unpack_payload(
    packs: list[ShardPack],
    n_bytes: int | None = None,
    *,
    redundancy: int = 1,
    use_stored_blocks: bool = True,
) -> bytes:
    """
    Unpack shards → bytes with optional majority vote over redundant copies.
    """
    redundancy = max(1, int(redundancy))
    if not packs:
        return b""

    if use_stored_blocks and all(p.block for p in packs):
        blocks = [p.block for p in packs]
    else:
        blocks = [reconstruct_block_from_qs(p.q, p.scale) for p in packs]

    if redundancy == 1:
        out = b"".join(blocks)
    else:
        n_logical = len(blocks) // redundancy
        voted: list[bytes] = []
        for i in range(n_logical):
            group = blocks[i * redundancy : (i + 1) * redundancy]
            # Byte-wise majority
            arr = np.stack([np.frombuffer(g, dtype=np.uint8) for g in group], axis=0)
            # mode along axis 0
            maj = np.zeros(4, dtype=np.uint8)
            for j in range(4):
                vals, counts = np.unique(arr[:, j], return_counts=True)
                maj[j] = vals[int(np.argmax(counts))]
            voted.append(maj.tobytes())
        out = b"".join(voted)

    if n_bytes is not None:
        return out[: int(n_bytes)]
    return out


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """Simple CRC-8 for payload integrity checks."""
    crc = init & 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def byte_error_rate(a: bytes, b: bytes) -> float:
    """Fraction of differing bytes (padded to equal length)."""
    n = max(len(a), len(b), 1)
    aa = a.ljust(n, b"\x00")
    bb = b.ljust(n, b"\x00")
    return float(sum(x != y for x, y in zip(aa, bb)) / n)


def bit_error_rate(a: bytes, b: bytes) -> float:
    n = max(len(a), len(b), 1)
    aa = np.frombuffer(a.ljust(n, b"\x00"), dtype=np.uint8)
    bb = np.frombuffer(b.ljust(n, b"\x00"), dtype=np.uint8)
    xor = np.bitwise_xor(aa, bb)
    bits = np.unpackbits(xor)
    return float(bits.mean()) if bits.size else 0.0


# --- Legacy API (approximate unit-only path) ---------------------------------

def bytes_to_quaternion_shards(payload: bytes, n_shards: int = 8) -> list[Quaternion]:
    """Pack payload bytes into unit-quaternion shards (scale discarded)."""
    return [p.q for p in pack_payload(payload, n_shards=n_shards, redundancy=1)]


def quaternion_shards_to_bytes(shards: list[Quaternion], n_bytes: int | None = None) -> bytes:
    """Approximate inverse assuming scale≈1 (lossy legacy path)."""
    out = bytearray()
    for q in shards:
        out.extend(vec4_to_bytes(q.as_array()))
    if n_bytes is not None:
        return bytes(out[:n_bytes])
    return bytes(out)
