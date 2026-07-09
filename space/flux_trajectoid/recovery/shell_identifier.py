"""Unique shape matching via Fourier fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..shell.generator import ShellGeometry, generate_shell
from ..utils.fourier_descriptors import compute_fourier_descriptors, match_fingerprints


@dataclass
class ShellMatchResult:
    matched: bool
    similarity: float
    fingerprint: np.ndarray
    reference_fingerprint: np.ndarray | None
    metadata: dict[str, Any]


def identify_shell(
    observed_vertices: np.ndarray,
    reference: ShellGeometry | None = None,
    *,
    fingerprint: np.ndarray | None = None,
    tol: float = 0.15,
    n_harmonics: int = 12,
) -> ShellMatchResult:
    """
    Match an observed shell silhouette against a reference fingerprint.

    Similarity is cosine similarity of Fourier descriptor magnitudes.
    ``matched`` is True when similarity >= 1 - tol.
    """
    fd = compute_fourier_descriptors(observed_vertices, n_harmonics=n_harmonics)
    obs_fp = fd["fingerprint"]

    ref_fp = fingerprint
    if ref_fp is None and reference is not None:
        ref_fp = reference.fourier_fingerprint

    if ref_fp is None:
        return ShellMatchResult(
            matched=False,
            similarity=0.0,
            fingerprint=obs_fp,
            reference_fingerprint=None,
            metadata={"reason": "no_reference"},
        )

    sim = match_fingerprints(obs_fp, ref_fp)
    # Map cosine ∈ [-1,1] → treat high positive as match
    matched = sim >= (1.0 - tol)
    return ShellMatchResult(
        matched=matched,
        similarity=sim,
        fingerprint=obs_fp,
        reference_fingerprint=np.asarray(ref_fp),
        metadata={"tol": tol, "threshold": 1.0 - tol},
    )


def identify_from_payload_guess(
    observed_vertices: np.ndarray,
    payload_candidates: list[str | bytes],
    seed: int = 42,
    **kwargs,
) -> tuple[ShellMatchResult, str | bytes | None]:
    """Brute-force shell ID against candidate payloads (routing demo)."""
    best: ShellMatchResult | None = None
    best_payload: str | bytes | None = None
    for p in payload_candidates:
        shell = generate_shell(p, seed)
        result = identify_shell(observed_vertices, reference=shell, **kwargs)
        if best is None or result.similarity > best.similarity:
            best = result
            best_payload = p
    assert best is not None
    return best, best_payload
