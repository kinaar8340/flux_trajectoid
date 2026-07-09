"""Tests for Photon Seed Asteroid pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from flux_trajectoid import PhotonSeedAsteroid, generate_shell
from flux_trajectoid.recovery.shell_identifier import identify_shell
from flux_trajectoid.utils.fourier_descriptors import (
    compute_fourier_descriptors,
    match_fingerprints,
    unroll_curve,
)
from flux_trajectoid.utils.quaternion_utils import (
    Quaternion,
    bytes_to_quaternion_shards,
    quaternion_shards_to_bytes,
)


def test_shell_deterministic():
    a = generate_shell("abc", seed=1)
    b = generate_shell("abc", seed=1)
    c = generate_shell("abc", seed=2)
    assert np.allclose(a.vertices, b.vertices)
    assert not np.allclose(a.vertices, c.vertices)
    assert a.fourier_fingerprint is not None
    assert a.total_length > 0


def test_fourier_and_unroll():
    shell = generate_shell("fourier-test", seed=0, n_points=128)
    fd = compute_fourier_descriptors(shell.vertices, n_harmonics=8)
    assert fd["fingerprint"].shape[0] == 8
    assert abs(np.linalg.norm(fd["fingerprint"]) - 1.0) < 1e-6
    unrolled = unroll_curve(shell.vertices, n_samples=64)
    assert unrolled["path"].shape[0] == 64
    assert unrolled["total_length"] > 0


def test_shell_self_identify():
    shell = generate_shell("identity", seed=99)
    match = identify_shell(shell.vertices, reference=shell, tol=0.2)
    assert match.matched
    assert match.similarity > 0.95


def test_different_payloads_different_fingerprints():
    s1 = generate_shell("alpha", seed=0)
    s2 = generate_shell("beta", seed=0)
    sim = match_fingerprints(s1.fourier_fingerprint, s2.fourier_fingerprint)
    # Not required to be orthogonal, but should not be identical
    assert sim < 0.999


def test_quaternion_roundtrip_approx():
    raw = b"test"
    shards = bytes_to_quaternion_shards(raw, n_shards=2)
    assert all(abs(q.norm() - 1.0) < 1e-9 for q in shards)
    back = quaternion_shards_to_bytes(shards, n_bytes=4)
    assert len(back) == 4


def test_quaternion_multiply_identity():
    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()
    ident = Quaternion(1, 0, 0, 0)
    p = q.multiply(ident)
    assert np.allclose(p.as_array(), q.as_array())


def test_build_propagate_recover():
    msg = "photon seed"
    ast = PhotonSeedAsteroid(msg, seed=42).build(n_shards=4, lattice_nx=12)
    assert ast.shell is not None
    assert ast.quaternion is not None
    assert ast.flux_state is not None
    assert ast.flux_state.protected_field is not None

    prop = ast.propagate(turbulence_level=0.2, n_steps=8)
    assert 0.0 <= prop.fidelity_proxy <= 1.0
    assert len(prop.mean_twist_trace) == 8

    rec = ast.recover()
    assert rec.payload_text == msg
    assert rec.shell_match is not None
    assert rec.shell_match.matched
    assert rec.flywheel_readout is not None


def test_summary_keys():
    ast = PhotonSeedAsteroid("x", seed=1).build(n_shards=2, lattice_nx=8)
    s = ast.summary()
    assert s["built"] is True
    assert "shell_hash" in s
    assert "n_shards" in s
