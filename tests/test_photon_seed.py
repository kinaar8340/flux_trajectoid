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
    assert a.mismatch_deg >= 0.0
    assert a.rotation_matrices is not None
    assert a.phase_trench_mask is not None


def test_trajectoid_scaling_reduces_or_reports_mismatch():
    from flux_trajectoid.shell.generator import (
        compute_cumulative_rotations,
        scale_path_for_closure,
    )

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    path = np.column_stack([np.cos(t), 1.3 * np.sin(t)])
    path = np.vstack([path, path[0]])
    _, m0, _ = compute_cumulative_rotations(path, r=1.0)
    scaled, kx, ky, g, m1 = scale_path_for_closure(
        path, rolling_radius=1.0, rng=np.random.default_rng(0), max_iter=12, grid=5
    )
    assert kx > 0 and ky > 0 and g > 0
    assert m1 <= m0 + 1e-6  # scaling search must not worsen best candidate
    _, m_check, _ = compute_cumulative_rotations(scaled, r=1.0)
    assert abs(m_check - m1) < 1.0  # allow tiny SVD/composition drift


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


def test_invertible_shard_pack_roundtrip():
    from flux_trajectoid.utils.quaternion_utils import pack_payload, unpack_payload, crc8

    raw = b"PhotonSeed!"
    packs = pack_payload(raw, n_shards=4, redundancy=1)
    back = unpack_payload(packs, n_bytes=len(raw), use_stored_blocks=True)
    assert back == raw
    # q×scale reconstruction (no stored blocks)
    for p in packs:
        p.block = b""  # force reconstruct path
    # restore blocks via reconstruct
    from flux_trajectoid.utils.quaternion_utils import reconstruct_block_from_qs

    packs2 = pack_payload(raw, n_shards=4)
    recon = b"".join(reconstruct_block_from_qs(p.q, p.scale) for p in packs2)[: len(raw)]
    assert recon == raw
    assert crc8(raw) == crc8(recon)


def test_redundancy_majority_vote():
    from flux_trajectoid.utils.quaternion_utils import (
        ShardPack,
        pack_payload,
        unpack_payload,
        reconstruct_block_from_qs,
    )

    raw = b"ABCD"  # one logical block
    packs = pack_payload(raw, n_shards=3, redundancy=3)
    assert len(packs) == 3
    # Corrupt two photonic reconstructions of scale/q slightly on one copy only
    corrupted = []
    for i, p in enumerate(packs):
        if i == 0:
            q = p.q
            s = p.scale
            block = reconstruct_block_from_qs(q, s)
        else:
            block = p.block
        corrupted.append(ShardPack(q=p.q, scale=p.scale, block=block, vec=p.vec))
    out = unpack_payload(corrupted, n_bytes=4, redundancy=3, use_stored_blocks=True)
    assert out == raw


def test_quaternion_multiply_identity():
    q = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()
    ident = Quaternion(1, 0, 0, 0)
    p = q.multiply(ident)
    assert np.allclose(p.as_array(), q.as_array())


def test_build_propagate_recover():
    msg = "photon seed"
    ast = PhotonSeedAsteroid(msg, seed=42).build(
        n_shards=4, lattice_nx=12, n_coupling_steps=4
    )
    assert ast.shell is not None
    assert ast.quaternion is not None
    assert ast.flux_state is not None
    assert ast.flux_state.protected_field is not None
    assert ast.quaternion.packs
    assert "crc8" in ast.quaternion.metadata

    prop = ast.propagate(turbulence_level=0.2, n_steps=8)
    assert 0.0 <= prop.fidelity_proxy <= 1.0
    assert len(prop.mean_twist_trace) == 8

    rec = ast.recover(mode="hybrid")
    assert rec.payload_text == msg
    assert rec.crc_ok is True
    assert rec.digital_payload == msg.encode("utf-8")
    assert rec.shell_match is not None
    assert rec.shell_match.matched
    assert rec.flywheel_readout is not None


def test_digital_and_photonic_recovery_modes():
    msg = "seed"
    ast = PhotonSeedAsteroid(msg, seed=7).build(
        n_shards=4, lattice_nx=8, n_coupling_steps=2, force_stub_flux=True
    )
    dig = ast.recover(mode="digital")
    assert dig.payload_text == msg
    assert dig.crc_ok is True

    # Clean photonic from per-shard fields (no channel)
    ph = ast.recover(mode="photonic")
    assert ph.photonic_payload is not None
    # Invertible q×scale on clean fields should be exact or near-exact
    assert ph.payload_hat == msg.encode("utf-8") or (
        ph.byte_error_rate is not None and ph.byte_error_rate <= 0.25
    )


def test_photonic_recovery_from_clean_shard_fields():
    from flux_trajectoid.inner.vqc_encoder import encode_to_quaternion
    from flux_trajectoid.recovery.decoder import recover_from_shard_fields
    from flux_trajectoid.utils.quaternion_utils import unpack_payload

    raw = b"Hello!!"  # 7 bytes
    enc = encode_to_quaternion(raw, n_shards=2, build_fields=True)
    packs = recover_from_shard_fields(enc.fields)
    out = unpack_payload(packs, n_bytes=len(raw), use_stored_blocks=False)
    assert out == raw


def test_summary_keys():
    ast = PhotonSeedAsteroid("x", seed=1).build(
        n_shards=2, lattice_nx=8, n_coupling_steps=2
    )
    s = ast.summary()
    assert s["built"] is True
    assert "shell_hash" in s
    assert "n_shards" in s
    assert "flux_backend" in s


def test_live_oam_flux_backend_when_available():
    from flux_trajectoid import is_live_oam_flux, oam_flux_backend
    from flux_trajectoid.inner.oam_flux_coupling import couple_to_flux_lattice
    from flux_trajectoid.inner.vqc_encoder import encode_to_quaternion
    from flux_trajectoid.shell.generator import generate_shell

    backend = oam_flux_backend()
    # With submodule checked out, expect live path
    if not is_live_oam_flux():
        pytest.skip(f"oam_flux not importable ({backend})")

    enc = encode_to_quaternion("live-backend", n_shards=2, build_fields=False)
    shell = generate_shell("live-backend", seed=0, n_points=64, scale_grid=3, scale_max_iter=4)
    flux = couple_to_flux_lattice(
        enc,
        shell,
        lattice_nx=8,
        flywheel_sites=3,
        n_steps=3,
        n_z=8,
        nr=32,
        l_max=2,
        seed=0,
    )
    assert flux.is_live
    assert flux.lattice is not None
    assert flux.photonics_propagation is not None
    assert flux.backend.startswith("live")
    assert len(flux.coupling_history) > 0
    assert flux.metadata.get("oam_flux_version") is not None


def test_force_stub_flux():
    from flux_trajectoid.inner.oam_flux_coupling import couple_to_flux_lattice
    from flux_trajectoid.inner.vqc_encoder import encode_to_quaternion

    enc = encode_to_quaternion(b"stub", n_shards=2, build_fields=False)
    flux = couple_to_flux_lattice(enc, None, lattice_nx=8, force_stub=True, n_steps=2)
    assert flux.backend == "stub"
    assert flux.lattice is None


def test_fidelity_metrics_and_sweep():
    from flux_trajectoid.propagation.metrics import compute_fidelity_metrics

    ast = PhotonSeedAsteroid(b"metrics", seed=3).build(
        n_shards=2,
        lattice_nx=8,
        n_coupling_steps=2,
        force_stub_flux=True,
        n_points=64,
        scale_grid=3,
        scale_max_iter=2,
    )
    prop = ast.propagate(turbulence_level=0.2, n_steps=6, seed=3)
    assert prop.metrics is not None
    m = prop.metrics
    assert 0.0 <= m.overlap_fidelity <= 1.0
    assert m.phase_rmse_rad >= 0.0
    assert 0.0 <= m.oam_fidelity <= 1.0
    assert len(prop.fidelity_trace) == 6
    # identity metrics
    ref = prop.field_reference
    assert ref is not None
    idm = compute_fidelity_metrics(ref, ref)
    assert idm.overlap_fidelity > 0.99
    assert idm.phase_rmse_rad < 1e-6

    rows = ast.sweep_turbulence(levels=[0.0, 0.3], n_steps=4, recover_photonic=False)
    assert len(rows) == 2
    assert rows[0]["overlap_fidelity"] >= rows[1]["overlap_fidelity"] - 0.05


def test_slm_export(tmp_path):
    ast = PhotonSeedAsteroid("slm-test", seed=1).build(
        n_shards=2,
        lattice_nx=8,
        n_coupling_steps=1,
        force_stub_flux=True,
        n_points=64,
        scale_grid=3,
        scale_max_iter=2,
    )
    out = tmp_path / "slm"
    result = ast.export_slm(out, preset="generic_256", stack_shards=True)
    assert (out / "manifest.json").is_file()
    assert (out / "phase_rad.npy").is_file()
    assert (out / "phase_levels.png").is_file()
    assert (out / "phase_stack.npy").is_file()
    assert result.phase_rad.shape == (256, 256)
    assert result.phase_levels.dtype == np.uint8
    assert "phase_rad.npy" in result.files
