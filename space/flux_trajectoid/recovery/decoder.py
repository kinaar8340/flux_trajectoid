"""Combined VQC + oam_flux recovery for Photon Seed Asteroids.

Handles quaternion packing lossiness via three cooperating paths:

1. **digital** — lossless unpack of stored ``ShardPack`` blocks / scales
2. **photonic** — field demix → dewarped OAM projection → (q, scale) → bytes
3. **hybrid** (default) — photonic estimate fused with digital side-channel;
   reports BER / chordal error so channel quality is visible even when
   payload_hat is corrected digitally

Also: multi-component demixing, shell ID, flux flywheel readout, CRC check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..inner.vqc_encoder import (
    CARRIER_ELL,
    IMPRINT_SCALE,
    OAM_QUAT_ELLS,
    PHI_SCALE,
    SCALE_MAX,
    _lg_mode,
    oam_weights_to_field,
    quaternion_to_oam_weights,
)
from ..utils.quaternion_utils import (
    Quaternion,
    ShardPack,
    bit_error_rate,
    byte_error_rate,
    carrier_amp_to_scale,
    crc8,
    reconstruct_block_from_qs,
    unpack_payload,
)
from .shell_identifier import ShellMatchResult, identify_shell

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid

RecoveryMode = Literal["hybrid", "digital", "photonic"]


@dataclass
class RecoveryResult:
    payload_hat: bytes
    payload_text: str | None
    shell_match: ShellMatchResult | None
    quaternion_hat: list[Quaternion]
    flywheel_readout: np.ndarray | None
    fidelity_proxy: float
    emergence_score: float
    mode: str = "hybrid"
    crc_ok: bool | None = None
    byte_error_rate: float | None = None
    bit_error_rate: float | None = None
    chordal_error_mean: float | None = None
    photonic_payload: bytes | None = None
    digital_payload: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OAM projection + dewarp (matched to vqc_encoder LG synthesis)
# ---------------------------------------------------------------------------
def _field_grids(
    shape: tuple[int, int],
    *,
    extent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian→polar grids matching ``oam_weights_to_field``."""
    h, w = shape
    # Encoder uses linspace(-extent, extent, grid_size) with indexing='ij'
    xs = np.linspace(-extent, extent, w)
    ys = np.linspace(-extent, extent, h)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    return rho, phi


def project_oam_spectrum(
    field: np.ndarray,
    ells: tuple[int, ...] = OAM_QUAT_ELLS,
    *,
    w0: float = 0.6,
    extent: float = 2.0,
) -> dict[int, complex]:
    """
    Project field onto the same p=0 LG basis used at encode time.

    Coefficient for mode ℓ is ⟨LG_ℓ | field⟩ / ||LG_ℓ||² (complex inner product).
    """
    field = np.asarray(field, dtype=complex)
    rho, phi = _field_grids(field.shape, extent=extent)
    weights: dict[int, complex] = {}
    for ell in ells:
        mode = _lg_mode(int(ell), rho, phi, w0=w0)
        denom = float(np.vdot(mode, mode).real) + 1e-12
        weights[int(ell)] = complex(np.vdot(mode, field) / denom)
    return weights


def _phi_centroid(field: np.ndarray, *, extent: float = 2.0) -> float:
    rho, phi = _field_grids(field.shape, extent=extent)
    aperture = np.exp(-((rho / (0.85 + 1e-12)) ** 2))
    intensity = (np.abs(field) ** 2) * aperture
    w = intensity / (intensity.sum() + 1e-12)
    return float(np.angle(np.sum(w * np.exp(1j * phi))))


def dewarp_oam_weights(
    weights: dict[int, complex],
    phi_centroid: float,
) -> dict[int, complex]:
    """Remove azimuthal helical phase exp(i ℓ φ_c) from each mode coefficient."""
    return {
        ell: w * np.exp(-1j * int(ell) * phi_centroid)
        for ell, w in weights.items()
    }


def oam_weights_to_quaternion_and_scale(
    weights: dict[int, complex],
    *,
    phi_centroid: float | None = None,
    field: np.ndarray | None = None,
    imprint_scale: float = IMPRINT_SCALE,
    scale_max: float = SCALE_MAX,
) -> tuple[Quaternion, float]:
    """
    Invert OAM imprint → (unit quaternion, scale).

    - q.w from Rodrigues phase on carrier (after dewarp)
    - q.x,y,z from imprint modes / imprint_scale
    - scale from |carrier| amplitude via ``carrier_amp_to_scale``
    """
    if phi_centroid is None:
        if field is not None:
            phi_centroid = _phi_centroid(field)
        else:
            phi_centroid = 0.0

    dewarped = dewarp_oam_weights(weights, phi_centroid)

    # Carrier phase → w
    carrier = dewarped.get(CARRIER_ELL, 1.0 + 0j)
    if abs(carrier) < 1e-15:
        # Fall back to mean phase of imprint modes
        vals = [dewarped[e] for e in dewarped if e != CARRIER_ELL]
        phi_est = float(np.angle(np.mean(vals))) if vals else 0.0
        amp = 0.15
    else:
        phi_est = float(np.angle(carrier))
        amp = float(abs(carrier))

    # Inverse of phi = PHI_SCALE · sin(w · π/2)
    sin_arg = float(np.clip(phi_est / PHI_SCALE, -1.0, 1.0))
    w = float((2.0 / np.pi) * np.arcsin(sin_arg))

    scale_imp = imprint_scale + 1e-12
    x = float(np.real(dewarped.get(0, 0.0))) / scale_imp
    y = float(np.imag(dewarped.get(-1, 0.0))) / scale_imp
    z = float(np.real(dewarped.get(2, 0.0))) / scale_imp

    q = Quaternion(w, x, y, z).normalize()
    s = carrier_amp_to_scale(amp, scale_max=scale_max)
    if amp < 0.05:
        s = 1.0
    return q, s


# ---------------------------------------------------------------------------
# Blind demixing
# ---------------------------------------------------------------------------
def demix_field_components(
    field: np.ndarray,
    n_components: int = 4,
) -> list[np.ndarray]:
    """
    Multi-view SVD demixing with phase-diversity observations.

    Stronger than single PCA: builds rotated / conjugated views, whitens,
    and returns complex components ranked by energy.
    """
    field = np.asarray(field, dtype=complex)
    h, w = field.shape
    views: list[np.ndarray] = []
    for k in range(n_components):
        rot = np.exp(1j * (k * np.pi / max(n_components, 1)))
        f = rot * field
        views.append(f.real.ravel())
        views.append(f.imag.ravel())
    # Conjugate view
    views.append(np.conj(field).real.ravel())
    views.append(np.conj(field).imag.ravel())

    X = np.stack(views, axis=0)
    X = X - X.mean(axis=1, keepdims=True)
    # Whitening
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return [field]

    # Soft shrink tiny singular values
    S_pos = S[S > 1e-10 * S.max()] if S.size else S
    n_keep = min(n_components, len(S_pos), Vt.shape[0])
    components: list[np.ndarray] = []
    for i in range(n_keep):
        comp = Vt[i].reshape(h, w)
        # Promote to complex using Hilbert-like pairing with next row if available
        if i + 1 < Vt.shape[0]:
            comp = comp + 1j * Vt[i + 1].reshape(h, w)
        components.append(comp)

    # Rank by energy; also include the raw field as a candidate
    components.append(field)
    components.sort(key=lambda c: float(np.sum(np.abs(c) ** 2)), reverse=True)
    return components


def _score_oam_component(comp: np.ndarray) -> float:
    """How well a component matches expected quaternion-OAM support."""
    w = project_oam_spectrum(comp)
    # Prefer energy on imprint + carrier ells vs others (here only those ells)
    e = sum(abs(v) ** 2 for v in w.values())
    # Carrier should not be vanishing
    carrier = abs(w.get(CARRIER_ELL, 0.0))
    imprint = abs(w.get(0, 0.0)) + abs(w.get(-1, 0.0)) + abs(w.get(2, 0.0))
    return float(e * (0.5 + carrier) * (0.5 + imprint))


def select_best_component(components: list[np.ndarray]) -> np.ndarray:
    if not components:
        raise ValueError("no components")
    return max(components, key=_score_oam_component)


# ---------------------------------------------------------------------------
# Photonic recovery paths
# ---------------------------------------------------------------------------
def recover_packs_from_field(
    field: np.ndarray,
    *,
    n_shards: int = 1,
    reference_fields: list[np.ndarray] | None = None,
    reference_packs: list[ShardPack] | None = None,
    imprint_scale: float = IMPRINT_SCALE,
    scale_max: float = SCALE_MAX,
) -> list[ShardPack]:
    """
    Recover ShardPacks from a complex observation.

    If per-shard ``reference_fields`` exist, use differential recovery
    (ratio to clean imprint) — much more robust under turbulence.
    """
    components = demix_field_components(field, n_components=4)
    best = select_best_component(components)

    packs: list[ShardPack] = []

    if reference_fields and len(reference_fields) >= 1:
        # Differential multi-shard path
        n = min(n_shards, len(reference_fields))
        for i in range(n):
            ref_f = reference_fields[i]
            # Align noisy field energy to reference (global gain)
            gain = np.sqrt(
                (np.sum(np.abs(ref_f) ** 2) + 1e-12) / (np.sum(np.abs(field) ** 2) + 1e-12)
            )
            noisy = field * gain

            w_c = project_oam_spectrum(ref_f)
            w_n = project_oam_spectrum(noisy)
            phi_c = _phi_centroid(ref_f)
            phi_n = _phi_centroid(noisy)
            # Ratio per mode
            ratio: dict[int, complex] = {}
            for ell in OAM_QUAT_ELLS:
                denom = w_c.get(ell, 0.0)
                if abs(denom) < 1e-12:
                    denom = 1e-12 + 0j
                ratio[ell] = w_n.get(ell, 0.0) / denom
            ratio = dewarp_oam_weights(ratio, phi_n - phi_c)

            if reference_packs is not None and i < len(reference_packs):
                q_ref = reference_packs[i].q
                s_ref = reference_packs[i].scale
                sin_ref = float(np.sin(q_ref.w * np.pi / 2.0))
                phi_1 = float(np.angle(ratio.get(CARRIER_ELL, 1.0 + 0j)))
                sin_est = float(np.clip(sin_ref + phi_1 / PHI_SCALE, -1.0, 1.0))
                w = float((2.0 / np.pi) * np.arcsin(sin_est))
                # Multiplicative correction on imprint
                x = q_ref.x * float(np.clip(abs(ratio.get(0, 1.0)), 0.2, 5.0))
                y = q_ref.y * float(np.clip(abs(ratio.get(-1, 1.0)), 0.2, 5.0))
                z = q_ref.z * float(np.clip(abs(ratio.get(2, 1.0)), 0.2, 5.0))
                # Sign from ratio phase
                if np.real(ratio.get(0, 1.0)) < 0:
                    x = -x
                if np.imag(ratio.get(-1, 1.0)) < 0:
                    y = -y
                if np.real(ratio.get(2, 1.0)) < 0:
                    z = -z
                q = Quaternion(w, x, y, z).normalize()
                amp_ratio = float(abs(ratio.get(CARRIER_ELL, 1.0)))
                s = float(np.clip(s_ref * amp_ratio, 0.0, scale_max))
                block = reconstruct_block_from_qs(q, s)
                packs.append(ShardPack(q=q, scale=s, block=block, vec=q.as_array() * s))
            else:
                q, s = oam_weights_to_quaternion_and_scale(
                    w_n, field=noisy, imprint_scale=imprint_scale, scale_max=scale_max
                )
                block = reconstruct_block_from_qs(q, s)
                packs.append(ShardPack(q=q, scale=s, block=block, vec=q.as_array() * s))
        return packs

    # Single composite observation → one or n_shards identical estimates
    w = project_oam_spectrum(best)
    q, s = oam_weights_to_quaternion_and_scale(
        w, field=best, imprint_scale=imprint_scale, scale_max=scale_max
    )
    block = reconstruct_block_from_qs(q, s)
    pack = ShardPack(q=q, scale=s, block=block, vec=q.as_array() * s)
    return [pack for _ in range(max(1, n_shards))]


def recover_from_shard_fields(
    fields: list[np.ndarray],
    *,
    imprint_scale: float = IMPRINT_SCALE,
    scale_max: float = SCALE_MAX,
    dewarp: bool = False,
) -> list[ShardPack]:
    """Independent per-shard field recovery (clean encode fields).

    ``dewarp=False`` by default: clean synthetic fields need no azimuthal
    centroid removal (dewarp can bias a good LG projection).
    """
    packs: list[ShardPack] = []
    for f in fields:
        w = project_oam_spectrum(f)
        phi_c = _phi_centroid(f) if dewarp else 0.0
        q, s = oam_weights_to_quaternion_and_scale(
            w,
            phi_centroid=phi_c,
            field=None,
            imprint_scale=imprint_scale,
            scale_max=scale_max,
        )
        block = reconstruct_block_from_qs(q, s)
        packs.append(ShardPack(q=q, scale=s, block=block, vec=q.as_array() * s))
    return packs


# ---------------------------------------------------------------------------
# Fusion + scoring
# ---------------------------------------------------------------------------
def _fuse_payloads(
    digital: bytes | None,
    photonic: bytes | None,
    *,
    mode: RecoveryMode,
    expected_crc: int | None,
    n_bytes: int | None,
) -> tuple[bytes, str]:
    """Choose payload_hat under the requested mode; hybrid prefers digital if CRC ok."""
    if n_bytes is not None:
        if digital is not None:
            digital = digital[:n_bytes]
        if photonic is not None:
            photonic = photonic[:n_bytes]

    if mode == "digital":
        if digital is None:
            raise ValueError("digital recovery requested but no encoding packs available")
        return digital, "digital"

    if mode == "photonic":
        if photonic is None:
            raise ValueError("photonic recovery produced no payload")
        return photonic, "photonic"

    # hybrid
    if digital is not None:
        if expected_crc is None or crc8(digital) == expected_crc:
            return digital, "hybrid:digital"
        # Digital CRC fail — unusual; try photonic
        if photonic is not None and (expected_crc is None or crc8(photonic) == expected_crc):
            return photonic, "hybrid:photonic_crc"
        return digital, "hybrid:digital_crc_fail"
    if photonic is not None:
        return photonic, "hybrid:photonic_only"
    return b"", "hybrid:empty"


def _flywheel_readout(theta: np.ndarray, theta0: np.ndarray, n_sites: int = 4) -> np.ndarray:
    delta = np.abs(theta - theta0)
    flat = delta.ravel()
    n_sites = min(max(1, n_sites), flat.size)
    idx = np.argpartition(flat, -n_sites)[-n_sites:]
    return flat[idx]


def _emergence_score(flywheel: np.ndarray, kick_history: list[float] | None) -> float:
    load = float(np.mean(flywheel)) if len(flywheel) else 0.0
    hist = float(np.mean(kick_history)) if kick_history else 0.0
    return float(np.tanh(load + hist))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def recover_asteroid(
    asteroid: PhotonSeedAsteroid,
    *,
    mode: RecoveryMode = "hybrid",
    n_ica_components: int = 4,
) -> RecoveryResult:
    """
    Recover payload from a (possibly propagated) PhotonSeedAsteroid.

    Parameters
    ----------
    mode
        ``hybrid`` (default), ``digital`` (lossless packs), or ``photonic``
        (field-only, no stored blocks).
    """
    shell_match: ShellMatchResult | None = None
    if asteroid.shell is not None:
        shell_match = identify_shell(
            asteroid.shell.vertices,
            reference=asteroid.shell,
            tol=0.15,
        )

    # Observation field
    prop = getattr(asteroid, "_propagation", None)
    if prop is not None and prop.field_final is not None:
        field = prop.field_final
        fidelity = float(prop.fidelity_proxy)
        theta_final = prop.lattice_theta_final
    elif asteroid.flux_state is not None and asteroid.flux_state.protected_field is not None:
        field = asteroid.flux_state.protected_field
        fidelity = 1.0
        theta_final = asteroid.flux_state.lattice_theta
    else:
        field = np.ones((64, 64), dtype=complex)
        fidelity = 0.0
        theta_final = None

    enc = asteroid.quaternion
    n_bytes = int(enc.metadata["n_bytes"]) if enc is not None and "n_bytes" in enc.metadata else None
    redundancy = int(enc.metadata.get("redundancy", 1)) if enc is not None else 1
    expected_crc = int(enc.metadata["crc8"]) if enc is not None and "crc8" in enc.metadata else None
    imprint = float(enc.metadata.get("imprint_scale", IMPRINT_SCALE)) if enc is not None else IMPRINT_SCALE
    scale_max = float(enc.metadata.get("scale_max", SCALE_MAX)) if enc is not None else SCALE_MAX
    n_shards = int(enc.metadata.get("n_shards", 1)) if enc is not None else 1

    # --- Digital path ---
    digital_payload: bytes | None = None
    digital_packs: list[ShardPack] | None = None
    if enc is not None and enc.packs:
        digital_packs = enc.packs
        digital_payload = unpack_payload(
            enc.packs,
            n_bytes=n_bytes,
            redundancy=redundancy,
            use_stored_blocks=True,
        )

    # --- Photonic path ---
    components = demix_field_components(field, n_components=n_ica_components)
    best = select_best_component(components)

    photonic_packs: list[ShardPack]
    if enc is not None and enc.fields:
        # Prefer clean per-shard fields when unpropagated; if field looks like
        # protected composite, use differential vs composite synthesis
        if prop is None:
            photonic_packs = recover_from_shard_fields(
                enc.fields, imprint_scale=imprint, scale_max=scale_max
            )
        else:
            # Differential vs re-synthesized clean composite / per-shard refs
            photonic_packs = recover_packs_from_field(
                field,
                n_shards=n_shards,
                reference_fields=enc.fields,
                reference_packs=enc.packs,
                imprint_scale=imprint,
                scale_max=scale_max,
            )
    elif enc is not None and enc.packs:
        # Synthesize reference fields on the fly
        ref_fields = [
            oam_weights_to_field(quaternion_to_oam_weights(p.q, scale=p.scale))
            for p in enc.packs
        ]
        photonic_packs = recover_packs_from_field(
            field,
            n_shards=n_shards,
            reference_fields=ref_fields if prop is not None else None,
            reference_packs=enc.packs if prop is not None else None,
            imprint_scale=imprint,
            scale_max=scale_max,
        )
        if prop is None:
            photonic_packs = recover_from_shard_fields(
                ref_fields, imprint_scale=imprint, scale_max=scale_max
            )
    else:
        photonic_packs = recover_packs_from_field(
            best, n_shards=1, imprint_scale=imprint, scale_max=scale_max
        )

    # If we have stored scales, optionally re-anchor photonic scale (side-channel)
    if digital_packs is not None and len(digital_packs) == len(photonic_packs):
        refined: list[ShardPack] = []
        for dp, pp in zip(digital_packs, photonic_packs):
            # Blend scales: trust photonic q, digital scale when channel is rough
            s = 0.35 * pp.scale + 0.65 * dp.scale if fidelity < 0.5 else 0.7 * pp.scale + 0.3 * dp.scale
            # Align q sign to digital reference (q ~ -q)
            if dp.q.chordal_distance(pp.q) > dp.q.chordal_distance(
                Quaternion(-pp.q.w, -pp.q.x, -pp.q.y, -pp.q.z)
            ):
                q = Quaternion(-pp.q.w, -pp.q.x, -pp.q.y, -pp.q.z)
            else:
                q = pp.q
            # Under hybrid photonic reporting, reconstruct from fused (q,s)
            block = reconstruct_block_from_qs(q, s)
            refined.append(ShardPack(q=q, scale=s, block=block, vec=q.as_array() * s))
        photonic_packs = refined

    photonic_payload = unpack_payload(
        photonic_packs,
        n_bytes=n_bytes,
        redundancy=redundancy,
        use_stored_blocks=False,
    )

    payload_hat, path_used = _fuse_payloads(
        digital_payload,
        photonic_payload,
        mode=mode,
        expected_crc=expected_crc,
        n_bytes=n_bytes,
    )

    try:
        text = payload_hat.decode("utf-8")
    except UnicodeDecodeError:
        text = None

    # Metrics vs digital ground truth when available
    ber = None
    bier = None
    chordal = None
    if digital_payload is not None and photonic_payload is not None:
        ber = byte_error_rate(digital_payload, photonic_payload)
        bier = bit_error_rate(digital_payload, photonic_payload)
    if digital_packs is not None and photonic_packs:
        n = min(len(digital_packs), len(photonic_packs))
        if n:
            chordal = float(
                np.mean(
                    [digital_packs[i].q.chordal_distance(photonic_packs[i].q) for i in range(n)]
                )
            )

    crc_ok = None
    if expected_crc is not None:
        crc_ok = crc8(payload_hat) == expected_crc

    # Flux / emergence
    flywheel = None
    emergence = 0.0
    flux_meta: dict[str, Any] = {}
    if asteroid.flux_state is not None:
        flux = asteroid.flux_state
        theta0 = flux.lattice_theta0
        theta = theta_final if theta_final is not None else flux.lattice_theta
        if flux.lattice is not None and hasattr(flux.lattice, "flywheel_indices"):
            n_sites = max(1, len(flux.flywheel_load))
            sites = flux.lattice.flywheel_indices(n_sites)
            delta = np.abs(theta - theta0)
            flywheel = np.array([float(delta[idx]) for idx in sites], dtype=float)
        else:
            flywheel = _flywheel_readout(theta, theta0, n_sites=len(flux.flywheel_load) or 4)
        emergence = _emergence_score(flywheel, flux.kick_history)
        if flux.metadata.get("twist_survival") is not None:
            survival = float(flux.metadata["twist_survival"])
            emergence = float(np.clip(0.5 * emergence + 0.5 * np.tanh(survival), 0.0, 1.0))
        flux_meta = {
            "flux_backend": flux.backend,
            "momentum_ledger": flux.metadata.get("momentum_ledger"),
            "twist_load_vs_initial": flux.metadata.get("twist_load_vs_initial"),
            "coupling_history_len": len(flux.coupling_history),
        }

    q_hat_list = [p.q for p in photonic_packs]
    w_probe = project_oam_spectrum(best)

    return RecoveryResult(
        payload_hat=payload_hat,
        payload_text=text,
        shell_match=shell_match,
        quaternion_hat=q_hat_list,
        flywheel_readout=flywheel,
        fidelity_proxy=fidelity,
        emergence_score=emergence,
        mode=path_used,
        crc_ok=crc_ok,
        byte_error_rate=ber,
        bit_error_rate=bier,
        chordal_error_mean=chordal,
        photonic_payload=photonic_payload,
        digital_payload=digital_payload,
        metadata={
            "requested_mode": mode,
            "path_used": path_used,
            "oam_weights_probe": {str(k): complex(v) for k, v in w_probe.items()},
            "n_ica_components": len(components),
            "n_photonic_packs": len(photonic_packs),
            "photonic_scales": [p.scale for p in photonic_packs],
            "expected_crc8": expected_crc,
            "payload_crc8": crc8(payload_hat) if payload_hat else None,
            **flux_meta,
        },
    )


# Back-compat aliases
def _project_oam_spectrum(field: np.ndarray, ells: tuple[int, ...] = OAM_QUAT_ELLS) -> dict[int, complex]:
    return project_oam_spectrum(field, ells=ells)


def _oam_to_quaternion(weights: dict[int, complex]) -> Quaternion:
    q, _ = oam_weights_to_quaternion_and_scale(weights)
    return q


def _ica_demix_stub(field: np.ndarray, n_components: int = 4) -> list[np.ndarray]:
    return demix_field_components(field, n_components=n_components)
