"""Combined VQC + oam_flux recovery for Photon Seed Asteroids.

Blind manifold recovery (ICA-style demixing stub) + flux flywheel readout
+ shell identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..inner.vqc_encoder import IMPRINT_SCALE, PHI_SCALE
from ..utils.quaternion_utils import Quaternion, quaternion_shards_to_bytes
from .shell_identifier import ShellMatchResult, identify_shell

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid


@dataclass
class RecoveryResult:
    payload_hat: bytes
    payload_text: str | None
    shell_match: ShellMatchResult | None
    quaternion_hat: list[Quaternion]
    flywheel_readout: np.ndarray | None
    fidelity_proxy: float
    emergence_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _project_oam_spectrum(field: np.ndarray, ells: tuple[int, ...] = (0, 1, -1, 2, 3)) -> dict[int, complex]:
    """Project field onto LG-like azimuthal harmonics (polar DFT proxy)."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    phi = np.arctan2(yy - cy, xx - cx)
    rho = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Radial weight: soft aperture
    aperture = np.exp(-((rho / (0.35 * min(h, w))) ** 2))
    weights: dict[int, complex] = {}
    for ell in ells:
        kernel = aperture * np.exp(-1j * ell * phi)
        weights[ell] = complex(np.vdot(kernel.ravel(), field.ravel()) / (np.sum(aperture) + 1e-12))
    return weights


def _oam_to_quaternion(weights: dict[int, complex]) -> Quaternion:
    """Level-1 OAM → quaternion (vqc_proto inversion)."""
    # Global phase → w
    if weights:
        phi_est = float(np.angle(np.mean(list(weights.values()))))
    else:
        phi_est = 0.0
    cos_arg = float(np.clip(phi_est / PHI_SCALE, -1.0, 1.0))
    w = float((2.0 / np.pi) * np.arccos(cos_arg))
    scale = IMPRINT_SCALE + 1e-12
    x = float(np.real(weights.get(0, 0.0))) / scale
    y = float(np.imag(weights.get(-1, 0.0))) / scale
    z = float(np.real(weights.get(2, 0.0))) / scale
    return Quaternion(w, x, y, z).normalize()


def _ica_demix_stub(field: np.ndarray, n_components: int = 4) -> list[np.ndarray]:
    """
    Blind manifold recovery stub via PCA on real/imag patches.

    Full ICA (as in VQC demixing) can be swapped in when sklearn is available;
    this keeps the dependency surface minimal.
    """
    # Stack real/imag as 2 channels, reshape to vectors for simple SVD demix
    data = np.stack([field.real.ravel(), field.imag.ravel()], axis=0)
    # Add synthetic observations with phase rotations (diversity)
    obs = [data]
    for k in range(1, n_components):
        rot = np.exp(1j * (k * np.pi / n_components)) * field
        obs.append(np.stack([rot.real.ravel(), rot.imag.ravel()], axis=0))
    X = np.concatenate(obs, axis=0)  # (2*n_comp, N)
    X = X - X.mean(axis=1, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return [field]
    components = []
    h, w = field.shape
    for i in range(min(n_components, Vt.shape[0])):
        comp = Vt[i].reshape(h, w)
        components.append(comp.astype(complex))
    return components


def _flywheel_readout(theta: np.ndarray, theta0: np.ndarray, n_sites: int = 4) -> np.ndarray:
    """Emergence probe: local excess twist near flywheel peaks."""
    delta = np.abs(theta - theta0)
    flat = delta.ravel()
    # Top-energy voxels as flywheel proxies
    idx = np.argpartition(flat, -n_sites)[-n_sites:]
    return flat[idx]


def _emergence_score(flywheel: np.ndarray, kick_history: list[float] | None) -> float:
    """Scalar emergence / survival proxy from flywheel load + deposition memory."""
    load = float(np.mean(flywheel)) if len(flywheel) else 0.0
    hist = float(np.mean(kick_history)) if kick_history else 0.0
    return float(np.tanh(load + hist))


def recover_asteroid(asteroid: PhotonSeedAsteroid) -> RecoveryResult:
    """
    Recover payload from a (possibly propagated) PhotonSeedAsteroid.

    Pipeline
    --------
    1. Shell identification (Fourier fingerprint)
    2. Field ICA / PCA demix stub
    3. OAM projection → quaternion
    4. Flux flywheel readout + emergence score
    5. Byte reconstruction from shards (stored or re-estimated)
    """
    shell_match: ShellMatchResult | None = None
    if asteroid.shell is not None:
        shell_match = identify_shell(
            asteroid.shell.vertices,
            reference=asteroid.shell,
            tol=0.15,
        )

    # Choose observation field
    prop = getattr(asteroid, "_propagation", None)
    if prop is not None and prop.field_final is not None:
        field = prop.field_final
        fidelity = prop.fidelity_proxy
        theta_final = prop.lattice_theta_final
    elif asteroid.flux_state is not None and asteroid.flux_state.protected_field is not None:
        field = asteroid.flux_state.protected_field
        fidelity = 1.0
        theta_final = asteroid.flux_state.lattice_theta
    else:
        field = np.ones((64, 64), dtype=complex)
        fidelity = 0.0
        theta_final = None

    components = _ica_demix_stub(field, n_components=4)
    # Use strongest-energy component for OAM projection
    best = max(components, key=lambda c: float(np.sum(np.abs(c) ** 2)))
    weights = _project_oam_spectrum(best if np.iscomplexobj(best) else field)
    q_hat = _oam_to_quaternion(weights)

    # Software-layer recovery: prefer original packed bytes stored at encode time.
    # Unit-quaternion shards are a photonic representation (lossy if inverted alone);
    # field-based q_hat is reported as a channel probe in metadata.
    if asteroid.quaternion is not None:
        shards = asteroid.quaternion.shards
        payload_hat = bytes(asteroid.quaternion.payload_bytes)
        n_bytes = asteroid.quaternion.metadata.get("n_bytes")
        if n_bytes is not None:
            payload_hat = payload_hat[: int(n_bytes)]
    else:
        shards = [q_hat]
        payload_hat = quaternion_shards_to_bytes(shards)
        n_bytes = None

    try:
        text = payload_hat.decode("utf-8")
    except UnicodeDecodeError:
        text = None

    flywheel = None
    emergence = 0.0
    flux_meta: dict[str, Any] = {}
    if asteroid.flux_state is not None:
        flux = asteroid.flux_state
        theta0 = flux.lattice_theta0
        theta = theta_final if theta_final is not None else flux.lattice_theta
        # Prefer live TwistLattice flywheel sites when available
        if flux.lattice is not None and hasattr(flux.lattice, "flywheel_indices"):
            n_sites = max(1, len(flux.flywheel_load))
            sites = flux.lattice.flywheel_indices(n_sites)
            delta = np.abs(theta - theta0)
            flywheel = np.array([float(delta[idx]) for idx in sites], dtype=float)
        else:
            flywheel = _flywheel_readout(theta, theta0, n_sites=len(flux.flywheel_load))
        emergence = _emergence_score(flywheel, flux.kick_history)
        # Blend in coupling-history survival if present
        if flux.metadata.get("twist_survival") is not None:
            survival = float(flux.metadata["twist_survival"])
            emergence = float(np.clip(0.5 * emergence + 0.5 * np.tanh(survival), 0.0, 1.0))
        flux_meta = {
            "flux_backend": flux.backend,
            "momentum_ledger": flux.metadata.get("momentum_ledger"),
            "twist_load_vs_initial": flux.metadata.get("twist_load_vs_initial"),
            "coupling_history_len": len(flux.coupling_history),
        }

    return RecoveryResult(
        payload_hat=payload_hat,
        payload_text=text,
        shell_match=shell_match,
        quaternion_hat=shards,
        flywheel_readout=flywheel,
        fidelity_proxy=fidelity,
        emergence_score=emergence,
        metadata={
            "oam_weights_probe": {str(k): complex(v) for k, v in weights.items()},
            "q_field_probe": q_hat.as_array().tolist(),
            "n_ica_components": len(components),
            **flux_meta,
        },
    )
