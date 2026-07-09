"""Photon Seed Asteroid — main orchestrator.

Layers a trajectoid outer shell over VQC quaternion + oam_flux helical
lattice payload for robust photonic data carriers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .inner.oam_flux_coupling import FluxState, couple_to_flux_lattice
from .inner.vqc_encoder import QuaternionEncoding, encode_to_quaternion
from .propagation.simulator import PropagationResult, propagate_asteroid
from .recovery.decoder import RecoveryResult, recover_asteroid
from .shell.generator import ShellGeometry, generate_shell


@dataclass
class PhotonSeedAsteroid:
    """
    Layered biomimetic photonic data carrier (macadamia + trajectoid).

    Parameters
    ----------
    payload
        Message or raw bytes to encode in the inner nut.
    seed
        RNG / geometry seed for reproducible shell fingerprints.
    """

    payload: str | bytes
    seed: int = 42
    shell: ShellGeometry | None = None
    quaternion: QuaternionEncoding | None = None
    flux_state: FluxState | None = None
    _propagation: PropagationResult | None = field(default=None, repr=False)
    config: dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        *,
        n_shards: int = 8,
        lattice_nx: int = 16,
        flywheel_sites: int = 4,
        **shell_kwargs,
    ) -> PhotonSeedAsteroid:
        """Assemble outer shell + inner VQC encoding + oam_flux coupling."""
        self.shell = generate_shell(self.payload, self.seed, **shell_kwargs)
        self.quaternion = encode_to_quaternion(self.payload, n_shards=n_shards)
        self.flux_state = couple_to_flux_lattice(
            self.quaternion,
            self.shell,
            lattice_nx=lattice_nx,
            flywheel_sites=flywheel_sites,
            seed=self.seed,
        )
        return self

    def propagate(self, turbulence_level: float = 0.3, **kwargs) -> PropagationResult:
        """Transmit through turbulent channel + Hopf lattice medium."""
        return propagate_asteroid(self, turbulence_level, **kwargs)

    def recover(self) -> RecoveryResult:
        """Identify shell and decode inner payload."""
        return recover_asteroid(self)

    def summary(self) -> dict[str, Any]:
        """Compact status dict for logging / demos."""
        out: dict[str, Any] = {
            "seed": self.seed,
            "payload_type": type(self.payload).__name__,
            "built": self.shell is not None,
        }
        if isinstance(self.payload, str):
            out["payload_preview"] = self.payload[:64]
        elif isinstance(self.payload, (bytes, bytearray)):
            out["payload_len"] = len(self.payload)
        if self.shell is not None:
            out["shell_length"] = self.shell.total_length
            out["shell_hash"] = self.shell.metadata.get("payload_hash")
            fp = self.shell.fourier_fingerprint
            out["fingerprint_l2"] = float(np_l2(fp)) if fp is not None else None
        if self.quaternion is not None:
            out["n_shards"] = len(self.quaternion.shards)
            out["oam_ells"] = list(self.quaternion.composite_weights.keys())
        if self.flux_state is not None:
            out["deposited_momentum"] = self.flux_state.deposited_momentum
            out["mean_twist"] = self.flux_state.metadata.get("mean_twist")
        if self._propagation is not None:
            out["fidelity_proxy"] = self._propagation.fidelity_proxy
            out["turbulence_level"] = self._propagation.turbulence_level
        return out


def np_l2(arr) -> float:
    import numpy as np

    return float(np.linalg.norm(arr))
