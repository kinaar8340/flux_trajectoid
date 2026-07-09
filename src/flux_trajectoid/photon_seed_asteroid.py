"""Photon Seed Asteroid — main orchestrator.

Layers a trajectoid outer shell over VQC quaternion + oam_flux helical
lattice payload for robust photonic data carriers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .inner.oam_flux_coupling import FluxState, couple_to_flux_lattice
from .inner.vqc_encoder import QuaternionEncoding, encode_to_quaternion
from .export.slm import SLMExportResult, export_slm_package
from .propagation.metrics import FidelityMetrics, sweep_turbulence
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
        use_tpt: bool = True,
        force_stub_flux: bool = False,
        n_coupling_steps: int = 16,
        redundancy: int = 1,
        **shell_kwargs,
    ) -> PhotonSeedAsteroid:
        """Assemble outer shell + inner VQC encoding + oam_flux coupling.

        Shell generation uses real trajectoid rolling constraints (path
        scaling + SO(3) mismatch + optional two-period TPT closure).

        Inner coupling prefers the live ``oam_flux`` submodule
        (``TwistLattice`` + VQC multi-ℓ deposition); set
        ``force_stub_flux=True`` to force the numpy stub.
        """
        self.shell = generate_shell(
            self.payload,
            self.seed,
            use_tpt=use_tpt,
            **shell_kwargs,
        )
        self.quaternion = encode_to_quaternion(
            self.payload, n_shards=n_shards, redundancy=redundancy
        )
        self.flux_state = couple_to_flux_lattice(
            self.quaternion,
            self.shell,
            lattice_nx=lattice_nx,
            flywheel_sites=flywheel_sites,
            seed=self.seed,
            n_steps=n_coupling_steps,
            force_stub=force_stub_flux,
        )
        return self

    def propagate(self, turbulence_level: float = 0.3, **kwargs) -> PropagationResult:
        """Transmit through turbulent channel + Hopf lattice medium.

        Result includes ``metrics`` (:class:`FidelityMetrics`) scorecard.
        """
        return propagate_asteroid(self, turbulence_level, **kwargs)

    def recover(self, mode: str = "hybrid", **kwargs) -> RecoveryResult:
        """Identify shell and decode inner payload.

        mode: ``hybrid`` | ``digital`` | ``photonic``
          - digital: lossless ShardPack blocks
          - photonic: field-only OAM inversion (q × scale)
          - hybrid: digital payload + photonic BER/chordal metrics
        """
        return recover_asteroid(self, mode=mode, **kwargs)  # type: ignore[arg-type]

    def export_slm(self, out_dir: str, **kwargs) -> SLMExportResult:
        """Export phase-only SLM hologram package (manifest + phase arrays)."""
        return export_slm_package(self, out_dir, **kwargs)

    def sweep_turbulence(
        self,
        levels: list[float] | None = None,
        *,
        n_steps: int = 12,
        channel_seed: int | None = None,
        apply_bmgl: bool = True,
        recover_photonic: bool = True,
        force_stub_flux: bool = True,
        **build_kwargs,
    ) -> list[dict[str, Any]]:
        """Propagate copies of this seed across turbulence levels; collect metrics."""
        levels = levels if levels is not None else [0.0, 0.1, 0.2, 0.3, 0.5]
        payload = self.payload
        seed = self.seed
        n_shards = (
            int(self.quaternion.metadata.get("n_shards", 8)) if self.quaternion else 8
        )
        lattice_nx = (
            int(self.flux_state.metadata.get("lattice_nx", 12)) if self.flux_state else 12
        )

        def _factory() -> PhotonSeedAsteroid:
            kw = {
                "n_shards": n_shards,
                "lattice_nx": lattice_nx,
                "n_coupling_steps": 4,
                "force_stub_flux": force_stub_flux,
                "n_points": 96,
                "scale_grid": 3,
                "scale_max_iter": 2,
            }
            kw.update(build_kwargs)
            return PhotonSeedAsteroid(payload, seed=seed).build(**kw)

        return sweep_turbulence(
            _factory,
            levels,
            n_steps=n_steps,
            seed=channel_seed if channel_seed is not None else seed,
            apply_bmgl=apply_bmgl,
            recover_photonic=recover_photonic,
        )

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
            out["mismatch_deg"] = self.shell.mismatch_deg
            out["tilt_deg"] = self.shell.tilt_deg
            out["kx"] = self.shell.kx
            out["ky"] = self.shell.ky
            out["global_scale"] = self.shell.global_scale
            out["use_tpt"] = self.shell.use_tpt
            out["is_3d"] = self.shell.is_3d
            out["volume_proxy"] = self.shell.volume_proxy
            if self.shell.mesh_vertices is not None:
                out["mesh_vertices"] = int(self.shell.mesh_vertices.shape[0])
            fp = self.shell.fourier_fingerprint
            out["fingerprint_l2"] = float(np_l2(fp)) if fp is not None else None
        if self.quaternion is not None:
            out["n_shards"] = len(self.quaternion.shards)
            out["oam_ells"] = list(self.quaternion.composite_weights.keys())
        if self.flux_state is not None:
            out["deposited_momentum"] = self.flux_state.deposited_momentum
            out["mean_twist"] = self.flux_state.metadata.get("mean_twist")
            out["flux_backend"] = self.flux_state.backend
            out["momentum_ledger"] = self.flux_state.metadata.get("momentum_ledger")
            out["oam_flux_version"] = self.flux_state.metadata.get("oam_flux_version")
        if self._propagation is not None:
            out["fidelity_proxy"] = self._propagation.fidelity_proxy
            out["turbulence_level"] = self._propagation.turbulence_level
            if self._propagation.metrics is not None:
                m = self._propagation.metrics
                out["oam_fidelity"] = m.oam_fidelity
                out["phase_rmse_rad"] = m.phase_rmse_rad
                out["strehl_proxy"] = m.strehl_proxy
                out["power_retention"] = m.power_retention
        return out


def np_l2(arr) -> float:
    import numpy as np

    return float(np.linalg.norm(arr))
