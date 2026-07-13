"""flux_trajectoid — Photon Seed Asteroids (trajectoid shells + VQC + oam_flux)."""

from .export.slm import SLMConfig, SLMExportResult, export_slm_package
from .inner.oam_flux_coupling import (
    couple_to_flux_lattice,
    is_live_oam_flux,
    oam_flux_backend,
)
from .inner.vqc_encoder import encode_to_quaternion
from .photon_seed_asteroid import PhotonSeedAsteroid
from .propagation.metrics import FidelityMetrics, compute_fidelity_metrics, sweep_turbulence
from .propagation.phase_screens import convex_defect_available, make_phase_screen_engine
from .propagation.simulator import propagate_asteroid
from .recovery.decoder import recover_asteroid
from .shell.generator import ShellGeometry, generate_shell

__version__ = "0.2.0"

__all__ = [
    "PhotonSeedAsteroid",
    "ShellGeometry",
    "generate_shell",
    "encode_to_quaternion",
    "couple_to_flux_lattice",
    "is_live_oam_flux",
    "oam_flux_backend",
    "propagate_asteroid",
    "recover_asteroid",
    "FidelityMetrics",
    "compute_fidelity_metrics",
    "sweep_turbulence",
    "convex_defect_available",
    "make_phase_screen_engine",
    "SLMConfig",
    "SLMExportResult",
    "export_slm_package",
    "__version__",
]
