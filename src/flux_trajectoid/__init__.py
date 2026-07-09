"""flux_trajectoid — Photon Seed Asteroids (trajectoid shells + VQC + oam_flux)."""

from .photon_seed_asteroid import PhotonSeedAsteroid
from .shell.generator import ShellGeometry, generate_shell
from .inner.vqc_encoder import encode_to_quaternion
from .inner.oam_flux_coupling import (
    couple_to_flux_lattice,
    is_live_oam_flux,
    oam_flux_backend,
)
from .propagation.simulator import propagate_asteroid
from .recovery.decoder import recover_asteroid

__version__ = "0.1.2"

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
    "__version__",
]
