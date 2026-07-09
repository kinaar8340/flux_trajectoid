"""Inner payload: VQC quaternion encoding + oam_flux lattice coupling."""

from .oam_flux_coupling import (
    FluxState,
    OAMPacket,
    couple_to_flux_lattice,
    is_live_oam_flux,
    oam_flux_backend,
    relax_lattice_steps,
)
from .vqc_encoder import QuaternionEncoding, encode_to_quaternion

__all__ = [
    "QuaternionEncoding",
    "encode_to_quaternion",
    "FluxState",
    "OAMPacket",
    "couple_to_flux_lattice",
    "is_live_oam_flux",
    "oam_flux_backend",
    "relax_lattice_steps",
]
