"""Helical twist packets + Hopf lattice flux flywheels (adapted from oam_flux).

Couples VQC quaternion/OAM state into a photon "seed" (LG OAM modes)
and deposits flux onto a gauged Hopf lattice of flywheel resonators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..shell.generator import ShellGeometry
from ..shell.modulator import ShellModulation, apply_modulation, shell_to_phase_mask
from .vqc_encoder import QuaternionEncoding, oam_weights_to_field


def helical_seed(nx: int, *, pitch: float = 0.35, amplitude: float = 1.2) -> np.ndarray:
    """Two-gyro helical IC (oam_flux lattice convention)."""
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return amplitude * (0.5 + 0.5 * np.sin(pitch * (x + 2 * y - z)))


@dataclass
class OAMPacket:
    """Propagating ℓ-mode twist packet with kinetic flux bookkeeping."""

    ell: int = 3
    lambda_nm: float = 1550.0
    w0: float = 1.0
    energy_scale: float = 1.0
    z: float = 0.0

    @property
    def helical_phase_gradient(self) -> float:
        return float(self.ell)

    @property
    def momentum(self) -> float:
        # Proxy: p ∝ |ℓ| · E_scale (oam_flux kinetic momentum scaling)
        return abs(self.ell) * self.energy_scale


@dataclass
class FluxState:
    """Inner nut state after coupling to Hopf lattice flywheels."""

    packets: list[OAMPacket]
    lattice_theta: np.ndarray
    lattice_theta0: np.ndarray
    deposited_momentum: float
    flywheel_load: np.ndarray
    protected_field: np.ndarray | None
    shell_modulation: ShellModulation | None
    kick_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _hopf_fiber_coords(nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 3-torus voxels to Hopf-style (rho, phi, eta)."""
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    rho = np.sqrt((x - np.pi) ** 2 + (y - np.pi) ** 2)
    rho = rho / (rho.max() + 1e-12)
    phi = np.arctan2(y - np.pi, x - np.pi)
    eta = np.mod(x + y + z, 2 * np.pi)
    return rho, phi, eta


def _flywheel_indices(nx: int, n_sites: int, seed: int = 0) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in rng.integers(0, nx, size=3)) for _ in range(n_sites)]


def couple_to_flux_lattice(
    encoding: QuaternionEncoding,
    shell: ShellGeometry | None = None,
    *,
    lattice_nx: int = 16,
    flywheel_sites: int = 4,
    kick_strength: float = 0.05,
    seed: int = 42,
) -> FluxState:
    """
    Couple VQC OAM state into helical packets + Hopf lattice flux.

    1. Build OAMPackets from composite mode spectrum
    2. Optionally apply shell phase-mask protection
    3. Deposit flux kicks onto flywheel resonator neighborhoods
    """
    weights = encoding.composite_weights
    packets = [
        OAMPacket(ell=int(ell), energy_scale=float(abs(amp)))
        for ell, amp in sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    ]
    if not packets:
        packets = [OAMPacket(ell=1, energy_scale=1.0)]

    # Shell protection
    modulation: ShellModulation | None = None
    protected: np.ndarray | None = None
    if shell is not None:
        modulation = shell_to_phase_mask(shell, grid_size=64)
        field = oam_weights_to_field(weights, grid_size=64)
        # Use dominant ell for bias
        dom_ell = max(weights, key=lambda e: abs(weights[e]))
        protected = apply_modulation(field, modulation, ell=int(dom_ell))

    # Lattice + deposit
    theta = helical_seed(lattice_nx)
    theta0 = theta.copy()
    rho, phi, eta = _hopf_fiber_coords(lattice_nx)
    sites = _flywheel_indices(lattice_nx, flywheel_sites, seed=seed)

    total_mom = 0.0
    kick_history: list[float] = []
    flywheel_load = np.zeros(flywheel_sites)

    for i, pkt in enumerate(packets):
        # Helical OAM winding along Hopf fiber
        helical = np.cos(pkt.ell * phi + 0.5 * eta)
        radial = np.exp(-((rho * 2.0) ** 2)) * (rho ** min(abs(pkt.ell), 4))
        kick = kick_strength * pkt.energy_scale * radial * helical

        # Restrict to flywheel neighborhoods
        mask = np.zeros_like(theta)
        for j, idx in enumerate(sites):
            local = np.zeros_like(mask)
            local[idx] = 1.0
            for ax in range(3):
                local = 0.25 * (np.roll(local, 1, ax) + np.roll(local, -1, ax) + 2 * local)
            mask += local
            flywheel_load[j] += float(np.abs(kick[idx]))
        mask /= max(mask.max(), 1e-12)
        kick = kick * mask

        mom = float(np.abs(kick).sum() * kick_strength)
        theta = np.clip(theta + kick, 0.01, 2 * np.pi - 0.01)
        total_mom += mom
        kick_history.append(mom)

        # Soft shell fingerprint bias on mean twist
        if shell is not None and shell.fourier_fingerprint is not None:
            fp = shell.fourier_fingerprint
            theta = theta + 0.01 * float(fp[i % len(fp)])

    return FluxState(
        packets=packets,
        lattice_theta=theta,
        lattice_theta0=theta0,
        deposited_momentum=total_mom,
        flywheel_load=flywheel_load,
        protected_field=protected,
        shell_modulation=modulation,
        kick_history=kick_history,
        metadata={
            "lattice_nx": lattice_nx,
            "flywheel_sites": flywheel_sites,
            "kick_strength": kick_strength,
            "n_packets": len(packets),
            "ells": [p.ell for p in packets],
            "mean_twist": float(theta.mean()),
            "twist_variance": float(theta.var()),
        },
    )
