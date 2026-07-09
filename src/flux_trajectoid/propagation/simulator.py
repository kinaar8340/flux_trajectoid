"""Turbulent channel + Hopf lattice propagation for Photon Seed Asteroids.

Combines Kolmogorov-like phase screens / jitter with oam_flux lattice
relaxation. Optional BMGL-style gating stub for turbulence mitigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid


@dataclass
class PropagationResult:
    """State after propagating an asteroid through a channel."""

    turbulence_level: float
    n_steps: int
    field_final: np.ndarray | None
    lattice_theta_final: np.ndarray | None
    fidelity_proxy: float
    mean_twist_trace: list[float] = field(default_factory=list)
    intensity_trace: list[float] = field(default_factory=list)
    phase_screen_rms: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _kolmogorov_phase_screen(
    size: int,
    level: float,
    rng: np.random.Generator,
    alpha: float = 5.0 / 3.0,
) -> np.ndarray:
    """Simplified Kolmogorov-like phase screen in Fourier domain."""
    fx = np.fft.fftfreq(size)
    fy = np.fft.fftfreq(size)
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid div0
    # Power spectrum ~ k^{-(alpha+2)} rough proxy for phase
    psd = k2 ** (-(alpha + 2) / 2.0)
    psd[0, 0] = 0.0
    noise = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    screen = np.fft.ifft2(noise * np.sqrt(psd)).real
    screen = screen - screen.mean()
    rms = screen.std() + 1e-12
    return (level * screen / rms).astype(float)


def _lattice_relax_step(
    theta: np.ndarray,
    *,
    dt: float = 0.001,
    D: float = 0.05,
    kappa: float = 0.85,
    delta_omega: float = 0.002,
    recovery_alpha: float = 0.0,
    theta0: np.ndarray | None = None,
) -> np.ndarray:
    """Single Hopf-lattice PDE step (oam_flux TwistLattice.relax_step simplified)."""
    t = theta
    lap = (
        np.roll(t, 1, 0)
        + np.roll(t, -1, 0)
        + np.roll(t, 1, 1)
        + np.roll(t, -1, 1)
        + np.roll(t, 1, 2)
        + np.roll(t, -1, 2)
        - 6 * t
    ) * (theta.shape[0] ** 2)
    mean_twist = float(t.mean())
    gauge = -kappa * mean_twist
    rhs = D * lap + delta_omega + gauge
    t = np.clip(t + dt * rhs, 0.01, 2 * np.pi - 0.01)
    if recovery_alpha > 0.0 and theta0 is not None:
        t = np.clip(t + recovery_alpha * (theta0 - t), 0.01, 2 * np.pi - 0.01)
    return t


def _bmgl_gate(field: np.ndarray, turbulence_level: float) -> np.ndarray:
    """
    Lightweight BMGL-style gating stub (VQC turbulence mitigation).

    Soft-thresholds high-spatial-frequency noise while preserving
    low-order OAM structure.
    """
    F = np.fft.fft2(field)
    h, w = field.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Keep core modes; gate more aggressively as turbulence rises
    cutoff = max(2.0, 0.35 * min(h, w) * (1.0 - 0.5 * turbulence_level))
    gate = np.exp(-(r**2) / (2.0 * cutoff**2))
    gate = np.fft.ifftshift(gate)
    return np.fft.ifft2(F * gate)


def propagate_asteroid(
    asteroid: PhotonSeedAsteroid,
    turbulence_level: float = 0.3,
    *,
    n_steps: int = 32,
    jitter_std: float = 0.02,
    lattice_memory: float = 0.15,
    seed: int | None = None,
    apply_bmgl: bool = True,
) -> PropagationResult:
    """
    Propagate a built PhotonSeedAsteroid through a turbulent + lattice medium.
    """
    if asteroid.flux_state is None:
        raise RuntimeError("Asteroid must be built before propagate() — call build() first")

    rng = np.random.default_rng(seed if seed is not None else asteroid.seed)
    flux = asteroid.flux_state

    # Start field: protected field or synthesize from encoding
    if flux.protected_field is not None:
        field = flux.protected_field.astype(complex).copy()
    elif asteroid.quaternion is not None and asteroid.quaternion.fields:
        field = asteroid.quaternion.fields[0].astype(complex).copy()
    else:
        field = np.ones((64, 64), dtype=complex)

    theta = flux.lattice_theta.copy()
    theta0 = flux.lattice_theta0.copy()
    I0 = float(np.sum(np.abs(field) ** 2)) + 1e-12

    mean_twist_trace: list[float] = []
    intensity_trace: list[float] = []
    phase_rms: list[float] = []

    recovery_alpha = lattice_memory * (1.0 - np.exp(-1.0 / 25.0))

    for step in range(n_steps):
        # Turbulent phase screen + tip/tilt jitter
        screen = _kolmogorov_phase_screen(field.shape[0], turbulence_level, rng)
        jitter = rng.normal(0.0, jitter_std * turbulence_level, size=2)
        yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
        tip_tilt = jitter[0] * (xx / field.shape[1]) + jitter[1] * (yy / field.shape[0])
        field = field * np.exp(1j * (screen + tip_tilt))

        if apply_bmgl:
            field = _bmgl_gate(field, turbulence_level)

        # Shell protection each step: phase bias + soft trench (do NOT
        # re-multiply the full amplitude envelope — that would decay as env^n).
        if flux.shell_modulation is not None:
            mod = flux.shell_modulation
            phase = mod.phase_mask
            trench = mod.potential_trench
            if phase.shape != field.shape:
                from scipy.ndimage import zoom

                zy = field.shape[0] / phase.shape[0]
                zx = field.shape[1] / phase.shape[1]
                phase = zoom(phase, (zy, zx), order=1)
                trench = zoom(trench, (zy, zx), order=1)
            suppress = np.exp(-0.05 * trench)
            field = field * suppress * np.exp(1j * 0.15 * phase)

        # Lattice medium evolution
        external = turbulence_level * 0.01 * rng.normal(size=theta.shape)
        theta = _lattice_relax_step(
            theta + external,
            recovery_alpha=recovery_alpha,
            theta0=theta0,
        )

        mean_twist_trace.append(float(theta.mean()))
        intensity_trace.append(float(np.sum(np.abs(field) ** 2)))
        phase_rms.append(float(screen.std()))

    If = float(np.sum(np.abs(field) ** 2))
    # Fidelity proxy: intensity retention × phase coherence with initial
    if flux.protected_field is not None:
        ref = flux.protected_field
        # Pad/crop if needed
        if ref.shape != field.shape:
            ref = field  # fallback
        overlap = np.vdot(ref.ravel(), field.ravel())
        fidelity = float(np.abs(overlap) ** 2 / ((np.vdot(ref, ref).real + 1e-12) * (If + 1e-12)))
    else:
        fidelity = float(If / I0)

    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    # Store final state back on asteroid for recovery
    asteroid._propagation = PropagationResult(
        turbulence_level=turbulence_level,
        n_steps=n_steps,
        field_final=field,
        lattice_theta_final=theta,
        fidelity_proxy=fidelity,
        mean_twist_trace=mean_twist_trace,
        intensity_trace=intensity_trace,
        phase_screen_rms=phase_rms,
        metadata={
            "jitter_std": jitter_std,
            "lattice_memory": lattice_memory,
            "apply_bmgl": apply_bmgl,
            "I0": I0,
            "If": If,
        },
    )
    return asteroid._propagation
