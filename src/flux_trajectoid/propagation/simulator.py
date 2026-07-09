"""Turbulent channel + Hopf lattice propagation for Photon Seed Asteroids.

Combines Kolmogorov-like phase screens / jitter with oam_flux lattice
relaxation. Optional BMGL-style gating stub for turbulence mitigation.

Emits a full :class:`~flux_trajectoid.propagation.metrics.FidelityMetrics`
scorecard on each run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .metrics import FidelityMetrics, compute_fidelity_metrics

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid


@dataclass
class PropagationResult:
    """State after propagating an asteroid through a channel."""

    turbulence_level: float
    n_steps: int
    field_final: np.ndarray | None
    field_reference: np.ndarray | None
    lattice_theta_final: np.ndarray | None
    fidelity_proxy: float
    metrics: FidelityMetrics | None = None
    mean_twist_trace: list[float] = field(default_factory=list)
    intensity_trace: list[float] = field(default_factory=list)
    phase_screen_rms: list[float] = field(default_factory=list)
    fidelity_trace: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def metrics_dict(self) -> dict[str, Any]:
        if self.metrics is None:
            return {"fidelity_proxy": self.fidelity_proxy}
        return self.metrics.as_dict()


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
    k2[0, 0] = 1.0
    psd = k2 ** (-(alpha + 2) / 2.0)
    psd[0, 0] = 0.0
    noise = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    screen = np.fft.ifft2(noise * np.sqrt(psd)).real
    screen = screen - screen.mean()
    rms = screen.std() + 1e-12
    return (level * screen / rms).astype(float)


def _bmgl_gate(field: np.ndarray, turbulence_level: float) -> np.ndarray:
    """Lightweight BMGL-style Fourier gate (VQC turbulence mitigation stub)."""
    F = np.fft.fft2(field)
    h, w = field.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
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
    track_fidelity: bool = True,
) -> PropagationResult:
    """
    Propagate a built PhotonSeedAsteroid through a turbulent + lattice medium.

    Computes a multi-metric fidelity scorecard against the pre-channel
    protected field (or first shard field).
    """
    if asteroid.flux_state is None:
        raise RuntimeError("Asteroid must be built before propagate() — call build() first")

    rng = np.random.default_rng(seed if seed is not None else asteroid.seed)
    flux = asteroid.flux_state

    if flux.protected_field is not None:
        field0 = flux.protected_field.astype(complex).copy()
    elif asteroid.quaternion is not None and asteroid.quaternion.fields:
        field0 = asteroid.quaternion.fields[0].astype(complex).copy()
    else:
        field0 = np.ones((64, 64), dtype=complex)

    field = field0.copy()
    ref = field0.copy()
    I0 = float(np.sum(np.abs(field) ** 2)) + 1e-12

    mean_twist_trace: list[float] = []
    intensity_trace: list[float] = []
    phase_rms: list[float] = []
    fidelity_trace: list[float] = []

    from ..inner.oam_flux_coupling import relax_lattice_steps
    from .metrics import field_overlap_fidelity

    use_live_lattice = flux.lattice is not None and hasattr(flux.lattice, "relax_step")

    for step in range(n_steps):
        screen = _kolmogorov_phase_screen(field.shape[0], turbulence_level, rng)
        jitter = rng.normal(0.0, jitter_std * turbulence_level, size=2)
        yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
        tip_tilt = jitter[0] * (xx / field.shape[1]) + jitter[1] * (yy / field.shape[0])
        field = field * np.exp(1j * (screen + tip_tilt))

        if apply_bmgl:
            field = _bmgl_gate(field, turbulence_level)

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

        step_means = relax_lattice_steps(
            flux,
            1,
            recovery_memory=lattice_memory,
            pump_active=False,
            external_turbulence=turbulence_level,
            rng=rng,
        )
        mean_twist_trace.extend(step_means)
        intensity_trace.append(float(np.sum(np.abs(field) ** 2)))
        phase_rms.append(float(screen.std()))
        if track_fidelity:
            fidelity_trace.append(field_overlap_fidelity(ref, field))

    theta_final = flux.lattice_theta
    metrics = compute_fidelity_metrics(
        ref,
        field,
        turbulence_level=turbulence_level,
        n_steps=n_steps,
        extras={
            "I0": I0,
            "If": float(np.sum(np.abs(field) ** 2)),
            "mean_phase_screen_rms": float(np.mean(phase_rms)) if phase_rms else 0.0,
            "apply_bmgl": apply_bmgl,
            "jitter_std": jitter_std,
        },
    )

    asteroid._propagation = PropagationResult(
        turbulence_level=turbulence_level,
        n_steps=n_steps,
        field_final=field,
        field_reference=ref,
        lattice_theta_final=theta_final.copy() if theta_final is not None else None,
        fidelity_proxy=metrics.overlap_fidelity,
        metrics=metrics,
        mean_twist_trace=mean_twist_trace,
        intensity_trace=intensity_trace,
        phase_screen_rms=phase_rms,
        fidelity_trace=fidelity_trace,
        metadata={
            "jitter_std": jitter_std,
            "lattice_memory": lattice_memory,
            "apply_bmgl": apply_bmgl,
            "I0": I0,
            "If": float(np.sum(np.abs(field) ** 2)),
            "lattice_backend": "live" if use_live_lattice else "stub",
            "flux_backend": getattr(flux, "backend", "unknown"),
            "metrics_summary": metrics.summary_line(),
        },
    )
    return asteroid._propagation
