"""Transmission / simulation layer."""

from .metrics import FidelityMetrics, compute_fidelity_metrics, sweep_turbulence
from .phase_screens import (
    PhaseScreenConfig,
    PhaseScreenEngine,
    convex_defect_available,
    kolmogorov_phase_screen,
    make_phase_screen_engine,
)
from .simulator import PropagationResult, propagate_asteroid

__all__ = [
    "PropagationResult",
    "propagate_asteroid",
    "FidelityMetrics",
    "compute_fidelity_metrics",
    "sweep_turbulence",
    "PhaseScreenConfig",
    "PhaseScreenEngine",
    "kolmogorov_phase_screen",
    "make_phase_screen_engine",
    "convex_defect_available",
]
