"""Outer shell layer: trajectoid geometry, Fourier ID, modulation."""

from .generator import (
    ShellGeometry,
    compute_cumulative_rotations,
    generate_shell,
    scale_path_for_closure,
    two_period_trajectoid_closure,
)
from .mesh3d import TrajectoidMesh3D, build_trajectoid_mesh3d
from .modulator import ShellModulation, apply_modulation, shell_to_phase_mask

__all__ = [
    "ShellGeometry",
    "generate_shell",
    "compute_cumulative_rotations",
    "scale_path_for_closure",
    "two_period_trajectoid_closure",
    "TrajectoidMesh3D",
    "build_trajectoid_mesh3d",
    "ShellModulation",
    "shell_to_phase_mask",
    "apply_modulation",
]
