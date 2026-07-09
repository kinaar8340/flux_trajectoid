"""Outer shell layer: trajectoid geometry, Fourier ID, modulation."""

from .generator import ShellGeometry, generate_shell
from .modulator import ShellModulation, apply_modulation, shell_to_phase_mask

__all__ = [
    "ShellGeometry",
    "generate_shell",
    "ShellModulation",
    "shell_to_phase_mask",
    "apply_modulation",
]
