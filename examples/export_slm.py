#!/usr/bin/env python3
"""Export SLM hologram package for a Photon Seed Asteroid."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trajectoid import PhotonSeedAsteroid


def main() -> None:
    out = ROOT / "outputs" / "slm_package"
    ast = PhotonSeedAsteroid("SLM photon seed", seed=42).build(
        n_shards=4,
        lattice_nx=10,
        n_coupling_steps=2,
        force_stub_flux=True,
        n_points=96,
        scale_grid=3,
        scale_max_iter=2,
    )
    result = ast.export_slm(
        out,
        preset="generic_256",
        source="protected",
        stack_shards=True,
        include_shell_bias=True,
        use_gs=False,
    )
    print("SLM export →", result.out_dir)
    print("files:", ", ".join(result.files))
    print("phase shape:", result.phase_rad.shape)
    print("levels dtype:", result.phase_levels.dtype, "range",
          int(result.phase_levels.min()), "–", int(result.phase_levels.max()))
    if result.phase_stack is not None:
        print("stack frames:", result.phase_stack.shape[0])
    print("manifest device:", result.manifest.get("device"))


if __name__ == "__main__":
    main()
