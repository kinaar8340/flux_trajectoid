#!/usr/bin/env python3
"""Create a Photon Seed Asteroid and print / plot shell + inner summary."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from flux_trajectoid import PhotonSeedAsteroid, oam_flux_backend


def main() -> None:
    payload = "hello photon seed asteroid"
    print(f"oam_flux backend: {oam_flux_backend()}")
    asteroid = PhotonSeedAsteroid(payload, seed=42).build(
        lattice_nx=12,
        n_coupling_steps=8,
    )
    summary = asteroid.summary()

    print("=== Photon Seed Asteroid ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    assert asteroid.shell is not None
    assert asteroid.quaternion is not None
    assert asteroid.flux_state is not None

    print("\nShell (trajectoid):")
    print(f"  is_3d: {asteroid.shell.is_3d}")
    print(f"  vertices (silhouette): {asteroid.shell.vertices.shape}")
    print(f"  total_length: {asteroid.shell.total_length:.4f}")
    print(
        f"  kx={asteroid.shell.kx:.4f}  ky={asteroid.shell.ky:.4f}  "
        f"global_scale={asteroid.shell.global_scale:.4f}"
    )
    print(
        f"  mismatch_deg: {asteroid.shell.mismatch_deg:.4f}  "
        f"(base={asteroid.shell.metadata.get('mismatch_base_deg', float('nan')):.4f}, "
        f"scaled={asteroid.shell.metadata.get('mismatch_scaled_deg', float('nan')):.4f})"
    )
    print(f"  tilt_deg: {asteroid.shell.tilt_deg:.4f}")
    print(f"  use_tpt: {asteroid.shell.use_tpt}")
    if asteroid.shell.is_3d and asteroid.shell.mesh_vertices is not None:
        print(
            f"  mesh: V={asteroid.shell.mesh_vertices.shape[0]} "
            f"F={asteroid.shell.mesh_faces.shape[0] if asteroid.shell.mesh_faces is not None else 0}"
        )
        print(
            f"  volume_proxy={asteroid.shell.volume_proxy:.4f}  "
            f"mean_r={asteroid.shell.metadata.get('mean_radius')}  "
            f"contact_coverage={asteroid.shell.metadata.get('contact_coverage')}"
        )
    print(f"  fingerprint: {np.array2string(asteroid.shell.fourier_fingerprint, precision=3)}")
    if asteroid.shell.phase_trench_mask is not None:
        print(
            f"  phase_trench_mask: len={len(asteroid.shell.phase_trench_mask)} "
            f"rms={float(np.std(asteroid.shell.phase_trench_mask)):.4f}"
        )

    print("\nInner VQC:")
    print(f"  shards: {len(asteroid.quaternion.shards)}")
    q0 = asteroid.quaternion.primary_quaternion
    print(f"  primary q: w={q0.w:.3f} x={q0.x:.3f} y={q0.y:.3f} z={q0.z:.3f}")
    print(f"  composite ells: {list(asteroid.quaternion.composite_weights.keys())}")

    print("\nFlux lattice:")
    print(f"  backend: {asteroid.flux_state.backend}")
    print(f"  packets: {asteroid.flux_state.metadata.get('ells')}")
    print(f"  deposited_momentum: {asteroid.flux_state.deposited_momentum:.6f}")
    print(f"  flywheel_load: {asteroid.flux_state.flywheel_load}")
    print(f"  mean_twist: {asteroid.flux_state.metadata.get('mean_twist'):.4f}")
    if asteroid.flux_state.metadata.get("momentum_ledger") is not None:
        print(f"  momentum_ledger: {asteroid.flux_state.metadata['momentum_ledger']}")
    if asteroid.flux_state.metadata.get("oam_flux_version"):
        print(f"  oam_flux_version: {asteroid.flux_state.metadata['oam_flux_version']}")
    if asteroid.flux_state.lattice is not None:
        print(f"  live TwistLattice nx={asteroid.flux_state.lattice.nx}")
    if asteroid.flux_state.coupling_history:
        print(f"  coupling_history steps: {len(asteroid.flux_state.coupling_history)}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131, projection="3d")
        if asteroid.shell.is_3d and asteroid.shell.surface is not None:
            s = asteroid.shell.surface
            ax1.plot_surface(
                s[:, :, 0], s[:, :, 1], s[:, :, 2],
                rstride=2, cstride=2, color="C0", alpha=0.75, linewidth=0,
            )
            if asteroid.shell.path_on_body is not None:
                p = asteroid.shell.path_on_body
                ax1.plot(p[:, 0], p[:, 1], p[:, 2], color="C3", lw=1.5)
            ax1.set_title("3D trajectoid shell")
        else:
            v = asteroid.shell.vertices
            ax1.plot(v[:, 0], v[:, 1], v[:, 2], color="C0")
            ax1.set_title("Trajectoid path")

        ax2 = fig.add_subplot(132)
        if asteroid.shell.radial_map is not None:
            ax2.imshow(asteroid.shell.radial_map, origin="lower", cmap="viridis", aspect="auto")
            ax2.set_title("Radial map (trench / shave)")
        else:
            ax2.plot(asteroid.shell.curvature_signal)
            ax2.set_title("Curvature signal")

        ax3 = fig.add_subplot(133)
        if asteroid.flux_state.protected_field is not None:
            ax3.imshow(
                np.abs(asteroid.flux_state.protected_field) ** 2,
                origin="lower",
                cmap="magma",
            )
            ax3.set_title("|protected field|²")
        else:
            ax3.plot(asteroid.shell.curvature_signal)
            ax3.set_title("Unrolled curvature")

        fig.tight_layout()
        path = out_dir / "create_seed.png"
        fig.savefig(path, dpi=120)
        print(f"\nWrote {path}")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"\n(plot skipped: {exc})")

    print("\nOK — seed built successfully.")


if __name__ == "__main__":
    main()
