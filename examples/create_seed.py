#!/usr/bin/env python3
"""Create a Photon Seed Asteroid and print / plot shell + inner summary."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from flux_trajectoid import PhotonSeedAsteroid


def main() -> None:
    payload = "hello photon seed asteroid"
    asteroid = PhotonSeedAsteroid(payload, seed=42).build()
    summary = asteroid.summary()

    print("=== Photon Seed Asteroid ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    assert asteroid.shell is not None
    assert asteroid.quaternion is not None
    assert asteroid.flux_state is not None

    print("\nShell (trajectoid):")
    print(f"  vertices: {asteroid.shell.vertices.shape}")
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
    print(f"  packets: {asteroid.flux_state.metadata.get('ells')}")
    print(f"  deposited_momentum: {asteroid.flux_state.deposited_momentum:.6f}")
    print(f"  flywheel_load: {asteroid.flux_state.flywheel_load}")
    print(f"  mean_twist: {asteroid.flux_state.metadata.get('mean_twist'):.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121, projection="3d")
        v = asteroid.shell.vertices
        ax1.plot(v[:, 0], v[:, 1], v[:, 2], color="C0")
        ax1.set_title("Trajectoid shell curve")

        ax2 = fig.add_subplot(122)
        if asteroid.flux_state.protected_field is not None:
            ax2.imshow(
                np.abs(asteroid.flux_state.protected_field) ** 2,
                origin="lower",
                cmap="magma",
            )
            ax2.set_title("|protected field|²")
        else:
            ax2.plot(asteroid.shell.curvature_signal)
            ax2.set_title("Unrolled curvature signal")

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
