#!/usr/bin/env python3
"""Build a seed, propagate through turbulence, report fidelity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trajectoid import PhotonSeedAsteroid


def main() -> None:
    asteroid = PhotonSeedAsteroid(b"flux_trajectoid_demo", seed=7).build()
    levels = [0.0, 0.15, 0.3, 0.5]

    print("turbulence  fidelity  mean_twist_end  I_final/I0")
    for level in levels:
        a = PhotonSeedAsteroid(b"flux_trajectoid_demo", seed=7).build()
        prop = a.propagate(turbulence_level=level, n_steps=24)
        ratio = prop.metadata["If"] / (prop.metadata["I0"] + 1e-12)
        print(
            f"  {level:5.2f}     {prop.fidelity_proxy:7.4f}  "
            f"{prop.mean_twist_trace[-1]:12.4f}  {ratio:8.4f}"
        )

    # Keep last asteroid for a quick recover peek
    rec = a.recover()
    print(f"\nRecovered text: {rec.payload_text!r}")
    print(f"Shell match: {rec.shell_match.matched if rec.shell_match else None} "
          f"(sim={rec.shell_match.similarity if rec.shell_match else None:.4f})")
    print(f"Emergence: {rec.emergence_score:.4f}")


if __name__ == "__main__":
    main()
