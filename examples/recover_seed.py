#!/usr/bin/env python3
"""End-to-end: create → propagate → recover."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trajectoid import PhotonSeedAsteroid


def main() -> None:
    message = "macadamia trajectoid photon seed"
    asteroid = PhotonSeedAsteroid(message, seed=42).build()
    print("Built:", asteroid.summary())

    prop = asteroid.propagate(turbulence_level=0.25, n_steps=32)
    print(f"Propagated: fidelity={prop.fidelity_proxy:.4f} steps={prop.n_steps}")

    rec = asteroid.recover()
    print("\n=== Recovery ===")
    print(f"  payload_text: {rec.payload_text!r}")
    print(f"  matches original: {rec.payload_text == message}")
    if rec.shell_match:
        print(f"  shell matched: {rec.shell_match.matched} sim={rec.shell_match.similarity:.4f}")
    print(f"  fidelity_proxy: {rec.fidelity_proxy:.4f}")
    print(f"  emergence_score: {rec.emergence_score:.4f}")
    if rec.flywheel_readout is not None:
        print(f"  flywheel_readout: {rec.flywheel_readout}")


if __name__ == "__main__":
    main()
