#!/usr/bin/env python3
"""End-to-end: create → propagate → recover (digital / photonic / hybrid)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trajectoid import PhotonSeedAsteroid


def _print_rec(label: str, rec) -> None:
    print(f"\n=== Recovery ({label}) ===")
    print(f"  mode/path: {rec.mode}")
    print(f"  payload_text: {rec.payload_text!r}")
    print(f"  crc_ok: {rec.crc_ok}")
    if rec.byte_error_rate is not None:
        print(f"  photonic BER (bytes): {rec.byte_error_rate:.4f}")
        print(f"  photonic BER (bits):  {rec.bit_error_rate:.4f}")
    if rec.chordal_error_mean is not None:
        print(f"  mean chordal S³ error: {rec.chordal_error_mean:.4f}")
    if rec.shell_match:
        print(f"  shell matched: {rec.shell_match.matched} sim={rec.shell_match.similarity:.4f}")
    print(f"  fidelity_proxy: {rec.fidelity_proxy:.4f}")
    print(f"  emergence_score: {rec.emergence_score:.4f}")


def main() -> None:
    message = "macadamia trajectoid photon seed"
    asteroid = PhotonSeedAsteroid(message, seed=42).build(
        n_shards=8,
        redundancy=1,
        lattice_nx=12,
        n_coupling_steps=4,
    )
    print("Built:", asteroid.summary())

    # Clean (no channel) hybrid + photonic
    rec_clean = asteroid.recover(mode="hybrid")
    _print_rec("clean/hybrid", rec_clean)
    rec_ph_clean = asteroid.recover(mode="photonic")
    _print_rec("clean/photonic", rec_ph_clean)

    prop = asteroid.propagate(turbulence_level=0.25, n_steps=16)
    print(f"\nPropagated: fidelity={prop.fidelity_proxy:.4f} steps={prop.n_steps}")

    rec = asteroid.recover(mode="hybrid")
    _print_rec("after channel/hybrid", rec)
    print(f"  matches original: {rec.payload_text == message}")

    rec_ph = asteroid.recover(mode="photonic")
    _print_rec("after channel/photonic", rec_ph)

    rec_dig = asteroid.recover(mode="digital")
    _print_rec("digital", rec_dig)
    if rec.flywheel_readout is not None:
        print(f"  flywheel_readout: {rec.flywheel_readout}")


if __name__ == "__main__":
    main()
