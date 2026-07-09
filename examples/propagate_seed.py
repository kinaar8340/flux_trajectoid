#!/usr/bin/env python3
"""Build a seed, propagate through turbulence, report multi-metric fidelity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flux_trajectoid import PhotonSeedAsteroid


def main() -> None:
    print(
        f"{'turb':>6}  {'F':>7}  {'Icorr':>7}  {'Strehl':>7}  "
        f"{'OAMf':>7}  {'φrms':>7}  {'P':>7}  {'BER':>7}"
    )
    levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    last = None
    for level in levels:
        a = PhotonSeedAsteroid(b"flux_trajectoid_demo", seed=7).build(
            n_shards=4,
            lattice_nx=10,
            n_coupling_steps=3,
            force_stub_flux=True,
            n_points=96,
            scale_grid=3,
            scale_max_iter=2,
        )
        prop = a.propagate(turbulence_level=level, n_steps=12, seed=7)
        m = prop.metrics
        rec = a.recover(mode="photonic")
        ber = rec.byte_error_rate if rec.byte_error_rate is not None else float("nan")
        assert m is not None
        print(
            f"  {level:4.2f}  {m.overlap_fidelity:7.4f}  {m.intensity_correlation:7.4f}  "
            f"{m.strehl_proxy:7.4f}  {m.oam_fidelity:7.4f}  {m.phase_rmse_rad:7.3f}  "
            f"{m.power_retention:7.4f}  {ber:7.3f}"
        )
        last = a

    assert last is not None
    print("\n--- sweep_turbulence() ---")
    rows = last.sweep_turbulence(levels=[0.0, 0.25, 0.5], n_steps=8)
    for r in rows:
        print(
            f"  L={r['turbulence_level']:.2f}  F={r['overlap_fidelity']:.4f}  "
            f"OAM={r['oam_fidelity']:.4f}  BER={r.get('photonic_byte_ber')}"
        )

    rec = last.recover(mode="hybrid")
    print(f"\nHybrid recover: {rec.payload_text!r}  crc={rec.crc_ok}")


if __name__ == "__main__":
    main()
