# Metrics interpretation guide

Every `propagate()` call attaches a `FidelityMetrics` object to `PropagationResult.metrics`.

## Scorecard fields

| Field | Symbol | Range | Interpretation |
|-------|--------|-------|----------------|
| `overlap_fidelity` | F | [0, 1] | Pure-state field fidelity \|⟨ref\|obs⟩\|² / (‖ref‖²‖obs‖²). Primary quality number. |
| `intensity_correlation` | Icorr | [−1, 1] | Pearson correlation of intensity maps. High when structure survives even if phase is scrambled. |
| `power_retention` | P | ≥ 0 | ∫\|obs\|² / ∫\|ref\|². Drops under BMGL gating + trench re-application. |
| `strehl_proxy` | Strehl | ≥ 0 | Peak intensity ratio. Sensitive to focusing / constructive interference. |
| `phase_rmse_rad` | φrms | ≥ 0 rad | Intensity-weighted RMS phase error after global phase alignment. |
| `oam_fidelity` | OAMf | [0, 1] | Cosine similarity of LG mode *magnitude* spectra. Measures angular-momentum structure. |
| `oam_phase_error_rad` | — | ≥ 0 | Mean \|Δphase\| on shared OAM modes. |
| `tip_tilt_rms` | tt | ≥ 0 | Weighted phase-gradient energy (residual pointing). |

Quick print:

```python
print(prop.metrics.summary_line())
# F=0.21  Icorr=0.34  P=0.08  Strehl=0.04  φrms=1.29  OAM=0.99  tt=…
```

## How to read a turbulence sweep

Typical qualitative pattern in this codebase:

| Turbulence ↑ | Field F | OAMf | Photonic BER |
|--------------|---------|------|--------------|
| Low | high → moderate | stays high | may stay low on clean refs |
| High | low | still relatively high | often high (channel limited) |

**Takeaway:** OAM spectral structure is more robust than coherent field overlap. That is why the seed can keep an “identity” in angular momentum even when raw hologram fidelity collapses — and why hybrid recovery reports photonic BER honestly while still delivering CRC-correct digital payload for software demos.

## `sweep_turbulence()`

```python
rows = ast.sweep_turbulence(
    levels=[0.0, 0.1, 0.2, 0.3, 0.5],
    n_steps=12,
    recover_photonic=True,
)
for r in rows:
    print(r["turbulence_level"], r["overlap_fidelity"], r["oam_fidelity"], r.get("photonic_byte_ber"))
```

Each row includes the full scorecard keys plus optional:

- `photonic_byte_ber` / `photonic_bit_ber`
- `crc_ok`, `chordal_error`

## Recovery metrics (post-channel)

| Field | Meaning |
|-------|---------|
| `crc_ok` | Payload matches encode-time CRC-8 |
| `byte_error_rate` | Digital vs photonic byte disagreement |
| `bit_error_rate` | Bit-level BER |
| `chordal_error_mean` | Mean S³ chordal distance of recovered quaternions |
| `emergence_score` | Flywheel excess-twist proxy from the lattice |

## Design notes

1. **Reference field** is the pre-channel protected field (or first shard).  
2. **Global phase** is aligned before phase RMSE.  
3. **BMGL gating** intentionally reduces high-k content; power retention will drop even at zero turbulence.  
4. For hardware campaigns, track **OAMf + BER** alongside F — not F alone.  
