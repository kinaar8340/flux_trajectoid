# Architecture: Photon Seed Asteroid

**Version:** 0.2 · **Package:** `flux_trajectoid`

## Motivation

Optical free-space and fiber channels suffer from turbulence, mode crosstalk, and identification / routing ambiguity. **flux_trajectoid** packages data as a **Photon Seed Asteroid**:

1. A **hard outer shell** — 3D trajectoid body with unique geometry (fingerprint + protection).
2. A **dense inner nut** — VQC quaternion shards + OAM modes, coupled into `oam_flux` Hopf-lattice flywheels.
3. **Shell-modulated propagation** — geometry constrains and protects the inner field (potential trench).
4. **Hybrid recovery + metrics** — lossless digital path, photonic BER, multi-metric turbulence scorecard.
5. **SLM export** — phase-only holograms for hardware validation.

```
                    ┌─────────────────────────────┐
                    │   PhotonSeedAsteroid        │
                    │   build → propagate →       │
                    │   recover → export_slm      │
                    └─────────────┬───────────────┘
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   ┌─────────────┐        ┌──────────────┐        ┌─────────────┐
   │ 3D Shell    │        │ Inner Nut    │        │ Channel     │
   │ trajectoid  │───────▶│ VQC + OAM    │───────▶│ turbulence  │
   │ mesh+trench │ modulate│ oam_flux     │        │ + lattice   │
   └─────────────┘        └──────────────┘        └──────┬──────┘
                                                         ▼
                                                  ┌─────────────┐
                                                  │ Recovery    │
                                                  │ hybrid/CRC  │
                                                  └─────────────┘
```

---

## Layer details

### 1. Data ingestion

`PhotonSeedAsteroid(payload, seed)` accepts `str` or `bytes`. The seed controls geometric reproducibility; the payload hash drives shell harmonics and packing.

### 2. Outer shell (`shell/`)

#### Planar rolling plan

1. SHA-256 digest → harmonic planar curve  
2. **Path scaling** `(kx, ky)` + perimeter normalization (avoids shrink-to-zero)  
3. Optional **Two-Period Trajectoid (TPT)** closure  
4. Cumulative pure-roll SO(3): `angle = ds / r` about `e_z × tangent`  
5. Fourier fingerprint + arc-length curvature signal  

#### 3D body (`shell/mesh3d.py`) — default `build_3d=True`

1. UV sphere of radius `rolling_radius`  
2. **Contact curve** in body frame: `p_i = R_iᵀ (−r ê_z)`  
3. **Oriented cutting planes** from body-frame ground normals (shaving)  
4. **Contact trench** groove along the spherical image of the path  
5. Triangle mesh + `radial_map` for modulation  

Fourier ID uses an orthographic silhouette of the 3D mesh so identity remains path-unique.

#### Modulation (`shell/modulator.py`)

- 2D path mode: silhouette + boundary trench + azimuthal curvature phase  
- 3D mode: projects `radial_map` deficit → envelope + potential trench + phase  

### 3. Inner payload (`inner/`)

#### Encoding (`vqc_encoder.py`)

Each 4-byte block → `vec ∈ R⁴`, then:

- `q = vec / ‖vec‖` — imprints LG₀ / LG₋₁ / LG₂  
- `s = ‖vec‖` — carrier **amplitude** on LG₁  
- Phase: `φ = Φ · sin(w · π/2)` (odd → sign of `w` recoverable)  

Optional `redundancy` for majority-vote shards. CRC-8 over payload.

#### Flux coupling (`oam_flux_coupling.py`)

Prefers **live** `oam_flux` (submodule `external/oam_flux`):

- `PhotonicsConfig` + `propagate_multi_ell_vectorized`  
- Shared `TwistLattice` + per-ℓ `VQCCouplingState` / `run_vqc_coupling_step`  
- `deposit_on_flywheels` + momentum ledger  
- Shell mismatch attenuates kick strength  

Falls back to numpy Hopf-kick stub (`force_stub=True` or missing package).  
Probe: `oam_flux_backend()` / `is_live_oam_flux()`.

### 4. Transmission (`propagation/`)

- Kolmogorov-like phase screens + tip/tilt jitter  
- Soft BMGL-style Fourier gate  
- Lattice PDE (`TwistLattice.relax_step` when live)  
- **FidelityMetrics** on every run (`metrics.py`)  
- `sweep_turbulence()` for multi-level scorecards + optional photonic BER  

### 5. Recovery (`recovery/`)

| Mode | Use |
|------|-----|
| `digital` | Lossless `ShardPack` blocks |
| `photonic` | LG-matched projection → `(q, s)` → bytes |
| `hybrid` | Digital payload + photonic BER / chordal S³ + CRC-8 |

Also: shell Fourier match, flywheel excess-twist → emergence score.

### 6. Hardware (`export/slm.py`)

Phase-only holograms from protected / composite / shard / propagated fields:

- Device presets (generic, Holoeye, Meadowlark, Thorlabs-class)  
- Optional Gerchberg–Saxton  
- Shell phase bias from trench / curvature  
- Package: `manifest.json`, `.npy`, `.raw`, `.png`, stack montage  

---

## Data flow

```
payload + seed
  → generate_shell (2D plan + 3D mesh)
  → encode_to_quaternion (ShardPacks + OAM fields)
  → couple_to_flux_lattice (TwistLattice + protected field)
  → propagate_asteroid (channel + FidelityMetrics)
  → recover_asteroid (hybrid / digital / photonic)
  → export_slm (optional hardware package)
```

---

## Design principles

1. **Determinism** — same `(payload, seed)` → same shell + packing.  
2. **Graceful backends** — live `oam_flux` when present; stubs always run.  
3. **Honest metrics** — hybrid recovery does not hide photonic BER.  
4. **Separation** — shell ID independent of inner decode; modulation is the coupling surface.  
5. **Hardware path** — SLM export is a first-class hook, not an afterthought.  

---

## Key modules

| Module | Responsibility |
|--------|----------------|
| `photon_seed_asteroid.py` | Orchestrator |
| `shell/generator.py` | Path + rolling + shell assembly |
| `shell/mesh3d.py` | 3D trajectoid mesh |
| `shell/modulator.py` | Phase / trench masks |
| `inner/vqc_encoder.py` | Quaternion / OAM packing |
| `inner/oam_flux_coupling.py` | Lattice coupling |
| `propagation/simulator.py` | Channel |
| `propagation/metrics.py` | Fidelity scorecard |
| `recovery/decoder.py` | Hybrid recovery |
| `export/slm.py` | SLM package |

---

## Extension points

| Direction | Where |
|-----------|--------|
| Finer shaving / developable surface | `shell/mesh3d.py` |
| Full ICA / Isomap demix | `recovery/decoder.py` |
| Emergence / golden-angle sweeps | `oam_flux` + `sweep_turbulence` |
| Real SLM driver | consume `phase_levels.raw` from export package |
