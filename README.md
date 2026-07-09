# flux_trajectoid

**Photon Seed Asteroids** — trajectoid shells + VQC quaternion + oam_flux helical lattice for robust photonic data carriers.

Inspired by macadamia nuts (hard shell, dense kernel) and [trajectoids](https://en.wikipedia.org/wiki/Trajectoid) (bodies whose rolling path encodes a closed curve), this project models a layered biomimetic photonic packet that propagates as a protected “seed” through simulated optical channels.

## Core concept

| Layer | Role |
|-------|------|
| **Outer shell** | Hard, uniquely shaped geometry (trajectoid-inspired). Fourier descriptors + arc-length unrolling for ID / metadata / protection. |
| **Inner nut** | Dense payload via VQC (quaternion + OAM) and oam_flux (helical twist packets + Hopf lattice flux flywheels). |
| **Whole unit** | Propagates as a protected seed; shell modulates/protects inner modes (“potential trench”). |

## Architecture

```
Photon Seed Asteroid (flux_trajectoid)
│
├── 1. DATA INGESTION LAYER
│   └── packet → metadata + raw_data
│
├── 2. OUTER SHELL LAYER (Trajectoid Geometry)
│   ├── shell/generator.py
│   │   ├── Parametric closed surface (trajectoid-style)
│   │   ├── Fourier descriptors (boundary → coefficients)
│   │   └── Arc-length unrolling → 1D path (metadata signal)
│   └── Protection + Identification
│       └── Unique shape fingerprint (routing / error resilience)
│
├── 3. INNER PAYLOAD LAYER (VQC + oam_flux)
│   ├── inner/vqc_encoder.py       ← Quaternion shards + Orbital Braille
│   ├── inner/oam_flux_coupling.py ← Helical twist packets + Hopf lattice
│   └── Inner data packing (DNA-like density)
│
├── 4. MODULATION & PROTECTION LAYER
│   └── shell/modulator.py
│       └── Shell → phase mask / constraint on inner OAM modes
│
├── 5. TRANSMISSION / SIMULATION LAYER
│   └── propagation/simulator.py
│       ├── Turbulent channel (Kolmogorov + jitter)
│       └── Hopf lattice medium + BMGL gating stub
│
└── 6. RECOVERY / DECODING LAYER
    ├── recovery/shell_identifier.py
    └── recovery/decoder.py
```

## Related projects

| Repo | Role |
|------|------|
| [vqc_proto](https://github.com/kinaar8340/vqc_proto) | Quaternion + Orbital Braille / OAM imprint |
| [oam_flux](https://github.com/kinaar8340/oam_flux) | Helical packets, Hopf lattice, flux flywheels |
| [vqc_sims_public](https://github.com/kinaar8340/vqc_sims_public) | Photonics simulation lineage |

Optional git submodules under `external/` after clone (see below).

## Install

```bash
cd ~/Projects/flux_trajectoid
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from flux_trajectoid import PhotonSeedAsteroid

seed = PhotonSeedAsteroid("hello photon seed", seed=42).build()
print(seed.summary())

prop = seed.propagate(turbulence_level=0.3)
rec = seed.recover(mode="hybrid")  # digital | photonic | hybrid
print(rec.payload_text, rec.crc_ok, rec.byte_error_rate, rec.chordal_error_mean)
```

Recovery handles quaternion lossiness by packing **unit q + scale** (carrier amplitude side-channel), with CRC-8 and optional shard redundancy.

Examples:

```bash
python examples/create_seed.py
python examples/propagate_seed.py
python examples/recover_seed.py
```

## Project layout

```
flux_trajectoid/
├── configs/default.yaml
├── src/flux_trajectoid/
│   ├── photon_seed_asteroid.py   # orchestrator
│   ├── shell/                    # trajectoid geometry + modulator
│   ├── inner/                    # VQC + oam_flux coupling
│   ├── propagation/              # turbulence + lattice
│   ├── recovery/                 # shell ID + decoder
│   └── utils/                    # Fourier + quaternions
├── examples/
├── tests/
└── docs/architecture.md
```

## Submodules

```bash
git clone --recurse-submodules https://github.com/kinaar8340/flux_trajectoid.git
# or after clone:
git submodule update --init --recursive
```

| Path | Role |
|------|------|
| `external/oam_flux` | **Live** Hopf lattice + VQC flux deposition (used by default when present) |
| `external/vqc_proto` | Quaternion / Orbital Braille reference |
| `external/vqc_sims_public` | Photonics simulation lineage |

Override discovery with `FLUX_TRAJECTOID_OAM_FLUX_PATH=/path/to/oam_flux/src`.

```python
from flux_trajectoid import oam_flux_backend, is_live_oam_flux
print(oam_flux_backend(), is_live_oam_flux())
```

## License

MIT — see [LICENSE](LICENSE).
