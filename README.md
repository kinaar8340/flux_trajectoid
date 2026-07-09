# flux_trajectoid

**Photon Seed Asteroids** — 3D trajectoid shells + VQC quaternion/OAM + live `oam_flux` Hopf lattice for robust photonic data carriers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Inspired by **macadamia nuts** (hard shell, dense kernel) and **[trajectoids](https://en.wikipedia.org/wiki/Trajectoid)** (3D bodies whose rolling path encodes a prescribed curve), this project models a layered biomimetic photonic packet that propagates as a protected “seed” through simulated optical channels — and exports SLM-ready holograms.

---

## Why this works

| Idea | Role in flux_trajectoid |
|------|-------------------------|
| **Trajectoid scaling** | Anisotropic `(kx, ky)` + perimeter lock minimizes SO(3) rolling mismatch so the shell geometry is a reproducible ID fingerprint. |
| **3D shaved sphere** | Contact curve + oriented cutting planes turn the rolling plan into a real asteroid mesh with a potential trench. |
| **q × scale packing** | Unit quaternions alone lose bytes (S³ is 3-DOF). Scale rides on the OAM carrier amplitude → near-lossless photonic recovery on clean fields. |
| **OAM robustness** | Metrics show OAM spectral fidelity decays slower under turbulence than raw field overlap — angular momentum structure is hard to scramble. |
| **Hybrid recovery** | Digital CRC-perfect path + photonic BER scorecard so you always know channel quality. |

---

## Architecture (layered)

```
Photon Seed Asteroid (flux_trajectoid)
│
├── 1. DATA INGESTION
│   └── str | bytes + seed → deterministic identity
│
├── 2. OUTER SHELL (3D trajectoid)
│   ├── Planar path from payload hash
│   ├── Path scaling + TPT closure + SO(3) rolling
│   ├── 3D mesh: sphere + contact trench + oriented shaves
│   ├── Fourier fingerprint (silhouette) + curvature signal
│   └── shell/modulator → phase mask / potential trench
│
├── 3. INNER NUT (VQC + oam_flux)
│   ├── Quaternion shards (q) + scale (s) → LG OAM imprint
│   ├── Live TwistLattice + deposit_on_flywheels
│   └── Shell-attenuated multi-ℓ coupling
│
├── 4. TRANSMISSION
│   ├── Kolmogorov phase screens + tip/tilt + BMGL gate
│   ├── Lattice PDE evolution
│   └── FidelityMetrics scorecard + sweep_turbulence()
│
├── 5. RECOVERY
│   ├── Shell ID (Fourier cosine match)
│   ├── digital | photonic | hybrid (CRC-8)
│   └── Flywheel emergence probe
│
└── 6. HARDWARE HOOK
    └── export/slm → phase maps, GS option, device presets
```

See **[docs/architecture.md](docs/architecture.md)** for the full design write-up, **[docs/metrics.md](docs/metrics.md)** for fidelity interpretation, and **[docs/gallery.md](docs/gallery.md)** for usage vignettes.

---

## Install

```bash
git clone --recurse-submodules https://github.com/kinaar8340/flux_trajectoid.git
cd flux_trajectoid
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Submodules (already linked if you used `--recurse-submodules`):

| Path | Role |
|------|------|
| `external/oam_flux` | Live Hopf lattice + VQC flux deposition |
| `external/vqc_proto` | Quaternion / Orbital Braille reference |
| `external/vqc_sims_public` | Photonics simulation lineage |

```python
from flux_trajectoid import oam_flux_backend, is_live_oam_flux
print(oam_flux_backend(), is_live_oam_flux())
```

---

## End-to-end quick start

```python
from flux_trajectoid import PhotonSeedAsteroid

ast = PhotonSeedAsteroid("Hello from the shell", seed=42).build(
    lattice_nx=16,
    n_coupling_steps=12,
    use_tpt=True,
    build_3d=True,       # 3D shaved-sphere shell (default)
)

print(ast.summary())
# is_3d, mesh_vertices, mismatch_deg, flux_backend, ...

prop = ast.propagate(turbulence_level=0.35, n_steps=24)
print(prop.metrics.summary_line())
# F=… Icorr=… Strehl=… φrms=… OAM=… tt=…

rec = ast.recover(mode="hybrid")
print(rec.payload_text, rec.crc_ok, rec.byte_error_rate, rec.chordal_error_mean)

pkg = ast.export_slm("outputs/slm_package", preset="generic_256")
print(pkg.files)
```

### Recovery modes

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Lossless digital payload + photonic BER / chordal metrics |
| `digital` | ShardPack blocks only (CRC-8) |
| `photonic` | Field-only LG projection → `(q, s)` → bytes |

### Examples

```bash
python examples/create_seed.py       # shell + mesh + protected field plot
python examples/propagate_seed.py    # turbulence scorecard table
python examples/recover_seed.py      # digital / photonic / hybrid
python examples/export_slm.py        # SLM package on disk
```

---

## Metrics at a glance

After `propagate()`, inspect `prop.metrics`:

| Field | Meaning |
|-------|---------|
| `overlap_fidelity` | Primary field fidelity \|⟨ref\|obs⟩\|² |
| `oam_fidelity` | Cosine similarity of LG weight spectra |
| `phase_rmse_rad` | Intensity-weighted phase error |
| `strehl_proxy` | Peak intensity ratio |
| `power_retention` | Total power ratio |
| `tip_tilt_rms` | Residual tip/tilt energy |

OAM fidelity typically stays high while raw field F drops with turbulence — a signature of structured angular-momentum encoding.

```python
rows = ast.sweep_turbulence(levels=[0.0, 0.25, 0.5], n_steps=12)
```

Full guide: **[docs/metrics.md](docs/metrics.md)**.

---

## Project layout

```
flux_trajectoid/
├── configs/default.yaml
├── docs/
│   ├── architecture.md
│   ├── metrics.md
│   └── gallery.md
├── examples/
├── src/flux_trajectoid/
│   ├── photon_seed_asteroid.py   # orchestrator
│   ├── shell/                    # 3D trajectoid + modulator
│   ├── inner/                    # VQC + live oam_flux
│   ├── propagation/              # channel + metrics
│   ├── recovery/                 # hybrid decoder
│   ├── export/                   # SLM holograms
│   └── utils/
├── tests/
└── external/                     # git submodules
```

---

## Related projects

| Repo | Role |
|------|------|
| [vqc_proto](https://github.com/kinaar8340/vqc_proto) | Quaternion + Orbital Braille / OAM imprint |
| [oam_flux](https://github.com/kinaar8340/oam_flux) | Helical packets, Hopf lattice, flux flywheels |
| [vqc_sims_public](https://github.com/kinaar8340/vqc_sims_public) | Photonics simulation lineage |

---

## Local Space demo (Gradio)

Single-page visual app (HF Space: [kinaar111/flux_trajectoid](https://huggingface.co/spaces/kinaar111/flux_trajectoid)):

```bash
pip install -r space/requirements.txt
PYTHONPATH=src python space/app.py
# → http://127.0.0.1:7860
```

Top nav · multi-column viewports · glass layers at **opacity 0.3** · Nature-style trajectoid hero art in `space/assets/`.

## License

MIT — see [LICENSE](LICENSE).
