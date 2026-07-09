# Usage gallery

Short vignettes for common workflows. All assume:

```bash
pip install -e ".[dev]"
# optional: git submodule update --init --recursive
```

---

## 1. Build a 3D Photon Seed Asteroid

```python
from flux_trajectoid import PhotonSeedAsteroid

ast = PhotonSeedAsteroid("gallery seed", seed=42).build(
    use_tpt=True,
    build_3d=True,
    n_lat=40,
    n_lon=80,
    lattice_nx=12,
    n_coupling_steps=8,
)
s = ast.shell
print(s.is_3d, s.mesh_vertices.shape, s.volume_proxy, s.mismatch_deg)
```

**Expect:** `is_3d=True`, mesh with thousands of vertices, positive volume proxy, finite mismatch degrees.

**Plot:** `python examples/create_seed.py` → `outputs/create_seed.png`  
(3D surface + radial map + protected field).

---

## 2. Turbulence scorecard

```python
prop = ast.propagate(turbulence_level=0.3, n_steps=24, seed=42)
print(prop.metrics.summary_line())
print("trace F:", prop.fidelity_trace[0], "→", prop.fidelity_trace[-1])
```

**Table sweep:**

```bash
python examples/propagate_seed.py
```

---

## 3. Hybrid recovery after the channel

```python
rec = ast.recover(mode="hybrid")
print(rec.payload_text)          # digital path (CRC)
print(rec.crc_ok)
print("photonic BER", rec.byte_error_rate)
print("chordal", rec.chordal_error_mean)

# Channel-only stress:
rec_ph = ast.recover(mode="photonic")
print(rec_ph.payload_hat[:16], rec_ph.crc_ok)
```

---

## 4. Multi-level sweep with BER

```python
rows = ast.sweep_turbulence(levels=[0.0, 0.2, 0.4], n_steps=10)
for r in rows:
    print(
        f"L={r['turbulence_level']:.2f}  F={r['overlap_fidelity']:.3f}  "
        f"OAM={r['oam_fidelity']:.3f}  BER={r.get('photonic_byte_ber')}"
    )
```

---

## 5. SLM hologram package

```python
pkg = ast.export_slm(
    "outputs/slm_gallery",
    preset="generic_256",       # or holoeye_pluto_2, meadowlark_512, ...
    source="protected",
    stack_shards=True,
    include_shell_bias=True,
    use_gs=False,               # set True for Gerchberg–Saxton
)
print(pkg.out_dir)
print(pkg.files)
```

**Artifacts:** `manifest.json`, `phase_rad.npy`, `phase_levels.png`, `phase_stack.npy`, `preview_montage.png`.

```bash
python examples/export_slm.py
```

---

## 6. Live vs stub oam_flux

```python
from flux_trajectoid import oam_flux_backend, is_live_oam_flux

print(oam_flux_backend())
ast_live = PhotonSeedAsteroid(b"live", seed=1).build(force_stub_flux=False)
ast_stub = PhotonSeedAsteroid(b"stub", seed=1).build(force_stub_flux=True)
print(ast_live.flux_state.backend, ast_stub.flux_state.backend)
```

With submodules initialized, expect `live:.../external/oam_flux/src`.

---

## 7. Legacy 2D shell only

```python
from flux_trajectoid import generate_shell

s = generate_shell("flat", seed=0, build_3d=False)
assert not s.is_3d
```

---

## 8. Redundant shards

```python
ast = PhotonSeedAsteroid(b"ABCD", seed=0).build(
    n_shards=6,
    redundancy=3,   # each 4-byte block thrice
    force_stub_flux=True,
)
rec = ast.recover(mode="digital")
assert rec.payload_text is not None or rec.payload_hat[:4] == b"ABCD"
```
