# Architecture: Photon Seed Asteroid

## Motivation

Optical free-space and fiber channels suffer from turbulence, mode crosstalk, and identification / routing ambiguity. **flux_trajectoid** packages data as a **Photon Seed Asteroid**:

1. A **hard outer shell** whose unique closed geometry acts as a fingerprint (trajectoid-inspired).
2. A **dense inner nut** encoded with VQC quaternion shards and OAM modes, coupled into oam_flux Hopf-lattice flywheels.
3. **Shell-modulated propagation** so the geometry constrains and protects the inner field (potential trench).

## Layer details

### 1. Data ingestion

`PhotonSeedAsteroid(payload, seed)` accepts `str` or `bytes`. The seed controls geometric reproducibility; the payload hash drives shell harmonics and packing.

### 2. Outer shell (`shell/`)

- **`generator.generate_shell`**: SHA-256 digest → harmonic planar curve → **path scaling (kx, ky)** minimizing SO(3) rolling mismatch → optional **Two-Period Trajectoid (TPT)** closure → cumulative rolling rotations → Fourier + arc-length + 1D phase/trench mask.
- **Rolling constraint**: pure-roll increments `angle = ds / r` about the in-plane left normal; mismatch is the magnitude of the total cumulative rotation (degrees).
- **Fourier descriptors**: complex DFT of centered boundary; magnitude fingerprint for matching.
- **Arc-length unroll**: uniform-s reparameterization + curvature signal as 1D metadata channel.
- **`modulator.shell_to_phase_mask`**: silhouette → amplitude envelope + boundary trench potential + azimuthal phase from curvature; per-ℓ phase bias from fingerprint.

### 3. Inner payload (`inner/`)

- **`vqc_encoder.encode_to_quaternion`**: bytes → unit-quaternion shards (from vqc_proto `quaternion_codec` pattern) → OAM weights on `{0, 1, −1, 2}` with Rodrigues-style `q.w` phase and imprint on spatial modes.
- **`oam_flux_coupling.couple_to_flux_lattice`**: prefers **live** `oam_flux` (submodule `external/oam_flux`):
  - `PhotonicsConfig` + `propagate_multi_ell_vectorized`
  - shared `TwistLattice` + per-ℓ `VQCCouplingState` / `run_vqc_coupling_step`
  - `deposit_on_flywheels` momentum ledger + shell mismatch attenuation
  - Falls back to a numpy Hopf-kick stub if the package is missing (`force_stub=True` or import failure).
- Backend probe: `oam_flux_backend()` / `is_live_oam_flux()`.

### 4. Modulation & protection

Applied at couple-time and re-applied each propagation step so the trench continues to gate energy near the shell boundary.

### 5. Transmission (`propagation/simulator.py`)

- Kolmogorov-like phase screens + tip/tilt jitter.
- Soft BMGL-style Fourier gate (VQC turbulence mitigation stub).
- Hopf-lattice PDE step with memory recovery toward the initial helical IC.

### 6. Recovery (`recovery/`)

- **Shell ID**: cosine similarity of Fourier fingerprints.
- **Decoder**: PCA/SVD demix stub → OAM projection → quaternion probe; stored shards yield byte recovery in the software stack; flywheel excess-twist → emergence score.

## Data flow

```
payload + seed
    → generate_shell → ShellGeometry (vertices, fingerprint, curvature)
    → encode_to_quaternion → QuaternionEncoding (shards, OAM weights, fields)
    → couple_to_flux_lattice → FluxState (packets, lattice θ, protected field)
    → propagate_asteroid → PropagationResult (field_final, fidelity)
    → recover_asteroid → RecoveryResult (payload_hat, shell_match, emergence)
```

## Design notes

- **Starters, not full physics**: LG modes, Kolmogorov screens, and lattice PDEs are intentionally simplified so the repo runs with `numpy`/`scipy` only. Deeper fidelity should pull from `oam_flux` and `vqc_proto` submodules.
- **Determinism**: same `(payload, seed)` → same shell fingerprint and packing.
- **Separation of concerns**: shell ID is independent of inner decode; modulation is the only tight coupling surface.

## Extension points

| Next step | Where |
|-----------|--------|
| Real trajectoid development / rolling constraint | `shell/generator.py` |
| Full ICA / Isomap blind recovery | `recovery/decoder.py` |
| Wire live `oam_flux.TwistLattice` | `inner/oam_flux_coupling.py` |
| Hardware SLM holograms | adapt `vqc_proto` typehead |
