# Why OAM fidelity holds under convex_defect screens

**Question.** Structured `convex_defect` / hybrid phase screens often **lower field overlap F** more than Kolmogorov at the same `turbulence_level`, while **OAM magnitude fidelity (OAMf)** stays high — sometimes *higher* than under Kolmogorov.

**Answer (short).** OAMf only compares **LG weight magnitudes** `|c_ℓ|`. Kolmogorov phase is **scale-free and mid-band correlated**, which efficiently **mixes OAM modes**. Convex screens, after RMS normalization to the same L, are often dominated by **fine residual texture** (misalignment noise whitened to target RMS): that kills field overlap F and power, but tends to **average out** in LG projections and **preserve the shape** of `|c_ℓ|` — so OAMf stays high.

---

## What the metrics actually measure

| Metric | Definition | Sensitive to |
|--------|------------|--------------|
| **F** (`overlap_fidelity`) | \|⟨ref\|obs⟩\|² / (‖ref‖²‖obs‖²) | Full complex field: any phase structure |
| **OAMf** (`oam_fidelity`) | Cosine similarity of **\|c_ℓ\|** vectors for ℓ∈{0,1,−1,2} | OAM *power* distribution only |
| **oam_phase_error** | Mean \|Δarg c_ℓ\| | Per-mode phase, not used in OAMf |
| **φrms** | Intensity-weighted phase RMSE after global align | Local phase errors |

Important: **OAMf deliberately ignores OAM coefficient phases.** A channel can rotate or scramble absolute phase, drop F to near zero, and still leave `|c_ℓ|` almost unchanged → OAMf ≈ 1.

That is by design for photonic data carriers: quaternion imprinting lives largely in the **OAM power + relative phases** of a small ℓ set; pure global phase is uninformative.

---

## Mechanism

### 1. Phase multiplication

Both screen models apply the same channel step:

\[
\psi \;\leftarrow\; \psi \cdot e^{i\phi(\mathbf{r})}
\]

(with tip/tilt, BMGL gate, shell trench, lattice relax — common path).

Differences come from the **statistics of \(\phi\)**.

### 2. Kolmogorov \(\phi\)

- Fourier synthesis with \(k^{-11/3}\)-like PSD → **power concentrated at low/mid k**.
- Phase is **spatially correlated** on scales that match LG beam structure.
- That correlated mid-band phase is efficient at **ℓ ↔ ℓ′ coupling**.

Empirically (ensemble @ L≈0.35): lower **LG diagonal retention** (~0.96) than convex, and OAMf falls sooner as L grows.

Effect: OAM *and* field are scrambled together — F drops, OAMf drops more than under pure convex.

### 3. Convex-defect \(\phi\)

Pipeline: local misalignment \(x_{ij}\) → ρ (or multi-scale Σ ρₖ wₖ) → **RMS-normalize to L**.

Two regimes appear in diagnostics:

1. **Structured regime** — real pointer/misalignment patches; ρ envelope is Gaussian-like / multi-scale.
2. **Whitened residual regime** — when \(x_{ij}\) is nearly uniform, residual grid noise is **amplified by RMS normalization** to carry almost all of the phase power. Cartesian PSD then looks **high-k heavy**, and \|∇φ\| is large.

Counter-intuitively, that can still **protect OAMf**:

- Rapid fine-scale phase tends to **average inside LG inner products** (amplitude damping / power loss) rather than systematically transfer power between helical modes.
- Field overlap F and power retention P collapse (global phase mess + BMGL).
- Relative **shape** of `|c_ℓ|` can remain close to the reference → high cosine OAMf even when `|c|` L1 is large (overall coefficient scale changes).

Empirically: **LG diagonal retention ~0.999**, **OAMf ≳ Kolmogorov**, **F ≤ Kolmogorov** at equal L.

### 4. Multi-scale convex

Evolving ρ(s) and integrating over bins adds multi-scale texture. Channel F can fall even faster (richer structure under RMS norm); OAMf typically remains high for the same reason as single-scale convex (shape of `|c_ℓ|` survives).

### 5. Hybrid

Mixes Kolmogorov mid-band coupling with convex residual texture. Often the **worst of both for OAMf** at high L (e.g. hybrid OAMf falling below either pure model) while F sits in between — a reminder that hybrid_weight is a real design knob, not free lunch.

---

## Diagnostic toolkit

```bash
PYTHONPATH=src:../convex_defect/src python examples/analyze_oam_screen_fidelity.py
```

Outputs under `outputs/oam_screen_analysis/`:

| Artifact | Content |
|----------|---------|
| `fidelity_F_vs_OAMf.png` | F, OAMf, OAMf−F vs L |
| `oam_magnitude_spectra.png` | \|c_ℓ\| bars at fixed L |
| `oam_leakage_matrices.png` | LG in→out power under e^{iφ} |
| `screen_structure_bars.png` | high-k, azimuthal roughness, LG diag |
| `SUMMARY.md` / JSON | Numeric tables |

API (importable):

```python
from flux_trajectoid.propagation.screen_diagnostics import (
    phase_screen_structure,
    oam_leakage_under_screen,
    sample_screen_ensemble,
    channel_metric_row,
)
```

---

## How to read a result table

At fixed L (e.g. 0.3), compare:

1. **OAMf − F** — larger ⇒ “OAM advantage” (structure survives better than full field).
2. **|c| L1** — total relative change of magnitude spectrum (should track OAMf inversely).
3. **high-k fraction** of φ — Kolmogorov high, convex lower.
4. **LG diagonal retention** — mean power staying on the input ℓ under pure e^{iφ} on LG probes.

Typical qualitative ordering at moderate L (from `analyze_oam_screen_fidelity.py`):

```text
F:         kolmogorov ≳ hybrid ~ convex > convex_ms     (convex hurts F more)
OAMf:      convex ≳ convex_ms ≳ kolmogorov ≫ hybrid     (hybrid can lose OAM)
OAMf − F:  largest for convex / convex_ms
LG_diag:   convex ≈ convex_ms > hybrid > kolmogorov
high-k*:   convex > hybrid > kolmogorov
           (*cartesian PSD after RMS norm — convex residual looks whitened)
```

Illustrative L=0.3 run (stub lattice, shared seed):

| model | F | OAMf | OAMf−F |
|-------|---|------|--------|
| kolmogorov | 0.228 | 0.979 | 0.75 |
| convex_defect | 0.193 | 0.995 | 0.80 |
| convex multi-scale | 0.101 | 0.980 | 0.88 |
| hybrid | 0.188 | 0.928 | 0.74 |

Absolute numbers depend on BMGL gate, shell trench, and field support; the **relative** pattern is the robust takeaway.

---

## Implications for Photon Seed Asteroids

1. **Hybrid recovery is the right story:** digital CRC path for payload truth; OAMf + BER for channel honesty.
2. **Structured turbulence is not “just worse Kolmogorov.”** It is a different error channel — often harsher on coherent F / power, softer on OAM **labels** (spectrum shape).
3. **Hardware campaigns** should log OAMf, `|c|` L1, and power alongside F; F alone overstates (or mis-types) damage from convex-like media.
4. **Mitigation differs by model:**
   - Kolmogorov → high-k / BMGL-style gates, tip-tilt removal.
   - Convex → **pointer realignment**, κ-tuning, reducing grid_noise / RMS-gain; pure Fourier gates may not target the right structure.
5. **RMS-normalization caveat:** equal L does *not* mean equal PSD shape. Convex residuals can be whitened to L and look high-k while still behaving differently for LG projections than Kolmogorov mid-band phase.

---

## Relation to convex_defect theory

- Misalignment-driven Gaussian + fractal \(s^{-\delta}\) → multi-scale **opacity** and **holonomy memory**.
- In the channel, that memory appears as **structured φ**, not white phase noise.
- Multi-scale mode uses **evolved ρ(s)** for holonomy in the defect simulator; in flux_trajectoid screens, multi-scale φ is the spatial integral of that dynamical texture.

See also: `docs/metrics.md`, `convex_defect` theory notes, `examples/compare_phase_screens.py`.
