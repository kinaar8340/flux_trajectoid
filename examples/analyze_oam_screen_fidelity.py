#!/usr/bin/env python3
"""Why OAM fidelity often holds under convex_defect screens while F drops.

Compares Kolmogorov / convex_defect / hybrid (/ multi_scale) on:

1. Channel scorecard (F, OAMf, φrms, tip-tilt, |c_ℓ| L1 drift)
2. Phase-screen structure (high-k power, azimuthal roughness, ∇φ)
3. LG leakage matrices (how pure modes scramble under e^{iφ})

    PYTHONPATH=src:../convex_defect/src python examples/analyze_oam_screen_fidelity.py

Writes plots + JSON under outputs/oam_screen_analysis/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "convex_defect" / "src"))

from flux_trajectoid import PhotonSeedAsteroid, convex_defect_available  # noqa: E402
from flux_trajectoid.propagation.phase_screens import make_phase_screen_engine  # noqa: E402
from flux_trajectoid.propagation.screen_diagnostics import (  # noqa: E402
    channel_metric_row,
    explain_oam_vs_f,
    oam_leakage_under_screen,
    phase_screen_structure,
    sample_screen_ensemble,
)

OUT = ROOT / "outputs" / "oam_screen_analysis"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [0.0, 0.15, 0.3, 0.5]
N_STEPS = 10
SEED = 7


def _factory():
    return PhotonSeedAsteroid(b"oam_screen_analysis", seed=SEED).build(
        n_shards=4,
        lattice_nx=10,
        n_coupling_steps=2,
        force_stub_flux=True,
        n_points=80,
        scale_grid=3,
        scale_max_iter=2,
    )


def _models() -> list[tuple[str, dict]]:
    models: list[tuple[str, dict]] = [("kolmogorov", {})]
    if convex_defect_available():
        models.append(("convex_defect", {"multi_scale": False}))
        models.append(("convex_defect", {"multi_scale": True, "n_scales": 8}))
        models.append(("hybrid", {"hybrid_weight": 0.55, "multi_scale": False}))
    return models


def _label(model: str, kw: dict) -> str:
    if model == "convex_defect" and kw.get("multi_scale"):
        return "convex_ms"
    if model == "hybrid":
        return "hybrid"
    return model


def run_channel_sweep() -> list[dict]:
    rows = []
    models = _models()
    print("=== Channel metrics (build → propagate) ===")
    print(
        f"{'label':>12}  {'L':>5}  {'F':>7}  {'OAMf':>7}  {'Δ':>7}  "
        f"{'|c|L1':>7}  {'φrms':>7}  {'tt':>7}"
    )
    for model, kw in models:
        lab = _label(model, kw)
        for level in LEVELS:
            row = channel_metric_row(
                _factory,
                level=level,
                model=model,
                n_steps=N_STEPS,
                seed=SEED,
                convex_f=1.5,
                convex_s=0.5,
                **kw,
            )
            row["label"] = lab
            rows.append(row)
            print(
                f"{lab:>12}  {level:5.2f}  {row['F']:7.4f}  {row['OAMf']:7.4f}  "
                f"{row['OAMf_minus_F']:7.4f}  {row['mag_l1']:7.4f}  "
                f"{row['phi_rms']:7.3f}  {row['tt']:7.3f}"
            )
    return rows


def run_structure_and_leakage() -> list[dict]:
    print("\n=== Screen structure + LG diagonal retention (L=0.35) ===")
    stats = []
    level = 0.35
    for model, kw in _models():
        lab = _label(model, kw)
        ens = sample_screen_ensemble(
            model,
            level,
            size=64,
            n_samples=6,
            seed=0,
            convex_f=1.5,
            convex_s=0.5,
            **kw,
        )
        ens["label"] = lab
        stats.append(ens)
        print(
            f"{lab:>12}  high_k={ens['high_k_fraction_mean']:.3f}±{ens['high_k_fraction_std']:.3f}  "
            f"az={ens['azimuthal_roughness_mean']:.3f}  "
            f"∇φ={ens['gradient_rms_mean']:.3f}  "
            f"LG_diag={ens['oam_diag_retention_mean']:.3f}±{ens['oam_diag_retention_std']:.3f}"
        )
    return stats


def plot_fidelity_curves(rows: list[dict]) -> None:
    labels = sorted({r["label"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    for lab in labels:
        sub = [r for r in rows if r["label"] == lab]
        xs = [r["level"] for r in sub]
        axes[0].plot(xs, [r["F"] for r in sub], "o-", label=lab)
        axes[1].plot(xs, [r["OAMf"] for r in sub], "s-", label=lab)
        axes[2].plot(xs, [r["OAMf_minus_F"] for r in sub], "^-", label=lab)
    axes[0].set_title("Field overlap F")
    axes[1].set_title("OAM magnitude fidelity")
    axes[2].set_title("OAMf − F (OAM advantage)")
    for ax in axes:
        ax.set_xlabel("turbulence_level L")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Structured screens: F drops faster than OAMf")
    path = OUT / "fidelity_F_vs_OAMf.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_oam_spectra(rows: list[dict], level: float = 0.3) -> None:
    sub = [r for r in rows if abs(r["level"] - level) < 1e-9]
    if not sub:
        return
    ells = list(sub[0]["ells"])
    x = np.arange(len(ells))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / max(len(sub), 1)
    # reference from first row
    ax.bar(x - 0.4, sub[0]["mag_ref"], width=width, label="ref", alpha=0.35, color="k")
    for i, r in enumerate(sub):
        ax.bar(
            x - 0.4 + (i + 1) * width,
            r["mag_obs"],
            width=width,
            label=f"{r['label']} (OAMf={r['OAMf']:.3f})",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in ells])
    ax.set_xlabel("ℓ")
    ax.set_ylabel("|c_ℓ|")
    ax.set_title(f"OAM magnitude spectra at L={level}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    path = OUT / "oam_magnitude_spectra.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_leakage_matrices() -> None:
    if not convex_defect_available():
        return
    models = [
        ("kolmogorov", {}),
        ("convex_defect", {"multi_scale": False}),
        ("convex_defect", {"multi_scale": True, "n_scales": 8}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    level = 0.35
    for ax, (model, kw) in zip(axes, models):
        rng = np.random.default_rng(1)
        eng = make_phase_screen_engine(
            64, 64, level, rng, screen_model=model, convex_f=1.5, convex_s=0.5, **kw
        )
        screen = eng.next_screen()
        for _ in range(4):
            screen = eng.next_screen()
        leak = oam_leakage_under_screen(screen)
        im = ax.imshow(leak.leakage_matrix, origin="upper", cmap="viridis", vmin=0, vmax=1)
        lab = _label(model, kw)
        ax.set_title(f"{lab}\ndiag={leak.diagonal_retention:.3f}")
        ax.set_xticks(range(len(leak.ells)))
        ax.set_yticks(range(len(leak.ells)))
        ax.set_xticklabels([str(e) for e in leak.ells])
        ax.set_yticklabels([str(e) for e in leak.ells])
        ax.set_xlabel("out ℓ")
        ax.set_ylabel("in ℓ")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("LG power leakage under e^{iφ} (L=0.35)")
    path = OUT / "oam_leakage_matrices.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_structure_bars(stats: list[dict]) -> None:
    labels = [s["label"] for s in stats]
    metrics = [
        ("high_k_fraction_mean", "high-k power fraction"),
        ("azimuthal_roughness_mean", "azimuthal roughness m≥2"),
        ("oam_diag_retention_mean", "LG diagonal retention"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, (key, title) in zip(axes, metrics):
        vals = [s[key] for s in stats]
        ax.bar(x, vals, color=["C0", "C1", "C2", "C3"][: len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Screen structure → OAM scrambling (ensemble @ L=0.35)")
    path = OUT / "screen_structure_bars.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def write_summary(rows: list[dict], stats: list[dict]) -> None:
    # pair structure with mid-level channel rows
    struct_by = {s["label"]: s for s in stats}
    mid = [r for r in rows if abs(r["level"] - 0.3) < 1e-9]
    lines = [
        "# OAM vs field fidelity under phase screens — auto summary",
        "",
        "## Mechanism (short)",
        "",
        "1. **OAMf uses |c_ℓ| only** (cosine of magnitude spectra). Global phase "
        "collapse hurts F hard but need not reshape |c_ℓ|.",
        "2. **Kolmogorov** mid-band correlated phase couples LG modes "
        "(lower LG diagonal retention; OAMf falls with L).",
        "3. **convex_defect** ρ→φ is RMS-normalized; residual grid texture is often "
        "whitened to target L (high cartesian high-k fraction). That damps F/power "
        "but averages in LG projections, so |c_ℓ| *shape* (OAMf) stays high.",
        "4. **hybrid** can combine mid-band mixing with residual noise — watch OAMf "
        "at high L (sometimes worse than either pure model).",
        "",
        "## At L=0.3",
        "",
        "| model | F | OAMf | OAMf−F | |c| L1 | high-k | LG diag |",
        "|-------|---|------|--------|-------|--------|---------|",
    ]
    for r in mid:
        st = struct_by.get(r["label"], {})
        lines.append(
            f"| {r['label']} | {r['F']:.3f} | {r['OAMf']:.3f} | "
            f"{r['OAMf_minus_F']:.3f} | {r['mag_l1']:.3f} | "
            f"{st.get('high_k_fraction_mean', float('nan')):.3f} | "
            f"{st.get('oam_diag_retention_mean', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Narratives",
            "",
        ]
    )
    for r in mid:
        st = struct_by.get(r["label"])
        lines.append(f"- {explain_oam_vs_f(r, st)}")
    lines.append("")
    path = OUT / "SUMMARY.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")

    # JSON dump (convert arrays)
    serializable = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in ("mag_ref", "mag_obs", "ells")}
        d["mag_ref"] = list(map(float, r["mag_ref"]))
        d["mag_obs"] = list(map(float, r["mag_obs"]))
        d["ells"] = list(r["ells"])
        serializable.append(d)
    (OUT / "channel_metrics.json").write_text(json.dumps(serializable, indent=2))
    (OUT / "screen_structure.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT / 'channel_metrics.json'}")


def main() -> None:
    print(f"convex_defect available: {convex_defect_available()}")
    print(f"output → {OUT}")
    rows = run_channel_sweep()
    stats = run_structure_and_leakage()
    plot_fidelity_curves(rows)
    plot_oam_spectra(rows, level=0.3)
    plot_leakage_matrices()
    plot_structure_bars(stats)
    write_summary(rows, stats)
    print("\ndone.")


if __name__ == "__main__":
    main()
