#!/usr/bin/env python3
"""Sweep multi-scale screen knobs: n_scales × scale_coupling → F vs OAMf.

Uses convex_defect multi-scale phase screens in the Photon Seed channel.

    PYTHONPATH=src:../convex_defect/src python examples/sweep_multiscale_fidelity.py

Writes tables, heatmaps, and a short summary under
``outputs/multiscale_param_sweep/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "convex_defect" / "src"))

from flux_trajectoid import PhotonSeedAsteroid, convex_defect_available  # noqa: E402
from flux_trajectoid.propagation.screen_diagnostics import (  # noqa: E402
    channel_metric_row,
    sample_screen_ensemble,
)

OUT = ROOT / "outputs" / "multiscale_param_sweep"

# Default grids (overridable via CLI)
DEFAULT_N_SCALES = [4, 8, 12, 16, 24]
DEFAULT_COUPLING = [0.0, 0.05, 0.1, 0.2, 0.35]
DEFAULT_LEVELS = [0.15, 0.3, 0.5]


def _factory(seed: int):
    return PhotonSeedAsteroid(b"ms_param_sweep", seed=seed).build(
        n_shards=4,
        lattice_nx=10,
        n_coupling_steps=2,
        force_stub_flux=True,
        n_points=80,
        scale_grid=3,
        scale_max_iter=2,
    )


def run_grid(
    *,
    n_scales_list: list[int],
    coupling_list: list[float],
    levels: list[float],
    n_steps: int,
    seed: int,
    convex_f: float,
    convex_s: float,
    include_structure: bool,
) -> list[dict]:
    rows: list[dict] = []
    print(
        f"{'ns':>4}  {'c':>5}  {'L':>5}  {'F':>7}  {'OAMf':>7}  "
        f"{'Δ':>7}  {'|c|L1':>7}  {'P':>7}"
    )
    for ns in n_scales_list:
        for c in coupling_list:
            struct = None
            if include_structure:
                # structure at mid level for this (ns, c)
                struct = sample_screen_ensemble(
                    "convex_defect",
                    0.35,
                    size=48,
                    n_samples=4,
                    seed=seed,
                    multi_scale=True,
                    n_scales=ns,
                    scale_coupling=c,
                    convex_f=convex_f,
                    convex_s=convex_s,
                )
            for level in levels:
                row = channel_metric_row(
                    lambda s=seed: _factory(s),
                    level=level,
                    model="convex_defect",
                    n_steps=n_steps,
                    seed=seed,
                    multi_scale=True,
                    n_scales=ns,
                    scale_coupling=c,
                    convex_f=convex_f,
                    convex_s=convex_s,
                )
                row["n_scales"] = ns
                row["scale_coupling"] = c
                if struct is not None:
                    row["high_k"] = struct["high_k_fraction_mean"]
                    row["lg_diag"] = struct["oam_diag_retention_mean"]
                    row["grad_rms"] = struct["gradient_rms_mean"]
                rows.append(row)
                print(
                    f"{ns:4d}  {c:5.2f}  {level:5.2f}  {row['F']:7.4f}  "
                    f"{row['OAMf']:7.4f}  {row['OAMf_minus_F']:7.4f}  "
                    f"{row['mag_l1']:7.4f}  {row['P']:7.4f}"
                )
    return rows


def _pivot(rows: list[dict], level: float, key: str) -> tuple[np.ndarray, list[int], list[float]]:
    n_scales = sorted({r["n_scales"] for r in rows})
    couplings = sorted({r["scale_coupling"] for r in rows})
    mat = np.full((len(couplings), len(n_scales)), np.nan)
    for r in rows:
        if abs(r["level"] - level) > 1e-12:
            continue
        i = couplings.index(r["scale_coupling"])
        j = n_scales.index(r["n_scales"])
        mat[i, j] = r[key]
    return mat, n_scales, couplings


def plot_heatmaps(rows: list[dict], levels: list[float]) -> None:
    metrics = [
        ("F", "Field overlap F"),
        ("OAMf", "OAM magnitude fidelity"),
        ("OAMf_minus_F", "OAMf − F (OAM advantage)"),
    ]
    for level in levels:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for ax, (key, title) in zip(axes, metrics):
            mat, n_scales, couplings = _pivot(rows, level, key)
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks(range(len(n_scales)))
            ax.set_xticklabels([str(n) for n in n_scales])
            ax.set_yticks(range(len(couplings)))
            ax.set_yticklabels([f"{c:.2f}" for c in couplings])
            ax.set_xlabel("n_scales")
            ax.set_ylabel("scale_coupling")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
            # annotate
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="w",
                            fontsize=7,
                        )
        fig.suptitle(f"Multi-scale param grid @ L={level}")
        path = OUT / f"heatmap_L{str(level).replace('.', 'p')}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def plot_pareto(rows: list[dict], levels: list[float]) -> None:
    """Scatter F vs OAMf colored by n_scales, marker size by coupling."""
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.0), squeeze=False)
    for ax, level in zip(axes[0], levels):
        sub = [r for r in rows if abs(r["level"] - level) < 1e-12]
        ns = np.array([r["n_scales"] for r in sub], dtype=float)
        cc = np.array([r["scale_coupling"] for r in sub], dtype=float)
        F = np.array([r["F"] for r in sub])
        O = np.array([r["OAMf"] for r in sub])
        sc = ax.scatter(
            F,
            O,
            c=ns,
            s=40 + 400 * cc,
            cmap="viridis",
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
        for r in sub:
            ax.annotate(
                f"n={r['n_scales']}\nc={r['scale_coupling']:.2f}",
                (r["F"], r["OAMf"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                alpha=0.7,
            )
        ax.set_xlabel("F")
        ax.set_ylabel("OAMf")
        ax.set_title(f"L={level}  (size∝coupling)")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="n_scales")
    fig.suptitle("F–OAMf tradeoff under multi-scale screens")
    path = OUT / "pareto_F_vs_OAMf.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_slices(rows: list[dict], level: float) -> None:
    """1D slices: OAMf & F vs n_scales at fixed coupling; vs coupling at fixed n."""
    couplings = sorted({r["scale_coupling"] for r in rows})
    n_scales = sorted({r["n_scales"] for r in rows})
    sub = [r for r in rows if abs(r["level"] - level) < 1e-12]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True)

    # vs n_scales for each coupling
    for c in couplings:
        xs, f, o = [], [], []
        for ns in n_scales:
            match = [
                r
                for r in sub
                if r["n_scales"] == ns and abs(r["scale_coupling"] - c) < 1e-12
            ]
            if match:
                xs.append(ns)
                f.append(match[0]["F"])
                o.append(match[0]["OAMf"])
        axes[0].plot(xs, f, "o--", alpha=0.7, label=f"F c={c:.2f}")
        axes[0].plot(xs, o, "s-", label=f"OAMf c={c:.2f}")
    axes[0].set_xlabel("n_scales")
    axes[0].set_ylabel("fidelity")
    axes[0].set_title(f"vs n_scales @ L={level}")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # vs coupling for each n_scales
    for ns in n_scales:
        xs, f, o = [], [], []
        for c in couplings:
            match = [
                r
                for r in sub
                if r["n_scales"] == ns and abs(r["scale_coupling"] - c) < 1e-12
            ]
            if match:
                xs.append(c)
                f.append(match[0]["F"])
                o.append(match[0]["OAMf"])
        axes[1].plot(xs, f, "o--", alpha=0.7, label=f"F n={ns}")
        axes[1].plot(xs, o, "s-", label=f"OAMf n={ns}")
    axes[1].set_xlabel("scale_coupling")
    axes[1].set_ylabel("fidelity")
    axes[1].set_title(f"vs scale_coupling @ L={level}")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    path = OUT / f"slices_L{str(level).replace('.', 'p')}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def write_summary(rows: list[dict], levels: list[float]) -> None:
    lines = [
        "# Multi-scale parameter sweep — F vs OAMf",
        "",
        "Screen: `convex_defect` with `multi_scale=True`.",
        "",
        "Knobs: **n_scales** (log bins over s) and **scale_coupling** "
        "(neighbor mix on the scale axis each channel step).",
        "",
    ]
    for level in levels:
        sub = [r for r in rows if abs(r["level"] - level) < 1e-12]
        if not sub:
            continue
        best_f = max(sub, key=lambda r: r["F"])
        best_o = max(sub, key=lambda r: r["OAMf"])
        best_adv = max(sub, key=lambda r: r["OAMf_minus_F"])
        # compromise: maximize OAMf subject to F not in bottom quartile
        f_cut = float(np.percentile([r["F"] for r in sub], 50))
        pool = [r for r in sub if r["F"] >= f_cut] or sub
        best_bal = max(pool, key=lambda r: r["OAMf"])

        lines.extend(
            [
                f"## L = {level}",
                "",
                f"| optimum | n_scales | coupling | F | OAMf | OAMf−F |",
                f"|---------|----------|----------|---|------|--------|",
                f"| max F | {best_f['n_scales']} | {best_f['scale_coupling']:.2f} | "
                f"{best_f['F']:.4f} | {best_f['OAMf']:.4f} | {best_f['OAMf_minus_F']:.4f} |",
                f"| max OAMf | {best_o['n_scales']} | {best_o['scale_coupling']:.2f} | "
                f"{best_o['F']:.4f} | {best_o['OAMf']:.4f} | {best_o['OAMf_minus_F']:.4f} |",
                f"| max OAMf−F | {best_adv['n_scales']} | {best_adv['scale_coupling']:.2f} | "
                f"{best_adv['F']:.4f} | {best_adv['OAMf']:.4f} | {best_adv['OAMf_minus_F']:.4f} |",
                f"| balanced (F≥median) | {best_bal['n_scales']} | {best_bal['scale_coupling']:.2f} | "
                f"{best_bal['F']:.4f} | {best_bal['OAMf']:.4f} | {best_bal['OAMf_minus_F']:.4f} |",
                "",
            ]
        )

        # simple trend notes
        ns_vals = sorted({r["n_scales"] for r in sub})
        mean_f_by_ns = {
            ns: float(np.mean([r["F"] for r in sub if r["n_scales"] == ns]))
            for ns in ns_vals
        }
        mean_o_by_ns = {
            ns: float(np.mean([r["OAMf"] for r in sub if r["n_scales"] == ns]))
            for ns in ns_vals
        }
        lines.append(
            f"- Mean F by n_scales: "
            + ", ".join(f"{ns}→{mean_f_by_ns[ns]:.3f}" for ns in ns_vals)
        )
        lines.append(
            f"- Mean OAMf by n_scales: "
            + ", ".join(f"{ns}→{mean_o_by_ns[ns]:.3f}" for ns in ns_vals)
        )
        c_vals = sorted({r["scale_coupling"] for r in sub})
        mean_f_by_c = {
            c: float(np.mean([r["F"] for r in sub if abs(r["scale_coupling"] - c) < 1e-12]))
            for c in c_vals
        }
        mean_o_by_c = {
            c: float(np.mean([r["OAMf"] for r in sub if abs(r["scale_coupling"] - c) < 1e-12]))
            for c in c_vals
        }
        lines.append(
            f"- Mean F by coupling: "
            + ", ".join(f"{c:.2f}→{mean_f_by_c[c]:.3f}" for c in c_vals)
        )
        lines.append(
            f"- Mean OAMf by coupling: "
            + ", ".join(f"{c:.2f}→{mean_o_by_c[c]:.3f}" for c in c_vals)
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- **n_scales**: log bins over s; with per-scale spatial textures "
            "(scale_texture_amp), more bins change the integrated screen PSD.",
            "- **scale_coupling**: mixes neighboring scale bins each step — "
            "blends fine and coarse textures; large coupling homogenizes scales.",
            "- Non-separable rho(x,s) is required so knobs survive RMS normalization.",
            "- Prefer reading **OAMf - F** as the OAM advantage, not F alone.",
            "",
            "See also: `docs/oam_screen_fidelity.md`, "
            "`examples/analyze_oam_screen_fidelity.py`.",
            "",
        ]
    )
    path = OUT / "SUMMARY.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def _serializable(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = {
            k: v
            for k, v in r.items()
            if k not in ("mag_ref", "mag_obs", "ells")
        }
        if "mag_ref" in r:
            d["mag_ref"] = [float(x) for x in r["mag_ref"]]
            d["mag_obs"] = [float(x) for x in r["mag_obs"]]
            d["ells"] = list(r["ells"])
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--f", type=float, default=1.5, dest="convex_f")
    p.add_argument("--s", type=float, default=0.5, dest="convex_s")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid for a fast smoke run",
    )
    p.add_argument(
        "--no-structure",
        action="store_true",
        help="Skip ensemble structure diagnostics (faster)",
    )
    args = p.parse_args(argv)

    if not convex_defect_available():
        print(
            "convex_defect not available. "
            "Use: PYTHONPATH=src:../convex_defect/src ...",
            file=sys.stderr,
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    if args.quick:
        n_scales_list = [4, 8, 16]
        coupling_list = [0.0, 0.1, 0.3]
        levels = [0.3]
    else:
        n_scales_list = list(DEFAULT_N_SCALES)
        coupling_list = list(DEFAULT_COUPLING)
        levels = list(DEFAULT_LEVELS)

    print(f"output → {OUT}")
    print(
        f"grid: n_scales={n_scales_list}  coupling={coupling_list}  "
        f"L={levels}  steps={args.steps}"
    )

    rows = run_grid(
        n_scales_list=n_scales_list,
        coupling_list=coupling_list,
        levels=levels,
        n_steps=args.steps,
        seed=args.seed,
        convex_f=args.convex_f,
        convex_s=args.convex_s,
        include_structure=not args.no_structure,
    )

    plot_heatmaps(rows, levels)
    plot_pareto(rows, levels)
    for level in levels:
        plot_slices(rows, level)
    write_summary(rows, levels)

    (OUT / "sweep_results.json").write_text(
        json.dumps(
            {
                "n_scales": n_scales_list,
                "scale_coupling": coupling_list,
                "levels": levels,
                "n_steps": args.steps,
                "seed": args.seed,
                "convex_f": args.convex_f,
                "convex_s": args.convex_s,
                "rows": _serializable(rows),
            },
            indent=2,
        )
    )
    print(f"wrote {OUT / 'sweep_results.json'}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
