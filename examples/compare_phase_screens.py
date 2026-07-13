#!/usr/bin/env python3
"""Compare Kolmogorov vs convex_defect vs hybrid phase screens under turbulence.

Requires optional convex_defect on PYTHONPATH (sibling repo):

    PYTHONPATH=src:../convex_defect/src python examples/compare_phase_screens.py

Writes a small table to stdout and optional plots under outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# sibling convex_defect checkout
sys.path.insert(0, str(ROOT.parent / "convex_defect" / "src"))

from flux_trajectoid import PhotonSeedAsteroid, convex_defect_available  # noqa: E402
from flux_trajectoid.propagation.phase_screens import make_phase_screen_engine  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _build():
    return PhotonSeedAsteroid(b"phase_screen_compare", seed=7).build(
        n_shards=4,
        lattice_nx=10,
        n_coupling_steps=2,
        force_stub_flux=True,
        n_points=80,
        scale_grid=3,
        scale_max_iter=2,
    )


def main() -> None:
    print(f"convex_defect available: {convex_defect_available()}")
    models = ["kolmogorov"]
    if convex_defect_available():
        models.extend(["convex_defect", "hybrid"])
    else:
        print("  (install/sibling convex_defect to enable structured screens)")

    levels = [0.0, 0.15, 0.3, 0.5]
    print(
        f"{'model':>14}  {'L':>5}  {'F':>7}  {'OAMf':>7}  {'φrms':>7}  {'P':>7}"
    )
    rows = []
    for model in models:
        for level in levels:
            a = _build()
            prop = a.propagate(
                turbulence_level=level,
                n_steps=10,
                seed=7,
                screen_model=model,
                hybrid_weight=0.55,
                convex_f=1.5,
                convex_s=0.5,
                apply_bmgl=True,
            )
            m = prop.metrics
            assert m is not None
            print(
                f"{model:>14}  {level:5.2f}  {m.overlap_fidelity:7.4f}  "
                f"{m.oam_fidelity:7.4f}  {m.phase_rmse_rad:7.3f}  "
                f"{m.power_retention:7.4f}"
            )
            rows.append((model, level, m, prop))

    # Visualize one mid-level screen per model
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        eng = make_phase_screen_engine(
            64,
            64,
            0.35,
            rng,
            screen_model=model,
            hybrid_weight=0.55,
            convex_f=1.5,
            convex_s=0.5,
        )
        # a few steps so convex grid has texture
        screen = eng.next_screen()
        for _ in range(4):
            screen = eng.next_screen()
        im = ax.imshow(screen, origin="lower", cmap="twilight")
        ax.set_title(model)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Phase screens (L=0.35, after warm-up steps)")
    path = OUT / "phase_screen_models.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")

    # Fidelity vs level curves
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model in models:
        xs = [r[1] for r in rows if r[0] == model]
        ys = [r[2].overlap_fidelity for r in rows if r[0] == model]
        oam = [r[2].oam_fidelity for r in rows if r[0] == model]
        ax.plot(xs, ys, "o-", label=f"{model} F")
        ax.plot(xs, oam, "s--", alpha=0.7, label=f"{model} OAM")
    ax.set_xlabel("turbulence_level")
    ax.set_ylabel("fidelity")
    ax.set_title("Kolmogorov vs convex_defect channel fidelity")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    path2 = OUT / "phase_screen_fidelity_compare.png"
    fig.savefig(path2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path2}")


if __name__ == "__main__":
    main()
