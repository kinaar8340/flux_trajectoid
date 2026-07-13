"""Diagnostics for phase-screen structure vs OAM / field fidelity.

Helps explain why structured convex_defect screens often degrade field
overlap F more than Kolmogorov while preserving (or improving) OAM
magnitude fidelity:

1. **OAMf is magnitude-only** — cosine similarity of |c_ℓ|, not field phase.
2. **Kolmogorov is mid-band correlated** — scale-free phase couples LG modes
   (lower LG diagonal retention).
3. **Convex ρ→φ after RMS norm** often amplifies fine residual texture
   (high cartesian high-k fraction) that damps field overlap / power but
   averages inside LG projections, leaving |c_ℓ| *shape* intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..inner.vqc_encoder import OAM_QUAT_ELLS
from ..recovery.decoder import project_oam_spectrum
from .metrics import field_overlap_fidelity, oam_spectrum_metrics
from .phase_screens import ScreenModel, make_phase_screen_engine


@dataclass
class ScreenStructureStats:
    """Structural diagnostics of a single phase screen (radians)."""

    rms: float
    high_k_fraction: float
    """Power in k > k_nyquist/4 relative to total PSD power (excl. DC)."""
    azimuthal_roughness: float
    """RMS of azimuthal Fourier coeffs m≥2 on a mid-radius ring (normalized)."""
    radial_variance: float
    """Variance of radial mean phase profile (structure along ρ)."""
    gradient_rms: float
    """RMS of |∇φ| (high → strong mode mixing)."""

    def as_dict(self) -> dict[str, float]:
        return {
            "rms": self.rms,
            "high_k_fraction": self.high_k_fraction,
            "azimuthal_roughness": self.azimuthal_roughness,
            "radial_variance": self.radial_variance,
            "gradient_rms": self.gradient_rms,
        }


@dataclass
class OAMLeakageReport:
    """Approximate OAM power redistribution under a phase screen."""

    ells: tuple[int, ...]
    leakage_matrix: NDArray[np.floating]
    """L[i,j] ≈ power remaining/leaked from mode ells[i] into ells[j]."""
    diagonal_retention: float
    """Mean of diag(L) — higher means less OAM scrambling."""
    offdiag_mass: float
    """Mean off-diagonal mass per input mode."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ells": list(self.ells),
            "leakage_matrix": self.leakage_matrix.tolist(),
            "diagonal_retention": self.diagonal_retention,
            "offdiag_mass": self.offdiag_mass,
        }


def phase_screen_structure(screen: NDArray[np.floating]) -> ScreenStructureStats:
    """Characterize spatial roughness of a zero-mean phase map."""
    phi = np.asarray(screen, dtype=float)
    phi = phi - float(phi.mean())
    rms = float(phi.std())

    # Fourier high-k fraction
    F = np.fft.fftshift(np.fft.fft2(phi))
    psd = np.abs(F) ** 2
    h, w = phi.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    kr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    kmax = min(cy, cx)
    total = float(psd.sum() - psd[cy, cx]) + 1e-30
    high = float(psd[kr > 0.25 * kmax].sum())
    high_k_fraction = high / total

    # Azimuthal roughness on mid-radius ring
    rho = kr
    r0 = 0.35 * kmax
    ring = (rho > r0 - 1.5) & (rho < r0 + 1.5)
    # sample angles via phase values on ring in polar reorder
    ys, xs = np.where(ring)
    if len(xs) < 16:
        az_rough = 0.0
    else:
        ang = np.arctan2(ys - cy, xs - cx)
        order = np.argsort(ang)
        ring_vals = phi[ys[order], xs[order]]
        # interpolate to uniform angle grid
        th = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        ring_u = np.interp(th, ang[order], ring_vals, period=2 * np.pi)
        spec = np.fft.rfft(ring_u - ring_u.mean())
        # m≥2 energy / total
        power = np.abs(spec) ** 2
        az_rough = float(power[2:].sum() / (power.sum() + 1e-30))

    # Radial profile variance
    r_bins = np.linspace(0, kmax, 24)
    radial_means = []
    for i in range(len(r_bins) - 1):
        mask = (rho >= r_bins[i]) & (rho < r_bins[i + 1])
        if np.any(mask):
            radial_means.append(float(phi[mask].mean()))
    radial_variance = float(np.var(radial_means)) if radial_means else 0.0

    gy, gx = np.gradient(phi)
    gradient_rms = float(np.sqrt(np.mean(gx**2 + gy**2)))

    return ScreenStructureStats(
        rms=rms,
        high_k_fraction=high_k_fraction,
        azimuthal_roughness=az_rough,
        radial_variance=radial_variance,
        gradient_rms=gradient_rms,
    )


def oam_leakage_under_screen(
    screen: NDArray[np.floating],
    *,
    ells: tuple[int, ...] = OAM_QUAT_ELLS,
    size: int | None = None,
) -> OAMLeakageReport:
    """Apply e^{iφ} to pure LG modes; measure power leakage among ells.

    For each input ℓ, project the phase-modulated LG_ℓ onto the LG basis and
    record |c_j|² / Σ|c|². Diagonal retention high ⇒ OAM labels survive.
    """
    from ..inner.vqc_encoder import _lg_mode
    from ..recovery.decoder import _field_grids

    phi = np.asarray(screen, dtype=float)
    if size is not None and phi.shape[0] != size:
        # simple center crop/pad
        out = np.zeros((size, size), dtype=float)
        a = min(size, phi.shape[0])
        s0 = (phi.shape[0] - a) // 2
        d0 = (size - a) // 2
        out[d0 : d0 + a, d0 : d0 + a] = phi[s0 : s0 + a, s0 : s0 + a]
        phi = out
    h, w = phi.shape
    rho, ang = _field_grids((h, w))
    n = len(ells)
    L = np.zeros((n, n), dtype=float)
    for i, ell_in in enumerate(ells):
        mode = _lg_mode(int(ell_in), rho, ang, w0=0.6)
        field = mode * np.exp(1j * phi)
        weights = project_oam_spectrum(field, ells=ells)
        mags2 = np.array([abs(weights[e]) ** 2 for e in ells], dtype=float)
        mags2 = mags2 / (mags2.sum() + 1e-30)
        L[i, :] = mags2
    diag = float(np.mean(np.diag(L)))
    off = float(np.mean(L - np.diag(np.diag(L))))
    return OAMLeakageReport(
        ells=ells,
        leakage_matrix=L,
        diagonal_retention=diag,
        offdiag_mass=off,
    )


def sample_screen_ensemble(
    model: ScreenModel | str,
    level: float,
    *,
    size: int = 64,
    n_samples: int = 8,
    seed: int = 0,
    multi_scale: bool = False,
    n_scales: int = 8,
    **screen_kwargs: Any,
) -> dict[str, Any]:
    """Draw an ensemble of screens; return mean structure + leakage stats."""
    structs: list[ScreenStructureStats] = []
    diags: list[float] = []
    highs: list[float] = []
    azs: list[float] = []
    grads: list[float] = []
    for i in range(n_samples):
        rng = np.random.default_rng(seed + 17 * i)
        eng = make_phase_screen_engine(
            size,
            size,
            level,
            rng,
            screen_model=model,  # type: ignore[arg-type]
            multi_scale=multi_scale,
            n_scales=n_scales,
            **screen_kwargs,
        )
        # warm-up so multi-scale / convex grid has texture
        screen = eng.next_screen()
        for _ in range(3):
            screen = eng.next_screen()
        st = phase_screen_structure(screen)
        structs.append(st)
        highs.append(st.high_k_fraction)
        azs.append(st.azimuthal_roughness)
        grads.append(st.gradient_rms)
        leak = oam_leakage_under_screen(screen, size=size)
        diags.append(leak.diagonal_retention)

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        a = np.asarray(xs, dtype=float)
        return float(a.mean()), float(a.std())

    hk_m, hk_s = _mean_std(highs)
    az_m, az_s = _mean_std(azs)
    gr_m, gr_s = _mean_std(grads)
    dg_m, dg_s = _mean_std(diags)
    return {
        "model": model,
        "level": level,
        "multi_scale": multi_scale,
        "high_k_fraction_mean": hk_m,
        "high_k_fraction_std": hk_s,
        "azimuthal_roughness_mean": az_m,
        "azimuthal_roughness_std": az_s,
        "gradient_rms_mean": gr_m,
        "gradient_rms_std": gr_s,
        "oam_diag_retention_mean": dg_m,
        "oam_diag_retention_std": dg_s,
        "n_samples": n_samples,
    }


def channel_metric_row(
    asteroid_factory,
    *,
    level: float,
    model: str,
    n_steps: int = 10,
    seed: int = 7,
    multi_scale: bool = False,
    **prop_kwargs: Any,
) -> dict[str, Any]:
    """Build → propagate once; return scorecard + OAM magnitude vectors."""
    ast = asteroid_factory()
    prop = ast.propagate(
        turbulence_level=level,
        n_steps=n_steps,
        seed=seed,
        screen_model=model,
        multi_scale=multi_scale,
        apply_bmgl=True,
        **prop_kwargs,
    )
    m = prop.metrics
    assert m is not None
    ref = prop.field_reference
    obs = prop.field_final
    assert ref is not None and obs is not None
    oam_fid, oam_ph, wr, wo = oam_spectrum_metrics(ref, obs)
    ells = OAM_QUAT_ELLS
    mag_r = np.array([abs(complex(wr[str(e)])) for e in ells])
    mag_o = np.array([abs(complex(wo[str(e)])) for e in ells])
    # relative L1 change of magnitude spectrum
    mag_l1 = float(np.sum(np.abs(mag_o - mag_r)) / (np.sum(mag_r) + 1e-12))
    return {
        "model": model,
        "level": level,
        "multi_scale": multi_scale,
        "F": m.overlap_fidelity,
        "OAMf": oam_fid,
        "oam_phase_err": oam_ph,
        "Icorr": m.intensity_correlation,
        "phi_rms": m.phase_rmse_rad,
        "P": m.power_retention,
        "Strehl": m.strehl_proxy,
        "tt": m.tip_tilt_rms,
        "mag_l1": mag_l1,
        "mag_ref": mag_r,
        "mag_obs": mag_o,
        "ells": ells,
        "F_over_OAMf": m.overlap_fidelity / (oam_fid + 1e-12),
        "OAMf_minus_F": oam_fid - m.overlap_fidelity,
    }


def explain_oam_vs_f(row: dict[str, Any], structure: dict[str, Any] | None = None) -> str:
    """One-paragraph interpretation for a channel metric row."""
    parts = [
        f"model={row['model']}"
        + (" multi_scale" if row.get("multi_scale") else ""),
        f"L={row['level']:.2f}",
        f"F={row['F']:.3f}",
        f"OAMf={row['OAMf']:.3f}",
        f"Δ(OAMf−F)={row['OAMf_minus_F']:.3f}",
        f"|c|_L1={row['mag_l1']:.3f}",
    ]
    if structure is not None:
        parts.append(f"high_k={structure['high_k_fraction_mean']:.3f}")
        parts.append(f"az_rough={structure['azimuthal_roughness_mean']:.3f}")
        parts.append(f"LG_diag={structure['oam_diag_retention_mean']:.3f}")
    return "  ".join(parts)
