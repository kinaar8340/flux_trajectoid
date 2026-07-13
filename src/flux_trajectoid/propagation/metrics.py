"""Turbulence / channel fidelity metrics for Photon Seed Asteroids.

Provides a multi-metric scorecard beyond a single overlap fidelity:

- Field overlap (Uhlmann / pure-state fidelity proxy)
- Intensity correlation + power retention
- Strehl-like peak ratio + phase RMSE
- OAM spectral fidelity (LG mode weight correlation)
- Tip/tilt residual estimate
- Optional sweep over turbulence levels
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..inner.vqc_encoder import OAM_QUAT_ELLS
from ..recovery.decoder import project_oam_spectrum

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid


@dataclass
class FidelityMetrics:
    """Structured channel quality report."""

    overlap_fidelity: float
    """|⟨ref|obs⟩|² / (‖ref‖² ‖obs‖²) — primary field fidelity ∈ [0, 1]."""

    intensity_correlation: float
    """Pearson correlation of |ψ|² maps."""

    power_retention: float
    """∫|obs|² / ∫|ref|²."""

    strehl_proxy: float
    """Peak |obs|² / peak |ref|² (Strehl-like)."""

    phase_rmse_rad: float
    """RMS phase error after global-phase alignment (radians)."""

    oam_fidelity: float
    """Cosine similarity of OAM weight magnitude spectra."""

    oam_phase_error_rad: float
    """Mean absolute phase error on shared OAM modes (radians)."""

    tip_tilt_rms: float
    """Estimated residual tip/tilt amplitude (phase slope RMS)."""

    turbulence_level: float = 0.0
    n_steps: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def summary_line(self) -> str:
        return (
            f"F={self.overlap_fidelity:.4f}  Icorr={self.intensity_correlation:.4f}  "
            f"P={self.power_retention:.4f}  Strehl={self.strehl_proxy:.4f}  "
            f"φrms={self.phase_rmse_rad:.3f}  OAM={self.oam_fidelity:.4f}  "
            f"tt={self.tip_tilt_rms:.3f}"
        )


def _align_global_phase(ref: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Multiply obs by e^{-iφ} maximizing Re⟨ref|obs⟩."""
    overlap = np.vdot(ref.ravel(), obs.ravel())
    if abs(overlap) < 1e-15:
        return obs
    return obs * np.exp(-1j * np.angle(overlap))


def field_overlap_fidelity(ref: np.ndarray, obs: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=complex)
    obs = np.asarray(obs, dtype=complex)
    num = abs(np.vdot(ref.ravel(), obs.ravel())) ** 2
    den = float(np.vdot(ref, ref).real * np.vdot(obs, obs).real) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


def intensity_correlation(ref: np.ndarray, obs: np.ndarray) -> float:
    a = (np.abs(ref) ** 2).ravel()
    b = (np.abs(obs) ** 2).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def power_retention(ref: np.ndarray, obs: np.ndarray) -> float:
    p_ref = float(np.sum(np.abs(ref) ** 2)) + 1e-12
    p_obs = float(np.sum(np.abs(obs) ** 2))
    return float(p_obs / p_ref)


def strehl_proxy(ref: np.ndarray, obs: np.ndarray) -> float:
    peak_ref = float(np.max(np.abs(ref) ** 2)) + 1e-12
    peak_obs = float(np.max(np.abs(obs) ** 2))
    return float(peak_obs / peak_ref)


def phase_rmse(ref: np.ndarray, obs: np.ndarray, *, intensity_weight: bool = True) -> float:
    """RMS phase difference after global phase alignment."""
    obs_a = _align_global_phase(ref, obs)
    dphi = np.angle(ref * np.conj(obs_a))  # wrapped difference
    if intensity_weight:
        w = np.abs(ref) ** 2
        w = w / (w.sum() + 1e-12)
        return float(np.sqrt(np.sum(w * dphi**2)))
    mask = np.abs(ref) > 0.05 * np.max(np.abs(ref))
    if not np.any(mask):
        return float(np.sqrt(np.mean(dphi**2)))
    return float(np.sqrt(np.mean(dphi[mask] ** 2)))


def oam_spectrum_metrics(
    ref: np.ndarray,
    obs: np.ndarray,
    ells: tuple[int, ...] = OAM_QUAT_ELLS,
) -> tuple[float, float, dict[str, complex], dict[str, complex]]:
    """Return (magnitude cosine sim, mean |Δphase|, ref_w, obs_w)."""
    wr = project_oam_spectrum(ref, ells=ells)
    wo = project_oam_spectrum(obs, ells=ells)
    mag_r = np.array([abs(wr[e]) for e in ells], dtype=float)
    mag_o = np.array([abs(wo[e]) for e in ells], dtype=float)
    den = float(np.linalg.norm(mag_r) * np.linalg.norm(mag_o)) + 1e-12
    cos_sim = float(np.clip(np.dot(mag_r, mag_o) / den, 0.0, 1.0))

    phase_errs = []
    for e in ells:
        if abs(wr[e]) > 1e-12 and abs(wo[e]) > 1e-12:
            phase_errs.append(abs(np.angle(wr[e] * np.conj(wo[e]))))
    phase_err = float(np.mean(phase_errs)) if phase_errs else 0.0
    return cos_sim, phase_err, {str(k): v for k, v in wr.items()}, {str(k): v for k, v in wo.items()}


def estimate_tip_tilt_rms(field: np.ndarray, ref: np.ndarray | None = None) -> float:
    """
    Estimate residual tip/tilt from phase gradient energy.

    If ``ref`` is given, uses phase of ref*conj(obs) after alignment.
    """
    if ref is not None:
        obs = _align_global_phase(ref, field)
        phase = np.angle(ref * np.conj(obs))
    else:
        phase = np.angle(field)
    # Central difference gradients
    gy, gx = np.gradient(phase)
    # unwrap-ish weight by intensity
    w = np.abs(field) ** 2
    w = w / (w.sum() + 1e-12)
    return float(np.sqrt(np.sum(w * (gx**2 + gy**2))))


def compute_fidelity_metrics(
    ref: np.ndarray,
    obs: np.ndarray,
    *,
    turbulence_level: float = 0.0,
    n_steps: int = 0,
    extras: dict[str, Any] | None = None,
) -> FidelityMetrics:
    """Full scorecard comparing reference and observed complex fields."""
    ref = np.asarray(ref, dtype=complex)
    obs = np.asarray(obs, dtype=complex)
    if ref.shape != obs.shape:
        # Center-crop / pad to match
        obs = _match_shape(obs, ref.shape)

    oam_fid, oam_ph, wr, wo = oam_spectrum_metrics(ref, obs)
    return FidelityMetrics(
        overlap_fidelity=field_overlap_fidelity(ref, obs),
        intensity_correlation=intensity_correlation(ref, obs),
        power_retention=power_retention(ref, obs),
        strehl_proxy=strehl_proxy(ref, obs),
        phase_rmse_rad=phase_rmse(ref, obs),
        oam_fidelity=oam_fid,
        oam_phase_error_rad=oam_ph,
        tip_tilt_rms=estimate_tip_tilt_rms(obs, ref=ref),
        turbulence_level=float(turbulence_level),
        n_steps=int(n_steps),
        extras={
            "oam_weights_ref": wr,
            "oam_weights_obs": wo,
            **(extras or {}),
        },
    )


def _match_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=arr.dtype)
    slices_src = []
    slices_dst = []
    for a, s in zip(arr.shape, shape):
        if a >= s:
            start = (a - s) // 2
            slices_src.append(slice(start, start + s))
            slices_dst.append(slice(0, s))
        else:
            start = (s - a) // 2
            slices_src.append(slice(0, a))
            slices_dst.append(slice(start, start + a))
    out[tuple(slices_dst)] = arr[tuple(slices_src)]
    return out


def sweep_turbulence(
    factory: Callable[[], PhotonSeedAsteroid],
    levels: list[float] | np.ndarray,
    *,
    n_steps: int = 16,
    seed: int | None = None,
    apply_bmgl: bool = True,
    recover_photonic: bool = True,
    **propagate_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Rebuild → propagate at each turbulence level → collect metrics (+ optional BER).

    ``factory`` should return a **built** PhotonSeedAsteroid (or unbuilt; we call build).

    Extra ``propagate_kwargs`` are forwarded to ``ast.propagate`` (e.g.
    ``screen_model="convex_defect"``, ``convex_f=1.5``, ``hybrid_weight=0.6``).
    """
    from ..utils.quaternion_utils import bit_error_rate, byte_error_rate

    rows: list[dict[str, Any]] = []
    for level in levels:
        ast = factory()
        if ast.shell is None:
            ast.build()
        # Fresh copy of protected field reference
        ref = None
        if ast.flux_state is not None and ast.flux_state.protected_field is not None:
            ref = ast.flux_state.protected_field.copy()
        prop = ast.propagate(
            turbulence_level=float(level),
            n_steps=n_steps,
            seed=seed,
            apply_bmgl=apply_bmgl,
            **propagate_kwargs,
        )
        metrics = prop.metrics
        row: dict[str, Any] = {
            "turbulence_level": float(level),
            "n_steps": n_steps,
            "screen_model": prop.metadata.get("screen_model", "kolmogorov"),
            "overlap_fidelity": metrics.overlap_fidelity if metrics else prop.fidelity_proxy,
            "intensity_correlation": metrics.intensity_correlation if metrics else None,
            "power_retention": metrics.power_retention if metrics else None,
            "strehl_proxy": metrics.strehl_proxy if metrics else None,
            "phase_rmse_rad": metrics.phase_rmse_rad if metrics else None,
            "oam_fidelity": metrics.oam_fidelity if metrics else None,
            "tip_tilt_rms": metrics.tip_tilt_rms if metrics else None,
            "fidelity_proxy": prop.fidelity_proxy,
        }
        if recover_photonic:
            rec = ast.recover(mode="photonic")
            dig = ast.quaternion.payload_bytes if ast.quaternion else b""
            row["photonic_byte_ber"] = (
                byte_error_rate(dig, rec.payload_hat) if dig else None
            )
            row["photonic_bit_ber"] = bit_error_rate(dig, rec.payload_hat) if dig else None
            row["crc_ok"] = rec.crc_ok
            row["chordal_error"] = rec.chordal_error_mean
        if ref is not None and prop.field_final is not None:
            row["ref_power"] = float(np.sum(np.abs(ref) ** 2))
        rows.append(row)
    return rows
