"""Phase-screen generators for turbulent channel propagation.

Models
------
- ``kolmogorov`` — classic Fourier-domain Kolmogorov-like screen (default).
- ``convex_defect`` — structured screens from local pointer misalignment via
  the optional ``convex_defect`` package (``grid_to_phase_screen``).
- ``hybrid`` — convex-weighted mix of Kolmogorov + convex_defect.

Convex-defect screens are soft-optional: Kolmogorov always works; convex /
hybrid require ``convex_defect`` on ``PYTHONPATH`` or installed.

Conceptual map
--------------
``turbulence_level`` → global misalignment scale and phase RMS.
``convex_f`` / ``convex_s`` / ``convex_kappa`` → frequency, fractal scale,
and gauge detuning of the defect density ρ.
Evolving local grid → multi-step “memory” texture along the channel
(analogue of pointer + local x_ij in convex_defect.simulator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

ScreenModel = Literal["kolmogorov", "convex_defect", "hybrid"]


def convex_defect_available() -> bool:
    """True if the convex_defect package can be imported."""
    try:
        import convex_defect  # noqa: F401

        return True
    except ImportError:
        return False


def kolmogorov_phase_screen(
    size: int,
    level: float,
    rng: np.random.Generator,
    *,
    alpha: float = 5.0 / 3.0,
    height: int | None = None,
    width: int | None = None,
) -> NDArray[np.floating]:
    """Simplified Kolmogorov-like phase screen in Fourier domain.

    Returns a zero-mean real phase map with RMS ≈ ``level``.
    Supports non-square fields when height/width are set.
    """
    h = int(height if height is not None else size)
    w = int(width if width is not None else size)
    if level <= 0.0:
        return np.zeros((h, w), dtype=float)

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    psd = k2 ** (-(alpha + 2) / 2.0)
    psd[0, 0] = 0.0
    noise = rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))
    screen = np.fft.ifft2(noise * np.sqrt(psd)).real
    screen = screen - screen.mean()
    rms = float(screen.std()) + 1e-12
    return (level * screen / rms).astype(float)


def _require_convex_defect() -> Any:
    try:
        import convex_defect as cd
    except ImportError as exc:
        raise ImportError(
            "screen_model requires the convex_defect package. "
            "Install from ~/Projects/convex_defect (pip install -e .) "
            "or add it to PYTHONPATH."
        ) from exc
    return cd


def _scale_to_rms(screen: NDArray[np.floating], level: float) -> NDArray[np.floating]:
    """Zero-mean, scale to RMS = level (phase radians proxy)."""
    s = np.asarray(screen, dtype=float)
    s = s - float(s.mean())
    rms = float(s.std()) + 1e-12
    if level <= 0.0:
        return np.zeros_like(s)
    return (level * s / rms).astype(float)


@dataclass
class ConvexScreenState:
    """Evolving local-misalignment grid for multi-step convex screens."""

    grid: NDArray[np.floating]
    x_global: float
    t: float = 0.0
    step_index: int = 0


@dataclass
class PhaseScreenConfig:
    """Parameters for :class:`PhaseScreenEngine`."""

    model: ScreenModel = "kolmogorov"
    kolmogorov_alpha: float = 5.0 / 3.0
    # convex_defect knobs
    convex_f: float = 1.0
    convex_s: float = 1.0
    convex_kappa: float | None = None  # None → package κ*
    convex_gain: float = 1.0
    grid_correlation: float = 0.85
    grid_noise_frac: float = 0.25  # local noise = this × turbulence_level
    pointer_gamma: float = 0.35  # realign rate of global misalignment proxy
    hybrid_weight: float = 0.5  # fraction of convex in hybrid ∈ [0, 1]
    # map turbulence_level → initial |x| scale
    x_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.model not in ("kolmogorov", "convex_defect", "hybrid"):
            raise ValueError(f"unknown screen model: {self.model!r}")
        if not (0.0 <= self.hybrid_weight <= 1.0):
            raise ValueError("hybrid_weight must lie in [0, 1]")
        if self.convex_f <= 0 or self.convex_s <= 0:
            raise ValueError("convex_f and convex_s must be positive")


@dataclass
class PhaseScreenEngine:
    """Stateful multi-step phase-screen generator.

    Parameters
    ----------
    height, width :
        Field shape.
    level :
        Turbulence level (phase RMS target).
    rng :
        NumPy Generator.
    config :
        Screen model + convex/hybrid knobs.
    """

    height: int
    width: int
    level: float
    rng: np.random.Generator
    config: PhaseScreenConfig = field(default_factory=PhaseScreenConfig)
    _state: ConvexScreenState | None = field(default=None, init=False, repr=False)
    _cd: Any = field(default=None, init=False, repr=False)
    _kappa: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.model in ("convex_defect", "hybrid"):
            self._cd = _require_convex_defect()
            if self.config.convex_kappa is None:
                self._kappa = float(self._cd.KAPPA_STAR_DEFAULT)
            else:
                self._kappa = float(self.config.convex_kappa)
            self._init_convex_state()

    def _init_convex_state(self) -> None:
        assert self._cd is not None
        x0 = float(self.config.x_scale * self.level)
        noise = self.config.grid_noise_frac * max(self.level, 1e-6)
        grid = np.full((self.height, self.width), x0, dtype=float)
        if noise > 0.0:
            grid = grid + noise * self.rng.normal(size=grid.shape)
        self._state = ConvexScreenState(grid=grid, x_global=x0, t=0.0, step_index=0)

    def _step_convex_grid(self, dt: float = 1.0) -> NDArray[np.floating]:
        """Evolve global misalignment proxy + local grid one channel step."""
        assert self._state is not None
        cfg = self.config
        st = self._state
        # pointer-like realignment toward 0 + mild drive from residual turbulence
        drive = 0.15 * self.level * float(self.rng.normal())
        st.x_global = (
            st.x_global * (1.0 - cfg.pointer_gamma * dt)
            + drive * dt
        )
        noise = cfg.grid_noise_frac * max(self.level, 1e-6)
        c = cfg.grid_correlation
        st.grid = c * st.grid + (1.0 - c) * st.x_global
        if noise > 0.0:
            st.grid = st.grid + noise * self.rng.normal(size=st.grid.shape)
        st.t += dt
        st.step_index += 1
        return st.grid

    def _convex_rho_screen(self, grid: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._cd is not None and self._kappa is not None
        cfg = self.config
        # Prefer package helper when available
        if hasattr(self._cd, "grid_to_phase_screen"):
            rho = self._cd.grid_to_phase_screen(
                grid,
                cfg.convex_f,
                self._kappa,
                cfg.convex_s,
                gain=cfg.convex_gain,
            )
        else:
            rho = cfg.convex_gain * self._cd.defect_density(
                grid, cfg.convex_f, self._kappa, cfg.convex_s
            )
        return np.asarray(rho, dtype=float)

    def next_screen(self) -> NDArray[np.floating]:
        """Generate the next phase screen (shape height×width)."""
        cfg = self.config
        level = float(self.level)

        if cfg.model == "kolmogorov":
            return kolmogorov_phase_screen(
                max(self.height, self.width),
                level,
                self.rng,
                alpha=cfg.kolmogorov_alpha,
                height=self.height,
                width=self.width,
            )

        # convex path (and hybrid base)
        grid = self._step_convex_grid()
        rho = self._convex_rho_screen(grid)
        convex_phase = _scale_to_rms(rho, level)

        if cfg.model == "convex_defect":
            return convex_phase

        # hybrid
        kol = kolmogorov_phase_screen(
            max(self.height, self.width),
            level,
            self.rng,
            alpha=cfg.kolmogorov_alpha,
            height=self.height,
            width=self.width,
        )
        w = cfg.hybrid_weight
        mixed = (1.0 - w) * kol + w * convex_phase
        return _scale_to_rms(mixed, level)

    def metadata(self) -> dict[str, Any]:
        """Diagnostics for PropagationResult.metadata."""
        cfg = self.config
        out: dict[str, Any] = {
            "screen_model": cfg.model,
            "kolmogorov_alpha": cfg.kolmogorov_alpha,
            "convex_defect_available": convex_defect_available(),
        }
        if cfg.model in ("convex_defect", "hybrid"):
            out.update(
                {
                    "convex_f": cfg.convex_f,
                    "convex_s": cfg.convex_s,
                    "convex_kappa": self._kappa,
                    "convex_gain": cfg.convex_gain,
                    "grid_correlation": cfg.grid_correlation,
                    "grid_noise_frac": cfg.grid_noise_frac,
                    "pointer_gamma": cfg.pointer_gamma,
                    "hybrid_weight": cfg.hybrid_weight if cfg.model == "hybrid" else None,
                    "x_global": None if self._state is None else float(self._state.x_global),
                    "grid_steps": None if self._state is None else int(self._state.step_index),
                }
            )
        return out

    @property
    def grid_final(self) -> NDArray[np.floating] | None:
        if self._state is None:
            return None
        return np.asarray(self._state.grid, dtype=float)


def make_phase_screen_engine(
    height: int,
    width: int,
    level: float,
    rng: np.random.Generator,
    *,
    screen_model: ScreenModel = "kolmogorov",
    **kwargs: Any,
) -> PhaseScreenEngine:
    """Factory: build a :class:`PhaseScreenEngine` from keyword knobs.

    Extra kwargs map onto :class:`PhaseScreenConfig` field names
    (``convex_f``, ``hybrid_weight``, …).
    """
    cfg_fields = {f.name for f in PhaseScreenConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
    unknown = set(kwargs) - cfg_fields
    if unknown:
        raise TypeError(f"unknown phase-screen config keys: {sorted(unknown)}")
    cfg = PhaseScreenConfig(model=screen_model, **cfg_kwargs)
    return PhaseScreenEngine(
        height=height,
        width=width,
        level=level,
        rng=rng,
        config=cfg,
    )
