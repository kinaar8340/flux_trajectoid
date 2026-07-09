"""Helical twist packets + Hopf lattice flux flywheels.

Primary path uses the live ``oam_flux`` package (git submodule
``external/oam_flux`` or an installed wheel):

- ``TwistLattice`` PDE + flywheel sites
- ``OAMPacket`` kinetic momentum bookkeeping
- ``PhotonicsConfig`` / ``propagate_multi_ell_vectorized``
- ``deposit_on_flywheels`` / ``VQCCouplingState`` / ``run_vqc_coupling_step``

Falls back to a lightweight numpy stub when ``oam_flux`` is unavailable so
examples and CI still run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..shell.generator import ShellGeometry
from ..shell.modulator import ShellModulation, apply_modulation, shell_to_phase_mask
from .oam_backend import try_import_oam_flux
from .vqc_encoder import QuaternionEncoding, oam_weights_to_field

_OAM, _LIVE, _BACKEND_LABEL = try_import_oam_flux()


# ---------------------------------------------------------------------------
# Packet type: live OAMPacket when available, else local stub
# ---------------------------------------------------------------------------
if _LIVE and _OAM is not None:
    OAMPacket = _OAM.OAMPacket  # type: ignore[misc, assignment]
else:

    @dataclass
    class OAMPacket:  # type: ignore[no-redef]
        """Propagating ℓ-mode twist packet (stub)."""

        ell: int = 3
        lambda_nm: float = 1550.0
        w0: float = 1.0
        energy_scale: float = 1.0
        z: float = 0.0

        @property
        def helical_phase_gradient(self) -> float:
            return float(self.ell)

        @property
        def momentum(self) -> float:
            return abs(self.ell) * self.energy_scale


@dataclass
class FluxState:
    """Inner nut state after coupling to Hopf lattice flywheels."""

    packets: list[Any]
    lattice_theta: np.ndarray
    lattice_theta0: np.ndarray
    deposited_momentum: float
    flywheel_load: np.ndarray
    protected_field: np.ndarray | None
    shell_modulation: ShellModulation | None
    kick_history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Live oam_flux handles (None on stub path)
    lattice: Any | None = None
    photonics_propagation: Any | None = None
    coupling_history: list[dict[str, float]] = field(default_factory=list)
    backend: str = "stub"

    @property
    def is_live(self) -> bool:
        return self.backend.startswith("live")


def oam_flux_backend() -> str:
    """Return current backend label (``live``, ``live:<path>``, or ``stub``)."""
    return _BACKEND_LABEL


def is_live_oam_flux() -> bool:
    return _LIVE


# ---------------------------------------------------------------------------
# Stub helpers (used only when live package is missing)
# ---------------------------------------------------------------------------
def _stub_helical_seed(nx: int, *, pitch: float = 0.35, amplitude: float = 1.2) -> np.ndarray:
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return amplitude * (0.5 + 0.5 * np.sin(pitch * (x + 2 * y - z)))


def _stub_hopf_fiber_coords(nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    rho = np.sqrt((x - np.pi) ** 2 + (y - np.pi) ** 2)
    rho = rho / (rho.max() + 1e-12)
    phi = np.arctan2(y - np.pi, x - np.pi)
    eta = np.mod(x + y + z, 2 * np.pi)
    return rho, phi, eta


def _stub_couple(
    encoding: QuaternionEncoding,
    shell: ShellGeometry | None,
    *,
    lattice_nx: int,
    flywheel_sites: int,
    kick_strength: float,
    seed: int,
    modulation: ShellModulation | None,
    protected: np.ndarray | None,
) -> FluxState:
    weights = encoding.composite_weights
    packets = [
        OAMPacket(ell=int(ell), energy_scale=float(abs(amp)))
        for ell, amp in sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    ]
    if not packets:
        packets = [OAMPacket(ell=1, energy_scale=1.0)]

    theta = _stub_helical_seed(lattice_nx)
    theta0 = theta.copy()
    rho, phi, eta = _stub_hopf_fiber_coords(lattice_nx)
    rng = np.random.default_rng(seed)
    sites = [tuple(int(v) for v in rng.integers(0, lattice_nx, size=3)) for _ in range(flywheel_sites)]

    total_mom = 0.0
    kick_history: list[float] = []
    flywheel_load = np.zeros(flywheel_sites)

    for i, pkt in enumerate(packets):
        helical = np.cos(pkt.ell * phi + 0.5 * eta)
        radial = np.exp(-((rho * 2.0) ** 2)) * (rho ** min(abs(pkt.ell), 4))
        kick = kick_strength * pkt.energy_scale * radial * helical
        mask = np.zeros_like(theta)
        for j, idx in enumerate(sites):
            local = np.zeros_like(mask)
            local[idx] = 1.0
            for ax in range(3):
                local = 0.25 * (np.roll(local, 1, ax) + np.roll(local, -1, ax) + 2 * local)
            mask += local
            flywheel_load[j] += float(np.abs(kick[idx]))
        mask /= max(mask.max(), 1e-12)
        kick = kick * mask
        mom = float(np.abs(kick).sum() * kick_strength)
        theta = np.clip(theta + kick, 0.01, 2 * np.pi - 0.01)
        total_mom += mom
        kick_history.append(mom)
        if shell is not None and shell.fourier_fingerprint is not None:
            fp = shell.fourier_fingerprint
            theta = theta + 0.01 * float(fp[i % len(fp)])

    return FluxState(
        packets=packets,
        lattice_theta=theta,
        lattice_theta0=theta0,
        deposited_momentum=total_mom,
        flywheel_load=flywheel_load,
        protected_field=protected,
        shell_modulation=modulation,
        kick_history=kick_history,
        backend="stub",
        metadata={
            "lattice_nx": lattice_nx,
            "flywheel_sites": flywheel_sites,
            "kick_strength": kick_strength,
            "n_packets": len(packets),
            "ells": [p.ell for p in packets],
            "mean_twist": float(theta.mean()),
            "twist_variance": float(theta.var()),
            "backend": "stub",
        },
    )


# ---------------------------------------------------------------------------
# Live oam_flux path
# ---------------------------------------------------------------------------
def _shell_kick_bias(shell: ShellGeometry | None, lattice_theta: np.ndarray, i: int) -> np.ndarray:
    if shell is None or shell.fourier_fingerprint is None:
        return lattice_theta
    fp = shell.fourier_fingerprint
    return lattice_theta + 0.01 * float(fp[i % len(fp)])


def _live_couple(
    encoding: QuaternionEncoding,
    shell: ShellGeometry | None,
    *,
    lattice_nx: int,
    flywheel_sites: int,
    kick_strength: float,
    seed: int,
    modulation: ShellModulation | None,
    protected: np.ndarray | None,
    n_steps: int,
    n_z: int,
    nr: int,
    l_max: int,
    lambda_nm: float,
    w0: float,
    conserve_momentum: bool,
    recovery_memory: float,
) -> FluxState:
    assert _OAM is not None
    TwistLattice = _OAM.TwistLattice
    PhotonicsConfig = _OAM.PhotonicsConfig
    propagate_multi_ell_vectorized = _OAM.propagate_multi_ell_vectorized
    VQCCouplingState = _OAM.VQCCouplingState
    run_vqc_coupling_step = _OAM.run_vqc_coupling_step
    OAMPacketLive = _OAM.OAMPacket

    weights = encoding.composite_weights
    if not weights:
        weights = {1: 1.0 + 0j}

    # Dominant ells ordered by |amp|
    ordered = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    ells = [int(e) for e, _ in ordered]
    l_max_eff = max(int(l_max), max(abs(e) for e in ells) + 1, 2)

    # Seed lattice IC for reproducibility (helical_seed is deterministic in nx)
    lattice = TwistLattice(nx=int(lattice_nx))
    # Optional fingerprint micro-bias on IC (shell identity imprint)
    if shell is not None and shell.fourier_fingerprint is not None:
        fp = shell.fourier_fingerprint
        lattice.theta = np.clip(
            lattice.theta + 0.005 * float(fp[seed % len(fp)]),
            0.01,
            2 * np.pi - 0.01,
        )
        lattice.theta_initial = lattice.theta.copy()

    photonics = PhotonicsConfig(
        l_max=l_max_eff,
        w0=float(w0),
        lambda_nm=float(lambda_nm),
        nr=int(nr),
        z_start=0.0,
        z_end=5.0,
        n_z=int(n_z),
        turbulence=0.0,
        chirp=0.0,
        qec_suppression=1,
    )
    # Deterministic turbulence noise not used (turbulence=0); pin numpy RNG anyway
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        propagation = propagate_multi_ell_vectorized(photonics)
    finally:
        np.random.set_state(rng_state)

    # Kinetic momentum helper
    try:
        from oam_flux.momentum import oam_kinetic_momentum  # type: ignore
    except ImportError:
        oam_kinetic_momentum = None  # type: ignore

    packets: list[Any] = []
    kick_history: list[float] = []
    coupling_history: list[dict[str, float]] = []
    total_deposited = 0.0
    ledger0 = float(lattice.momentum_ledger)

    steps = max(1, min(int(n_steps), int(propagation.n_z)))

    for i, (ell, amp) in enumerate(ordered):
        e_scale = float(max(abs(amp), 1e-6))
        # Shell mismatch attenuates coupling slightly (worse shell → softer deposit)
        shell_atten = 1.0
        if shell is not None:
            shell_atten = float(np.clip(1.0 - shell.mismatch_deg / 360.0, 0.35, 1.0))

        k_eff = float(kick_strength) * e_scale * shell_atten
        if oam_kinetic_momentum is not None:
            p0 = float(oam_kinetic_momentum(energy_scale=e_scale, ell=int(ell), lambda_nm=lambda_nm))
        else:
            p0 = e_scale * abs(int(ell))

        pkt = OAMPacketLive(
            ell=int(ell),
            lambda_nm=float(lambda_nm),
            w0=float(w0),
            energy_scale=e_scale,
        )
        packets.append(pkt)

        state = VQCCouplingState(
            lattice=lattice,
            propagation=propagation,
            ell=int(ell),
            kick_strength=k_eff,
            energy_scale=e_scale,
            flywheel_sites=int(flywheel_sites),
            conserve_momentum=bool(conserve_momentum),
            z_index=0,
            lambda_nm=float(lambda_nm),
            photon_reservoir=p0,
            initial_total_momentum=p0,
            recovery_memory=float(recovery_memory),
        )

        mean_before = lattice.mean_twist
        for step in range(steps):
            run_vqc_coupling_step(state, step)
        coupling_history.extend(state.history)

        # Deposited ≈ ledger change + reservoir drain proxy
        delta_ledger = abs(float(lattice.momentum_ledger) - ledger0)
        ledger0 = float(lattice.momentum_ledger)
        deposited = delta_ledger if delta_ledger > 0 else abs(mean_before - lattice.mean_twist)
        # Prefer history if available
        if state.history:
            deposited = abs(
                float(state.history[-1].get("lattice_received", 0.0))
                - float(state.history[0].get("lattice_received", 0.0))
            ) or deposited
        kick_history.append(float(deposited))
        total_deposited += float(deposited)

        lattice.theta = _shell_kick_bias(shell, lattice.theta, i)

    # Flywheel load from |θ − θ₀| at resonator sites
    sites = lattice.flywheel_indices(int(flywheel_sites))
    delta = np.abs(lattice.theta - lattice.theta_initial)
    flywheel_load = np.array([float(delta[idx]) for idx in sites], dtype=float)

    # Optional emergence probe (best-effort)
    emergence_meta: dict[str, Any] = {}
    try:
        if hasattr(_OAM, "emergence_report") and coupling_history:
            # Lightweight survival proxy from history
            twists = [h.get("mean_twist", 0.0) for h in coupling_history]
            if twists:
                emergence_meta["mean_twist_trace_len"] = len(twists)
                emergence_meta["mean_twist_final"] = float(twists[-1])
                emergence_meta["mean_twist_initial"] = float(twists[0])
                emergence_meta["twist_survival"] = float(
                    abs(twists[-1]) / (abs(twists[0]) + 1e-12)
                )
    except Exception:
        pass

    return FluxState(
        packets=packets,
        lattice_theta=lattice.theta.copy(),
        lattice_theta0=lattice.theta_initial.copy(),
        deposited_momentum=float(total_deposited),
        flywheel_load=flywheel_load,
        protected_field=protected,
        shell_modulation=modulation,
        kick_history=kick_history,
        lattice=lattice,
        photonics_propagation=propagation,
        coupling_history=coupling_history,
        backend=_BACKEND_LABEL,
        metadata={
            "lattice_nx": lattice_nx,
            "flywheel_sites": flywheel_sites,
            "kick_strength": kick_strength,
            "n_packets": len(packets),
            "ells": [int(p.ell) for p in packets],
            "mean_twist": float(lattice.mean_twist),
            "twist_variance": float(lattice.twist_variance),
            "twist_load_vs_initial": float(lattice.twist_load_vs_initial()),
            "momentum_ledger": float(lattice.momentum_ledger),
            "backend": _BACKEND_LABEL,
            "n_coupling_steps": steps,
            "photonics_n_z": int(propagation.n_z),
            "photonics_l_max": l_max_eff,
            "lambda_nm": lambda_nm,
            "conserve_momentum": conserve_momentum,
            "oam_flux_version": getattr(_OAM, "__version__", "unknown"),
            **emergence_meta,
        },
    )


def couple_to_flux_lattice(
    encoding: QuaternionEncoding,
    shell: ShellGeometry | None = None,
    *,
    lattice_nx: int = 16,
    flywheel_sites: int = 4,
    kick_strength: float = 0.05,
    seed: int = 42,
    n_steps: int = 16,
    n_z: int = 32,
    nr: int = 128,
    l_max: int = 4,
    lambda_nm: float = 1550.0,
    w0: float = 1.0,
    conserve_momentum: bool = True,
    recovery_memory: float = 0.0,
    force_stub: bool = False,
) -> FluxState:
    """
    Couple VQC OAM state into helical packets + Hopf lattice flux.

    1. Build shell-protected complex field (phase mask / trench)
    2. Live path: multi-ℓ VQC photonics → ``deposit_on_flywheels`` via
       ``run_vqc_coupling_step`` on a shared ``TwistLattice``
    3. Stub path: helical Hopf kick (legacy numpy)

    Parameters
    ----------
    force_stub
        If True, skip live backend (tests / offline demos).
    """
    weights = encoding.composite_weights

    # Shell protection (shared by both backends)
    modulation: ShellModulation | None = None
    protected: np.ndarray | None = None
    if shell is not None:
        modulation = shell_to_phase_mask(shell, grid_size=64)
        field = oam_weights_to_field(weights or {1: 1.0 + 0j}, grid_size=64)
        dom_ell = max(weights, key=lambda e: abs(weights[e])) if weights else 1
        protected = apply_modulation(field, modulation, ell=int(dom_ell))

    if _LIVE and not force_stub:
        try:
            return _live_couple(
                encoding,
                shell,
                lattice_nx=lattice_nx,
                flywheel_sites=flywheel_sites,
                kick_strength=kick_strength,
                seed=seed,
                modulation=modulation,
                protected=protected,
                n_steps=n_steps,
                n_z=n_z,
                nr=nr,
                l_max=l_max,
                lambda_nm=lambda_nm,
                w0=w0,
                conserve_momentum=conserve_momentum,
                recovery_memory=recovery_memory,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            # Fall through to stub with note
            state = _stub_couple(
                encoding,
                shell,
                lattice_nx=lattice_nx,
                flywheel_sites=flywheel_sites,
                kick_strength=kick_strength,
                seed=seed,
                modulation=modulation,
                protected=protected,
            )
            state.metadata["backend"] = "stub:fallback"
            state.metadata["live_error"] = repr(exc)
            state.backend = "stub:fallback"
            return state

    return _stub_couple(
        encoding,
        shell,
        lattice_nx=lattice_nx,
        flywheel_sites=flywheel_sites,
        kick_strength=kick_strength,
        seed=seed,
        modulation=modulation,
        protected=protected,
    )


def relax_lattice_steps(
    flux: FluxState,
    n_steps: int,
    *,
    recovery_memory: float = 0.15,
    pump_active: bool = False,
    external_turbulence: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """
    Advance lattice PDE for propagation layer.

    Uses live ``TwistLattice.relax_step`` when ``flux.lattice`` is present;
    otherwise a compact numpy Laplacian step on ``flux.lattice_theta``.
    """
    rng = rng or np.random.default_rng(0)
    mean_trace: list[float] = []

    if flux.lattice is not None and hasattr(flux.lattice, "relax_step"):
        lat = flux.lattice
        for _ in range(n_steps):
            external = None
            if external_turbulence > 0:
                external = external_turbulence * 0.01 * rng.normal(size=lat.theta.shape)
            lat.relax_step(
                external_torque=external,
                pump_active=pump_active,
                recovery_memory=recovery_memory,
            )
            mean_trace.append(float(lat.mean_twist))
        flux.lattice_theta = lat.theta.copy()
        return mean_trace

    # Stub PDE
    theta = flux.lattice_theta.copy()
    theta0 = flux.lattice_theta0
    nx = theta.shape[0]
    dt, D, kappa, delta_omega = 0.001, 0.05, 0.85, 0.002
    for _ in range(n_steps):
        lap = (
            np.roll(theta, 1, 0)
            + np.roll(theta, -1, 0)
            + np.roll(theta, 1, 1)
            + np.roll(theta, -1, 1)
            + np.roll(theta, 1, 2)
            + np.roll(theta, -1, 2)
            - 6 * theta
        ) * (nx**2)
        rhs = D * lap + delta_omega - kappa * float(theta.mean())
        if external_turbulence > 0:
            rhs = rhs + external_turbulence * 0.01 * rng.normal(size=theta.shape)
        theta = np.clip(theta + dt * rhs, 0.01, 2 * np.pi - 0.01)
        if recovery_memory > 0 and not pump_active:
            alpha = (1.0 - recovery_memory) * (1.0 - np.exp(-1.0 / 25.0))
            theta = np.clip(theta + alpha * (theta0 - theta), 0.01, 2 * np.pi - 0.01)
        mean_trace.append(float(theta.mean()))
    flux.lattice_theta = theta
    return mean_trace
