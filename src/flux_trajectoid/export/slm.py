"""SLM hologram export hook for Photon Seed Asteroids.

Converts the protected (or composite) complex field into phase-only
patterns suitable for spatial light modulators, with optional Gerchberg–Saxton
refinement and a self-contained export package (manifest + arrays + previews).

Inspired by vqc_proto ``orbital_braille.slm_typehead`` but wired to
flux_trajectoid shell/VQC fields rather than typehead orbs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from ..photon_seed_asteroid import PhotonSeedAsteroid
    from ..shell.generator import ShellGeometry
    from ..inner.vqc_encoder import QuaternionEncoding


@dataclass
class SLMDevicePreset:
    name: str
    resolution_x: int
    resolution_y: int
    pitch_um: float
    wavelength_nm: float = 1550.0
    bit_depth: int = 8
    notes: str = ""


SLM_PRESETS: dict[str, SLMDevicePreset] = {
    "generic_512": SLMDevicePreset(
        name="generic_512",
        resolution_x=512,
        resolution_y=512,
        pitch_um=8.0,
        notes="Default simulation grid.",
    ),
    "generic_256": SLMDevicePreset(
        name="generic_256",
        resolution_x=256,
        resolution_y=256,
        pitch_um=8.0,
        notes="Fast preview / CI.",
    ),
    "holoeye_pluto_2": SLMDevicePreset(
        name="holoeye_pluto_2",
        resolution_x=1920,
        resolution_y=1080,
        pitch_um=8.0,
        bit_depth=8,
        notes="Holoeye PLUTO-2 class (1920×1080, 8 µm).",
    ),
    "meadowlark_512": SLMDevicePreset(
        name="meadowlark_512",
        resolution_x=512,
        resolution_y=512,
        pitch_um=15.0,
        bit_depth=16,
        notes="Meadowlark 512×512 (16-bit phase).",
    ),
    "thorlabs_1080p": SLMDevicePreset(
        name="thorlabs_1080p",
        resolution_x=1920,
        resolution_y=1080,
        pitch_um=6.4,
        bit_depth=8,
        notes="1080p LCOS class.",
    ),
}


@dataclass
class SLMConfig:
    resolution_x: int = 512
    resolution_y: int = 512
    pitch_um: float = 8.0
    wavelength_nm: float = 1550.0
    extent: float = 2.0  # field units matching encoder grid
    bit_depth: int = 8
    phase_wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi"
    w0: float = 0.6

    @classmethod
    def from_preset(cls, preset: str | SLMDevicePreset, **kwargs) -> SLMConfig:
        p = SLM_PRESETS[preset] if isinstance(preset, str) else preset
        cfg = cls(
            resolution_x=p.resolution_x,
            resolution_y=p.resolution_y,
            pitch_um=p.pitch_um,
            wavelength_nm=p.wavelength_nm,
            bit_depth=p.bit_depth,
        )
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class SLMExportResult:
    """Paths and arrays from an SLM package export."""

    out_dir: Path
    phase_rad: np.ndarray
    phase_levels: np.ndarray
    phase_stack: np.ndarray | None
    manifest: dict[str, Any]
    files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "files": self.files,
            "manifest": self.manifest,
            "phase_shape": list(self.phase_rad.shape),
            "levels_dtype": str(self.phase_levels.dtype),
        }


def field_to_phase(
    field: np.ndarray,
    *,
    wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi",
) -> np.ndarray:
    """Extract phase of a complex field, wrapped."""
    phase = np.angle(np.asarray(field, dtype=complex))
    if wrap == "0_2pi":
        return np.mod(phase, 2 * np.pi)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def phase_to_levels(
    phase: np.ndarray,
    bit_depth: int = 8,
    wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi",
) -> np.ndarray:
    """Map radians → integer SLM drive levels."""
    if wrap == "0_2pi":
        p = np.mod(phase, 2 * np.pi)
        norm = p / (2 * np.pi)
    else:
        p = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        norm = (p + np.pi) / (2 * np.pi)
    max_val = (1 << bit_depth) - 1
    levels = np.round(norm * max_val)
    return levels.astype(np.uint16 if bit_depth > 8 else np.uint8)


def resize_field(
    field: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Resize complex field via separate real/imag zoom."""
    from scipy.ndimage import zoom

    field = np.asarray(field, dtype=complex)
    if field.shape == shape:
        return field
    zy = shape[0] / field.shape[0]
    zx = shape[1] / field.shape[1]
    re = zoom(field.real, (zy, zx), order=1)
    im = zoom(field.imag, (zy, zx), order=1)
    # zoom may be 1 off due to rounding
    re = _crop_or_pad(re, shape)
    im = _crop_or_pad(im, shape)
    return re + 1j * im


def _crop_or_pad(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=arr.dtype)
    h = min(shape[0], arr.shape[0])
    w = min(shape[1], arr.shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def gerchberg_saxton(
    target_amp: np.ndarray,
    n_iter: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """GS phase retrieval; returns SLM-plane phase in [0, 2π)."""
    rng = np.random.default_rng(seed)
    target_amp = np.asarray(target_amp, dtype=float)
    target_amp = target_amp / (target_amp.max() + 1e-12)
    ny, nx = target_amp.shape
    slm_phase = rng.uniform(0, 2 * np.pi, (ny, nx))

    for _ in range(n_iter):
        slm_field = np.exp(1j * slm_phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
        far = target_amp * np.exp(1j * np.angle(far))
        slm_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far)))
        slm_phase = np.mod(np.angle(slm_field), 2 * np.pi)
    return slm_phase


def compose_seed_field(
    asteroid: PhotonSeedAsteroid,
    *,
    source: Literal["protected", "composite", "shard0", "propagated"] = "protected",
) -> np.ndarray:
    """Pick the complex field used for hologram synthesis."""
    if source == "propagated":
        prop = getattr(asteroid, "_propagation", None)
        if prop is not None and prop.field_final is not None:
            return np.asarray(prop.field_final, dtype=complex)
        source = "protected"

    if source == "protected" and asteroid.flux_state is not None:
        if asteroid.flux_state.protected_field is not None:
            return np.asarray(asteroid.flux_state.protected_field, dtype=complex)

    if source == "shard0" and asteroid.quaternion is not None and asteroid.quaternion.fields:
        return np.asarray(asteroid.quaternion.fields[0], dtype=complex)

    # composite from OAM weights
    if asteroid.quaternion is not None:
        from ..inner.vqc_encoder import oam_weights_to_field

        return oam_weights_to_field(asteroid.quaternion.composite_weights)

    return np.ones((64, 64), dtype=complex)


def apply_shell_phase_bias(
    phase: np.ndarray,
    shell: ShellGeometry | None,
) -> np.ndarray:
    """Blend trajectoid phase/trench mask into SLM phase (identification imprint)."""
    if shell is None or shell.phase_trench_mask is None:
        return phase
    from scipy.ndimage import zoom

    mask = shell.phase_trench_mask
    # Map 1D curvature/phase signal → 2D azimuthal pattern
    h, w = phase.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    phi = np.arctan2(yy - cy, xx - cx)
    idx = ((phi + np.pi) / (2 * np.pi) * (len(mask) - 1)).astype(int)
    idx = np.clip(idx, 0, len(mask) - 1)
    bias = 0.35 * mask[idx]
    # Optional surface silhouette amplitude as soft aperture on bias
    if shell.vertices is not None:
        pass
    return np.mod(phase + bias, 2 * np.pi)


def build_phase_stack(
    encoding: QuaternionEncoding | None,
    cfg: SLMConfig,
    *,
    use_gs: bool = False,
    gs_iter: int = 16,
    include_shell: ShellGeometry | None = None,
) -> np.ndarray:
    """
    Stack one phase frame per shard field (or single composite).

    Shape: (n_frames, H, W)
    """
    from ..inner.vqc_encoder import oam_weights_to_field

    shape = (cfg.resolution_y, cfg.resolution_x)
    frames: list[np.ndarray] = []

    if encoding is not None and encoding.fields:
        sources = encoding.fields
    elif encoding is not None:
        sources = [
            oam_weights_to_field(w, grid_size=min(shape), w0=cfg.w0)
            for w in encoding.oam_weights
        ]
    else:
        sources = [np.ones((64, 64), dtype=complex)]

    for i, f in enumerate(sources):
        f_r = resize_field(f, shape)
        if use_gs:
            target = np.abs(f_r)
            ph = gerchberg_saxton(target, n_iter=gs_iter, seed=i)
        else:
            ph = field_to_phase(f_r, wrap=cfg.phase_wrap)
            if cfg.phase_wrap != "0_2pi":
                ph = np.mod(ph, 2 * np.pi)
        ph = apply_shell_phase_bias(ph, include_shell)
        frames.append(ph)

    return np.stack(frames, axis=0)


def _save_gray_png(levels: np.ndarray, path: Path) -> None:
    """Save grayscale via matplotlib (no Pillow required)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = levels.astype(float)
    if levels.dtype == np.uint16:
        arr = arr / 65535.0
    else:
        arr = arr / 255.0
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def _save_preview(stack: np.ndarray, path: Path, max_frames: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(max_frames, stack.shape[0])
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            ax.imshow(stack[i], cmap="twilight", vmin=0, vmax=2 * np.pi)
            ax.set_title(f"shard {i}")
        ax.axis("off")
    fig.suptitle("flux_trajectoid SLM phase stack", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def export_slm_package(
    asteroid: PhotonSeedAsteroid,
    out_dir: str | Path,
    *,
    preset: str = "generic_256",
    source: Literal["protected", "composite", "shard0", "propagated"] = "protected",
    use_gs: bool = False,
    gs_iter: int = 16,
    include_shell_bias: bool = True,
    stack_shards: bool = True,
    cfg: SLMConfig | None = None,
) -> SLMExportResult:
    """
    Export a hardware-oriented SLM package for a built PhotonSeedAsteroid.

    Writes
    ------
    - ``manifest.json`` — device + payload metadata
    - ``phase_rad.npy`` — primary phase map (radians)
    - ``phase_levels.npy`` / ``phase_levels.raw`` — quantized drive levels
    - ``phase_levels.png`` — grayscale preview
    - ``phase_stack.npy`` — per-shard phases (if ``stack_shards``)
    - ``preview_montage.png`` — tiled stack preview
    - ``field_complex.npy`` — complex field used (debug / replay)
    """
    if asteroid.shell is None and asteroid.quaternion is None:
        raise RuntimeError("Asteroid must be built before SLM export")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SLMConfig.from_preset(preset)
    shape = (cfg.resolution_y, cfg.resolution_x)

    field = compose_seed_field(asteroid, source=source)
    field_r = resize_field(field, shape)

    if use_gs:
        phase = gerchberg_saxton(np.abs(field_r), n_iter=gs_iter, seed=asteroid.seed)
    else:
        phase = field_to_phase(field_r, wrap="0_2pi")

    shell = asteroid.shell if include_shell_bias else None
    phase = apply_shell_phase_bias(phase, shell)
    levels = phase_to_levels(phase, bit_depth=cfg.bit_depth, wrap="0_2pi")

    files: list[str] = []

    def _write(name: str, saver) -> None:
        path = out / name
        saver(path)
        files.append(name)

    np.save(out / "phase_rad.npy", phase)
    files.append("phase_rad.npy")
    np.save(out / "phase_levels.npy", levels)
    files.append("phase_levels.npy")
    (out / "phase_levels.raw").write_bytes(levels.tobytes())
    files.append("phase_levels.raw")
    np.save(out / "field_complex.npy", field_r)
    files.append("field_complex.npy")
    _save_gray_png(levels, out / "phase_levels.png")
    files.append("phase_levels.png")

    stack = None
    if stack_shards and asteroid.quaternion is not None:
        stack = build_phase_stack(
            asteroid.quaternion,
            cfg,
            use_gs=use_gs,
            gs_iter=gs_iter,
            include_shell=shell,
        )
        np.save(out / "phase_stack.npy", stack)
        files.append("phase_stack.npy")
        _save_preview(stack, out / "preview_montage.png")
        files.append("preview_montage.png")

    payload_preview = None
    if isinstance(asteroid.payload, str):
        payload_preview = asteroid.payload[:128]
    elif isinstance(asteroid.payload, (bytes, bytearray)):
        payload_preview = bytes(asteroid.payload[:64]).hex()

    manifest: dict[str, Any] = {
        "generator": "flux_trajectoid/export/slm.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "payload_preview": payload_preview,
        "seed": asteroid.seed,
        "source": source,
        "use_gs": use_gs,
        "gs_iter": gs_iter if use_gs else None,
        "include_shell_bias": include_shell_bias,
        "device": {
            "preset": preset if isinstance(preset, str) else getattr(preset, "name", "custom"),
            **{k: getattr(cfg, k) for k in (
                "resolution_x", "resolution_y", "pitch_um", "wavelength_nm",
                "bit_depth", "phase_wrap", "extent", "w0",
            )},
        },
        "shell": {
            "hash": asteroid.shell.metadata.get("payload_hash") if asteroid.shell else None,
            "mismatch_deg": asteroid.shell.mismatch_deg if asteroid.shell else None,
            "kx": asteroid.shell.kx if asteroid.shell else None,
            "ky": asteroid.shell.ky if asteroid.shell else None,
        },
        "encoding": {
            "n_shards": asteroid.quaternion.metadata.get("n_shards") if asteroid.quaternion else None,
            "crc8": asteroid.quaternion.metadata.get("crc8") if asteroid.quaternion else None,
            "n_bytes": asteroid.quaternion.metadata.get("n_bytes") if asteroid.quaternion else None,
            "ells": list(asteroid.quaternion.composite_weights.keys()) if asteroid.quaternion else None,
        },
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    files.append("manifest.json")
    manifest["files"] = files

    return SLMExportResult(
        out_dir=out,
        phase_rad=phase,
        phase_levels=levels,
        phase_stack=stack,
        manifest=manifest,
        files=files,
    )
