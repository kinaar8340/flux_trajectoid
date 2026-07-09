"""Locate and import the live ``oam_flux`` package (submodule or install).

Search order
------------
1. Already importable ``oam_flux`` (pip / PYTHONPATH)
2. ``$FLUX_TRAJECTOID_OAM_FLUX_PATH`` if set (path to package *src* or root)
3. Repo submodule ``external/oam_flux/src``
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _repo_root() -> Path:
    # inner/oam_backend.py → repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _candidate_src_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("FLUX_TRAJECTOID_OAM_FLUX_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "oam_flux").is_dir():
            dirs.append(p)
        elif (p / "src" / "oam_flux").is_dir():
            dirs.append(p / "src")
        elif p.name == "oam_flux" and (p.parent / "oam_flux").is_dir():
            dirs.append(p.parent)
    root = _repo_root()
    sub = root / "external" / "oam_flux" / "src"
    if sub.is_dir():
        dirs.append(sub)
    # Sibling checkout (dev convenience)
    sibling = root.parent / "oam_flux" / "src"
    if sibling.is_dir():
        dirs.append(sibling)
    return dirs


def try_import_oam_flux() -> tuple[ModuleType | None, bool, str]:
    """
    Returns ``(module_or_None, is_live, backend_label)``.

    ``backend_label`` is ``"live"``, ``"live:path"``, or ``"stub"``.
    """
    try:
        mod = importlib.import_module("oam_flux")
        return mod, True, "live"
    except ImportError:
        pass

    for src in _candidate_src_dirs():
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)
        try:
            # Drop stale failed import if any
            if "oam_flux" in sys.modules:
                del sys.modules["oam_flux"]
            mod = importlib.import_module("oam_flux")
            return mod, True, f"live:{src}"
        except ImportError:
            continue

    return None, False, "stub"


def require_oam_attr(mod: ModuleType, name: str) -> Any:
    if not hasattr(mod, name):
        raise AttributeError(f"oam_flux missing attribute {name!r}")
    return getattr(mod, name)
