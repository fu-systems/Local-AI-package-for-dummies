"""Locating data files, both in a source checkout and inside a frozen build.

PyInstaller unpacks bundled data under ``sys._MEIPASS``. In a onedir build
since PyInstaller 6.0 that is ``<app>/_internal``, not the directory holding
the executable, so nothing may assume the old flat layout.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a PyInstaller build rather than a source tree."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_root() -> Path:
    """The directory that bundled data files live under."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # toolshed/resources.py -> toolshed/ -> repository root
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Resolve a bundled data path, e.g. ``resource_path("catalog", "recipes")``."""
    return bundle_root().joinpath(*parts)
