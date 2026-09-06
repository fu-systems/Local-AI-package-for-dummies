"""Putting the ready-made workflows where ComfyUI will show them.

The Workflows sidebar is a plain directory listing of
``<user-directory>/default/workflows``. There is no index to register with and
no API to call: a ``.json`` dropped in the right folder appears, and
subdirectories become folders in the tree.

Everything goes under a single ``Toolshed/`` namespace so uninstalling removes
exactly what we added and never touches a workflow the user saved themselves.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from toolshed import resources

NAMESPACE = "Toolshed"

# Which workflow folders belong to which pack. A pack the user did not choose
# should not leave a workflow behind referencing models they do not have.
PACK_FOLDERS: dict[str, tuple[str, ...]] = {
    "image.zimage": ("image/01 Text to picture (Z-Image).json",),
    "image.sdxl": ("image/02 Text to picture (SDXL).json",),
    "image.qwen_edit": ("image/03 Edit a picture.json",),
    "video.wan22": ("video/01 Text or picture to video.json",),
    "audio.acestep": ("audio/01 Make music.json",),
    "model3d.trellis2": ("3d/01 Photo to 3D model.json",),
}

# Sidebar order comes from the folder name, so the numbers are load-bearing.
SECTION = {
    "image": "01 Pictures",
    "video": "02 Video",
    "audio": "03 Music",
    "3d": "04 3D models",
}


@dataclass(frozen=True)
class Injected:
    source: Path
    dest: Path


def workflows_dir(base_directory: Path) -> Path:
    """Where ComfyUI reads user workflows from, given --base-directory."""
    return base_directory / "user" / "default" / "workflows" / NAMESPACE


def injectable(pack_ids: list[str] | tuple[str, ...]) -> list[Path]:
    """Relative workflow paths for the chosen packs, in catalogue order."""
    out: list[Path] = []
    for pack_id in pack_ids:
        for rel in PACK_FOLDERS.get(pack_id, ()):
            out.append(Path(rel))
    return out


def inject(pack_ids: list[str] | tuple[str, ...], base_directory: Path) -> list[Injected]:
    """Copy the workflows for these packs into ComfyUI's user directory.

    Idempotent: re-running replaces our files and leaves everything else alone.
    """
    target_root = workflows_dir(base_directory)
    target_root.mkdir(parents=True, exist_ok=True)

    done: list[Injected] = []
    for rel in injectable(pack_ids):
        source = resources.resource_path("workflows", str(rel))
        if not source.is_file():
            raise FileNotFoundError(f"workflow missing from the bundle: {rel}")
        # Validate before copying: a workflow that will not parse is worse than
        # one that is absent, because the user only finds out on click.
        try:
            json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel} is not valid JSON: {exc}") from exc

        section = SECTION.get(rel.parts[0], rel.parts[0])
        dest = target_root / section / rel.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        done.append(Injected(source, dest))
    return done


def remove(base_directory: Path) -> int:
    """Remove only our namespace. Returns how many files went."""
    root = workflows_dir(base_directory)
    if not root.is_dir():
        return 0
    count = sum(1 for _ in root.rglob("*.json"))
    shutil.rmtree(root)
    return count
