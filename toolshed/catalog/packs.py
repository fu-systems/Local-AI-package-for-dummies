"""Load the packs a user chooses between.

Two files feed this, and the split matters:

* ``catalog/packs.yaml`` is hand-written and holds only human-facing text and
  the hardware floor.
* ``catalog/recipes/*.generated.yaml`` is derived from Comfy Org's official
  templates and holds everything machine-checkable, including the download
  size.

The size is read from the generated file rather than copied into the
hand-written one, so the number on the confirmation screen cannot drift away
from the templates it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from toolshed import resources

PENDING = "PENDING_FREEZE"


@dataclass(frozen=True)
class Pack:
    id: str
    modality: str
    name: str
    blurb: str
    recipe: str
    vram_gb_min: float
    licence: str
    default_checked: bool = False
    experimental_on: tuple[str, ...] = ()
    download_bytes: int | None = None

    @property
    def download_gb(self) -> float | None:
        return None if self.download_bytes is None else round(self.download_bytes / 1e9, 1)

    def size_text(self) -> str:
        gb = self.download_gb
        return "size unknown" if gb is None else f"{gb:g} GB download"

    def is_experimental_for(self, vendor: str) -> bool:
        return vendor in self.experimental_on

    def unavailable_reason(self, vram_gb: float | None) -> str | None:
        """Why this pack cannot be offered, in words a beginner can act on."""
        if vram_gb is not None and vram_gb < self.vram_gb_min:
            return (
                f"Needs a graphics card with at least {self.vram_gb_min:g} GB of memory. "
                f"Yours has {vram_gb:g} GB."
            )
        return None


def _recipe_size(recipe: str, catalog_dir: Path) -> int | None:
    path = catalog_dir / "recipes" / f"{recipe}.generated.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    size = data.get("estimated_download_bytes")
    return size if isinstance(size, int) else None


@lru_cache(maxsize=1)
def load_packs() -> tuple[Pack, ...]:
    """Read the catalogue. Cached: it does not change while the app runs."""
    catalog_dir = resources.resource_path("catalog")
    doc = yaml.safe_load((catalog_dir / "packs.yaml").read_text(encoding="utf-8")) or {}

    packs: list[Pack] = []
    for entry in doc.get("packs", []):
        packs.append(
            Pack(
                id=entry["id"],
                modality=entry["modality"],
                name=entry["name"],
                blurb=" ".join(entry["blurb"].split()),  # unwrap the YAML folding
                recipe=entry["recipe"],
                vram_gb_min=float(entry["vram_gb_min"]),
                licence=entry["licence"],
                default_checked=bool(entry.get("default_checked", False)),
                experimental_on=tuple(entry.get("experimental_on", ())),
                download_bytes=_recipe_size(entry["recipe"], catalog_dir),
            )
        )
    return tuple(packs)


def modalities(packs: tuple[Pack, ...]) -> list[str]:
    """Modality names in catalogue order, without repeats."""
    seen: list[str] = []
    for pack in packs:
        if pack.modality not in seen:
            seen.append(pack.modality)
    return seen


def total_bytes(packs: list[Pack]) -> int:
    """Total download for a selection.

    Packs share large files -- the video and picture packs both pull a text
    encoder, for instance -- so this over-counts. Deduplication happens by
    sha256 once the manifest is frozen; until then the honest thing is to
    present this as an upper bound rather than a precise figure.
    """
    return sum(p.download_bytes or 0 for p in packs)
