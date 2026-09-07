"""Video sizes offered per graphics card.

Easy mode's promise is that the button works. For pictures that is nearly free
-- a picture that is too big fails in seconds and costs nothing but a retry.
Video is not like that: a job runs for minutes, gets all the way to the decode,
and dies there with nothing to show, which is the single worst failure this
product can hand somebody.

So the sizes on offer are the ones the card can do, and the numbers behind them
are data (``catalog/video_presets.yaml``) rather than code, for the same reason
the pack catalogue is: replacing an estimate with a measurement should be a
YAML edit, not a release.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from toolshed import resources


@dataclass(frozen=True)
class VideoPreset:
    """One entry in the length dropdown."""

    id: str
    label: str
    width: int
    height: int
    length: int
    default: bool = False

    @property
    def frames(self) -> int:
        return self.length


@lru_cache(maxsize=1)
def _tiers() -> tuple[tuple[float, tuple[VideoPreset, ...]], ...]:
    path = resources.resource_path("catalog") / "video_presets.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    out: list[tuple[float, tuple[VideoPreset, ...]]] = []
    for tier in doc.get("tiers", []):
        presets = tuple(
            VideoPreset(
                id=entry["id"],
                label=entry["label"],
                width=int(entry["width"]),
                height=int(entry["height"]),
                length=int(entry["length"]),
                default=bool(entry.get("default", False)),
            )
            for entry in tier.get("presets", [])
        )
        out.append((float(tier["vram_gb_min"]), presets))

    # Sorted here rather than trusted from the file: a tier added in the wrong
    # place would otherwise silently hand a 24 GB card the 8 GB sizes.
    out.sort(key=lambda pair: pair[0], reverse=True)
    return tuple(out)


def presets_for(vram_gb: float | None) -> tuple[VideoPreset, ...]:
    """The sizes to offer a card with this much memory.

    ``None`` -- VRAM could not be read -- gets the smallest tier rather than
    the largest. Detection failing is not evidence of a big card, and the cost
    of being wrong is asymmetric: too small wastes some of a good card, too big
    wastes ten minutes and produces nothing.
    """
    tiers = _tiers()
    if not tiers:
        return ()
    if vram_gb is None:
        return tiers[-1][1]
    for floor, presets in tiers:
        if vram_gb >= floor:
            return presets
    return tiers[-1][1]


def default_preset(presets: tuple[VideoPreset, ...]) -> VideoPreset | None:
    """What the dropdown starts on: the marked one, else the first."""
    if not presets:
        return None
    for preset in presets:
        if preset.default:
            return preset
    return presets[0]
