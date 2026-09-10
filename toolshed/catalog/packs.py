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
    # Is there a recipe file at all? Distinct from whether its facts are
    # frozen: a missing file is a mistake, an unfrozen one is a step not taken.
    recipe_found: bool = True
    # A warning shown on the row, written by hand. Overrides the automatic one.
    caution_text: str = ""
    # Explicit sexual content. Kept out of the modality groups, off by default,
    # and shown only after the person has said they want to see it -- see
    # docs/ADULT-PACKS.md for why this is a flag on the pack rather than a
    # separate catalogue.
    adult: bool = False

    @property
    def download_gb(self) -> float | None:
        return None if self.download_bytes is None else round(self.download_bytes / 1e9, 1)

    def size_text(self) -> str:
        gb = self.download_gb
        return "size unknown" if gb is None else f"{gb:g} GB download"

    @property
    def is_frozen(self) -> bool:
        """Have this pack's downloads been verified against their publisher?

        The download total is the last thing tools/freeze_manifest.py writes,
        and it only writes it once every file's size is known -- so a size is
        exactly the signal that the whole recipe is frozen.
        """
        return self.download_bytes is not None

    def is_experimental_for(self, vendor: str) -> bool:
        return vendor in self.experimental_on

    def licence_text(self) -> str:
        """The licence as shown on screen.

        The catalogue keeps PENDING_FREEZE where nobody has read the model card
        yet, and that token is what fails a release build -- but it is not a
        thing to put in front of a beginner, so it is translated here rather
        than removed from the data.
        """
        if PENDING in self.licence:
            return "Licence not confirmed yet — read the model card before you rely on it"
        return self.licence

    def caution(self) -> str | None:
        """Something the person should know, that does not stop the install.

        Deliberately separate from unavailable_reason. That one greys the row
        out; this one lets them proceed knowing what is unverified, which is
        the difference between a safeguard and a wall.

        An authored `caution:` in packs.yaml wins, because the automatic one
        below reasons from is_frozen -- which really means "do we know the
        download size" -- and size is rarely the most important thing to say.
        A model whose own author warns it returns explicit images from prompts
        that did not ask for any needs that on the row, and would have got a
        sentence about byte counts instead.
        """
        if self.caution_text:
            return " ".join(self.caution_text.split())
        if not self.is_frozen:
            return ("We do not know this download's size in advance, and the file it "
                    "names has not been confirmed from here — if the publisher has "
                    "renamed or removed it, the download will fail and say so. What "
                    "does arrive is still checked against the publisher's own hash as "
                    "it lands, the same as every other pack.")
        return None

    def unavailable_reason(self, vram_gb: float | None) -> str | None:
        """Why this pack cannot be offered, in words a beginner can act on.

        An unfrozen recipe is NOT one of those reasons any more, and the block
        it used to raise said something untrue: "the download has not been
        checked against its publisher". No pack's downloads have been, at
        release time -- image_sdxl_simple ships with sha256 and size_bytes both
        PENDING_FREEZE, exactly like this one, and resolves them from the
        publisher at install time (see planner/plan.py Download). The only
        thing an unfrozen pack really lacked was estimated_download_bytes, a
        figure for the confirmation screen, which is not a safety property and
        is a poor reason to refuse to install anything.

        So it is a caution() now, not a wall. The owner asked for these to be
        installable knowing they are unverified; that is a decision the person
        who ships this gets to make, and the honest way to carry it out is to
        say what is unknown rather than to invent it.
        """
        if vram_gb is not None and vram_gb < self.vram_gb_min:
            return (
                f"Needs a graphics card with at least {self.vram_gb_min:g} GB of memory. "
                f"Yours has {vram_gb:g} GB."
            )
        return None


# Generated first, then hand-authored. Almost every recipe is derived from an
# upstream template and ends in .generated.yaml; a pack whose graph is an
# existing template but whose model is not (a different checkpoint in the same
# SDXL workflow, say) has nothing upstream to derive from and is written by
# hand. Both are read the same way, and the suffix says which kind it is
# without anyone having to open it.
RECIPE_SUFFIXES = (".generated.yaml", ".authored.yaml")


def _recipe_facts(recipe: str, catalog_dir: Path) -> tuple[bool, int | None]:
    """(is there a recipe file, what size does it declare).

    The two are separate answers and conflating them hid a real distinction. A
    missing file is a typo in `recipe:` and must fail the build. A file that is
    present but still carries PENDING_FREEZE is a pack whose facts nobody has
    verified yet -- honest, expected, and not a reason to refuse the whole
    catalogue.
    """
    for suffix in RECIPE_SUFFIXES:
        path = catalog_dir / "recipes" / f"{recipe}{suffix}"
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        size = data.get("estimated_download_bytes")
        return True, (size if isinstance(size, int) else None)
    return False, None


@lru_cache(maxsize=1)
def load_packs() -> tuple[Pack, ...]:
    """Read the catalogue. Cached: it does not change while the app runs."""
    catalog_dir = resources.resource_path("catalog")
    doc = yaml.safe_load((catalog_dir / "packs.yaml").read_text(encoding="utf-8")) or {}

    packs: list[Pack] = []
    for entry in doc.get("packs", []):
        adult = bool(entry.get("adult", False))
        found, size = _recipe_facts(entry["recipe"], catalog_dir)
        if adult and entry.get("default_checked"):
            # Not a warning to be tidied up later: a pre-ticked adult pack means
            # someone clicking Continue through the defaults downloads porn they
            # never asked for. Refuse the catalogue rather than the checkbox.
            raise ValueError(f"{entry['id']}: an adult pack cannot be default_checked")
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
                download_bytes=size,
                recipe_found=found,
                caution_text=entry.get("caution", ""),
                adult=adult,
            )
        )
    return tuple(packs)


def general(packs: tuple[Pack, ...]) -> tuple[Pack, ...]:
    """Everything that belongs in the ordinary modality groups."""
    return tuple(p for p in packs if not p.adult)


def adult(packs: tuple[Pack, ...]) -> tuple[Pack, ...]:
    """The packs that go behind the gate, in catalogue order."""
    return tuple(p for p in packs if p.adult)


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
