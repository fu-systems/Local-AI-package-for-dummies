"""Tests for the adult content gate.

The catalogue ships no adult pack yet, so every test here builds its own. That
is deliberate and worth keeping even once one exists: these are tests of the
gate, and they should not start passing or failing because somebody edited
packs.yaml.

What they defend is a short list, and all of it is about the same failure --
explicit material appearing in front of somebody who did not ask for it:

* it is never pre-ticked, and the catalogue is refused if it says otherwise;
* it is not in the modality groups, so it cannot be reached by scrolling;
* the rows do not exist on screen until the confirmation is accepted;
* nothing it holds counts as selected while it is shut.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")

from PySide6 import QtWidgets  # noqa: E402

from toolshed.catalog.packs import Pack, adult, general, modalities  # noqa: E402
from toolshed.hw.detect import Gpu, HardwareReport  # noqa: E402
from toolshed.ui.choose import ChoosePage  # noqa: E402

BIG_CARD = HardwareReport(os="linux", gpus=(Gpu(vendor="amd", vram_mb=24576, gfx="gfx1100"),))
SMALL_CARD = HardwareReport(os="linux", gpus=(Gpu(vendor="nvidia", name="GTX 1660", vram_mb=6144),))


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])
    yield app


def pack(pack_id: str, *, adult: bool = False, vram: float = 8, **kw) -> Pack:
    return Pack(
        id=pack_id,
        modality=kw.get("modality", "Pictures"),
        name=kw.get("name", pack_id),
        blurb="a blurb",
        recipe="some_recipe",
        vram_gb_min=vram,
        licence="Some licence",
        default_checked=kw.get("default_checked", False),
        download_bytes=kw.get("download_bytes", 7_000_000_000),
        adult=adult,
    )


ORDINARY = pack("image.zimage", default_checked=True)
EXPLICIT = pack("image.adult", adult=True, name="Adult pictures")
EXPLICIT_BIG = pack("video.adult", adult=True, name="Adult video", modality="Video", vram=16)


class TestTheCatalogueLayer:
    def test_packs_split_into_general_and_adult(self):
        packs = (ORDINARY, EXPLICIT)
        assert general(packs) == (ORDINARY,)
        assert adult(packs) == (EXPLICIT,)

    def test_a_catalogue_with_no_adult_packs_is_the_normal_case(self):
        assert adult((ORDINARY,)) == ()

    def test_adult_packs_are_kept_out_of_the_modality_groups(self):
        """Otherwise "Pictures" quietly grows an explicit row on the main list."""
        packs = (ORDINARY, EXPLICIT, EXPLICIT_BIG)
        assert modalities(general(packs)) == ["Pictures"]

    def test_a_default_checked_adult_pack_is_refused(self, tmp_path, monkeypatch):
        """The one thing that must fail loudly rather than be corrected quietly:
        accepting the defaults must never download porn."""
        from toolshed import resources
        from toolshed.catalog import packs as packs_module

        (tmp_path / "packs.yaml").write_text(
            "schema_version: 1\n"
            "packs:\n"
            "  - id: image.adult\n"
            "    modality: Pictures\n"
            '    name: "Adult pictures"\n'
            '    blurb: "explicit"\n'
            "    recipe: nope\n"
            "    vram_gb_min: 8\n"
            '    licence: "Some licence"\n'
            "    adult: true\n"
            "    default_checked: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(resources, "resource_path", lambda _name: tmp_path)
        packs_module.load_packs.cache_clear()
        with pytest.raises(ValueError, match="cannot be default_checked"):
            packs_module.load_packs()
        packs_module.load_packs.cache_clear()


class TestTheGateOnScreen:
    def test_no_adult_packs_means_no_section_at_all(self, qapp):
        """Today's catalogue. The screen must look exactly as it did before."""
        page = ChoosePage((ORDINARY,), BIG_CARD)
        assert page.adult_section is None

    def test_the_rows_are_hidden_until_confirmed(self, qapp):
        page = ChoosePage((ORDINARY, EXPLICIT), BIG_CARD)
        assert page.adult_section is not None
        assert not page.adult_section.unlocked
        assert not page.adult_section.holder.isVisible()

    def test_nothing_explicit_is_selected_while_the_section_is_shut(self, qapp):
        """Belt and braces: even if a checkbox were somehow ticked, a shut
        section contributes nothing to the install plan."""
        page = ChoosePage((ORDINARY, EXPLICIT), BIG_CARD)
        assert page.adult_section is not None
        for row in page.adult_section.rows:
            row.checkbox.setChecked(True)
        assert [p.id for p in page.selected()] == ["image.zimage"]

    def test_unlocking_reveals_the_rows_still_unticked(self, qapp):
        """Saying "show me" is not the same as saying "install it"."""
        page = ChoosePage((ORDINARY, EXPLICIT), BIG_CARD)
        assert page.adult_section is not None
        page.adult_section.unlock()
        assert page.adult_section.unlocked
        assert all(not r.checkbox.isChecked() for r in page.adult_section.rows)
        assert [p.id for p in page.selected()] == ["image.zimage"]

    def test_an_unlocked_pack_can_then_be_chosen(self, qapp):
        page = ChoosePage((ORDINARY, EXPLICIT), BIG_CARD)
        assert page.adult_section is not None
        page.adult_section.unlock()
        page.adult_section.rows[0].checkbox.setChecked(True)
        assert sorted(p.id for p in page.selected()) == ["image.adult", "image.zimage"]

    def test_hardware_floors_apply_behind_the_gate_too(self, qapp):
        """The gate must not become a way to sidestep the VRAM check and hand
        someone a 16 GB pack on a 6 GB card."""
        page = ChoosePage((ORDINARY, EXPLICIT_BIG), SMALL_CARD)
        assert page.adult_section is not None
        page.adult_section.unlock()
        row = page.adult_section.rows[0]
        assert not row.checkbox.isEnabled()
        assert row.reason and "16 GB" in row.reason
        row.checkbox.setChecked(True)
        assert page.selected() == []

    def test_the_total_updates_when_an_adult_pack_is_chosen(self, qapp):
        """The confirmation figure has to include it, or the download is a
        surprise."""
        page = ChoosePage((ORDINARY, EXPLICIT), BIG_CARD)
        assert page.adult_section is not None
        page.adult_section.unlock()
        page.adult_section.rows[0].checkbox.setChecked(True)
        assert "2 things" in page.total_label.text()


class TestTheAuthoredRecipeIsReadable:
    """A pack whose model is not in any upstream template has no .generated
    file to derive from, so the loader has to find a hand-authored one too."""

    def test_both_recipe_kinds_are_found(self, tmp_path, monkeypatch):
        from toolshed import resources
        from toolshed.catalog import packs as packs_module

        (tmp_path / "recipes").mkdir()
        (tmp_path / "recipes" / "derived.generated.yaml").write_text(
            "estimated_download_bytes: 5000000000\n", encoding="utf-8")
        (tmp_path / "recipes" / "byhand.authored.yaml").write_text(
            "estimated_download_bytes: 7000000000\n", encoding="utf-8")
        monkeypatch.setattr(resources, "resource_path", lambda _name: tmp_path)

        assert packs_module._recipe_size("derived", tmp_path) == 5_000_000_000
        assert packs_module._recipe_size("byhand", tmp_path) == 7_000_000_000
        assert packs_module._recipe_size("missing", tmp_path) is None

    def test_an_unfrozen_recipe_reports_no_size_rather_than_a_wrong_one(self, tmp_path):
        from toolshed.catalog import packs as packs_module

        (tmp_path / "recipes").mkdir()
        (tmp_path / "recipes" / "x.authored.yaml").write_text(
            "estimated_download_bytes: PENDING_FREEZE\n", encoding="utf-8")
        assert packs_module._recipe_size("x", tmp_path) is None

    def test_the_adult_recipe_names_a_model_but_is_not_yet_frozen(self):
        """The repo and path are a proposal for freeze_manifest.py to verify.
        The hash is the thing that must never be a proposal: it is what every
        downloaded byte is checked against, so it stays PENDING until the API
        has said what it is."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        try:
            from freeze_manifest import PENDING as PENDING_TOKEN
            from freeze_manifest import pending_in
        finally:
            sys.path.pop(0)
        import yaml

        from toolshed import resources

        path = (resources.resource_path("catalog") / "recipes"
                / "image_sdxl_adult.authored.yaml")
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc["id"] == "image.sdxl_adult"
        assert doc["default_checked"] is False
        assert doc["custom_nodes"] == []
        pending = pending_in(doc)
        assert doc["files"]["checkpoint"]["repo"] != PENDING_TOKEN
        assert doc["files"]["checkpoint"]["path"].endswith(".safetensors")
        assert "files.checkpoint.sha256" in pending, (
            "a hash must come from the API, never from a recipe author")
        assert "files.checkpoint.size_bytes" in pending
        assert "estimated_download_bytes" in pending, (
            "the download total shown to the user cannot be a guess")

    def test_no_adult_pack_is_offered_while_it_is_unfrozen(self):
        """The catalogue guard: a pack with no frozen size cannot be offered,
        so the gate stays invisible until the facts exist."""
        from toolshed.catalog.packs import adult, load_packs

        assert adult(load_packs()) == ()
