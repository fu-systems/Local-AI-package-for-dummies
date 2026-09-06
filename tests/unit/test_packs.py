"""Tests for the pack catalogue: the thing the user actually chooses from."""

from __future__ import annotations

import pytest

from toolshed.catalog.packs import load_packs, modalities, total_bytes


@pytest.fixture(scope="module")
def packs():
    return load_packs()


def test_catalogue_is_not_empty(packs):
    assert packs, "packs.yaml produced nothing; the choose screen would be blank"


def test_every_pack_has_a_size_derived_from_its_recipe(packs):
    """Sizes come from catalog/recipes/*.generated.yaml, which is derived from
    Comfy Org's templates. A missing size means packs.yaml points at a recipe
    that does not exist, and the user would see 'size unknown'."""
    missing = [p.id for p in packs if p.download_bytes is None]
    assert not missing, f"no derived size for: {missing}"


def test_sizes_are_plausible(packs):
    """A generative model pack is gigabytes. Catching a units mistake here is
    cheaper than shipping a confirmation screen that says 0 GB."""
    for p in packs:
        assert 1e9 < p.download_bytes < 200e9, f"{p.id}: {p.download_bytes} bytes"


def test_modalities_keep_catalogue_order_without_repeats(packs):
    assert modalities(packs) == ["Pictures", "Video", "Music", "3D models"]


def test_blurbs_are_unwrapped(packs):
    """YAML folded scalars keep newlines; they would render as hard breaks."""
    for p in packs:
        assert "\n" not in p.blurb
        assert "  " not in p.blurb


class TestAvailability:
    def test_card_below_the_floor_gets_a_reason_naming_both_numbers(self, packs):
        pack = next(p for p in packs if p.vram_gb_min == 8)
        reason = pack.unavailable_reason(6.0)
        assert reason and "8 GB" in reason and "6 GB" in reason

    def test_card_at_the_floor_is_offered(self, packs):
        pack = next(p for p in packs if p.vram_gb_min == 8)
        assert pack.unavailable_reason(8.0) is None

    def test_unknown_vram_does_not_block(self, packs):
        """Failing to read VRAM must not lock someone out of their own card;
        setup's smoke test settles it later."""
        for p in packs:
            assert p.unavailable_reason(None) is None

    def test_amd_3d_is_flagged_experimental(self, packs):
        pack = next(p for p in packs if p.id == "model3d.trellis2")
        assert pack.is_experimental_for("amd")
        assert not pack.is_experimental_for("nvidia")


def test_total_is_a_sum_over_the_selection(packs):
    assert total_bytes([]) == 0
    two = list(packs)[:2]
    assert total_bytes(two) == sum(p.download_bytes for p in two)


def test_exactly_one_pack_is_ticked_by_default(packs):
    """More than one and a beginner accepting the defaults downloads tens of
    gigabytes they did not ask for."""
    defaults = [p.id for p in packs if p.default_checked]
    assert len(defaults) == 1, f"default-checked packs: {defaults}"
