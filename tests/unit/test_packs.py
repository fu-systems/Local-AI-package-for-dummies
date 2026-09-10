"""Tests for the pack catalogue: the thing the user actually chooses from."""

from __future__ import annotations

import pytest

from toolshed.catalog.packs import load_packs, modalities, total_bytes


@pytest.fixture(scope="module")
def packs():
    return load_packs()


def test_catalogue_is_not_empty(packs):
    assert packs, "packs.yaml produced nothing; the choose screen would be blank"


def test_every_pack_points_at_a_recipe_that_exists(packs):
    """A typo in `recipe:` is a mistake and must fail the build. This used to
    be checked by way of the download size, which conflated it with a recipe
    that exists but is not frozen yet -- a different and legitimate state."""
    missing = [p.id for p in packs if not p.recipe_found]
    assert not missing, f"packs.yaml points at no such recipe: {missing}"


def test_sizes_are_plausible(packs):
    """A generative model pack is gigabytes. Catching a units mistake here is
    cheaper than shipping a confirmation screen that says 0 GB.

    Only the frozen ones have a size to check; an unfrozen pack is covered by
    the test below instead."""
    for p in packs:
        if p.is_frozen:
            assert 1e9 < p.download_bytes < 200e9, f"{p.id}: {p.download_bytes} bytes"


def test_an_unfrozen_pack_is_installable_with_a_caution(packs):
    """Third position on this, and the last one is the owner's decision.

    It went: failing the suite (which hid the pack), then greying the row out
    (which showed it and refused to install it), and now installable with what
    is unknown said on the row.

    The block was also justified by something untrue -- "the download has not
    been checked against its publisher" -- when image_sdxl_simple ships with
    sha256 and size_bytes both PENDING_FREEZE and installs fine. What an
    unfrozen pack actually lacks is a size estimate for the confirmation
    screen, which is not a safety property.
    """
    for p in packs:
        if not p.is_frozen:
            assert p.unavailable_reason(None) is None, (
                f"{p.id}: an unfrozen recipe must not block the install")
            caution = p.caution()
            assert caution, f"{p.id} is unverified and says nothing about it"
            assert "not been confirmed" in caution
            assert p.size_text() == "size unknown"


def test_no_pack_shows_the_placeholder_token_on_screen(packs):
    """PENDING_FREEZE stays in the data -- it is what fails a release build --
    but it is not a thing to show a beginner now that these packs install."""
    for p in packs:
        assert "PENDING_FREEZE" not in p.licence_text()
        assert "PENDING_FREEZE" not in p.size_text()


def test_a_frozen_pack_is_not_blocked_by_this(packs):
    """The check must not quietly disable the whole catalogue."""
    frozen = [p for p in packs if p.is_frozen]
    assert frozen, "no frozen packs at all would mean nothing is installable"
    for p in frozen:
        assert p.unavailable_reason(None) is None


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
        setup's smoke test settles it later. An unfrozen pack is blocked for a
        different reason entirely, so it is not evidence either way here."""
        for p in packs:
            if p.is_frozen:
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
