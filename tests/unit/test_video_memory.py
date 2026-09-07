"""Tests for not running the card out of memory on a video job.

The failure these defend against is the worst one this product has: a video
job samples happily for several minutes, reaches the VAE decode, allocates
every frame at full resolution in one go, and dies with nothing to show. No
engine flag prevents it -- --reserve-vram, --lowvram and dynamic VRAM all
decide where model *weights* live, and the weights were never the problem.

Two things fix it, and both are tested here:

* the sizes easy mode offers are the ones the card can finish, chosen from
  `catalog/video_presets.yaml` by detected VRAM;
* the decode is tiled on its way to the engine, so the peak allocation is a
  slice of frames rather than all of them.
"""

from __future__ import annotations

import pytest

from toolshed.catalog.presets import default_preset, presets_for
from toolshed.easy.knobs import Settings, analyse, apply, use_tiled_decode


class TestPresetsMatchTheCard:
    def test_a_big_card_is_offered_more_than_a_small_one(self):
        big = presets_for(24)
        small = presets_for(8)
        assert max(p.length for p in big) > max(p.length for p in small)

    def test_every_tier_offers_something(self):
        """A card with no entry would leave the dropdown empty and video
        undrivable from easy mode."""
        for vram in (6, 8, 12, 16, 20, 24, 48):
            assert presets_for(vram), f"no presets for {vram} GB"

    def test_unreadable_vram_gets_the_smallest_tier(self):
        """Detection failing is not evidence of a big card. Too small wastes
        some of a good card; too big wastes ten minutes and produces nothing."""
        assert presets_for(None) == presets_for(0)

    def test_the_twenty_gig_card_that_prompted_this_is_not_offered_121_frames(self):
        """The reported failure, verbatim: 1280x704x121 on a 20 GB card."""
        offered = presets_for(20)
        assert all(not (p.length >= 121 and p.width >= 1280) for p in offered)

    def test_tiers_are_sorted_even_if_the_file_is_not(self):
        vrams = [24, 16, 12, 8]
        lengths = [max(p.length for p in presets_for(v)) for v in vrams]
        assert lengths == sorted(lengths, reverse=True)

    def test_frame_counts_sit_on_the_grid_the_node_accepts(self):
        """Wan22ImageToVideoLatent declares length min 1 step 4, so 4n+1.
        An off-grid value is a validation error on somebody's first video."""
        for vram in (0, 8, 12, 16, 24):
            for preset in presets_for(vram):
                assert (preset.length - 1) % 4 == 0, preset

    def test_sizes_sit_on_the_grid_the_node_accepts(self):
        """width and height are declared min 32, step 32."""
        for vram in (0, 8, 12, 16, 24):
            for preset in presets_for(vram):
                assert preset.width % 32 == 0 and preset.height % 32 == 0, preset

    def test_one_preset_per_tier_is_the_default(self):
        for vram in (0, 12, 16, 24):
            presets = presets_for(vram)
            assert len([p for p in presets if p.default]) <= 1
            assert default_preset(presets) is not None


VIDEO_GRAPH = {
    "1": {"class_type": "Wan22ImageToVideoLatent",
          "inputs": {"width": 1280, "height": 704, "length": 121, "batch_size": 1}},
    "2": {"class_type": "KSampler",
          "inputs": {"seed": 1, "steps": 20, "latent_image": ["1", 0]}},
    "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["9", 0]}},
}

PICTURE_GRAPH = {
    "1": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "2": {"class_type": "KSampler",
          "inputs": {"seed": 1, "steps": 20, "latent_image": ["1", 0]}},
    "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["9", 0]}},
}


class FakeSpec:
    def __init__(self, inputs):
        self.inputs = inputs


# What the engine reports for the tiled node, per nodes.py at v0.34.0.
SPECS = {"VAEDecodeTiled": FakeSpec(
    ["samples", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"])}


class TestLengthIsADrivableKnob:
    def test_a_video_latent_is_recognised_as_video(self):
        assert analyse(VIDEO_GRAPH).is_video

    def test_a_picture_latent_is_not(self):
        """The picture workflows must not grow a video length dropdown."""
        assert not analyse(PICTURE_GRAPH).is_video

    def test_length_is_written_into_the_graph(self):
        knobs = analyse(VIDEO_GRAPH)
        out = apply(VIDEO_GRAPH, knobs, Settings(length=49, width=832, height=480))
        assert out["1"]["inputs"]["length"] == 49
        assert out["1"]["inputs"]["width"] == 832
        assert out["1"]["inputs"]["height"] == 480

    def test_leaving_length_alone_leaves_the_workflow_as_it_was(self):
        knobs = analyse(VIDEO_GRAPH)
        out = apply(VIDEO_GRAPH, knobs, Settings(prompt="a cat"))
        assert out["1"]["inputs"]["length"] == 121

    def test_length_does_not_also_appear_in_the_long_list(self):
        """It has its own control now; offering it twice invites setting it in
        both places and getting whichever is written last."""
        knobs = analyse(VIDEO_GRAPH)
        assert not any(c.input_name == "length" for c in knobs.advanced)


class TestTiledDecode:
    def test_the_decode_is_swapped_for_the_tiled_one(self):
        out = use_tiled_decode(VIDEO_GRAPH, SPECS)
        assert out["3"]["class_type"] == "VAEDecodeTiled"

    def test_it_decodes_a_slice_of_frames_at_a_time(self):
        """temporal_size is the whole point: 121 frames in passes of 64,
        rather than one allocation of 121."""
        out = use_tiled_decode(VIDEO_GRAPH, SPECS)
        assert out["3"]["inputs"]["temporal_size"] == 64

    def test_the_links_into_the_node_survive(self):
        out = use_tiled_decode(VIDEO_GRAPH, SPECS)
        assert out["3"]["inputs"]["samples"] == ["2", 0]
        assert out["3"]["inputs"]["vae"] == ["9", 0]

    def test_nothing_else_is_touched(self):
        out = use_tiled_decode(VIDEO_GRAPH, SPECS)
        assert out["1"] == VIDEO_GRAPH["1"]
        assert out["2"] == VIDEO_GRAPH["2"]

    def test_the_original_graph_is_not_mutated(self):
        """It is cached per workflow and reused for every run."""
        use_tiled_decode(VIDEO_GRAPH, SPECS)
        assert VIDEO_GRAPH["3"]["class_type"] == "VAEDecode"

    @pytest.mark.parametrize("specs", [None, {}, {"SomethingElse": FakeSpec([])}])
    def test_an_engine_without_the_node_is_left_alone(self, specs):
        """Sending a node the engine does not have turns a job that might have
        finished into one that certainly fails validation."""
        out = use_tiled_decode(VIDEO_GRAPH, specs)
        assert out["3"]["class_type"] == "VAEDecode"

    def test_an_input_the_engine_does_not_declare_is_dropped(self):
        """A renamed input upstream must not become a rejected prompt."""
        specs = {"VAEDecodeTiled": FakeSpec(["samples", "vae", "tile_size"])}
        out = use_tiled_decode(VIDEO_GRAPH, specs)
        assert "temporal_size" not in out["3"]["inputs"]
        assert out["3"]["inputs"]["tile_size"] == 512


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="PySide6 is not installed")
    from PySide6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])


class TestEasyModeUsesThem:
    """The wiring, which is where this would quietly not work."""

    def page(self, tmp_path, vram):
        from toolshed.ui.make import MakePage

        return MakePage(tmp_path, vram_gb=vram)

    def test_the_length_control_appears_for_video(self, qapp, tmp_path):
        page = self.page(tmp_path, 20)
        page._on_inspected(analyse(VIDEO_GRAPH))
        assert page._is_video
        assert page.video_size.count() == len(presets_for(20))

    def test_it_stays_away_from_the_picture_workflows(self, qapp, tmp_path):
        page = self.page(tmp_path, 20)
        page._on_inspected(analyse(PICTURE_GRAPH))
        assert not page._is_video
        assert page.current_video_preset() is None

    def test_the_preset_reaches_the_settings(self, qapp, tmp_path):
        """The whole point: leaving everything alone must still ask for a size
        the card can finish, not the template's 1280x704x121."""
        page = self.page(tmp_path, 20)
        page._on_inspected(analyse(VIDEO_GRAPH))
        settings = page.settings()
        chosen = default_preset(presets_for(20))
        assert settings.length == chosen.length
        assert (settings.width, settings.height) == (chosen.width, chosen.height)
        assert settings.length < 121

    def test_the_optional_panel_shows_what_will_be_asked_for(self, qapp, tmp_path):
        """Not the template's 1280x704 while something else is sent."""
        page = self.page(tmp_path, 12)
        page._on_inspected(analyse(VIDEO_GRAPH))
        chosen = default_preset(presets_for(12))
        assert (page.width.value(), page.height.value()) == (chosen.width, chosen.height)

    def test_a_typed_size_still_wins(self, qapp, tmp_path):
        """Someone who opened the optional panel and set a number is being
        specific; the preset must not overrule them."""
        page = self.page(tmp_path, 20)
        page._on_inspected(analyse(VIDEO_GRAPH))
        page.width.setValue(512)
        page.height.setValue(288)
        settings = page.settings()
        assert (settings.width, settings.height) == (512, 288)

    def test_choosing_another_length_updates_the_size(self, qapp, tmp_path):
        page = self.page(tmp_path, 24)
        page._on_inspected(analyse(VIDEO_GRAPH))
        page.video_size.setCurrentIndex(2)                  # the 'long' entry
        assert page.settings().length == 121

    def test_reset_returns_to_the_preset_not_the_template(self, qapp, tmp_path):
        page = self.page(tmp_path, 12)
        page._on_inspected(analyse(VIDEO_GRAPH))
        page.width.setValue(2048)
        page.reset_settings()
        chosen = default_preset(presets_for(12))
        assert page.width.value() == chosen.width

    def test_the_preset_survives_the_page_not_being_on_screen(self, qapp, tmp_path):
        """Qt calls an unshown widget invisible, so asking the widget rather
        than the graph would drop the preset whenever another page was up."""
        page = self.page(tmp_path, 20)
        page._on_inspected(analyse(VIDEO_GRAPH))
        assert not page.video_row.isVisible()      # never shown in this test
        assert page.current_video_preset() is not None

    def test_a_small_card_is_given_smaller_video(self, qapp, tmp_path):
        small = self.page(tmp_path, 8)
        small._on_inspected(analyse(VIDEO_GRAPH))
        big = self.page(tmp_path, 24)
        big._on_inspected(analyse(VIDEO_GRAPH))
        assert small.settings().width < big.settings().width
