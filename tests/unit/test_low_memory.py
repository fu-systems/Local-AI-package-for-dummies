"""Tests for the low-memory switch.

Prompted by a user still hitting out-of-memory, holding advice from elsewhere
that recommended:

    --lowvram --async-offload 3 --reserve-vram 2.0 --disable-pinned-memory
    --force-non-blocking

Read against comfy/cli_args.py at the tag we ship, most of that is wrong here,
and two parts are actively harmful -- so what this switch turns on is a
different, verified set. The tests below pin both halves: what it passes, and
what it must never pass.

  --async-offload N  is the exact path AMD_SAFEGUARDS disables after a fault
                     that killed two finished jobs
  --reserve-vram N   withholds memory FROM ComfyUI; the Linux default is
                     0.4 GB, so 2.0 makes an out-of-memory MORE likely
  --lowvram          does not stream a model layer by layer; its own help says
                     it moves text encoders to the CPU, and only when dynamic
                     vram is off -- which is our state on AMD, so it is kept
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolshed.exec.engine import (
    AMD_SAFEGUARDS,
    CACHE_GROUP,
    LOW_MEMORY_OPTIONS,
    VRAM_GROUP,
    choose_layer_streaming,
    choose_low_memory,
    read_layer_streaming,
    read_low_memory,
    write_layer_streaming,
    write_low_memory,
)

CLI_ARGS = '''
parser.add_argument("--disable-smart-memory", action="store_true")
parser.add_argument("--cache-none", action="store_true")
parser.add_argument("--lowvram", action="store_true")
parser.add_argument("--novram", action="store_true")
parser.add_argument("--async-offload", nargs="?")
parser.add_argument("--reserve-vram", type=float)
parser.add_argument("--force-non-blocking", action="store_true")
'''


def engine_tree(root: Path, source: str = CLI_ARGS) -> Path:
    engine = root / "engine" / "comfyui"
    (engine / "comfy").mkdir(parents=True, exist_ok=True)
    (engine / "comfy" / "cli_args.py").write_text(source, encoding="utf-8")
    return engine


class TestWhatItTurnsOn:
    def test_off_by_default_it_adds_nothing(self, tmp_path):
        got = choose_low_memory(engine_tree(tmp_path), enabled=False)
        assert got.flags == []

    def test_on_it_adds_the_three_verified_options(self, tmp_path):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True)
        assert got.flags == ["--disable-smart-memory", "--lowvram", "--cache-none"]
        assert got.unavailable == ()

    def test_an_engine_that_renamed_one_reports_it(self, tmp_path):
        """Rather than passing an unknown flag, which stops ComfyUI starting
        at all -- a worse failure than the one we are treating."""
        engine = engine_tree(tmp_path, 'parser.add_argument("--listen")\n')
        got = choose_low_memory(engine, enabled=True)
        assert got.flags == []
        assert len(got.unavailable) == len(LOW_MEMORY_OPTIONS)


class TestItNeverPassesTheHarmfulAdvice:
    """These are not oversights to be added later. Each is excluded because it
    would make things worse on the card that prompted the request."""

    @pytest.mark.parametrize("flag", ["--async-offload", "--reserve-vram",
                                      "--force-non-blocking"])
    def test_the_excluded_flags_stay_excluded(self, tmp_path, flag):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True)
        assert flag not in got.flags

    def test_low_memory_alone_does_not_stream_layers(self, tmp_path):
        """--novram is a separate, louder choice with its own switch. Turning
        it on as a side effect of "use less memory" would make every job on a
        card that was coping suddenly crawl."""
        got = choose_low_memory(engine_tree(tmp_path), enabled=True)
        assert "--novram" not in got.flags

    def test_it_does_not_undo_a_safeguard(self, tmp_path):
        """The safeguards exist because of an observed crash. A memory setting
        that switched one back on would trade a slow job for a lost one."""
        got = choose_low_memory(engine_tree(tmp_path), enabled=True)
        opposites = {g.flag.replace("--disable-", "--") for g in AMD_SAFEGUARDS}
        assert not opposites.intersection(got.flags)


class TestItCannotStopComfyuiStarting:
    """--lowvram and --cache-none each belong to a mutually exclusive group.
    Passing two members of one makes argparse exit before the server starts."""

    def test_a_typed_vram_choice_is_left_alone(self, tmp_path):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True,
                                extra=["--novram"])
        assert "--lowvram" not in got.flags
        assert [o.flag for o in got.overridden] == ["--lowvram"]
        assert "--disable-smart-memory" in got.flags, "the rest still applies"

    def test_a_typed_cache_choice_is_left_alone(self, tmp_path):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True,
                                extra=["--cache-classic"])
        assert "--cache-none" not in got.flags
        assert [o.flag for o in got.overridden] == ["--cache-none"]

    def test_repeating_our_own_flag_is_not_doubled(self, tmp_path):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True,
                                extra=["--lowvram"])
        assert got.flags.count("--lowvram") == 0

    def test_every_group_member_is_recognised(self, tmp_path):
        """Missing one would let argparse see two members of a group."""
        engine = engine_tree(tmp_path)
        for flag in VRAM_GROUP:
            got = choose_low_memory(engine, enabled=True, extra=[flag])
            assert "--lowvram" not in got.flags, flag
        for flag in CACHE_GROUP:
            got = choose_low_memory(engine, enabled=True, extra=[flag])
            assert "--cache-none" not in got.flags, flag


class TestItIsRemembered:
    def test_off_until_switched_on(self, tmp_path):
        assert not read_low_memory(tmp_path)

    def test_it_survives_a_restart(self, tmp_path):
        write_low_memory(tmp_path, True)
        assert read_low_memory(tmp_path)

    def test_and_can_be_switched_back_off(self, tmp_path):
        write_low_memory(tmp_path, True)
        write_low_memory(tmp_path, False)
        assert not read_low_memory(tmp_path)

    def test_switching_off_twice_is_not_an_error(self, tmp_path):
        write_low_memory(tmp_path, False)
        write_low_memory(tmp_path, False)
        assert not read_low_memory(tmp_path)


class TestStreamingModelLayers:
    """The demand this was built for: not moving whole models off the card
    between steps, but streaming ONE model's weights in a block at a time, so a
    model larger than the card still runs.

    At v0.34.0 that is --novram, traced through comfy/model_management.py:
    args.novram sets VRAMState.NO_VRAM, NO_VRAM forces lowvram_model_memory to
    0.1, and that byte budget is what model_load passes to partially_load. An
    earlier version of this file asserted --novram must never be passed, on the
    grounds that it was too blunt. That was the wrong call and it is reversed.
    """

    def test_off_by_default(self, tmp_path):
        assert choose_layer_streaming(engine_tree(tmp_path), enabled=False).flags == []

    def test_on_it_passes_novram(self, tmp_path):
        got = choose_layer_streaming(engine_tree(tmp_path), enabled=True)
        assert got.flags == ["--novram"]

    def test_an_engine_without_it_says_so(self, tmp_path):
        engine = engine_tree(tmp_path, 'parser.add_argument("--listen")\n')
        got = choose_layer_streaming(engine, enabled=True)
        assert got.flags == []
        assert [o.flag for o in got.unavailable] == ["--novram"]

    @pytest.mark.parametrize("flag", VRAM_GROUP)
    def test_a_memory_mode_the_user_typed_wins(self, tmp_path, flag):
        """Every member of the group, because passing two stops ComfyUI
        starting -- and that is a worse outcome than not streaming."""
        got = choose_layer_streaming(engine_tree(tmp_path), enabled=True, extra=[flag])
        assert got.flags == []
        assert [o.flag for o in got.overridden] == ["--novram"]

    def test_it_is_remembered(self, tmp_path):
        assert not read_layer_streaming(tmp_path)
        write_layer_streaming(tmp_path, True)
        assert read_layer_streaming(tmp_path)
        write_layer_streaming(tmp_path, False)
        assert not read_layer_streaming(tmp_path)


class TestTheTwoSwitchesTogether:
    """--novram and --lowvram are members of one argparse group. Both switched
    on must not produce both flags, or ComfyUI exits before the server starts."""

    def both(self, tmp_path, extra=()):
        engine = engine_tree(tmp_path)
        streaming = choose_layer_streaming(engine, enabled=True, extra=list(extra))
        thrift = choose_low_memory(engine, enabled=True,
                                   extra=[*extra, *streaming.flags])
        return [*streaming.flags, *thrift.flags]

    def test_streaming_wins_and_lowvram_stands_down(self, tmp_path):
        flags = self.both(tmp_path)
        assert "--novram" in flags
        assert "--lowvram" not in flags

    def test_at_most_one_member_of_the_vram_group_is_ever_passed(self, tmp_path):
        flags = self.both(tmp_path)
        assert len([f for f in flags if f in VRAM_GROUP]) <= 1

    def test_the_rest_of_low_memory_mode_still_applies(self, tmp_path):
        """Standing --lowvram down must not throw away the other two."""
        flags = self.both(tmp_path)
        assert "--disable-smart-memory" in flags
        assert "--cache-none" in flags

    def test_a_typed_choice_still_beats_both(self, tmp_path):
        flags = self.both(tmp_path, extra=["--highvram"])
        assert not [f for f in flags if f in VRAM_GROUP]
        assert "--disable-smart-memory" in flags
