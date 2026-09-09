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
    choose_low_memory,
    read_low_memory,
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
                                      "--force-non-blocking", "--novram"])
    def test_the_excluded_flags_stay_excluded(self, tmp_path, flag):
        got = choose_low_memory(engine_tree(tmp_path), enabled=True)
        assert flag not in got.flags

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
