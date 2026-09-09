"""Tests for ``toolshed --selftest``, the check every build job runs.

These exist because of a build that failed while the whole test suite was
green. `--selftest` is the last gate before a binary ships, and nothing ran it
except the build itself -- so its checks could rot against the catalogue for as
long as nobody dispatched a build, and the first sign was a red build on main
rather than a red test on the branch.

Two of its checks had drifted apart in exactly that way. The catalogue loader
learned that "no download size" and "no recipe file" are different states; the
selftest still read the first and reported the second. The first deliberately
unfrozen pack therefore failed the build with a message blaming a recipe that
was present in the bundle.

So the point of this file is narrow and worth keeping narrow: run the real
thing against the real catalogue, and pin the distinctions its failures are
supposed to draw.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")

from toolshed.__main__ import selftest  # noqa: E402
from toolshed.catalog.packs import Pack  # noqa: E402


def pack(pack_id: str, **kw) -> Pack:
    return Pack(
        id=pack_id,
        modality="Pictures",
        name=pack_id,
        blurb="a blurb",
        recipe="some_recipe",
        vram_gb_min=8,
        licence="Some licence",
        download_bytes=kw.get("download_bytes", 7_000_000_000),
        recipe_found=kw.get("recipe_found", True),
    )


class TestAgainstTheRealCatalogue:
    def test_the_shipped_catalogue_passes(self, capsys):
        """The regression itself. This failed on main while every other test
        passed, and only a dispatched build said so."""
        assert selftest() == 0, capsys.readouterr().out

    def test_an_unfrozen_pack_is_reported_without_failing(self, capsys):
        """It has to appear in the report -- a pack that cannot be installed is
        something the person reading a build log needs to see -- but appearing
        is not the same as failing."""
        selftest()
        out = capsys.readouterr().out
        assert "awaiting freeze" in out
        assert "SELFTEST OK" in out

    def test_the_report_file_matches_stdout(self, tmp_path, capsys):
        """A windowed Windows build has no stdout, so the file is the only
        channel CI can read there. If the two ever disagreed, the Windows job
        would be judging a different report from the Linux one."""
        report = tmp_path / "selftest.txt"
        selftest(report)
        assert report.read_text(encoding="utf-8") == capsys.readouterr().out


class TestWhatItRefuses:
    """The distinctions the failure messages have to keep straight."""

    def test_a_missing_recipe_file_fails_and_is_named(self, monkeypatch, capsys):
        """A typo in `recipe:` is a mistake and must stop the build."""
        # selftest imports load_packs inside the function, so the name that
        # matters is the one in its own module, not a copy in __main__.
        monkeypatch.setattr("toolshed.catalog.packs.load_packs",
                            lambda: (pack("image.good"), pack("image.typo", recipe_found=False)))
        assert selftest() == 1
        out = capsys.readouterr().out
        assert "no such recipe" in out
        assert "image.typo" in out, "a failure has to say which pack"
        assert "image.good" not in out.split("SELFTEST FAIL")[-1]

    def test_an_unfrozen_pack_alone_does_not_fail(self, monkeypatch, capsys):
        """The bug this file was written for: unfrozen is a legitimate state
        and must not be reported as a missing recipe."""
        monkeypatch.setattr("toolshed.catalog.packs.load_packs",
                            lambda: (pack("image.frozen"),
                                     pack("image.pending", download_bytes=None)))
        assert selftest() == 0
        out = capsys.readouterr().out
        assert "image.pending" in out
        assert "no such recipe" not in out

    def test_a_catalogue_with_no_sizes_at_all_fails(self, monkeypatch, capsys):
        """The half of the old check worth keeping. Recipes bundled empty or
        truncated would leave the files present and every size gone, which
        `recipe_found` cannot see -- and a build where nothing is installable
        is not a usable binary."""
        monkeypatch.setattr("toolshed.catalog.packs.load_packs",
                            lambda: (pack("image.a", download_bytes=None),
                                     pack("image.b", download_bytes=None)))
        assert selftest() == 1
        assert "bundled recipes are unusable" in capsys.readouterr().out
