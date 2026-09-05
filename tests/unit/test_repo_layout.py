"""Guards against the bug that broke the first build.

`toolshed/` was created as a set of empty directories and committed. Git does
not track empty directories, so on a fresh checkout the package simply was not
there: CI's `ruff check tools tests toolshed` failed with E902, and the real
problem -- "the application does not exist" -- arrived disguised as a style
error.

These tests are cheap and they fail loudly for the whole class of mistake:
a path that exists on the developer's disk but not in a clone.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    return set(out.stdout.split())


def test_no_empty_directories_anywhere():
    """An empty directory is invisible to git, so it is a landmine for anything
    that references it by path."""
    empty = [
        p.relative_to(REPO)
        for p in REPO.rglob("*")
        if p.is_dir()
        and not any(part in {".git", ".cache", "build", "dist", "__pycache__", ".ruff_cache",
                             ".pytest_cache", "stage", "out", ".venv"}
                    for part in p.relative_to(REPO).parts)
        and not any(p.iterdir())
    ]
    assert not empty, f"empty directories will not survive a clone: {empty}"


def test_the_package_is_actually_tracked():
    """The specific failure, asserted directly."""
    tracked = tracked_files()
    for required in ("toolshed/__init__.py", "toolshed/__main__.py",
                     "toolshed/_streams.py", "toolshed/resources.py"):
        assert required in tracked, f"{required} is not committed"


def test_console_script_target_exists():
    """pyproject declares `toolshed = "toolshed.__main__:main"`. If that entry
    point ever stops resolving, `pip install .` produces a broken command."""
    from toolshed.__main__ import main

    assert callable(main)


@pytest.mark.parametrize("spec", ["packaging/linux/toolshed.spec",
                                  "packaging/windows/toolshed.spec"])
def test_spec_files_only_reference_paths_that_exist(spec):
    """Every `REPO / "..."` path in a spec must resolve, or the build fails on a
    clean checkout while working perfectly for whoever wrote it."""
    text = (REPO / spec).read_text(encoding="utf-8")
    chains = re.findall(r'REPO(?:\s*/\s*"[^"]+")+', text)
    assert chains, f"{spec} references no repository paths at all; has it been gutted?"
    for chain in chains:
        parts = re.findall(r'"([^"]+)"', chain)
        target = REPO.joinpath(*parts)
        # build/version_info.txt is generated at build time and guarded by
        # is_file() in the spec itself.
        if parts[:1] == ["build"]:
            continue
        assert target.exists(), f"{spec} references {'/'.join(parts)}, which does not exist"


def test_data_dirs_bundled_by_the_specs_exist_and_are_populated():
    """The selftest asserts on these inside the frozen bundle, so an empty one
    would turn into a build that passes and a binary that fails."""
    import sys

    sys.path.insert(0, str(REPO / "packaging"))
    from _spec_common import DATA_DIRS

    for name in DATA_DIRS:
        directory = REPO / name
        assert directory.is_dir(), f"{name}/ is bundled by the specs but does not exist"
        files = [f for f in directory.rglob("*") if f.is_file()]
        assert files, f"{name}/ is empty; the bundle would ship nothing"
