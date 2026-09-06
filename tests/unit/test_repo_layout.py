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


def test_excludes_contain_no_setuptools_alias_targets():
    """PyInstaller aliases setuptools' vendored copies onto their bare names on
    Python 3.12+, and `alias_module` raises if the name is already an excluded
    node. Putting any packaging tool in `excludes` therefore breaks the build
    the moment the module graph reaches it -- which happened on Windows and not
    on Linux, so it cost a full CI round to find.

    This test fails at the point the mistake is made, rather than eleven
    minutes into a Windows build.
    """
    import sys

    sys.path.insert(0, str(REPO / "packaging"))
    from _spec_common import FORBIDDEN_EXCLUDES, OTHER_EXCLUDES, QT_EXCLUDES

    offenders = FORBIDDEN_EXCLUDES.intersection(OTHER_EXCLUDES + QT_EXCLUDES)
    assert not offenders, (
        f"{sorted(offenders)} must not be excluded: PyInstaller aliases the "
        f"setuptools-vendored copies onto these names and the build dies with "
        f'ValueError: Target module "..." already imported as ExcludedModule'
    )


BUILD_WORKFLOWS = ["build-linux", "build-windows"]


@pytest.mark.parametrize("name", BUILD_WORKFLOWS)
def test_builds_are_manual_only(name):
    """Builds must never start on their own.

    An unbounded wait in a build step, triggered automatically on merge, burned
    hours of runner time before anyone could react. The owner starts builds and
    nothing else does -- no push, no pull_request, no tag, no cron, no agent.
    """
    import yaml

    doc = yaml.safe_load((REPO / ".github" / "workflows" / f"{name}.yml").read_text())
    # PyYAML parses the bare key `on` as the boolean True.
    triggers = doc[True] if True in doc else doc["on"]
    assert list(triggers) == ["workflow_dispatch"], (
        f"{name}.yml must be workflow_dispatch only, found {list(triggers)}. "
        f"See the note at the top of the file, README.md and CLAUDE.md."
    )


@pytest.mark.parametrize("name", ["ci", *BUILD_WORKFLOWS])
def test_every_job_has_a_timeout(name):
    """GitHub's default job timeout is 360 minutes. Without an explicit one, a
    single hung step sits there for six hours burning runner minutes."""
    import yaml

    doc = yaml.safe_load((REPO / ".github" / "workflows" / f"{name}.yml").read_text())
    missing = [j for j, cfg in doc["jobs"].items() if not cfg.get("timeout-minutes")]
    assert not missing, f"{name}.yml jobs without timeout-minutes: {missing}"


def test_no_unbounded_process_wait_on_windows():
    """`Start-Process -Wait` has no deadline. On a headless runner a GUI process
    that shows a modal dialog waits forever; that is what hung the build."""
    text = (REPO / ".github" / "workflows" / "build-windows.yml").read_text()
    offenders = [
        line.strip()
        for line in text.splitlines()
        if "Start-Process" in line and "-Wait" in line and not line.strip().startswith("#")
    ]
    assert not offenders, f"unbounded waits: {offenders}"


def test_the_no_agent_builds_rule_is_written_down():
    """The rule has to survive this conversation, so it lives in the files a
    human reads and the file an agent loads."""
    needle = "NO AI AGENT MAY EVER TRIGGER A BUILD"
    for rel in ("README.md", "CLAUDE.md"):
        assert needle in (REPO / rel).read_text(), f"{rel} is missing the rule"
    for name in BUILD_WORKFLOWS:
        header = (REPO / ".github" / "workflows" / f"{name}.yml").read_text()[:1200]
        assert "NO AI AGENT MAY EVER TRIGGER A BUILD" in header.upper(), \
            f"{name}.yml header is missing the rule"
