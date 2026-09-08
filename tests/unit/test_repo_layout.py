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


# --- Dependencies: one source of truth ---------------------------------------
#
# Three CI failures on main came from the same mistake: a new third-party
# import whose package was in pyproject.toml (so `pip install .` in the build
# workflows had it) but not in ci.yml's hand-typed list. pyyaml, then PySide6,
# then httpx. Each was found by a red main rather than by a test.
#
# Both halves are now asserted here: that everything imported is declared, and
# that CI installs what is declared rather than retyping it.

# Import name -> distribution name, where they differ. Kept explicit rather
# than resolved through importlib.metadata, so the check works from a bare
# checkout with nothing installed.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "PySide6": "pyside6-essentials",
}


def declared_dependencies() -> set[str]:
    """Distribution names from pyproject's [project].dependencies, normalised
    per PEP 503 and stripped of version specifiers and extras."""
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    names = set()
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~;\[\s]", spec, maxsplit=1)[0]
        names.add(re.sub(r"[-_.]+", "-", name).lower())
    return names


def third_party_imports_under(package: Path) -> dict[str, set[str]]:
    """Top-level module name -> the files that import it, for every import in
    `package` that is neither stdlib nor first-party."""
    import ast
    import sys

    found: dict[str, set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which is first-party by
                # definition and has no module name to resolve.
                roots = [node.module.split(".")[0]] if node.level == 0 and node.module else []
            else:
                continue
            for root in roots:
                if root in sys.stdlib_module_names or root == "toolshed":
                    continue
                found.setdefault(root, set()).add(str(path.relative_to(REPO)))
    return found


def test_every_third_party_import_is_a_declared_dependency():
    """The check that would have caught httpx before it reached main.

    An import the project does not declare works fine on the machine that has
    it installed and fails everywhere else -- which is precisely how this
    failed three times, each time discovered by CI going red after a merge.
    """
    declared = declared_dependencies()
    undeclared = {
        module: sorted(files)
        for module, files in third_party_imports_under(REPO / "toolshed").items()
        if IMPORT_TO_DISTRIBUTION.get(module, module).replace("_", "-").lower() not in declared
    }
    assert not undeclared, (
        f"imported but not in pyproject.toml [project].dependencies: {undeclared}. "
        f"Add the distribution there -- do not add it to a workflow -- or the "
        f"frozen build and CI will disagree about what the app needs."
    )


def test_ci_installs_the_project_rather_than_naming_packages():
    """ci.yml must get its dependencies from pyproject.toml.

    Naming packages in the workflow creates a second, silently-drifting list.
    That list is what was missing pyyaml, then PySide6, then httpx.
    """
    import yaml

    doc = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    runs = [
        step.get("run", "")
        for job in doc["jobs"].values()
        for step in job.get("steps", [])
    ]
    installs = [
        line.strip()
        for run in runs
        for line in run.splitlines()
        if re.search(r"\bpip install\b", line) and not line.strip().startswith("#")
    ]
    assert installs, "ci.yml installs nothing"

    assert any(re.search(r'pip install .*-e ["\']?\.', line) for line in installs), \
        "ci.yml must install the project itself, e.g. `pip install -e \".[dev]\"`"

    # Anything that is neither the project, an extra of it, nor pip itself is a
    # hand-typed package name and therefore a second source of truth.
    allowed = re.compile(r'pip install\s+(--upgrade\s+pip|(--upgrade\s+)?-e\s+["\']?\.)')
    offenders = [line for line in installs if not allowed.search(line)]
    assert not offenders, (
        f"ci.yml names packages by hand: {offenders}. "
        f"Declare them in pyproject.toml instead; CI installs from there."
    )


def test_the_engine_flags_were_verified_against_the_engine_we_ship():
    """A bumped ComfyUI must not silently invalidate the safeguard flags.

    The launcher passes --disable-async-offload and --disable-pinned-memory to
    stop a graphics memory fault that killed two finished jobs. Both spellings
    were read from ComfyUI's own cli_args.py at one tag. If a later version
    renames either, the launcher drops it -- safely, because an unknown flag
    would stop the engine starting, but the crash comes back.

    So bumping the engine fails here until someone re-reads the flags at the
    new tag and moves FLAGS_VERIFIED_AGAINST with it.
    """
    from toolshed.exec.engine import FLAGS_VERIFIED_AGAINST
    from toolshed.planner.plan import ENGINE_TAG

    assert FLAGS_VERIFIED_AGAINST == ENGINE_TAG, (
        f"ComfyUI moved to {ENGINE_TAG} but the engine flags were last read at "
        f"{FLAGS_VERIFIED_AGAINST}. Re-read comfy/cli_args.py at {ENGINE_TAG}, "
        f"confirm --disable-async-offload and --disable-pinned-memory are still "
        f"spelt that way, then update FLAGS_VERIFIED_AGAINST."
    )


def test_the_workflows_come_from_the_templates_the_engine_ships():
    """The graphs we inject must match the node schemas the engine has.

    They did not have to be pinned to notice this: a 3D run reported
    "Required input is missing: qef" on RemeshMesh, a node whose inputs vary by
    a DynamicCombo branch. The workflow turned out to be identical to the
    pinned template, so that particular failure was upstream's -- but the file
    that built it was reading `main` while the engine sat at a fixed tag, and a
    graph fetched from a moving branch is one upstream edit away from exactly
    that error, with nothing anywhere connecting the two.

    derive_catalog.py already refuses a branch for recipes. This is the same
    rule for the workflows those recipes ship.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    try:
        from build_workflows import TEMPLATES_REF, TEMPLATES_VERIFIED_FOR_ENGINE
    finally:
        sys.path.pop(0)
    from toolshed.planner.plan import ENGINE_TAG

    assert not TEMPLATES_REF.endswith("main"), "a branch lets the graph move under a pinned engine"
    assert TEMPLATES_VERIFIED_FOR_ENGINE == ENGINE_TAG, (
        f"ComfyUI moved to {ENGINE_TAG} but the workflow templates were last "
        f"pinned for {TEMPLATES_VERIFIED_FOR_ENGINE}. Read "
        f"comfyui-workflow-templates in requirements.txt at {ENGINE_TAG}, set "
        f"TEMPLATES_REF to that version, rerun tools/build_workflows.py, then "
        f"move TEMPLATES_VERIFIED_FOR_ENGINE."
    )
