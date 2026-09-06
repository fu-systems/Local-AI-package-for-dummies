"""The GPL guard must catch bundling, and must not catch prose.

ComfyUI is GPL-3.0-or-later. Toolshed downloads it and drives it as a separate
process, and bundling it would redistribute GPL code -- so CI has a job that
refuses the merge if it finds ComfyUI in anything under packaging/.

That job used to be a bare case-insensitive grep for the word, and it failed on
this, in the uninstaller:

    # ComfyUI engine, its user directory and database, injected workflows, the

A comment describing what gets *deleted* is the opposite of bundling. The check
now ignores comments -- which loosens a safety check, so the point of this file
is to show it still catches the thing it is for.

The check lives in tools/check_gpl_boundary.sh rather than inline in ci.yml
precisely so these tests can run it. A safety check nobody can exercise is a
wish, and this one had been unexercised long enough to pass vacuously once
already (a missing toolshed/ made grep exit 2, the `if` read that as false).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "tools" / "check_gpl_boundary.sh"

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shell script")


def run(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", str(CHECK)], cwd=cwd, capture_output=True,
                          text=True, timeout=60)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal project the check will pass."""
    (tmp_path / "toolshed").mkdir()
    (tmp_path / "toolshed" / "__init__.py").write_text("__version__ = '0'\n")
    (tmp_path / "packaging" / "linux").mkdir(parents=True)
    (tmp_path / "packaging" / "linux" / "install.sh").write_text("#!/bin/sh\ntrue\n")
    return tmp_path


class TestItPassesWhatItShould:
    def test_the_real_repository_passes(self):
        result = run(REPO)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "GPL boundary intact" in result.stdout

    def test_a_clean_tree_passes(self, tree):
        assert run(tree).returncode == 0

    def test_a_shell_comment_naming_comfyui_is_allowed(self, tree):
        """The exact line that broke the merge."""
        (tree / "packaging" / "linux" / "uninstall.sh").write_text(
            "#!/bin/sh\n"
            "# ComfyUI engine, its user directory and database, injected workflows\n"
            "rm -rf \"$ROOT/engine\"\n")
        result = run(tree)
        assert result.returncode == 0, result.stdout

    def test_an_indented_comment_is_allowed_too(self, tree):
        (tree / "packaging" / "linux" / "install.sh").write_text(
            "#!/bin/sh\n"
            "if true; then\n"
            "    # we never bundle ComfyUI, we download it\n"
            "    true\n"
            "fi\n")
        assert run(tree).returncode == 0

    def test_an_inno_setup_comment_is_allowed(self, tree):
        (tree / "packaging" / "windows").mkdir()
        (tree / "packaging" / "windows" / "toolshed.iss").write_text(
            "[Code]\n"
            "// the ComfyUI engine lives outside {app} and is removed separately\n"
            "; ComfyUI is never shipped in this installer\n")
        assert run(tree).returncode == 0


class TestItStillCatchesBundling:
    """Each of these is a real way ComfyUI could end up redistributed."""

    def test_a_pyinstaller_spec_that_bundles_the_engine(self, tree):
        (tree / "packaging" / "linux" / "toolshed.spec").write_text(
            "a = Analysis([], datas=[('engine/comfyui', 'comfyui')])\n")
        result = run(tree)
        assert result.returncode == 1
        assert "must never be bundled" in result.stdout

    def test_a_spec_tree_of_the_engine(self, tree):
        (tree / "packaging" / "linux" / "toolshed.spec").write_text(
            "trees = [Tree('engine/ComfyUI', prefix='comfyui')]\n")
        assert run(tree).returncode == 1

    def test_an_inno_files_entry(self, tree):
        (tree / "packaging" / "windows").mkdir()
        (tree / "packaging" / "windows" / "toolshed.iss").write_text(
            "[Files]\n"
            'Source: "..\\\\engine\\\\comfyui\\\\*"; DestDir: "{app}\\\\comfyui"\n')
        assert run(tree).returncode == 1

    def test_an_inno_preprocessor_define_is_not_a_comment(self, tree):
        """'#' starts a preprocessor directive in Inno Setup, not a comment.
        Treating it as one would let a #define naming ComfyUI straight past."""
        (tree / "packaging" / "windows").mkdir()
        (tree / "packaging" / "windows" / "toolshed.iss").write_text(
            '#define ComfyUIDir "..\\\\engine\\\\comfyui"\n')
        result = run(tree)
        assert result.returncode == 1, "a #define naming ComfyUI was treated as a comment"

    def test_an_install_script_that_copies_the_engine_in(self, tree):
        (tree / "packaging" / "linux" / "install.sh").write_text(
            "#!/bin/sh\ncp -a comfyui \"$APP_DIR/\"\n")
        assert run(tree).returncode == 1

    def test_a_trailing_comment_does_not_launder_a_real_line(self, tree):
        """Only a line that *starts* as a comment is exempt."""
        (tree / "packaging" / "linux" / "install.sh").write_text(
            "#!/bin/sh\ncp -a comfyui \"$APP_DIR/\"   # harmless, honest\n")
        assert run(tree).returncode == 1

    def test_it_names_the_file_and_line_it_objected_to(self, tree):
        (tree / "packaging" / "linux" / "install.sh").write_text(
            "#!/bin/sh\ncp -a comfyui \"$APP_DIR/\"\n")
        result = run(tree)
        assert "install.sh" in result.stdout
        assert "cp -a comfyui" in result.stdout


class TestItStillCatchesImports:
    @pytest.mark.parametrize("line", [
        "import comfy",
        "from comfy import model_management",
        "import folder_paths",
        "from nodes import NODE_CLASS_MAPPINGS",
        "    import comfy.utils",
    ])
    def test_importing_comfyui_fails(self, tree, line):
        (tree / "toolshed" / "engine.py").write_text(f"{line}\n")
        result = run(tree)
        assert result.returncode == 1
        assert "must not import ComfyUI" in result.stdout

    def test_the_word_comfy_in_a_string_is_not_an_import(self, tree):
        """toolshed/exec/engine.py legitimately names the engine directory."""
        (tree / "toolshed" / "engine.py").write_text(
            'ENGINE = "engine/comfyui/main.py"\n'
            '# drives comfy as a subprocess, never imports it\n')
        assert run(tree).returncode == 0


class TestTheGuardCannotPassVacuously:
    def test_a_missing_package_is_a_failure_not_a_pass(self, tmp_path):
        """It passed having checked nothing once, because grep exits 2 on a
        missing directory and `if` reads that as false."""
        result = run(tmp_path)
        assert result.returncode == 1
        assert "toolshed/ package is missing" in result.stdout


class TestCiRunsThisExactScript:
    def test_the_workflow_calls_it(self):
        workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
        assert "tools/check_gpl_boundary.sh" in workflow

    def test_the_workflow_has_no_second_copy_of_the_check(self):
        """Two copies of a safety check drift, and the tested one is not
        necessarily the one that runs."""
        workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text()
        assert "grep -rniE 'comfyui'" not in workflow

    def test_it_is_committed_and_executable_shell(self):
        assert CHECK.is_file()
        assert subprocess.run(["sh", "-n", str(CHECK)],
                              capture_output=True).returncode == 0
