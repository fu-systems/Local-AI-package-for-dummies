"""Uninstalling must leave the models and nothing else.

The old uninstaller removed the app and deliberately left the entire data root:
the private Python workspace, the ComfyUI engine and its database, injected
workflows, the manifest, logs, part-finished downloads. So a reinstall landed
on top of a half-finished one and inherited its problems -- a venv from an
interrupted run, a manifest describing files that were no longer there, an
engine tree from another version. Reinstalling to get a clean slate did not
give you one.

These tests run the **actual shipped script** against a data root built to look
like a real leftover install, rather than a Python reimplementation of it that
could agree with itself while the shipped file did something else. The script
is what gets packaged, so the script is what is tested.

The Windows half is Inno Pascal and cannot be run here at all. It is checked
for agreement on the two things that must match -- what is kept, and where the
data root is -- and is otherwise unverified, which is stated rather than
glossed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "packaging" / "linux" / "uninstall.sh"
ISS = REPO / "packaging" / "windows" / "toolshed.iss"

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the shipped uninstaller is a POSIX shell script")


def make_install(home: Path) -> Path:
    """A data root and app install with something from every stage in it."""
    root = home / "Toolshed"

    # The models, which must survive.
    for folder, files in (
        ("checkpoints", ["sd_xl_base_1.0.safetensors"]),
        ("diffusion_models", ["trellis_2_int8_convrot.safetensors"]),
        ("vae", ["trellis_2_texture_vae_bf16.safetensors"]),
        ("clip_vision", []),          # an empty one: the structure matters too
        ("loras", []),
    ):
        directory = root / "models" / folder
        directory.mkdir(parents=True)
        for name in files:
            (directory / name).write_bytes(b"weights" * 100)

    # Everything else, which must not.
    leftovers = [
        "runtime/venv/bin/python",
        "runtime/venv/pyvenv.cfg",
        "runtime/uv/uv",
        "runtime/uv-cache/some-wheel.whl",
        "runtime/python/cpython-3.12/bin/python3.12",
        "engine/comfyui/main.py",
        "engine/comfyui/requirements.txt",
        "comfy/user/default/workflows/Toolshed/image/01 Text to picture.json",
        "comfy/user/comfyui.db",
        "comfy/input/photo.png",
        "comfy/temp/scratch.bin",
        "comfy/custom_nodes/.gitkeep",
        "state/manifest.json",
        "state/logs/comfyui.log",
        "state/engine-flags.txt",
        "state/downloads/half-a-model.safetensors.part",
        "output/picture.png",
        "input/reference.png",
    ]
    for rel in leftovers:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("leftover")

    # Dotfiles at the top level: a shell glob would walk straight past these.
    (root / ".hidden-state").write_text("x")
    (root / ".cache").mkdir()
    (root / ".cache" / "thing").write_text("x")

    # The app itself.
    for rel in (
        ".local/share/toolshed/app/toolshed",
        ".local/share/toolshed/app/_internal/base_library.zip",
        ".local/bin/toolshed",
        ".local/share/applications/toolshed.desktop",
        ".local/share/icons/hicolor/256x256/apps/toolshed.png",
    ):
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("app")

    # Something of the user's that has nothing to do with us.
    (home / "Documents").mkdir()
    (home / "Documents" / "taxes.ods").write_text("mine")
    (home / ".bashrc").write_text("mine")
    return root


def run_uninstall(home: Path, root: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(home)}
    if root is not None:
        env["TOOLSHED_ROOT"] = str(root)
    else:
        env.pop("TOOLSHED_ROOT", None)
    return subprocess.run(["sh", str(SCRIPT)], capture_output=True, text=True,
                          timeout=120, env=env)


class TestItLeavesTheModelsAndNothingElse:
    @pytest.fixture
    def done(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        root = make_install(home)
        result = run_uninstall(home)
        assert result.returncode == 0, result.stderr
        return home, root, result

    def test_the_model_files_are_untouched(self, done):
        _, root, _ = done
        assert (root / "models" / "checkpoints" / "sd_xl_base_1.0.safetensors").is_file()
        assert (root / "models" / "diffusion_models"
                / "trellis_2_int8_convrot.safetensors").is_file()
        assert (root / "models" / "vae"
                / "trellis_2_texture_vae_bf16.safetensors").read_bytes() == b"weights" * 100

    def test_the_empty_model_folders_are_kept_too(self, done):
        """"except the empty file structure for the models" -- the skeleton is
        part of what survives, not just the files in it."""
        _, root, _ = done
        assert (root / "models" / "clip_vision").is_dir()
        assert (root / "models" / "loras").is_dir()

    def test_models_is_the_only_thing_left_in_the_data_root(self, done):
        _, root, _ = done
        assert sorted(p.name for p in root.iterdir()) == ["models"]

    @pytest.mark.parametrize("gone", [
        "runtime", "engine", "comfy", "state", "output", "input",
    ])
    def test_every_other_folder_is_gone(self, done, gone):
        _, root, _ = done
        assert not (root / gone).exists(), f"{gone} survived the uninstall"

    def test_dotfiles_and_dot_directories_go_too(self, done):
        """A shell glob skips these, which is how "delete everything" quietly
        becomes "delete everything visible"."""
        _, root, _ = done
        assert not (root / ".hidden-state").exists()
        assert not (root / ".cache").exists()

    def test_part_finished_downloads_go(self, done):
        _, root, _ = done
        assert not (root / "state" / "downloads").exists()

    def test_the_app_is_removed(self, done):
        home, _, _ = done
        assert not (home / ".local" / "share" / "toolshed").exists()
        assert not (home / ".local" / "bin" / "toolshed").exists()
        assert not (home / ".local" / "share" / "applications"
                    / "toolshed.desktop").exists()
        assert not (home / ".local" / "share" / "icons" / "hicolor" / "256x256"
                    / "apps" / "toolshed.png").exists()

    def test_it_says_what_it_kept(self, done):
        _, _, result = done
        assert "models" in result.stdout
        assert "Nothing else was left" in result.stdout

    def test_nothing_of_the_users_is_touched(self, done):
        home, _, _ = done
        assert (home / "Documents" / "taxes.ods").read_text() == "mine"
        assert (home / ".bashrc").read_text() == "mine"


class TestWhenThereAreNoModels:
    def test_the_whole_data_root_goes(self, tmp_path):
        """Nothing worth keeping means nothing is kept, including the folder."""
        home = tmp_path / "home"
        (home / "Toolshed" / "runtime" / "venv").mkdir(parents=True)
        (home / "Toolshed" / "state").mkdir(parents=True)

        result = run_uninstall(home)
        assert result.returncode == 0, result.stderr
        assert not (home / "Toolshed").exists()
        assert "Nothing was left anywhere" in result.stdout

    def test_it_is_fine_with_nothing_installed_at_all(self, tmp_path):
        """Running it twice, or on a machine that never had it, must not fail."""
        home = tmp_path / "home"
        home.mkdir()
        result = run_uninstall(home)
        assert result.returncode == 0, result.stderr


class TestItRefusesToDeleteTheWrongThing:
    """This script removes directory trees. A wrong root here is not a bug, it
    is somebody's home directory."""

    @pytest.mark.parametrize("bad", ["/", ".", "/*"])
    def test_it_refuses_an_obviously_wrong_root(self, tmp_path, bad):
        home = tmp_path / "home"
        home.mkdir()
        (home / "keepme.txt").write_text("mine")

        result = run_uninstall(home, root=Path(bad))
        assert result.returncode == 1, f"it accepted {bad!r} as a data folder"
        assert "Refusing" in result.stdout
        assert (home / "keepme.txt").is_file()

    def test_an_empty_override_falls_back_rather_than_deleting_everything(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        (home / "keepme.txt").write_text("mine")

        result = subprocess.run(
            ["sh", str(SCRIPT)], capture_output=True, text=True, timeout=60,
            env={**os.environ, "HOME": str(home), "TOOLSHED_ROOT": ""})
        assert result.returncode == 0, result.stderr
        assert (home / "keepme.txt").is_file()

    def test_it_refuses_the_home_directory_itself(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        (home / "keepme.txt").write_text("mine")

        result = run_uninstall(home, root=home)
        assert result.returncode == 1
        assert "Refusing" in result.stdout
        assert (home / "keepme.txt").is_file()

    def test_it_refuses_the_filesystem_root(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        result = run_uninstall(home, root=Path("/"))
        assert result.returncode == 1
        assert "Refusing" in result.stdout


class TestTheTwoUninstallersAgree:
    """One is shell, one is Inno Pascal, and only the shell one can be run
    here. They must at least agree on the two facts that define the behaviour."""

    def test_both_keep_the_same_folder(self):
        assert 'KEEP="models"' in SCRIPT.read_text()
        assert "Keep := 'models';" in ISS.read_text()

    def test_both_honour_the_same_override(self):
        assert "TOOLSHED_ROOT" in SCRIPT.read_text()
        assert "GetEnv('TOOLSHED_ROOT')" in ISS.read_text()

    def test_both_find_the_data_root_where_the_app_puts_it(self):
        """toolshed/ui/ready.py decides this: $HOME/Toolshed on Linux,
        %SYSTEMDRIVE%\\Toolshed on Windows. An uninstaller looking somewhere
        else would report success having deleted nothing."""
        from toolshed.ui.ready import default_data_root

        assert default_data_root().name == "Toolshed"
        assert "${HOME}/Toolshed" in SCRIPT.read_text()
        assert "{%SYSTEMDRIVE|C:}\\Toolshed" in ISS.read_text()

    def test_the_windows_one_still_guards_against_a_bare_path(self):
        assert "Pos('Toolshed', Root) = 0" in ISS.read_text()

    def test_neither_claims_the_models_were_deleted(self):
        """The old pair told the user their models were kept *and* that the
        whole data folder was left behind. Only the first is true now."""
        text = SCRIPT.read_text() + ISS.read_text()
        assert "folder you chose during setup" not in text


class TestTheScriptIsShipped:
    def test_the_build_packages_it(self):
        workflow = (REPO / ".github" / "workflows" / "build-linux.yml").read_text()
        assert "packaging/linux/uninstall.sh" in workflow

    def test_it_is_valid_shell(self):
        assert subprocess.run(["sh", "-n", str(SCRIPT)],
                              capture_output=True).returncode == 0


class TestTheUninstallerIsActuallyOnTheMachine:
    """install.sh used to delete uninstall.sh as it copied the app in.

    So once the tarball was gone there was no uninstaller anywhere, and
    "uninstall" became "delete some folders and hope" -- which is how a
    half-removed install ends up being reinstalled on top of.
    """

    def test_install_keeps_the_uninstaller(self):
        text = (REPO / "packaging" / "linux" / "install.sh").read_text()
        assert 'rm -f "$APP_DIR/install.sh" "$APP_DIR/uninstall.sh"' not in text
        assert '"$APP_DIR/uninstall.sh"' in text

    def test_install_leaves_a_command_to_run_it(self):
        text = (REPO / "packaging" / "linux" / "install.sh").read_text()
        assert "toolshed-uninstall" in text

    def test_the_uninstaller_removes_that_command_too(self):
        assert "${BIN_DIR}/toolshed-uninstall" in SCRIPT.read_text()

    def test_it_runs_from_a_copy_before_deleting_its_own_directory(self):
        """A POSIX shell reads a script as it goes, so deleting the app
        directory while running from inside it can leave the rest of the file
        unread -- the exact half-done state this script exists to clean up.

        This asserts the mechanism is present, not that it is load-bearing:
        whether the hazard bites depends on the shell's read buffering and the
        size of the script, and removing the relocation does not fail any test
        here. Kept because the hazard is real and the guard costs nothing, but
        recorded as unproven rather than left looking covered.
        """
        assert "TOOLSHED_UNINSTALL_RELOCATED" in SCRIPT.read_text()

    def test_it_finishes_even_when_run_from_inside_the_app_directory(self, tmp_path):
        """The real arrangement: the script sits in the directory it deletes."""
        home = tmp_path / "home"
        home.mkdir()
        root = make_install(home)
        installed = home / ".local" / "share" / "toolshed" / "app" / "uninstall.sh"
        installed.write_text(SCRIPT.read_text())
        installed.chmod(0o755)

        result = subprocess.run(["sh", str(installed)], capture_output=True, text=True,
                                timeout=120, env={**os.environ, "HOME": str(home)})
        assert result.returncode == 0, result.stderr
        # The tail of the script ran: it printed its closing lines and the data
        # root was purged, not left half-done.
        assert "Toolshed has been removed." in result.stdout
        assert sorted(p.name for p in root.iterdir()) == ["models"]
        assert not (home / ".local" / "share" / "toolshed").exists()
