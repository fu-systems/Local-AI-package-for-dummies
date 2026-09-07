"""Regressions for the faults an adversarial audit of the whole suite confirmed.

Each class names the fault as a user would have met it. Most were found by
reading rather than by anyone hitting them, which is the point: an installer
that moves 40 GB and drives a graphics card gets one chance per person, and a
fault met on that one chance is met by someone who cannot diagnose it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from toolshed.exec import download as download_module
from toolshed.exec import proc, uvtool
from toolshed.exec.download import Cancelled, DownloadError, download_file
from toolshed.exec.manifest import Manifest
from toolshed.exec.proc import child_environment, run
from toolshed.exec.runner import InstallFailed, Runner
from toolshed.hw.detect import Gpu, HardwareReport
from toolshed.planner.plan import PENDING, Download, InstallPlan, Kind, Step, build_plan
from toolshed.planner.torchsel import choose_torch
from toolshed.ui.ready import windows_data_root

REPO = Path(__file__).resolve().parents[2]

# gfx1031 (RX 6700 XT) is a card ROCm does not list; the planner tells it to
# pass as gfx1030 with HSA_OVERRIDE_GFX_VERSION=10.3.0.
AMD_NEEDS_OVERRIDE = HardwareReport(
    os="linux", gpus=(Gpu("amd", "RX 6700 XT", 12288, gfx="gfx1031"),))


# A uv that cannot exist. A bare "uv" would be found on the PATH of any
# machine that happens to have one installed, and a test that quietly runs the
# real tool passes for a reason the CI runner does not share.
FAKE_UV = Path("/nonexistent/toolshed-test/uv")


def _ok_result(stdout: str = "") -> proc.Result:
    return proc.Result(0, stdout, "")


# ---------------------------------------------------------------------------
# Stop, during a step that is a subprocess


class TestStopWorksDuringASubprocess:
    """Stop used to be honoured between steps and inside downloads only. Pressed
    during the twenty-minute pip install it did nothing until pip finished."""

    def test_a_cancelled_child_is_killed_and_reported_as_cancelled(self):
        started = time.monotonic()
        with pytest.raises(Cancelled):
            run([sys.executable, "-c", "import time; time.sleep(60)"],
                timeout=60, should_cancel=lambda: True)
        assert time.monotonic() - started < 10, "Stop waited for the child to finish"

    def test_the_cancel_check_is_polled_not_read_once(self):
        """Pressed a moment *after* the child starts, not before."""
        polls = []

        def later():
            polls.append(1)
            return len(polls) > 3

        with pytest.raises(Cancelled):
            run([sys.executable, "-c", "import time; time.sleep(60)"],
                timeout=60, should_cancel=later)
        assert len(polls) > 3

    def test_every_uv_step_forwards_the_check(self, tmp_path, monkeypatch):
        seen = []

        def fake_run(cmd, **kw):
            seen.append(kw.get("should_cancel"))
            return _ok_result('{"version": "2.14.0+rocm7.2", "available": true, '
                              '"devices": 1, "name": "gpu"}')

        monkeypatch.setattr(uvtool, "run", fake_run)
        check = lambda: False  # noqa: E731
        uvtool.install_python(FAKE_UV, tmp_path, "3.12", should_cancel=check)
        uvtool.create_venv(FAKE_UV, tmp_path, "3.12", should_cancel=check)
        uvtool.pip_install(FAKE_UV, tmp_path, ["torch"], should_cancel=check)
        uvtool.verify_torch(tmp_path, "+rocm7.2", should_cancel=check)
        assert seen == [check] * 4


# ---------------------------------------------------------------------------
# What a child inherits from the frozen app


class TestChildrenDoNotInheritTheBundle:
    """PyInstaller points LD_LIBRARY_PATH at the bundle's own libraries. A
    child that is *another* Python -- the venv's, running PyTorch -- then loads
    our libstdc++ instead of its own: works from source, breaks in the tarball."""

    def test_the_original_is_restored_when_frozen(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/toolshed/_internal")
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/local/lib")
        env = child_environment()
        assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"
        assert "LD_LIBRARY_PATH_ORIG" not in env

    def test_it_is_removed_rather_than_left_pointing_at_us(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/toolshed/_internal")
        monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
        assert "LD_LIBRARY_PATH" not in child_environment()

    def test_from_source_nothing_is_touched(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/my/libs")
        assert child_environment()["LD_LIBRARY_PATH"] == "/my/libs"

    def test_run_actually_uses_it(self, monkeypatch):
        """The helper existing is not the fix; every child going through it is."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/toolshed/_internal")
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/local/lib")
        result = run([sys.executable, "-c",
                      "import os; print(os.environ.get('LD_LIBRARY_PATH', '<unset>'))"],
                     timeout=60)
        assert result.stdout.strip() == "/usr/local/lib"


# ---------------------------------------------------------------------------
# uv writing outside the data root


class TestUvStaysInsideTheDataRoot:
    def test_python_install_never_drops_a_launcher_in_local_bin(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(uvtool, "run", lambda cmd, **kw: (seen.append([str(c) for c in cmd]),
                                                              _ok_result())[1])
        uvtool.install_python(FAKE_UV, tmp_path, "3.12")
        assert "--no-bin" in seen[0], "uv python install writes a shim to ~/.local/bin without it"
        assert seen[0].index("--no-bin") < seen[0].index("3.12")


# ---------------------------------------------------------------------------
# The AMD override reaching everything that needs it


class TestTheHsaOverrideReachesEveryProbe:
    """gfx1031 needs HSA_OVERRIDE_GFX_VERSION before ROCm will drive it. The
    installer knew that, set it for nothing, and then probed torch without it
    -- "cannot see your graphics card" on exactly the machines the override
    exists for."""

    def test_the_plan_gives_the_verify_step_the_same_environment(self, tmp_path):
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        install = next(s for s in plan.steps if s.kind == Kind.INSTALL_TORCH)
        verify = next(s for s in plan.steps if s.kind == Kind.VERIFY_TORCH)
        assert install.payload["env"].get("HSA_OVERRIDE_GFX_VERSION")
        assert verify.payload["env"] == install.payload["env"]

    def test_verify_torch_runs_the_probe_under_it(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, **kw):
            seen["env"] = kw.get("env")
            return _ok_result('{"version": "2.14.0+rocm7.2", "cuda": null, "hip": "7.2",'
                              '"available": true, "devices": 1, "name": "AMD Radeon"}')

        monkeypatch.setattr(uvtool, "run", fake_run)
        uvtool.verify_torch(tmp_path, "+rocm7.2", env={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        assert seen["env"] == {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}

    def test_the_runner_records_it_for_the_launcher(self, tmp_path, monkeypatch):
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        torch_steps = tuple(s for s in plan.steps
                            if s.kind in (Kind.INSTALL_TORCH, Kind.VERIFY_TORCH))
        plan = InstallPlan(torch_steps, plan.torch, tmp_path, ())
        expected = torch_steps[0].payload["env"]

        probed = {}
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [])
        monkeypatch.setattr(uvtool, "pip_install", lambda *a, **kw: _ok_result())

        def fake_verify(runtime, tag, *, env=None, log=None, should_cancel=None):
            probed["env"] = env
            return uvtool.TorchCheck(True, "fine")

        monkeypatch.setattr(uvtool, "verify_torch", fake_verify)
        Runner(plan).run()

        assert probed["env"] == expected
        assert Manifest.load(tmp_path).torch_env == expected, \
            "the launcher reads the manifest; an override not written there never reaches ComfyUI"

    def test_the_manifest_round_trips_it(self, tmp_path):
        manifest = Manifest(data_root=tmp_path, torch_env={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
        manifest.save()
        assert Manifest.load(tmp_path).torch_env == {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}

    def test_an_old_manifest_without_it_still_loads(self, tmp_path):
        path = tmp_path / "state" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema": 1, "packs": [], "files": []}))
        assert Manifest.load(tmp_path).torch_env == {}


# ---------------------------------------------------------------------------
# The record of what landed


MODEL_A = b"a" * 4096
MODEL_B = b"b" * 4096


def _download_plan(root: Path) -> InstallPlan:
    dest = root / "models" / "checkpoints"
    files = tuple(
        Download(url=f"http://origin.invalid/{name}", dest=dest, filename=name,
                 sha256=PENDING, size_bytes=len(body))
        for name, body in (("a.safetensors", MODEL_A), ("b.safetensors", MODEL_B)))
    step = Step(Kind.DOWNLOAD, "Downloading", bytes_total=8192,
                payload={"id": "image.test"}, downloads=files)
    return InstallPlan((step,), choose_torch(AMD_NEEDS_OVERRIDE), root, ())


class TestTheRecordIsWrittenAsEachFileLands:
    """The manifest used to be saved when the whole pack step finished. An
    install that died on the third file had two verified models on disk and
    no record of either, so the next run fetched them again."""

    @pytest.fixture
    def failed_on_the_second_file(self, tmp_path, monkeypatch):
        def fake_download(url, dest, **kw):
            if dest.name == "a.safetensors":
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(MODEL_A)
                return download_module.file_digest(dest)
            raise DownloadError("the connection dropped")

        monkeypatch.setattr("toolshed.exec.runner.download_file", fake_download)
        with pytest.raises(InstallFailed):
            Runner(_download_plan(tmp_path)).run()
        return tmp_path

    def test_the_first_file_is_on_record_on_disk(self, failed_on_the_second_file):
        root = failed_on_the_second_file
        recorded = {e.path for e in Manifest.load(root).files}
        assert recorded == {"models/checkpoints/a.safetensors"}

    def test_and_in_the_sidecar_beside_the_models(self, failed_on_the_second_file):
        root = failed_on_the_second_file
        sidecar = json.loads((root / "models" / ".toolshed-hashes.json").read_text())
        assert "models/checkpoints/a.safetensors" in sidecar
        assert sidecar["models/checkpoints/a.safetensors"]["size_bytes"] == len(MODEL_A)

    def test_the_next_run_keeps_it(self, failed_on_the_second_file, monkeypatch):
        root = failed_on_the_second_file
        fetched = []

        def fake_download(url, dest, **kw):
            fetched.append(dest.name)
            dest.write_bytes(MODEL_B)
            return download_module.file_digest(dest)

        monkeypatch.setattr("toolshed.exec.runner.download_file", fake_download)
        Runner(_download_plan(root)).run()
        assert fetched == ["b.safetensors"], "a verified file was fetched again"


class TestKeptModelsAreRecognisedAfterUninstall:
    """Uninstall keeps the models and deletes the manifest. The next install
    then found 40 GB it had no record of, and -- correctly refusing to trust
    an unknown file -- downloaded it all again."""

    @pytest.fixture
    def reinstalling(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        root = home / "Toolshed"
        (root / "state").mkdir(parents=True)

        def fake_download(url, dest, **kw):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(MODEL_A if dest.name.startswith("a") else MODEL_B)
            return download_module.file_digest(dest)

        monkeypatch.setattr("toolshed.exec.runner.download_file", fake_download)
        Runner(_download_plan(root)).run()

        # The real uninstaller, not a simulation of it.
        result = subprocess.run(
            ["sh", str(REPO / "packaging" / "linux" / "uninstall.sh")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HOME": str(home), "TOOLSHED_ROOT": str(root)})
        assert result.returncode == 0, result.stderr
        assert not (root / "state").exists()
        return root

    @pytest.mark.skipif(os.name != "posix", reason="runs the shipped POSIX uninstaller")
    def test_the_sidecar_survives_with_the_models(self, reinstalling):
        assert (reinstalling / "models" / ".toolshed-hashes.json").is_file()

    @pytest.mark.skipif(os.name != "posix", reason="runs the shipped POSIX uninstaller")
    def test_nothing_is_downloaded_again(self, reinstalling, monkeypatch):
        fetched = []
        monkeypatch.setattr("toolshed.exec.runner.download_file",
                            lambda url, dest, **kw: fetched.append(dest.name))
        manifest = Runner(_download_plan(reinstalling)).run()
        assert fetched == []
        assert {e.path for e in manifest.files} == {
            "models/checkpoints/a.safetensors", "models/checkpoints/b.safetensors"}, \
            "the new manifest must cover the kept files, not just the new ones"

    @pytest.mark.skipif(os.name != "posix", reason="runs the shipped POSIX uninstaller")
    def test_a_file_that_changed_meanwhile_is_not_trusted(self, reinstalling, monkeypatch):
        (reinstalling / "models" / "checkpoints" / "a.safetensors").write_bytes(b"x" * len(MODEL_A))
        fetched = []

        def fake_download(url, dest, **kw):
            fetched.append(dest.name)
            dest.write_bytes(MODEL_A)
            return download_module.file_digest(dest)

        monkeypatch.setattr("toolshed.exec.runner.download_file", fake_download)
        Runner(_download_plan(reinstalling)).run()
        assert fetched == ["a.safetensors"]


class TestAddingAPackKeepsTheOthers:
    def test_packs_accumulate_across_runs(self, tmp_path):
        Manifest(data_root=tmp_path, packs=["image.zimage"]).save()
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        plan = InstallPlan((), plan.torch, tmp_path, ())
        from toolshed.catalog.packs import Pack
        extra = Pack(id="audio.ace", modality="Music", name="Make music", blurb="b",
                     recipe="r", vram_gb_min=8, licence="Apache 2.0", download_bytes=1)
        plan = InstallPlan((), plan.torch, tmp_path, (extra,))
        assert Runner(plan).run().packs == ["audio.ace", "image.zimage"]


# ---------------------------------------------------------------------------
# Resuming a download


BODY = b"model" * 2000


class RangeOrigin(BaseHTTPRequestHandler):
    """Serves one file, honouring Range the way a CDN does: 206 for a range
    inside the file, 416 with the true length for one past its end."""

    hits: list[int] = []

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()

    def do_GET(self):
        header = self.headers.get("Range", "")
        start = int(header[len("bytes="):].rstrip("-")) if header.startswith("bytes=") else 0
        if start >= len(BODY):
            type(self).hits.append(416)
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(BODY)}")
            self.end_headers()
            return
        chunk = BODY[start:]
        type(self).hits.append(206 if start else 200)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(chunk)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def range_origin():
    RangeOrigin.hits = []
    server = HTTPServer(("127.0.0.1", 0), RangeOrigin)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/model.bin"
    server.shutdown()


class TestAStalePartIsNotTakenForTheWholeFile:
    """A .part longer than the file the server has -- a model re-published
    smaller under the same name -- gets 416 to its Range request, and that was
    taken to mean "already have it all". The stale bytes became the model."""

    def test_with_the_size_known(self, range_origin, tmp_path):
        dest = tmp_path / "model.bin"
        (tmp_path / "model.bin.part").write_bytes(b"stale" * 3000)
        download_file(range_origin, dest, size_bytes=len(BODY))
        assert dest.read_bytes() == BODY
        assert RangeOrigin.hits == [416, 200], "it did not start over after the 416"

    def test_with_only_the_servers_word_for_the_size(self, range_origin, tmp_path):
        dest = tmp_path / "model.bin"
        (tmp_path / "model.bin.part").write_bytes(b"stale" * 3000)
        download_file(range_origin, dest)
        assert dest.read_bytes() == BODY

    def test_a_complete_part_is_still_just_renamed(self, range_origin, tmp_path):
        dest = tmp_path / "model.bin"
        (tmp_path / "model.bin.part").write_bytes(BODY)
        download_file(range_origin, dest, size_bytes=len(BODY))
        assert dest.read_bytes() == BODY
        assert RangeOrigin.hits == [416], "a finished .part was downloaded again"

    def test_a_genuine_resume_still_resumes(self, range_origin, tmp_path):
        dest = tmp_path / "model.bin"
        (tmp_path / "model.bin.part").write_bytes(BODY[:1000])
        download_file(range_origin, dest, size_bytes=len(BODY))
        assert dest.read_bytes() == BODY
        assert RangeOrigin.hits == [206]


class TestResumingAsksForRoomForTheRemainder:
    """With 30 of 40 GB on disk the space check demanded room for 40, and
    refused exactly the machines that had made the most progress."""

    def test_only_the_missing_bytes_are_demanded(self, tmp_path, monkeypatch):
        asked = {}

        def fake_check(dest_dir, needed):
            asked["needed"] = needed
            raise DownloadError("stop here")

        monkeypatch.setattr(download_module, "check_space", fake_check)
        (tmp_path / "m.bin.part").write_bytes(b"x" * 300)
        with pytest.raises(DownloadError):
            download_file("http://origin.invalid/m.bin", tmp_path / "m.bin",
                          size_bytes=1000, attempts=1)
        assert asked["needed"] == 700


# ---------------------------------------------------------------------------
# Where an install lands on Windows


class TestTheWindowsDataRoot:
    """SYSTEMDRIVE is "C:" with no backslash, and Path("C:") / "Toolshed" is
    "C:Toolshed": relative to whatever the current directory on C: happens to
    be. Every install would have landed somewhere different."""

    def test_it_is_an_absolute_path_on_the_system_drive(self):
        root = windows_data_root("C:")
        assert root.is_absolute()
        assert str(root) == r"C:\Toolshed"

    @pytest.mark.parametrize("drive", ["D:", "D:\\", " D: ", ""])
    def test_however_the_variable_is_spelt(self, drive):
        root = windows_data_root(drive)
        assert root.is_absolute()
        assert root.name == "Toolshed" and str(root.parent) == root.anchor


# ---------------------------------------------------------------------------
# The uninstallers


class TestTheUninstallerAcceptsATrailingSlash:
    """"$ROOT/" is what tab completion produces, and it defeated the string
    comparisons that keep the script away from / and $HOME."""

    pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shell script")

    def _run(self, home: Path, root: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", str(REPO / "packaging" / "linux" / "uninstall.sh")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HOME": str(home), "TOOLSHED_ROOT": root})

    def test_a_root_with_a_trailing_slash_still_works(self, tmp_path):
        home = tmp_path / "home"
        root = home / "Toolshed"
        (root / "models" / "vae").mkdir(parents=True)
        (root / "models" / "vae" / "v.safetensors").write_bytes(b"w")
        (root / "state").mkdir()
        (root / "state" / "manifest.json").write_text("{}")
        result = self._run(home, f"{root}/")
        assert result.returncode == 0, result.stderr
        assert (root / "models" / "vae" / "v.safetensors").exists()
        assert not (root / "state").exists()

    def test_something_running_from_the_root_is_still_found_and_stopped(self, tmp_path):
        """The one place the slash really bit: processes are matched on
        "$ROOT/engine/", and with ROOT ending in "/" that is "…//engine/",
        which matches nothing. The engine was left running while the folder
        it runs from was deleted under it."""
        home = tmp_path / "home"
        root = home / "Toolshed"
        (root / "engine" / "comfyui").mkdir(parents=True)
        (root / "models").mkdir()
        engine = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)",
             str(root / "engine" / "comfyui" / "main.py")])
        try:
            result = self._run(home, f"{root}/")
            assert result.returncode == 0, result.stderr
            deadline = time.monotonic() + 10
            while engine.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert engine.poll() is not None, "the engine was left running from a deleted folder"
        finally:
            if engine.poll() is None:
                engine.kill()
                engine.wait()

    def test_the_home_directory_with_a_trailing_slash_is_still_refused(self, tmp_path):
        home = tmp_path / "home"
        (home / "Documents").mkdir(parents=True)
        (home / "Documents" / "taxes.ods").write_text("mine")
        result = self._run(home, f"{home}/")
        assert result.returncode != 0
        assert (home / "Documents" / "taxes.ods").exists()


class TestTheWindowsUninstallerIsSilentWhenAsked:
    def test_no_message_box_under_silent(self):
        text = (REPO / "packaging" / "windows" / "toolshed.iss").read_text(encoding="utf-8")
        body = text[text.index("procedure CurUninstallStepChanged"):]
        body = body[:body.index("\nend;")]
        guard = body.index("if not UninstallSilent")
        assert all(guard < i for i in _all_indexes(body, "MsgBox(")), \
            "a MsgBox outside the UninstallSilent guard hangs an unattended /VERYSILENT uninstall"


def _all_indexes(text: str, needle: str) -> list[int]:
    found, start = [], 0
    while (i := text.find(needle, start)) != -1:
        found.append(i)
        start = i + 1
    return found


# ---------------------------------------------------------------------------
# The screens


pyside = pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])


class TestStopIsNotADeadEnd:
    """Stop during an install changed one label to "Stopped." and left a Stop
    button that did nothing: no Try again, no Close."""

    def test_the_install_page_reports_a_stop(self, qapp):
        from toolshed.ui.install import InstallPage

        page = InstallPage()
        got = []
        page.stopped.connect(lambda: got.append(True))
        page._on_cancelled()
        assert got and page.current.text() == "Stopped."

    def test_the_window_offers_to_carry_on(self, qapp):
        from toolshed.hw.verdict import verdict_for
        from toolshed.ui.app import MainWindow

        window = MainWindow(AMD_NEEDS_OVERRIDE, verdict_for(AMD_NEEDS_OVERRIDE))
        window._on_install_stopped()
        assert window.install_page.heading.text() == "Stopped"
        assert window.next_button.text() == "Try again"
        assert window.back_button.isVisibleTo(window) and window.back_button.text() == "Close"
        assert not window._installing

    def test_shutting_the_page_with_no_worker_is_fine(self, qapp):
        from toolshed.ui.install import InstallPage

        assert InstallPage().shutdown() is True

    def test_closing_the_window_stops_the_installer(self, qapp, monkeypatch):
        from toolshed.hw.verdict import verdict_for
        from toolshed.ui.app import MainWindow

        window = MainWindow(AMD_NEEDS_OVERRIDE, verdict_for(AMD_NEEDS_OVERRIDE))
        calls = []
        monkeypatch.setattr(window.install_page, "shutdown",
                            lambda *a, **kw: calls.append(1) or True)
        window.close()
        assert calls, "the install worker would outlive the window and abort the process"


class TestStoppingAGenerationIsReported:
    def test_a_cancelled_run_puts_the_button_back(self, qapp, tmp_path):
        from toolshed.exec.comfy_api import ComfyError
        from toolshed.ui.make import GenerateWorker

        class Client:
            def object_info(self):
                raise ComfyError("stopped", reason_key="cancelled")

        got = []
        worker = GenerateWorker(Client(), tmp_path / "w.json", None, {})
        worker.failed.connect(lambda m, d: got.append(m))
        worker.run()
        assert got == ["Stopped."], "nothing told the screen the run had ended"


class TestTheSizeBoxesFollowTheWorkflow:
    def test_changing_workflow_forgets_the_last_ones_size(self, qapp):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.set_packs(ALL_PACK_IDS)
        assert page.what.count() >= 2, "need two workflows to switch between"
        page.what.setCurrentIndex(0)
        page._on_analysed(1024, 1024)
        page.what.setCurrentIndex(1)
        assert (page.width.value(), page.height.value()) == (0, 0), \
            "a picture workflow's size was carried into the next workflow as an override"


def _pack_ids() -> list[str]:
    from toolshed.catalog.packs import load_packs
    return [p.id for p in load_packs()]


ALL_PACK_IDS = _pack_ids()


# ---------------------------------------------------------------------------
# uv's silence while it downloads PyTorch


class TestASilentDownloadIsNotMistakenForAHang:
    """The install sat on "Using Python 3.12.14 environment at: …" for minutes
    while uv fetched several gigabytes of PyTorch in silence. Two faults: the
    screen gave no sign anything was happening, and a fixed 60-minute budget
    would have killed a slow connection's download at the finish line."""

    def test_output_resets_the_deadline(self):
        """A child that keeps talking is alive, however long it takes."""
        script = "import sys,time\nfor i in range(6):\n    print(i, flush=True); time.sleep(0.3)"
        result = run([sys.executable, "-c", script], timeout=1.0)
        assert result.ok, result.stderr

    def test_a_heartbeat_reporting_progress_resets_it_too(self):
        result = run([sys.executable, "-c", "import time; time.sleep(1.8)"],
                     timeout=0.7, heartbeat=lambda: True, heartbeat_every=0.2)
        assert result.ok, result.stderr

    def test_no_output_and_no_progress_is_a_stall(self):
        result = run([sys.executable, "-c", "import time; time.sleep(30)"],
                     timeout=0.7, heartbeat=lambda: False, heartbeat_every=0.2)
        assert not result.ok
        assert "no sign of progress" in result.stderr

    def test_pip_install_forwards_the_heartbeat(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(uvtool, "run",
                            lambda cmd, **kw: (seen.update(kw), _ok_result())[1])
        beat = lambda: True  # noqa: E731
        uvtool.pip_install(FAKE_UV, tmp_path, ["torch"], heartbeat=beat)
        assert seen["heartbeat"] is beat

    def test_the_cache_growing_is_reported_as_bytes_received(self, tmp_path, monkeypatch):
        """What the user sees under the step while uv says nothing."""
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        install = next(s for s in plan.steps if s.kind == Kind.INSTALL_TORCH)
        runner = Runner(InstallPlan((install,), plan.torch, tmp_path, ()))
        events = []
        runner.on_event = events.append
        cache = tmp_path / "runtime" / "uv-cache" / "archive-v0" / "abc"
        cache.mkdir(parents=True)

        def fake_pip(uv, runtime, packages, *, heartbeat=None, **kw):
            assert heartbeat is not None, "torch is installed with nothing watching the download"
            assert heartbeat() is False, "nothing has arrived yet"
            (cache / "torch.so").write_bytes(b"x" * 5_000_000)
            assert heartbeat() is True, "5 MB arrived and was not noticed"
            return _ok_result()

        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [])
        monkeypatch.setattr(uvtool, "pip_install", fake_pip)
        runner._install_torch(install)

        said = [e.message for e in events if e.kind == "progress"]
        assert said and "0.01 GB received so far" in said[-1] and "MB/s" in said[-1]

    def test_a_failed_uv_run_says_what_uv_said(self, tmp_path, monkeypatch):
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        install = next(s for s in plan.steps if s.kind == Kind.INSTALL_TORCH)
        runner = Runner(InstallPlan((install,), plan.torch, tmp_path, ()))
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [])
        monkeypatch.setattr(uvtool, "pip_install",
                            lambda *a, **kw: proc.Result(-1, "", "uv showed no sign of progress"))
        with pytest.raises(InstallFailed) as exc:
            runner._install_torch(install)
        assert "no sign of progress" in str(exc.value)


# ---------------------------------------------------------------------------
# uv's downloads get the same bar as the models


# Verbatim from `uv pip install --dry-run -v` with uv 0.12.10, two packages.
UV_DRY_RUN = """\
DEBUG uv 0.12.10 (x86_64-unknown-linux-gnu)
DEBUG At least one requirement is not satisfied: attrs==25.3.0
DEBUG No cache entry for: https://pypi.org/simple/six/
DEBUG Sending fresh GET request for: https://pypi.org/simple/six/
DEBUG Selecting: six==1.17.0 [compatible] (six-1.17.0-py2.py3-none-any.whl)
DEBUG No cache entry for: https://files.pythonhosted.org/packages/77/06/bb80f5f86020c4551da315d78b3ab75e8228f89f0162f2c3a819e407941a/attrs-25.3.0-py3-none-any.whl.metadata
DEBUG Selecting: attrs==25.3.0 [compatible] (attrs-25.3.0-py3-none-any.whl)
DEBUG Tried 2 versions: attrs 1, six 1
DEBUG marker environment resolution took 0.113s
DEBUG Identified uncached distribution: six==1.17.0
DEBUG Identified uncached distribution: attrs==25.3.0
 + attrs==25.3.0
 + six==1.17.0
"""

SIX_WHEEL = b"PK\x05\x06" + b"\x00" * 18 + b"six" * 1000
SIX_NAME = "six-1.17.0-py2.py3-none-any.whl"


class FakeIndex(BaseHTTPRequestHandler):
    """A PEP 503 simple index with one project, hashes in the fragment."""

    heads: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/simple/six/":
            import hashlib
            sha = hashlib.sha256(SIX_WHEEL).hexdigest()
            body = (f'<html><body><h1>Links for six</h1>'
                    f'<a href="../../files/six-1.17.0-py2.py3-none-any.whl#sha256={sha}">'
                    f'six-1.17.0-py2.py3-none-any.whl</a><br/>'
                    f'<a href="../../files/six-1.17.0.tar.gz#sha256=abc">six-1.17.0.tar.gz</a>'
                    f'</body></html>').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/files/six-1.17.0-py2.py3-none-any.whl":
            self.send_response(200)
            self.send_header("Content-Length", str(len(SIX_WHEEL)))
            self.end_headers()
            self.wfile.write(SIX_WHEEL)
        else:
            self.send_error(404)

    def do_HEAD(self):
        type(self).heads.append(self.path)
        if self.path == "/files/six-1.17.0-py2.py3-none-any.whl":
            self.send_response(200)
            self.send_header("Content-Length", str(len(SIX_WHEEL)))
            self.end_headers()
        else:
            self.send_error(404)


@pytest.fixture
def fake_index():
    FakeIndex.heads = []
    server = HTTPServer(("127.0.0.1", 0), FakeIndex)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/simple"
    server.shutdown()


class TestAskingUvWhatItWillFetch:
    def test_every_chosen_file_is_read_from_the_dry_run(self):
        files = uvtool.parse_selected(UV_DRY_RUN)
        assert [(f.name, f.version, f.filename) for f in files] == [
            ("six", "1.17.0", "six-1.17.0-py2.py3-none-any.whl"),
            ("attrs", "25.3.0", "attrs-25.3.0-py3-none-any.whl"),
        ]

    def test_a_repeated_line_is_one_file(self):
        assert len(uvtool.parse_selected(UV_DRY_RUN + UV_DRY_RUN)) == 2

    def test_a_dry_run_that_fetches_nothing_is_empty(self):
        assert uvtool.parse_selected("Audited 12 packages in 3ms\n") == []

    def test_the_dry_run_is_a_dry_run(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(uvtool, "run", lambda cmd, **kw: (
            seen.update(cmd=[str(c) for c in cmd]), _ok_result(UV_DRY_RUN))[1])
        files = uvtool.plan_install(FAKE_UV, tmp_path, ["torch"], index_url="https://i/whl")
        assert "--dry-run" in seen["cmd"] and "-v" in seen["cmd"]
        assert seen["cmd"][seen["cmd"].index("--index-url") + 1] == "https://i/whl"
        assert [f.name for f in files] == ["six", "attrs"]

    def test_a_failed_dry_run_is_an_empty_list_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uvtool, "run", lambda cmd, **kw: proc.Result(1, "", "boom"))
        assert uvtool.plan_install(FAKE_UV, tmp_path, ["torch"]) == []


class TestLookingTheFilesUpOnTheIndex:
    def test_url_hash_and_size_come_from_the_simple_page(self, fake_index):
        import hashlib
        [six] = uvtool.locate([uvtool.WheelFile("six", "1.17.0", SIX_NAME)], fake_index)
        assert six.url == fake_index.replace("/simple", "") + "/files/" + SIX_NAME
        assert six.sha256 == hashlib.sha256(SIX_WHEEL).hexdigest()
        assert six.size_bytes == len(SIX_WHEEL)
        assert "#" not in six.url

    def test_a_file_the_index_does_not_list_comes_back_without_a_url(self, fake_index):
        [odd] = uvtool.locate([uvtool.WheelFile("six", "1.17.0", "six-9.9.9-py3-none-any.whl")],
                              fake_index)
        assert odd.url == "" and odd.size_bytes == 0

    def test_a_project_that_is_not_there_at_all(self, fake_index):
        [gone] = uvtool.locate([uvtool.WheelFile("nope", "1", "nope-1-py3-none-any.whl")],
                               fake_index)
        assert gone.url == ""

    @pytest.mark.parametrize("name, key", [
        ("typing_extensions", "typing-extensions"), ("Jinja2", "jinja2"),
        ("pytorch-triton-rocm", "pytorch-triton-rocm"), ("zope.interface", "zope-interface"),
    ])
    def test_project_names_are_normalised_for_the_page_url(self, name, key):
        assert uvtool.normalise(name) == key


class TestUvInstallsFromWhatWeDownloaded:
    def test_find_links_means_offline_and_no_index(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(uvtool, "run", lambda cmd, **kw: (
            seen.update(cmd=[str(c) for c in cmd]), _ok_result())[1])
        uvtool.pip_install(FAKE_UV, tmp_path, ["torch"], index_url="https://i/whl",
                           find_links=tmp_path / "wheels")
        cmd = seen["cmd"]
        assert "--no-index" in cmd and "--offline" in cmd
        assert cmd[cmd.index("--find-links") + 1] == str(tmp_path / "wheels")
        assert "--index-url" not in cmd, "an index alongside --no-index is a contradiction"

    def test_without_it_the_index_is_used(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(uvtool, "run", lambda cmd, **kw: (
            seen.update(cmd=[str(c) for c in cmd]), _ok_result())[1])
        uvtool.pip_install(FAKE_UV, tmp_path, ["torch"], index_url="https://i/whl")
        assert "--index-url" in seen["cmd"] and "--no-index" not in seen["cmd"]


class TestTheGraphicsCardStepHasARealBar:
    """What the user asked for: the torch download looking like the model
    downloads -- how big it is, and how far along."""

    def _runner(self, tmp_path, events):
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        install = next(s for s in plan.steps if s.kind == Kind.INSTALL_TORCH)
        runner = Runner(InstallPlan((install,), plan.torch, tmp_path, ()))
        runner.on_event = events.append
        return runner, install

    def test_the_size_is_known_before_a_byte_is_fetched_and_the_bar_fills(
            self, tmp_path, monkeypatch, fake_index):
        events = []
        runner, step = self._runner(tmp_path, events)
        installed = {}

        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [
            uvtool.WheelFile("six", "1.17.0", "six-1.17.0-py2.py3-none-any.whl")])
        real_locate = uvtool.locate
        monkeypatch.setattr(uvtool, "locate",
                            lambda files, index, **kw: real_locate(files, fake_index))

        def fake_pip(uv, runtime, packages, *, find_links=None, **kw):
            installed["find_links"] = find_links
            installed["present"] = sorted(p.name for p in find_links.iterdir())
            installed["bytes"] = (find_links / "six-1.17.0-py2.py3-none-any.whl").read_bytes()
            return _ok_result()

        monkeypatch.setattr(uvtool, "pip_install", fake_pip)
        runner._install_torch(step)

        bars = [e for e in events if e.kind == "progress" and e.bytes_total]
        assert bars, "no event carried a total, so the row could not show X / Y GB"
        assert {e.bytes_total for e in bars} == {len(SIX_WHEEL)}
        assert bars[-1].bytes_done == len(SIX_WHEEL) and bars[-1].fraction == 1.0
        # The total must ride on the events *during* the transfer -- the ones
        # with a speed -- not only on the final "done" one.
        during = [e for e in events if e.kind == "progress" and "MB/s" in e.message]
        assert during and all(e.bytes_total == len(SIX_WHEEL) for e in during)
        assert installed["present"] == ["six-1.17.0-py2.py3-none-any.whl"]
        assert installed["bytes"] == SIX_WHEEL, "uv was handed something other than the wheel"
        assert not installed["find_links"].exists(), "gigabytes of wheels left behind after install"

    def test_a_corrupt_download_is_not_handed_to_uv(self, tmp_path, monkeypatch, fake_index):
        events = []
        runner, step = self._runner(tmp_path, events)
        runner.attempts = 1
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [
            uvtool.WheelFile("six", "1.17.0", "six-1.17.0-py2.py3-none-any.whl")])
        monkeypatch.setattr(uvtool, "locate", lambda files, index, **kw: [
            uvtool.WheelFile("six", "1.17.0", "six-1.17.0-py2.py3-none-any.whl",
                             url=fake_index.replace("/simple", "") + "/files/" + SIX_NAME,
                             sha256="0" * 64, size_bytes=len(SIX_WHEEL))])
        monkeypatch.setattr(uvtool, "pip_install",
                            lambda *a, **kw: pytest.fail("uv was given a file with the wrong hash"))
        with pytest.raises(InstallFailed) as exc:
            runner.run()
        assert exc.value.reason_key == "checksum_mismatch"

    def test_when_the_index_cannot_size_it_uv_fetches_as_before(self, tmp_path, monkeypatch):
        events = []
        runner, step = self._runner(tmp_path, events)
        seen = {}
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [
            uvtool.WheelFile("torch", "2.14.0+rocm7.2", "torch-2.14.0+rocm7.2-cp312-cp312-x.whl")])
        monkeypatch.setattr(uvtool, "locate", lambda files, index, **kw: list(files))  # no url

        def fake_pip(uv, runtime, packages, *, find_links=None, heartbeat=None, **kw):
            seen.update(find_links=find_links, heartbeat=heartbeat)
            return _ok_result()

        monkeypatch.setattr(uvtool, "pip_install", fake_pip)
        runner._install_torch(step)
        assert seen["find_links"] is None and seen["heartbeat"] is not None

    def test_nothing_to_fetch_means_no_download_and_a_plain_install(self, tmp_path, monkeypatch):
        events = []
        runner, step = self._runner(tmp_path, events)
        calls = []
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [])
        monkeypatch.setattr(uvtool, "locate", lambda *a, **kw: pytest.fail("nothing to look up"))
        monkeypatch.setattr(uvtool, "pip_install",
                            lambda *a, **kw: (calls.append(kw.get("find_links")), _ok_result())[1])
        runner._install_torch(step)
        assert calls == [None]

    def test_if_uv_rejects_the_folder_it_fetches_itself(self, tmp_path, monkeypatch, fake_index):
        events = []
        runner, step = self._runner(tmp_path, events)
        calls = []
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: FAKE_UV)
        monkeypatch.setattr(uvtool, "plan_install", lambda *a, **kw: [
            uvtool.WheelFile("six", "1.17.0", "six-1.17.0-py2.py3-none-any.whl")])
        real_locate = uvtool.locate
        monkeypatch.setattr(uvtool, "locate",
                            lambda files, index, **kw: real_locate(files, fake_index))

        def fake_pip(uv, runtime, packages, *, find_links=None, **kw):
            calls.append(find_links)
            return proc.Result(1, "", "no solution") if find_links else _ok_result()

        monkeypatch.setattr(uvtool, "pip_install", fake_pip)
        runner._install_torch(step)
        assert calls[0] is not None and calls[1] is None


# ---------------------------------------------------------------------------
# The crash that ate seven minutes of work, twice


# The shape of ComfyUI's own cli_args.py, quoted from v0.34.0. The help text
# for async offload says Nvidia; the AMD install log says otherwise.
CLI_ARGS_SOURCE = '''
parser.add_argument("--async-offload", nargs="?", const=2, type=int,
                    help="Use async weight offloading. An optional argument "
                         "controls the amount of offload streams. Default is 2. "
                         "Enabled by default on Nvidia.")
parser.add_argument("--disable-async-offload", action="store_true",
                    help="Disable async weight offloading.")
parser.add_argument("--disable-pinned-memory", action="store_true",
                    help="Disable pinned memory use.")
parser.add_argument("--disable-dynamic-vram", action="store_true",
                    help="Disable dynamic VRAM and use estimate based model loading.")
'''


def _engine_tree(root: Path, source: str = CLI_ARGS_SOURCE) -> Path:
    engine = root / "engine" / "comfyui"
    (engine / "comfy").mkdir(parents=True)
    (engine / "comfy" / "cli_args.py").write_text(source, encoding="utf-8")
    return engine


class TestTheFlagsWePassAreTheOnesComfyuiHas:
    """A flag ComfyUI does not know makes argparse exit before the server
    starts. That turns "crashes at the end of a long job" into "never starts",
    and we have no card here to rehearse it on."""

    def test_the_exact_spellings(self):
        from toolshed.exec.engine import AMD_SAFEGUARDS

        assert [s.flag for s in AMD_SAFEGUARDS] == [
            "--disable-async-offload", "--disable-pinned-memory",
            "--disable-dynamic-vram"]

    def test_a_flag_the_engine_declares_is_recognised(self, tmp_path):
        from toolshed.exec.engine import engine_understands

        engine = _engine_tree(tmp_path)
        assert engine_understands(engine, "--disable-async-offload")
        assert engine_understands(engine, "--disable-pinned-memory")

    def test_one_it_does_not_is_not(self, tmp_path):
        from toolshed.exec.engine import engine_understands

        assert not engine_understands(_engine_tree(tmp_path), "--disable-warp-drive")

    def test_a_mention_in_prose_is_not_a_flag(self, tmp_path):
        """"see --disable-async-offload" in a comment must not count."""
        from toolshed.exec.engine import engine_understands

        engine = _engine_tree(tmp_path, "# renamed; was --disable-async-offload\n")
        assert not engine_understands(engine, "--disable-async-offload")

    def test_an_engine_that_is_not_there_yet_gets_nothing(self, tmp_path):
        from toolshed.exec.engine import engine_understands

        assert not engine_understands(tmp_path / "nope", "--disable-async-offload")


class TestWhichSafeguardsAMachineGets:
    def test_rocm_gets_all_three(self, tmp_path):
        """Three, not two. The first two left dynamic VRAM's own transfer path
        running, and that is what was still faulting at every VAE load."""
        from toolshed.exec.engine import choose_safeguards

        got = choose_safeguards(_engine_tree(tmp_path), rocm=True)
        assert got.flags == ["--disable-async-offload", "--disable-pinned-memory",
                             "--disable-dynamic-vram"]
        assert got.unavailable == ()

    def test_nvidia_gets_none(self, tmp_path):
        from toolshed.exec.engine import choose_safeguards

        got = choose_safeguards(_engine_tree(tmp_path), rocm=False)
        assert got.flags == [] and got.unavailable == ()

    def test_the_user_taking_it_over_means_we_stand_aside(self, tmp_path):
        """Typing --async-offload in Extra options puts the default back, and
        we must not then argue with it by passing the opposite flag too."""
        from toolshed.exec.engine import choose_safeguards

        got = choose_safeguards(_engine_tree(tmp_path), rocm=True,
                                extra=["--async-offload", "4"])
        assert got.flags == ["--disable-pinned-memory", "--disable-dynamic-vram"]
        assert [s.flag for s in got.overridden] == ["--disable-async-offload"]
        assert got.unavailable == (), "a deliberate choice is not a missing safeguard"

    def test_and_repeating_our_own_flag_is_not_doubled(self, tmp_path):
        from toolshed.exec.engine import choose_safeguards

        got = choose_safeguards(_engine_tree(tmp_path), rocm=True,
                                extra=["--disable-pinned-memory"])
        assert got.flags == ["--disable-async-offload", "--disable-dynamic-vram"]

    def test_asking_for_dynamic_vram_back_is_respected(self, tmp_path):
        """--enable-dynamic-vram is somebody deciding they want it, perhaps to
        see whether a newer ComfyUI fixed the fault. We do not argue."""
        from toolshed.exec.engine import choose_safeguards

        got = choose_safeguards(_engine_tree(tmp_path), rocm=True,
                                extra=["--enable-dynamic-vram"])
        assert "--disable-dynamic-vram" not in got.flags
        assert [s.flag for s in got.overridden] == ["--disable-dynamic-vram"]
        assert got.unavailable == (), "a deliberate choice is not a missing safeguard"

    def test_an_engine_that_renamed_a_flag_reports_it_rather_than_dropping_it(self, tmp_path):
        """The way this whole crash comes back: a future ComfyUI renames the
        flag, we quietly stop passing it, and long jobs start dying again with
        nothing anywhere saying why."""
        from toolshed.exec.engine import choose_safeguards

        engine = _engine_tree(tmp_path, 'parser.add_argument("--listen")\n')
        got = choose_safeguards(engine, rocm=True)
        assert got.flags == [], "an unknown flag would stop ComfyUI starting at all"
        assert [s.flag for s in got.unavailable] == ["--disable-async-offload",
                                                     "--disable-pinned-memory",
                                                     "--disable-dynamic-vram"]


class TestTheSafeguardsReachTheCommandLine:
    def test_they_are_in_the_argv_before_the_users_own(self, tmp_path):
        from toolshed.exec.engine import Engine

        engine = Engine(root=tmp_path, port=8188,
                        safe_args=["--disable-async-offload", "--disable-pinned-memory"],
                        extra_args=["--reserve-vram", "2"])
        argv = engine.command()
        assert "--disable-async-offload" in argv and "--disable-pinned-memory" in argv
        assert argv.index("--disable-pinned-memory") < argv.index("--reserve-vram"), \
            "the user's options must still come last, so they keep the final word"

    def test_a_machine_that_needs_none_gets_a_clean_command(self, tmp_path):
        from toolshed.exec.engine import Engine

        assert not [a for a in Engine(root=tmp_path).command() if a.startswith("--disable-")
                    and a != "--disable-auto-launch"]


class TestTheLauncherKnowsItIsOnRocm:
    def test_from_the_index_the_installer_used(self, qapp, tmp_path):
        from toolshed.exec.manifest import Manifest
        from toolshed.ui.launch import LaunchPage

        page = LaunchPage(tmp_path)
        rocm = Manifest(data_root=tmp_path,
                        torch_index="https://download.pytorch.org/whl/rocm7.2")
        cuda = Manifest(data_root=tmp_path,
                        torch_index="https://download.pytorch.org/whl/cu130")
        assert page._on_rocm(rocm) is True
        assert page._on_rocm(cuda) is False

    def test_an_older_install_falls_back_to_the_machine(self, qapp, tmp_path, monkeypatch):
        import importlib

        from toolshed.exec.manifest import Manifest
        from toolshed.ui.launch import LaunchPage

        # `toolshed.hw` re-exports detect, so `import toolshed.hw.detect as x`
        # binds the function, not the module. Ask for the module by name.
        detect_module = importlib.import_module("toolshed.hw.detect")
        monkeypatch.setattr(detect_module, "detect",
                            lambda: HardwareReport(os="linux", gpus=(Gpu("amd", "RX 7900 XT"),)))
        assert LaunchPage(tmp_path)._on_rocm(Manifest(data_root=tmp_path)) is True

    def test_starting_the_engine_applies_them(self, qapp, tmp_path, monkeypatch):
        """End to end from the button: an AMD install starts ComfyUI with both
        defaults switched off, and says so in the log."""
        from PySide6 import QtCore

        import toolshed.ui.launch as launch_module
        from toolshed.exec.manifest import Manifest
        from toolshed.ui.launch import LaunchPage

        _engine_tree(tmp_path)
        Manifest(data_root=tmp_path,
                 torch_index="https://download.pytorch.org/whl/rocm7.2",
                 torch_env={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}).save()

        class StubWorker(QtCore.QThread):
            line = QtCore.Signal(str)
            ready = QtCore.Signal(str)
            failed = QtCore.Signal(str, str)
            died = QtCore.Signal(str)

            def __init__(self, engine):
                super().__init__()
                self.engine = engine

            def run(self):
                pass

        monkeypatch.setattr(launch_module, "StubWorker", StubWorker, raising=False)
        monkeypatch.setattr(launch_module, "EngineWorker", StubWorker)
        page = LaunchPage(tmp_path)
        page.start_engine()
        try:
            assert page.engine is not None
            assert page.engine.safe_args == ["--disable-async-offload",
                                             "--disable-pinned-memory",
                                             "--disable-dynamic-vram"]
            assert page.engine.env == {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}
            said = page.log.toPlainText()
            assert "--disable-async-offload" in said and "swapped out" in said
        finally:
            page.worker.wait(5000)


class TestAMissingSafeguardIsSaidOutLoud:
    """The way this crash returns: a future ComfyUI renames a flag, the
    launcher quietly stops passing it, and long jobs die again with nothing
    anywhere connecting the two."""

    def test_the_launcher_warns_instead_of_starting_quietly(self, qapp, tmp_path, monkeypatch):
        from PySide6 import QtCore

        import toolshed.ui.launch as launch_module
        from toolshed.exec.manifest import Manifest
        from toolshed.ui.launch import LaunchPage

        _engine_tree(tmp_path, 'parser.add_argument("--listen")\n')   # flags renamed away
        Manifest(data_root=tmp_path,
                 torch_index="https://download.pytorch.org/whl/rocm7.2").save()

        class StubWorker(QtCore.QThread):
            line = QtCore.Signal(str)
            ready = QtCore.Signal(str)
            failed = QtCore.Signal(str, str)
            died = QtCore.Signal(str)

            def __init__(self, engine):
                super().__init__()
                self.engine = engine

            def run(self):
                pass

        monkeypatch.setattr(launch_module, "EngineWorker", StubWorker)
        page = LaunchPage(tmp_path)
        page.start_engine()
        try:
            assert page.engine.safe_args == [], "an unknown flag stops ComfyUI starting"
            said = page.log.toPlainText()
            assert "--disable-async-offload" in said and "--disable-pinned-memory" in said
            assert "Warning" in said and "first thing to suspect" in said
        finally:
            page.worker.wait(5000)


class TestTheLogSaysWhichCrashItWas:
    """An exit code says the process aborted. Only the log says why, and the
    user should not have to send two hundred lines to someone who can read it."""

    def test_the_fault_from_the_real_log_is_recognised(self):
        from toolshed.exec.engine import explain_crash

        said = explain_crash(
            "Requested to load WanVAE\n"
            "Memory access fault by GPU node-1 (Agent handle: 0x1f768820) on address "
            "0x7f1396928000. Reason: Page not present or supervisor privilege.\n"
            "Fatal Python error: Aborted\n")
        assert said and "graphics memory fault" in said
        # All three transfer paths are off by default now, so the next step is
        # no longer a fourth flag -- it is taking the VAE off the card.
        assert "--cpu-vae" in said, "no next step offered"

    def test_an_ordinary_log_is_not_explained_away(self):
        from toolshed.exec.engine import explain_crash

        assert explain_crash("Prompt executed in 12.3 seconds\n") is None

    def test_the_engine_prefers_the_log_to_the_exit_code(self, tmp_path):
        from toolshed.exec.engine import Engine

        engine = Engine(root=tmp_path)
        engine._log.append("Memory access fault by GPU node-1 on address 0x7f13")
        assert "graphics memory fault" in engine._explain_exit(-6)

    def test_without_that_the_exit_code_still_speaks(self, tmp_path):
        from toolshed.exec.engine import Engine

        engine = Engine(root=tmp_path)
        engine._log.append("Prompt executed in 12.3 seconds")
        assert "graphics driver" in engine._explain_exit(-6)
        assert engine._explain_exit(0) == "ComfyUI closed on its own."


class TestTheShoppingListIsOnlyAnOptimisation:
    """Asking uv what it would fetch exists to put a progress bar on a
    download. It is not the install, and it must never be the thing that
    decides an install is impossible -- CI found this by having no uv on its
    PATH where the machine it was written on did."""

    def test_a_uv_that_cannot_be_run_is_an_empty_list_not_a_crash(self, tmp_path):
        got = uvtool.plan_install(Path("/nonexistent/toolshed-test/uv"), tmp_path, ["torch"])
        assert got == []

    def test_and_the_install_still_happens_the_ordinary_way(self, tmp_path, monkeypatch):
        plan = build_plan(AMD_NEEDS_OVERRIDE, [], tmp_path)
        install = next(s for s in plan.steps if s.kind == Kind.INSTALL_TORCH)
        runner = Runner(InstallPlan((install,), plan.torch, tmp_path, ()))
        seen = {}

        monkeypatch.setattr(uvtool, "uv_path",
                            lambda _r: Path("/nonexistent/toolshed-test/uv"))
        monkeypatch.setattr(uvtool, "locate",
                            lambda *a, **kw: pytest.fail("nothing was planned to look up"))
        monkeypatch.setattr(uvtool, "pip_install",
                            lambda *a, **kw: (seen.update(kw), _ok_result())[1])
        runner._install_torch(install)
        assert seen["find_links"] is None, "uv must fetch for itself when we could not plan"
