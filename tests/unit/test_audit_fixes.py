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
        uvtool.install_python(Path("uv"), tmp_path, "3.12", should_cancel=check)
        uvtool.create_venv(Path("uv"), tmp_path, "3.12", should_cancel=check)
        uvtool.pip_install(Path("uv"), tmp_path, ["torch"], should_cancel=check)
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
        uvtool.install_python(Path("uv"), tmp_path, "3.12")
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
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: Path("uv"))
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
