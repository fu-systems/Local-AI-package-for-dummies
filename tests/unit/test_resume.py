"""Running the install again must carry on, not start over or fall over.

This is the property the first version of the runner did not have, and the way
it failed was the worst possible shape: the install died at step three, and
every subsequent attempt died at the same place with

    error: Failed to create virtual environment
      Caused by: A virtual environment already exists at: Toolshed/runtime/venv

An existing workspace is the *desired* state. Treating it as an error left the
user with an install they could neither finish nor restart, and the only
apparent way out -- sudo -- would have made it permanently worse by leaving the
whole data root owned by root.

An install that fetches tens of gigabytes will be interrupted. "Run it again"
is the normal case, so every step is checked here for what it does when the
work is already done.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from toolshed.catalog.packs import Pack
from toolshed.exec import uvtool
from toolshed.exec.manifest import Manifest
from toolshed.exec.runner import InstallFailed, Runner
from toolshed.hw.detect import Gpu, HardwareReport
from toolshed.planner.plan import PENDING, Download, InstallPlan, Kind, Step
from toolshed.planner.torchsel import choose_torch

MODEL = b"model-weights" * 5000
ENGINE_TAG = "v0.34.0"


def engine_tarball() -> bytes:
    """A stand-in for the ComfyUI source tarball: one top-level directory with
    the requirements.txt the runner checks for."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, body in (("ComfyUI-0.34.0/requirements.txt", b"torch\n"),
                           ("ComfyUI-0.34.0/main.py", b"# engine\n")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
    return buf.getvalue()


ROUTES = {"/w.safetensors": MODEL, "/engine.tar.gz": engine_tarball()}


class CountingOrigin(BaseHTTPRequestHandler):
    """Serves the routes above and counts every body actually sent, so a test
    can assert that a second run transferred nothing."""

    hits: Counter = Counter()

    def log_message(self, *a):
        pass

    def _body(self):
        return ROUTES.get(self.path)

    def do_HEAD(self):
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self):
        body = self._body()
        if body is None:
            self.send_error(404)
            return
        type(self).hits[self.path] += 1
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def origin():
    CountingOrigin.hits = Counter()
    server = HTTPServer(("127.0.0.1", 0), CountingOrigin)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


AMD = HardwareReport(os="linux", gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))
PACK = Pack(id="image.zimage", modality="Pictures", name="Make pictures", blurb="b",
            recipe="image_z_image_turbo_int8", vram_gb_min=8, licence="Apache 2.0",
            download_bytes=len(MODEL))


def plan_with_unfrozen_hash(root: Path, base: str) -> InstallPlan:
    """The situation as it actually ships today.

    Every hash in the catalogue is PENDING_FREEZE, so download_file has nothing
    to compare a file on disk against and cannot skip it. If the runner does not
    remember what it fetched, a re-run re-downloads every byte -- which for a
    real pack is tens of gigabytes over domestic broadband.
    """
    dl = Download(url=f"{base}/w.safetensors", dest=root / "models" / "diffusion_models",
                  filename="w.safetensors", sha256=PENDING, size_bytes=len(MODEL))
    steps = (
        Step(Kind.MAKE_DIRS, "Creating folders", payload={"model_dirs": ["diffusion_models"]}),
        Step(Kind.FETCH_ENGINE, "Installing the AI engine",
             payload={"url": f"{base}/engine.tar.gz", "tag": ENGINE_TAG}),
        Step(Kind.DOWNLOAD, "Downloading make pictures", bytes_total=len(MODEL),
             payload={"id": PACK.id}, downloads=(dl,)),
        Step(Kind.INJECT_WORKFLOW, "Adding the workflow", payload={"id": PACK.id}),
        Step(Kind.WRITE_SETTINGS, "Setting sensible defaults"),
    )
    return InstallPlan(steps, choose_torch(AMD), root, (PACK,))


class TestRunningItAgain:
    def test_a_second_run_succeeds_and_transfers_nothing(self, origin, tmp_path):
        """The headline property. Both runs finish; only the first downloads."""
        plan = plan_with_unfrozen_hash(tmp_path, origin)

        Runner(plan).run()
        after_first = dict(CountingOrigin.hits)
        assert after_first == {"/w.safetensors": 1, "/engine.tar.gz": 1}

        logs = []
        Runner(plan, on_event=lambda e: logs.append(e.message)).run()

        assert dict(CountingOrigin.hits) == after_first, (
            "the second run re-downloaded something it already had"
        )
        assert any("already here" in m for m in logs)
        assert any("already installed" in m for m in logs)

    def test_the_model_is_still_there_and_correct_afterwards(self, origin, tmp_path):
        """Skipping must not be the same as losing track of the file."""
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        Runner(plan).run()
        manifest = Runner(plan).run()

        model = tmp_path / "models" / "diffusion_models" / "w.safetensors"
        assert model.read_bytes() == MODEL
        recorded = [e for e in manifest.files if e.path.endswith("w.safetensors")]
        assert len(recorded) == 1, "the file was recorded twice"
        assert recorded[0].sha256 == hashlib.sha256(MODEL).hexdigest()

    def test_a_corrupted_file_is_fetched_again_rather_than_trusted(self, origin, tmp_path):
        """The record says the file is fine; the disk says otherwise. The disk
        wins, because building on a corrupt model is the failure this whole
        verify-then-rename design exists to prevent."""
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        Runner(plan).run()

        model = tmp_path / "models" / "diffusion_models" / "w.safetensors"
        model.write_bytes(b"x" * len(MODEL))     # same size, wrong content

        Runner(plan).run()
        assert CountingOrigin.hits["/w.safetensors"] == 2
        assert model.read_bytes() == MODEL

    def test_a_deleted_file_is_fetched_again(self, origin, tmp_path):
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        Runner(plan).run()
        (tmp_path / "models" / "diffusion_models" / "w.safetensors").unlink()

        Runner(plan).run()
        assert CountingOrigin.hits["/w.safetensors"] == 2

    def test_a_half_extracted_engine_is_fetched_again(self, origin, tmp_path):
        """The manifest records the tag only after the tree is in place, but a
        tree can still be damaged afterwards. requirements.txt is what the next
        step needs, so its absence means fetch again."""
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        Runner(plan).run()
        (tmp_path / "engine" / "comfyui" / "requirements.txt").unlink()

        Runner(plan).run()
        assert CountingOrigin.hits["/engine.tar.gz"] == 2
        assert (tmp_path / "engine" / "comfyui" / "requirements.txt").is_file()

    def test_a_different_engine_version_replaces_the_old_one(self, origin, tmp_path):
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        Runner(plan).run()

        assert Manifest.load(tmp_path).engine_tag == ENGINE_TAG

        newer = plan_with_unfrozen_hash(tmp_path, origin)
        bumped = tuple(
            Step(s.kind, s.title, s.detail, s.bytes_total,
                 {**s.payload, "tag": "v0.35.0"}, s.downloads)
            if s.kind is Kind.FETCH_ENGINE else s
            for s in newer.steps
        )
        Runner(InstallPlan(bumped, newer.torch, tmp_path, newer.packs)).run()
        assert CountingOrigin.hits["/engine.tar.gz"] == 2
        assert Manifest.load(tmp_path).engine_tag == "v0.35.0"


class TestTheManifestOnlyClaimsWhatHappened:
    def test_the_engine_tag_is_absent_until_the_engine_lands(self, origin, tmp_path):
        """A record written before the fact survives the failure and lies about
        it. The next run would then skip fetching an engine that is not there."""
        plan = plan_with_unfrozen_hash(tmp_path, origin)
        broken = tuple(
            Step(s.kind, s.title, s.detail, s.bytes_total,
                 {**s.payload, "url": f"{origin}/gone.tar.gz"}, s.downloads)
            if s.kind is Kind.FETCH_ENGINE else s
            for s in plan.steps
        )
        with pytest.raises(InstallFailed):
            Runner(InstallPlan(broken, plan.torch, tmp_path, plan.packs), attempts=1).run()

        assert Manifest.load(tmp_path).engine_tag == "", \
            "the manifest claims an engine version that was never installed"


class TestTheWorkspaceStep:
    """The exact step the user's install died on, in all four states."""

    def _runner(self, tmp_path) -> Runner:
        plan = plan_with_unfrozen_hash(tmp_path, "http://127.0.0.1:1")
        return Runner(plan)

    def _run_step(self, runner, monkeypatch, *, existing_version, venv_exists):
        calls = {}

        def fake_create(uv, runtime, version, *, clear=False, log=None):
            calls["clear"] = clear
            calls["version"] = version
            class R:
                ok = True
                stdout = stderr = ""
            return R()

        monkeypatch.setattr(uvtool, "venv_python_version", lambda _r: existing_version)
        monkeypatch.setattr(uvtool, "create_venv", fake_create)
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: Path("uv"))
        if venv_exists:
            (runner.runtime / "venv").mkdir(parents=True, exist_ok=True)

        logs = []
        runner.on_event = lambda e: logs.append(e.message)
        runner._create_venv(Step(Kind.CREATE_VENV, "Making a private workspace"))
        return calls, logs

    def test_a_working_workspace_is_kept(self, tmp_path, monkeypatch):
        """The user's question, answered: if it is there and it works, that is
        a good thing and we move on."""
        runner = self._runner(tmp_path)
        calls, logs = self._run_step(runner, monkeypatch,
                                     existing_version="3.12", venv_exists=True)
        assert calls == {}, "it rebuilt a workspace that was already fine"
        assert any("already here" in m for m in logs)

    def test_no_workspace_yet_creates_one_without_clearing(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)
        calls, _ = self._run_step(runner, monkeypatch,
                                  existing_version=None, venv_exists=False)
        assert calls["clear"] is False

    def test_a_half_made_workspace_is_replaced(self, tmp_path, monkeypatch):
        """Directory there, no working interpreter in it -- an interrupted
        create. This is the state that produced the reported error."""
        runner = self._runner(tmp_path)
        calls, logs = self._run_step(runner, monkeypatch,
                                     existing_version=None, venv_exists=True)
        assert calls["clear"] is True
        assert any("half-made" in m for m in logs)

    def test_a_workspace_on_the_wrong_python_is_replaced(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)
        calls, logs = self._run_step(runner, monkeypatch,
                                     existing_version="3.11", venv_exists=True)
        assert calls["clear"] is True
        assert any("3.11" in m for m in logs)

    def test_the_failure_message_names_the_path_and_the_cause(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path)

        def fake_create(uv, runtime, version, *, clear=False, log=None):
            class R:
                ok = False
                stdout = "error: Failed to create virtual environment\nno space left"
                stderr = ""
            return R()

        monkeypatch.setattr(uvtool, "venv_python_version", lambda _r: None)
        monkeypatch.setattr(uvtool, "create_venv", fake_create)
        monkeypatch.setattr(uvtool, "uv_path", lambda _r: Path("uv"))

        with pytest.raises(InstallFailed) as exc:
            runner._create_venv(Step(Kind.CREATE_VENV, "Making a private workspace"))
        assert "runtime/venv" in str(exc.value).replace("\\", "/")
        assert "no space left" in str(exc.value)


class TestUvFlags:
    """--clear and --allow-existing were read from `uv venv --help` for the
    pinned version, not remembered. --force is never passed: it lets --clear
    delete a directory that is not a virtual environment at all."""

    def test_clear_is_passed_only_when_asked(self, tmp_path):
        seen = []

        def fake_run(cmd, **kw):
            seen.append([str(c) for c in cmd])
            class R:
                ok = True
                stdout = stderr = ""
            return R()

        import toolshed.exec.uvtool as mod
        original, mod.run = mod.run, fake_run
        try:
            mod.create_venv(Path("uv"), tmp_path, "3.12")
            mod.create_venv(Path("uv"), tmp_path, "3.12", clear=True)
        finally:
            mod.run = original

        assert "--clear" not in seen[0]
        assert "--clear" in seen[1]
        assert not any("--force" in c for c in seen), \
            "--force lets --clear delete a directory that is not a venv"


class TestSudo:
    def test_installing_as_root_is_refused_with_a_reason(self, tmp_path, monkeypatch):
        """Reaching for sudo is the natural reaction to a stuck install. It
        would leave the whole data root owned by root, so the next ordinary run
        fails on permissions -- a worse state than the one being escaped."""
        import os

        monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setenv("SUDO_USER", "james")
        plan = plan_with_unfrozen_hash(tmp_path, "http://127.0.0.1:1")

        with pytest.raises(InstallFailed) as exc:
            Runner(plan).run()
        assert exc.value.reason_key == "running_as_root"
        assert "sudo" in str(exc.value)
        assert not (tmp_path / "models").exists(), "it started work before refusing"
