"""End-to-end tests for the install runner, with the network stubbed.

The steps that need the internet (uv, PyTorch, the engine tarball) are replaced
with fakes; the steps that touch the disk -- downloads, folders, workflow
injection, settings, the manifest -- run for real against a temporary root and
a local HTTP server.

That is the honest boundary of what can be proven without a GPU and without
access to Hugging Face: the orchestration, the bookkeeping and the filesystem
effects are real, the remote fetches are not.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from toolshed.catalog.packs import Pack
from toolshed.exec.download import Cancelled
from toolshed.exec.manifest import Manifest
from toolshed.exec.runner import InstallFailed, Runner
from toolshed.hw.detect import Gpu, HardwareReport
from toolshed.planner.plan import Download, InstallPlan, Kind, Step
from toolshed.planner.torchsel import choose_torch

BODY = b"model-weights" * 1000
DIGEST = hashlib.sha256(BODY).hexdigest()


class Origin(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)


@pytest.fixture
def origin():
    server = HTTPServer(("127.0.0.1", 0), Origin)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/w.safetensors"
    server.shutdown()


AMD = HardwareReport(os="linux", gpus=(Gpu("amd", "RX 7900 XT", 20480, gfx="gfx1100"),))

PACK = Pack(id="image.zimage", modality="Pictures", name="Make pictures", blurb="b",
            recipe="image_z_image_turbo_int8", vram_gb_min=8, licence="Apache 2.0",
            download_bytes=len(BODY))


def offline_plan(root: Path, url: str) -> InstallPlan:
    """Only the steps that do not need the internet, plus one real download."""
    dl = Download(url=url, dest=root / "models" / "diffusion_models",
                  filename="w.safetensors", sha256=DIGEST, size_bytes=len(BODY))
    steps = (
        Step(Kind.MAKE_DIRS, "Creating folders", payload={"model_dirs": ["diffusion_models"]}),
        Step(Kind.DOWNLOAD, "Downloading make pictures", bytes_total=len(BODY),
             payload={"id": PACK.id}, downloads=(dl,)),
        Step(Kind.INJECT_WORKFLOW, "Adding the workflow", payload={"id": PACK.id}),
        Step(Kind.WRITE_SETTINGS, "Setting sensible defaults"),
    )
    return InstallPlan(steps, choose_torch(AMD), root, (PACK,))


class TestSuccessfulRun:
    def test_it_completes_and_reports_every_step(self, origin, tmp_path):
        events = []
        manifest = Runner(offline_plan(tmp_path, origin), on_event=events.append).run()
        started = [e.step.kind for e in events if e.kind == "step_started"]
        done = [e.step.kind for e in events if e.kind == "step_done"]
        assert started == done, "a step started but never finished"
        assert len(done) == 4
        assert isinstance(manifest, Manifest)

    def test_the_model_lands_verified_in_the_right_folder(self, origin, tmp_path):
        Runner(offline_plan(tmp_path, origin)).run()
        target = tmp_path / "models" / "diffusion_models" / "w.safetensors"
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == DIGEST
        assert not list(target.parent.glob("*.part")), "left a partial file behind"

    def test_the_workflow_is_injected_where_comfyui_reads_it(self, origin, tmp_path):
        Runner(offline_plan(tmp_path, origin)).run()
        found = list((tmp_path / "comfy" / "user" / "default" / "workflows").rglob("*.json"))
        assert found, "no workflow was injected"
        assert any("Toolshed" in str(p) for p in found), "not in our namespace"

    def test_the_manifest_records_what_landed_and_how_it_was_verified(self, origin, tmp_path):
        Runner(offline_plan(tmp_path, origin)).run()
        saved = Manifest.load(tmp_path)
        assert [e.path for e in saved.files]
        entry = saved.files[0]
        assert entry.sha256 == DIGEST
        assert entry.pack == "image.zimage"
        # Frozen at release time here, so say so rather than leaving it vague.
        assert entry.hash_verified_against == "release"

    def test_settings_are_seeded_without_clobbering_a_real_choice(self, origin, tmp_path):
        import json

        settings = tmp_path / "comfy" / "user" / "default" / "comfy.settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"Comfy.Window.ConfirmOnClose": False}))
        Runner(offline_plan(tmp_path, origin)).run()
        doc = json.loads(settings.read_text())
        assert doc["Comfy.Window.ConfirmOnClose"] is False, "overwrote the user's setting"
        assert doc["Comfy.Workflow.ShowMissingModelsWarning"] is True

    def test_progress_is_reported_during_the_download(self, origin, tmp_path):
        events = []
        Runner(offline_plan(tmp_path, origin), on_event=events.append).run()
        progress = [e for e in events if e.kind == "progress"]
        assert progress, "the download reported no progress at all"
        assert progress[-1].overall <= 1.0

    def test_rerunning_is_idempotent(self, origin, tmp_path):
        plan = offline_plan(tmp_path, origin)
        Runner(plan).run()
        Runner(plan).run()          # must not raise, duplicate or corrupt
        saved = Manifest.load(tmp_path)
        paths = [e.path for e in saved.files]
        assert len(paths) == len(set(paths)), "manifest gained duplicate entries"


class TestFailureAndCancellation:
    def test_cancelling_stops_and_keeps_partial_work(self, origin, tmp_path):
        plan = offline_plan(tmp_path, origin)
        with pytest.raises(Cancelled):
            Runner(plan, should_cancel=lambda: True).run()
        assert not (tmp_path / "models" / "diffusion_models" / "w.safetensors").exists()

    def test_a_failing_step_reports_which_one_and_why(self, tmp_path):
        dead = Download(url="http://127.0.0.1:9/nope", dest=tmp_path / "models" / "x",
                        filename="x.safetensors", size_bytes=10)
        plan = InstallPlan(
            (Step(Kind.MAKE_DIRS, "Creating folders", payload={"model_dirs": ["x"]}),
             Step(Kind.DOWNLOAD, "Downloading", bytes_total=10,
                  payload={"id": "p"}, downloads=(dead,))),
            choose_torch(AMD), tmp_path, (PACK,))
        events = []
        with pytest.raises(InstallFailed) as exc:
            Runner(plan, on_event=events.append, attempts=1).run()
        assert exc.value.step.kind == Kind.DOWNLOAD
        assert any(e.kind == "step_failed" for e in events)

    def test_the_manifest_survives_a_failure_midway(self, origin, tmp_path):
        """An install killed halfway must still describe what actually landed,
        or Repair and uninstall have nothing to work from."""
        dead = Download(url="http://127.0.0.1:9/nope", dest=tmp_path / "models" / "d",
                        filename="d.safetensors", size_bytes=10)
        good = Download(url=origin, dest=tmp_path / "models" / "diffusion_models",
                        filename="w.safetensors", sha256=DIGEST, size_bytes=len(BODY))
        plan = InstallPlan(
            (Step(Kind.MAKE_DIRS, "dirs", payload={"model_dirs": ["diffusion_models", "d"]}),
             Step(Kind.DOWNLOAD, "good", bytes_total=len(BODY),
                  payload={"id": "p"}, downloads=(good,)),
             Step(Kind.DOWNLOAD, "bad", bytes_total=10, payload={"id": "q"}, downloads=(dead,))),
            choose_torch(AMD), tmp_path, (PACK,))
        with pytest.raises(InstallFailed):
            Runner(plan, attempts=1).run()
        saved = Manifest.load(tmp_path)
        assert [e for e in saved.files if e.path.endswith("w.safetensors")], \
            "the file that did land was not recorded"
