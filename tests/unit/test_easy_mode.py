"""Easy mode, driven end to end against a stand-in ComfyUI.

The brief asked for something like Automatic1111: a prompt box and a button,
rather than a node graph. Everything behind that button is protocol work --
convert the workflow, write the prompt in, queue it, follow the websocket,
fetch what came out -- and every step is somewhere the whole thing can silently
do nothing useful. The worst outcome is not a crash: it is a button that works,
ignores what you typed, and returns the template's stock picture every time.

So the server here speaks ComfyUI's actual protocol, on one port, as the real
one does: ``/object_info``, ``POST /prompt``, ``/history/{id}``, ``/view``, and
a ``/ws`` pushing ``progress_state`` then ``execution_success``. Every name and
shape was read out of v0.34.0's ``server.py`` and
``comfy_execution/progress.py``.

What it cannot check is whether the picture is any good. It checks that a
picture is asked for correctly, that the words typed reach the model, and that
what the engine says on the way is passed on rather than swallowed.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("websockets")

from comfy_fixtures import OBJECT_INFO, load_workflow  # noqa: E402
from comfy_fixtures import SPECS as OBJECT_INFO_SPECS  # noqa: E402
from websockets.datastructures import Headers  # noqa: E402
from websockets.http11 import Response  # noqa: E402
from websockets.sync.server import serve  # noqa: E402

from toolshed.easy.convert import to_api  # noqa: E402
from toolshed.easy.knobs import Settings, analyse, apply  # noqa: E402
from toolshed.exec.comfy_api import (  # noqa: E402
    ComfyClient,
    ComfyError,
    Output,
    outputs_from_history,
)


def _tiny_png() -> bytes:
    """A real 2x2 PNG, built here rather than pasted as a blob.

    It has to decode: the preview path loads it with QPixmap, and bytes that
    merely start with the PNG magic would exercise only the failure branch.
    """
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * 2 for _ in range(2))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


PICTURE_BYTES = _tiny_png()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp_easy():
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    return qtwidgets.QApplication.instance() or qtwidgets.QApplication(["tests"])


class FakeComfy:
    """A stand-in engine. HTTP and websocket on one port, like the real one."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.reject: dict | None = None
        self.fail_during_run = False
        self.server = serve(self._websocket, "127.0.0.1", 0,
                            process_request=self._http)
        self.port = self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.server.shutdown()

    # -- plain HTTP ---------------------------------------------------------

    def _http(self, connection, request) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None                        # let it upgrade

        if path == "/object_info":
            return self._json(OBJECT_INFO)
        if path.startswith("/history/"):
            return self._json({path.rsplit("/", 1)[-1]: {
                "outputs": {"9": {"images": [
                    {"filename": "Toolshed_00001_.png", "subfolder": "", "type": "output"}]}}}})
        if path == "/view":
            return Response(200, "OK", Headers({"Content-Type": "image/png",
                                                "Content-Length": str(len(PICTURE_BYTES))}),
                            PICTURE_BYTES)
        if path == "/prompt" and request.headers.get("Content-Length"):
            # websockets hands us the request before the body; the client's
            # submission is read on the websocket path instead, so this branch
            # only exists for completeness of the surface.
            return self._json({"prompt_id": "p1"})
        if path == "/interrupt":
            return self._json({})
        return Response(404, "Not Found", Headers({"Content-Length": "0"}), b"")

    @staticmethod
    def _json(payload) -> Response:
        body = json.dumps(payload).encode()
        return Response(200, "OK", Headers({"Content-Type": "application/json",
                                            "Content-Length": str(len(body))}), body)

    # -- websocket ----------------------------------------------------------

    def _websocket(self, connection) -> None:
        """Pretend to run a job as soon as a client connects.

        The real engine only reports once something is queued; here the client
        always submits immediately after connecting, so this is close enough to
        exercise the ordering the client depends on.
        """
        try:
            connection.send(json.dumps({"type": "status", "data": {"status": {}}}))
            for step in (1, 2, 3):
                connection.send(json.dumps({
                    "type": "progress_state",
                    "data": {"prompt_id": "p1", "nodes": {"5": {
                        "value": step, "max": 3, "state": "running",
                        "node_id": "5", "display_node_id": "5"}}}}))
            if self.fail_during_run:
                connection.send(json.dumps({
                    "type": "execution_error",
                    "data": {"prompt_id": "p1", "node_id": "5", "node_type": "KSampler",
                             "exception_message": "Allocation on device: out of memory"}}))
                return
            connection.send(json.dumps({
                "type": "execution_success", "data": {"prompt_id": "p1"}}))
        except Exception:                       # noqa: BLE001 -- client went away
            pass


@pytest.fixture
def engine():
    fake = FakeComfy()
    yield fake
    fake.close()


class TestTalkingToTheEngine:
    def test_it_reads_what_the_engine_can_do(self, engine):
        client = ComfyClient(base_url=engine.url)
        info = client.object_info()
        assert "KSampler" in info

    def test_a_run_reports_progress_and_returns_the_files(self, engine, monkeypatch):
        client = ComfyClient(base_url=engine.url)
        monkeypatch.setattr(client, "submit", lambda prompt: "p1")

        seen = []
        outputs = client.run({}, on_progress=seen.append)

        assert [p.value for p in seen] == [1, 2, 3]
        assert all(p.maximum == 3 for p in seen)
        assert seen[-1].fraction == 1.0
        assert outputs == [Output("Toolshed_00001_.png", "", "output", "images")]

    def test_the_picture_can_be_fetched_back(self, engine, monkeypatch, tmp_path):
        client = ComfyClient(base_url=engine.url)
        monkeypatch.setattr(client, "submit", lambda prompt: "p1")
        item = client.run({})[0]
        saved = client.download(item, tmp_path / "out.png")
        assert saved.read_bytes() == PICTURE_BYTES

    def test_an_engine_failure_is_translated_not_swallowed(self, engine, monkeypatch):
        """Out of memory is the commonest real failure and the least
        self-explanatory, so it gets said in words."""
        engine.fail_during_run = True
        client = ComfyClient(base_url=engine.url)
        monkeypatch.setattr(client, "submit", lambda prompt: "p1")

        with pytest.raises(ComfyError) as exc:
            client.run({})
        assert exc.value.reason_key == "execution_error"
        assert "ran out of memory" in str(exc.value)
        assert "smaller size" in str(exc.value)

    def test_cancelling_stops_waiting(self, engine, monkeypatch):
        client = ComfyClient(base_url=engine.url)
        monkeypatch.setattr(client, "submit", lambda prompt: "p1")
        with pytest.raises(ComfyError) as exc:
            client.run({}, should_cancel=lambda: True)
        assert exc.value.reason_key == "cancelled"

    def test_the_websocket_url_is_derived_from_the_engine_url(self):
        client = ComfyClient(base_url="http://127.0.0.1:8188", client_id="abc")
        assert client._ws_url() == "ws://127.0.0.1:8188/ws?clientId=abc"


class TestReadingWhatCameOut:
    def test_pictures_audio_video_and_models_are_all_found(self):
        history = {"outputs": {
            "1": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "2": {"audio": [{"filename": "b.mp3", "subfolder": "audio", "type": "output"}]},
            "3": {"video": [{"filename": "c.mp4", "subfolder": "", "type": "output"}]},
            "4": {"3d": [{"filename": "d.glb", "subfolder": "", "type": "output"}]},
        }}
        kinds = {o.kind for o in outputs_from_history(history)}
        assert kinds == {"images", "audio", "video", "3d"}

    def test_a_run_that_made_nothing_is_not_an_error(self):
        assert outputs_from_history({"outputs": {}}) == []

    def test_junk_entries_are_ignored(self):
        history = {"outputs": {"1": {"images": [{}, {"filename": "ok.png"}]}}}
        assert [o.filename for o in outputs_from_history(history)] == ["ok.png"]


class TestTheWholeChain:
    """Workflow on disk -> API graph -> the person's words written in."""

    def test_what_you_type_reaches_the_model(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="a red bicycle in the rain"))

        texts = [n["inputs"]["text"] for n in graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        assert "a red bicycle in the rain" in texts

    def test_the_template_text_is_replaced_not_appended(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="a red bicycle"))
        positive = graph[knobs.positive[0].node_id]["inputs"]["text"]
        assert positive == "a red bicycle"
        assert "marble statue" not in positive

    def test_the_negative_box_does_not_overwrite_the_prompt(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="a cat", negative="blurry"))
        assert graph[knobs.positive[0].node_id]["inputs"]["text"] == "a cat"
        assert graph[knobs.negative[0].node_id]["inputs"]["text"] == "blurry"

    def test_the_prompt_reaches_inside_a_subgraph(self):
        """Z-Image is the default pack and hides its encoder inside a subgraph.
        If this fails, easy mode silently makes the template's picture."""
        workflow = load_workflow("image/01 Text to picture (Z-Image).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="a paper boat"))
        texts = [n["inputs"]["text"] for n in graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        assert texts == ["a paper boat"]

    def test_a_shared_encoder_is_never_rewritten_by_the_negative_box(self):
        """Z-Image has no separate negative encoder: it takes the same
        CLIPTextEncode and runs it through ConditioningZeroOut. Both branches
        therefore trace back to the one node holding the actual prompt, and
        writing the negative box into it would replace what the user asked for
        with what they asked to avoid."""
        workflow = load_workflow("image/01 Text to picture (Z-Image).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)

        assert knobs.positive, "no prompt box target found"
        shared = {t.node_id for t in knobs.positive} & {t.node_id for t in knobs.negative}
        assert not shared, "the negative box points at the node holding the prompt"

        graph = apply(prompt, knobs, Settings(prompt="a paper boat", negative="blurry"))
        texts = [n["inputs"]["text"] for n in graph.values()
                 if n["class_type"] == "CLIPTextEncode"]
        assert texts == ["a paper boat"]

    def test_every_run_gets_a_new_seed(self):
        """Pressing the button twice and getting the identical picture reads as
        the button being broken. It is the most confusing thing these tools do."""
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        seeds = {apply(prompt, knobs, Settings(prompt="x"))[knobs.seeds[0].node_id]
                 ["inputs"]["seed"] for _ in range(8)}
        assert len(seeds) > 1, "the same seed came back every time"

    def test_a_fixed_seed_is_honoured(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="x", seed=1234))
        assert graph[knobs.seeds[0].node_id]["inputs"]["seed"] == 1234

    def test_the_size_can_be_changed(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="x", width=768, height=512))
        latent = next(n for n in graph.values() if n["class_type"] == "EmptyLatentImage")
        assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (768, 512)

    def test_one_run_does_not_inherit_the_last_ones_settings(self):
        """apply() must copy. Writing in place would mean clearing the prompt
        box left the previous prompt in the graph."""
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        original = prompt[knobs.positive[0].node_id]["inputs"]["text"]
        apply(prompt, knobs, Settings(prompt="something else"))
        assert prompt[knobs.positive[0].node_id]["inputs"]["text"] == original

    def test_leaving_a_box_empty_keeps_the_templates_value(self):
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt=None))
        assert graph[knobs.positive[0].node_id]["inputs"]["text"].startswith("a giant")


class TestWhatEasyModeOffers:
    def test_only_installed_packs_are_offered(self):
        from toolshed.ui.make import recipes_for

        offered = recipes_for(["image.sdxl", "audio.acestep"])
        assert [r.pack_id for r in offered] == ["image.sdxl", "audio.acestep"]

    def test_a_pack_that_is_not_installed_is_not_offered(self):
        from toolshed.ui.make import recipes_for

        assert recipes_for([]) == []

    def test_an_unknown_pack_is_skipped_rather_than_crashing(self):
        from toolshed.ui.make import recipes_for

        assert recipes_for(["nonsense.pack"]) == []

    def test_every_pack_has_a_button_label_and_a_workflow(self):
        from toolshed.exec.inject import PACK_FOLDERS
        from toolshed.ui.make import VERB, recipes_for

        for pack_id in PACK_FOLDERS:
            assert pack_id in VERB, f"{pack_id} has no button label"
            assert recipes_for([pack_id]), f"{pack_id} has no workflow in the bundle"


class SubmitServer(BaseHTTPRequestHandler):
    """A plain HTTP stand-in, so the actual POST /prompt path is exercised.

    The websocket server above hands `process_request` the request before its
    body arrives, so the graph we send could not be inspected there -- and the
    submission is precisely where the graph goes.
    """

    received: list[dict] = []
    reject_with: dict | None = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).received.append(json.loads(body))
        if type(self).reject_with is not None:
            payload = json.dumps(type(self).reject_with).encode()
            self.send_response(400)
        else:
            payload = json.dumps({"prompt_id": "p1"}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def submit_server():
    SubmitServer.received = []
    SubmitServer.reject_with = None
    server = HTTPServer(("127.0.0.1", 0), SubmitServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


class TestSubmitting:
    def _client(self, server) -> ComfyClient:
        return ComfyClient(base_url=f"http://127.0.0.1:{server.server_port}")

    def test_the_graph_and_the_client_id_are_sent(self, submit_server):
        client = self._client(submit_server)
        workflow = load_workflow("image/02 Text to picture (SDXL).json")
        prompt = to_api(workflow, OBJECT_INFO_SPECS)
        knobs = analyse(prompt)
        graph = apply(prompt, knobs, Settings(prompt="a red bicycle"))

        assert client.submit(graph) == "p1"

        sent = SubmitServer.received[0]
        assert sent["client_id"] == client.client_id
        texts = [n["inputs"]["text"] for n in sent["prompt"].values()
                 if n["class_type"] == "CLIPTextEncode"]
        assert "a red bicycle" in texts
        assert "MarkdownNote" not in {n["class_type"] for n in sent["prompt"].values()}

    def test_a_missing_model_is_explained_in_plain_words(self, submit_server):
        """The commonest real rejection. "value_not_in_list: [] " means nothing
        to someone who just wanted a picture."""
        SubmitServer.reject_with = {
            "error": {"type": "prompt_outputs_failed_validation",
                      "message": "Prompt outputs failed validation"},
            "node_errors": {"15": {"class_type": "CheckpointLoaderSimple", "errors": [
                {"type": "value_not_in_list", "message": "Value not in list",
                 "details": "ckpt_name: 'sd_xl_base_1.0.safetensors' not in []"}]}},
        }
        client = self._client(submit_server)
        with pytest.raises(ComfyError) as exc:
            client.submit({})
        assert exc.value.reason_key == "model_missing"
        assert "not on your computer" in str(exc.value)
        assert "sd_xl_base_1.0.safetensors" in exc.value.detail

    def test_another_rejection_keeps_the_engines_own_words(self, submit_server):
        SubmitServer.reject_with = {
            "error": {"type": "required_input_missing",
                      "message": "Required input is missing"},
            "node_errors": {},
        }
        client = self._client(submit_server)
        with pytest.raises(ComfyError) as exc:
            client.submit({})
        assert "Required input is missing" in str(exc.value)

    def test_an_engine_that_is_not_there_says_so(self):
        client = ComfyClient(base_url="http://127.0.0.1:1")
        with pytest.raises(ComfyError) as exc:
            client.submit({})
        assert exc.value.reason_key == "unreachable"


class TestTheSizeBoxes:
    """Opening More settings must not silently change what you asked for."""

    def test_untouched_size_boxes_mean_leave_it_alone(self, qapp_easy):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.more.setChecked(True)
        settings = page.settings()
        assert settings.width is None and settings.height is None, \
            "merely opening More settings asked for a different size"

    def test_the_boxes_say_so_rather_than_showing_a_misleading_number(self, qapp_easy):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        assert page.width.value() == 0
        assert page.width.specialValueText()

    def test_a_chosen_size_is_sent(self, qapp_easy):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.width.setValue(768)
        page.height.setValue(512)
        settings = page.settings()
        assert (settings.width, settings.height) == (768, 512)

    def test_the_workflows_own_size_fills_the_boxes_in(self, qapp_easy):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page._on_analysed(1024, 1024)
        assert (page.width.value(), page.height.value()) == (1024, 1024)

    def test_a_size_the_user_chose_is_not_overwritten(self, qapp_easy):
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.width.setValue(768)
        page._on_analysed(1024, 1024)
        assert page.width.value() == 768


class TestTheReadyMessage:
    def test_it_stops_saying_start_comfyui_once_it_is_started(self, qapp_easy):
        from toolshed.exec.comfy_api import ComfyClient
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.set_packs(["image.sdxl"])
        assert "Start ComfyUI" in page.status.text()

        page.set_engine(ComfyClient(base_url="http://127.0.0.1:1"))
        assert page.status.text() == ""
        assert page.go.isEnabled()

    def test_it_says_so_when_no_pack_can_be_driven(self, qapp_easy):
        from toolshed.exec.comfy_api import ComfyClient
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        page.set_engine(ComfyClient(base_url="http://127.0.0.1:1"))
        page.set_packs([])
        assert "Nothing installed" in page.status.text()
        assert not page.go.isEnabled()
