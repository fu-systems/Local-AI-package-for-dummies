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
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

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
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"

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
            # PreviewUI3D: "result" is [model file, camera info, ...]; the
            # file is a plain string, not an {filename, subfolder} record.
            "4": {"result": ["3d/d.glb", {"position": [0, 0, 1]}]},
        }}
        found = outputs_from_history(history)
        assert {o.kind for o in found} == {"images", "audio", "video", "3d"}
        model = next(o for o in found if o.kind == "3d")
        assert (model.filename, model.subfolder) == ("d.glb", "3d")

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
    uploads: list[bytes] = []
    reject_with: dict | None = None
    upload_reply: dict | None = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.startswith("/upload/"):
            type(self).uploads.append(body)
            payload = json.dumps(type(self).upload_reply or {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
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
    SubmitServer.uploads = []
    SubmitServer.reject_with = None
    SubmitServer.upload_reply = None
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
        try:
            page.set_packs(["image.sdxl"])
            assert "Start ComfyUI" in page.status.text()

            page.set_engine(ComfyClient(base_url="http://127.0.0.1:1"))
            assert page.status.text() == ""
            assert page.go.isEnabled()
        finally:
            page.shutdown()

    def test_it_says_so_when_no_pack_can_be_driven(self, qapp_easy):
        from toolshed.exec.comfy_api import ComfyClient
        from toolshed.ui.make import MakePage

        page = MakePage(Path("/tmp/nowhere"))
        try:
            page.set_engine(ComfyClient(base_url="http://127.0.0.1:1"))
            page.set_packs([])
            assert "Nothing installed" in page.status.text()
            assert not page.go.isEnabled()
        finally:
            page.shutdown()


IMAGE_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "LoadImage", "inputs": [],
         "widgets_values_named": {"image": "viking_wolf_rune_axe.png", "upload": "image"},
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}]},
        {"id": 2, "type": "VAEDecode",
         "inputs": [{"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None}],
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}]},
        {"id": 3, "type": "SaveImage",
         "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
         "widgets_values_named": {"filename_prefix": "out"}, "outputs": []},
    ],
    "links": [[1, 1, 0, 3, 0, "IMAGE"]],
}


class TestAWorkflowThatStartsFromAPicture:
    """The 3D pack could be chosen and then had nowhere to put the photo.

    The shipped workflow carries the template's own filename --
    viking_wolf_rune_axe.png -- which is not on anybody else's machine, so
    running it unchanged asks the engine for a file it does not have.
    """

    def test_the_picture_input_is_found(self):
        knobs = analyse(to_api(IMAGE_WORKFLOW, OBJECT_INFO_SPECS), OBJECT_INFO_SPECS)
        assert knobs.takes_picture
        assert knobs.images[0].input_name == "image"

    def test_it_is_found_from_the_engines_metadata_not_the_class_name(self):
        """image_upload marks it, so a node other than LoadImage works too."""
        from toolshed.easy.convert import specs_from_object_info

        specs = specs_from_object_info({
            **OBJECT_INFO,
            "LoadSomethingElse": {"input": {"required": {
                "picture": [["a.png"], {"image_upload": True}]}}, "output": ["IMAGE"]},
        })
        workflow = {
            "nodes": [{"id": 1, "type": "LoadSomethingElse", "inputs": [],
                       "widgets_values_named": {"picture": "a.png"},
                       "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}]},
                      {"id": 2, "type": "SaveImage",
                       "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                       "widgets_values_named": {"filename_prefix": "o"}, "outputs": []}],
            "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        }
        knobs = analyse(to_api(workflow, specs), specs)
        assert knobs.takes_picture
        assert knobs.images[0].input_name == "picture"

    def test_the_chosen_picture_replaces_the_templates_filename(self):
        prompt = to_api(IMAGE_WORKFLOW, OBJECT_INFO_SPECS)
        knobs = analyse(prompt, OBJECT_INFO_SPECS)
        graph = apply(prompt, knobs, Settings(image="my-photo.png"))
        loader = next(n for n in graph.values() if n["class_type"] == "LoadImage")
        assert loader["inputs"]["image"] == "my-photo.png"
        assert "viking" not in str(loader["inputs"])

    def test_the_real_3d_workflow_has_one(self):
        """The pack the complaint was about."""
        import json

        workflow = load_workflow("3d/01 Photo to 3D model.json")
        loaders = [n for n in workflow["nodes"] if n["type"] == "LoadImage"]
        assert loaders, "the 3D workflow has no LoadImage to attach a photo to"
        assert "viking" in json.dumps(loaders[0].get("widgets_values_named")), \
            "the template default changed; the picture box still has to override it"


class TestUploadingThePicture:
    def test_it_is_sent_and_the_engine_names_it(self, submit_server, tmp_path):
        photo = tmp_path / "my photo.png"
        photo.write_bytes(PICTURE_BYTES)

        SubmitServer.upload_reply = {"name": "my photo.png", "subfolder": "",
                                     "type": "input"}
        client = ComfyClient(base_url=f"http://127.0.0.1:{submit_server.server_port}")
        assert client.upload_image(photo) == "my photo.png"

        sent = SubmitServer.uploads[0]
        assert b"my photo.png" in sent
        assert PICTURE_BYTES in sent

    def test_a_subfolder_is_included_in_the_name(self, submit_server, tmp_path):
        photo = tmp_path / "p.png"
        photo.write_bytes(PICTURE_BYTES)
        SubmitServer.upload_reply = {"name": "p.png", "subfolder": "toolshed",
                                     "type": "input"}
        client = ComfyClient(base_url=f"http://127.0.0.1:{submit_server.server_port}")
        assert client.upload_image(photo) == "toolshed/p.png"

    def test_an_engine_that_will_not_take_it_says_so(self, tmp_path):
        photo = tmp_path / "p.png"
        photo.write_bytes(PICTURE_BYTES)
        client = ComfyClient(base_url="http://127.0.0.1:1")
        with pytest.raises(ComfyError) as exc:
            client.upload_image(photo)
        assert exc.value.reason_key == "upload_failed"


class TestEverySettingIsReachable:
    """"all settings and options available that we can do" -- every widget the
    engine reports, not the handful with friendly names."""

    def _knobs(self, rel="image/02 Text to picture (SDXL).json"):
        prompt = to_api(load_workflow(rel), OBJECT_INFO_SPECS)
        return prompt, analyse(prompt, OBJECT_INFO_SPECS)

    def test_the_sampler_settings_are_all_offered(self):
        _, knobs = self._knobs()
        offered = {c.input_name for c in knobs.advanced}
        assert {"cfg", "sampler_name", "scheduler", "denoise", "batch_size"} <= offered

    def test_a_choice_carries_the_engines_own_options(self):
        _, knobs = self._knobs()
        sampler = next(c for c in knobs.advanced if c.input_name == "sampler_name")
        assert sampler.kind == "choice"
        assert "euler" in sampler.choices

    def test_numbers_carry_their_bounds(self):
        _, knobs = self._knobs()
        cfg = next(c for c in knobs.advanced if c.input_name == "cfg")
        assert cfg.kind == "float"
        assert cfg.maximum == 100.0

    def test_wired_inputs_are_not_offered_as_boxes(self):
        """A connected input is driven by another node; a box for it would be
        offering a value the engine is going to ignore."""
        _, knobs = self._knobs()
        offered = {(c.node_id, c.input_name) for c in knobs.controls}
        assert not any(name in {"model", "positive", "negative", "latent_image",
                                "samples", "vae", "clip", "images"}
                       for _, name in offered)

    def test_a_widget_input_that_is_wired_up_is_not_offered(self):
        """The case the type check alone does not cover.

        width is an INT -- a widget by type -- but here another node supplies
        it. A box for it would take a number the engine is going to ignore, and
        the user would be left wondering why the size never changed.
        """
        from toolshed.easy.convert import specs_from_object_info

        specs = specs_from_object_info({
            **OBJECT_INFO,
            "PrimitiveInt": {"input": {"required": {
                "value": ["INT", {"default": 512, "min": 16, "max": 16384}]}},
                "output": ["INT"]},
        })
        workflow = {
            "nodes": [
                {"id": 1, "type": "PrimitiveInt", "inputs": [],
                 "widgets_values_named": {"value": 768},
                 "outputs": [{"name": "INT", "type": "INT", "links": [1]}]},
                {"id": 2, "type": "EmptyLatentImage",
                 "inputs": [{"name": "width", "type": "INT",
                             "widget": {"name": "width"}, "link": 1}],
                 "widgets_values_named": {"width": 512, "height": 512, "batch_size": 1},
                 "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}]},
            ],
            "links": [[1, 1, 0, 2, 0, "INT"]],
        }
        prompt = to_api(workflow, specs)
        assert isinstance(prompt["2"]["inputs"]["width"], list), "not wired in the graph"

        knobs = analyse(prompt, specs)
        offered = {(c.node_id, c.input_name) for c in knobs.controls}
        assert ("2", "width") not in offered, "offered a box for a value the engine ignores"
        assert ("2", "height") in offered, "the unwired ones should still be offered"

    def test_model_filenames_are_offered_as_a_choice(self):
        """These were hidden on the reasoning that the pack decided them. That
        was removing a function rather than defaulting one -- the engine builds
        the list from the files actually on disk, so every option is a model
        the user has, and switching checkpoint is one of the first things
        anyone wants to try."""
        _, knobs = self._knobs()
        checkpoint = next(c for c in knobs.advanced if c.input_name == "ckpt_name")
        assert checkpoint.kind == "choice"
        assert checkpoint.choices
        assert checkpoint.default == "sd_xl_base_1.0.safetensors"

    def test_the_only_thing_withheld_is_not_an_input_at_all(self):
        """control_after_generate is invented by the editor to sit beside a
        seed. The engine has never heard of it."""
        _, knobs = self._knobs()
        assert not any(c.input_name == "control_after_generate" for c in knobs.controls)

    def test_every_control_remembers_what_it_started_as(self):
        """Which is what lets the screen say that leaving it alone is fine,
        and put it back when it is not."""
        _, knobs = self._knobs()
        assert all(c.default is not None for c in knobs.advanced)

    def test_the_named_controls_are_not_repeated_in_the_list(self):
        """Showing the prompt twice invites setting it in both places and
        getting whichever the code writes last."""
        _, knobs = self._knobs()
        assert all(not c.role for c in knobs.advanced)
        assert {c.role for c in knobs.controls if c.role} >= {
            "prompt", "negative", "width", "height", "seed", "steps"}

    def test_an_override_reaches_the_graph(self):
        prompt, knobs = self._knobs()
        cfg = next(c for c in knobs.advanced if c.input_name == "cfg")
        graph = apply(prompt, knobs, Settings(overrides={cfg.key: 3.5}))
        assert graph[cfg.node_id]["inputs"]["cfg"] == 3.5

    def test_an_override_beats_an_inferred_value(self):
        """An explicit edit is the user being specific."""
        prompt, knobs = self._knobs()
        width = knobs.width[0]
        graph = apply(prompt, knobs,
                      Settings(width=512, overrides={(width.node_id, "width"): 768}))
        assert graph[width.node_id]["inputs"]["width"] == 768

    def test_without_object_info_there_are_no_controls(self):
        """Type and bounds are not knowable from a value alone, and inventing
        them would produce boxes that accept what the engine rejects."""
        prompt = to_api(load_workflow("image/02 Text to picture (SDXL).json"),
                        OBJECT_INFO_SPECS)
        assert analyse(prompt).controls == []
        assert analyse(prompt).positive, "roles should still be found"


class TestThePictureBoxOnScreen:
    """The complaint: the 3D pack could be chosen and there was nowhere to put
    the photo. These drive the real page rather than the plumbing under it."""

    def _page(self, tmp_path, workflow: dict, pack="model3d.trellis2"):
        from toolshed.ui.make import MakePage, Recipe

        path = tmp_path / "wf.json"
        path.write_text(json.dumps(workflow))
        page = MakePage(tmp_path / "root")
        page.recipes = [Recipe(pack, "Turn a photo into a 3D model", path)]
        page.what.addItem(page.recipes[0].name)
        return page

    def _inspected(self, page, engine):
        import time

        from toolshed.exec.comfy_api import ComfyClient

        page.set_engine(ComfyClient(base_url=engine.url))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and page.knobs is None:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.02)
        QtWidgets.QApplication.processEvents()
        return page.knobs is not None

    def test_a_picture_workflow_gets_a_picture_box(self, qapp_easy, engine, tmp_path):
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine), "the workflow was never inspected"
            assert not page.picture_row.isHidden(), "no way to add the picture"
            assert page.knobs.takes_picture
        finally:
            page.shutdown()

    def test_it_runs_without_choosing_anything(self, qapp_easy, engine, tmp_path):
        """Easy mode means the button always works.

        This used to leave the button unpressable until a picture was chosen,
        which is the opposite of the point. A picture ships with Toolshed and
        is filled in, so pressing the button makes something -- the template's
        own sample filename, which is on nobody else's machine, never gets
        asked for.
        """
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine)
            assert page.go.isEnabled(), "the button was not pressable out of the box"
            assert page.picture is not None, "no starting picture was filled in"
            assert page.picture.is_file()
        finally:
            page.shutdown()

    def test_the_filled_in_picture_says_it_is_a_default(self, qapp_easy, engine,
                                                        tmp_path):
        """Optional is only useful if it is understood as optional."""
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine)
            assert "came with Toolshed" in page.picture_name.text()
        finally:
            page.shutdown()

    def test_choosing_one_enables_it_and_clears_the_message(self, qapp_easy, engine,
                                                            tmp_path):
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine)
            photo = tmp_path / "my-photo.png"
            photo.write_bytes(PICTURE_BYTES)
            page.set_picture(photo)

            assert page.go.isEnabled()
            assert "starting picture" not in page.status.text(), \
                "the message outlived the thing it was asking for"
            assert page.picture_name.text() == "my-photo.png"
        finally:
            page.shutdown()

    def test_clearing_returns_to_the_starter_rather_than_to_nothing(
            self, qapp_easy, engine, tmp_path):
        """There is no state in which the button stops working."""
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine)
            photo = tmp_path / "p.png"
            photo.write_bytes(PICTURE_BYTES)
            page.set_picture(photo)
            assert page.picture == photo

            page.clear_picture()
            assert page.picture is not None
            assert page.go.isEnabled()
            assert "came with Toolshed" in page.picture_name.text()
        finally:
            page.shutdown()

    def test_a_text_workflow_has_no_picture_box(self, qapp_easy, engine, tmp_path):
        page = self._page(tmp_path, load_workflow("image/02 Text to picture (SDXL).json"),
                          pack="image.sdxl")
        try:
            assert self._inspected(page, engine)
            assert page.picture_row.isHidden()
            assert not page.prompt.isHidden()
            assert page.go.isEnabled()
        finally:
            page.shutdown()

    def test_a_picture_workflow_with_no_words_hides_the_prompt_box(
            self, qapp_easy, engine, tmp_path):
        """Asking for a description when nothing reads one is a box that does
        nothing."""
        page = self._page(tmp_path, IMAGE_WORKFLOW)
        try:
            assert self._inspected(page, engine)
            assert page.prompt.isHidden()
        finally:
            page.shutdown()

    def test_the_settings_panel_is_built_from_the_workflow(self, qapp_easy, engine,
                                                           tmp_path):
        page = self._page(tmp_path, load_workflow("image/02 Text to picture (SDXL).json"),
                          pack="image.sdxl")
        try:
            assert self._inspected(page, engine)
            names = {row.control.input_name for row in page.rows}
            assert {"cfg", "sampler_name", "scheduler", "denoise", "ckpt_name"} <= names
            assert "control_after_generate" not in names
            assert page.settings().overrides, "nothing would be sent"
        finally:
            page.shutdown()

    def test_a_worker_left_running_does_not_take_the_app_down(self, qapp_easy, tmp_path):
        """Qt aborts the process if a QThread is destroyed while running, so a
        page closed mid-inspection would kill the app."""
        from toolshed.exec.comfy_api import ComfyClient
        from toolshed.ui.make import MakePage

        page = self._page(tmp_path, IMAGE_WORKFLOW)
        page.set_engine(ComfyClient(base_url="http://127.0.0.1:1"))
        page.shutdown()
        assert page.inspector is None
        assert isinstance(page, MakePage)


class TestPressingGoWithoutTouchingAnything:
    """Easy mode's one promise: the button works, always.

    Every workflow we ship, straight after being picked, with nothing typed
    and nothing opened, must produce a graph the engine would accept.
    """

    def _page(self, tmp_path, rel, pack):
        from toolshed.ui.make import MakePage, Recipe

        page = MakePage(tmp_path / "root")
        page.recipes = [Recipe(pack, "whatever", WORKFLOWS / rel)]
        page.what.addItem(page.recipes[0].name)
        return page

    def test_the_starter_picture_is_shipped_and_is_a_real_image(self):
        from toolshed.ui.make import starter_picture

        starter = starter_picture()
        assert starter is not None and starter.is_file()
        assert starter.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        pixmap = QtWidgets.QApplication.instance() and None
        del pixmap
        from PySide6 import QtGui
        image = QtGui.QImage(str(starter))
        assert not image.isNull(), "the starter picture does not decode"
        assert image.width() >= 256, "too small to be a useful subject"

    def test_it_is_bundled_into_the_build(self):
        """A resource the frozen app cannot find is not shipped at all."""
        import sys

        sys.path.insert(0, str(REPO_ROOT / "packaging"))
        from _spec_common import DATA_DIRS

        assert "assets" in DATA_DIRS

    def test_the_defaults_alone_produce_a_runnable_graph(self):
        """Nothing typed, nothing opened: what would be sent?"""
        for rel in ["image/02 Text to picture (SDXL).json",
                    "image/01 Text to picture (Z-Image).json"]:
            prompt = to_api(load_workflow(rel), OBJECT_INFO_SPECS)
            knobs = analyse(prompt, OBJECT_INFO_SPECS)
            graph = apply(prompt, knobs, Settings())

            assert graph, rel
            for node_id, node in graph.items():
                for name, value in node["inputs"].items():
                    assert value is not None, f"{rel}: {node['class_type']}.{name} is unset"
                    if isinstance(value, list) and len(value) == 2:
                        assert value[0] in graph, f"{rel}: {name} points nowhere"
                assert node_id

    def test_an_untouched_settings_panel_changes_nothing(self, qapp_easy, engine,
                                                         tmp_path):
        """Opening the panel and closing it again must not alter the result."""
        page = self._page(tmp_path, "image/02 Text to picture (SDXL).json", "image.sdxl")
        try:
            import time

            from toolshed.exec.comfy_api import ComfyClient
            page.set_engine(ComfyClient(base_url=engine.url))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and page.knobs is None:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.02)
            assert page.knobs is not None

            prompt = to_api(load_workflow("image/02 Text to picture (SDXL).json"),
                            OBJECT_INFO_SPECS)
            knobs = analyse(prompt, OBJECT_INFO_SPECS)
            untouched = apply(prompt, knobs, page.settings())

            for row in page.rows:
                node = untouched[row.control.node_id]["inputs"]
                assert node[row.control.input_name] == row.control.default, \
                    f"{row.control.input_name} changed just by being shown"
        finally:
            page.shutdown()

    def test_reset_puts_everything_back(self, qapp_easy, engine, tmp_path):
        page = self._page(tmp_path, "image/02 Text to picture (SDXL).json", "image.sdxl")
        try:
            import time

            from toolshed.exec.comfy_api import ComfyClient
            page.set_engine(ComfyClient(base_url=engine.url))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and page.knobs is None:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.02)
            assert page.rows

            for row in page.rows:
                if row.control.input_name == "cfg":
                    row.widget.setValue(99.0)
                if row.control.input_name == "sampler_name":
                    row.widget.setCurrentText("euler")
            page.negative.setText("blurry")
            page.width.setValue(512)

            page.reset_settings()

            for row in page.rows:
                assert row.value() == row.control.default or \
                    str(row.value()) == str(row.control.default), row.control.input_name
            assert page.negative.text() == ""
        finally:
            page.shutdown()

    def test_the_panel_says_it_is_optional(self, qapp_easy, tmp_path):
        from toolshed.ui.make import MakePage

        page = MakePage(tmp_path / "root")
        try:
            assert "optional" in page.all_settings.title().lower()
        finally:
            page.shutdown()
