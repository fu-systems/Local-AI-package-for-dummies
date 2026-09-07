"""Talking to a running ComfyUI over HTTP, as any other client would.

This is the GPL boundary in practice: ComfyUI is a separate process and we are
one of its API clients. Nothing here imports it.

Every route, field and message name below was read from ComfyUI v0.34.0's own
``server.py`` and ``comfy_execution/progress.py``:

* ``POST /prompt`` takes ``{"prompt": ..., "client_id": ...}`` and answers with
  ``prompt_id``, or a 400 carrying ``error`` and ``node_errors``.
* ``GET /history/{prompt_id}`` holds the outputs once a run finishes.
* ``GET /view`` fetches one produced file by filename, subfolder and type.
* ``GET /object_info`` describes every node the engine has.
* ``/ws?clientId=...`` streams ``progress_state``, ``executing``, ``executed``,
  ``execution_error``, ``execution_cached``, ``execution_interrupted`` and
  ``execution_success``.

Progress comes over the websocket because there is no HTTP equivalent -- and
"it froze at 70 percent with no idea what it was doing" is the complaint this
whole product exists to avoid, so a spinner alone is not good enough.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

# Long enough for a cold model load inside a single node.
HTTP_TIMEOUT = httpx.Timeout(30.0, read=300.0)
# How long the engine may say nothing before we assume it is wedged. Checked
# in short slices so a Stop is honoured within a second, not at the end of a
# five-minute model load.
WS_SILENCE_BUDGET = 300.0
WS_POLL = 1.0
# The engine emits execution_success *before* it stores the history entry
# (execution.py: add_message at 824, history_result set at 831, stored by
# task_done). Ask a few times rather than read an empty record once.
HISTORY_ATTEMPTS = 40
HISTORY_WAIT = 0.25

# Every call here is to the loopback engine. Proxies must not be consulted:
# httpx honours HTTP(S)_PROXY by default, and behind one every request to
# 127.0.0.1 was sent to the proxy instead, so readiness never came.
NO_PROXY = {"trust_env": False}


class ComfyError(RuntimeError):
    """The engine refused or failed a request, with something worth showing."""

    def __init__(self, message: str, *, reason_key: str = "comfy_error",
                 detail: str = "") -> None:
        super().__init__(message)
        self.reason_key = reason_key
        self.detail = detail


@dataclass(frozen=True)
class Output:
    """One file a run produced."""

    filename: str
    subfolder: str
    type: str
    kind: str          # images | audio | video | 3d | files

    @property
    def is_picture(self) -> bool:
        return self.kind == "images"

    @property
    def is_saved(self) -> bool:
        """Is this a file that is still there after the run?

        Previews go to ComfyUI's temp directory, which it empties. Telling
        someone their model is in the output folder when it was a preview is
        how "it says it worked and there is nothing there" happens.
        """
        return self.type == "output"


@dataclass
class Progress:
    """How far along a run is, in terms a person can read."""

    value: int = 0
    maximum: int = 0
    node_title: str = ""
    stage: str = ""

    @property
    def fraction(self) -> float:
        return 0.0 if self.maximum <= 0 else min(1.0, self.value / self.maximum)


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]

# Which history output keys hold files. Read from comfy_api/latest/_ui.py, not
# assumed: SaveImage writes "images"; SaveAudio writes "audio"; SaveVideo goes
# through ui.PreviewVideo, which writes "images" with "animated": [true]; the
# 3D savers go through ui.PreviewUI3D(Advanced), which writes
# "result": [model_file, camera_info, ...] with model_file a plain filename.
# The first version of this table had a "3d" key that nothing ever writes, so
# a finished 3D model was reported as "produced no files".
OUTPUT_KEYS = ("images", "audio", "video", "gifs", "files")

# 3D nodes that write a preview into the temp directory and report it in the
# same "result" shape a saver uses. Read from comfy_extras/nodes_load_3d.py at
# v0.34.0, where Preview3DAdvanced does:
#
#     filename = f"preview3d_advanced_{uuid.uuid4().hex}.{model_3d.format}"
#     model_3d.save_to(os.path.join(folder_paths.get_temp_directory(), filename))
#     ... ui=UI.PreviewUI3DAdvanced(filename, ...)
#
# while Save3DAdvanced goes through _save_file3d_to_output, which writes into
# folder_paths.get_output_directory() and returns "<subfolder>/<name>".
#
# History does not say which is which -- both arrive as {"result": [name, ...]}
# -- so the node class is the only honest way to tell them apart, and the
# converted graph is where that lives. Calling all of them "output" made the 3D
# pack report three finished models where one had been saved, and sent anyone
# looking for the other two to a folder that never held them.
PREVIEW_3D_CLASSES = frozenset({
    "Preview3D", "Preview3DAdvanced", "PreviewGaussianSplat",
})
PREVIEW_3D_VERIFIED_AGAINST = "v0.34.0"


@dataclass
class ComfyClient:
    """One client per engine. Cheap to make, safe to keep."""

    base_url: str
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    # -- reading ------------------------------------------------------------

    def object_info(self) -> dict:
        """Every node class the engine has, with its inputs.

        This is what makes the workflow converter authoritative rather than a
        table of names in our source that would rot at the next ComfyUI
        release.
        """
        try:
            reply = httpx.get(self._url("/object_info"), timeout=HTTP_TIMEOUT, trust_env=False)
            reply.raise_for_status()
            return reply.json()
        except httpx.HTTPError as exc:
            raise ComfyError("Could not ask ComfyUI what it can do.",
                             reason_key="no_object_info", detail=str(exc)) from exc

    def history(self, prompt_id: str) -> dict:
        try:
            reply = httpx.get(self._url(f"/history/{prompt_id}"),
                              timeout=HTTP_TIMEOUT, trust_env=False)
            reply.raise_for_status()
            return reply.json().get(prompt_id) or {}
        except httpx.HTTPError:
            return {}

    def view_url(self, item: Output) -> str:
        params = httpx.QueryParams(
            filename=item.filename, subfolder=item.subfolder, type=item.type)
        return self._url(f"/view?{params}")

    def upload_image(self, path: Path, *, subfolder: str = "") -> str:
        """Put a local picture where the engine can load it, and say what to
        call it.

        A workflow that starts from a photo -- turning one into a 3D model, or
        editing it -- needs the file inside ComfyUI's input directory, which
        may be on another machine as far as this API is concerned. So it goes
        over ``POST /upload/image`` (multipart, field name ``image``) exactly
        as the editor's own uploader does.

        Overwrite is deliberately not set. Without it the engine compares
        hashes, reuses the name when the file is identical, and otherwise picks
        an unused one -- so uploading the same photo twice does not pile up
        copies, and a different photo never quietly replaces someone else's.
        The name it chose is what LoadImage wants, prefixed by the subfolder
        when there is one.
        """
        import mimetypes

        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            reply = httpx.post(
                self._url("/upload/image"), timeout=HTTP_TIMEOUT, trust_env=False,
                files={"image": (path.name, data, mime)},
                data={"type": "input", "subfolder": subfolder})
            reply.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyError(f"Could not send {path.name} to ComfyUI.",
                             reason_key="upload_failed", detail=str(exc)) from exc

        body = reply.json()
        name = body.get("name")
        if not name:
            raise ComfyError("ComfyUI accepted the picture but did not name it.",
                             reason_key="upload_unnamed")
        folder = body.get("subfolder") or ""
        return f"{folder}/{name}" if folder else name

    def download(self, item: Output, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        reply = httpx.get(self.view_url(item), timeout=HTTP_TIMEOUT, trust_env=False)
        reply.raise_for_status()
        dest.write_bytes(reply.content)
        return dest

    # -- running ------------------------------------------------------------

    def submit(self, prompt: dict) -> str:
        """Queue a graph. Returns the prompt id.

        A rejected graph comes back as a 400 whose body names the node and the
        reason. That is far more useful than "it did not work", so it is
        unpacked rather than swallowed.
        """
        try:
            reply = httpx.post(self._url("/prompt"), timeout=HTTP_TIMEOUT, trust_env=False,
                               json={"prompt": prompt, "client_id": self.client_id})
        except httpx.HTTPError as exc:
            raise ComfyError("Could not reach ComfyUI.", reason_key="unreachable",
                             detail=str(exc)) from exc

        if reply.status_code >= 400:
            raise _explain_rejection(reply)

        prompt_id = reply.json().get("prompt_id")
        if not prompt_id:
            raise ComfyError("ComfyUI accepted the job but did not say which one.",
                             reason_key="no_prompt_id")
        return prompt_id

    def interrupt(self) -> None:
        # Best effort: the caller is leaving either way, and a stop that cannot
        # be delivered must not become an error on the way out.
        with httpx.Client(timeout=10.0, trust_env=False) as client, \
                contextlib.suppress(httpx.HTTPError):
            client.post(self._url("/interrupt"))

    def run(
        self,
        prompt: dict,
        *,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
        titles: dict[str, str] | None = None,
    ) -> list[Output]:
        """Queue a graph and wait for it, reporting progress as it goes.

        The websocket is opened **before** the job is submitted. Opening it
        afterwards loses whatever the engine reported in between, which on a
        cached or very fast run can be the entire job -- leaving the caller
        waiting for messages that have already been and gone.
        """
        titles = titles or {}
        import time

        with ws_connect(self._ws_url(), open_timeout=30) as socket:
            prompt_id = self.submit(prompt)
            quiet_since = time.monotonic()
            while True:
                if should_cancel and should_cancel():
                    # Tell the engine, or it carries on rendering something
                    # nobody is waiting for and the next job queues behind it.
                    self.interrupt()
                    raise ComfyError("Stopped.", reason_key="cancelled")

                try:
                    raw = socket.recv(timeout=WS_POLL)
                except TimeoutError:
                    if time.monotonic() - quiet_since > WS_SILENCE_BUDGET:
                        self.interrupt()
                        raise ComfyError(
                            "ComfyUI stopped reporting progress.",
                            reason_key="ws_timeout") from None
                    continue
                except ConnectionClosed as exc:
                    raise ComfyError(
                        "ComfyUI went away while working. If it crashed, its log says why.",
                        reason_key="engine_gone", detail=str(exc)) from exc
                quiet_since = time.monotonic()

                if isinstance(raw, bytes):
                    continue          # a live preview image; nothing to do with it

                message = _decode(raw)
                if message is None:
                    continue
                kind, data = message
                if data.get("prompt_id") not in (None, prompt_id):
                    continue          # another client's job on a shared engine

                if kind == "progress_state" and on_progress:
                    on_progress(_progress_from(data, titles))
                elif kind == "execution_error":
                    raise ComfyError(_explain_execution_error(data),
                                     reason_key="execution_error",
                                     detail=json.dumps(data, indent=2))
                elif kind == "execution_interrupted":
                    raise ComfyError("Stopped.", reason_key="cancelled")
                elif kind == "execution_success":
                    break
                elif kind == "executing" and data.get("node") is None:
                    # The older end-of-run signal. Kept because a run whose
                    # outputs were all cached can finish on this alone.
                    break

        return outputs_from_history(self._history_when_ready(prompt_id), prompt)

    def _history_when_ready(self, prompt_id: str) -> dict:
        """The history entry, once the engine has actually written it."""
        import time

        entry: dict = {}
        for _ in range(HISTORY_ATTEMPTS):
            entry = self.history(prompt_id)
            if entry.get("outputs"):
                return entry
            time.sleep(HISTORY_WAIT)
        return entry

    def _ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1].rstrip("/")
        return f"{scheme}://{host}/ws?clientId={self.client_id}"


def _decode(raw: str) -> tuple[str, dict] | None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(message, dict):
        return None
    return message.get("type", ""), (message.get("data") or {})


def _progress_from(data: dict, titles: dict[str, str]) -> Progress:
    """Turn per-node progress into one number and a name.

    ComfyUI reports every running node separately. A beginner does not want six
    bars, so the node that is actually working is the one shown.
    """
    nodes = (data.get("nodes") or {}).values()
    running = [n for n in nodes if n.get("state") == "running"] or list(nodes)
    if not running:
        return Progress()
    current = max(running, key=lambda n: n.get("max") or 0)
    node_id = str(current.get("display_node_id") or current.get("node_id") or "")
    return Progress(
        value=int(current.get("value") or 0),
        maximum=int(current.get("max") or 0),
        node_title=titles.get(node_id, ""),
        stage=str(current.get("state") or ""),
    )


def outputs_from_history(entry: dict, graph: dict | None = None) -> list[Output]:
    """Every file a finished run produced, in node order.

    ``graph`` is the converted prompt that was sent. It is what tells a saved
    3D model apart from a preview of one, because the history entry does not:
    both report ``{"result": [name, ...]}`` and only the node class says which
    directory the name is in. Without it, previews are still assumed to be
    saved files -- the old behaviour, and wrong, but not worse than it was.
    """
    found: list[Output] = []
    for node_id, node_output in (entry.get("outputs") or {}).items():
        animated = bool((node_output.get("animated") or [False])[0]) \
            if isinstance(node_output.get("animated"), (list, tuple)) else False
        for key in OUTPUT_KEYS:
            for item in node_output.get(key) or []:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                kind = key
                if key == "gifs" or (key == "images" and animated):
                    kind = "video"
                found.append(Output(
                    filename=item["filename"],
                    subfolder=item.get("subfolder", ""),
                    type=item.get("type", "output"),
                    kind=kind,
                ))
        # 3D: "result": [model_file, camera_info, ...]; the file is a string.
        result = node_output.get("result")
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], str):
            name = result[0]
            subfolder, _, filename = name.rpartition("/")
            node_class = ((graph or {}).get(node_id) or {}).get("class_type", "")
            preview = node_class in PREVIEW_3D_CLASSES
            found.append(Output(filename=filename or name, subfolder=subfolder,
                                type="temp" if preview else "output", kind="3d"))
    return found


def _explain_rejection(reply: httpx.Response) -> ComfyError:
    """Turn a 400 from /prompt into something worth reading.

    ComfyUI answers a bad graph with ``error`` and a per-node ``node_errors``
    naming the class and what was wrong with it. The commonest case by far is a
    model file the workflow references that is not on disk, which surfaces as
    value_not_in_list -- and "sd_xl_base_1.0.safetensors not in []" means
    nothing to the person who just wanted a picture.
    """
    try:
        body = reply.json()
    except ValueError:
        return ComfyError("ComfyUI refused the job.", reason_key="rejected",
                          detail=reply.text[:2000])

    error = body.get("error") or {}
    node_errors = body.get("node_errors") or {}

    parts = []
    for node_id, problem in node_errors.items():
        name = problem.get("class_type", node_id)
        for detail in problem.get("errors") or []:
            parts.append(
                f"{name}: {detail.get('message', '')} {detail.get('details', '')}".strip())

    headline = error.get("message") or "ComfyUI refused the job."
    reason = "rejected"
    if "value_not_in_list" in json.dumps(node_errors):
        headline = ("A model this workflow needs is not on your computer. "
                    "Setting its pack up again will fetch it.")
        reason = "model_missing"

    return ComfyError(headline, reason_key=reason,
                      detail="\n".join(parts) or json.dumps(body, indent=2))


def _explain_execution_error(data: dict) -> str:
    node = data.get("node_type") or data.get("node_id") or "a step"
    message = (data.get("exception_message") or "").strip()
    if "out of memory" in message.lower():
        # "Try a smaller size" is good advice for a picture and useless for a
        # 3D model, where nothing on screen is a size and the run has already
        # cost three minutes. Say which step ran out, and give a step that
        # applies: a workflow chaining six models keeps them all on the card,
        # and that is the memory the next one cannot have.
        return (f"Your graphics card ran out of memory at {node}. "
                "If this is a picture or a video, ask for a smaller one. "
                "Otherwise the workflow is holding several models on the card "
                "at once: add --disable-smart-memory to Extra ComfyUI options, "
                "which makes it put each one back when it is done. That is "
                "slower and it fits in far less memory.")
    return f"{node} failed: {message}" if message else f"{node} failed."
