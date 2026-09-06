"""Starting ComfyUI, and knowing whether it actually came up.

The install used to finish by telling the user to "open ComfyUI" and giving
them no way to do it. That is the whole gap this closes: a supervised child
process, a real readiness check, and a URL to open.

**The GPL boundary is the reason this file looks the way it does.** ComfyUI is
GPL-3.0-or-later. Toolshed downloads it and drives it as a separate process
over HTTP, and must never import it, never bundle it, and never place our code
under ``custom_nodes/``. So there is no ``import comfy`` here and never will
be: everything below is argv, a socket and a subprocess.

Every command-line flag used here was read from ComfyUI v0.34.0's own
``comfy/cli_args.py`` and ``folder_paths.py``, not remembered. See
docs/UPSTREAM.md for the table and where each was verified.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from toolshed.exec.proc import popen_kwargs, terminate_tree
from toolshed.exec.uvtool import venv_python

# ComfyUI's own default. Tried first so that a user who already knows the
# address gets the address they expect; if something else holds it we move.
DEFAULT_PORT = 8188
# Loopback only. This is somebody's personal machine, and ComfyUI has no
# authentication whatsoever -- binding it to the network would publish an
# unauthenticated remote code execution endpoint on their LAN.
HOST = "127.0.0.1"

# Cold start loads torch and scans models; on a slow disk that is not quick.
READY_TIMEOUT = 300.0
STOP_TIMEOUT = 20.0
LOG_LINES_KEPT = 400

LogFn = Callable[[str], None]


class EngineError(RuntimeError):
    """The engine could not be started, or died on the way up."""

    def __init__(self, message: str, *, reason_key: str = "engine_failed",
                 detail: str = "") -> None:
        super().__init__(message)
        self.reason_key = reason_key
        self.detail = detail


def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def choose_port(preferred: int = DEFAULT_PORT) -> int:
    """The preferred port if nothing holds it, otherwise one the OS picks.

    Binding to port 0 and reading the assignment back leaves a brief window in
    which something else could take it. That is acceptable here and the
    alternative is not: refusing to start because 8188 is busy would mean a
    second ComfyUI, or anything else on that port, blocks Toolshed entirely.
    """
    if port_is_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


@dataclass(frozen=True)
class Layout:
    """Where everything lives, given the data root.

    ``--base-directory`` sets ComfyUI's base for models, custom_nodes, input,
    output, temp and user; ``--models-directory`` then overrides the models
    folder alone. Both were read from cli_args.py.

    That pairing is load-bearing. The installer downloads models to
    ``<root>/models`` while ComfyUI's own state belongs under ``<root>/comfy``,
    and with only ``--base-directory`` the engine would look for models in
    ``<root>/comfy/models`` and find an empty shelf -- every workflow reporting
    missing models on a machine where all of them are present.
    """

    root: Path

    @property
    def engine_dir(self) -> Path:
        return self.root / "engine" / "comfyui"

    @property
    def main_py(self) -> Path:
        return self.engine_dir / "main.py"

    @property
    def python(self) -> Path:
        return venv_python(self.root / "runtime")

    @property
    def comfy_base(self) -> Path:
        return self.root / "comfy"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def log_file(self) -> Path:
        return self.root / "state" / "logs" / "comfyui.log"

    def missing_pieces(self) -> list[str]:
        """What is not installed yet, in the user's words rather than paths."""
        missing = []
        if not self.python.is_file():
            missing.append("the private Python workspace")
        if not self.main_py.is_file():
            missing.append("the ComfyUI engine")
        if not self.models_dir.is_dir():
            missing.append("the models folder")
        return missing


@dataclass
class Engine:
    """A supervised ComfyUI process.

    One instance per launch. Starting is not the same as being ready: the
    process is up in milliseconds and serving in tens of seconds, and reporting
    the first as though it were the second is how a launcher ends up opening a
    browser on a connection-refused page.
    """

    root: Path
    env: dict[str, str] = field(default_factory=dict)
    port: int = 0
    extra_args: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES_KEPT))
    _reader: threading.Thread | None = None

    @property
    def layout(self) -> Layout:
        return Layout(self.root)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def command(self) -> list[str]:
        """The exact argv. Pure, so it can be asserted without launching.

        ``--disable-auto-launch`` because we open the browser ourselves, once
        the server actually answers. ``--log-stdout`` because ComfyUI logs to
        stderr by default and one ordered stream is far easier to show a user
        than two interleaved ones.
        """
        layout = self.layout
        return [
            str(layout.python),
            str(layout.main_py),
            "--base-directory", str(layout.comfy_base),
            "--models-directory", str(layout.models_dir),
            # Overrides the output folder inside --base-directory, so finished
            # pictures land somewhere a beginner can find rather than four
            # levels down beside the engine's own state.
            "--output-directory", str(layout.output_dir),
            "--listen", HOST,
            "--port", str(self.port),
            "--disable-auto-launch",
            "--log-stdout",
            # Whatever the user added. Last, so it can override anything above
            # -- argparse takes the later value for a repeated option, which is
            # what makes this an escape hatch rather than a suggestion box.
            *self.extra_args,
        ]

    # -- lifecycle ----------------------------------------------------------

    def start(self, *, on_line: LogFn | None = None) -> None:
        missing = self.layout.missing_pieces()
        if missing:
            raise EngineError(
                "ComfyUI is not installed yet: missing " + ", ".join(missing) + ".",
                reason_key="not_installed")

        # is_valid_directory in cli_args.py rejects a --models-directory that
        # does not exist, so the engine would refuse to start rather than start
        # empty. Create it here too; the install makes it, but a user who
        # deleted it should get a working engine, not an argparse error.
        self.layout.models_dir.mkdir(parents=True, exist_ok=True)
        self.layout.comfy_base.mkdir(parents=True, exist_ok=True)
        self.layout.output_dir.mkdir(parents=True, exist_ok=True)
        self.layout.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.port:
            self.port = choose_port()

        import os

        # Inherit the user's environment and lay ours over it. This carries
        # HSA_OVERRIDE_GFX_VERSION for the AMD cards that need it; dropping it
        # does not fail loudly, it just means the engine cannot use the GPU.
        environment = {**os.environ, **self.env}
        # Unbuffered, or the log stays empty for a minute and the user watches
        # a blank box while the engine is in fact starting normally.
        environment["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(  # noqa: S603 -- argv is ours, no shell
            self.command(),
            cwd=str(self.layout.engine_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs(new_group=True),
        )
        self._reader = threading.Thread(target=self._drain, args=(on_line,), daemon=True)
        self._reader.start()

    def _drain(self, on_line: LogFn | None) -> None:
        """Copy the engine's output to a file and a ring buffer.

        On its own thread: reading the pipe from the caller would block the
        readiness poll, and letting the pipe fill would deadlock the engine
        once it had written 64 KB nobody was collecting.
        """
        assert self.process and self.process.stdout
        try:
            with self.layout.log_file.open("a", encoding="utf-8") as fh:
                fh.write(f"\n--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                for line in self.process.stdout:
                    line = line.rstrip("\n")
                    self._log.append(line)
                    fh.write(line + "\n")
                    fh.flush()
                    if on_line:
                        on_line(line)
        except (OSError, ValueError):
            # The pipe closing under us is how a stopped engine ends. Not news.
            pass

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def responds(self, timeout: float = 2.0) -> bool:
        """Does the server answer? /system_stats is a real handler in v0.34.0's
        server.py, so a 200 means the app is up, not merely the socket."""
        try:
            reply = httpx.get(f"{self.url}/system_stats", timeout=timeout)
        except httpx.HTTPError:
            return False
        return reply.status_code == 200

    def wait_until_ready(
        self,
        timeout: float = READY_TIMEOUT,
        *,
        should_cancel: Callable[[], bool] | None = None,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        """Block until the server answers, or explain why it never will.

        Watches the process as well as the socket. Without that, an engine that
        dies two seconds in -- a missing dependency, a CUDA error -- would be
        waited on for the full five minutes before reporting a timeout, and the
        real reason would be sitting unread in the log the whole time.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if should_cancel and should_cancel():
                self.stop()
                raise EngineError("Stopped.", reason_key="cancelled")
            if self.responds():
                return
            if not self.is_running():
                raise EngineError(
                    "ComfyUI stopped while it was starting up.",
                    reason_key="engine_died", detail=self.tail())
            if on_wait:
                on_wait(max(0.0, deadline - time.monotonic()))
            time.sleep(0.5)

        self.stop()
        raise EngineError(
            f"ComfyUI did not finish starting within {int(timeout)} seconds.",
            reason_key="engine_timeout", detail=self.tail())

    def stop(self) -> None:
        """Stop the engine and everything it started. Safe to call twice."""
        if self.process is None:
            return
        if self.process.poll() is None:
            terminate_tree(self.process)
            try:
                self.process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=2.0)
        self.process = None

    def tail(self, lines: int = 25) -> str:
        """The last of the engine's output, for showing when it goes wrong."""
        return "\n".join(list(self._log)[-lines:])


def flags_file(root: Path) -> Path:
    return root / "state" / "engine-flags.txt"


def read_extra_flags(root: Path) -> list[str]:
    """Extra engine options the user has set, if any.

    ComfyUI has real levers for the failures we cannot fix from out here --
    --fp32-vae and --cpu-vae for a VAE the card will not run, --cuda-device to
    pick between two graphics cards, --reserve-vram to leave the desktop some
    room. Without somewhere to put them, someone hitting one of those has no
    move at all except to stop using the app.

    Parsed with shlex so quoting behaves, and passed as argv to a process we
    spawn without a shell, so there is nothing here to inject into.
    """
    import shlex

    path = flags_file(root)
    if not path.is_file():
        return []
    try:
        return shlex.split(path.read_text(encoding="utf-8"), comments=True)
    except (OSError, ValueError):
        return []


def write_extra_flags(root: Path, text: str) -> None:
    path = flags_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def open_in_browser(url: str) -> bool:
    """Open the engine in the user's browser. False if we could not.

    Never fatal. On a machine with no default browser -- a bare window manager,
    a locked-down desktop -- the address itself is still useful, so the caller
    shows it rather than reporting a failure.
    """
    import webbrowser

    try:
        return webbrowser.open(url)
    except Exception:      # noqa: BLE001 -- any backend failure is the same to us
        return False


def is_wayland_or_x11() -> bool:
    """Whether a browser could plausibly be opened at all."""
    import os

    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
