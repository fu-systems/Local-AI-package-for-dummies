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

import contextlib
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from toolshed.exec.proc import child_environment, popen_kwargs, terminate_tree
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
class Safeguard:
    """One ComfyUI default we turn off because it crashes this kind of machine.

    ``concern`` is the substring that means the user has taken this decision
    over in Extra ComfyUI options. Typing ``--async-offload`` there puts the
    default back, and we then add nothing about it -- rather than trying to
    out-argue argparse over which of a pair of opposing flags wins.
    """

    flag: str
    concern: str
    plain_english: str


# ComfyUI turns on async weight offloading and pinned host memory by default.
# Its own help text says async offload is "Enabled by default on Nvidia", but
# a gfx1100 on ROCm 7.2 logs "Using async weight offloading with 2 streams" and
# "Enabled pinned memory 14909" -- so it is on for AMD too, and the help is
# describing an intention rather than the behaviour.
#
# Both move weights between the card and main memory by direct memory access,
# and that is where this configuration dies. Twice, at the moment a large model
# is swapped out to make room for a VAE:
#
#     Requested to load WanVAE
#     Memory access fault by GPU node-1 ... on address 0x7f1396928000.
#     Reason: Page not present or supervisor privilege.
#
# 0x7f… is a host address. A graphics card faulting on host memory means a
# transfer was set up against pages that were not mapped for it. Both crashes
# came after minutes of successful sampling, at the first model swap, which is
# exactly when these two paths are used and never used before.
#
# Turning them off slows model swapping and nothing else -- sampling is
# untouched. The trade is a few seconds per model change against losing seven
# minutes of finished work to an abort.
# The two above were not enough. A machine that hit this at every attempt to
# load the Wan VAE -- not twice in a long session, every single time -- sent us
# back to model_management.py at v0.34.0, where the reason is visible:
#
# Dynamic VRAM is its own transfer path, and neither of the first two flags
# switches it off. `enables_dynamic_vram()` in cli_args.py is true unless
# --disable-dynamic-vram, --highvram, --gpu-only, --novram or --cpu is given,
# so it survives both of ours. It then does this for every dynamic model, at
# the point one is swapped:
#
#     pin_state[subset] = (comfy_aimdo.host_buffer.HostBuffer(
#         0, 8 * 1024 * 1024, pinned_hostbuf_size(model.model_size())), ...)
#
# --disable-pinned-memory only drives pinned_hostbuf_size() to zero. The aimdo
# host buffer is still constructed and the path is still live, which is why
# turning off async offload and pinned memory made the fault rarer without
# making it stop.
#
# So the third flag, which is the one that actually takes that path out:
# "Disable dynamic VRAM and use estimate based model loading."
#
# The trade is real and worth stating. Estimate-based loading is the older,
# blunter scheme, and it can misjudge a tight card where the dynamic one would
# have coped -- so this may cost an out-of-memory on a job that used to fit.
# An out-of-memory reports itself and leaves the machine usable. A page fault
# aborts the process and takes the finished work with it. Given a card that
# cannot load a VAE without dying, that trade is not close.
AMD_SAFEGUARDS = (
    Safeguard("--disable-async-offload", "async-offload",
              "moving model weights in the background while the card works"),
    Safeguard("--disable-pinned-memory", "pinned-memory",
              "reserving main memory the card can read from directly"),
    Safeguard("--disable-dynamic-vram", "dynamic-vram",
              "streaming model weights between the card and main memory as it goes"),
)


# The tag whose cli_args.py these flags were read from. An engine that renames
# one is handled safely -- engine_understands drops it rather than producing a
# command line ComfyUI refuses -- but *safely* is not the same as *silently*,
# and a dropped safeguard means the crash comes back. So: bumping ENGINE_TAG
# fails a test until someone re-reads the flags and moves this with it.
FLAGS_VERIFIED_AGAINST = "v0.34.0"


@dataclass(frozen=True)
class Safeguards:
    """What we could and could not do about a machine's known failure.

    ``unavailable`` is the field that matters. A safeguard we meant to apply
    and could not is the exact shape of the original bug returning unannounced,
    so it is carried out of here to be said out loud rather than dropped.
    """

    applied: tuple[Safeguard, ...] = ()
    unavailable: tuple[Safeguard, ...] = ()
    overridden: tuple[Safeguard, ...] = ()

    @property
    def flags(self) -> list[str]:
        return [s.flag for s in self.applied]


# Verbatim from two crash logs on a gfx1100, seven minutes into a job each
# time. Only strings actually seen in a log belong here: a guessed one would
# either never match or, worse, explain the wrong thing confidently.
CRASH_SIGNS = (
    ("Memory access fault by GPU node",
     "ComfyUI hit a graphics memory fault and was stopped by the driver. "
     "That is not your workflow and not something you did, but the work in "
     "progress is lost. Toolshed already starts AMD cards with all three "
     "transfer paths that cause it switched off. If this still happens, the "
     "VAE is the usual place: add --cpu-vae to Extra ComfyUI options to run "
     "that step on the processor instead. It is slower and it always works."),
)


def explain_crash(log_tail: str) -> str | None:
    """A specific explanation when the log carries a known fault, else None.

    An exit code says the process aborted; the log says why. Reading it here is
    the difference between a user being told what happened and a user copying
    two hundred lines of module names to someone who can read them.
    """
    for sign, explanation in CRASH_SIGNS:
        if sign in log_tail:
            return explanation
    return None


def engine_understands(engine_dir: Path, flag: str) -> bool:
    """Does the installed ComfyUI accept this option?

    Read from its own ``cli_args.py`` as text. Passing an option ComfyUI does
    not know makes argparse print usage and exit before the server starts, so
    an unrecognised flag would turn "crashes at the end of a long job" into
    "never starts at all" -- a worse failure, and one we cannot rehearse here
    because we do not have the engine or the card.

    Read, not imported: importing ComfyUI is the line this project does not
    cross. Unreadable means no, so an engine laid out differently by a future
    version gets the stock defaults rather than a broken command line.
    """
    try:
        source = (engine_dir / "comfy" / "cli_args.py").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return False
    return f'"{flag}"' in source or f"'{flag}'" in source


def choose_safeguards(engine_dir: Path, *, rocm: bool,
                      extra: Sequence[str] = ()) -> Safeguards:
    """Which safeguards this machine needs, and which of them we can apply."""
    if not rocm:
        return Safeguards()
    typed = " ".join(extra)
    applied, unavailable, overridden = [], [], []
    for guard in AMD_SAFEGUARDS:
        if guard.concern in typed:
            overridden.append(guard)          # the user has taken this over
        elif engine_understands(engine_dir, guard.flag):
            applied.append(guard)
        else:
            unavailable.append(guard)         # say so; never drop it quietly
    return Safeguards(tuple(applied), tuple(unavailable), tuple(overridden))


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
    safe_args: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES_KEPT))
    _reader: threading.Thread | None = None
    _stopping: bool = False
    _ready: bool = False

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
            # Defaults switched off because they crash this machine. Before the
            # user's own options, not after, so anything they type still has
            # the last word.
            *self.safe_args,
            # Whatever the user added. Last, so it can override anything above
            # -- argparse takes the later value for a repeated option, which is
            # what makes this an escape hatch rather than a suggestion box.
            *self.extra_args,
        ]

    # -- lifecycle ----------------------------------------------------------

    def start(self, *, on_line: LogFn | None = None,
              on_died: Callable[[str], None] | None = None) -> None:
        """Launch the engine.

        ``on_died`` is called if it exits without being asked to. That is not a
        rare case to be tidy about: a GPU driver fault takes the whole process
        down mid-generation with SIGABRT, and without this the app goes on
        saying "ComfyUI is running" over a program that is gone, while whatever
        was waiting on it waits for a reply that will never come.
        """
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

        # ComfyUI needs these to exist; folder_paths.py only creates input/.
        for sub in ("custom_nodes", "input", "temp", "user"):
            (self.layout.comfy_base / sub).mkdir(parents=True, exist_ok=True)

        # An engine left over from a Toolshed that crashed would still hold the
        # port, so the next launch either fails readiness or starts a second
        # copy beside it. The pid file names ours; anything else on the port is
        # not ours to touch.
        self._stop_orphan()

        # An explicit --port in the user's extra options wins, and readiness
        # must poll that port rather than the one we would have chosen.
        wanted = _port_from_args(self.extra_args)
        if wanted:
            self.port = wanted
        if not self.port:
            self.port = choose_port()

        # Inherit the user's environment, minus the frozen app's own loader
        # paths, with ours laid over it. Ours carries HSA_OVERRIDE_GFX_VERSION
        # for the AMD cards that need it; dropping it does not fail loudly, it
        # just means the engine cannot use the GPU.
        environment = child_environment(self.env)
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
        self._stopping = False
        self._ready = False
        self._write_pidfile()
        self._reader = threading.Thread(target=self._drain, args=(on_line, on_died),
                                        daemon=True)
        self._reader.start()

    def _drain(self, on_line: LogFn | None,
               on_died: Callable[[str], None] | None = None) -> None:
        """Copy the engine's output to a file and a ring buffer.

        On its own thread: reading the pipe from the caller would block the
        readiness poll, and letting the pipe fill would deadlock the engine
        once it had written 64 KB nobody was collecting.
        """
        assert self.process and self.process.stdout
        # The pipe must be drained whatever happens to the log file. A reader
        # that gave up because the file could not be opened would leave the
        # engine blocked on a full pipe after 64 KB of output -- frozen, with
        # nothing to say why.
        try:
            fh = self.layout.log_file.open("a", encoding="utf-8")
        except OSError:
            fh = None
        try:
            if fh:
                fh.write(f"\n--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            for line in self.process.stdout:
                line = line.rstrip("\n")
                self._log.append(line)
                if fh:
                    try:
                        fh.write(line + "\n")
                        fh.flush()
                    except OSError:
                        fh = None
                if on_line:
                    on_line(line)
        except (OSError, ValueError):
            # The pipe closing under us is how a stopped engine ends. Not news.
            pass
        finally:
            if fh:
                fh.close()

        # The loop above ends when the engine's output does, which means it has
        # exited -- or is about to. EOF arrives before the kernel has reaped
        # the child, so poll() can still say "running" here; wait briefly so
        # the exit code is real rather than None.
        if on_died and not self._stopping and self.process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
            code = self.process.poll()
            # Only once it was up: a death during startup is reported by
            # wait_until_ready, and reporting it here too showed the user two
            # conflicting messages for one event.
            if code is not None and self._ready:
                on_died(self._explain_exit(code))

    def _explain_exit(self, code: int) -> str:
        """Say what happened, in words, for the codes that mean something.

        A GPU page fault -- "Memory access fault by GPU node-1" on ROCm -- kills
        the process with SIGABRT, which arrives here as -6. It is not something
        the user did, and it is not out of memory, so it should not be reported
        as either.

        The log is consulted before the exit code, because -6 covers every
        abort and the log says which one this was.
        """
        if code != 0 and (known := explain_crash(self.tail(200))):
            return known
        if code == 0:
            return "ComfyUI closed on its own."
        if sys.platform == "win32":
            # Windows has no signals; a crash is an NTSTATUS in the exit code.
            unsigned = code & 0xFFFFFFFF
            if unsigned == 0xC0000005:
                return "ComfyUI crashed (access violation). The graphics driver is the usual cause."
            if unsigned >= 0xC0000000:
                return f"ComfyUI crashed (Windows error 0x{unsigned:08X})."
            return f"ComfyUI stopped unexpectedly (exit code {code})."
        if code == -6:
            return ("ComfyUI was stopped by the graphics driver. This is usually a "
                    "driver-level fault rather than anything you did.")
        if code == -9:
            return ("ComfyUI was killed, most likely by the system running out of "
                    "memory. Closing other programs may help.")
        if code < 0:
            return f"ComfyUI was stopped by signal {-code}."
        return f"ComfyUI stopped unexpectedly (exit code {code})."

    def _write_pidfile(self) -> None:
        try:
            pid_file(self.root).parent.mkdir(parents=True, exist_ok=True)
            pid_file(self.root).write_text(str(self.process.pid), encoding="utf-8")
        except OSError:
            pass

    def _stop_orphan(self) -> None:
        """Stop a ComfyUI a previous Toolshed left behind, and only that.

        Matched by the pid file *and* the process's own command line naming
        our engine directory, so a pid reused by something else since is left
        alone. On platforms without /proc the command line cannot be read, so
        nothing is killed rather than something wrong.
        """
        import os

        path = pid_file(self.root)
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return
        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            argv = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            path.unlink(missing_ok=True)
            return
        if str(self.layout.engine_dir) in argv:
            with contextlib.suppress(OSError):
                os.kill(pid, 15)
            deadline = time.monotonic() + STOP_TIMEOUT
            while time.monotonic() < deadline and cmdline.exists():
                time.sleep(0.1)
            if cmdline.exists():
                with contextlib.suppress(OSError):
                    os.kill(pid, 9)
        path.unlink(missing_ok=True)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def responds(self, timeout: float = 2.0) -> bool:
        """Does the server answer? /system_stats is a real handler in v0.34.0's
        server.py, so a 200 means the app is up, not merely the socket."""
        try:
            # trust_env=False: loopback must never go through a proxy, and
            # httpx would otherwise honour HTTP(S)_PROXY for 127.0.0.1 too.
            reply = httpx.get(f"{self.url}/system_stats", timeout=timeout, trust_env=False)
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
                self._ready = True
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
        self._stopping = True
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


def _port_from_args(args: list[str]) -> int:
    """A --port the user put in the extra options, or 0."""
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args) and args[i + 1].isdigit():
            return int(args[i + 1])
        if arg.startswith("--port=") and arg[7:].isdigit():
            return int(arg[7:])
    return 0


def pid_file(root: Path) -> Path:
    return root / "state" / "engine.pid"


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
        # posix=False on Windows, or every backslash in a path is eaten.
        return shlex.split(path.read_text(encoding="utf-8"), comments=True,
                           posix=(sys.platform != "win32"))
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
