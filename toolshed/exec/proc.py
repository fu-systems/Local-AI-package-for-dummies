"""Running child processes so that stopping actually stops them.

Two rules, both learned from other people's bug trackers:

* Never wait on a child without a deadline. A process that blocks on something
  invisible -- a prompt, a modal, a dead socket -- otherwise hangs the app for
  as long as the operating system will let it.
* Killing the parent is not killing the job. Installers and Python launchers
  spawn children; on Windows only a Job Object reliably reaps the tree, and on
  Linux only a process group does.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from toolshed.exec.download import Cancelled

IS_WINDOWS = sys.platform == "win32"

# Keep a console window from flashing up behind a GUI app on Windows.
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def popen_kwargs(*, new_group: bool = True) -> dict:
    """Platform flags that make a child killable as a group."""
    if IS_WINDOWS:
        flags = CREATE_NO_WINDOW
        if new_group:
            flags |= CREATE_NEW_PROCESS_GROUP
        return {"creationflags": flags}
    return {"start_new_session": True} if new_group else {}


def terminate_tree(process: subprocess.Popen) -> None:
    """Stop a child and everything it started. Never raises."""
    if process.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                           capture_output=True, check=False, timeout=30)
        else:
            os.killpg(os.getpgid(process.pid), 9)
    except (OSError, subprocess.SubprocessError):
        with contextlib.suppress(OSError):
            process.kill()


def run(
    cmd: Sequence[str | Path],
    *,
    timeout: float = 900,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Result:
    """Run a command to completion, streaming its output, with a hard deadline.

    ``on_line`` receives each line as it appears, so the user sees a log moving
    rather than a frozen window during a five-minute pip install.

    ``should_cancel`` is polled while the child runs. Stop used to work only
    between steps and inside downloads; pressed during a twenty-minute pip
    install it did nothing until the install finished, which is not what a
    button labelled Stop means. When it returns true the child is killed with
    its whole tree and ``Cancelled`` is raised, the same signal the downloads
    use.
    """
    argv = [str(c) for c in cmd]
    full_env = child_environment(env)
    # Unbuffered, or a streamed log arrives in one lump at the end.
    full_env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=full_env,
        cwd=str(cwd) if cwd else None,
        **popen_kwargs(),
    )
    lines: list[str] = []

    # The reader runs on its own thread. Draining stdout on this one and only
    # then calling wait(timeout=...) would mean the deadline never applies to a
    # process that simply goes quiet -- which is exactly the shape of hang that
    # matters: no output, no exit, no way out.
    def drain() -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if on_line:
                on_line(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout
    try:
        while True:
            if process.poll() is not None:
                break
            if should_cancel and should_cancel():
                terminate_tree(process)
                reader.join(timeout=5)
                raise Cancelled()
            if time.monotonic() >= deadline:
                terminate_tree(process)
                reader.join(timeout=5)
                return Result(
                    -1, "\n".join(lines),
                    f"{argv[0]} did not finish within {timeout:.0f} seconds.")
            time.sleep(0.05)
    except BaseException:
        terminate_tree(process)
        raise

    reader.join(timeout=10)
    return Result(process.returncode, "\n".join(lines), "")


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a child should inherit from a possibly-frozen app.

    PyInstaller's bootloader points LD_LIBRARY_PATH at the bundle's own
    ``_internal`` so the frozen app finds its Qt and libstdc++. Every child
    inherits that, and a child that is *another* Python -- the venv's, running
    PyTorch -- then loads our bundled libstdc++ and friends instead of its own.
    That is exactly the "works from source, breaks in the tarball" bug. The
    bootloader keeps the original in LD_LIBRARY_PATH_ORIG, so it is restored;
    when there was none, the variable is removed rather than left pointing at
    us. Same for the other loader variables the install wrapper clears.
    """
    env = dict(os.environ)
    frozen = getattr(sys, "frozen", False)
    if frozen:
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            original = env.pop(f"{var}_ORIG", None)
            if original:
                env[var] = original
            else:
                env.pop(var, None)
        for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QML2_IMPORT_PATH",
                    "PYTHONHOME", "PYTHONPATH"):
            env.pop(var, None)
    if extra:
        env.update(extra)
    return env
