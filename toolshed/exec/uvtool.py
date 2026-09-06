"""Providing Python, without touching whatever Python the user already has.

uv is a single static binary that needs no system Python and no admin rights.
It installs a relocatable interpreter of our choosing and creates the virtual
environment the engine runs in, entirely inside the data root, so uninstalling
is deleting a folder.

Everything here is invoked by absolute path. A GUI launched from a file manager
inherits a stale environment, and resolving tools from PATH is how installers
end up reporting that a program is missing when it is plainly installed.
"""

from __future__ import annotations

import platform
import sys
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from toolshed.exec.download import download_file
from toolshed.exec.proc import Result, run

UV_VERSION = "0.12.10"
UV_BASE = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}"

LogFn = Callable[[str], None]


def uv_asset() -> tuple[str, str]:
    """(asset filename, path of the binary inside it) for this machine."""
    machine = platform.machine().lower()
    if sys.platform == "win32":
        arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
        return f"uv-{arch}-pc-windows-msvc.zip", "uv.exe"
    if sys.platform == "darwin":  # not supported, but do not lie about the name
        arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
        return f"uv-{arch}-apple-darwin.tar.gz", "uv"
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    return f"uv-{arch}-unknown-linux-gnu.tar.gz", "uv"


def uv_path(runtime_dir: Path) -> Path:
    return runtime_dir / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")


def ensure_uv(runtime_dir: Path, *, log: LogFn | None = None) -> Path:
    """Download and unpack uv if it is not already there."""
    target = uv_path(runtime_dir)
    if target.is_file():
        return target

    asset, inner = uv_asset()
    archive = runtime_dir / "uv" / asset
    if log:
        log(f"Fetching uv {UV_VERSION}")
    download_file(f"{UV_BASE}/{asset}", archive)

    extract_to = archive.parent
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_to)
    else:
        with tarfile.open(archive) as tf:
            # filter="data" refuses absolute paths and traversal in the archive.
            tf.extractall(extract_to, filter="data")

    found = next((p for p in extract_to.rglob(inner) if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"{inner} not found inside {asset}")
    if found != target:
        found.replace(target)
    target.chmod(0o755)
    archive.unlink(missing_ok=True)
    return target


def uv_env(runtime_dir: Path) -> dict[str, str]:
    """Keep every byte uv writes inside the data root.

    The cache must share a filesystem with the venv: otherwise uv silently
    falls back to copying instead of linking, which triples install time and
    disk use on a machine with a separate drive for models.
    """
    return {
        "UV_CACHE_DIR": str(runtime_dir / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(runtime_dir / "python"),
        "UV_HTTP_TIMEOUT": "600",
        "UV_NO_PROGRESS": "1",
    }


def install_python(
    uv: Path, runtime_dir: Path, version: str, *, log: LogFn | None = None
) -> Result:
    return run([uv, "python", "install", version],
               env=uv_env(runtime_dir), timeout=900, on_line=log)


def create_venv(
    uv: Path,
    runtime_dir: Path,
    version: str,
    *,
    clear: bool = False,
    log: LogFn | None = None,
) -> Result:
    """Create the virtual environment the engine runs in.

    ``clear`` replaces whatever is at the target path. uv refuses by default if
    anything is there, which is correct for a one-shot command and wrong for an
    installer: a run interrupted anywhere after this step would otherwise be
    unable to start again. The caller decides, having first checked whether the
    existing environment is usable.

    ``--force`` is deliberately never passed. It lets ``--clear`` delete a
    directory that is not a virtual environment at all, and a bug that reaches
    it would delete a folder of the user's making.
    """
    cmd: list[str | Path] = [uv, "venv", "--python", version]
    if clear:
        cmd.append("--clear")
    cmd.append(str(runtime_dir / "venv"))
    return run(cmd, env=uv_env(runtime_dir), timeout=600, on_line=log)


def venv_python(runtime_dir: Path) -> Path:
    venv = runtime_dir / "venv"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def venv_python_version(runtime_dir: Path) -> str | None:
    """``"3.12"`` if a working virtual environment is already there, else None.

    Asks the interpreter rather than reading pyvenv.cfg, because the question
    that matters is whether it *runs*. A venv whose interpreter was deleted, or
    which points at a Python that has since been removed, has a perfectly
    well-formed config file and cannot execute anything.

    Never raises: every answer other than a working interpreter is None.
    """
    python = venv_python(runtime_dir)
    if not python.is_file():
        return None
    try:
        result = run([python, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                     timeout=60)
    except OSError:
        return None
    if not result.ok:
        return None
    line = next((ln.strip() for ln in reversed(result.stdout.splitlines()) if ln.strip()), "")
    return line or None


def pip_install(
    uv: Path,
    runtime_dir: Path,
    packages: list[str],
    *,
    index_url: str | None = None,
    log: LogFn | None = None,
    timeout: float = 1800,
) -> Result:
    """Install into our venv.

    ``--index-url``, never ``--extra-index-url``: with an extra index the
    resolver may legitimately prefer the PyPI wheel, and torch's Windows PyPI
    wheel is the CPU build. That single flag is the most common cause of
    "Torch not compiled with CUDA enabled".
    """
    cmd: list[str | Path] = [uv, "pip", "install", "--python", venv_python(runtime_dir)]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages
    return run(cmd, env=uv_env(runtime_dir), timeout=timeout, on_line=log)


TORCH_PROBE = (
    "import json,torch;"
    "print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,"
    "'hip':getattr(torch.version,'hip',None),'available':torch.cuda.is_available(),"
    "'devices':torch.cuda.device_count()}))"
)


@dataclass(frozen=True)
class TorchCheck:
    """The outcome of looking at the PyTorch that was just installed."""

    ok: bool
    message: str
    warning: str = ""


def verify_torch(runtime_dir: Path, expect_tag: str, *, log: LogFn | None = None) -> TorchCheck:
    """Prove the graphics card is really usable before downloading 40 GB.

    Catching a CPU-only build here costs ninety seconds. Catching it after the
    models costs an hour and the user's patience. That -- and only that -- is
    what this step is for, so only that stops the install:

    * PyTorch will not import, or sees no device: **fail**. Nothing downstream
      can work, and 40 GB of models would be wasted.
    * PyTorch sees the card but carries a different build tag than we asked
      for: **warn and carry on**. It works. Refusing here would be rejecting a
      functioning machine over a string.

    The second case used to be fatal, and killed a healthy two-GPU ROCm install
    at 26 percent. A check that is stricter than the thing it protects against
    does not make the install safer, it just makes it fail.
    """
    result = run([venv_python(runtime_dir), "-c", TORCH_PROBE], timeout=300, on_line=log)
    if not result.ok:
        return TorchCheck(False, "PyTorch could not be loaded at all.")

    import json

    line = next((ln for ln in reversed(result.stdout.splitlines())
                 if ln.strip().startswith("{")), "")
    try:
        info = json.loads(line)
    except json.JSONDecodeError:
        return TorchCheck(False, "PyTorch did not report its configuration.")

    if not info.get("available") or not info.get("devices"):
        return TorchCheck(False, "PyTorch is installed but cannot see your graphics card.")

    version = info.get("version", "")
    devices = info.get("devices")
    plural = "" if devices == 1 else "s"
    good = f"{version} sees {devices} graphics card{plural}."

    if expect_tag and expect_tag not in version:
        return TorchCheck(True, good, warning=(
            f"This is the {version} build; we asked for {expect_tag}. "
            f"Your card works, so setup is carrying on."))
    return TorchCheck(True, good)
