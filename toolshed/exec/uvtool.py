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

import os
import platform
import re
import sys
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from toolshed.exec.download import CancelFn, download_file, remote_size
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


def ensure_uv(runtime_dir: Path, *, log: LogFn | None = None,
              should_cancel: CancelFn | None = None) -> Path:
    """Download and unpack uv if it is not already there."""
    target = uv_path(runtime_dir)
    if target.is_file():
        return target

    asset, inner = uv_asset()
    archive = runtime_dir / "uv" / asset
    if log:
        log(f"Fetching uv {UV_VERSION}")
    download_file(f"{UV_BASE}/{asset}", archive, should_cancel=should_cancel)

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
    uv: Path, runtime_dir: Path, version: str, *, log: LogFn | None = None,
    should_cancel: CancelFn | None = None,
) -> Result:
    # --no-bin: without it uv also drops a `python3.12` launcher into
    # ~/.local/bin, outside the data root. Nothing of ours belongs there --
    # it would shadow a Python the user installed themselves, and uninstall
    # would not know to remove it.
    return run([uv, "python", "install", "--no-bin", version],
               env=uv_env(runtime_dir), timeout=900, on_line=log,
               should_cancel=should_cancel)


def create_venv(
    uv: Path,
    runtime_dir: Path,
    version: str,
    *,
    clear: bool = False,
    log: LogFn | None = None,
    should_cancel: CancelFn | None = None,
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
    return run(cmd, env=uv_env(runtime_dir), timeout=600, on_line=log,
               should_cancel=should_cancel)


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
    should_cancel: CancelFn | None = None,
    heartbeat: Callable[[], bool] | None = None,
    find_links: Path | None = None,
) -> Result:
    """Install into our venv.

    With ``find_links`` the install is offline, from files already on disk:
    ``--no-index`` so nothing is fetched, ``--offline`` so nothing can be. That
    is how a download we did ourselves -- with a real progress bar -- is handed
    to uv to unpack. Without it uv fetches from ``index_url`` as usual.

    ``heartbeat`` is polled while uv runs, because uv is silent while it
    downloads: progress bars are suppressed (UV_NO_PROGRESS, and it would not
    draw them into a pipe anyway) and its summary lines come only at phase
    boundaries. The PyTorch ROCm build is gigabytes, so without something else
    to watch the screen sits on one line for many minutes and looks hung.

    ``--index-url``, never ``--extra-index-url``: with an extra index the
    resolver may legitimately prefer the PyPI wheel, and torch's Windows PyPI
    wheel is the CPU build. That single flag is the most common cause of
    "Torch not compiled with CUDA enabled".
    """
    cmd: list[str | Path] = [uv, "pip", "install", "--python", venv_python(runtime_dir)]
    if find_links is not None:
        cmd += ["--no-index", "--offline", "--find-links", str(find_links)]
    elif index_url:
        cmd += ["--index-url", index_url]
    cmd += packages
    return run(cmd, env=uv_env(runtime_dir), timeout=timeout, on_line=log,
               should_cancel=should_cancel, heartbeat=heartbeat)


# -- knowing what uv will fetch before it fetches it -------------------------
#
# uv is silent while it downloads, and the PyTorch build is gigabytes. The
# models get a real bar -- "2.3 / 12.4 GB" -- because we fetch them ourselves
# and know every size in advance. To give uv's downloads the same bar we ask
# uv what it *would* fetch (a dry run), look each file up on the index for its
# size and hash, download them with our own machinery, and hand uv the folder.

PYPI_SIMPLE = "https://pypi.org/simple"

# One line per chosen distribution in `uv pip install --dry-run -v`, verified
# against uv 0.12.10:
#     DEBUG Selecting: six==1.17.0 [compatible] (six-1.17.0-py2.py3-none-any.whl)
SELECTING = re.compile(
    r"Selecting: (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+) "
    r"\[[^\]]*\] \((?P<file>\S+\.(?:whl|tar\.gz|zip))\)")


@dataclass(frozen=True)
class WheelFile:
    """One file an install needs: what uv chose, and where the index keeps it."""

    name: str
    version: str
    filename: str
    url: str = ""
    sha256: str = ""
    size_bytes: int = 0


def parse_selected(output: str) -> list[WheelFile]:
    """The distributions a verbose dry run said it would fetch, in order."""
    found: list[WheelFile] = []
    seen: set[str] = set()
    for match in SELECTING.finditer(output):
        if match["file"] in seen:
            continue
        seen.add(match["file"])
        found.append(WheelFile(match["name"], match["version"], match["file"]))
    return found


def plan_install(
    uv: Path,
    runtime_dir: Path,
    packages: list[str],
    *,
    index_url: str | None = None,
    should_cancel: CancelFn | None = None,
) -> list[WheelFile]:
    """Ask uv which files an install would fetch, without fetching them.

    Empty when uv would fetch nothing (everything is installed already) and
    also when the dry run fails or says something we do not recognise; the
    caller then installs the ordinary way and lets uv report its own error.

    That includes uv not being runnable at all. This step exists only to put a
    progress bar on a download -- it is not the install, and it must not be
    the thing that decides an install is impossible. Whatever is really wrong
    surfaces from the install itself, where the message means something.
    """
    cmd: list[str | Path] = [uv, "pip", "install", "--dry-run", "-v",
                             "--python", venv_python(runtime_dir)]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages
    try:
        result = run(cmd, env=uv_env(runtime_dir), timeout=600, should_cancel=should_cancel)
    except OSError:
        return []
    if not result.ok:
        return []
    return parse_selected(result.stdout)


def normalise(name: str) -> str:
    """PEP 503 project-name normalisation: `typing_extensions` -> `typing-extensions`."""
    return re.sub(r"[-_.]+", "-", name).lower()


class _Anchors(HTMLParser):
    """filename -> href from a PEP 503 simple index page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, str] = {}
        self._href: str | None = None
        self._text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = ""

    def handle_data(self, data):
        if self._href is not None:
            self._text += data

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links[self._text.strip()] = self._href
            self._href = None


def locate(
    files: list[WheelFile],
    index_url: str,
    *,
    client: httpx.Client | None = None,
    token: str | None = None,
) -> list[WheelFile]:
    """Fill in each file's URL, hash and size from the index.

    Every file uv chose is listed on its project's simple page, as an anchor
    whose text is the filename and whose href carries the hash as a
    ``#sha256=`` fragment (PEP 503). The size comes from a HEAD request. A file
    the page does not list is returned with an empty URL; the caller treats
    any of those as "cannot do this properly" and lets uv fetch instead.
    """
    owned = client is None
    # Internet-facing, like download_file, so the user's proxy settings apply;
    # trust_env=False is for our loopback calls to ComfyUI only.
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    # Ask for the HTML flavour explicitly: PyPI also speaks PEP 691 JSON.
    accept = {"Accept": "application/vnd.pypi.simple.v1+html, text/html;q=0.9, */*;q=0.1"}
    pages: dict[str, tuple[str, dict[str, str]]] = {}
    try:
        located = []
        for item in files:
            key = normalise(item.name)
            if key not in pages:
                page_url = f"{index_url.rstrip('/')}/{key}/"
                links: dict[str, str] = {}
                try:
                    reply = client.get(page_url, headers=accept)
                    if reply.status_code == 200:
                        parser = _Anchors()
                        parser.feed(reply.text)
                        links = parser.links
                except httpx.HTTPError:
                    pass
                pages[key] = (page_url, links)
            page_url, links = pages[key]
            href = links.get(item.filename)
            if not href:
                located.append(item)
                continue
            url = urljoin(page_url, href)
            parts = urlsplit(url)
            sha = parse_qs(parts.fragment).get("sha256", [""])[0]
            clean = parts._replace(fragment="").geturl()
            located.append(replace(item, url=clean, sha256=sha,
                                   size_bytes=remote_size(clean, token=token, client=client)))
        return located
    finally:
        if owned:
            client.close()


def cache_bytes(runtime_dir: Path) -> int:
    """How much uv has put in its cache so far. It unzips wheels as they
    arrive, so this grows while a download is in flight."""
    total = 0
    stack = [runtime_dir / "uv-cache"]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


# Reports the name of device 0 as well as the count, because the count alone
# is misleading on ROCm: it exposes the CPU as an HSA agent, so a machine with
# one Radeon and a Ryzen reports two "devices" and ComfyUI lists them as
#     Device: cuda:0 AMD Radeon Graphics
#     Device: cuda:1 AMD Ryzen 7 7800X3D 8-Core Processor
# Telling someone they have two graphics cards on that basis is wrong, and
# sends them off setting --cuda-device 1, which selects the processor.
TORCH_PROBE = (
    "import json,torch;"
    "print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,"
    "'hip':getattr(torch.version,'hip',None),'available':torch.cuda.is_available(),"
    "'devices':torch.cuda.device_count(),"
    "'name':(torch.cuda.get_device_name(0) if torch.cuda.device_count() else '')}))"
)


@dataclass(frozen=True)
class TorchCheck:
    """The outcome of looking at the PyTorch that was just installed."""

    ok: bool
    message: str
    warning: str = ""


def verify_torch(
    runtime_dir: Path,
    expect_tag: str,
    *,
    env: dict[str, str] | None = None,
    log: LogFn | None = None,
    should_cancel: CancelFn | None = None,
) -> TorchCheck:
    """Prove the graphics card is really usable before downloading 40 GB.

    ``env`` is whatever the installer decided PyTorch needs to see the card --
    the HSA_OVERRIDE_GFX_VERSION for AMD cards ROCm does not list. Probing
    without it would report "cannot see your graphics card" on exactly the
    machines the override exists for, and stop an install that would have
    worked.

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
    result = run([venv_python(runtime_dir), "-c", TORCH_PROBE], timeout=300, on_line=log,
                 env=env, should_cancel=should_cancel)
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
    name = (info.get("name") or "").strip()
    good = f"{version} sees your {name}." if name else f"{version} sees your graphics card."

    if expect_tag and expect_tag not in version:
        return TorchCheck(True, good, warning=(
            f"This is the {version} build; we asked for {expect_tag}. "
            f"Your card works, so setup is carrying on."))
    return TorchCheck(True, good)
