"""Executing an install plan, and reporting honestly while it happens.

The plan says what should happen; this makes it happen. Every step reports
start, progress and outcome through a callback, so the UI can show a named step
moving rather than one opaque bar -- "install freezes at 70 percent with no
diagnosable state" is the most common complaint about every tool in this space.

Cancellation is cooperative and safe at any point: partial downloads survive as
.part files and the next run resumes them.

**Every step must be safe to run again.** An install that fetches tens of
gigabytes will be interrupted -- a closed laptop, a dropped connection, a
failure three steps later -- so "run it again" is the normal case, not the
exceptional one. A step that has already been done reports that and returns;
only a step that finds its work half-finished or wrong does it over. The first
version of this file got that wrong in one place, and a second run died on
"a virtual environment already exists" with no way past it.
"""

from __future__ import annotations

import shutil
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from toolshed.exec import inject, uvtool
from toolshed.exec.download import (
    Cancelled,
    DownloadError,
    download_file,
    file_digest,
    remote_size,
)
from toolshed.exec.manifest import Entry, Manifest
from toolshed.planner.plan import MODEL_DIRS, InstallPlan, Kind, Step

# How long a subprocess may go with no output *and* no bytes arriving before
# we call it stuck. Not a total budget: a 4 GB download on a slow line takes
# longer than any total we would dare set, and is fine as long as it moves.
STALL_SECONDS = 900


class Level(StrEnum):
    INFO = "info"
    LOG = "log"
    ERROR = "error"


@dataclass
class Event:
    kind: str                 # step_started | progress | step_done | step_failed | log
    step: Step | None = None
    message: str = ""
    level: Level = Level.INFO
    fraction: float = 0.0     # of this step
    overall: float = 0.0      # of the whole install
    bytes_done: int = 0
    bytes_total: int = 0


EventFn = Callable[[Event], None]
CancelFn = Callable[[], bool]


class InstallFailed(RuntimeError):
    def __init__(self, message: str, *, step: Step, reason_key: str = "step_failed") -> None:
        super().__init__(message)
        self.step = step
        self.reason_key = reason_key


class Runner:
    """Runs a plan. One instance per install."""

    def __init__(
        self,
        plan: InstallPlan,
        *,
        on_event: EventFn | None = None,
        should_cancel: CancelFn | None = None,
        hf_token: str | None = None,
        attempts: int = 5,
    ) -> None:
        self.plan = plan
        self.on_event = on_event or (lambda _e: None)
        self.should_cancel = should_cancel or (lambda: False)
        self.hf_token = hf_token
        self.attempts = attempts
        self.root = plan.data_root
        self.runtime = self.root / "runtime"
        self.comfy = self.root / "comfy"
        self.engine = self.root / "engine" / "comfyui"
        self.manifest = Manifest.load(self.root)
        self._done_steps = 0

    # -- reporting ----------------------------------------------------------

    def _emit(self, kind: str, **kw) -> None:
        overall = self._done_steps / max(len(self.plan.steps), 1)
        self.on_event(Event(kind=kind, overall=overall, **kw))

    def _log(self, message: str) -> None:
        self._emit("log", message=message, level=Level.LOG)

    def _check_cancelled(self) -> None:
        if self.should_cancel():
            raise Cancelled()

    # -- the loop -----------------------------------------------------------

    def run(self) -> Manifest:
        handlers = {
            Kind.ENSURE_UV: self._ensure_uv,
            Kind.ENSURE_PYTHON: self._ensure_python,
            Kind.CREATE_VENV: self._create_venv,
            Kind.INSTALL_TORCH: self._install_torch,
            Kind.VERIFY_TORCH: self._verify_torch,
            Kind.FETCH_ENGINE: self._fetch_engine,
            Kind.INSTALL_ENGINE_REQS: self._install_engine_reqs,
            Kind.MAKE_DIRS: self._make_dirs,
            Kind.DOWNLOAD: self._download,
            Kind.INJECT_WORKFLOW: self._inject,
            Kind.WRITE_SETTINGS: self._write_settings,
            Kind.SMOKE_TEST: self._smoke_test,
        }
        self._refuse_to_run_as_root()
        # packs is a statement of intent and is true from the start. engine_tag
        # and torch_index describe what actually landed, so they are written by
        # the steps that land them -- not here, where they would be a claim
        # made before the fact and left behind by a failure.
        #
        # Added to, never replaced: a second run that adds one pack must not
        # make the manifest forget the ones already installed, or the launcher
        # stops offering them.
        self.manifest.packs = sorted(set(self.manifest.packs) | {p.id for p in self.plan.packs})

        for step in self.plan.steps:
            self._check_cancelled()
            self._emit("step_started", step=step, message=step.title)
            try:
                handlers[step.kind](step)
            except Cancelled:
                raise
            except InstallFailed:
                raise
            except (DownloadError, OSError, RuntimeError) as exc:
                self._emit("step_failed", step=step, message=str(exc), level=Level.ERROR)
                raise InstallFailed(str(exc), step=step,
                                    reason_key=getattr(exc, "reason_key", "step_failed")) from exc
            self._done_steps += 1
            self._emit("step_done", step=step, message=step.title)
            self.manifest.save()
        return self.manifest

    # -- preflight ----------------------------------------------------------

    def _refuse_to_run_as_root(self) -> None:
        """Do not let sudo turn a stuck install into a broken home directory.

        Nothing here needs administrator rights: uv, Python, the engine and the
        models all live under the user's own data root. Running as root writes
        that whole tree owned by root, and the *next* ordinary run then fails on
        permissions -- a far worse and far more confusing state than whatever
        prompted someone to reach for sudo in the first place.
        """
        import os

        if hasattr(os, "geteuid") and os.geteuid() == 0 and os.environ.get("SUDO_USER"):
            raise InstallFailed(
                "Please run Toolshed normally, not with sudo. Nothing it installs "
                "needs administrator rights, and installing as root would leave "
                f"{self.root} owned by root and unusable from your own account.",
                step=self.plan.steps[0],
                reason_key="running_as_root",
            )

    # -- steps --------------------------------------------------------------

    def _ensure_uv(self, step: Step) -> None:
        uvtool.ensure_uv(self.runtime, log=self._log, should_cancel=self.should_cancel)

    def _ensure_python(self, step: Step) -> None:
        uv = uvtool.uv_path(self.runtime)
        result = uvtool.install_python(uv, self.runtime, step.payload["version"], log=self._log,
                                       should_cancel=self.should_cancel)
        if not result.ok:
            raise InstallFailed(f"Could not set up Python: {result.stderr or 'see the log'}",
                                step=step)

    def _create_venv(self, step: Step) -> None:
        """The private workspace: an isolated Python the engine runs inside.

        It exists so that installing PyTorch and ComfyUI's dependencies cannot
        touch, upgrade or break whatever Python the user already has, and so
        that uninstalling is deleting one folder.

        An existing one is good news, not an obstacle -- provided it works.
        Three cases, and only the last one is a failure:

        * it runs and is the right Python -> keep it, say so, move on;
        * it is there but broken or the wrong version -> replace it;
        * something is at that path that is not a virtual environment at all
          -> stop, because deleting it is not ours to decide.
        """
        from toolshed.planner.plan import PYTHON_VERSION

        venv = self.runtime / "venv"
        existing = uvtool.venv_python_version(self.runtime)

        if existing == PYTHON_VERSION:
            self._log(f"The private workspace is already here (Python {existing}); keeping it.")
            return

        if existing:
            self._log(f"Replacing the workspace: it has Python {existing}, "
                      f"and the engine needs {PYTHON_VERSION}.")
        elif venv.exists():
            self._log("The workspace was left half-made by an earlier run; starting it again.")

        uv = uvtool.uv_path(self.runtime)
        result = uvtool.create_venv(uv, self.runtime, PYTHON_VERSION,
                                    clear=venv.exists(), log=self._log,
                                    should_cancel=self.should_cancel)
        if not result.ok:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            hint = detail[-1] if detail else "see the log"
            raise InstallFailed(
                f"Could not create the private workspace at {venv}. {hint}", step=step)

    def _install_torch(self, step: Step) -> None:
        index = step.payload.get("index_url")
        if not index:
            raise InstallFailed("No suitable PyTorch build for this machine.", step=step)
        self._pip_install_watched(step, ["torch", "torchvision", "torchaudio"],
                                  index_url=index, what="the graphics card software")
        self.manifest.torch_index = index
        # Recorded so the launcher starts the engine with the same environment
        # the card was verified under. Without it an AMD card that needs the
        # HSA override passes the check here and vanishes at run time.
        self.manifest.torch_env = dict(step.payload.get("env") or {})

    def _verify_torch(self, step: Step) -> None:
        env = dict(step.payload.get("env") or self.manifest.torch_env or {})
        check = uvtool.verify_torch(self.runtime, step.payload.get("expect_tag", ""),
                                    env=env, log=self._log, should_cancel=self.should_cancel)
        if not check.ok:
            raise InstallFailed(check.message, step=step, reason_key="torch_unusable")
        if check.warning:
            # Recorded as well as logged: a build that is not the one we asked
            # for is worth knowing about later, when something behaves oddly.
            self.manifest.notes.append(check.warning)
            self._log(check.warning)
        self._log(check.message)

    def _fetch_engine(self, step: Step) -> None:
        tag = step.payload["tag"]
        # The manifest records the tag only once the tree is fully in place, so
        # an interrupted extraction is never mistaken for a finished one.
        if self.manifest.engine_tag == tag and (self.engine / "requirements.txt").is_file():
            self._log(f"ComfyUI {tag} is already installed; keeping it.")
            return

        self.engine.parent.mkdir(parents=True, exist_ok=True)
        archive = self.root / "state" / "downloads" / f"comfyui-{step.payload['tag']}.tar.gz"
        download_file(step.payload["url"], archive,
                      on_progress=lambda p: self._emit(
                          "progress", step=step, fraction=p.fraction,
                          bytes_done=p.downloaded, bytes_total=p.total),
                      should_cancel=self.should_cancel)

        # Replace wholesale: the engine tree is disposable by design, so an
        # interrupted extraction can never leave a half-updated engine behind.
        staging = self.engine.with_suffix(".new")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with tarfile.open(archive) as tf:
            tf.extractall(staging, filter="data")
        inner = next((p for p in staging.iterdir() if p.is_dir()), staging)
        if self.engine.exists():
            shutil.rmtree(self.engine)
        inner.replace(self.engine)
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
        self.manifest.engine_tag = tag

    def _install_engine_reqs(self, step: Step) -> None:
        reqs = self.engine / "requirements.txt"
        if not reqs.is_file():
            raise InstallFailed("The engine download looks incomplete.", step=step)
        # Against PyPI, not the torch index: the previous step replaced the
        # index entirely, and these are ordinary packages.
        self._pip_install_watched(step, ["-r", str(reqs)], index_url=None,
                                  what="the engine's dependencies")

    def _pip_install_watched(self, step: Step, packages: list[str], *,
                             index_url: str | None, what: str) -> None:
        """A pip install that looks like every other download on the screen.

        uv says nothing while it fetches, and the PyTorch build is gigabytes.
        So: ask uv what it would fetch, look the files up for their sizes,
        download them ourselves -- the same bar, the same "2.3 / 4.6 GB", the
        same resume -- and hand uv the folder to install from offline. If any
        part of that cannot be worked out, uv fetches as before and our count
        of its cache stands in for a bar.
        """
        uv = uvtool.uv_path(self.runtime)
        self._emit("progress", step=step, message="Working out which files are needed…")
        wanted = uvtool.plan_install(uv, self.runtime, packages, index_url=index_url,
                                     should_cancel=self.should_cancel)
        self._check_cancelled()

        find_links = None
        if wanted:
            located = uvtool.locate(wanted, index_url or uvtool.PYPI_SIMPLE, token=None)
            if all(f.url for f in located):
                find_links = self.root / "state" / "downloads" / "wheels"
                self._fetch_wheels(step, located, find_links)
            else:
                missing = [f.filename for f in located if not f.url]
                self._log(f"Could not find {missing[0]} on the index to size it up; "
                          f"letting uv fetch instead.")

        install = uvtool.pip_install(
            uv, self.runtime, packages, index_url=index_url, log=self._log,
            timeout=STALL_SECONDS, should_cancel=self.should_cancel,
            heartbeat=self._watch_uv_cache(step), find_links=find_links)
        if not install.ok and find_links is not None:
            # The files are all there, so this is uv disagreeing with its own
            # dry run. Rare; let it fetch for itself rather than fail.
            self._log("Installing from the downloaded files did not work; "
                      "asking uv to fetch them itself.")
            install = uvtool.pip_install(
                uv, self.runtime, packages, index_url=index_url, log=self._log,
                timeout=STALL_SECONDS, should_cancel=self.should_cancel,
                heartbeat=self._watch_uv_cache(step))
        if not install.ok:
            raise InstallFailed(f"Could not install {what}. "
                                + (install.stderr or "See the log for what uv said."),
                                step=step)
        if find_links is not None:
            # uv has unpacked them into the workspace; gigabytes of wheel files
            # kept as well would double what this step costs in disk.
            shutil.rmtree(find_links, ignore_errors=True)

    def _fetch_wheels(self, step: Step, files: list[uvtool.WheelFile], into) -> None:
        """Download uv's shopping list with the models' own progress bar."""
        into.mkdir(parents=True, exist_ok=True)
        total = sum(f.size_bytes for f in files)
        self._log(f"{len(files)} files, {total / 1e9:.1f} GB to download.")
        done = 0
        for item in files:
            self._check_cancelled()
            target = into / item.filename
            if (target.is_file() and item.size_bytes
                    and target.stat().st_size == item.size_bytes and not item.sha256):
                done += item.size_bytes
                continue

            def progress(p, base=done, name=item.filename):
                self._emit("progress", step=step,
                           fraction=(base + p.downloaded) / max(total, 1),
                           bytes_done=base + p.downloaded, bytes_total=total,
                           message=f"{name} — {p.bytes_per_second / 1e6:.1f} MB/s")

            download_file(item.url, target, sha256=item.sha256 or None,
                          size_bytes=item.size_bytes, attempts=self.attempts,
                          on_progress=progress, should_cancel=self.should_cancel)
            done += target.stat().st_size
        self._emit("progress", step=step, fraction=1.0, bytes_done=done, bytes_total=total,
                   message="Downloaded. Unpacking…")

    def _watch_uv_cache(self, step: Step) -> Callable[[], bool]:
        """Something to show while uv downloads in silence.

        uv unzips wheels into its cache as the bytes arrive, so the cache's size
        is a live count of what has been received. Reported as our own line
        under the step, with a rate, so a twenty-minute download reads as a
        download and not as a hang. Returns whether anything arrived since the
        last look, which is what keeps the stall deadline from firing on a slow
        but healthy connection.
        """
        start_bytes = last_bytes = uvtool.cache_bytes(self.runtime)
        start_time = last_time = time.monotonic()

        def beat() -> bool:
            nonlocal last_bytes, last_time
            now_bytes = uvtool.cache_bytes(self.runtime)
            now = time.monotonic()
            grew = now_bytes > last_bytes
            if grew:
                rate = (now_bytes - last_bytes) / max(now - last_time, 1e-6)
                received = now_bytes - start_bytes
                self._emit("progress", step=step,
                           message=f"{received / 1e9:.2f} GB received so far — "
                                   f"{rate / 1e6:.1f} MB/s")
            elif now - start_time > 20 and now_bytes == start_bytes:
                self._emit("progress", step=step,
                           message="Waiting for the download to start… "
                                   "(working out which files are needed)")
            last_bytes, last_time = now_bytes, now
            return grew

        return beat

    def _make_dirs(self, step: Step) -> None:
        for name in step.payload.get("model_dirs", MODEL_DIRS):
            (self.root / "models" / name).mkdir(parents=True, exist_ok=True)
        for rel in ("comfy/user/default", "comfy/input", "comfy/temp", "comfy/custom_nodes",
                    "output/image", "output/video", "output/audio", "output/3d",
                    "state/downloads", "state/logs"):
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    def _already_have(self, target, item) -> str | None:
        """The file's sha256 if it is already here, complete and unchanged since
        we fetched it; None if it has to be downloaded.

        Only ``download_file`` can answer that from the catalogue, and only once
        the catalogue's hashes are frozen; until then every hash is
        PENDING_FREEZE and it has nothing to compare against. So our own record
        answers instead: what the file hashed to when it landed, from the
        manifest or -- after an uninstall that kept the models and removed the
        manifest -- from the sidecar kept beside the models.

        Checked rather than assumed. Re-hashing 40 GB takes under a minute and
        catches a truncated or edited file; trusting the record instead would
        make a re-run quietly build on a corrupt model. The size is compared
        first because that rules most stale files out without reading them.
        """
        if item.hash_is_frozen:
            return None         # download_file does this check itself, better.
        if not target.is_file():
            return None
        try:
            key = str(target.relative_to(self.root))
        except ValueError:
            return None         # outside the data root; the manifest cannot speak for it
        prior = self.manifest.prior_hash(key)
        if prior is None or prior[1] != target.stat().st_size:
            return None
        self._log(f"Checking {item.filename} is still intact…")
        return prior[0] if file_digest(target) == prior[0] else None

    def _download(self, step: Step) -> None:
        pack_id = step.payload.get("id", "")
        done_bytes = 0
        for item in step.downloads:
            self._check_cancelled()
            target = item.dest / item.filename

            # A re-run must not fetch tens of gigabytes it already has. This is
            # what makes "just start it again" a reasonable thing to tell
            # someone whose install died two hours in.
            if kept := self._already_have(target, item):
                done_bytes += target.stat().st_size
                self._log(f"{item.filename} is already here; skipping.")
                # Written down again: after an uninstall the manifest is
                # gone, and this run's record must cover the kept files too.
                self._remember(target, item, kept, pack_id)
                continue

            size = item.size_bytes or remote_size(item.url, token=self.hf_token)
            self._log(f"{item.filename} ({size / 1e9:.1f} GB)" if size else item.filename)

            # base and name are bound now, not when the callback fires: a
            # late-bound `item` would label every file with the last one's name.
            def progress(p, base=done_bytes, name=item.filename):
                self._emit("progress", step=step,
                           fraction=(base + p.downloaded) / max(step.bytes_total, 1),
                           bytes_done=base + p.downloaded, bytes_total=step.bytes_total,
                           message=f"{name} — {p.bytes_per_second / 1e6:.1f} MB/s")

            digest = download_file(
                item.url, target,
                sha256=item.sha256 if item.hash_is_frozen else None,
                size_bytes=item.size_bytes, token=self.hf_token, attempts=self.attempts,
                on_progress=progress, should_cancel=self.should_cancel)
            done_bytes += target.stat().st_size
            self._remember(target, item, digest, pack_id)

    def _remember(self, target, item, digest: str, pack_id: str) -> None:
        """Record a landed file, and write the record out *now*.

        The manifest used to be saved only when the whole pack step finished.
        A pack is several files and many gigabytes; an install that died on
        the third file had two complete, verified models on disk and no record
        of either, so the next run hashed nothing, trusted nothing, and fetched
        them again. Each file is written down the moment it lands.
        """
        entry = Entry(
            path=str(target.relative_to(self.root)), sha256=digest,
            size_bytes=target.stat().st_size, source=item.url, pack=pack_id,
            hash_verified_against="release" if item.hash_is_frozen else "publisher")
        self.manifest.record(entry)
        self.manifest.remember_hash(entry)
        self.manifest.save()

    def _inject(self, step: Step) -> None:
        inject.inject([step.payload["id"]], self.comfy)

    def _write_settings(self, step: Step) -> None:
        import json

        settings = self.comfy / "user" / "default" / "comfy.settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        current = {}
        if settings.is_file():
            try:
                current = json.loads(settings.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        # Only defaults a beginner benefits from; never clobber a real choice.
        current.setdefault("Comfy.Workflow.ShowMissingModelsWarning", True)
        current.setdefault("Comfy.Workflow.ShowMissingNodesWarning", True)
        current.setdefault("Comfy.Workflow.ConfirmDelete", True)
        current.setdefault("Comfy.Window.ConfirmOnClose", True)
        settings.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def _smoke_test(self, step: Step) -> None:
        # Generating a real artefact needs the engine running; that lands with
        # the launcher. Recorded rather than silently skipped.
        self.manifest.notes.append(f"smoke test not yet run for {step.payload.get('id')}")
        self._log("Skipped for now — the launcher lands next.")
