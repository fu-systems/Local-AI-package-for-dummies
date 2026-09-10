"""The install plan: an ordered list of typed, idempotent steps.

The plan is *data*. Building it touches nothing -- no network, no disk -- which
is what lets the confirmation screen show exactly what is about to happen, and
lets the whole policy be tested without a GPU or an internet connection.

Executing it is somebody else's job (toolshed/exec/runner.py). Keeping those
apart is deliberate: it means a plan can be inspected, diffed, logged and
tested without any risk of it running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from toolshed.catalog.packs import RECIPE_SUFFIXES, Pack
from toolshed.hw.detect import HardwareReport
from toolshed.planner.torchsel import TorchChoice, choose_torch

PENDING = "PENDING_FREEZE"


class Kind(StrEnum):
    ENSURE_UV = "ensure_uv"
    ENSURE_PYTHON = "ensure_python"
    CREATE_VENV = "create_venv"
    INSTALL_TORCH = "install_torch"
    VERIFY_TORCH = "verify_torch"
    FETCH_ENGINE = "fetch_engine"
    INSTALL_ENGINE_REQS = "install_engine_reqs"
    MAKE_DIRS = "make_dirs"
    DOWNLOAD = "download"
    INJECT_WORKFLOW = "inject_workflow"
    WRITE_SETTINGS = "write_settings"
    SMOKE_TEST = "smoke_test"


@dataclass(frozen=True)
class Download:
    """One file to fetch.

    ``sha256`` is the frozen hash from the catalogue when there is one. Where
    the catalogue still says PENDING_FREEZE we resolve size and hash from the
    publisher at install time and verify against that instead: weaker, because
    it trusts the publisher at that moment rather than at release time, but it
    still catches every truncated or corrupted transfer. Which of the two was
    used is recorded in the manifest, so it is never a mystery afterwards.
    """

    url: str
    dest: Path
    filename: str
    sha256: str = PENDING
    size_bytes: int = 0
    licence: str = ""

    @property
    def hash_is_frozen(self) -> bool:
        return self.sha256 != PENDING and len(self.sha256) == 64


@dataclass(frozen=True)
class Step:
    kind: Kind
    title: str                       # shown to the user, in their words
    detail: str = ""
    bytes_total: int = 0
    payload: dict = field(default_factory=dict)
    downloads: tuple[Download, ...] = ()

    @property
    def id(self) -> str:
        suffix = self.payload.get("id") or self.title
        return f"{self.kind.value}:{suffix}"


@dataclass(frozen=True)
class InstallPlan:
    steps: tuple[Step, ...]
    torch: TorchChoice
    data_root: Path
    packs: tuple[Pack, ...] = ()

    @property
    def bytes_total(self) -> int:
        return sum(s.bytes_total for s in self.steps)

    @property
    def downloads(self) -> tuple[Download, ...]:
        return tuple(d for s in self.steps for d in s.downloads)

    @property
    def has_unfrozen_hashes(self) -> bool:
        return any(not d.hash_is_frozen for d in self.downloads)


# Pinned engine. Bumping these is a deliberate, tested act; see docs/UPSTREAM.md.
ENGINE_TAG = "v0.34.0"
ENGINE_URL = f"https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/refs/tags/{ENGINE_TAG}"
PYTHON_VERSION = "3.12"

# Model folders the engine expects. Created up front because the engine
# validates these paths at startup and aborts with a bare usage error if they
# are missing.
MODEL_DIRS = (
    "checkpoints", "diffusion_models", "text_encoders", "clip_vision", "vae",
    "vae_approx", "loras", "controlnet", "upscale_models", "embeddings",
    "audio_encoders", "model_patches", "geometry_estimation", "background_removal",
)


def build_plan(
    report: HardwareReport,
    packs: tuple[Pack, ...] | list[Pack],
    data_root: Path,
    *,
    installed: set[str] | None = None,
) -> InstallPlan:
    """Turn a hardware report and a selection into an ordered plan.

    ``installed`` is the set of download filenames already present and verified,
    so adding a pack later re-downloads nothing. Pure: it decides, it does not act.
    """
    packs = tuple(packs)
    installed = installed or set()
    torch = choose_torch(report)

    steps: list[Step] = [
        Step(Kind.ENSURE_UV, "Preparing the installer",
             "Fetching uv, which manages Python for us."),
        Step(Kind.ENSURE_PYTHON, "Setting up Python",
             f"Python {PYTHON_VERSION}, kept separate from anything already on your computer.",
             payload={"version": PYTHON_VERSION}),
        Step(Kind.CREATE_VENV, "Making a private workspace"),
        Step(Kind.INSTALL_TORCH, "Setting up your graphics card",
             torch.message,
             payload={"index_url": torch.index_url, "env": dict(torch.env)}),
        # Immediately after, never later: catching a CPU-only build here costs
        # ninety seconds, catching it after the models costs an hour and 40 GB.
        Step(Kind.VERIFY_TORCH, "Checking your graphics card is really being used",
             payload={"expect_tag": torch.expected_local_tag, "env": dict(torch.env)}),
        Step(Kind.FETCH_ENGINE, "Installing the AI engine",
             f"ComfyUI {ENGINE_TAG}",
             payload={"url": ENGINE_URL, "tag": ENGINE_TAG}),
        Step(Kind.INSTALL_ENGINE_REQS, "Installing the engine's dependencies"),
        Step(Kind.MAKE_DIRS, "Creating folders",
             payload={"model_dirs": list(MODEL_DIRS)}),
    ]

    # One download step per pack, so progress is reported in terms the user
    # recognises ("Wan 2.2 video model") rather than as one opaque bar.
    for pack in packs:
        files = tuple(d for d in _downloads_for(pack, data_root)
                      if d.filename not in installed)
        if not files:
            continue
        steps.append(Step(
            Kind.DOWNLOAD,
            f"Downloading {pack.name.lower()}",
            pack.blurb,
            bytes_total=sum(d.size_bytes for d in files) or (pack.download_bytes or 0),
            payload={"id": pack.id},
            downloads=files,
        ))

    for pack in packs:
        steps.append(Step(Kind.INJECT_WORKFLOW, f"Adding the {pack.name.lower()} workflow",
                          payload={"id": pack.id, "recipe": pack.recipe}))

    steps.append(Step(Kind.WRITE_SETTINGS, "Setting sensible defaults"))
    for pack in packs:
        steps.append(Step(Kind.SMOKE_TEST, f"Testing {pack.name.lower()}",
                          "Making one real thing, to prove it works.",
                          payload={"id": pack.id}))

    return InstallPlan(tuple(steps), torch, data_root, packs)


def _downloads_for(pack: Pack, data_root: Path) -> tuple[Download, ...]:
    """Read the files a pack needs from its derived recipe."""
    import yaml

    from toolshed import resources

    # Both recipe kinds, and for the same reason packs.py reads both: a pack
    # whose graph is an existing template but whose model is not has nothing
    # upstream to derive from and is written by hand.
    #
    # This read .generated.yaml alone, so image.sdxl_adult -- the one authored
    # recipe we ship -- resolved to no downloads at all. Nothing failed: the
    # pack installed, fetched nothing, and the first generation died on a
    # missing checkpoint, a long way from the cause. The catalogue loader
    # learned about authored recipes when they were introduced and this did
    # not, which is exactly the kind of split RECIPE_SUFFIXES exists to stop.
    doc: dict = {}
    for suffix in RECIPE_SUFFIXES:
        path = resources.resource_path("catalog", "recipes", f"{pack.recipe}{suffix}")
        if path.is_file():
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            break
    else:
        return ()

    out: list[Download] = []
    for entry in (doc.get("files") or {}).values():
        dest, filename = entry.get("dest"), entry.get("filename")
        if not all((dest, filename)):
            continue

        # A direct URL, for a model that is not on Hugging Face at all. Every
        # derived recipe names a repo and a path because that is what the
        # templates give us, but a community checkpoint is often published
        # somewhere with no such structure, and the alternative to expressing
        # that here is not expressing it anywhere.
        url = entry.get("url")
        if url and PENDING not in url:
            pass
        elif url:
            continue                      # named, but nobody has filled it in
        else:
            repo, rel = entry.get("repo"), entry.get("path")
            if not all((repo, rel)) or PENDING in (repo, rel):
                continue
            revision = entry.get("revision")
            revision = "main" if not revision or revision == PENDING else revision
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{rel}"

        size = entry.get("size_bytes")
        out.append(Download(
            url=url,
            dest=data_root / "models" / dest,
            filename=filename,
            sha256=entry.get("sha256") or PENDING,
            size_bytes=size if isinstance(size, int) else 0,
            licence=pack.licence,
        ))
    return tuple(out)
