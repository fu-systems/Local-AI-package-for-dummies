#!/usr/bin/env python3
"""Derive recipe skeletons from Comfy Org's official workflow templates.

The catalogue is DERIVED, never hand-written. Every model filename and every
destination folder in a Toolshed recipe must come from the upstream template
that recipe's workflow is adapted from, because those are the only values we
can prove ComfyUI will accept. Hand-typing them produces `value_not_in_list`
on a beginner's first run, which is the exact failure this project exists to
prevent.

This tool reads two things from `Comfy-Org/workflow_templates` at a pinned ref:

  templates/index.json   -> title, minComfyUIVersion, total download size
  templates/<name>.json  -> properties.models[] on every node, and the graph
                            shape (flat, or subgraph-based with typed inputs)

and emits a recipe skeleton per template with `PENDING_FREEZE` wherever a fact
must come from Hugging Face instead (byte size, sha256, gated flag). Those are
filled in later by tools/freeze_manifest.py.

Nothing here contacts Hugging Face. Nothing here writes a hash. If a value is
not present upstream, this tool emits PENDING_FREEZE rather than a guess.

Usage:
    python3 tools/derive_catalog.py --list
    python3 tools/derive_catalog.py --all --out catalog/recipes
    python3 tools/derive_catalog.py image_z_image_turbo_int8 --print
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{ref}/templates/{name}"
DEFAULT_REF = "main"
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "templates"

PENDING = "PENDING_FREEZE"

# Destination folders we are willing to write into. Anything else means the
# template needs a `folder_paths` key we have not reviewed, and that is a
# decision for a human, not a default.
KNOWN_DESTS = {
    "checkpoints", "diffusion_models", "unet", "text_encoders", "clip",
    "clip_vision", "vae", "vae_approx", "loras", "controlnet", "upscale_models",
    "embeddings", "audio_encoders", "model_patches", "style_models",
    "latent_upscale_models", "geometry_estimation", "background_removal",
    "photomaker", "gligen", "hypernetworks",
}

# Extension allowlist. Pickle formats (.ckpt/.pt/.pth/.bin) execute arbitrary
# code on load and are refused outright, with no per-model exception.
ALLOWED_SUFFIXES = (".safetensors", ".sft", ".gguf")

# Templates that back a v1 recipe, and the recipe id each maps to.
V1_TEMPLATES = {
    "image_z_image_turbo":            "image.zimage_turbo",
    "image_z_image_turbo_int8":       "image.zimage_turbo",
    "image_sdxl_simple":              "image.sdxl",
    "image_qwen_image_edit_2511_int8": "image.qwen_edit",
    "video_wan2_2_5B_ti2v":           "video.wan22_5b",
    # The big one. Only reachable on a 20 GB card with layer streaming
    # switched on, which is why it was not here before that existed.
    "video_wan2_2_14B_t2v":           "video.wan22_14b",
    "audio_ace_step_1_5_checkpoint":  "audio.acestep",
    "3d_pixal3d_trellis2_image_to_model": "model3d.trellis2",
}


class DeriveError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def fetch(name: str, ref: str, refresh: bool = False) -> Any:
    """Fetch a template file, caching it under .cache/ so reruns are offline."""
    cache_path = CACHE / ref / name
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = RAW.format(ref=ref, name=name)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise DeriveError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DeriveError(f"cannot reach {url}: {exc.reason}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(body, encoding="utf-8")
    return json.loads(body)


def resolve_ref(ref: str) -> str:
    """Turn a branch name into the commit SHA it currently points at.

    Recipes pin an immutable commit, never a branch: a branch would let the
    upstream graph change under a frozen hash.
    """
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    url = f"https://api.github.com/repos/Comfy-Org/workflow_templates/commits/{ref}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.sha"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            sha = resp.read().decode("utf-8").strip()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"warning: could not resolve {ref} to a commit ({exc}); "
              f"emitting {PENDING} for the pin", file=sys.stderr)
        return PENDING
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else PENDING


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

@dataclass
class Model:
    node_type: str
    filename: str
    dest: str
    repo: str
    path: str
    url: str

    @property
    def problems(self) -> list[str]:
        out = []
        if self.dest not in KNOWN_DESTS:
            out.append(f"unreviewed destination folder {self.dest!r}")
        if not self.filename.endswith(ALLOWED_SUFFIXES):
            out.append(f"disallowed extension on {self.filename!r} "
                       f"(allowed: {', '.join(ALLOWED_SUFFIXES)})")
        if not self.repo:
            out.append(f"model url is not a Hugging Face resolve URL: {self.url}")
        return out


@dataclass
class Template:
    name: str
    title: str = ""
    min_comfyui: str = ""
    total_size_bytes: int | None = None
    shape: str = "flat"                       # "flat" | "subgraph"
    subgraph_inputs: list[str] = field(default_factory=list)
    subgraph_name: str = ""
    node_classes: set[str] = field(default_factory=set)
    models: list[Model] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        return [f"{m.filename}: {p}" for m in self.models for p in m.problems]


HF_RESOLVE = re.compile(
    r"https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)"
)


def parse_model(node_type: str, entry: dict) -> Model:
    url = (entry.get("url") or "").strip()
    m = HF_RESOLVE.match(url)
    repo = m.group("repo") if m else ""
    path = m.group("path") if m else ""
    filename = (entry.get("name") or "").strip()

    # Upstream truncates some `url` values mid-filename (an authoring bug in a
    # few templates). The `name` field is authoritative for the filename, so
    # repair the path tail from it rather than emitting a broken URL.
    if path and filename and not path.endswith(filename):
        path = f"{path.rsplit('/', 1)[0]}/{filename}" if "/" in path else filename

    return Model(
        node_type=node_type,
        filename=filename,
        dest=(entry.get("directory") or "").strip(),
        repo=repo,
        path=path,
        url=url,
    )


def iter_nodes(doc: dict) -> Iterator[tuple[dict, bool]]:
    """Yield (node, is_inside_subgraph) for every node in a template."""
    for node in doc.get("nodes", []) or []:
        yield node, False
    for sub in (doc.get("definitions") or {}).get("subgraphs", []) or []:
        for node in sub.get("nodes", []) or []:
            yield node, True


# Frontend-only nodes. They never execute and must never reach POST /prompt.
VIRTUAL_NODES = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def parse_template(name: str, doc: dict, index_entry: dict | None) -> Template:
    tpl = Template(name=name)
    if index_entry:
        tpl.title = index_entry.get("title", "")
        tpl.min_comfyui = str(index_entry.get("minComfyUIVersion") or "")
        size = index_entry.get("size")
        tpl.total_size_bytes = size if isinstance(size, int) else None

    subgraphs = (doc.get("definitions") or {}).get("subgraphs") or []
    if subgraphs:
        tpl.shape = "subgraph"
        first = subgraphs[0]
        tpl.subgraph_name = first.get("name", "")
        tpl.subgraph_inputs = [i.get("name", "") for i in (first.get("inputs") or [])]
        if len(subgraphs) > 1:
            print(f"note: {name} defines {len(subgraphs)} subgraphs; "
                  f"only the first is described", file=sys.stderr)

    seen: set[tuple[str, str]] = set()
    for node, _inside in iter_nodes(doc):
        ntype = node.get("type", "")
        if ntype and ntype not in VIRTUAL_NODES and not UUID_RE.match(ntype):
            tpl.node_classes.add(ntype)
        for entry in (node.get("properties") or {}).get("models") or []:
            model = parse_model(ntype, entry)
            key = (model.filename, model.dest)
            if key in seen:
                continue
            seen.add(key)
            tpl.models.append(model)

    return tpl


# --------------------------------------------------------------------------
# emitting
# --------------------------------------------------------------------------

def slug(filename: str) -> str:
    """A stable YAML key for a model file."""
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def to_recipe(tpl: Template, ref: str) -> dict:
    files: dict[str, Any] = {}
    for m in tpl.models:
        files[slug(m.filename)] = {
            "source": "huggingface",
            "repo": m.repo or PENDING,
            "revision": PENDING,          # immutable commit SHA of the MODEL repo
            "path": m.path or PENDING,
            "dest": m.dest,
            "filename": m.filename,
            "sha256": PENDING,            # from HF paths-info: lfs.oid
            "size_bytes": PENDING,
            "gated": PENDING,             # from GET /api/models/{repo}: "gated"
            "licence": PENDING,
            "from_node": m.node_type,     # provenance, for review
        }

    workflow: dict[str, Any] = {
        "id": PENDING,
        "title": tpl.title or tpl.name,
        "inject_as": PENDING,
        "derived_from": {
            "repo": "Comfy-Org/workflow_templates",
            "commit": ref,
            "template": tpl.name,
        },
        "shape": tpl.shape,
        "bind_mode": "subgraph_inputs" if tpl.shape == "subgraph" else "node_ids",
    }
    if tpl.shape == "subgraph":
        workflow["subgraph_name"] = tpl.subgraph_name
        workflow["available_inputs"] = tpl.subgraph_inputs
        workflow["bind"] = {name: name for name in tpl.subgraph_inputs}
    else:
        workflow["bind"] = {}
        workflow["_note"] = ("flat graph: bind by node id, filled in when the "
                             "workflow triple is authored")

    return {
        "schema_version": 1,
        "id": V1_TEMPLATES.get(tpl.name, PENDING),
        "name": PENDING,
        "blurb": PENDING,
        "modality": PENDING,
        "default_checked": False,
        "requires": {
            "comfyui_min": tpl.min_comfyui or PENDING,
            "vendors": ["nvidia", "amd"],
            "node_classes": sorted(tpl.node_classes),
        },
        "licence": {"id": PENDING, "gate": PENDING},
        "custom_nodes": [],
        "estimated_download_bytes": tpl.total_size_bytes or PENDING,
        "variants": [
            {"id": PENDING, "label": PENDING, "when": {"vram_gb_min": PENDING},
             "files": sorted(files)}
        ],
        "files": files,
        "workflows": [workflow],
        "presets": [],
    }


HEADER = """\
# GENERATED by tools/derive_catalog.py -- do not hand-edit model filenames or
# destination folders. They are derived from the upstream template named under
# workflows[].derived_from and re-checked by tools/lint_catalog.py.
#
# Every PENDING_FREEZE below is a fact this tool refuses to guess:
#   revision / sha256 / size_bytes / gated  -> tools/freeze_manifest.py (Hugging Face)
#   id / name / blurb / modality / licence / variants / presets -> a human
#
# A release build fails while any PENDING_FREEZE remains.
"""


def write_recipe(tpl: Template, ref: str, out_dir: Path) -> Path:
    import yaml

    recipe = to_recipe(tpl, ref)
    dest = out_dir / f"{tpl.name}.generated.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True, width=100)
    dest.write_text(HEADER + body, encoding="utf-8")
    return dest


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def load_index(ref: str, refresh: bool) -> dict[str, dict]:
    raw = fetch("index.json", ref, refresh)
    out: dict[str, dict] = {}
    for category in raw:
        for entry in category.get("templates", []) or []:
            entry = dict(entry)
            entry["_category"] = category.get("title", "")
            out[entry["name"]] = entry
    return out


def human_bytes(n: int | None) -> str:
    if not isinstance(n, int):
        return "?"
    return f"{n / 1e9:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("templates", nargs="*", help="template names to derive")
    ap.add_argument("--all", action="store_true", help="derive every v1 template")
    ap.add_argument("--list", action="store_true", help="list the v1 templates and exit")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print a summary instead of writing YAML")
    ap.add_argument("--out", type=Path, default=Path("catalog/recipes"))
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help=f"upstream ref to read (default {DEFAULT_REF})")
    ap.add_argument("--refresh", action="store_true", help="ignore the local cache")
    args = ap.parse_args(argv)

    try:
        index = load_index(args.ref, args.refresh)
    except DeriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for name, recipe_id in sorted(V1_TEMPLATES.items()):
            entry = index.get(name)
            if entry is None:
                print(f"  {name:38s} MISSING FROM UPSTREAM INDEX")
                continue
            print(f"  {name:38s} -> {recipe_id:20s} "
                  f"{human_bytes(entry.get('size')):>8s}  "
                  f"needs ComfyUI {entry.get('minComfyUIVersion')}")
        return 0

    names = list(V1_TEMPLATES) if args.all else args.templates
    if not names:
        ap.error("give one or more template names, or --all, or --list")

    pinned = resolve_ref(args.ref)
    if pinned == PENDING:
        print("warning: recipes will carry PENDING_FREEZE for the template commit",
              file=sys.stderr)
    else:
        print(f"pinning Comfy-Org/workflow_templates at {pinned}")

    problems = 0
    for name in names:
        if name not in index:
            print(f"error: {name!r} is not in the upstream template index", file=sys.stderr)
            problems += 1
            continue
        try:
            doc = fetch(f"{name}.json", args.ref, args.refresh)
        except DeriveError as exc:
            print(f"error: {exc}", file=sys.stderr)
            problems += 1
            continue

        tpl = parse_template(name, doc, index.get(name))
        print(f"\n{tpl.name}  [{tpl.shape}]  {human_bytes(tpl.total_size_bytes)}  "
              f"needs ComfyUI {tpl.min_comfyui or '?'}")
        if tpl.shape == "subgraph":
            print(f"  subgraph {tpl.subgraph_name!r} inputs: "
                  f"{', '.join(tpl.subgraph_inputs) or '(none)'}")
        for m in tpl.models:
            print(f"  model  {m.filename:52s} -> {m.dest:20s} ({m.repo or 'NON-HF'})")
        if not tpl.models:
            print("  model  (none declared upstream -- verify by hand)")

        for problem in tpl.problems:
            print(f"  PROBLEM  {problem}", file=sys.stderr)
            problems += 1

        if not args.print_only:
            dest = write_recipe(tpl, pinned, args.out)
            print(f"  wrote  {dest}")

    if problems:
        print(f"\n{problems} problem(s) need a human decision", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
