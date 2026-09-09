#!/usr/bin/env python3
"""Turn Comfy Org's official templates into the workflows Toolshed injects.

A beginner opening ComfyUI for the first time sees a node graph and no idea
which box to type in. Every injected workflow therefore gains a large
"START HERE" note, positioned to the left of the graph so it is the first thing
on screen, written in plain language.

The graphs themselves are otherwise left exactly as Comfy Org authored them.
They are tested upstream, they reference the model filenames our catalogue
downloads, and diverging from them would mean owning every future breakage.

    python3 tools/build_workflows.py            # from the cached templates
    python3 tools/build_workflows.py --refresh  # re-fetch them first

Output is committed; this only needs rerunning when a note or template changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "templates"
OUT = REPO / "workflows"
# The templates version the engine we pin actually ships. ComfyUI v0.34.0's
# requirements.txt reads:
#
#     comfyui-workflow-templates==0.11.48
#
# so that is the tag these graphs come from. Not `main`: a branch lets the
# graph change under a pinned engine, and then a node's schema and the widget
# values we ship for it disagree with nothing saying so -- the user sees
# "Required input is missing" on a workflow that has never been edited.
#
# derive_catalog.py already refuses a branch for exactly this reason
# ("Recipes pin an immutable commit, never a branch"). This file was the one
# place still reading a moving target.
#
# Raising ENGINE_TAG means re-reading requirements.txt at the new tag and
# moving this with it; tests/unit/test_repo_layout.py fails until you do.
TEMPLATES_REF = "v0.11.48"
# The engine tag whose requirements.txt named that version. Same guard as
# engine.FLAGS_VERIFIED_AGAINST: raising ENGINE_TAG fails a test until someone
# has re-read the pin and moved this to match.
TEMPLATES_VERIFIED_FOR_ENGINE = "v0.34.0"
RAW = ("https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
       "{ref}/templates/{name}.json")

# The note's look. Matches the colours the official templates already use, so
# it does not read as a foreign object bolted onto the graph.
NOTE_COLOR, NOTE_BG = "#432", "#653"

# (template, output path, friendly title, what to tell the user)
WORKFLOWS: list[tuple[str, str, str, str]] = [
    (
        "image_z_image_turbo_int8",
        "image/01 Text to picture (Z-Image).json",
        "Text to picture",
        "Type what you want to see, then press **Run**.\n\n"
        "The picture appears in the box on the right and is saved into your "
        "`output` folder automatically.",
    ),
    (
        "image_sdxl_simple",
        "image/02 Text to picture (SDXL).json",
        "Text to picture, classic model",
        "The same idea as the first workflow, using the older SDXL model.\n\n"
        "It runs on smaller graphics cards and has a huge library of community "
        "add-ons, so it is worth keeping around.",
    ),
    (
        "image_qwen_image_edit_2511_int8",
        "image/03 Edit a picture.json",
        "Edit a picture",
        "Load a photo in the **Load Image** box, then describe the change you "
        "want in words -- \"make the car red\", \"remove the person on the left\".\n\n"
        "Press **Run**. The edited picture appears on the right.",
    ),
    (
        "video_wan2_2_5B_ti2v",
        "video/01 Text or picture to video.json",
        "Make a short video",
        "Describe the shot you want and press **Run**.\n\n"
        "To animate a picture instead, load one in the **Load Image** box first.\n\n"
        "A few seconds of video takes a few minutes to make. That is normal -- "
        "the progress bar is moving even when it looks stuck.",
    ),
    (
        "video_wan2_2_14B_t2v",
        "video/02 Make a video (big model).json",
        "Make a video with the big model",
        "The same idea as the first video workflow, using the much larger 14B "
        "model. It looks considerably better and it is a 38 GB download.\n\n"
        "**This will not fit on most graphics cards on its own.** Turn on "
        "**Stream model layers into the card** on the Toolshed launch screen "
        "before running it. The two speed-up LoRAs mean only four sampling "
        "steps, so streaming costs far less here than it would on a twenty-step "
        "job.\n\n"
        "Describe the shot you want and press **Run**.",
    ),
    (
        "audio_ace_step_1_5_checkpoint",
        "audio/01 Make music.json",
        "Make music",
        "There are two boxes that matter.\n\n"
        "**Tags** is the style: `pop, female vocal, upbeat, guitar`.\n"
        "**Lyrics** is what gets sung. Leave it empty for an instrumental.\n\n"
        "Press **Run**. A full song takes well under a minute.",
    ),
    (
        "3d_pixal3d_trellis2_image_to_model",
        "3d/01 Photo to 3D model.json",
        "Turn a photo into a 3D model",
        "Load one photo of a single object -- a mug, a shoe, a toy -- in the "
        "**Load Image** box, then press **Run**.\n\n"
        "A plain background works best. The finished model is saved as a `.glb` "
        "file in your `output` folder, which Blender and Windows 3D Viewer can "
        "both open.\n\n"
        "This is the slowest thing here: a few minutes per model.",
    ),
]

FOOTER = (
    "\n\n---\n\n"
    "Nothing here can break your setup. If you change something by mistake, "
    "close the tab without saving and open the workflow again.\n\n"
    "*Added by Toolshed. The graph itself is Comfy Org's official template.*"
)


def fetch(name: str, refresh: bool) -> dict:
    path = CACHE / TEMPLATES_REF / f"{name}.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    url = RAW.format(ref=TEMPLATES_REF, name=name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise SystemExit(f"cannot fetch {url}: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return json.loads(body)


def leftmost(nodes: list[dict]) -> tuple[float, float]:
    """Top-left of the graph, so the note sits beside it rather than on it."""
    positioned = [n for n in nodes if isinstance(n.get("pos"), list) and len(n["pos"]) >= 2]
    if not positioned:
        return (0.0, 0.0)
    return (min(n["pos"][0] for n in positioned), min(n["pos"][1] for n in positioned))


def add_start_here(doc: dict, title: str, body: str) -> dict:
    """Prepend a MarkdownNote. Virtual node: it never executes."""
    nodes = doc.setdefault("nodes", [])
    next_id = max([n.get("id", 0) for n in nodes] + [doc.get("last_node_id", 0)]) + 1
    x, y = leftmost(nodes)

    text = f"# START HERE\n\n## {title}\n\n{body}{FOOTER}"
    nodes.insert(0, {
        "id": next_id,
        "type": "MarkdownNote",
        "pos": [x - 520, y],
        "size": [480, 420],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": "START HERE",
        "properties": {},
        "widgets_values": [text],
        "widgets_values_named": {"text": text},
        "color": NOTE_COLOR,
        "bgcolor": NOTE_BG,
    })
    doc["last_node_id"] = next_id
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch the templates")
    args = ap.parse_args()

    provenance: list[tuple[str, str]] = []
    for template, rel, title, body in WORKFLOWS:
        doc = add_start_here(fetch(template, args.refresh), title, body)
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        provenance.append((rel, template))
        print(f"  wrote {rel}  (from {template})")

    notice = OUT / "NOTICE"
    lines = notice.read_text(encoding="utf-8").split("Adapted from")[0].rstrip()
    lines += (f"\n\nAdapted from Comfy-Org/workflow_templates at {TEMPLATES_REF},\n"
              f"the version ComfyUI's own requirements.txt pins at the engine tag we ship:\n\n")
    width = max(len(rel) for rel, _ in provenance)
    for rel, template in sorted(provenance):
        lines += f"  {rel.ljust(width)}  <- {template}\n"
    lines += (
        "\nRegenerate with tools/build_workflows.py. The graphs are Comfy Org's,\n"
        "unchanged apart from an added START HERE note, which never executes.\n"
    )
    notice.write_text(lines, encoding="utf-8")
    print(f"  updated {notice.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
