"""Finding the few things a beginner actually wants to change.

Easy mode shows a prompt box and three or four controls. Behind it is a graph
of forty nodes it never shows anyone. Something has to connect "what you typed"
to "the text input of the node feeding the sampler's positive conditioning",
and this is it.

That connection is **worked out from the graph**, never written down as node
ids. Ids are assigned during conversion and change whenever a template is
updated upstream, so a table of them would be wrong at the next version bump
and wrong silently -- easy mode would still run, just ignoring the prompt and
returning the template's stock picture. Deriving it means a workflow that
changes shape either still works or reports honestly that it cannot be driven.

The rules come from the structure every one of these graphs shares: a sampler
has ``positive`` and ``negative`` conditioning inputs, conditioning is produced
by an encoder with a ``text`` widget, and latent size comes from a node with
``width`` and ``height``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# The full 64-bit range ComfyUI's seed input accepts, from nodes.py:
# {"default": 0, "min": 0, "max": 0xffffffffffffffff}.
SEED_MAX = 0xFFFFFFFFFFFFFFFF


# Inputs a settings screen must never offer. Editing them would either break
# the graph or waste the user's time.
NEVER_OFFER = frozenset({
    # The editor's synthetic seed companion; not a backend input at all.
    "control_after_generate",
    # Model filenames. They are chosen by the pack that was installed, and the
    # only other values the engine would accept are other packs' files.
    "ckpt_name", "unet_name", "vae_name", "clip_name", "lora_name",
    "model_name", "bg_removal_name", "style_model_name", "control_net_name",
    "upscale_model_name", "clip_vision_name", "audio_encoder_name",
})


@dataclass(frozen=True)
class Control:
    """One thing the user may change, described well enough to draw a widget.

    Everything here comes from the engine's own /object_info -- the type, the
    bounds, the list of choices -- so a node nobody here has heard of still
    gets a sensible control rather than a text box and a guess.
    """

    node_id: str
    input_name: str
    node_title: str
    kind: str                          # int | float | bool | choice | text | string
    value: Any = None
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    tooltip: str = ""
    role: str = ""                     # prompt | negative | width | ... | ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.node_id, self.input_name)

    @property
    def label(self) -> str:
        return self.input_name.replace("_", " ")


@dataclass(frozen=True)
class Target:
    """One input on one node that easy mode may write to."""

    node_id: str
    input_name: str
    current: Any = None


@dataclass
class Knobs:
    """What this particular workflow lets a person change."""

    positive: list[Target] = field(default_factory=list)
    negative: list[Target] = field(default_factory=list)
    seeds: list[Target] = field(default_factory=list)
    width: list[Target] = field(default_factory=list)
    height: list[Target] = field(default_factory=list)
    steps: list[Target] = field(default_factory=list)
    images: list[Target] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)

    @property
    def advanced(self) -> list[Control]:
        """Everything that does not already have its own control up top.

        Showing the prompt twice -- once in the big box and once in a list of
        forty inputs -- invites someone to set it in both places and get
        whichever the code happens to write last.
        """
        return [c for c in self.controls if not c.role]

    @property
    def takes_text(self) -> bool:
        return bool(self.positive)

    @property
    def takes_picture(self) -> bool:
        return bool(self.images)

    @property
    def has_size(self) -> bool:
        return bool(self.width and self.height)

    @property
    def is_drivable(self) -> bool:
        """Can easy mode do anything useful with this at all?

        A workflow with neither a prompt nor an image to feed it is one we can
        only run unchanged, which is not easy mode -- it is a button that makes
        the same thing every time.
        """
        return self.takes_text or self.takes_picture

    @property
    def current_size(self) -> tuple[int, int] | None:
        if not self.has_size:
            return None
        width, height = self.width[0].current, self.height[0].current
        if isinstance(width, int) and isinstance(height, int):
            return width, height
        return None


def _ref(value: Any) -> str | None:
    """The node id a connected input points at, if it is connected."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value[0]
    return None


def _text_source(prompt: dict[str, dict], start: str, seen: frozenset[str]) -> Target | None:
    """Walk back from a conditioning input to the box the words are typed in.

    Encoders are not always one hop away: templates put a ConditioningZeroOut
    or a style node between the encoder and the sampler, and the Z-Image
    template does exactly that on its negative branch.
    """
    if start in seen:
        return None
    node = prompt.get(start)
    if node is None:
        return None

    inputs = node.get("inputs", {})
    for name in ("text", "prompt", "tags", "lyrics"):
        if isinstance(inputs.get(name), str):
            return Target(start, name, inputs[name])

    seen = seen | {start}
    for value in inputs.values():
        upstream = _ref(value)
        if upstream is None:
            continue
        found = _text_source(prompt, upstream, seen)
        if found is not None:
            return found
    return None


def analyse(prompt: dict[str, dict], specs: dict | None = None) -> Knobs:
    """Work out what can be driven in an already-converted graph.

    ``specs`` is the engine's /object_info, read by convert.specs_from_object_info.
    Without it the roles below are still found -- they are inferred from the
    values -- but the full control list is empty, because the type and bounds
    of an input are not knowable from its current value alone. A width of 1024
    is an int; whether it is a slider from 16 to 16384 in steps of 8, or a
    dropdown, only the engine can say.
    """
    knobs = Knobs()

    for node_id, node in prompt.items():
        inputs = node.get("inputs", {})

        if isinstance(inputs.get("seed"), int):
            knobs.seeds.append(Target(node_id, "seed", inputs["seed"]))
        # Some samplers call it noise_seed; nodes.py declares
        # control_after_generate on both.
        if isinstance(inputs.get("noise_seed"), int):
            knobs.seeds.append(Target(node_id, "noise_seed", inputs["noise_seed"]))

        if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
            knobs.width.append(Target(node_id, "width", inputs["width"]))
            knobs.height.append(Target(node_id, "height", inputs["height"]))

        if isinstance(inputs.get("steps"), int):
            knobs.steps.append(Target(node_id, "steps", inputs["steps"]))

        # An input the engine marks with image_upload takes a picture that has
        # been sent to it. Asked of the engine rather than matched on the class
        # name: LoadImage is not the only node with one, and a table of class
        # names here would go stale the first time a workflow used another.
        spec = (specs or {}).get(node.get("class_type"))
        for name, value in inputs.items():
            if isinstance(value, list):
                continue                    # driven by another node
            detail = spec.spec_for(name) if spec else None
            if detail is not None and detail.options.get("image_upload"):
                knobs.images.append(Target(node_id, name, value))
            elif spec is None and node.get("class_type") == "LoadImage" and name == "image":
                # No object_info to ask; fall back to the node everyone knows.
                knobs.images.append(Target(node_id, name, value))

        for slot, bucket in (("positive", knobs.positive), ("negative", knobs.negative)):
            upstream = _ref(inputs.get(slot))
            if upstream is None:
                continue
            found = _text_source(prompt, upstream, frozenset())
            if found is not None and found not in bucket:
                bucket.append(found)

    # A node reached through both branches -- one encoder feeding positive and
    # negative alike -- must not be rewritten by the negative box, or typing a
    # negative prompt would silently replace what the user asked for.
    positive_ids = {t.node_id for t in knobs.positive}
    knobs.negative = [t for t in knobs.negative if t.node_id not in positive_ids]

    if specs:
        knobs.controls = _controls(prompt, specs, knobs)
    return knobs


def _roles_by_key(knobs: Knobs) -> dict[tuple[str, str], str]:
    """Which inputs already have a friendly control of their own."""
    named = {
        "prompt": knobs.positive, "negative": knobs.negative,
        "width": knobs.width, "height": knobs.height,
        "steps": knobs.steps, "seed": knobs.seeds, "image": knobs.images,
    }
    return {(t.node_id, t.input_name): role
            for role, targets in named.items() for t in targets}


def _controls(prompt: dict[str, dict], specs: dict, knobs: Knobs) -> list[Control]:
    """Every input the user could reasonably change, in graph order."""
    roles = _roles_by_key(knobs)
    found: list[Control] = []

    for node_id, node in prompt.items():
        spec = specs.get(node.get("class_type"))
        if spec is None:
            continue
        title = (node.get("_meta") or {}).get("title") or node.get("class_type", "")
        inputs = node.get("inputs", {})

        for name in spec.inputs:
            if name in NEVER_OFFER:
                continue
            detail = spec.spec_for(name)
            if detail is None or not detail.is_widget:
                continue
            value = inputs.get(name)
            # A connected input is driven by another node; offering a box for
            # it would be offering a value the engine is going to ignore.
            if isinstance(value, list):
                continue
            options = detail.options or {}
            found.append(Control(
                node_id=node_id,
                input_name=name,
                node_title=title,
                kind=detail.kind,
                value=value if value is not None else options.get("default"),
                choices=detail.choices,
                minimum=options.get("min"),
                maximum=options.get("max"),
                step=options.get("step"),
                tooltip=str(options.get("tooltip") or ""),
                role=roles.get((node_id, name), ""),
            ))
    return found


@dataclass
class Settings:
    """What the person chose. Anything left as None is left as the template had it."""

    prompt: str | None = None
    negative: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None          # None means "pick a new one"
    image: str | None = None         # a filename already uploaded to the engine
    # Anything else the user changed, keyed by (node id, input name). This is
    # what makes every setting reachable rather than the six with names.
    overrides: dict[tuple[str, str], Any] = field(default_factory=dict)


def apply(prompt: dict[str, dict], knobs: Knobs, settings: Settings) -> dict[str, dict]:
    """Return a copy of the graph with the person's choices written into it.

    A copy, because the converted graph is cached per workflow and reused for
    every generation. Writing in place would make each run inherit the last
    one's settings, so clearing the prompt box would not clear the prompt.
    """
    out = {node_id: {**node, "inputs": dict(node.get("inputs", {}))}
           for node_id, node in prompt.items()}

    def write(targets: list[Target], value: Any) -> None:
        for target in targets:
            if target.node_id in out:
                out[target.node_id]["inputs"][target.input_name] = value

    if settings.prompt is not None:
        write(knobs.positive, settings.prompt)
    if settings.negative is not None:
        write(knobs.negative, settings.negative)
    if settings.width is not None:
        write(knobs.width, settings.width)
    if settings.height is not None:
        write(knobs.height, settings.height)
    if settings.steps is not None:
        write(knobs.steps, settings.steps)
    if settings.image is not None:
        write(knobs.images, settings.image)

    # Written after the named ones: an explicit edit in the settings list is
    # the user being specific, and should win over anything inferred.
    for (node_id, input_name), value in (settings.overrides or {}).items():
        if node_id in out:
            out[node_id]["inputs"][input_name] = value

    # Always write a seed. Leaving the template's means pressing the button
    # twice gives the identical picture, which reads as the button being
    # broken -- and it is the single most confusing thing about these tools.
    write(knobs.seeds, settings.seed if settings.seed is not None
          else random.randrange(0, SEED_MAX))
    return out


def titles(prompt: dict[str, dict]) -> dict[str, str]:
    """Node id to a readable name, for saying which step is running."""
    return {node_id: (node.get("_meta") or {}).get("title") or node.get("class_type", "")
            for node_id, node in prompt.items()}
