"""Turning a saved ComfyUI workflow into something ``/prompt`` will accept.

ComfyUI has two graph formats. The **workflow** format is what the editor saves
and what we inject into the sidebar: nodes with positions, links as a separate
list, widget values, subgraphs. The **API** format is what ``POST /prompt``
takes: a flat dictionary of ``{node id: {class_type, inputs}}`` with every
connection written inline. Only the editor's JavaScript converts between them,
so driving a saved workflow from outside the browser means doing it here.

Nothing in this file is guessed. The rules below were each read off real data
-- the templates we ship -- or out of ComfyUI v0.34.0's own source:

* Newer templates carry ``widgets_values_named``, a name-to-value map, which
  makes top-level nodes unambiguous. Nodes *inside* a subgraph carry only the
  positional ``widgets_values``, so both are handled.
* Positional mapping cannot simply be zipped against the backend's inputs. The
  editor inserts an extra ``control_after_generate`` widget after any INT input
  whose options say so -- ``nodes.py`` declares it on KSampler's ``seed`` --
  which is why a KSampler stores seven widget values for six real inputs.
* Which inputs exist, and in what order, comes from the running engine's
  ``/object_info``. That is the authority; this module holds no table of node
  names. A class the engine does not report is a frontend-only node, like
  ``MarkdownNote``, and is dropped rather than sent.
* A subgraph's boundary nodes are ``-10`` (its inputs) and ``-20`` (its
  outputs); the ids come from the ``inputNode``/``outputNode`` keys in the
  templates that use them.

The GPL boundary holds here as everywhere: this reads JSON and produces JSON.
No part of ComfyUI is imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Boundary node ids inside a subgraph definition, from the templates' own
# inputNode/outputNode entries.
INPUT_BOUNDARY = -10
OUTPUT_BOUNDARY = -20

# node["mode"] in the workflow format.
MODE_MUTED = 2
MODE_BYPASSED = 4

# The synthetic widget the editor inserts after a seed. It is not a backend
# input and must never be sent.
CONTROL_SLOT = "__control_after_generate__"

# Everything typed into a box rather than arriving down a wire. A combo box is
# expressed as a list of choices instead of a type name.
SCALAR_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# A dropdown whose choice decides which further widgets exist. SaveVideo's
# `format`, SaveImageAdvanced's `format` and RemeshMesh's `sign_mode` are all
# one of these, so three of the six workflows we ship contain at least one.
#
# The engine wants the choice under its own name and each of the chosen
# branch's widgets under a DOTTED name. Read from comfy_api/latest/_io.py at
# v0.34.0: DynamicCombo._expand_schema_for_dynamic looks up `live_inputs[id]`
# to pick the branch, then parse_class_inputs recurses with that id as the
# prefix, and finalize_prefix joins with "." -- so `format` selects, and
# `format.bit_depth` is where the branch value has to be.
#
# Getting this wrong is silent in the worst way. Validation passes, because a
# missing `format` means the branch is never expanded and so nothing is ever
# reported missing; the node then reaches execute() without a required
# argument and dies with a TypeError -- after the picture or the whole video
# has already been made.
DYNAMIC_COMBO_TYPE = "COMFY_DYNAMICCOMBO_V3"

# The tag whose _io.py the naming above was read from. Same guard as the
# engine flags: a rename upstream must not fail quietly.
DYNAMIC_COMBO_VERIFIED_AGAINST = "v0.34.0"


class ConversionError(RuntimeError):
    """The workflow cannot be expressed as an API prompt."""


@dataclass(frozen=True)
class InputSpec:
    """One input, as the engine describes it.

    The options are kept, not just the type. They are what lets a settings
    screen show a slider with the right bounds, a dropdown with the real
    choices and a multi-line box for a prompt -- all of it from the engine's
    own description, so a node we have never heard of still gets sensible
    controls.
    """

    name: str
    type: Any                          # "INT", "STRING", ... or a list of choices
    options: dict = field(default_factory=dict)

    @property
    def is_widget(self) -> bool:
        return _is_widget(self.type)

    @property
    def is_dynamic_combo(self) -> bool:
        return self.type == DYNAMIC_COMBO_TYPE

    @property
    def branches(self) -> dict[str, tuple[str, ...]]:
        """Option key -> the widgets that option brings with it."""
        return _branch_inputs(self.options) if self.is_dynamic_combo else {}

    @property
    def choices(self) -> tuple[str, ...]:
        if isinstance(self.type, list):
            return tuple(str(c) for c in self.type)
        if self.type == "COMBO":
            return tuple(str(c) for c in (self.options.get("options") or ()))
        if self.is_dynamic_combo:
            # The branch keys are the dropdown's choices, so a settings screen
            # can offer them like any other combo.
            return tuple(self.branches)
        return ()

    @property
    def kind(self) -> str:
        """What sort of control this wants."""
        if self.choices:
            return "choice"
        if self.type == "BOOLEAN":
            return "bool"
        if self.type == "FLOAT":
            return "float"
        if self.type == "INT":
            return "int"
        if self.type == "STRING":
            return "text" if self.options.get("multiline") else "string"
        return "wired"                 # arrives down a link, not typed in


@dataclass(frozen=True)
class NodeSpec:
    """What the engine says a node class accepts."""

    name: str
    inputs: tuple[str, ...]          # every valid backend input, in order
    widget_slots: tuple[str, ...]    # the editor's widget order, with synthetics
    output_types: tuple[str, ...] = ()
    specs: tuple[InputSpec, ...] = ()

    @property
    def input_set(self) -> frozenset[str]:
        return frozenset(self.inputs)

    @property
    def dynamic_combos(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every dynamic combo on this node, with its branches."""
        return {s.name: s.branches for s in self.specs if s.is_dynamic_combo}

    def accepts(self, key: str) -> bool:
        """Is this a name the engine will take?

        A dotted name is valid when its base is a dynamic combo here and the
        leaf belongs to one of that combo's branches. Checking the leaf too
        means a stale value from a branch nobody selected cannot ride along.
        """
        if key in self.input_set:
            return True
        base, dot, leaf = key.partition(".")
        if not dot:
            return False
        branches = self.dynamic_combos.get(base)
        return bool(branches) and any(leaf in names for names in branches.values())

    def spec_for(self, input_name: str) -> InputSpec | None:
        return next((s for s in self.specs if s.name == input_name), None)


def _is_widget(type_: Any) -> bool:
    if isinstance(type_, list):
        return True                      # a V1 combo: a list of choices
    # A V3 node (comfy_api.latest, io.Combo) reports the string "COMBO" and puts
    # its choices under options["options"]. TRELLIS.2, the mesh nodes and
    # SaveVideo are all V3, so treating this as a wired input dropped every one
    # of their dropdown values and broke the 3D and video graphs outright.
    return type_ == "COMBO" or type_ == DYNAMIC_COMBO_TYPE or type_ in SCALAR_WIDGET_TYPES


def _branch_inputs(options: dict) -> dict[str, tuple[str, ...]]:
    """A dynamic combo's branches: option key -> its widget names, in order.

    Shape from /object_info, per DynamicCombo.Option.as_dict at the pinned tag:
    ``{"options": [{"key": "png", "inputs": {"required": {...}, "optional": {...}}}]}``
    Required before optional, matching the order the editor lays widgets out.
    """
    branches: dict[str, tuple[str, ...]] = {}
    for option in options.get("options") or ():
        if not isinstance(option, dict):
            continue
        key = option.get("key")
        inner = option.get("inputs")
        if key is None or not isinstance(inner, dict):
            continue
        names: list[str] = []
        for section in ("required", "optional"):
            block = inner.get(section)
            if isinstance(block, dict):
                names.extend(block)
        branches[str(key)] = tuple(names)
    return branches


def specs_from_object_info(doc: dict) -> dict[str, NodeSpec]:
    """Read ``/object_info`` into what the converter needs.

    Required inputs come before optional ones, matching the order the editor
    lays widgets out in, which is what makes positional mapping work at all.
    """
    specs: dict[str, NodeSpec] = {}
    for class_name, info in doc.items():
        block = info.get("input") or {}
        ordered: list[tuple[str, Any]] = []
        for section in ("required", "optional"):
            ordered.extend((block.get(section) or {}).items())

        names: list[str] = []
        slots: list[str] = []
        details: list[InputSpec] = []
        for input_name, definition in ordered:
            names.append(input_name)
            if not isinstance(definition, (list, tuple)) or not definition:
                details.append(InputSpec(input_name, None, {}))
                continue
            type_ = definition[0]
            options = definition[1] if len(definition) > 1 and isinstance(
                definition[1], dict) else {}
            details.append(InputSpec(input_name, type_, options))
            if _is_widget(type_):
                slots.append(input_name)
                if options.get("control_after_generate"):
                    slots.append(CONTROL_SLOT)

        specs[class_name] = NodeSpec(
            name=class_name,
            inputs=tuple(names),
            widget_slots=tuple(slots),
            output_types=tuple(info.get("output") or ()),
            specs=tuple(details),
        )
    return specs


@dataclass(frozen=True)
class Ref:
    """A connection: the flattened node it comes from, and which output."""

    uid: str
    slot: int


@dataclass
class _Link:
    origin_id: int
    origin_slot: int
    target_id: int
    target_slot: int


def _index_links(raw: Any) -> dict[int, _Link]:
    """Link lists come in two shapes and both appear in the shipped templates.

    Top level: ``[id, origin_id, origin_slot, target_id, target_slot, type]``.
    Inside a subgraph: an object with those as named keys.
    """
    out: dict[int, _Link] = {}
    for entry in raw or []:
        if isinstance(entry, dict):
            link_id = entry.get("id")
            link = _Link(entry.get("origin_id"), entry.get("origin_slot"),
                         entry.get("target_id"), entry.get("target_slot"))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 5:
            link_id = entry[0]
            link = _Link(entry[1], entry[2], entry[3], entry[4])
        else:
            continue
        if link_id is not None:
            out[link_id] = link
    return out


@dataclass
class _Placed:
    """A workflow node, plus enough context to resolve its inputs later."""

    node: dict
    links: dict[int, _Link]
    prefix: str
    external: dict[int, Any]


class _Flattener:
    """Inlines subgraphs so the result is one flat graph.

    Subgraph instances are replaced by their contents, with inner ids prefixed
    so two uses of the same subgraph cannot collide. Connections that crossed
    the boundary are rewritten to point at whatever is really on the other side,
    which is recorded as a redirect and followed once everything is placed --
    a subgraph whose output feeds another subgraph's input would otherwise
    depend on the order things happened to be visited in.
    """

    def __init__(self, specs: dict[str, NodeSpec], subgraphs: dict[str, dict]) -> None:
        self.specs = specs
        self.subgraphs = subgraphs
        self.placed: dict[str, _Placed] = {}
        self.redirects: dict[tuple[str, int], Ref | None] = {}
        self.skipped: dict[str, _Placed] = {}      # muted and bypassed

    # -- placing ------------------------------------------------------------

    def expand(self, scope: dict, prefix: str, external: dict[int, Any]) -> None:
        links = _index_links(scope.get("links"))
        for node in scope.get("nodes", []):
            uid = f"{prefix}{node['id']}"
            placed = _Placed(node=node, links=links, prefix=prefix, external=external)
            mode = node.get("mode", 0)

            if mode in (MODE_MUTED, MODE_BYPASSED):
                # Kept aside rather than discarded: a bypassed node still has to
                # pass a connection through it, and a muted one has to make
                # whatever depended on it drop that input.
                self.skipped[uid] = placed
                continue

            subgraph = self.subgraphs.get(node["type"])
            if subgraph is None:
                self.placed[uid] = placed
                continue

            self._inline(uid, node, subgraph, placed)

    def _inline(self, uid: str, node: dict, subgraph: dict, placed: _Placed) -> None:
        # What the instance's own inputs are wired to, in the parent scope.
        inner_external: dict[int, Any] = {}
        for slot, entry in enumerate(node.get("inputs", [])):
            inner_external[slot] = self._source(placed, entry)

        self.expand(subgraph, f"{uid}:", inner_external)

        # And what each of its outputs really comes from, inside.
        inner_links = _index_links(subgraph.get("links"))
        for out_slot in range(len(subgraph.get("outputs") or [])):
            source = next(
                (link for link in inner_links.values()
                 if link.target_id == OUTPUT_BOUNDARY and link.target_slot == out_slot),
                None)
            self.redirects[(uid, out_slot)] = (
                Ref(f"{uid}:{source.origin_id}", source.origin_slot) if source else None)

    def _source(self, placed: _Placed, entry: dict) -> Any:
        """What feeds one input: a Ref, a literal, or None for "use the widget".

        None is not a failure. A subgraph input left unconnected on the instance
        means the node inside falls back to its own stored widget value, which
        is exactly how the Z-Image template carries its prompt.
        """
        link_id = entry.get("link")
        if link_id is None:
            return None
        link = placed.links.get(link_id)
        if link is None:
            return None
        if link.origin_id == INPUT_BOUNDARY:
            return placed.external.get(link.origin_slot)
        return Ref(f"{placed.prefix}{link.origin_id}", link.origin_slot)

    # -- resolving ----------------------------------------------------------

    def resolve(self, value: Any, _seen: frozenset[tuple[str, int]] | None = None) -> Any:
        """Follow redirects and bypasses to a node that will actually run."""
        seen = _seen or frozenset()
        if not isinstance(value, Ref):
            return value

        key = (value.uid, value.slot)
        if key in seen:
            raise ConversionError(f"the workflow connects {value.uid} to itself")
        seen = seen | {key}

        if key in self.redirects:
            return self.resolve(self.redirects[key], seen)

        skipped = self.skipped.get(value.uid)
        if skipped is not None:
            return self.resolve(self._through(skipped, value.slot), seen)

        return value

    def _through(self, placed: _Placed, slot: int) -> Any:
        """What a bypassed node passes through on the given output.

        The editor matches by type: a bypassed node forwards the first input of
        the same type as the requested output. A node with nothing of that type
        -- a bypassed LoadImage, which is how the video template ships so that
        it makes video from text alone -- forwards nothing, and the input that
        wanted it is left unset.
        """
        if placed.node.get("mode") == MODE_MUTED:
            return None
        outputs = placed.node.get("outputs") or []
        if slot >= len(outputs):
            return None
        wanted = outputs[slot].get("type")
        for entry in placed.node.get("inputs", []):
            if entry.get("type") == wanted:
                return self._source(placed, entry)
        return None


def to_api(workflow: dict, specs: dict[str, NodeSpec]) -> dict[str, dict]:
    """Convert a saved workflow into an API prompt.

    ``specs`` comes from the running engine's ``/object_info``. Any class it
    does not know is a frontend-only node -- notes, reroutes, the editor's own
    furniture -- and is left out rather than sent to be rejected.
    """
    subgraphs = {
        sub["id"]: sub
        for sub in ((workflow.get("definitions") or {}).get("subgraphs") or [])
        if sub.get("id")
    }

    flat = _Flattener(specs, subgraphs)
    flat.expand(workflow, "", {})

    # Stable, simple ids. The uids carry colons from subgraph nesting and there
    # is no reason to make the engine's error messages harder to read.
    # Only nodes the engine knows get a number. A frontend-only node such as
    # the editor's PrimitiveNode is dropped below, so a link from it must not
    # resolve to an id the engine will never see.
    numbering = {uid: str(i + 1) for i, uid in enumerate(flat.placed)
                 if placed_spec(specs, flat.placed[uid]) is not None}

    prompt: dict[str, dict] = {}
    for uid, placed in flat.placed.items():
        spec = specs.get(placed.node["type"])
        if spec is None:
            continue                     # a node the engine does not have

        inputs = _widget_inputs(placed.node, spec)

        for entry in placed.node.get("inputs", []):
            name = entry.get("name")
            if name not in spec.input_set:
                continue
            resolved = flat.resolve(flat._source(placed, entry))
            if isinstance(resolved, Ref):
                target = numbering.get(resolved.uid)
                if target is None:
                    # The source is a node the engine does not have -- the
                    # editor's PrimitiveNode feeding a widget, typically. The
                    # editor writes that value into the target's own widget on
                    # save, so the right thing is to keep the widget value
                    # already in `inputs`, not to delete it. Deleting it turned
                    # "Make music" into required_input_missing on first press.
                    continue
                inputs[name] = [target, resolved.slot]
            elif resolved is not None:
                inputs[name] = resolved

        prompt[numbering[uid]] = {
            "class_type": placed.node["type"],
            "inputs": inputs,
            "_meta": {"title": placed.node.get("title") or placed.node["type"]},
        }

    if not prompt:
        raise ConversionError("the workflow has no nodes the engine can run")
    return prompt


def placed_spec(specs: dict[str, NodeSpec], placed: _Placed) -> NodeSpec | None:
    return specs.get(placed.node.get("type"))


def _widget_inputs(node: dict, spec: NodeSpec) -> dict[str, Any]:
    """The values typed into the node, by name.

    Prefers ``widgets_values_named`` where the template has it, because a name
    cannot be misaligned. Falls back to the positional list -- which is all a
    node inside a subgraph carries -- mapped against the editor's widget order,
    synthetic control_after_generate slots included so everything after a seed
    does not shift by one.
    """
    named = node.get("widgets_values_named")
    if isinstance(named, dict):
        # accepts(), not input_set: a dynamic combo's branch values arrive as
        # "format.bit_depth", which is not a top-level input name and so used
        # to be filtered out here -- taking the whole branch with it.
        return {k: v for k, v in named.items() if spec.accepts(k)}

    values = node.get("widgets_values")
    if not isinstance(values, list):
        return {}

    # Positional. A dynamic combo takes its own slot and is then followed by
    # the widgets of whichever branch it selected -- so how many slots it
    # consumes is not known until its value is read, and a static slot list
    # cannot express it. Walk instead.
    combos = spec.dynamic_combos
    out: dict[str, Any] = {}
    index = 0
    for slot_name in spec.widget_slots:
        if index >= len(values):
            break
        value = values[index]
        index += 1
        if slot_name == CONTROL_SLOT:
            continue
        if slot_name in combos:
            if slot_name in spec.input_set:
                out[slot_name] = value
            for nested in combos[slot_name].get(str(value), ()):
                if index >= len(values):
                    break
                out[f"{slot_name}.{nested}"] = values[index]
                index += 1
            continue
        if slot_name in spec.input_set:
            out[slot_name] = value
    return out


def load_workflow(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_nodes(prompt: dict[str, dict], class_type: str) -> list[str]:
    """Ids of every node of a class, in graph order."""
    return [nid for nid, node in prompt.items() if node.get("class_type") == class_type]
