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
from dataclasses import dataclass
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


class ConversionError(RuntimeError):
    """The workflow cannot be expressed as an API prompt."""


@dataclass(frozen=True)
class NodeSpec:
    """What the engine says a node class accepts."""

    name: str
    inputs: tuple[str, ...]          # every valid backend input, in order
    widget_slots: tuple[str, ...]    # the editor's widget order, with synthetics
    output_types: tuple[str, ...] = ()

    @property
    def input_set(self) -> frozenset[str]:
        return frozenset(self.inputs)


def _is_widget(type_: Any) -> bool:
    if isinstance(type_, list):
        return True                      # a combo: a list of choices
    return type_ in SCALAR_WIDGET_TYPES


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
        for input_name, definition in ordered:
            names.append(input_name)
            if not isinstance(definition, (list, tuple)) or not definition:
                continue
            type_ = definition[0]
            options = definition[1] if len(definition) > 1 and isinstance(
                definition[1], dict) else {}
            if _is_widget(type_):
                slots.append(input_name)
                if options.get("control_after_generate"):
                    slots.append(CONTROL_SLOT)

        specs[class_name] = NodeSpec(
            name=class_name,
            inputs=tuple(names),
            widget_slots=tuple(slots),
            output_types=tuple(info.get("output") or ()),
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
    numbering = {uid: str(i + 1) for i, uid in enumerate(flat.placed)}

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
                    # Points at something dropped; leave the input unset and
                    # let the engine say so, rather than sending a dangling id.
                    inputs.pop(name, None)
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
        return {k: v for k, v in named.items() if k in spec.input_set}

    values = node.get("widgets_values")
    if not isinstance(values, list):
        return {}

    out: dict[str, Any] = {}
    for slot_name, value in zip(spec.widget_slots, values, strict=False):
        if slot_name == CONTROL_SLOT or slot_name not in spec.input_set:
            continue
        out[slot_name] = value
    return out


def load_workflow(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_nodes(prompt: dict[str, dict], class_type: str) -> list[str]:
    """Ids of every node of a class, in graph order."""
    return [nid for nid, node in prompt.items() if node.get("class_type") == class_type]
