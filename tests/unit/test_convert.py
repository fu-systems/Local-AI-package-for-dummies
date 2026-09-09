"""Converting a saved workflow into what /prompt actually accepts.

ComfyUI's editor does this in JavaScript and nothing else does it at all, so
easy mode has to. The rules are not obvious and getting one wrong produces a
graph the engine rejects, or worse, one it runs with the wrong numbers.

The object_info fixture below is transcribed from ComfyUI v0.34.0's own source,
with the file and class named for each entry. It is here to test the
*algorithm*: at runtime the real thing is fetched from the running engine, so
this module holds no table of node names that could rot.

The case worth stating plainly, because it is the one that silently corrupts a
generation rather than failing: the editor inserts an extra
``control_after_generate`` widget after any INT input whose options declare it,
which ``nodes.py`` does on KSampler's ``seed``. A KSampler therefore stores
seven widget values for six real inputs, and a naive positional mapping shifts
everything after the seed by one -- sending ``steps="randomize"`` and a cfg of
25. It would not crash. It would just be wrong.
"""

from __future__ import annotations

import pytest
from comfy_fixtures import OBJECT_INFO, WORKFLOWS
from comfy_fixtures import load_workflow as load

from toolshed.easy.convert import (
    CONTROL_SLOT,
    ConversionError,
    NodeSpec,
    specs_from_object_info,
    to_api,
)

SPECS = specs_from_object_info(OBJECT_INFO)


class TestReadingObjectInfo:
    def test_widget_order_includes_the_editors_extra_seed_widget(self):
        """The whole reason positional mapping is not a plain zip."""
        assert SPECS["KSampler"].widget_slots == (
            "seed", CONTROL_SLOT, "steps", "cfg", "sampler_name", "scheduler", "denoise")

    def test_link_only_inputs_are_not_widgets(self):
        slots = SPECS["KSampler"].widget_slots
        for wired in ("model", "positive", "negative", "latent_image"):
            assert wired not in slots

    def test_a_combo_is_a_widget(self):
        assert SPECS["CheckpointLoaderSimple"].widget_slots == ("ckpt_name",)

    def test_optional_inputs_come_after_required_ones(self):
        assert SPECS["CLIPLoader"].inputs == ("clip_name", "type", "device")

    def test_every_input_is_known_even_when_it_is_not_a_widget(self):
        assert "clip" in SPECS["CLIPTextEncode"].input_set


class TestAFlatWorkflow:
    """The SDXL template: nine nodes, no subgraph, two of them notes."""

    @pytest.fixture
    def prompt(self):
        return to_api(load("image/02 Text to picture (SDXL).json"), SPECS)

    def test_the_editors_furniture_is_left_behind(self, prompt):
        """MarkdownNote is frontend-only. The engine has no such class and
        would reject the whole prompt for containing it."""
        classes = [n["class_type"] for n in prompt.values()]
        assert "MarkdownNote" not in classes
        assert sorted(classes) == [
            "CLIPTextEncode", "CLIPTextEncode", "CheckpointLoaderSimple",
            "EmptyLatentImage", "KSampler", "SaveImage", "VAEDecode"]

    def test_the_sampler_gets_the_numbers_that_were_actually_set(self, prompt):
        """The template shows 25 steps and cfg 7. Off-by-one on the seed widget
        would send steps="randomize" and cfg 25 -- which runs, and is wrong."""
        sampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
        assert sampler["inputs"]["steps"] == 25
        assert sampler["inputs"]["cfg"] == 7
        assert sampler["inputs"]["sampler_name"] == "dpmpp_2m"
        assert sampler["inputs"]["scheduler"] == "karras"
        assert sampler["inputs"]["denoise"] == 1
        assert isinstance(sampler["inputs"]["seed"], int)

    def test_the_synthetic_widget_is_never_sent(self, prompt):
        for node in prompt.values():
            assert "control_after_generate" not in node["inputs"]
            assert CONTROL_SLOT not in node["inputs"]

    def test_connections_become_node_references(self, prompt):
        sampler_id = next(i for i, n in prompt.items() if n["class_type"] == "KSampler")
        sampler = prompt[sampler_id]
        for wired in ("model", "positive", "negative", "latent_image"):
            ref = sampler["inputs"][wired]
            assert isinstance(ref, list) and len(ref) == 2
            assert ref[0] in prompt

        decode = next(n for n in prompt.values() if n["class_type"] == "VAEDecode")
        assert decode["inputs"]["samples"] == [sampler_id, 0]

    def test_the_checkpoint_name_survives(self, prompt):
        loader = next(n for n in prompt.values()
                      if n["class_type"] == "CheckpointLoaderSimple")
        assert loader["inputs"]["ckpt_name"].endswith(".safetensors")

    def test_the_size_is_carried_over(self, prompt):
        latent = next(n for n in prompt.values() if n["class_type"] == "EmptyLatentImage")
        assert latent["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}

    def test_nothing_points_at_a_node_that_is_not_there(self, prompt):
        for node in prompt.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in prompt, f"dangling reference to {value[0]}"


class TestASubgraphWorkflow:
    """The Z-Image template -- the default pack -- puts its whole pipeline
    inside a subgraph, leaving four visible nodes. Without flattening, easy
    mode would have nothing to run for the pack most people install."""

    @pytest.fixture
    def prompt(self):
        return to_api(load("image/01 Text to picture (Z-Image).json"), SPECS)

    def test_the_inside_of_the_subgraph_comes_out(self, prompt):
        classes = sorted(n["class_type"] for n in prompt.values())
        assert classes == [
            "CLIPLoader", "CLIPTextEncode", "ConditioningZeroOut",
            "EmptySD3LatentImage", "KSampler", "ModelSamplingAuraFlow",
            "SaveImage", "UNETLoader", "VAEDecode", "VAELoader"]

    def test_no_subgraph_placeholder_survives(self, prompt):
        """The instance node's type is a UUID. Sending that would be rejected."""
        for node in prompt.values():
            assert "-" not in node["class_type"] or node["class_type"].isidentifier()

    def test_the_prompt_text_survives_from_inside(self, prompt):
        """The instance node stores no value; the text lives on the inner node
        and reaches it because the boundary link resolves to nothing, so the
        node falls back to its own widget."""
        encoder = next(n for n in prompt.values() if n["class_type"] == "CLIPTextEncode")
        assert isinstance(encoder["inputs"]["text"], str)
        assert len(encoder["inputs"]["text"]) > 20

    def test_the_saver_outside_is_wired_to_the_decoder_inside(self, prompt):
        """The connection that crosses the subgraph boundary. If this is wrong
        the graph is two disconnected halves and nothing runs."""
        saver = next(n for n in prompt.values() if n["class_type"] == "SaveImage")
        ref = saver["inputs"]["images"]
        assert isinstance(ref, list)
        assert prompt[ref[0]]["class_type"] == "VAEDecode"

    def test_the_inner_sampler_kept_its_settings(self, prompt):
        """Inner nodes carry only positional widget values, so this is the
        control_after_generate case again with no named map to fall back on."""
        sampler = next(n for n in prompt.values() if n["class_type"] == "KSampler")
        assert sampler["inputs"]["steps"] == 8
        assert sampler["inputs"]["sampler_name"] == "res_multistep"
        assert sampler["inputs"]["scheduler"] == "simple"

    def test_nothing_points_at_a_node_that_is_not_there(self, prompt):
        for node in prompt.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in prompt, f"dangling reference to {value[0]}"


def permissive_specs(workflow: dict) -> dict[str, NodeSpec]:
    """Specs that accept every input name a workflow mentions.

    Used to exercise flattening and link resolution on the workflows whose node
    classes are not in the fixture. It proves the graph comes out structurally
    sound; it deliberately says nothing about widget values, which need the real
    /object_info.
    """
    specs: dict[str, NodeSpec] = {}
    scopes = [workflow] + ((workflow.get("definitions") or {}).get("subgraphs") or [])
    subgraph_ids = {s.get("id") for s in scopes[1:]}
    for scope in scopes:
        for node in scope.get("nodes", []):
            if node["type"] in subgraph_ids or node["type"] in {"MarkdownNote", "Note"}:
                continue
            names = tuple(e.get("name") for e in node.get("inputs", []) if e.get("name"))
            existing = specs.get(node["type"])
            merged = tuple(dict.fromkeys((existing.inputs if existing else ()) + names))
            specs[node["type"]] = NodeSpec(node["type"], merged, ())
    return specs


ALL_WORKFLOWS = sorted(p.relative_to(WORKFLOWS).as_posix() for p in WORKFLOWS.rglob("*.json"))


class TestEveryShippedWorkflow:
    def test_every_pack_that_claims_a_workflow_has_one(self):
        """A count was easier to write and said less. What matters is that
        PACK_FOLDERS and the files on disk agree: a pack naming a workflow that
        is not shipped is offered and then fails on click, and a workflow no
        pack names is dead weight in the bundle."""
        from toolshed.exec.inject import PACK_FOLDERS

        claimed = {rel for rels in PACK_FOLDERS.values() for rel in rels}
        shipped = set(ALL_WORKFLOWS)
        assert claimed - shipped == set(), f"packs name missing workflows: {claimed - shipped}"
        assert shipped - claimed == set(), f"workflows no pack names: {shipped - claimed}"

    @pytest.mark.parametrize("rel", ALL_WORKFLOWS)
    def test_it_converts_into_a_self_consistent_graph(self, rel):
        workflow = load(rel)
        prompt = to_api(workflow, permissive_specs(workflow))
        assert prompt
        for node_id, node in prompt.items():
            assert node["class_type"], node_id
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in prompt, \
                        f"{rel}: {node['class_type']}.{name} points at missing {value[0]}"

    @pytest.mark.parametrize("rel", ALL_WORKFLOWS)
    def test_no_editor_furniture_reaches_the_engine(self, rel):
        workflow = load(rel)
        prompt = to_api(workflow, permissive_specs(workflow))
        classes = {n["class_type"] for n in prompt.values()}
        assert not classes & {"MarkdownNote", "Note", "Reroute"}

    @pytest.mark.parametrize("rel", ALL_WORKFLOWS)
    def test_no_subgraph_instance_reaches_the_engine(self, rel):
        workflow = load(rel)
        ids = {s.get("id") for s in
               ((workflow.get("definitions") or {}).get("subgraphs") or [])}
        prompt = to_api(workflow, permissive_specs(workflow))
        assert not {n["class_type"] for n in prompt.values()} & ids


class TestNodesThatAreTurnedOff:
    """The video template ships with its LoadImage bypassed, so it makes video
    from text alone. A bypassed node must not be sent, and must not leave the
    node that used it pointing at nothing."""

    def test_a_bypassed_source_node_drops_the_input_it_fed(self):
        workflow = load("video/01 Text or picture to video.json")
        specs = permissive_specs(workflow)
        prompt = to_api(workflow, specs)

        assert "LoadImage" not in {n["class_type"] for n in prompt.values()}
        latent = next((n for n in prompt.values()
                       if n["class_type"] == "Wan22ImageToVideoLatent"), None)
        assert latent is not None
        assert "start_image" not in latent["inputs"], \
            "the bypassed image node was still referenced"

    def test_a_bypassed_node_passes_a_matching_type_through(self):
        """The general case: bypass forwards the first input of the same type
        as the output being asked for."""
        workflow = {
            "nodes": [
                {"id": 1, "type": "Source", "inputs": [],
                 "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}]},
                {"id": 2, "type": "Passthrough", "mode": 4,
                 "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                 "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}]},
                {"id": 3, "type": "Sink",
                 "inputs": [{"name": "image", "type": "IMAGE", "link": 2}],
                 "outputs": []},
            ],
            "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
        }
        specs = {
            "Source": NodeSpec("Source", (), ()),
            "Passthrough": NodeSpec("Passthrough", ("image",), ()),
            "Sink": NodeSpec("Sink", ("image",), ()),
        }
        prompt = to_api(workflow, specs)
        assert "Passthrough" not in {n["class_type"] for n in prompt.values()}
        sink = next(n for n in prompt.values() if n["class_type"] == "Sink")
        source_id = next(i for i, n in prompt.items() if n["class_type"] == "Source")
        assert sink["inputs"]["image"] == [source_id, 0]

    def test_a_muted_node_forwards_nothing(self):
        workflow = {
            "nodes": [
                {"id": 1, "type": "Source", "inputs": [],
                 "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}]},
                {"id": 2, "type": "Muted", "mode": 2,
                 "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                 "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}]},
                {"id": 3, "type": "Sink",
                 "inputs": [{"name": "image", "type": "IMAGE", "link": 2}],
                 "outputs": []},
            ],
            "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
        }
        specs = {
            "Source": NodeSpec("Source", (), ()),
            "Muted": NodeSpec("Muted", ("image",), ()),
            "Sink": NodeSpec("Sink", ("image",), ()),
        }
        prompt = to_api(workflow, specs)
        sink = next(n for n in prompt.values() if n["class_type"] == "Sink")
        assert "image" not in sink["inputs"]


class TestRefusals:
    def test_a_workflow_of_nothing_but_notes_says_so(self):
        workflow = {"nodes": [{"id": 1, "type": "MarkdownNote", "inputs": [],
                               "outputs": []}], "links": []}
        with pytest.raises(ConversionError):
            to_api(workflow, {})

    def test_a_loop_is_reported_rather_than_hung_on(self):
        workflow = {
            "nodes": [
                {"id": 1, "type": "A", "inputs": [{"name": "x", "type": "T", "link": 2}],
                 "outputs": [{"name": "T", "type": "T", "links": [1]}], "mode": 4},
                {"id": 2, "type": "B", "inputs": [{"name": "x", "type": "T", "link": 1}],
                 "outputs": [{"name": "T", "type": "T", "links": [2]}], "mode": 4},
                {"id": 3, "type": "Sink",
                 "inputs": [{"name": "x", "type": "T", "link": 1}], "outputs": []},
            ],
            "links": [[1, 1, 0, 3, 0, "T"], [2, 2, 0, 1, 0, "T"]],
        }
        specs = {"A": NodeSpec("A", ("x",), ()), "B": NodeSpec("B", ("x",), ()),
                 "Sink": NodeSpec("Sink", ("x",), ())}
        with pytest.raises(ConversionError, match="itself"):
            to_api(workflow, specs)
