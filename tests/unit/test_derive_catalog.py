"""Tests for tools/derive_catalog.py.

These are pure-unit tests over fixture documents. They never touch the network,
so they run on any machine with no GPU and no upstream access -- which matters,
because this project is developed on hardware that cannot run most of what it
installs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import derive_catalog as dc  # noqa: E402


def model_entry(name: str, directory: str, url: str) -> dict:
    return {"name": name, "directory": directory, "url": url}


SUBGRAPH_DOC = {
    "nodes": [
        {"type": "MarkdownNote"},
        {"type": "SaveImage"},
        {"type": "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"},
    ],
    "definitions": {
        "subgraphs": [
            {
                "name": "Text to Image (Z-Image-Turbo)",
                "inputs": [{"name": "text"}, {"name": "width"}, {"name": "seed"}],
                "nodes": [
                    {
                        "type": "UNETLoader",
                        "properties": {
                            "models": [
                                model_entry(
                                    "z_image_turbo_int8_convrot.safetensors",
                                    "diffusion_models",
                                    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
                                    "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
                                )
                            ]
                        },
                    },
                    {"type": "ModelSamplingAuraFlow"},
                    {"type": "ConditioningZeroOut"},
                ],
            }
        ]
    },
}

FLAT_DOC = {
    "nodes": [
        {
            "type": "CheckpointLoaderSimple",
            "properties": {
                "models": [
                    model_entry(
                        "sd_xl_base_1.0.safetensors",
                        "checkpoints",
                        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/"
                        "resolve/main/sd_xl_base_1.0.safetensors",
                    )
                ]
            },
        },
        {"type": "KSampler"},
        {"type": "MarkdownNote"},
    ]
}


class TestShapeDetection:
    def test_subgraph_template_exposes_typed_inputs(self):
        tpl = dc.parse_template("t", SUBGRAPH_DOC, None)
        assert tpl.shape == "subgraph"
        assert tpl.subgraph_inputs == ["text", "width", "seed"]
        assert tpl.subgraph_name == "Text to Image (Z-Image-Turbo)"

    def test_flat_template_has_no_subgraph_inputs(self):
        tpl = dc.parse_template("t", FLAT_DOC, None)
        assert tpl.shape == "flat"
        assert tpl.subgraph_inputs == []

    def test_bind_mode_follows_shape(self):
        sub = dc.to_recipe(dc.parse_template("t", SUBGRAPH_DOC, None), "abc")
        flat = dc.to_recipe(dc.parse_template("t", FLAT_DOC, None), "abc")
        assert sub["workflows"][0]["bind_mode"] == "subgraph_inputs"
        assert flat["workflows"][0]["bind_mode"] == "node_ids"


class TestNodeClasses:
    def test_virtual_and_uuid_nodes_are_excluded(self):
        """Note/MarkdownNote never execute and a UUID is a subgraph reference,
        so none of them may appear in requires.node_classes -- a contract test
        would look for them in /object_info and fail."""
        tpl = dc.parse_template("t", SUBGRAPH_DOC, None)
        assert "MarkdownNote" not in tpl.node_classes
        assert not any(dc.UUID_RE.match(c) for c in tpl.node_classes)

    def test_nodes_inside_subgraphs_are_collected(self):
        """The regression that motivated deriving rather than hand-listing:
        ModelSamplingAuraFlow lives inside the subgraph and sets the sampling
        shift Z-Image needs. Omitting it degrades output with no error."""
        tpl = dc.parse_template("t", SUBGRAPH_DOC, None)
        assert {"UNETLoader", "ModelSamplingAuraFlow", "ConditioningZeroOut"} <= tpl.node_classes


class TestModelExtraction:
    def test_hugging_face_url_splits_into_repo_and_path(self):
        tpl = dc.parse_template("t", SUBGRAPH_DOC, None)
        (m,) = tpl.models
        assert m.repo == "Comfy-Org/z_image_turbo"
        assert m.path == "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"
        assert m.dest == "diffusion_models"
        assert m.problems == []

    def test_truncated_upstream_url_is_repaired_from_the_name(self):
        """Some upstream templates truncate `url` mid-filename. `name` is
        authoritative, so the path tail is rebuilt from it."""
        doc = {"nodes": [{"type": "VAELoader", "properties": {"models": [model_entry(
            "trellis_2_texture_vae_bf16.safetensors", "vae",
            "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/"
            "trellis_2_texture_vae_bf16.safetensor",   # note: truncated
        )]}}]}
        (m,) = dc.parse_template("t", doc, None).models
        assert m.path == "vae/trellis_2_texture_vae_bf16.safetensors"

    def test_duplicate_models_are_collapsed(self):
        entry = model_entry("ae.safetensors", "vae",
                            "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/ae.safetensors")
        doc = {"nodes": [
            {"type": "VAELoader", "properties": {"models": [entry]}},
            {"type": "VAELoader", "properties": {"models": [entry]}},
        ]}
        assert len(dc.parse_template("t", doc, None).models) == 1

    @pytest.mark.parametrize("filename", ["model.ckpt", "model.pt", "model.pth", "model.bin"])
    def test_pickle_formats_are_rejected(self, filename):
        """Pickles execute arbitrary code on load. No per-model exception."""
        doc = {"nodes": [{"type": "UpscaleModelLoader", "properties": {"models": [model_entry(
            filename, "upscale_models",
            f"https://huggingface.co/someone/repo/resolve/main/{filename}")]}}]}
        (m,) = dc.parse_template("t", doc, None).models
        assert any("disallowed extension" in p for p in m.problems)

    def test_unreviewed_destination_folder_is_flagged(self):
        doc = {"nodes": [{"type": "Loader", "properties": {"models": [model_entry(
            "x.safetensors", "some_new_folder",
            "https://huggingface.co/a/b/resolve/main/x.safetensors")]}}]}
        (m,) = dc.parse_template("t", doc, None).models
        assert any("unreviewed destination folder" in p for p in m.problems)

    def test_non_hugging_face_url_is_flagged(self):
        doc = {"nodes": [{"type": "Loader", "properties": {"models": [model_entry(
            "x.safetensors", "vae", "https://example.com/x.safetensors")]}}]}
        (m,) = dc.parse_template("t", doc, None).models
        assert any("not a Hugging Face resolve URL" in p for p in m.problems)


class TestRecipeEmission:
    def test_every_unprovable_fact_is_pending_not_guessed(self):
        recipe = dc.to_recipe(dc.parse_template("t", SUBGRAPH_DOC, None), "deadbeef")
        f = recipe["files"]["z_image_turbo_int8_convrot"]
        for field in ("sha256", "size_bytes", "gated", "licence", "revision"):
            assert f[field] == dc.PENDING, f"{field} must never be guessed"

    def test_derived_from_pins_the_template_commit(self):
        recipe = dc.to_recipe(dc.parse_template("t", SUBGRAPH_DOC, None), "deadbeef")
        assert recipe["workflows"][0]["derived_from"]["commit"] == "deadbeef"

    def test_custom_nodes_is_empty_by_construction(self):
        """v1 invariant: no recipe may install a custom node."""
        recipe = dc.to_recipe(dc.parse_template("t", FLAT_DOC, None), "abc")
        assert recipe["custom_nodes"] == []

    def test_slug_is_stable_and_yaml_safe(self):
        assert dc.slug("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors") == \
            "qwen_image_edit_2511_lightning_4steps_v1_0_bf16"
        assert dc.slug("wan2.2_ti2v_5B_fp16.safetensors") == "wan2_2_ti2v_5b_fp16"
