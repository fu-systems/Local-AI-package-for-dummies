"""Tests for dropdowns whose choice decides which other widgets exist.

`SaveVideo.format`, `SaveImageAdvanced.format` and `RemeshMesh.sign_mode` are
all DynamicCombo inputs, so three of the six workflows we ship contain at least
one. The converter did not recognise the type, and dropped every one of their
values in both directions:

* the positional path never gave the combo a slot, so its value and its
  branch's values were never read;
* the named path filtered on the node's top-level input names, and
  "sign_mode.qef" is not one, so the dotted branch values were thrown away
  even when the template spelled them out.

The failure that causes is the worst shape there is. A missing `format` means
the engine never expands the branch, so nothing is reported missing and
validation PASSES -- then execute() is called without a required argument and
dies with a TypeError, after the picture or the entire video has been made.

The naming is read from comfy_api/latest/_io.py at v0.34.0:
DynamicCombo._expand_schema_for_dynamic looks up live_inputs[id] to choose the
branch, parse_class_inputs recurses with that id as prefix, and finalize_prefix
joins with "." -- so `format` selects and `format.bit_depth` carries.
"""

from __future__ import annotations

from toolshed.easy.convert import (
    DYNAMIC_COMBO_TYPE,
    _widget_inputs,
    specs_from_object_info,
)

# The shape /object_info reports for SaveImageAdvanced at v0.34.0: a String,
# then a DynamicCombo whose two branches carry different widgets.
OBJECT_INFO = {
    "SaveImageAdvanced": {
        "input": {
            "required": {
                "images": ["IMAGE", {}],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                "format": [DYNAMIC_COMBO_TYPE, {"options": [
                    {"key": "png", "inputs": {"required": {
                        "bit_depth": ["COMBO", {"options": ["8-bit", "16-bit"]}],
                        "input_color_space": ["COMBO", {"options": ["sRGB", "linear"]}],
                    }}},
                    {"key": "exr", "inputs": {"required": {
                        "exr_compression": ["COMBO", {"options": ["none", "zip"]}],
                    }}},
                ]}],
            }
        },
        "output": [],
    },
    "KSampler": {
        "input": {"required": {
            "seed": ["INT", {"control_after_generate": True}],
            "steps": ["INT", {}],
        }},
        "output": [],
    },
}


def spec():
    return specs_from_object_info(OBJECT_INFO)["SaveImageAdvanced"]


class TestTheSpec:
    def test_a_dynamic_combo_counts_as_a_widget(self):
        """It is typed into a box, so it takes a slot in the widget order.
        Treating it as a wired input is what shifted every later value."""
        assert "format" in spec().widget_slots

    def test_its_branches_are_read(self):
        assert spec().dynamic_combos["format"] == {
            "png": ("bit_depth", "input_color_space"),
            "exr": ("exr_compression",),
        }

    def test_its_choices_are_the_branch_keys(self):
        """So a settings screen can offer it like any other dropdown."""
        detail = spec().spec_for("format")
        assert detail.choices == ("png", "exr")
        assert detail.kind == "choice"

    def test_dotted_branch_names_are_accepted(self):
        assert spec().accepts("format.bit_depth")
        assert spec().accepts("format.exr_compression")

    def test_a_leaf_from_no_branch_is_not(self):
        """A stale value must not ride along into the prompt."""
        assert not spec().accepts("format.nonsense")

    def test_a_dotted_name_on_a_node_with_no_combo_is_not(self):
        assert not specs_from_object_info(OBJECT_INFO)["KSampler"].accepts("seed.foo")


class TestReadingPositionalValues:
    """Nodes inside a subgraph carry only the positional list."""

    def test_the_choice_and_its_branch_are_both_read(self):
        node = {"widgets_values": ["Qwen_Edit_2511", "png", "8-bit", "sRGB"]}
        assert _widget_inputs(node, spec()) == {
            "filename_prefix": "Qwen_Edit_2511",
            "format": "png",
            "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB",
        }

    def test_a_different_branch_consumes_a_different_number_of_slots(self):
        """Which is why a static slot list cannot express this: how far the
        combo reaches is not known until its value is read."""
        node = {"widgets_values": ["Shot", "exr", "zip"]}
        assert _widget_inputs(node, spec()) == {
            "filename_prefix": "Shot",
            "format": "exr",
            "format.exr_compression": "zip",
        }

    def test_a_truncated_list_does_not_raise(self):
        node = {"widgets_values": ["Shot", "png", "8-bit"]}
        got = _widget_inputs(node, spec())
        assert got["format.bit_depth"] == "8-bit"
        assert "format.input_color_space" not in got

    def test_an_unknown_branch_key_reads_no_nested_values(self):
        """Rather than misaligning everything after it."""
        node = {"widgets_values": ["Shot", "webp", "junk"]}
        got = _widget_inputs(node, spec())
        assert got["format"] == "webp"
        assert not [k for k in got if k.startswith("format.")]

    def test_control_after_generate_still_shifts_correctly(self):
        """The other synthetic slot must keep working alongside this."""
        ksampler = specs_from_object_info(OBJECT_INFO)["KSampler"]
        node = {"widgets_values": [42, "randomize", 20]}
        assert _widget_inputs(node, ksampler) == {"seed": 42, "steps": 20}


class TestReadingNamedValues:
    """Top-level nodes carry widgets_values_named, and the dotted keys in it
    were being filtered out -- this is the RemeshMesh failure."""

    def test_dotted_branch_values_survive(self):
        node = {"widgets_values_named": {
            "filename_prefix": "Qwen_Edit_2511",
            "format": "png",
            "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB",
        }}
        assert _widget_inputs(node, spec()) == {
            "filename_prefix": "Qwen_Edit_2511",
            "format": "png",
            "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB",
        }

    def test_a_key_the_engine_would_reject_is_still_dropped(self):
        node = {"widgets_values_named": {"format": "png", "made_up": 1,
                                         "format.nonsense": 2}}
        assert _widget_inputs(node, spec()) == {"format": "png"}
