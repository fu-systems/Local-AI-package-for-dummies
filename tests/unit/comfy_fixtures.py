"""A stand-in /object_info, transcribed from ComfyUI v0.34.0's own source.

Shared by the converter tests and the easy-mode tests so there is one copy to
keep honest. Each entry names the file and class it came from.

This exists to test *our* algorithms. At runtime the real thing is fetched from
the running engine, which is what keeps this project free of a table of node
names that would rot at the next ComfyUI release.
"""

from __future__ import annotations

import json
from pathlib import Path

from toolshed.easy.convert import specs_from_object_info

WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"


def load_workflow(rel: str) -> dict:
    return json.loads((WORKFLOWS / rel).read_text(encoding="utf-8"))


# Transcribed from ComfyUI v0.34.0. Each entry names where it came from.
# Combo inputs are written as a list of choices, exactly as /object_info
# serialises them, because that is what marks an input as a widget.
OBJECT_INFO = {
    # nodes.py, class KSampler
    "KSampler": {"input": {"required": {
        "model": ["MODEL", {}],
        "seed": ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                         "control_after_generate": True}],
        "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
        "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0,
                          "step": 0.1, "round": 0.01}],
        "sampler_name": [["euler", "dpmpp_2m", "res_multistep"], {}],
        "scheduler": [["normal", "karras", "simple"], {}],
        "positive": ["CONDITIONING", {}],
        "negative": ["CONDITIONING", {}],
        "latent_image": ["LATENT", {}],
        "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}],
    }}, "output": ["LATENT"]},
    # nodes.py, class CLIPTextEncode -- text comes before clip
    "CLIPTextEncode": {"input": {"required": {
        "text": ["STRING", {"multiline": True}],
        "clip": ["CLIP", {}],
    }}, "output": ["CONDITIONING"]},
    # nodes.py, class CheckpointLoaderSimple
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["sd_xl_base_1.0.safetensors"], {}],
    }}, "output": ["MODEL", "CLIP", "VAE"]},
    # nodes.py, class EmptyLatentImage
    "EmptyLatentImage": {"input": {"required": {
        "width": ["INT", {"default": 512, "min": 16, "max": 16384, "step": 8}],
        "height": ["INT", {"default": 512, "min": 16, "max": 16384, "step": 8}],
        "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
    }}, "output": ["LATENT"]},
    # nodes.py, class VAEDecode
    "VAEDecode": {"input": {"required": {
        "samples": ["LATENT", {}], "vae": ["VAE", {}],
    }}, "output": ["IMAGE"]},
    # nodes.py, class SaveImage -- images is a link, filename_prefix a widget
    "SaveImage": {"input": {"required": {
        "images": ["IMAGE", {}],
        "filename_prefix": ["STRING", {"default": "ComfyUI"}],
    }}, "output": []},
    # nodes.py, class CLIPLoader
    "CLIPLoader": {"input": {
        "required": {"clip_name": [["qwen.safetensors"], {}],
                     "type": [["stable_diffusion", "ace"], {}]},
        "optional": {"device": [["default", "cpu"], {}]},
    }, "output": ["CLIP"]},
    # nodes.py, class VAELoader
    "VAELoader": {"input": {"required": {
        "vae_name": [["ae.safetensors"], {}],
    }}, "output": ["VAE"]},
    # nodes.py, class UNETLoader
    "UNETLoader": {"input": {"required": {
        "unet_name": [["z_image.safetensors"], {}],
        "weight_dtype": [["default", "fp8_e4m3fn"], {}],
    }}, "output": ["MODEL"]},
    # nodes.py, class ConditioningZeroOut
    "ConditioningZeroOut": {"input": {"required": {
        "conditioning": ["CONDITIONING", {}],
    }}, "output": ["CONDITIONING"]},
    # comfy_extras/nodes_sd3.py, class EmptySD3LatentImage
    "EmptySD3LatentImage": {"input": {"required": {
        "width": ["INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}],
        "height": ["INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}],
        "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
    }}, "output": ["LATENT"]},
    # nodes.py, class LoadImage -- a combo of the files already in the input
    # directory, marked image_upload so the editor offers an upload button.
    "LoadImage": {"input": {"required": {
        "image": [["viking_wolf_rune_axe.png", "example.png"], {"image_upload": True}],
    }}, "output": ["IMAGE", "MASK"]},
    # comfy_extras/nodes_model_advanced.py, class ModelSamplingAuraFlow
    "ModelSamplingAuraFlow": {"input": {"required": {
        "model": ["MODEL", {}],
        "shift": ["FLOAT", {"default": 1.73, "min": 0.0, "max": 100.0, "step": 0.01}],
    }}, "output": ["MODEL"]},
}

SPECS = specs_from_object_info(OBJECT_INFO)
