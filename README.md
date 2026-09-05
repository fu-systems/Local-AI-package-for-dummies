# Toolshed

**Local AI, set up for you.**

An installer that hand-holds a complete beginner through setting up local
generative AI on **Linux and Windows**. You download one thing, tick boxes for
what you want to make, and it fetches the right models, puts them in the right
places, and configures everything.

It then **injects ready-made workflows into ComfyUI**, so you are never staring
at an empty node graph wondering what plugs into what, and gives you a simple
Generate window if you would rather not look at a node graph at all.

None of this is new technology. It is a bundle and a lot of hand-holding. All
of the value is in removing the twenty ways a beginner fails before their first
picture.

> **Status: early development.** Nothing is installable yet. See
> [docs/PLAN.md](docs/PLAN.md) for the full architecture and roadmap.

## What it will set up

| You want to make | What it installs |
|---|---|
| Pictures | Z-Image Turbo, with SDXL as a small-card fallback |
| Edited pictures | Qwen Image Edit 2511 |
| Short videos | Wan 2.2 5B, text-to-video and picture-to-video |
| Music | ACE-Step 1.5 |
| 3D models | TRELLIS.2, from a single photo |

Everything runs on your own machine. Nothing you make is sent anywhere, and
there is no telemetry, no account and no crash reporting.

## Hardware

| | Supported |
|---|---|
| **NVIDIA**, Windows and Linux | Yes, GTX 16-series and newer |
| **AMD**, Linux (ROCm) | Yes, RDNA2 and newer |
| AMD on Windows | Not yet — AMD's own Windows support for this is still an early preview |
| Intel Arc | Not yet |
| No dedicated graphics card | No, and we say so rather than installing something unusably slow |

You need roughly 8 GB of graphics memory for most things and 6 GB for pictures
alone. Toolshed tells you in plain English what your specific card can and
cannot do **before** it downloads anything.

macOS is out of scope.

### Linux system libraries

Toolshed bundles its own Python and Qt, but a few graphics libraries are
driver-coupled and must come from your distribution:

    Ubuntu / Debian   sudo apt install libgl1 libegl1 libglib2.0-0 libxkbcommon0 libxcb-cursor0
    Fedora            sudo dnf install mesa-libGL mesa-libEGL glib2 libxkbcommon xcb-util-cursor
    Arch              sudo pacman -S libglvnd glib2 libxkbcommon xcb-util-cursor

`install.sh` checks for these and tells you what is missing. Ubuntu 22.04 is the
oldest supported release.

## How it relates to ComfyUI

Toolshed downloads [ComfyUI](https://github.com/comfyanonymous/ComfyUI) from
upstream at install time and runs it as a separate process, talking to it over
its local HTTP API. It does not embed, patch, link against, or redistribute
ComfyUI.

There is an **"Open in ComfyUI"** button on every screen. When you outgrow the
simple interface, the real thing is right there with your workflows already
loaded and annotated. That is the point: Toolshed is a set of training wheels,
not a walled garden.

## Licence

Our code is Apache-2.0. Workflow files under `workflows/` are MIT, adapted from
[Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)
— see [workflows/NOTICE](workflows/NOTICE). Model weights are downloaded from
their publishers under their own licences and are never redistributed here;
Toolshed shows you each licence before it downloads anything.
