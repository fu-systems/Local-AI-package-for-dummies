# Video sizes, and how to replace the guesses with measurements

> **Two different video failures, and they are not the same bug.** If the log
> says `Memory access fault by GPU node` — usually right after
> `Requested to load WanVAE` — that is the AMD transfer-path fault, not this.
> It is handled by the safeguards in `toolshed/exec/engine.py:AMD_SAFEGUARDS`,
> and nothing on this page will help with it. This page is about the card
> genuinely not having room for the decode.


Every number in `catalog/video_presets.yaml` is a conservative estimate. This
is how to turn them into measurements, which is a couple of hours on one
machine and is worth doing before anyone else meets a card we guessed wrong
about.

## What goes wrong, so the fix is aimed at the right thing

A video job samples for minutes, reaches `VAEDecode`, turns every frame into
full-resolution pixels **in one allocation**, and dies there. The template we
ship asks for 1280×704×121 frames. On a 20 GB card that is over the line, and
the whole run is lost at the last step.

No engine flag prevents this. `--reserve-vram`, `--lowvram`, `--vram-headroom`
and dynamic VRAM all decide where model *weights* live and when they are
offloaded. The weights were never the problem — a single activation tensor was.
Only three things move that number:

* fewer frames (`length`), linear;
* smaller frames (`width`, `height`);
* decoding a slice at a time instead of all of it.

Easy mode does the third unconditionally for video, by swapping `VAEDecode` for
`VAEDecodeTiled` on the graph's way to the engine
(`toolshed/easy/knobs.py:use_tiled_decode`). The first two are what the presets
choose for you.

## Measure with tiling on

Easy mode always tiles, so the numbers that belong in the catalogue are the ones
taken **with tiling on**. Measuring the untiled peak measures a configuration
easy mode never sends, and would make every preset needlessly small.

The copy of the workflow in ComfyUI's sidebar is deliberately untiled — it is
Comfy Org's graph, unmodified, and that is the point of the "Open in ComfyUI"
path. So do not measure by opening the sidebar workflow and pressing Run.
Measure through easy mode.

## Taking the numbers

For each candidate size, on the card you are characterising:

1. Start easy mode, pick the video pack, choose the size, generate.
2. Watch VRAM during the run — `rocm-smi --showmemuse` on AMD, `nvidia-smi
   --query-gpu=memory.used --format=csv -l 1` on NVIDIA. The peak is what
   matters and it lands at the decode, near the end.
3. Record: card, total VRAM, width, height, length, peak VRAM, finished or died.
4. Repeat with a desktop session running normally. A card driving a compositor
   has less free than it has total, and the presets have to work on the machine
   as people actually use it, not on an idle headless one.

A size belongs in a tier when it finishes with room to spare — not when it
finishes exactly once. Two identical runs and a margin over the peak is the bar,
because the cost of being wrong is somebody losing ten minutes with nothing to
show.

## Constraints the numbers must respect

Read from ComfyUI v0.34.0, `comfy_extras/nodes_wan.py`:

```
Wan22ImageToVideoLatent   width    min 32,  step 32
                          height   min 32,  step 32
                          length   min 1,   step 4      -> 4n+1
```

The node's own default length is 49; the template we ship overrides it to 121.
`tests/unit/test_video_memory.py` asserts every preset sits on these grids, so
an off-grid value fails the suite rather than a user's first video.

Durations in the labels assume 24 fps, which is what the shipped workflow's
`CreateVideo` node uses.

## Updating the catalogue

Edit `catalog/video_presets.yaml`. Tiers are `vram_gb_min` floors, highest
first — though the loader sorts them anyway, so a tier added in the wrong place
cannot silently hand a 24 GB card the 8 GB sizes. The lowest tier must stay at
`vram_gb_min: 0`: it is also where a card whose memory could not be read lands,
deliberately, because detection failing is not evidence of a big card.

When the numbers are measured, delete the PROVISIONAL banner at the top of that
file. Leaving it there once it is untrue is worse than never having written it.
