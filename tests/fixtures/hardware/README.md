# Recorded hardware reports

The planner is a pure function of `(HardwareReport, selection, manifest)`, so
every hardware verdict is provable on a machine with no GPU. That is not a
nicety here: this project is developed on a single AMD Linux box, and these
fixtures are how the NVIDIA and Windows halves of the product get tested at all.

Each fixture is a `HardwareReport` captured from a real machine, plus the
verdict the planner must produce for it: the chosen torch index URL, the VRAM
tier, the set of offered recipes, and the refusal reason where applicable.

Coverage required before v0.3 (milestone M2 in docs/PLAN.md):

| Fixture | Why it exists |
|---|---|
| RTX 5070 (sm_120) | Blackwell needs cu130; cu126 must be hard-refused |
| RTX 4090 (sm_89, 24 GB) | Top tier, everything offered |
| RTX 4070 (sm_89, 12 GB) | The common case |
| RTX 3060 (sm_86, 8 GB) | Low tier; int8 variants |
| GTX 1660 (sm_75, 6 GB) | Pictures only; video and 3D hidden |
| GTX 1080 Ti (sm_61) | Legacy cu126 path, "end of the road" banner |
| below sm_5.0 | Friendly refusal |
| RX 9070 XT (gfx1201) | RDNA4, official ROCm |
| RX 7900 XTX (gfx1100) | RDNA3, the development machine |
| RX 6700 XT (gfx1031) | Unofficial, needs HSA_OVERRIDE consent; 3D blocked |
| Vega / Polaris | Below the ROCm floor, refusal |
| Intel Arc A770 | Refusal (D2) |
| AMD on Windows | Refusal (D2) |
| No discrete GPU | Refusal (D2) — never silently install CPU torch |
| Hybrid laptop iGPU + dGPU | iGPU ignored, VRAM never summed |
| Driver below floor, per CUDA line | Driver-update prompt, not a crash |
