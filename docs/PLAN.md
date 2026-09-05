# Toolshed — a hand-holding local AI installer for Linux and Windows

## 1. Context

`fu-systems/Local-AI-package-for-dummies` is empty apart from a one-paragraph README. The goal is a
downloadable program that sets up local generative AI for a complete beginner: they download one
thing, tick boxes for what they want to make, and the program fetches the right models, puts them in
the right places, and configures everything. It then **injects ready-made ComfyUI workflows** so a
novice never wires nodes, and offers an Automatic1111-style easy mode for people who never want to
see a node graph.

Nothing here is novel technology. The value is entirely in removing the twenty ways a beginner fails
before their first picture. macOS is out of scope.

### Decisions locked by the project owner

| # | Decision |
|---|---|
| D1 | v1 covers image, video, audio and 3D. 3D is **TRELLIS.2 only**. |
| D2 | NVIDIA CUDA on Windows and Linux, plus **AMD ROCm on Linux**. AMD-on-Windows, Intel and CPU-only get a friendly refusal, never a broken install. |
| D3 | The installer app is written in **Python**. |
| D4 | Easy mode is **built into our app**, driving a headless ComfyUI over its HTTP/WebSocket API. No third-party front-end. |
| D5 | **Six** ready-made workflows in v1, every one adapted from an official Comfy Org template. |
| D6 | One developer, no fixed deadline. Product name **Toolshed**. |
| D7 | Available test hardware is **one AMD card on Linux**. No NVIDIA box, no Windows box, no 24 GB card. |

D7 is the constraint that shapes the build order. The plan below builds and validates on Linux + AMD
first, because that is the only hardware that can actually run anything, and treats NVIDIA and
Windows as code written against fixtures and validated on rented cloud GPUs and by beta testers.
This is the reverse of the obvious order, and it is deliberate.

---

## 2. Ground truth verified during planning

These were checked against live sources on 2026-09-05 and several correct assumptions that a
reasonable person would make from memory. They are load-bearing.

**TRELLIS.2 needs no compiled dependencies.** As of ComfyUI v0.34.0, TRELLIS.2 and Pixal3D are
native in core (`comfy_extras/nodes_trellis2.py`). The upstream research repo's flash-attn, spconv,
nvdiffrast and kaolin nightmare does not apply. There is **one venv, one torch, four modalities** and
no prebuilt-wheel matrix. This is the single biggest de-risking fact in the project.

**Official templates are now subgraph-based, and both shapes are in play.** `image_z_image_turbo`
has three top-level nodes: a MarkdownNote, a SaveImage, and one node whose type is the literal UUID
`f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1`. The real pipeline lives in `definitions.subgraphs[0]` with
typed inputs `text, width, height, seed, steps, unet_name, clip_name, vae_name`. `image_sdxl_simple`,
`video_wan2_2_5B_ti2v`, `audio_ace_step_1_5_checkpoint` and the 3D template are still flat.
Consequence: **bind parameters to subgraph input names where a subgraph exists, node ids where it
does not.** Subgraph inputs are a stable, self-documenting, already-correct parameterisation surface.

**The template index is machine-readable and reachable.** `Comfy-Org/workflow_templates`
`templates/index.json` lists 617 templates with per-template `minComfyUIVersion` and a total `size`,
and each template JSON carries `properties.models[]` entries of `{name, url, directory}`. **The
catalogue is derived from this, not hand-written.** Hugging Face is only needed for byte sizes,
sha256 and the gated flag.

**Corrections this produced to the obvious design:**

- The low-VRAM Z-Image variant is `z_image_turbo_int8_convrot.safetensors` paired with a *different*
  text encoder, `qwen_3_4b_fp8_mixed.safetensors`. There is no fp8 variant. `int8_convrot` is Comfy
  Org's current quantisation convention across Z-Image, Qwen Image Edit and TRELLIS.2. **Variants
  must swap the whole file set, never reuse one encoder across tiers.**
- The 3D template's vision model is `dino_v3_L_naf_fp32.safetensors`, and it additionally requires
  `moge_2_vitl_normal_fp16.safetensors` (`geometry_estimation`) and `birefnet.safetensors`
  (`background_removal`). Its VAEs live in the `Comfy-Org/Pixal3D` repo, not the TRELLIS.2 repo.
  Only an int8 TRELLIS.2 exists upstream; **no bf16 variant is referenced anywhere, so v1 ships int8
  only.**
- `--preview-method auto` **can never select TAESD**. In `latent_preview.py::get_previewer`, `Auto`
  is rewritten to `Latent2RGB` before the TAESD branch is reached. There is no Z-Image entry in the
  TAESD decoder table at all, and SDXL would need `taesdxl_decoder`, not `taesd_decoder`. **Ship no
  TAESD decoder.** latent2rgb previews are what we actually get and they are adequate.
- ComfyUI's release cadence is roughly **one minor per week** (ten releases in the 64 days to
  v0.34.0), not monthly. Budget maintenance accordingly.
- ComfyUI has `--database-url`, which relocates the internal database that `--base-directory` does
  not move. Use it.

**Not verified, because huggingface.co was unreachable throughout:** every file size, every sha256,
and every `gated` flag. Template `size` totals give useful approximations (Z-Image int8 12.1 GB,
SDXL 7.0 GB, Wan 2.2 5B 18.1 GB, ACE-Step 10.7 GB, 3D 15.4 GB, Qwen Image Edit int8 31.0 GB) but
per-file data must be frozen in M0 before any catalogue is written.

---

## 3. Architecture

### Process model

| Process | Role |
|---|---|
| `toolshed` | PySide6 GUI. Hardware detection, catalogue, planner, executor, ComfyUI supervisor, Generate UI. |
| `toolshed-download` | Short-lived child running one download plan, NDJSON progress on stdout. Cancel = SIGKILL. |
| `comfyui` | Long-lived child on `127.0.0.1:<ephemeral port>`, headless, `--disable-all-custom-nodes`. |
| `uv` | Short-lived children for interpreter install, venv creation, pip. Always invoked by absolute path, never resolved from `PATH`. |

No daemon, no tray agent, no local database server. We never import ComfyUI, never place a `.py`
under `custom_nodes/`, and never bundle it — so no GPL-3.0 obligation attaches. A CI job enforces it.

### Modules, and why the boundaries are where they are

| Module | Job | Purity |
|---|---|---|
| `hw/` | Vendor, arch, VRAM, driver detection into a frozen `HardwareReport`. Never imports torch. | Pure |
| `catalog/` | Load, signature-verify and validate recipe YAML into typed objects. | Reads only |
| `planner/` | `(HardwareReport, selection, manifest) -> InstallPlan`. Every hardware verdict, VRAM tier, torch index and licence decision. | **Pure function** |
| `exec/` | The only module touching network, filesystem or subprocesses. | Impure by design |
| `engine/` | ComfyUI supervisor plus HTTP/WS client. | Impure |
| `ui/` | Qt only. Never spawns a subprocess itself. | — |

The planner being pure is what makes D2 provable on a machine with no GPU. Every hardware verdict
for cards we do not own is a unit test over a recorded fixture. Given D7, this is not a nicety, it is
how the NVIDIA half of the product gets tested at all.

**The planner takes the current manifest as an input from day one**, so "install images now, add
video on Wednesday" is an additive plan rather than a retrofit. This is an M1 design constraint.

### On-disk layout of an installation

App code is separate from data so uninstall is surgical. App lives in
`%LOCALAPPDATA%\Programs\Toolshed\` or `~/.local/share/toolshed/app/`. Data root is user-chosen,
defaulting to `C:\Toolshed` or `~/Toolshed`:

```
Toolshed/
├─ runtime/
│  ├─ uv/uv(.exe)          vendored uv binary
│  ├─ uv-cache/            UV_CACHE_DIR — MUST share a filesystem with the venv
│  ├─ python/3.12.x/       UV_PYTHON_INSTALL_DIR
│  └─ venv/                torch + ComfyUI requirements
├─ engine/comfyui/         pinned source — DISPOSABLE, replaced wholesale on update
├─ comfy/                  --base-directory : PERSISTENT ComfyUI-owned state
│  ├─ custom_nodes/        ships empty; survives every engine update
│  ├─ input/  temp/
│  └─ user/default/
│     ├─ comfy.settings.json
│     └─ workflows/Toolshed/    injected .json only
├─ models/                 --models-directory : never touched by updates
├─ output/                 --output-directory : image/ video/ audio/ 3d/
└─ state/
   ├─ manifest.json        every file we installed: path, sha256, size, source, recipe
   ├─ hardware.json        last HardwareReport plus measured throughput index
   ├─ comfyui.db           --database-url target
   ├─ catalog/  workflows/  presets/  downloads/  snapshots/  logs/
```

Two structural points carry the whole update and uninstall story. `engine/comfyui/` is disposable and
nothing the user cares about may live there. `comfy/custom_nodes/` sits under `--base-directory`, not
under the engine tree, so a user who graduates and installs a node pack keeps it across updates.

Paths in `state/manifest.json` are stored **relative to the data root**, so moving a 45 GB folder to
a bigger drive needs only the root updated.

### Tech stack

**PySide6-Essentials, Qt Widgets.** The decision is made on Linux, not Windows. By construction of D2
every Linux user owns a discrete GPU, and WebKitGTK's DMABUF renderer paints a blank window on the
NVIDIA proprietary driver unless an environment variable is set. Add the webkit2gtk 4.0 vs 4.1 vs 6.0
split across Ubuntu 22.04, 24.04 and Debian 13 and any system-webview shell becomes a per-distro
support matrix. Qt ships its own rendering stack and needs exactly one extra Linux package,
`libxcb-cursor0`. Essentials over full PySide6 saves roughly 400 MB installed by dropping QtWebEngine.
Widgets over QML because the entire UI is a wizard, a form, a thumbnail grid and a log view.

**PyInstaller `--onedir`**, never `--onefile`: onefile costs seconds of extraction on every launch and
triggers a full antivirus rescan each time, which a novice reads as "it's broken". Explicitly
`upx=False`, bootloader built from source so the binary does not share byte patterns with the malware
corpus built on the shipped prebuilt one, and null `sys.stdout`/`sys.stderr` shims at startup because
`--windowed` sets them to `None` and any stray `print()` raises.

Windows ships via **Inno Setup** with `PrivilegesRequired=lowest` (per-user, no UAC). Linux ships as
**`.tar.xz`** plus `install.sh` — not `.tar.zst`, because GNU tar shells out to a separate `zstd`
binary that is absent on minimal Ubuntu and Debian installs, and that failure lands before our app
has executed a single line. Built inside an `ubuntu:22.04` container, because the PySide6 wheel floors
us at glibc 2.34 and building on 24.04 silently produces a binary that dies on 22.04.

**Vendored uv, pinned CPython 3.12.** uv is a single static binary needing no system Python and no
admin rights. 3.12 is the only version inside every constraint at once. `UV_CACHE_DIR` must live on
the same filesystem as the venv or uv silently falls back to full copies.

**One download path we own**, in a killable child process: httpx with Range resume, `.part` files on
the destination filesystem, streaming sha256, atomic `os.replace`. `huggingface_hub` is a dependency
for **metadata only** — both of its transfer paths are documented broken on multi-tens-of-GB files.
Files land as real files in the top-level model folders, never a blob cache and never a symlink tree.

**Signature verification does not use the `minisign` PyPI package** — it is version 0.1.0 from 2020,
self-described as unaudited, and it would be the trust root for the entire catalogue. Verify ed25519
directly over `cryptography` or `PyNaCl` with golden fixtures for a valid signature, a tampered
payload, a wrong key and a truncated signature.

---

## 4. Repo layout

```
Local-AI-package-for-dummies/
├─ LICENSE                       Apache-2.0 (our code)
├─ NOTICE  THIRD-PARTY-NOTICES.md
├─ CONTRIBUTING.md               contains the GPL-boundary invariant
├─ docs/UPSTREAM.md              every ComfyUI behaviour we rely on + where it was verified
├─ pyproject.toml  uv.lock
├─ toolshed/
│  ├─ __main__.py                entrypoint, null-stream guard, single-instance lock
│  ├─ hw/                        detect.py nvidia.py amd.py windows_dxgi.py windows_registry.py verdict.py
│  ├─ catalog/                   schema.py loader.py resolve.py
│  ├─ planner/                   plan.py steps.py torchsel.py estimate.py     ← PURE
│  ├─ exec/                      runner.py download_worker.py download.py uvtool.py
│  │                             comfy_fetch.py inject.py manifest.py repair.py
│  ├─ engine/                    supervisor.py http.py ws.py outputs.py errors.py
│  ├─ ui/                        wizard/ generate/ manage/ widgets/
│  └─ util/                      paths.py diskspace.py secrets.py logging.py
├─ catalog/                      THE DATA — a new model is a change here only
│  ├─ bundles/2026.11.yaml       runtime pins, recipe ids, launch flags
│  ├─ recipes/*.yaml             one per checkbox
│  ├─ hardware/                  nvidia-sm-table.yaml amd-gfx-table.yaml
│  ├─ licences/*.md
│  └─ catalog.sig
├─ workflows/                    MIT — derived from Comfy-Org/workflow_templates (MIT)
│  ├─ image/ video/ audio/ 3d/   <name>.ui.json + .api.json + .binding.json
│  ├─ annotations/*.md           MarkdownNote bodies, reviewed as prose
│  ├─ NOTICE                     Comfy Org copyright + pinned commit + source template per file
│  └─ LICENSE
├─ tools/                        build-time only, never shipped
│  ├─ derive_catalog.py          template properties.models[] → recipe skeleton
│  ├─ freeze_manifest.py         HF paths-info → sha256 + size + revision
│  ├─ lint_catalog.py  check_workflows.py  contract_test.py
│  ├─ gpl_boundary_check.py  sign_catalog.py  gen_notice.py  vendor_uv.py
├─ packaging/windows/  packaging/linux/
├─ tests/                        unit/ fixtures/hardware/ contract/ golden/ e2e/
└─ .github/workflows/            ci build canary verify-catalog
```

`workflows/` is **MIT, not CC0**: the templates it derives from are MIT, copyright Comfy Org, and MIT
requires the notice be retained. Each workflow declares `derived_from: {repo, commit, template}` or
`derived_from: null`, and lint enforces it.

---

## 5. Recipe schema

Three layers. A **bundle** pins ComfyUI, torch, Python, uv and launch flags, and is what we test as a
unit. A **recipe** is exactly one checkbox in the wizard. **Tables** hold the GPU-to-wheel data so
hardware support is also a data change.

### Invariants, enforced by `tools/lint_catalog.py` and `catalog/schema.py`

1. Every downloadable carries `sha256` and `size_bytes`, frozen at release from HF `paths-info`
   (the LFS `oid` **is** the sha256). Integrity never depends on a live API at install time.
2. `revision` is always an immutable commit SHA, never `main`.
3. `dest` is a real `folder_paths` key validated against the pinned engine, and files land
   **top-level** in it — ComfyUI's missing-model detection only matches top-level filenames.
4. Host allowlist: `huggingface.co`, `cas-bridge.xethub.hf.co`, `github.com`. Extension allowlist:
   `.safetensors`, `.sft`, `.gguf` only. **No `.ckpt`/`.pt`/`.pth` pickles, no exceptions.**
5. `custom_nodes` must be `[]` for every v1 recipe. CI-enforced.
6. `default_checked: true` requires `gate in {none, notice}` **and** `gated == false` for every file
   in every variant. This is how "the happy path needs no Hugging Face account" becomes machine-checked.
7. No recipe may carry a `territory_excluded` key. Territorially restricted models are simply absent.
8. **sha256 is the only dedup key at execution time.** `dedupe_key` is an authoring-time lint hint
   that warns on same-key-different-hash (an authoring mistake) — it never drives behaviour.
9. `PENDING_FREEZE` is the single placeholder token, and lint fails any release build containing one.
10. `requires.node_classes` and `expect_class_type` are **generated** from the authored `api.json`
    by lint, never hand-listed, so they cannot drift.
11. Every `filename` and `dest` must match the source template's `properties.models[]` entry where
    one exists.

### Example — `catalog/recipes/image-fast.yaml` (abridged)

```yaml
schema_version: 1
id: image.zimage_turbo
name: "Make pictures — fast"
blurb: "Describe a picture, get one in a few seconds. The best place to start."
modality: image
default_checked: true
requires: {comfyui_min: "0.34.0", vendors: [nvidia, amd]}
licence: {id: apache-2.0, spdx: Apache-2.0, commercial_use: true, gate: none}
custom_nodes: []

variants:                       # a variant swaps the WHOLE file set, never one file
  - id: int8                    # upstream: image_z_image_turbo_int8 (12.1 GB total)
    label: "Balanced"
    when: {vram_gb_min: 8, vram_gb_max: 15.9}
    files: [zimage_int8, qwen3_4b_fp8, ae_vae]
  - id: bf16                    # upstream: image_z_image_turbo (20.8 GB total)
    label: "Best quality"
    when: {vram_gb_min: 16}
    files: [zimage_bf16, qwen3_4b_bf16, ae_vae]

files:
  zimage_int8:
    source: huggingface
    repo: "Comfy-Org/z_image_turbo"
    revision: PENDING_FREEZE
    path: "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"
    dest: diffusion_models
    filename: "z_image_turbo_int8_convrot.safetensors"
    sha256: PENDING_FREEZE
    size_bytes: PENDING_FREEZE
    gated: PENDING_FREEZE
    licence: apache-2.0
  qwen3_4b_fp8:
    repo: "Comfy-Org/z_image_turbo"
    path: "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"
    dest: text_encoders
    # ... paired with int8 ONLY; the bf16 variant uses qwen_3_4b.safetensors

workflows:
  - id: image.t2i.fast
    title: "Text to Image (fast)"
    inject_as: "01 Pictures/01 Text to picture (Z-Image).json"
    derived_from: {repo: Comfy-Org/workflow_templates, commit: PENDING_FREEZE,
                   template: image_z_image_turbo_int8}
    shape: subgraph                       # lint fails if this stops matching the file
    bind_mode: subgraph_inputs            # bind by NAME, not node id
    bind: {prompt: text, width: width, height: height, seed: seed, steps: steps}
    output_node: "9"
    output_kind: image

presets:
  - id: quick
    label: "Quick picture"
    default: true
    controls:
      - {bind: prompt, widget: textarea, rows: 4, label: "Describe your picture",
         placeholder: "a red fox asleep in autumn leaves, soft morning light"}
      - {bind: size, widget: size_picker, label: "Shape",
         options: [{label: Square, w: 1024, h: 1024, default: true},
                   {label: Portrait, w: 832, h: 1216}, {label: Landscape, w: 1216, h: 832}]}
      - {bind: seed, widget: seed, label: "Seed", default: random}
      - {bind: steps, widget: slider, label: "Effort", min: 4, max: 16, default: 8, advanced: true}
```

Note there is **no negative prompt control**. The official Z-Image subgraph has exactly one text
input; negative conditioning comes from a `ConditioningZeroOut` node, and at the model's CFG of 1.0 a
negative prompt would do nothing anyway. Shipping a visible control that does nothing is worse than
omitting it. Lint rejects a `negative` binding unless the graph really has a second text encode
feeding the sampler's negative input.

---

## 6. Install pipeline

Every step is a typed, idempotent, resumable object with a visible sub-step, a byte counter and a
timeout. Nothing requires a terminal, admin rights, git, a compiler, a CUDA toolkit or 7-Zip.

1. **Launch and catalogue verification.** Single-instance lock. Verify the catalogue signature
   against the embedded public key. **A floor catalogue ships inside the installer**, so a machine
   with no network still reaches the hardware verdict screen; the network fetch is a strictly
   optional upgrade guarded by `min_app_version` and a monotonic version.
2. **Preflight.** 64-bit process; on Windows `vcruntime140_1.dll` and the `LongPathsEnabled` state;
   on Linux `libGL.so.1`, `glib-2.0`, `libxcb-cursor0` and glibc ≥ 2.34, probed **by library, not by
   package manager**, so it works on Arch and Fedora, with only the remediation text naming packages;
   on AMD, `/dev/kfd` exists and the user is in `render` and `video`; free RAM.
3. **Hardware detection**, no network, never imports torch. Windows ladder: `nvml.dll` via ctypes,
   then DXGI `CreateDXGIFactory1`, then the registry `HardwareInformation.qwMemorySize`, then
   `nvidia-smi`. **Never `wmic.exe`** (removed by KB5067470) and **never
   `Win32_VideoController.AdapterRAM`** (uint32, saturates at 4 GB). Linux ladder: `/sys/class/drm/card*/device`,
   with `mem_info_vram_total` giving exact bytes for amdgpu, gfx from a PCI-ID table with `rocminfo`
   as fallback. iGPUs ignored when a discrete GPU exists; VRAM never summed. Re-checked every launch,
   so a driver update or GPU swap prompts re-provisioning instead of a kernel-image crash.
4. **Verdict gate (D2).** Unsupported hardware gets the friendly refusal here, **before any download.**
5. **Choose what to make**, then licence gates, then location.
6. **Location validation** refuses with a specific reason each: non-ASCII, spaces, any OneDrive or
   Dropbox root, `C:\Program Files`, a drive root, exFAT (4 GB per-file limit kills every checkpoint),
   UNC paths, and paths too deep for Windows. Then free space ≥ plan bytes × 1.15, refusing with
   exact numbers rather than starting a doomed download.
7. **Python runtime.** Extract vendored uv, verify checksum, `uv python install 3.12`, `uv venv`.
8. **Torch index selection is a table, not a heuristic** — `(vendor, compute capability or gfx,
   driver version, os)` maps to an index URL from `catalog/hardware/*.yaml`:

   | Condition | Index | Note |
   |---|---|---|
   | NVIDIA sm ≥ 7.5 | `download.pytorch.org/whl/cu130` | Turing through Blackwell |
   | NVIDIA sm 5.0–7.2 | `.../whl/cu126` | "end of the line" banner |
   | NVIDIA sm < 5.0 | refuse | |
   | AMD Linux, official gfx | `.../whl/rocm7.2` | wheels bundle userspace; no system ROCm needed |
   | AMD Linux gfx1031/1032 | `rocm7.2` + `HSA_OVERRIDE_GFX_VERSION=10.3.0` | explicit consent, 3D blocked |
   | AMD Windows, Intel, no dGPU | refuse (D2) | |

9. **Install torch alone, first, pinned.** `uv pip install torch==<v> torchvision==<v> torchaudio==<v>
   --index-url <chosen>`. **`--index-url`, never `--extra-index-url`** — with an extra index the
   resolver may legitimately prefer the PyPI wheel, and torch's Windows PyPI wheel is the **CPU**
   build. That single mistake is the most common cause of "Torch not compiled with CUDA enabled".
10. **Verify torch immediately.** Assert `torch.cuda.is_available()`, that the local version tag
    matches intent (`+cu130` or `+rocm`), and that the device arch is in `get_arch_list()`. Failing
    here costs ninety seconds, not three hours and 40 GB.
11. **ComfyUI**: download the pinned tag tarball from codeload, verify sha256, extract. No git needed.
12. **ComfyUI requirements**, a separate step against PyPI, run **with a constraints file re-pinning
    the torch family** to the exact installed local versions — ComfyUI's `requirements.txt` lists
    `torch`, `torchvision`, `torchaudio` and `torchsde` completely unpinned, and a PyPI resolve here
    is the one operation most likely to silently swap in a CPU build. **Re-run the full torch
    assertion after this step**, and abort before any model download if it fails.
13. **Create every directory** before ComfyUI runs; `--user-directory` and `--models-directory` must
    pre-exist or argparse aborts with a bare usage error.
14. **Model downloads.** Three concurrent files, no per-file chunking. Range resume; `.part` on the
    destination filesystem so `os.replace` is atomic rather than `EXDEV`; sha256 while writing;
    verify; replace. **A file under its final name is by definition complete and verified**, which
    deletes the entire corrupted-model bug class. Re-check free space every 256 MB. **Persist only
    the canonical `huggingface.co/{repo}/resolve/{sha}/{path}` URL and re-follow the redirect on
    every resume** — the CDN target is presigned and expires, which is exactly the overnight-resume
    case. Validate resumes with `If-Range` against the stored ETag.
15. **Workflow injection**, then **settings seed**.
16. **First launch and health check.** Flags: `--listen 127.0.0.1 --port <ephemeral>
    --disable-auto-launch --log-stdout --base-directory --models-directory --output-directory
    --database-url --preview-method auto --reserve-vram --disable-api-nodes --disable-all-custom-nodes`.
    On Windows, a Job Object with `KILL_ON_JOB_CLOSE` created **before** the spawn so an app crash
    reaps the GPU process; on Linux `start_new_session` plus `killpg`, both with a pidfile whose
    `create_time` is verified before any kill. Readiness is `GET /system_stats` returning 200, then
    assert the version, the VRAM, the torch build and that `system.argv` contains our flags.
    **Never `--lowvram`** — it is a documented no-op while DynamicVRAM is on, which is the 0.34.0
    default, and `--novram`/`--highvram` actively disable DynamicVRAM. The one exposed knob is a
    "leave VRAM for my desktop" slider mapping to `--reserve-vram`.
17. **Smoke test per installed modality** — one real generation at minimum settings. **The install is
    not "done" until an artefact exists for every ticked pack.** A failing pack disables itself with
    an honest message rather than shipping as a broken checkbox. On AMD this is the only way to know
    ROCm produced usable kernels. This run also records the machine's **throughput index** into
    `state/hardware.json`, which is what makes time estimates real.

---

## 7. Workflow injection and the six workflows

ComfyUI's Workflows sidebar is a raw directory listing of `<user-directory>/default/workflows/` over
the `/userdata` API — no index file is needed and subdirectories render as folders. Injection is a
plain copy of UI-format JSON into `comfy/user/default/workflows/Toolshed/`, with numeric prefixes
controlling order and the `Toolshed/` namespace making uninstall surgical.

We do **not** use the `comfyui-workflow-templates` package mechanism: it is Comfy Org's own channel
and any frontend upgrade would overwrite us. We also never rely on ComfyUI's built-in missing-model
Download button, which has multiple open 2026 regressions. We pre-download everything.

### The six, all adapted from official templates

| # | Workflow | Source template | Shape | Models | Approx |
|---|---|---|---|---|---|
| 1 | Text to picture | `image_z_image_turbo_int8` / `image_z_image_turbo` | subgraph | Z-Image Turbo int8 or bf16, paired Qwen3-4B encoder, `ae.safetensors` | 12.1 / 20.8 GB |
| 2 | Edit a picture | `image_qwen_image_edit_2511_int8` | subgraph | Qwen Image Edit 2511 int8, `qwen_2.5_vl_7b_fp8_scaled`, `qwen_image_vae`, a Lightning LoRA | 31.0 GB |
| 3 | Text to video | `video_wan2_2_5B_ti2v` | flat | `wan2.2_ti2v_5B_fp16`, `wan2.2_vae`, `umt5_xxl_fp8_e4m3fn_scaled` | 18.1 GB |
| 4 | Picture to video | same template, LoadImage entry | flat | shared with #3 | 0 extra |
| 5 | Make music | `audio_ace_step_1_5_checkpoint` | flat | `ace_step_1.5_turbo_aio` single file | 10.7 GB |
| 6 | Photo to 3D | `3d_pixal3d_trellis2_image_to_model` | flat, 66 nodes | `trellis_2_int8_convrot`, shape + texture VAEs, `dino_v3_L_naf_fp32`, `moge_2_vitl_normal_fp16`, `birefnet` | 15.4 GB |

`image_sdxl_simple` ships as a **low-VRAM fallback recipe**, not a headline workflow: one flat
8-node graph and one 7.0 GB checkpoint, and it is the only image option that fits a 6 GB card. It is
not default-checked.

Workflow #2 is the most expensive thing in the catalogue at 31 GB and needs a large card. It is
never default-checked and its size is shown prominently.

Two decisions to settle in M0 before any of this is authored:

- **Subgraph posture.** Bind to subgraph input names (`text`, `width`, `seed`, ...) rather than
  flattening the graph. That surface is stable, self-documenting and already exactly what the Generate
  UI wants, and it keeps us diffable against upstream. The alternative, unpacking subgraphs to
  restore node-id parity, makes every bundle bump a re-flatten. Add a lint rule that fails on an
  unexpected change in subgraph presence so a bump cannot silently change shape.
- **The Pixal3D branch.** The 3D template ships both pipelines behind a switch and pulls
  `pixal3d_int8_convrot.safetensors` as well as the TRELLIS.2 model. D1 says TRELLIS.2 only. Confirm
  whether MoGe feeds only the Pixal3D branch or the shared path; if removal is entangled, ship the
  template unmodified and describe the pack honestly. D1 is about the model, not the graph.

### Beginner annotation inside the graph

Official templates already use `Note` and `MarkdownNote` virtual nodes that never execute. We go
further, consistently across all six: a large **START HERE** note positioned left of the entry point;
small notes beside each bound input written in the second person; **consistent colour semantics**
(green for nodes you should touch, grey for plumbing, red-brown for loaders that must not be edited)
so a beginner learns "green means mine" once; a note beside every loader naming the exact model file
and folder; and node titles in English rather than class names. Annotation bodies live in
`workflows/annotations/*.md` and are reviewed as prose.

For subgraph templates the annotations sit at top level around the subgraph node, which is a real
limitation: opening one shows a single opaque box rather than a teachable graph. The mitigation is
that the top-level note explains what is inside and how to expand it.

---

## 8. Easy mode

One window. Left rail: Pictures, Video, Music, 3D, Library, Settings, with modality tabs appearing
only for installed packs. Every tab is the **same three-pane layout** — controls left, big preview
centre, results strip along the bottom — implemented as **one tab class driven by catalogue data**,
not four hand-written tabs. Adding a model later is one YAML file and one CI fixture with no UI code
change.

Labels are what the user wants, not what the node calls it: "Describe your picture", "Effort",
"Follow the description", with the real node name in a tooltip so the vocabulary stays learnable.
Advanced controls live under a collapsed disclosure. Presets are keyed **by binding name, never node
id**, so a workflow can be re-authored between releases without invalidating saved presets.

### API mechanics, roughly 500 lines

Connect to `ws://127.0.0.1:<port>/ws?clientId=<uuid4>` and send a `feature_flags` frame requesting
preview metadata. Mint `prompt_id` client-side and POST it with the API graph and `client_id`. Filter
every message on `data.prompt_id`. Progress comes from `progress_state`, never from counting
`executed` messages, which fire only for nodes returning a `ui` dict. The terminal condition is
`executing {node: null}`, which fires on success, error and interrupt alike and is sent **after**
history is written, so `/history/{id}` is immediately safe with no polling.

Output collection **never keys off `"images"`**: walk every key of every node output, collect every
dict carrying a `filename`, and additionally convert bare strings ending `.glb`/`.obj`/`.gltf` into
file references, because video publishes under `images`, audio under `audio` and 3D under `3d` or a
positional `result`.

Model dropdowns come from `GET /models/{folder}`, **except VAE**, which must come from
`/object_info/VAELoader` because that node synthesises names that are not files. Where we do read
`/object_info`, one tolerant parser handles both the V1 and V3 combo shapes — every pre-2025 snippet
handles only V1 and shows an empty model list on V3 nodes.

**No CORS flag, ever.** Without `--enable-cors-header` ComfyUI installs an origin-only middleware; a
native Python client sends no `Origin` header and is unaffected, whereas enabling the flag would open
the user's local server to any website they visit.

"Make 4" queues four separate prompts with different seeds rather than `batch_size=4`: incremental
results, individually cancellable, and no single large VRAM spike on the 8 GB cards our audience owns.

**Previews are latent2rgb**, not TAESD, for the reason in §2. Audio and 3D have no meaningful latent
preview at all, so their centre pane shows elapsed time and a per-node progress readout. Say that
plainly in the UI rather than implying visual parity across modalities.

**"Open in ComfyUI" is on every tab from day one**, opening the local server in the system browser
with the matching injected workflow named. That is the on-ramp from "for dummies" to "turbo nerd",
and it is the reason we inject workflows at all rather than generating graphs programmatically.

**Out of v1:** LoRA pickers, ControlNet, inpainting and masking, upscaling, node editing, batch runs,
multi-GPU selection, a 3D viewport. Each is a legitimate v1.x addition and each is how a beginner
tool becomes an intimidating one.

---

## 9. Novice UX

**The no-terminal rule.** No path through install, first generation, repair, update or uninstall
requires a command line. **Exactly two remediations can**, both because the OS demands it, both
detected in preflight before anything is downloaded, and both presented as a copyable one-liner with
a Copy button and an explanation of why: `usermod -a -G render,video` for AMD `/dev/kfd` permissions,
and installing `libxcb-cursor0`. This is an acknowledged crack in the core promise, and being honest
about it beats failing mysteriously after 40 GB.

**Wizard**: Welcome → Your computer → What do you want to make → Licences (only if needed) → Where
should it go → Ready → Setting up → Done. The install screen is a vertical checklist where every row
is a named step with its own sub-status and byte counter, because "install freezes at 70% with no
diagnosable state" is the single most common novice complaint across every comparable project. Every
long step has a timeout, and on timeout we say which step, what it was waiting for, and offer Retry,
Skip this pack, or Copy diagnostics. The Done screen shows a thumbnail of the image the smoke test
just generated on their machine.

**Verdicts in plain English**, with the jargon behind a "What does this mean?" expander:

- "Good news — your NVIDIA GeForce RTX 4070 (12 GB) will work well." Then per-modality specifics.
- "Your AMD Radeon RX 7900 XTX on Linux — supported. 3D on AMD is experimental; we'll test it during
  setup and tell you honestly."
- "Your GTX 1660 (6 GB) will work for pictures. Video and 3D need at least 8 GB, so we've left those
  out. You can add them later if you upgrade."
- "We can't set this up on Windows with an AMD graphics card yet... It works well on Linux, and we
  support that."
- "You need a dedicated graphics card." We **refuse** rather than silently installing CPU torch.

Every unsupported message ends with a "Tell us about your setup" link that pre-fills a GitHub issue
with the hardware report, turning a dead end into data.

**Time estimates are measured, never guessed.** A single scalar per workflow cannot span a GTX 1660
and a 4090. Recipe cost is stored as relative work units scaled by the machine's throughput index
from the smoke test. Until a real measurement exists, use qualitative language only ("a few
seconds", "a few minutes"); after it, show a range, never a point estimate. A beginner told "about 4
seconds" who waits ninety concludes the install is broken.

**Every failure** is classified and rendered with four fields: what happened, why, what to do now
(a button where possible), and Copy diagnostics. **No traceback ever appears in the primary view.**
An unclassified exception logs a distinct marker, and **unclassified-failure count is a metric driven
to zero across releases**. The error table maps at minimum: HF 401 vs 403 vs `gated: manual` (three
genuinely different messages — `GatedRepoError` subclasses `RepositoryNotFoundError`, so a naive
handler tells the user their model does not exist when the real answer is "click Agree"), an expired
download link (retried silently), `ENOSPC`, torch verification failure, engine start failure,
`value_not_in_list`, out-of-memory during sampling, and out-of-memory during 3D refinement.

**Three global recovery affordances**: Repair, Safe mode, Copy diagnostics. Diagnostics is a zip with
the hardware report, manifest, recent logs and `/system_stats`, with tokens and home paths redacted.

**Repair is two-stage.** A fast pass checks existence, size and mtime plus the torch and engine
assertions in seconds, which covers every error-table case that recommends Repair. "Check every file
thoroughly" is an explicit second step with a time estimate, because hashing 45 GB takes minutes and
a beginner watching an unexplained progress bar assumes it hung.

**Privacy** is stated in the UI: no analytics, no crash reporting, no install id. A panel lists every
host contacted. Note that `--disable-api-nodes` is always on, so the privacy panel states that as a
fact rather than offering a toggle that changes nothing.

---

## 10. Update, uninstall, coexistence

Three update tracks. The **app shell** checks GitHub Releases with a cached conditional request and
shows a non-modal banner; no binary auto-updater in v1. The **catalogue** is signed data fetched from
the release channel, so a moved URL or a changed licence is a data push, not a release cycle. The
**engine bundle** is transactional: install to `.new` trees, run the full smoke test, atomically
swap, keep rollback for one generation. **Models are never touched** — that is the whole reason they
live in a sibling directory behind `--models-directory`.

**Uninstall** offers two deliberately unbalanced options: "Remove Toolshed, keep my models and
creations" (prominent, default) and "Remove everything, including 31.4 GB of models" (quiet,
secondary, second confirmation, exact figures shown). Enforced invariants: delete only files that are
in the manifest **and** still hash-match, because anything else was modified or added by the user;
never `rm -rf` a tree; never touch a directory we merely referenced via `extra_model_paths.yaml`,
which is somebody else's 50 GB; `output/` is never deleted by either path. Leftovers are reported,
not force-deleted.

**Coexistence**: probe for existing ComfyUI, A1111, Forge, Stability Matrix and InvokeAI installs and
offer to reuse their models rather than re-downloading. Wire it by writing **our own**
`extra_model_paths.yaml` and passing `--extra-model-paths-config`; never edit the user's file in
place, because corrupting it breaks their other front-ends. Adopted directories are marked read-only
in the manifest and can never be deleted by us. Ports are ephemeral, so a ComfyUI already running on
8188 is a non-event the user never sees.

**Data root moved** gets its own screen: "We can't find your Toolshed folder at D:\Toolshed — did you
move it?" with Browse and hash validation, rather than crashing or offering a fresh 45 GB download.

---

## 11. Packaging, signing, CI

| Job | Runner | Purpose |
|---|---|---|
| `ci` | hosted, every PR | lint, unit, catalogue lint, contract tests, golden graphs |
| `gpl-boundary` | hosted, every PR | fails on any import of `comfy`/`nodes`/`folder_paths`, any of our `.py` under `custom_nodes/`, or ComfyUI in a PyInstaller spec |
| `build-windows` | `windows-latest` | onedir → Inno Setup → signed installer |
| `build-linux` | `ubuntu:22.04` container | onedir → `.tar.xz` + `install.sh` |
| `canary` | hosted, nightly | contract + golden against ComfyUI **master**, auto-filing an issue on drift |
| `verify-catalog` | hosted, weekly | re-reads gated flags, licence tags and sizes; blocks releases while open |

The canary must distinguish drift that breaks our pinned contract (file an issue) from upstream
adding things we do not use (log only). Against a roughly weekly-minor upstream, an undiscriminating
canary drowns a solo developer in noise.

**Windows signing is a scheduling risk, not a packaging task.** Azure Artifact Signing at about $10
a month is available to individual developers in the USA and Canada only; the EU and UK are
organisations only, and it requires a paid Azure subscription. Resolve entity type, jurisdiction and
eligibility in **week one**, not at packaging time. Fallbacks are a $400–900/year OV certificate or
SignPath Foundation, which displays "SignPath Foundation" as the publisher — actively bad for a
product whose job is reassuring a nervous beginner. If unresolved, ship the first build beta-only
with an explicit SmartScreen explainer on the download page.

Expect PyInstaller binaries to draw antivirus detections even when signed. Mitigate with onedir, no
UPX, a self-built bootloader, signing, and submitting every release to Microsoft's false-positive
portal before announcing it. Keep publisher identity stable forever, because SmartScreen reputation
attaches to it and resets when it changes.

Our code is **Apache-2.0**; `workflows/` is **MIT** with the Comfy Org notice retained.

---

## 12. Roadmap

One developer, no fixed deadline. Estimates are dev-weeks at a sustainable pace including tests.

| # | Milestone | dw | Cum. | Exit gate |
|---|---|---|---|---|
| **M0** | **De-risk on the hardware we have.** Stand up ComfyUI v0.34.0 via uv + ROCm on the AMD/Linux box by hand. Run all six official templates unmodified; record VRAM, wall time and whether each works on ROCm at all. **Settle TRELLIS.2 on ROCm** and the Pixal3D-branch question. Settle the subgraph binding posture. Freeze the catalogue from template `properties.models[]` plus HF `paths-info`; confirm the Comfy-Org repackages are ungated. Verify a 20 GB resumable download. **Start signing eligibility.** | 4 | 4 | Every catalogue assumption is measured, not guessed |
| **M1** | **Vertical slice: Linux + AMD + one picture, end to end.** Ugly Qt window but real: detect → verdict → uv → ROCm torch + assertion → ComfyUI tarball → downloads with sha256 in the child process → inject one workflow → supervised child → Generate tab with preview and cancel → a picture. Recipe schema, pure planner, downloader, manifest and API client are written **properly** here because everything else reuses them. Additive-install support designed in now. Contract tests and golden graphs land in CI. | 8 | 12 | Someone else with an AMD Linux box makes a picture from a dev build, no terminal |
| **M2** | **NVIDIA and Windows, written blind and validated deliberately.** Full detection both OSes, the torch index table with the whole fixture library, driver floors, every D2 refusal, all path validation. Validate on **rented cloud GPUs** (a few hours per milestone) and a **Windows VM with no GPU** for installer, paths, preflight and the refusal path. | 7 | 19 | Fixture suite green; one rented NVIDIA session produces a picture on Windows and on Linux |
| **M3** | **Packaging and first public artefact.** PyInstaller spec and AV mitigations, Inno Setup, Linux tarball and desktop entry, both build runners, release signing, the canary, `docs/UPSTREAM.md`. | 5 | 24 | **v0.1 — a signed installer a stranger downloads** |
| **M4** | **The remaining five workflows (D1, D5).** Generalise to packs, tiers and dedup; author the video, music, 3D and image-edit triples; per-modality Generate configs; the 3D self-test; licence gates; per-modality smoke tests. Image and audio can be authored during M2/M3 slack on the AMD box. | 8 | 32 | Four modalities smoke-tested on AMD; NVIDIA verified on rented hardware |
| **M5** | **Novice UX, recovery, lifecycle.** The real wizard and all verdict copy, presets, gallery, queue, the error table, diagnostics with redaction, two-stage Repair, Safe mode, transactional update with rollback, two-mode uninstall, adopt-existing-install, data-root-moved recovery. | 8 | 40 | An install can be broken five ways and repaired from the UI; a non-technical person installs unaided while observed |
| **M6** | **Test spine and chaos.** Network loss, disk full, corrupt file, killed process, expired presigned URL, AV quarantine; clean-VM end-to-end on Windows and minimal Ubuntu 22.04. | 4 | 44 | Reproducible, signed, red when upstream drifts |
| **M7** | **Beta on machines we cannot buy.** RTX 50-series, Pascal, RDNA4, hybrid-graphics laptops, non-English Windows, corporate TLS interception, OneDrive profiles. A pass rewriting every string that leaked jargon. | 6 | 50 | v0.95 |
| **M8** | **Hardening to 1.0.** Accessibility, docs, screenshots, licence and attribution review. | 4 | 54 | **v1.0** |

Roughly **54 dev-weeks**, so about **12 months** solo, with a signed public v0.1 at around month six.
Add 3–4 weeks if 3D on ROCm needs real debugging rather than probe-and-disable.

**Standing costs to plan for:** rented cloud GPU time for NVIDIA validation (small, but it must be
budgeted and scheduled per milestone); a bundle bump every 4–6 weeks at roughly 1–2 dev-weeks each
against a weekly-minor upstream; and a meaningful standing fraction of the developer's time on
support after 1.0, because a novice-facing product with 40 GB downloads generates tickets.

**If the schedule must be cut**, cut in this order: the image-edit workflow (31 GB, needs a big card)
→ adopt-existing-install → preset count → queue UI. Do **not** cut the recipe engine and pure planner
in M1: shipping the catalogue as hardcoded Python to save a few weeks turns every future model
addition into a release cycle. Do not cut AMD (D2) or 3D (D1); they are fixed constraints.

---

## 13. Verification

Seven layers, cheapest first. The first four need no GPU, which given D7 is what makes the NVIDIA
half of the product testable at all.

1. **Pure unit tests.** The planner is a pure function, so the whole D2 policy is provable without
   hardware. Committed fixtures cover at minimum: RTX 5070 (sm_120), 4090, 4070 12 GB, 3060 8 GB,
   GTX 1660, 1080 Ti, a card below sm_5.0, RX 9070 XT (gfx1201), 7900 XTX (gfx1100), 6700 XT
   (gfx1031), Vega, Intel Arc, AMD-on-Windows, no GPU, a dual-discrete laptop, and a
   driver-below-floor case per CUDA line. Each asserts the index URL, VRAM tier, offered recipes and
   refusal reason. Also unit-tested: path validation, the output walker, and the `/object_info`
   V1-vs-V3 parser.
2. **Catalogue lint.** Every invariant in §5, including that every filename and dest matches the
   source template, and that no `PENDING_FREEZE` survives on a release branch.
3. **Contract tests** against the pinned engine on CPU: every node class every recipe declares exists
   in `/object_info`; every CLI flag we pass is accepted; every `folder_paths` key we write to exists.
4. **Golden graphs.** POST every `api.json` to a CPU ComfyUI and assert **HTTP 200 with empty
   `node_errors`**. This requires **materialising zero-byte stub files** at each catalogue entry's
   filename in its dest folder first, generated from the catalogue so they cannot drift — ComfyUI
   validates combo widgets by filename enumeration, and validation runs before execution, so stubs
   make the check pass without weights or a GPU. A second assertion requires the stub set to equal
   the catalogue's file set exactly, so a workflow referencing a model no recipe installs fails the
   build. Without the stubs this layer is red on day one for every workflow and will get deleted.
5. **Real-hardware smoke.** The AMD/Linux box nightly, plus rented NVIDIA sessions per milestone. Each
   does a clean install and one real output per modality. This is the only thing that proves ROCm
   produced usable kernels and that the Windows Job Object reaps the GPU process.
6. **Canary against upstream master**, nightly, filing issues on contract-breaking drift only.
7. **Novice-path end-to-end and chaos**, pre-release, from clean Windows and **minimal** Ubuntu 22.04
   images — minimal, because a desktop image masks the missing-`zstd` and missing-`libGL` classes of
   failure. Assert no dialog contains "Traceback", every step reports progress within its timeout,
   and the default uninstall leaves models and outputs intact.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| **Every size, hash and gated flag is unverified.** Hugging Face was unreachable during research. | M0 derives the catalogue from template metadata and freezes the rest against the live Hub. Budget 1.5–2 dev-weeks, not half a week: the observed error rate suggests re-authoring, not just hash-filling. |
| **A Comfy-Org repackage turns out to be gated**, collapsing the no-account happy path. | Verified in M0. The token and terms-acceptance flow is built in v1 regardless. |
| **3D on ROCm is unverified by anyone, anywhere.** The only test machine is AMD. | M0 answers it first. The runtime self-test gates the checkbox per machine on every platform. If it fails everywhere, 3D ships labelled experimental — D1 forbids substituting another model, so the honest answer is a labelled experiment, never a silent substitution. |
| **NVIDIA and Windows are written without daily access to the hardware.** | Pure planner plus fixtures for policy, rented cloud GPUs per milestone for reality, a GPU-less Windows VM for the installer, and beta testers for the long tail. State the coverage limits in the README rather than implying more. |
| **TRELLIS.2 out-of-memories late**, during refinement rather than at load, so a VRAM check passes and the run dies minutes in. | Conservative 10 GB floor, resolution capped by detected VRAM, and a dedicated error-table row offering the smaller setting. |
| **Weekly upstream minors** mean a pinned engine drifts fast, undercutting the "Open in ComfyUI" graduation path. | 4–6 week bundle cadence, a discriminating canary, and the engine version stated plainly in the README. |
| **PyInstaller binaries draw antivirus detections**, and signing eligibility may not exist. | onedir, no UPX, self-built bootloader, FP-portal submission; signing resolved in week one with a documented fallback. |
| **The zero-custom-nodes position is load-bearing and fragile.** LoRAs, ControlNet and text-to-speech all break it. | Decided now: the first custom node gets its own release and its own isolation strategy. The Advanced toggle that enables nodes must also stop passing `--disable-all-custom-nodes`, or a graduating user installs a node that silently never loads. |
| **The 6–8 GB tier is thin.** | int8 variants are upstream's own low-VRAM route and sidestep the fp8-on-RDNA question entirely; SDXL covers 6 GB. Revisit if beta shows most users are on 8 GB. |
| **Deliberate v1 exclusions will draw complaints**: no text-to-speech, no FLUX, no LoRAs, no ControlNet, no upscaler, no macOS. | Each is a documented answer in the README with its reason. The small catalogue is what makes it possible for one person to test every combination offered. |

---

## 15. Open items to resolve during M0

1. Legal entity and jurisdiction for Windows code signing, plus a paid Azure subscription. Gates the
   first public build's trust story.
2. Whether TRELLIS.2 runs on ROCm at all. Nobody has published a result either way.
3. Whether the Pixal3D branch can be cleanly removed from the 3D template, or whether MoGe feeds the
   shared path.
4. Whether ComfyUI can export API format from a subgraph workflow, and whether "Unpack subgraph"
   round-trips. Determines the binding implementation before M1 writes the schema.
5. Gated status of every Comfy-Org repackage, and of `stabilityai/stable-diffusion-xl-base-1.0`.
6. Budget and cadence for rented NVIDIA validation sessions.

---

## 16. First implementation step

M0 is investigation on the AMD box and produces no application code. The first code to write is
`tools/derive_catalog.py`, which turns the official template index and each template's
`properties.models[]` into recipe skeletons, because everything downstream — the catalogue, the lint
rules, the golden-graph stubs — is derived from that ground truth rather than hand-authored.
