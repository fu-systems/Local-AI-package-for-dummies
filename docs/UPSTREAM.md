# Upstream behaviours we depend on

ComfyUI ships roughly one minor release a week. Every behaviour Toolshed relies
on is recorded here with where it was verified, so a version bump is a diff
review rather than an archaeology expedition. `.github/workflows/canary.yml`
runs the contract and golden-graph tests against ComfyUI `master` nightly and
opens an issue when something in this table stops being true.

Verified 2026-09-05 against ComfyUI v0.34.0 unless stated otherwise.

| Behaviour we rely on | Where verified | Breaks what if it changes |
|---|---|---|
| Workflows sidebar is a plain directory listing of `<user-directory>/default/workflows/`; no index file needed, subdirectories render as folders | `/userdata` API | Workflow injection entirely |
| `Note` / `MarkdownNote` are frontend-only virtual nodes that never execute | frontend node defs | Beginner annotations would reach `/prompt` and fail validation |
| Combo widgets validate against `folder_paths.get_filename_list(<key>)`, by filename, before execution | `execution.py` validation path | Golden-graph stub-file strategy (`tests/golden/`) |
| Missing-model detection only matches **top-level** filenames in a model folder | frontend missing-models dialog | Files nested in subfolders are reported missing forever |
| `--preview-method auto` is rewritten to `Latent2RGB` **before** the TAESD branch | `latent_preview.py::get_previewer` | Any plan to ship a TAESD decoder |
| Latent format `taesd_decoder_name` is undefined for Z-Image, ACE-Step and all TRELLIS.2 formats | `comfy/latent_formats.py` | Live preview expectations for those modalities |
| `--lowvram` is a no-op while DynamicVRAM is on (the 0.34.0 default); `--novram`/`--highvram` disable DynamicVRAM | `comfy/cli_args.py` help text | Low-VRAM handling; we expose `--reserve-vram` instead |
| `--database-url` relocates the internal database, which `--base-directory` does not | `comfy/cli_args.py` | DB lands in the disposable engine tree and dies on update |
| PyTorch names a wheel's local version after the index it is published under, **verbatim including dots**: `.../whl/rocm7.2` installs `2.14.0+rocm7.2`, `.../whl/cu130` installs `+cu130` | install log, AMD gfx1100 on Linux, 2026-09-06 | The post-install torch check; a mangled expectation rejects a working card |
| `--base-directory` sets base for models, custom_nodes, input, output, temp and user; `--models-directory` and `--output-directory` override those two alone, and `--models-directory` is rejected by `is_valid_directory` if it does not already exist | `comfy/cli_args.py`, `folder_paths.py` lines 15-72 | Launching: without the override the engine looks in `<base>/models` and every workflow reports missing models |
| `--disable-auto-launch` stops the engine opening a browser itself; `--log-stdout` moves its logging off stderr | `comfy/cli_args.py` | We open the browser only once `/system_stats` answers, and show one ordered log |
| `/system_stats` is a real route, so a 200 means the app is serving, not just that the socket is bound | `server.py` line 686 | The readiness check; anything weaker opens a browser on a connection-refused page |
| `/prompt` accepts **API-format** graphs, not the UI workflow format the templates ship | `server.py` line 1072 | Easy mode must build API graphs; injected templates are UI format |
| Templates carry `widgets_values_named` (name to value) on top-level nodes, but nodes *inside* a subgraph carry only the positional `widgets_values` | Comfy Org templates, `image_sdxl_simple` and `image_z_image_turbo` | Easy mode's widget mapping; both shapes must be handled |
| The editor inserts an extra `control_after_generate` widget after any INT input whose options declare it (`nodes.py` does, on KSampler's `seed`), so a KSampler stores 7 widget values for 6 inputs | `nodes.py` class KSampler | Positional widget mapping. Getting it wrong sends `steps="randomize"` and does not crash |
| A subgraph's boundary nodes are id `-10` (inputs) and `-20` (outputs); an unconnected boundary input means the inner node falls back to its own widget value | template `definitions.subgraphs[0]` | Flattening. The default pack, Z-Image, is entirely inside one |
| Websocket messages are `status`, `progress_state`, `executing`, `executed`, `execution_error`, `execution_cached`, `execution_interrupted`, `execution_success`; `/ws?clientId=` must match `/prompt`'s `client_id` | `server.py`, `comfy_execution/progress.py` | Easy mode's progress bar and completion detection |
| `POST /upload/image` is multipart with field `image` (plus optional `type`, `subfolder`, `overwrite`) and answers `{name, subfolder, type}`; without `overwrite` it compares hashes and reuses or renames | `server.py::image_upload` | Easy mode's starting picture; `name` (prefixed by `subfolder`) is what LoadImage wants |
| An input that accepts an upload is marked `image_upload: True` in its options, and `LoadImage.image` is a combo of the files already in the input directory | `nodes.py` class LoadImage | How easy mode finds the picture slot without a table of node names |
| The shipped 3D template stores its own sample filename (`viking_wolf_rune_axe.png`) in LoadImage | Comfy Org template `3d_pixal3d_trellis2_image_to_model` | Running it unchanged asks for a file nobody else has, so easy mode must require a picture rather than defaulting |
| Async weight offload (2 streams) and pinned host memory are both **on by default on AMD**, not just NVIDIA; `--disable-async-offload` and `--disable-pinned-memory` turn them off | `comfy/model_management.py` lines 1340-1353, 1581-1588 | Both move weights by DMA, the first suspects for a `Memory access fault by GPU node-N` on ROCm |
| ROCm counts the CPU as an HSA agent, so `torch.cuda.device_count()` is 2 on a one-GPU machine and ComfyUI lists `cuda:1` as the Ryzen | reported install log, gfx1100 | Never report the device count as a number of graphics cards, and never suggest `--cuda-device 1` from it |
| `--user-directory` and `--models-directory` are typed `is_valid_directory` and must pre-exist | `comfy/cli_args.py` | Install aborts with a bare argparse usage error |
| Without `--enable-cors-header`, an origin-only middleware 403s cross-site requests; a native client sends no `Origin` and is unaffected | `server.py` middleware | Why easy mode is native Qt, not a webview |
| Terminal condition is `executing {node: null, prompt_id}`, sent **after** history is written | `execution.py` / websocket | Generate UI would poll or hang |
| Node outputs publish under varying keys (`images`, `audio`, `3d`, positional `result`) | `comfy_extras/nodes_save_3d.py` and friends | Output collection must never key off `"images"` |
| `/object_info` combo entries have two shapes: V1 puts options in slot 0, V3 puts the literal `"COMBO"` there with options under `opts["options"]` | `/object_info` responses | Model dropdowns show empty on V3 nodes |
| `VAELoader.vae_list()` synthesises TAESD names and `pixel_space` that are not files | `nodes.py` | VAE dropdown must come from `/object_info/VAELoader`, not `/models/vae` |
| TRELLIS.2 and Pixal3D are native in core with no compiled dependencies | `comfy_extras/nodes_trellis2.py` (PR #14718, merged 2026-08-22) | The entire single-venv, no-build-tools 3D story |
| Official templates may be **subgraph-based**: top-level node whose `type` is a UUID, real graph under `definitions.subgraphs[0]` with typed `inputs` | `Comfy-Org/workflow_templates` `image_z_image_turbo` | Parameter binding strategy; `derive_catalog.py` handles both shapes |
| Template `properties.models[]` carries `{name, url, directory}` verbatim | any template JSON | `derive_catalog.py`, i.e. the whole catalogue |

## Known upstream issues we route around

- `--base-directory` does not relocate the internal database (ComfyUI #10264).
  Worked around with `--database-url`.
- The built-in missing-model Download button has several open 2026 regressions
  (browser downloads landing in the OS Downloads folder, dead "Download All",
  `Download (NaN undefined)` on Civitai). We pre-download everything and keep
  the warning on only as a diagnostic.
- Some template `url` values are truncated mid-filename. `derive_catalog.py`
  repairs the path tail from the authoritative `name` field.
