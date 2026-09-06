# Standing instructions for AI agents working in this repository

## Builds are triggered manually, by a human, only

**`build-linux` and `build-windows` run on `workflow_dispatch` and nothing else.**
Do not add `push`, `pull_request`, `schedule`, or tag triggers to them.

**NO AI AGENT MAY EVER TRIGGER A BUILD.** Not by pushing, not by merging, not by
dispatching a workflow, not by any other means. If a build is needed, say so and
let a human start it.

This is not a style preference. An unbounded wait in a build step, triggered
automatically on merge, burned hours of runner time before anyone could react.
The owner starts builds, at a moment of their choosing, and nothing else does.

Pushing a branch is fine. Opening a pull request only when asked is fine. Neither
may cause a build to run.

## Do not open pull requests unless asked

Push the branch and say it is ready. The owner decides when to open and merge.

## Never guess a fact

Model filenames, sizes, hashes, licence terms, version numbers, CLI flags: verify
them against a live source, or write `PENDING_FREEZE` and say it is unverified. An
acknowledged unknown is useful; an invented fact is worse than silence and
propagates into everything downstream.

## The GPL boundary with ComfyUI is absolute

Toolshed downloads ComfyUI and drives it as a separate process over HTTP. Never
import it, never bundle it, never place our code under `custom_nodes/`. CI
enforces this; see `CONTRIBUTING.md`.

## Verify builds on Python 3.12

`pyproject.toml` requires 3.12 and CI uses 3.12. Verifying a packaging change on
3.11 proves less than it appears to — 3.11 still has `distutils` in the standard
library and will pass a spec that 3.12 rejects.
