# Contributing

## The one invariant that is not negotiable

**Toolshed must never link against, embed, import, patch or redistribute
ComfyUI.** It downloads ComfyUI from upstream at install time and drives it as
a separate process over HTTP. ComfyUI is GPL-3.0-or-later; that boundary is
what keeps this project Apache-2.0, and it is enforced in CI by
`tools/gpl_boundary_check.py`, not by good intentions.

Concretely, a change may not:

- `import comfy`, `import nodes`, or `import folder_paths`
- place any `.py` file of ours under `custom_nodes/`
- add ComfyUI to a PyInstaller spec or any packaged artifact
- vendor ComfyUI source into this repository

If you need behaviour from ComfyUI, get it over the HTTP or WebSocket API and
record the dependency in `docs/UPSTREAM.md`.

## The catalogue is derived, not written

Model filenames and destination folders are **generated** by
`tools/derive_catalog.py` from Comfy Org's official workflow templates. Do not
hand-edit them. A typo produces `value_not_in_list` on a beginner's first run,
which is the exact failure this project exists to prevent.

```bash
python3 tools/derive_catalog.py --list          # what v1 covers
python3 tools/derive_catalog.py --all           # regenerate the skeletons
```

`PENDING_FREEZE` is the **only** placeholder token. It means "this is a fact we
have not proven yet". Never replace one with a plausible-looking value; either
prove it or leave it. A release build fails while any remain.

## Never guess a fact

Sizes, hashes, licences and gated flags come from a live source and are frozen
at release time. Version numbers, CLI flags and file paths come from upstream
source you have actually read. If you cannot verify something, say so in the
pull request and mark it — an acknowledged unknown is useful, an invented fact
poisons everything downstream of it.

## Beginner-facing text

Every string a user can see is part of the product. No tracebacks in the
primary view, no jargon without a plain-English gloss, no "error 0x80070005".
Every failure needs four things: what happened, why, what to do now, and a way
to copy diagnostics.

## Verify builds on Python 3.12, not whatever you have

`pyproject.toml` requires 3.12 and CI builds on 3.12. Verifying a PyInstaller
change on 3.11 proves less than it looks: 3.11 still has `distutils` in the
standard library, so it never exercises the setuptools-vendored aliasing that
3.12 depends on. A spec that builds cleanly on 3.11 can fail outright on 3.12.
This has already cost one round of red CI.

    python3.12 -m venv .venv-build && .venv-build/bin/pip install . pyinstaller
    .venv-build/bin/pyinstaller --clean --noconfirm packaging/linux/toolshed.spec

Relatedly: never put `distutils`, `setuptools`, `pkg_resources`, `pip` or
`wheel` in a spec's `excludes`. See the comment in `packaging/_spec_common.py`;
a unit test enforces it.

## Build environment

Linux builds run in an `ubuntu:22.04` container. This is not incidental: the
PySide6 wheel floors us at glibc 2.34, and building on 24.04 produces a binary
that dies on 22.04 with a linker error the user cannot act on.

## Running the tests

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

The unit suite is pure and needs no GPU and no network, which matters because
this project is developed on hardware that cannot run most of what it installs.
Hardware-specific policy is tested against recorded fixtures in
`tests/fixtures/hardware/` rather than against a real card.
