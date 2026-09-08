#!/usr/bin/env python3
"""Turn the PENDING_FREEZE facts in a recipe into verified ones.

`tools/derive_catalog.py` writes PENDING_FREEZE wherever a fact must come from
Hugging Face rather than from a workflow template. This is the tool that fills
those in, and it is what every pack in the catalogue is waiting on: nothing can
ship while a PENDING_FREEZE remains, and today all seven recipes carry between
18 and 48 of them.

Four fields per file, and where each comes from:

    revision    GET /api/models/{repo}                  -> "sha"
    sha256      POST /api/models/{repo}/paths-info/{rev} -> lfs.oid
    size_bytes  the same call                            -> lfs.size, else size
    gated       GET /api/models/{repo}                   -> "gated"

The LFS oid **is** the sha256 of the file, which is why integrity never has to
depend on a live API at install time. Those mappings are not invented here:
they are the ones already recorded in derive_catalog.py by the person who read
them.

Two rules this tool will not bend:

* **A fact it cannot read stays PENDING_FREEZE.** A partial freeze is a useful
  state -- it says exactly what is still unknown. A guessed hash is worse than
  no hash, because it fails at the end of a 20 GB download with a message
  blaming the network.
* **`revision` is a commit sha, never a branch.** A branch would let the file
  change under a frozen hash, which is the failure the hash exists to catch.

    python3 tools/freeze_manifest.py --check                # what is still pending
    python3 tools/freeze_manifest.py catalog/recipes/x.yaml # freeze one
    python3 tools/freeze_manifest.py --all                  # freeze everything

Nothing here writes a human-facing field. `name`, `blurb`, `modality` and the
variant labels are somebody's judgement, and this tool leaves them alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PENDING = "PENDING_FREEZE"
API = "https://huggingface.co"

# Only these are ours to write. Everything else in a recipe is either derived
# from a template (and re-checked by lint) or a human's decision.
FROZEN_FIELDS = ("revision", "sha256", "size_bytes", "gated", "licence")

SHA_LENGTH = 40
SHA256_LENGTH = 64


class FreezeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# talking to Hugging Face
# --------------------------------------------------------------------------

@dataclass
class HuggingFace:
    """The three calls this needs, behind one seam so tests can replace it."""

    token: str | None = None
    timeout: int = 60

    def _get(self, url: str, payload: dict | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(url, data=data)
        request.add_header("Accept", "application/json")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise FreezeError(
                    f"{url} needs authorisation (HTTP {exc.code}). If the repo is "
                    f"gated, accept its terms on the web and pass --token."
                ) from exc
            raise FreezeError(f"{url} returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise FreezeError(f"cannot reach {url}: {exc.reason}") from exc

    def model_info(self, repo: str) -> dict:
        return self._get(f"{API}/api/models/{urllib.parse.quote(repo)}")

    def paths_info(self, repo: str, revision: str, paths: list[str]) -> list[dict]:
        found = self._get(
            f"{API}/api/models/{urllib.parse.quote(repo)}/paths-info/"
            f"{urllib.parse.quote(revision)}",
            payload={"paths": paths},
        )
        return found if isinstance(found, list) else []


# --------------------------------------------------------------------------
# reading what the API said, carefully
# --------------------------------------------------------------------------

def commit_sha(info: dict) -> str:
    """The repo's current commit, or PENDING if it does not look like one."""
    sha = str(info.get("sha") or "").strip().lower()
    return sha if len(sha) == SHA_LENGTH and all(
        c in "0123456789abcdef" for c in sha) else PENDING


def gated_flag(info: dict) -> Any:
    """HF answers false, or a string naming the kind of gate.

    Both are facts. Only a missing key is unknown -- and `False` must survive
    the round trip, which a truthiness test would quietly turn into PENDING.
    """
    if "gated" not in info:
        return PENDING
    return info["gated"]


def licence_id(info: dict) -> str:
    """The licence the model card declares, if it declares one."""
    card = info.get("cardData")
    if isinstance(card, dict):
        name = card.get("license")
        if isinstance(name, str) and name.strip():
            return name.strip()
        if isinstance(name, list) and name and isinstance(name[0], str):
            return name[0].strip()
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].strip() or PENDING
    return PENDING


def file_facts(entry: dict) -> tuple[str, Any]:
    """(sha256, size_bytes) for one paths-info entry.

    The LFS oid is the sha256. A file small enough not to be in LFS has no oid
    at all, and there is no sha256 to be had from this API -- so it stays
    pending rather than becoming a plausible-looking wrong value.
    """
    lfs = entry.get("lfs")
    sha = PENDING
    size: Any = PENDING
    if isinstance(lfs, dict):
        oid = str(lfs.get("oid") or "").strip().lower()
        if len(oid) == SHA256_LENGTH and all(c in "0123456789abcdef" for c in oid):
            sha = oid
        if isinstance(lfs.get("size"), int):
            size = lfs["size"]
    if size is PENDING and isinstance(entry.get("size"), int):
        size = entry["size"]
    return sha, size


# --------------------------------------------------------------------------
# freezing a recipe
# --------------------------------------------------------------------------

@dataclass
class Report:
    frozen: list[str] = field(default_factory=list)
    still_pending: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.frozen)


def freeze_recipe(doc: dict, hf: HuggingFace) -> Report:
    """Fill in what Hugging Face can tell us. Mutates ``doc`` in place."""
    report = Report()
    files = doc.get("files") or {}
    if not files:
        report.problems.append("no files: block")
        return report

    # One model_info per repo, not per file: a recipe with five files from one
    # repo should ask once.
    by_repo: dict[str, list[str]] = {}
    for key, entry in files.items():
        repo = entry.get("repo")
        if not repo or repo == PENDING:
            report.problems.append(f"{key}: repo is {repo or 'missing'}")
            continue
        by_repo.setdefault(repo, []).append(key)

    for repo, keys in by_repo.items():
        try:
            info = hf.model_info(repo)
        except FreezeError as exc:
            report.problems.append(f"{repo}: {exc}")
            continue

        sha = commit_sha(info)
        gated = gated_flag(info)
        licence = licence_id(info)

        paths = [files[k].get("path") for k in keys]
        entries: dict[str, dict] = {}
        if sha != PENDING and all(p and p != PENDING for p in paths):
            try:
                for item in hf.paths_info(repo, sha, list(paths)):
                    if isinstance(item, dict) and item.get("path"):
                        entries[item["path"]] = item
            except FreezeError as exc:
                report.problems.append(f"{repo}: {exc}")

        for key in keys:
            entry = files[key]
            item = entries.get(entry.get("path"))
            file_sha, size = file_facts(item) if item is not None else (PENDING, PENDING)

            # The revision and the hash are one fact in two fields, and they
            # must move together. Advancing revision to a new commit while
            # leaving the old commit's sha256 in place builds a manifest that
            # points a new URL at an old hash: every install then downloads a
            # file that fails verification, and the message blames the user's
            # connection. So if the commit has moved and no new hash came back,
            # clear both and say so.
            moved = (entry.get("revision") not in (PENDING, None, sha)
                     and sha != PENDING)
            if moved and file_sha == PENDING:
                entry["revision"] = PENDING
                entry["sha256"] = PENDING
                entry["size_bytes"] = PENDING
                report.problems.append(
                    f"{key}: {repo} has moved to {sha[:12]} but its hash could not "
                    f"be read; revision and sha256 cleared rather than paired with "
                    f"the previous commit's hash")
                continue
            # A real re-freeze: the commit moved AND a hash came back with it,
            # so both may be replaced -- together, from the same commit.
            _set(entry, "revision", sha, key, report, overwrite=moved)
            _set(entry, "gated", gated, key, report)
            _set(entry, "licence", licence, key, report)
            if item is None:
                _note_pending(entry, key, ("sha256", "size_bytes"), report)
                continue
            _set(entry, "sha256", file_sha, key, report, overwrite=True)
            _set(entry, "size_bytes", size, key, report, overwrite=True)

    # The total the confirmation screen shows, once every part of it is known.
    sizes = [f.get("size_bytes") for f in files.values()]
    if sizes and all(isinstance(s, int) for s in sizes):
        # Only when it actually changes. Reporting it every time would make a
        # re-run rewrite a file it had nothing to add to, and "froze" would
        # stop meaning anything.
        total = sum(sizes)
        if doc.get("estimated_download_bytes") != total:
            doc["estimated_download_bytes"] = total
            report.frozen.append("estimated_download_bytes")

    return report


def _set(entry: dict, field_name: str, value: Any, key: str, report: Report,
         *, overwrite: bool = False) -> None:
    if value == PENDING or value is None:
        if entry.get(field_name) == PENDING:
            report.still_pending.append(f"{key}.{field_name}")
        return
    current = entry.get(field_name)
    if current == value:
        return
    if current not in (PENDING, None) and not overwrite:
        # A frozen fact that now reads differently is either an upstream
        # change or a mistake, and this tool cannot tell which. Say so rather
        # than quietly replacing it.
        report.problems.append(
            f"{key}.{field_name} is already frozen as {current!r} but the API now "
            f"says {value!r}; left alone -- reset it to {PENDING} to re-freeze")
        return
    entry[field_name] = value
    report.frozen.append(f"{key}.{field_name}")


def _note_pending(entry: dict, key: str, names: tuple[str, ...], report: Report) -> None:
    for name in names:
        if entry.get(name) == PENDING:
            report.still_pending.append(f"{key}.{name}")


def pending_in(doc: Any, path: str = "") -> list[str]:
    """Every PENDING_FREEZE left in a document, by dotted path."""
    found: list[str] = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            found += pending_in(value, f"{path}.{key}" if path else str(key))
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            found += pending_in(value, f"{path}[{index}]")
    elif doc == PENDING:
        found.append(path)
    return found


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _rewritten(text: str, doc: dict) -> str:
    """The original file with only the frozen scalars replaced.

    A YAML re-dump loses every comment, and in this repo the comments are the
    provenance -- why a model was chosen, which criteria it met, what is still
    unverified. Losing them on the first freeze would be worse than the freeze
    is worth. So the values are edited in place by key, at their own
    indentation, and nothing else in the file is disturbed.

    Deliberately literal about structure rather than clever: it only rewrites a
    scalar whose key it recognises, in a place it recognises, and leaves every
    other line exactly as it found it.
    """
    files = doc.get("files") or {}
    total = doc.get("estimated_download_bytes")

    def render(value: Any) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)

    out: list[str] = []
    in_files = False
    files_indent = 0
    file_key: str | None = None
    key_indent: int | None = None

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        name = stripped.split(":", 1)[0].strip().lstrip("- ")

        if in_files and indent <= files_indent:
            in_files, file_key, key_indent = False, None, None
        if not in_files and name == "files" and stripped.endswith(":"):
            in_files, files_indent = True, indent
            out.append(line)
            continue
        if (in_files and stripped.endswith(":") and name in files
                and (key_indent is None or indent <= key_indent)):
            file_key, key_indent = name, indent
            out.append(line)
            continue

        value: Any = None
        if not in_files and indent == 0 and name == "estimated_download_bytes" \
                and isinstance(total, int):
            value = total
        elif file_key and key_indent is not None and indent > key_indent \
                and name in FROZEN_FIELDS:
            entry = files.get(file_key) or {}
            if name in entry and entry[name] != PENDING:
                value = entry[name]

        if value is None:
            out.append(line)
            continue
        out.append(f"{' ' * indent}{name}: {render(value)}\n")
    return "".join(out)


def recipe_paths(args: argparse.Namespace) -> list[Path]:
    if args.recipes:
        # Named recipes win, even alongside --check. Silently scanning the
        # whole catalogue when someone asked about one file is a small lie
        # that reads as a much bigger answer.
        return [Path(p) for p in args.recipes]
    if args.all or args.check:
        return sorted((REPO / "catalog" / "recipes").glob("*.yaml"))
    return []


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipes", nargs="*", help="recipe files to freeze")
    ap.add_argument("--all", action="store_true", help="freeze every recipe")
    ap.add_argument("--check", action="store_true",
                    help="report what is still pending and change nothing")
    ap.add_argument("--token", help="Hugging Face token, for gated repos")
    args = ap.parse_args(argv)

    paths = recipe_paths(args)
    if not paths:
        ap.error("name a recipe, or pass --all or --check")

    if args.check:
        total = 0
        for path in paths:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            pending = pending_in(doc)
            total += len(pending)
            print(f"{path.name}: {len(pending)} pending")
            for item in pending:
                print(f"    {item}")
        print(f"\n{total} PENDING_FREEZE in {len(paths)} recipe(s)")
        return 0

    hf = HuggingFace(token=args.token)
    problems = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text) or {}
        print(f"\n{path.name}")
        report = freeze_recipe(doc, hf)

        for item in report.frozen:
            print(f"  froze    {item}")
        for item in report.still_pending:
            print(f"  PENDING  {item}")
        for item in report.problems:
            print(f"  PROBLEM  {item}", file=sys.stderr)
            problems += 1

        if report.changed:
            # Rewritten line by line rather than re-dumped. yaml.safe_dump
            # keeps no comments at all, and the adult recipe carries 35 lines
            # of indented rationale explaining which model was chosen and why
            # -- running the documented freeze command would have deleted every
            # word of it. Only the scalar values this tool is allowed to write
            # are touched; everything else in the file survives byte for byte.
            path.write_text(_rewritten(text, doc), encoding="utf-8")
            print(f"  wrote    {path}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
