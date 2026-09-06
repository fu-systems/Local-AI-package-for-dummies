"""A record of every file we put on this machine.

Written as each step completes, not at the end, so an install interrupted by a
crash or a power cut still describes what actually landed. It is what makes
Repair possible (diff the record against reality) and what makes uninstall
safe: we delete only what we recorded putting there, and only if it still
matches, because anything else is the user's.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = 1


@dataclass
class Entry:
    path: str
    sha256: str
    size_bytes: int
    source: str = ""
    pack: str = ""
    # Whether the hash was frozen at release time or resolved from the
    # publisher during this install. Never leave that ambiguous afterwards.
    hash_verified_against: str = "publisher"


@dataclass
class Manifest:
    data_root: Path
    schema: int = SCHEMA
    packs: list[str] = field(default_factory=list)
    files: list[Entry] = field(default_factory=list)
    engine_tag: str = ""
    torch_index: str = ""
    # The environment the installer chose for PyTorch -- on AMD, the
    # HSA_OVERRIDE_GFX_VERSION a card needs before ROCm will drive it. The
    # engine must be started with the same one, or a card that passed the
    # install-time check is invisible at run time.
    torch_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.data_root / "state" / "manifest.json"

    @property
    def hashes_path(self) -> Path:
        """A second record of the model hashes, kept *with* the models.

        Uninstall keeps the models folder and deletes everything else, this
        manifest included. Without this file the next install would have no
        way to tell whether the 40 GB it finds there is intact, and would
        fetch it all again -- which is exactly what keeping the models was
        meant to avoid.
        """
        return self.data_root / "models" / ".toolshed-hashes.json"

    @classmethod
    def load(cls, data_root: Path) -> Manifest:
        path = data_root / "state" / "manifest.json"
        if not path.is_file():
            return cls(data_root=data_root)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(data_root=data_root)
        return cls(
            data_root=data_root,
            schema=doc.get("schema", SCHEMA),
            packs=list(doc.get("packs", [])),
            files=[Entry(**e) for e in doc.get("files", [])],
            engine_tag=doc.get("engine_tag", ""),
            torch_index=doc.get("torch_index", ""),
            torch_env={str(k): str(v) for k, v in (doc.get("torch_env") or {}).items()},
            notes=list(doc.get("notes", [])),
        )

    def save(self) -> None:
        """Write atomically: a half-written manifest is worse than none."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema": self.schema,
            "packs": self.packs,
            "engine_tag": self.engine_tag,
            "torch_index": self.torch_index,
            "torch_env": self.torch_env,
            "notes": self.notes,
            "files": [asdict(e) for e in self.files],
        }
        _write_atomically(self.path, doc)

    def record(self, entry: Entry) -> None:
        self.files = [e for e in self.files if e.path != entry.path]
        self.files.append(entry)

    def remember_hash(self, entry: Entry) -> None:
        """Note a model's hash in the sidecar that survives uninstall."""
        doc = self._load_hashes()
        doc[entry.path] = {"sha256": entry.sha256, "size_bytes": entry.size_bytes}
        _write_atomically(self.hashes_path, doc)

    def prior_hash(self, rel_path: str) -> tuple[str, int] | None:
        """(sha256, size) we recorded for a file, from either record."""
        entry = next((e for e in self.files if e.path == rel_path), None)
        if entry is not None and entry.sha256:
            return entry.sha256, entry.size_bytes
        kept = self._load_hashes().get(rel_path)
        if isinstance(kept, dict) and kept.get("sha256"):
            return str(kept["sha256"]), int(kept.get("size_bytes", -1))
        return None

    def _load_hashes(self) -> dict:
        if not self.hashes_path.is_file():
            return {}
        try:
            doc = json.loads(self.hashes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return doc if isinstance(doc, dict) else {}

    @property
    def installed_filenames(self) -> set[str]:
        """Used to skip files a later pack already brought in."""
        return {Path(e.path).name for e in self.files}


def _write_atomically(path: Path, doc: dict) -> None:
    """Write atomically: a half-written record is worse than none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
