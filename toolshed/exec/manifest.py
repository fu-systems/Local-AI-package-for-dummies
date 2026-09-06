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
    notes: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.data_root / "state" / "manifest.json"

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
            "notes": self.notes,
            "files": [asdict(e) for e in self.files],
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def record(self, entry: Entry) -> None:
        self.files = [e for e in self.files if e.path != entry.path]
        self.files.append(entry)

    @property
    def installed_filenames(self) -> set[str]:
        """Used to skip files a later pack already brought in."""
        return {Path(e.path).name for e in self.files}
