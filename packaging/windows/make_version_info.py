#!/usr/bin/env python3
"""Generate the Windows VERSIONINFO resource embedded in the executables.

Without it the Properties tab reads 0.0.0.0, which looks like a broken or
untrustworthy download -- and this application is already asking a nervous
beginner to run an unsigned binary.

The structure's `filevers` field must be a 4-tuple of integers, so any
pre-release suffix is stripped for it while the full string is kept in the
human-readable FileVersion field.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TEMPLATE = """\
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vers}, prodvers={vers},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'fu.systems'),
      StringStruct('FileDescription', 'Toolshed — local AI, set up for you'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'toolshed'),
      StringStruct('LegalCopyright', 'Apache-2.0'),
      StringStruct('OriginalFilename', 'toolshed.exe'),
      StringStruct('ProductName', 'Toolshed'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def numeric_version(version: str) -> tuple[int, int, int, int]:
    """Strip any pre-release suffix and pad to the 4-tuple Windows requires."""
    core = re.split(r"[^0-9.]", version, maxsplit=1)[0]
    parts = [int(p) for p in core.split(".") if p.isdigit()][:4]
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        TEMPLATE.format(vers=numeric_version(args.version), version=args.version),
        encoding="utf-8",
    )
    print(f"wrote {args.out} for version {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
