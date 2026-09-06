"""XDG desktop-entry lookup.

Deliberately Qt-free. This is a filesystem question -- is there a .desktop file
for us? -- and it lived in the Qt module only because that is where it was
first needed. That forced anything testing it to import PySide6, which broke CI
where Qt is not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def desktop_file_installed(name: str) -> bool:
    """True if ``<name>.desktop`` exists in any XDG application directory.

    Follows the XDG base directory spec: ``$XDG_DATA_HOME`` (default
    ``~/.local/share``) then each entry of ``$XDG_DATA_DIRS`` (default
    ``/usr/local/share:/usr/share``).
    """
    if sys.platform != "linux":
        return True  # only the XDG portal cares; other platforms are unaffected

    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    for base in [data_home, *data_dirs.split(":")]:
        if base and Path(base, "applications", f"{name}.desktop").is_file():
            return True
    return False
