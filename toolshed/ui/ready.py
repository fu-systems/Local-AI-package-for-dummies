"""The confirmation screen: exactly what is about to happen, and where."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtWidgets

from toolshed.catalog.packs import Pack, total_bytes


def default_data_root() -> Path:
    """Where an installation would live, per docs/PLAN.md section 3."""
    if sys.platform == "win32":
        return Path(os.environ.get("SYSTEMDRIVE", "C:")) / "Toolshed"
    return Path.home() / "Toolshed"


class ReadyPage(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 8)
        layout.setSpacing(10)

        self.heading = QtWidgets.QLabel("Ready?")
        f = self.heading.font()
        f.setPointSize(f.pointSize() + 5)
        f.setBold(True)
        self.heading.setFont(f)
        layout.addWidget(self.heading)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.plan = QtWidgets.QTextEdit()
        self.plan.setReadOnly(True)
        layout.addWidget(self.plan, 1)

        # What pressing the button will actually do, before it is pressed.
        notice = QtWidgets.QLabel(
            "Nothing has been downloaded yet. When you press <b>Set it up</b> we will "
            "install a private copy of Python and the AI engine, fetch the models above, "
            "and add the ready-made workflows. You can stop at any point and pick up "
            "where you left off."
        )
        notice.setWordWrap(True)
        notice.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(notice)

    def set_selection(self, packs: list[Pack]) -> None:
        if not packs:
            self.summary.setText("Nothing selected. Go back and pick at least one thing.")
            self.plan.setPlainText("")
            return

        gb = total_bytes(packs) / 1e9
        noun = "thing" if len(packs) == 1 else "things"
        self.summary.setText(
            f"{len(packs)} {noun} to set up, at most {gb:.0f} GB to download, "
            f"into {default_data_root()}"
        )

        lines: list[str] = []
        for pack in packs:
            lines.append(f"{pack.name}")
            lines.append(f"    {pack.size_text()}")
            lines.append(f"    {pack.licence}")
            lines.append(f"    recipe: {pack.recipe}")
            lines.append("")
        lines.append(
            "Sizes are an upper bound: packs share large files, and the exact total is "
            "known once every file has been checked against its published hash."
        )
        self.plan.setPlainText("\n".join(lines))
