"""The screen this product exists for: tick what you want to make.

Everything on it comes from the catalogue. Adding a pack is a YAML change, not
a UI change -- which is the whole point of `catalog/packs.yaml` being data.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from toolshed.catalog.packs import Pack, adult, general, modalities, total_bytes
from toolshed.hw.detect import HardwareReport


class PackRow(QtWidgets.QWidget):
    """One tickable pack, or one greyed-out pack with the reason why."""

    toggled = QtCore.Signal()

    def __init__(self, pack: Pack, report: HardwareReport) -> None:
        super().__init__()
        self.pack = pack
        gpu = report.primary
        vram = gpu.vram_gb if gpu else None
        self.reason = pack.unavailable_reason(vram)
        available = self.reason is None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 8)
        layout.setSpacing(2)

        self.checkbox = QtWidgets.QCheckBox(pack.name)
        font = self.checkbox.font()
        font.setBold(True)
        self.checkbox.setFont(font)
        self.checkbox.setEnabled(available)
        # Never pre-tick something the user cannot actually run.
        self.checkbox.setChecked(pack.default_checked and available)
        self.checkbox.toggled.connect(self.toggled)
        layout.addWidget(self.checkbox)

        blurb = QtWidgets.QLabel(pack.blurb)
        blurb.setWordWrap(True)
        blurb.setEnabled(available)
        blurb.setContentsMargins(22, 0, 0, 0)
        layout.addWidget(blurb)

        bits = [pack.size_text(), pack.licence]
        if gpu and pack.is_experimental_for(gpu.vendor):
            bits.append("Experimental on AMD — we test it during setup and tell you honestly.")
        meta = QtWidgets.QLabel(" · ".join(bits))
        meta.setWordWrap(True)
        meta.setEnabled(False)
        meta.setContentsMargins(22, 0, 0, 0)
        layout.addWidget(meta)

        if self.reason:
            why = QtWidgets.QLabel(self.reason)
            why.setWordWrap(True)
            why.setContentsMargins(22, 0, 0, 0)
            layout.addWidget(why)

    def is_selected(self) -> bool:
        return self.checkbox.isChecked() and self.checkbox.isEnabled()


CONFIRM_TITLE = "Adult content"

# Said once, plainly, before anything explicit is on screen. Three separate
# things, because they are three separate things: what it is, where it goes,
# and whose problem it is.
CONFIRM_BODY = (
    "These options install models that make sexually explicit pictures and video.\n\n"
    "Everything is generated and stored on this computer. Nothing you make is "
    "uploaded, and there is no account and no telemetry.\n\n"
    "You are responsible for what you make with them, and for the law where you "
    "live. Making sexual images of real people without their consent, or of "
    "anyone under 18, is illegal in most countries — including where the images "
    "are generated rather than photographed."
)


class AdultSection(QtWidgets.QGroupBox):
    """The adult packs, behind one deliberate action.

    Not a filter to be switched off in a settings screen somewhere: the rows do
    not exist on screen until someone has read what they are and said yes. That
    is the difference between a product that offers this and a product that puts
    it in front of a beginner who came here to make a picture of a dog.
    """

    toggled = QtCore.Signal()

    def __init__(self, packs: tuple[Pack, ...], report: HardwareReport) -> None:
        super().__init__("Adult content")
        self.unlocked = False
        self.rows: list[PackRow] = []

        layout = QtWidgets.QVBoxLayout(self)

        self.blurb = QtWidgets.QLabel(
            "Models for sexually explicit pictures and video. Off unless you turn "
            "them on, and never part of the defaults."
        )
        self.blurb.setWordWrap(True)
        layout.addWidget(self.blurb)

        self.reveal = QtWidgets.QPushButton("Show adult content options…")
        self.reveal.clicked.connect(self._confirm)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.reveal)
        row.addStretch(1)
        layout.addLayout(row)

        # Built now, hidden until unlocked. Building them on demand would mean
        # the hardware floors and greying-out ran on a different code path from
        # every other pack, which is exactly where a "needs 16 GB" check gets
        # quietly skipped.
        self.holder = QtWidgets.QWidget()
        holder_layout = QtWidgets.QVBoxLayout(self.holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        for pack in packs:
            pack_row = PackRow(pack, report)
            pack_row.toggled.connect(self.toggled)
            self.rows.append(pack_row)
            holder_layout.addWidget(pack_row)
        self.holder.setVisible(False)
        layout.addWidget(self.holder)

    def _confirm(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(CONFIRM_TITLE)
        box.setText("Show adult content options?")
        box.setInformativeText(CONFIRM_BODY)
        yes = box.addButton("I am over 18 — show them", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("No thanks", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        # Cancel is the safe answer, so it is the one Enter and Escape both give.
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.NoButton)
        box.exec()
        if box.clickedButton() is yes:
            self.unlock()

    def unlock(self) -> None:
        self.unlocked = True
        self.holder.setVisible(True)
        self.reveal.setVisible(False)
        self.blurb.setVisible(False)
        self.toggled.emit()

    def selected(self) -> list[Pack]:
        """Nothing is selected while the section is shut, whatever the boxes say."""
        if not self.unlocked:
            return []
        return [r.pack for r in self.rows if r.is_selected()]


class ChoosePage(QtWidgets.QWidget):
    selection_changed = QtCore.Signal()

    def __init__(self, packs: tuple[Pack, ...], report: HardwareReport) -> None:
        super().__init__()
        self.rows: list[PackRow] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 8)
        outer.setSpacing(10)

        heading = QtWidgets.QLabel("What do you want to make?")
        f = heading.font()
        f.setPointSize(f.pointSize() + 5)
        f.setBold(True)
        heading.setFont(f)
        outer.addWidget(heading)

        sub = QtWidgets.QLabel(
            "Pick as many as you like. You can add more later without downloading "
            "anything twice."
        )
        sub.setWordWrap(True)
        outer.addWidget(sub)

        # Scroll, because six packs on a small laptop screen already overflow.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        everyday = general(packs)
        for modality in modalities(everyday):
            box = QtWidgets.QGroupBox(modality)
            box_layout = QtWidgets.QVBoxLayout(box)
            for pack in [p for p in everyday if p.modality == modality]:
                row = PackRow(pack, report)
                row.toggled.connect(self._on_toggle)
                self.rows.append(row)
                box_layout.addWidget(row)
            inner_layout.addWidget(box)

        # Last, and absent entirely when the catalogue has none -- which is the
        # state today. No empty group box hinting at something that is not there.
        self.adult_section: AdultSection | None = None
        explicit = adult(packs)
        if explicit:
            self.adult_section = AdultSection(explicit, report)
            self.adult_section.toggled.connect(self._on_toggle)
            inner_layout.addWidget(self.adult_section)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self.total_label = QtWidgets.QLabel()
        tf = self.total_label.font()
        tf.setBold(True)
        self.total_label.setFont(tf)
        outer.addWidget(self.total_label)
        self._on_toggle()

    def selected(self) -> list[Pack]:
        chosen = [r.pack for r in self.rows if r.is_selected()]
        if self.adult_section is not None:
            chosen += self.adult_section.selected()
        return chosen

    def _on_toggle(self) -> None:
        chosen = self.selected()
        if not chosen:
            self.total_label.setText("Nothing selected yet.")
        else:
            gb = total_bytes(chosen) / 1e9
            noun = "thing" if len(chosen) == 1 else "things"
            # "At most", because packs share large files and the exact total is
            # only known once the manifest is frozen and deduplicated by hash.
            self.total_label.setText(
                f"{len(chosen)} {noun} to set up · at most {gb:.0f} GB to download"
            )
        self.selection_changed.emit()
