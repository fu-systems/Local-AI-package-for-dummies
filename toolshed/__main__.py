"""Toolshed entry point.

Deliberately import-light. Everything imported at module scope runs on every
launch, is frozen into every build, and executes *before* the Qt platform
plugin has been proven to load -- so a failure here has no window to report
itself in. Qt is imported lazily, inside the functions that need it.

Absolute imports only: PyInstaller analyses whichever file the spec names as
the program's ``__main__``, and a relative import here would resolve against a
package that does not exist in the frozen build.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from toolshed import APP_NAME, __version__
from toolshed._streams import install_stream_guards

# The first real statement of the process. Must precede anything that can print.
install_stream_guards()


def diagnostics() -> str:
    from toolshed import resources  # noqa: PLC0415 -- keep module import cheap

    return "\n".join(
        [
            f"{APP_NAME} {__version__}",
            f"Python      {sys.version.split()[0]} ({platform.python_implementation()})",
            f"Platform    {platform.platform()}",
            f"Frozen      {resources.is_frozen()}",
            f"Bundle root {resources.bundle_root()}",
            f"Executable  {sys.executable}",
            f"stdout      {type(sys.stdout).__name__}",
        ]
    )


def selftest(report_path: Path | None = None) -> int:
    """Prove a built bundle is actually usable, then exit. This is what CI runs.

    It must touch everything that can only break once frozen: data files under
    ``sys._MEIPASS``, hardware detection, the Qt platform plugin load, and real
    widget construction. A build job that only checks PyInstaller exited zero
    proves nothing about whether the binary runs.
    """
    from toolshed import resources  # noqa: PLC0415

    lines = [diagnostics()]
    failures: list[str] = []

    recipes = sorted(resources.resource_path("catalog", "recipes").glob("*.yaml"))
    lines.append(f"catalog recipes: {len(recipes)}")
    if not recipes:
        failures.append("no catalogue recipes found in the bundle")

    workflows_dir = resources.resource_path("workflows")
    lines.append(f"workflows dir:   {'present' if workflows_dir.is_dir() else 'MISSING'}")
    if not workflows_dir.is_dir():
        failures.append("workflows directory missing from the bundle")

    # The packs the user chooses from. Loading them here means a bundle that
    # shipped without catalog/packs.yaml fails the build, not the beginner.
    from toolshed.catalog.packs import load_packs  # noqa: PLC0415

    packs = load_packs()
    missing = [p.id for p in packs if not p.recipe_found]
    unfrozen = [p.id for p in packs if not p.is_frozen]
    frozen = [p for p in packs if p.is_frozen]
    lines.append(f"packs:           {len(packs)} "
                 f"({len(frozen)} frozen, {len(unfrozen)} awaiting freeze)")
    if unfrozen:
        lines.append(f"awaiting freeze: {', '.join(unfrozen)}")
    if not packs:
        failures.append("no packs in the bundle; the choose screen would be blank")

    # A missing recipe file is a typo in `recipe:` and must fail the build. A
    # recipe that exists but still carries PENDING_FREEZE is a different and
    # legitimate state: the pack ships, greyed out, saying so on the choose
    # screen. This used to be one check reading the download size, which
    # conflated the two -- so the first deliberately unfrozen pack failed the
    # build with a message blaming a missing recipe that was sitting in the
    # bundle all along. tests/unit/test_packs.py separated them for the
    # catalogue loader; this is the same separation, here.
    if missing:
        failures.append(f"packs.yaml points at no such recipe: {', '.join(missing)}")
    # Worth keeping from the old check: if the data files were bundled empty or
    # truncated, every size would vanish while the files themselves still
    # existed, and nothing above would notice. A build where nothing at all can
    # be installed is not a usable binary.
    if packs and not frozen:
        failures.append("no pack has a download size; the bundled recipes are unusable")

    from toolshed.hw import detect, verdict_for  # noqa: PLC0415

    hw = detect()
    verdict = verdict_for(hw)
    lines.append(f"hardware:        os={hw.os} gpus={len(hw.gpus)}")
    lines.append(f"verdict:         supported={verdict.supported} reason={verdict.reason_key}")

    from toolshed.ui.app import build_application, qt_version  # noqa: PLC0415

    app, window = build_application(["toolshed-selftest"])
    window.show()
    from PySide6 import QtCore  # noqa: PLC0415

    QtCore.QTimer.singleShot(0, app.quit)
    app.exec()
    lines.append(f"qt:              {qt_version()} platform={app.platformName()}")

    lines.append("SELFTEST OK" if not failures else "SELFTEST FAIL: " + "; ".join(failures))
    text = "\n".join(lines)

    # A windowed Windows build has no stdout at all, so the report file is the
    # only channel CI can read. Write it before printing.
    if report_path is not None:
        report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toolshed",
        description=f"{APP_NAME} — local AI, set up for you.",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    parser.add_argument("--diagnostics", action="store_true",
                        help="print build and platform information, then exit")
    parser.add_argument("--selftest", action="store_true",
                        help="verify this build can run, then exit")
    parser.add_argument("--report", type=Path, default=None, metavar="PATH",
                        help="write --selftest output to this file as well as stdout")
    args = parser.parse_args(argv)

    if args.diagnostics:
        print(diagnostics())
        return 0
    if args.selftest:
        return selftest(args.report)

    from toolshed.ui.app import run_gui  # noqa: PLC0415

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
