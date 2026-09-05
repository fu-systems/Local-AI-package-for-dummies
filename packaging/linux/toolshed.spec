# -*- mode: python -*-
"""PyInstaller spec for the Linux build.

Built ONLY inside an ubuntu:22.04 container. The PySide6-Essentials wheel is
manylinux_2_34, so glibc 2.34 is the floor; building on 24.04 (glibc 2.39)
produces a binary that dies on 22.04 with a loader error a novice cannot act on.

Run from the repository root:
    pyinstaller --clean --noconfirm packaging/linux/toolshed.spec
"""

import sys
from pathlib import Path

REPO = Path(SPECPATH).resolve().parents[1]
sys.path.insert(0, str(REPO / "packaging"))

from _spec_common import DATA_DIRS, OTHER_EXCLUDES, QT_EXCLUDES, TREE_EXCLUDES

# Tree() yields (dest, src, typecode) triples destined for COLLECT, not the
# (src, dest) pairs Analysis(datas=...) takes. Passing them to Analysis fails
# with "too many values to unpack". They are splatted into COLLECT below.
data_trees = [
    Tree(str(REPO / name), prefix=name, excludes=TREE_EXCLUDES) for name in DATA_DIRS
]

a = Analysis(
    [str(REPO / "packaging" / "entry.py")],
    pathex=[str(REPO)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # keyring finds its backends through entry points. PyInstaller's core
        # hook already does the heavy lifting; the Linux backend chain reaches
        # secretstorage through a try/except that the module graph cannot see.
        "keyring.backends.SecretService",
        "keyring.backends.chainer",
        "keyring.backends.fail",
        "secretstorage",
        "jeepney",
        "jeepney.io.blocking",
    ],
    # hookspath: add packaging/hooks/ (with a real file in it) when a
    # custom hook is genuinely needed. An empty directory is not tracked
    # by git and would vanish on a fresh clone.
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=QT_EXCLUDES + OTHER_EXCLUDES,
    noarchive=False,
    # NOT 1 or 2: -O strips asserts, and asserts are used as invariants.
    # Silently deleting them in the shipped build is the wrong direction.
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="toolshed",
    debug=False,
    bootloader_ignore_signals=False,
    # The Qt wheel libraries are already stripped; stripping again buys almost
    # nothing and makes every future crash report from a beta tester useless.
    strip=False,
    # Explicit. UPX is the single largest antivirus false-positive generator
    # for Python freezers, and a false positive is fatal for a product whose
    # job is reassuring a nervous beginner.
    upx=False,
    upx_exclude=[],
    # PyInstaller's own docs: console is always True on Linux and does not
    # matter there. Writing False would be a lie that reads like a guarantee.
    # The null-stream guard still ships, because the same source tree builds
    # the windowed Windows executable where it is load-bearing.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon= is ignored on Linux. The icon is installed as a data file by
    # install.sh into the hicolor theme.
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    *data_trees,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="toolshed",
)
