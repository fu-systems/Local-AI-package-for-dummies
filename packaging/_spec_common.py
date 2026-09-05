"""Shared pieces of the PyInstaller specs.

Imported by packaging/{linux,windows}/toolshed.spec, which add sys.path for it.
Keeping the excludes and data trees here means the two platform specs cannot
silently diverge.
"""

from __future__ import annotations

# --- Qt modules we do not use -------------------------------------------------
# These are a TRIPWIRE, not a size optimisation. PyInstaller's Qt collection is
# demand-driven from the module graph, so an unused module already costs
# nothing. What this list buys is a loud build failure the day someone adds an
# import that would quietly add 60 MB to every user's download.
#
# Excluding a Python module does NOT remove a shared library that another
# collected library links against, so this cannot be used to slim the bundle.
#
# NOT excluded, deliberately:
#   QtNetwork -- the TLS/OpenSSL collection logic hangs off its hook
#   QtDBus    -- the xcb platform plugin links it; without it the plugin fails
#                to load with a near-contentless error
#   QtSvg     -- imageformats/libqsvg is loaded lazily; excluding it does not
#                error, it silently renders blank icons
QT_EXCLUDES = [
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets", "PySide6.QtQuick3D", "PySide6.QtQuickTest",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtSpatialAudio",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtWebView",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtGraphs", "PySide6.QtGraphsWidgets",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtUiTools",
    "PySide6.QtHelp", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtLocation",
    "PySide6.QtSerialPort", "PySide6.QtSerialBus",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors",
    "PySide6.QtTextToSpeech", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtStateMachine", "PySide6.QtNetworkAuth",
    # If any of these enters the graph, PyInstaller's "only one Qt bindings
    # package" guard produces a broken or aborted build.
    "PyQt5", "PyQt6", "PySide2",
]

OTHER_EXCLUDES = [
    "tkinter", "test", "lib2to3", "pydoc_data", "idlelib",
    "distutils", "setuptools", "pip", "wheel", "pkg_resources",
    "numpy", "PIL", "matplotlib", "IPython", "pytest", "_pytest",
]

# Never Tree(repo_root): .cache/ holds fetched upstream templates and whatever a
# developer left there. Enumerate the trees we actually want.
TREE_EXCLUDES = ["__pycache__", "*.pyc", "*.pyo", ".DS_Store", "*.part", ".gitignore"]

# Bundled rather than downloaded. Both are small, and a catalogue inside the
# installer is what lets a machine with no network still reach the hardware
# verdict screen instead of showing an error before the first sentence.
DATA_DIRS = ["catalog", "workflows"]
