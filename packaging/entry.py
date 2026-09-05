"""Frozen-build entry point.

Separate from toolshed/__main__.py on purpose. PyInstaller analyses whichever
file the spec names *as* the program's ``__main__``; pointing it at
``toolshed/__main__.py`` would make any future relative import in that file
resolve against a package that does not exist once frozen, breaking only in the
build and never in development. Two lines here removes the trap entirely and
keeps tracebacks naming a file that exists.
"""

from toolshed.__main__ import main

raise SystemExit(main())
