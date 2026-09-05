"""Toolshed -- local AI, set up for you."""

from __future__ import annotations

# The single source of version truth. hatchling reads this (see pyproject.toml
# [tool.hatch.version]), and the build workflows read it to name artifacts, to
# stamp the Windows VERSIONINFO resource and to set the Inno Setup AppVersion.
# Bumping it here bumps it everywhere; there is nowhere else to forget.
__version__ = "0.0.1"

APP_NAME = "Toolshed"

__all__ = ["APP_NAME", "__version__"]
