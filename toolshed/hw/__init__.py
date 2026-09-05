"""Hardware detection and the plain-English verdict derived from it."""

from __future__ import annotations

from toolshed.hw.detect import Gpu, HardwareReport, detect
from toolshed.hw.verdict import Verdict, verdict_for

__all__ = ["Gpu", "HardwareReport", "Verdict", "detect", "verdict_for"]
