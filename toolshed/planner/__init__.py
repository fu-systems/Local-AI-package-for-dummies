"""Turning hardware and a selection into an ordered, executable plan.

Everything here is a pure function of its inputs. No network, no filesystem, no
subprocesses. That is what lets the whole install policy -- including for cards
nobody on the project owns -- be proven from recorded fixtures on a machine
with no GPU.
"""

from __future__ import annotations

from toolshed.planner.plan import InstallPlan, Step, build_plan
from toolshed.planner.torchsel import TorchChoice, choose_torch

__all__ = ["InstallPlan", "Step", "TorchChoice", "build_plan", "choose_torch"]
