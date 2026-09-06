"""The only part of Toolshed that touches the network, the disk or subprocesses.

Everything else -- hardware detection, the catalogue, the planner -- is pure and
testable without any of that. Keeping the impure code in one place is what makes
that possible.
"""
