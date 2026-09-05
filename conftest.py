"""Root conftest.

Its presence puts the repository root on sys.path under pytest's default
prepend import mode, so `import toolshed` works in a source checkout without
installing the package. That matters here: pyproject requires Python 3.12-3.13,
so a developer on another interpreter can still run the pure unit suite.
"""
