"""Tests for bundle-relative resource resolution."""

from __future__ import annotations

import sys

import pytest

from toolshed import resources


def test_not_frozen_in_a_source_checkout():
    assert resources.is_frozen() is False


def test_bundle_root_is_the_repository_root_when_not_frozen():
    root = resources.bundle_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "toolshed" / "__init__.py").is_file()


def test_resource_path_finds_the_catalogue():
    """The selftest asserts on this path, so if it moves, builds must fail."""
    recipes = resources.resource_path("catalog", "recipes")
    assert recipes.is_dir()
    assert list(recipes.glob("*.yaml")), "no generated recipes found"


def test_frozen_layout_uses_meipass(monkeypatch, tmp_path):
    """In a PyInstaller onedir build since 6.0, sys._MEIPASS is <app>/_internal
    and NOT the directory holding the executable. Nothing may assume the old
    flat layout."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resources.is_frozen() is True
    assert resources.bundle_root() == tmp_path
    assert resources.resource_path("catalog", "recipes") == tmp_path / "catalog" / "recipes"


@pytest.mark.parametrize("parts", [("catalog",), ("catalog", "recipes"), ("workflows",)])
def test_resource_path_joins_without_escaping_the_root(parts):
    assert resources.bundle_root() in resources.resource_path(*parts).parents or \
        resources.resource_path(*parts) == resources.bundle_root() / parts[0]
