"""Where a pack's files are fetched from.

Two things are pinned here, and both come from the same mistake: the planner
knowing less about recipes than the catalogue loader does.

`_downloads_for` read only ``.generated.yaml``. The catalogue loader had
learned about ``.authored.yaml`` when hand-written recipes were introduced --
RECIPE_SUFFIXES exists for exactly that -- and the planner had not. So the one
authored recipe we ship resolved to an empty download list: the pack installed,
fetched nothing, reported success, and the first generation failed on a missing
checkpoint with nothing in between pointing at the cause.

The second is the direct URL. Every derived recipe names a Hugging Face repo
and path because that is the shape the upstream templates give us, and the URL
was built from those two fields alone. A community checkpoint is often
published somewhere with no such structure, so there was no way to express one
at all -- the recipe could name it and the planner would silently skip it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolshed.catalog.packs import Pack
from toolshed.planner.plan import PENDING, _downloads_for


@pytest.fixture
def recipes(tmp_path, monkeypatch):
    """A catalogue directory the test controls."""
    (tmp_path / "recipes").mkdir()

    def resource_path(*parts):
        return tmp_path.joinpath(*parts[1:]) if parts[0] == "catalog" else tmp_path

    monkeypatch.setattr("toolshed.resources.resource_path", resource_path)
    return tmp_path / "recipes"


def pack(recipe: str) -> Pack:
    return Pack(id="x.y", modality="Pictures", name="X", blurb="b",
                recipe=recipe, vram_gb_min=8, licence="Some licence")


def write(recipes: Path, name: str, files: str) -> None:
    (recipes / name).write_text(f"files:\n{files}", encoding="utf-8")


HF_ENTRY = """  checkpoint:
    repo: someone/some-model
    path: model.safetensors
    dest: checkpoints
    filename: model.safetensors
"""


class TestBothRecipeKindsAreRead:
    def test_a_generated_recipe_resolves(self, recipes):
        write(recipes, "r.generated.yaml", HF_ENTRY)
        assert len(_downloads_for(pack("r"), Path("/data"))) == 1

    def test_an_authored_recipe_resolves_too(self, recipes):
        """The regression. This returned nothing, and the pack installed
        nothing while reporting success."""
        write(recipes, "r.authored.yaml", HF_ENTRY)
        got = _downloads_for(pack("r"), Path("/data"))
        assert len(got) == 1, "a hand-written recipe must fetch its files"
        assert got[0].url.endswith("/model.safetensors")

    def test_a_recipe_that_does_not_exist_resolves_nothing(self, recipes):
        assert _downloads_for(pack("absent"), Path("/data")) == ()


class TestWhereTheFileComesFrom:
    def test_hugging_face_is_still_built_from_repo_and_path(self, recipes):
        write(recipes, "r.generated.yaml", HF_ENTRY)
        got = _downloads_for(pack("r"), Path("/data"))
        assert got[0].url == (
            "https://huggingface.co/someone/some-model/resolve/main/model.safetensors")

    def test_an_unfrozen_revision_becomes_main(self, recipes):
        write(recipes, "r.generated.yaml", HF_ENTRY + f"    revision: {PENDING}\n")
        assert "/resolve/main/" in _downloads_for(pack("r"), Path("/data"))[0].url

    def test_a_direct_url_is_used_as_given(self, recipes):
        """What a community model needs: somewhere with no repo/path shape."""
        write(recipes, "r.authored.yaml", """  checkpoint:
    url: https://example.invalid/files/community-model.safetensors
    dest: checkpoints
    filename: community-model.safetensors
""")
        got = _downloads_for(pack("r"), Path("/data"))
        assert len(got) == 1
        assert got[0].url == "https://example.invalid/files/community-model.safetensors"
        assert got[0].dest == Path("/data/models/checkpoints")

    def test_a_direct_url_wins_over_repo_and_path(self, recipes):
        """A recipe carrying both is saying "not the derived one"."""
        write(recipes, "r.authored.yaml", HF_ENTRY + "    url: https://example.invalid/m.safetensors\n")
        assert _downloads_for(pack("r"), Path("/data"))[0].url == (
            "https://example.invalid/m.safetensors")

    def test_an_unfilled_url_is_skipped_rather_than_fetched(self, recipes):
        """A recipe that names a URL nobody has supplied must not fall back to
        guessing a Hugging Face one -- that would fetch a different file than
        the recipe is describing."""
        write(recipes, "r.authored.yaml", HF_ENTRY + f"    url: {PENDING}\n")
        assert _downloads_for(pack("r"), Path("/data")) == ()


class TestIncompleteEntries:
    def test_an_entry_with_no_destination_is_skipped(self, recipes):
        write(recipes, "r.generated.yaml", """  checkpoint:
    url: https://example.invalid/m.safetensors
    filename: m.safetensors
""")
        assert _downloads_for(pack("r"), Path("/data")) == ()

    def test_an_unfrozen_repo_is_skipped(self, recipes):
        write(recipes, "r.generated.yaml", f"""  checkpoint:
    repo: {PENDING}
    path: m.safetensors
    dest: checkpoints
    filename: m.safetensors
""")
        assert _downloads_for(pack("r"), Path("/data")) == ()
