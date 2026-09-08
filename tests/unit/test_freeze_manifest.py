"""Tests for freezing a recipe's facts.

This is the tool the whole catalogue waits on: nothing ships while a
PENDING_FREEZE remains, and every recipe carries between 18 and 48 of them.

What matters here is not that it fills fields in. It is that it refuses to fill
them in wrongly. A guessed sha256 does not fail at freeze time -- it fails at
the end of somebody's 20 GB download, with a message about corruption that
blames their network. So every test below is really the same test: when the API
does not say, the value stays PENDING_FREEZE.

No network. The Hugging Face seam is replaced with recorded answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from freeze_manifest import (  # noqa: E402
    PENDING,
    Report,
    commit_sha,
    file_facts,
    freeze_recipe,
    gated_flag,
    licence_id,
    pending_in,
)

SHA = "a" * 40
OID = "b" * 64


class FakeHF:
    """Recorded answers, plus a note of what was asked."""

    def __init__(self, info: dict | None = None, paths: list[dict] | None = None):
        self._info = info if info is not None else {
            "sha": SHA, "gated": False, "cardData": {"license": "apache-2.0"}}
        self._paths = paths if paths is not None else [
            {"path": "model.safetensors", "size": 7,
             "lfs": {"oid": OID, "size": 7_000_000_000}}]
        self.info_calls: list[str] = []
        self.paths_calls: list[tuple[str, str, list[str]]] = []

    def model_info(self, repo):
        self.info_calls.append(repo)
        return self._info

    def paths_info(self, repo, revision, paths):
        self.paths_calls.append((repo, revision, list(paths)))
        return self._paths


def recipe(**overrides) -> dict:
    doc = {
        "schema_version": 1,
        "id": "image.example",
        "estimated_download_bytes": PENDING,
        "files": {
            "model": {
                "source": "huggingface",
                "repo": "someone/some-model",
                "revision": PENDING,
                "path": "model.safetensors",
                "dest": "checkpoints",
                "filename": "model.safetensors",
                "sha256": PENDING,
                "size_bytes": PENDING,
                "gated": PENDING,
                "licence": PENDING,
            }
        },
    }
    doc.update(overrides)
    return doc


class TestReadingWhatTheApiSaid:
    def test_a_commit_sha_is_taken(self):
        assert commit_sha({"sha": SHA}) == SHA

    @pytest.mark.parametrize("info", [
        {},                              # no key
        {"sha": "main"},                 # a branch, which is what we refuse
        {"sha": "abc123"},               # too short
        {"sha": "z" * 40},               # not hex
    ])
    def test_anything_that_is_not_a_commit_stays_pending(self, info):
        """revision must be immutable, or the hash guards nothing."""
        assert commit_sha(info) == PENDING

    def test_an_ungated_repo_records_false_rather_than_pending(self):
        """`False` is a fact. A truthiness test would lose it, and the whole
        "no account needed on the happy path" check is built on it."""
        assert gated_flag({"gated": False}) is False

    def test_a_gate_records_what_kind_it_is(self):
        assert gated_flag({"gated": "auto"}) == "auto"

    def test_a_missing_gate_key_is_unknown_not_ungated(self):
        assert gated_flag({}) == PENDING

    def test_the_licence_comes_from_the_card(self):
        assert licence_id({"cardData": {"license": "apache-2.0"}}) == "apache-2.0"

    def test_or_from_a_tag_when_the_card_has_none(self):
        assert licence_id({"tags": ["diffusers", "license:mit"]}) == "mit"

    def test_and_stays_pending_when_nothing_declares_one(self):
        assert licence_id({"tags": ["diffusers"]}) == PENDING


class TestFileFacts:
    def test_the_lfs_oid_is_the_sha256(self):
        sha, size = file_facts({"lfs": {"oid": OID, "size": 12}})
        assert sha == OID and size == 12

    def test_a_file_with_no_lfs_record_has_no_hash_to_take(self):
        """Small files are stored inline and have no oid. Inventing one is the
        exact failure this tool exists to avoid."""
        sha, size = file_facts({"size": 40})
        assert sha == PENDING and size == 40

    def test_a_malformed_oid_is_refused(self):
        sha, _ = file_facts({"lfs": {"oid": "nonsense", "size": 1}})
        assert sha == PENDING


class TestFreezingARecipe:
    def test_the_four_facts_are_written(self):
        doc = recipe()
        report = freeze_recipe(doc, FakeHF())
        entry = doc["files"]["model"]
        assert entry["revision"] == SHA
        assert entry["sha256"] == OID
        assert entry["size_bytes"] == 7_000_000_000
        assert entry["gated"] is False
        assert entry["licence"] == "apache-2.0"
        assert report.changed and not report.problems

    def test_the_download_total_is_the_sum_once_every_part_is_known(self):
        doc = recipe()
        freeze_recipe(doc, FakeHF())
        assert doc["estimated_download_bytes"] == 7_000_000_000

    def test_the_total_is_left_alone_while_any_size_is_unknown(self):
        """A total built on one known file and one guess is a wrong number on
        the confirmation screen, which is worse than an honest unknown."""
        doc = recipe()
        doc["files"]["other"] = dict(doc["files"]["model"], path="missing.safetensors")
        freeze_recipe(doc, FakeHF())
        assert doc["estimated_download_bytes"] == PENDING

    def test_one_call_per_repo_not_per_file(self):
        doc = recipe()
        doc["files"]["other"] = dict(doc["files"]["model"], path="other.safetensors")
        hf = FakeHF()
        freeze_recipe(doc, hf)
        assert hf.info_calls == ["someone/some-model"]

    def test_paths_info_is_asked_at_the_commit_never_a_branch(self):
        hf = FakeHF()
        freeze_recipe(recipe(), hf)
        assert hf.paths_calls and hf.paths_calls[0][1] == SHA

    def test_a_repo_whose_commit_is_unreadable_freezes_nothing_about_its_files(self):
        doc = recipe()
        freeze_recipe(doc, FakeHF(info={"sha": "main", "gated": False}))
        entry = doc["files"]["model"]
        assert entry["revision"] == PENDING
        assert entry["sha256"] == PENDING, "no commit means no trustworthy hash"

    def test_a_file_the_api_does_not_list_stays_pending(self):
        doc = recipe()
        report = freeze_recipe(doc, FakeHF(paths=[]))
        assert doc["files"]["model"]["sha256"] == PENDING
        assert "model.sha256" in report.still_pending

    def test_a_repo_that_cannot_be_reached_is_a_problem_not_a_guess(self):
        from freeze_manifest import FreezeError

        class Broken(FakeHF):
            def model_info(self, repo):
                raise FreezeError("cannot reach it")

        doc = recipe()
        report = freeze_recipe(doc, Broken())
        assert report.problems
        assert doc["files"]["model"]["revision"] == PENDING

    def test_a_recipe_with_no_repo_says_so(self):
        doc = recipe()
        doc["files"]["model"]["repo"] = PENDING
        report = freeze_recipe(doc, FakeHF())
        assert any("repo" in p for p in report.problems)

    def test_human_fields_are_never_written(self):
        """name, blurb and modality are somebody's judgement, not a lookup."""
        doc = recipe(name=PENDING, blurb=PENDING, modality=PENDING)
        freeze_recipe(doc, FakeHF())
        assert doc["name"] == PENDING
        assert doc["blurb"] == PENDING
        assert doc["modality"] == PENDING

    def test_freezing_twice_changes_nothing_the_second_time(self):
        doc = recipe()
        freeze_recipe(doc, FakeHF())
        second = freeze_recipe(doc, FakeHF())
        assert not second.changed


class TestFindingWhatIsLeft:
    def test_every_pending_is_reported_by_path(self):
        found = pending_in(recipe())
        assert "files.model.sha256" in found
        assert "estimated_download_bytes" in found

    def test_a_frozen_recipe_reports_none(self):
        doc = recipe()
        freeze_recipe(doc, FakeHF())
        assert pending_in(doc) == []

    def test_it_looks_inside_lists(self):
        assert pending_in({"variants": [{"id": PENDING}]}) == ["variants[0].id"]

    def test_an_empty_report_is_not_a_change(self):
        assert not Report().changed


class TestARefreezeCannotPairANewCommitWithAnOldHash:
    """The dangerous failure: revision advances, sha256 does not, and every
    install then downloads a file that fails verification with a message
    blaming the user's connection."""

    def frozen(self) -> dict:
        doc = recipe()
        freeze_recipe(doc, FakeHF())
        return doc

    def test_both_move_together_on_a_real_refreeze(self):
        doc = self.frozen()
        NEW_SHA, NEW_OID = "c" * 40, "d" * 64
        report = freeze_recipe(doc, FakeHF(
            info={"sha": NEW_SHA, "gated": False, "cardData": {"license": "apache-2.0"}},
            paths=[{"path": "model.safetensors", "lfs": {"oid": NEW_OID, "size": 9}}]))
        entry = doc["files"]["model"]
        assert entry["revision"] == NEW_SHA
        assert entry["sha256"] == NEW_OID, "the hash must come from the same commit"
        assert not report.problems

    def test_a_moved_commit_with_no_new_hash_clears_both(self):
        doc = self.frozen()
        report = freeze_recipe(doc, FakeHF(
            info={"sha": "c" * 40, "gated": False}, paths=[]))
        entry = doc["files"]["model"]
        assert entry["revision"] == PENDING
        assert entry["sha256"] == PENDING, "an old hash must never survive a new commit"
        assert entry["size_bytes"] == PENDING
        assert any("cleared" in p for p in report.problems)

    def test_an_unchanged_commit_is_not_disturbed(self):
        doc = self.frozen()
        before = dict(doc["files"]["model"])
        freeze_recipe(doc, FakeHF())
        assert doc["files"]["model"] == before

    def test_a_frozen_fact_that_now_reads_differently_is_reported_not_replaced(self):
        """The tool cannot tell an upstream change from a mistake, so it says
        so rather than choosing."""
        doc = self.frozen()
        report = freeze_recipe(doc, FakeHF(
            info={"sha": SHA, "gated": False, "cardData": {"license": "mit"}}))
        assert doc["files"]["model"]["licence"] == "apache-2.0"
        assert any("already frozen" in p for p in report.problems)


class TestRewritingKeepsTheFile:
    def test_indented_comments_survive(self, tmp_path):
        """The adult recipe carries 35 lines of indented rationale. A
        yaml.safe_dump round trip deletes every one of them, and the freeze
        command is the first thing anyone runs against it."""
        from freeze_manifest import _rewritten

        text = (
            "# top comment\n"
            "id: image.example\n"
            "estimated_download_bytes: PENDING_FREEZE\n"
            "files:\n"
            "  model:\n"
            "    # why this model and not another\n"
            "    #   criterion 2: the author's own repo\n"
            "    repo: someone/some-model\n"
            "    revision: PENDING_FREEZE\n"
            "    sha256: PENDING_FREEZE\n"
        )
        doc = {"estimated_download_bytes": 7,
               "files": {"model": {"repo": "someone/some-model",
                                   "revision": SHA, "sha256": OID}}}
        out = _rewritten(text, doc)
        assert "# why this model and not another" in out
        assert "#   criterion 2: the author's own repo" in out
        assert "# top comment" in out
        assert f"    revision: {SHA}" in out
        assert f"    sha256: {OID}" in out
        assert "estimated_download_bytes: 7" in out

    def test_the_real_adult_recipe_keeps_its_rationale(self, tmp_path):
        """Not a synthetic file: the one the documented command targets."""
        import yaml
        from freeze_manifest import _rewritten

        from toolshed import resources

        path = (resources.resource_path("catalog") / "recipes"
                / "image_sdxl_adult.authored.yaml")
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        doc["files"]["checkpoint"]["revision"] = SHA
        doc["files"]["checkpoint"]["sha256"] = OID
        out = _rewritten(text, doc)

        before = [ln for ln in text.splitlines() if ln.strip().startswith("#")]
        after = [ln for ln in out.splitlines() if ln.strip().startswith("#")]
        assert before == after, "every comment line must survive a freeze"
        assert f"    revision: {SHA}" in out
        assert yaml.safe_load(out)["files"]["checkpoint"]["sha256"] == OID

    def test_booleans_render_as_yaml_not_python(self):
        from freeze_manifest import _rewritten

        text = "files:\n  m:\n    gated: PENDING_FREEZE\n"
        out = _rewritten(text, {"files": {"m": {"gated": False}}})
        assert "gated: false" in out and "False" not in out


class TestNamedRecipesWin:
    def test_check_respects_the_files_it_was_given(self):
        """Scanning the whole catalogue when someone asked about one file is a
        small lie that reads as a much bigger answer."""
        import argparse

        from freeze_manifest import recipe_paths

        args = argparse.Namespace(recipes=["catalog/recipes/one.yaml"],
                                  all=False, check=True)
        assert recipe_paths(args) == [Path("catalog/recipes/one.yaml")]
