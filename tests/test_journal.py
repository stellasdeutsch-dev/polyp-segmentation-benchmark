"""Keeps research/journal.yaml valid: required fields, known problem sources, existing files."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
KINDS = {"ours", "published", "library", "data"}


def load():
    return yaml.safe_load((REPO / "research" / "journal.yaml").read_text())["iterations"]


def test_every_iteration_is_complete():
    ids = [it["id"] for it in load()]
    assert len(ids) == len(set(ids)), "iteration ids must be unique"
    for it in load():
        for key in ("id", "stage", "date", "title", "summary", "why", "results", "problems", "files"):
            assert key in it, f"{it.get('id')}: missing {key}"
        assert it["results"], f"{it['id']}: no results"


def test_problems_have_a_known_source_and_a_fix():
    for it in load():
        for p in it["problems"]:
            assert p["kind"] in KINDS, f"{it['id']}: unknown kind {p['kind']}"
            assert p["what"] and p["found"] and p["fix"], f"{it['id']}: incomplete problem"


def test_listed_files_exist():
    for it in load():
        for f in it["files"]:
            assert (REPO / f).exists(), f"{it['id']}: {f} does not exist"
