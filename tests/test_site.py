"""Keeps the explainer site honest: local links resolve, every page lists every stage page, and
the stage 3 page data agrees with the reports it is built from."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
PAGES = sorted(SITE.glob("*.html"))


def local_refs(page: Path) -> list[str]:
    refs = re.findall(r'(?:href|src)="([^"#]+)', page.read_text())
    return [r.split("?")[0] for r in refs if not re.match(r"^[a-z]+:", r)]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_local_links_resolve(page):
    missing = [r for r in local_refs(page) if not (page.parent / r).resolve().exists()]
    assert not missing, f"{page.name}: {missing}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_menu_lists_every_stage_page(page):
    stage_pages = sorted(p.name for p in SITE.glob("stage-*.html"))
    menu = re.search(r'<ul id="site-menu".*?</ul>', page.read_text(), re.S).group(0)
    assert all(f'href="{s}"' in menu for s in stage_pages), page.name


def test_stage3_page_data_matches_the_reports():
    spec = importlib.util.spec_from_file_location("build_site_data", REPO / "scripts" / "build_site_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.stage3_page()
    if data is None:
        pytest.skip("stage 3 reports not present")
    run = json.loads((REPO / "reports" / "stage3_pranet_weights.json").read_text())
    mps = next(r for r in run["runs"] if r["device"] == "mps")
    for page_row, report_row in zip(data["sets"], mps["sets"], strict=True):
        assert page_row["dataset"] == report_row["dataset"]
        assert page_row["ours"] == report_row["mdice"]
    ours = {r["image"]: r["mdice"] for r in mps["sets"][0]["per_image"]}
    assert all(abs(img["pillow"] - ours[img["image"]]) < 1e-5 for img in data["kvasir"]["images"])
    assert data["published_within"] <= data["published_rows"]
