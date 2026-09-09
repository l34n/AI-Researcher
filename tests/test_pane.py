"""In-app reader, structured daily brief, and agent handoff."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_researcher.config import Settings
from ai_researcher.db import Database, jdump
from ai_researcher.research.fallback import render_page
from ai_researcher.trends.validate import annotate_citations
from ai_researcher.util import content_hash, iso, local_day, url_hash, utcnow
from ai_researcher.web.app import create_app
from ai_researcher.web.pane import (
    build_frame,
    closer_look_why,
    handoff_markdown,
    metric_line,
)
from ai_researcher.web.queries import body_as_markdown


def _item(
    db,
    *,
    title,
    body="",
    url=None,
    source="src",
    category="research",
    verdict="watch",
    readiness=0.55,
    artifacts=None,
    reasons=None,
    hours_ago=2,
    engagement=0,
):
    now = utcnow()
    url = url or f"https://example.com/{source}/{title[:24]}"
    cur = db.execute(
        "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
        "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (source, f"{source}:{title[:32]}:{url}", url, url, url_hash(url),
         content_hash(title, body), title, "", body or title, iso(now - timedelta(hours=hours_ago)),
         iso(now), engagement, 0, "{}"),
    )
    item_id = cur.lastrowid
    db.execute(
        "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
        "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (item_id, f"Summary of {title}.", category, "[]", "[]", 0.7, "", "", iso(now)),
    )
    db.execute(
        "INSERT OR IGNORE INTO sources (key, name, kind, tier, weight) VALUES (?,?,?,?,?)",
        (source, source, "rss", "lab" if source == "lab" else "news", 1.0),
    )
    db.execute(
        "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
        "readiness, verdict, reasons, artifacts, model, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (item_id, readiness, readiness, readiness, readiness, readiness,
         verdict, jdump(reasons or []), jdump(artifacts or []), "", iso(now)),
    )
    return item_id


def _cluster(db, item_ids, *, label, score, source_count=None, ranking_why=""):
    now = utcnow()
    day = local_day()
    cur = db.execute(
        "INSERT INTO clusters (day, label, summary, category, score, size, source_count, "
        "first_seen, last_seen, entities, ranking_why, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (day, label, f"{label} summary.", "research", score, len(item_ids),
         source_count or len(item_ids), iso(now), iso(now), "[]", ranking_why, iso(now)),
    )
    cid = cur.lastrowid
    for i, item_id in enumerate(item_ids):
        db.execute(
            "INSERT INTO cluster_items (cluster_id, item_id, is_primary) VALUES (?,?,?)",
            (cid, item_id, 1 if i == 0 else 0),
        )
    return cid


def _research(db, item_id, *, title, cluster_id=None):
    now = iso(utcnow())
    adapt = render_page("adapt", {
        "title": title,
        "artifacts": ["https://github.com/acme/tool"],
        "judgment": {
            "quality": 0.8, "practicality": 0.78, "feasibility": 0.7,
            "usefulness": 0.72, "readiness": 0.8, "verdict": "adopt",
        },
    }, {})
    cur = db.execute(
        "INSERT INTO research (cluster_id, item_id, title, status, readiness, verdict, "
        "decision, model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cluster_id, item_id, title, "complete", 0.8, "adopt", "adopt", "", now, now),
    )
    rid = cur.lastrowid
    db.execute(
        "INSERT INTO research_pages (research_id, slug, title, markdown, turn, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (rid, "adapt", "Adapt", adapt, 4, now),
    )
    return rid


class TestCloserLookWhy:
    def test_artifact_and_undercovered(self):
        why = closer_look_why({
            "source_count": 1, "quality": 0.8, "usefulness": 0.5,
            "artifacts": ["https://github.com/acme/x"],
            "category": "research", "reasons": [],
        })
        assert "artifact" in why.lower()
        assert "under-covered" in why.lower() or "research" in why.lower()

    def test_falls_back_to_ranking_why(self):
        why = closer_look_why({
            "source_count": 4, "quality": 0.2, "usefulness": 0.2,
            "artifacts": [], "category": "opinion-analysis",
            "ranking_why": "4 sources · today",
        })
        assert "4 sources" in why


class TestMetricLine:
    def test_compact_chip(self):
        line = metric_line({
            "source_count": 3, "readiness": 0.81, "verdict": "research",
        })
        assert line == "3 src · 0.81 · research"


class TestCitationsAndBody:
    def test_bare_tokens_become_story_links(self):
        md = annotate_citations("Lead [S1] and also [S2].", [
            {"id": 11, "label": "A"}, {"id": 22, "label": "B"},
        ])
        assert "[S1](/story/11)" in md
        assert "[S2](/story/22)" in md

    def test_already_linked_tokens_are_not_doubled(self):
        md = annotate_citations("See [S1](/story/11).", [{"id": 11}])
        assert md.count("/story/11") == 1

    def test_plain_body_becomes_paragraphs(self):
        md = body_as_markdown("First graph.\n\nSecond graph.")
        assert md == "First graph.\n\nSecond graph."


class TestBuildFrame:
    def test_rows_link_in_app_and_closer_has_why(self):
        stories = [
            {"id": 1, "label": "Loud model drop", "summary": "Shipped.",
             "source_count": 5, "readiness": 0.5, "verdict": "watch",
             "category": "model-release", "primary": {"id": 10, "source_name": "HN"}},
            {"id": 2, "label": "Quiet paper", "summary": "A method.",
             "source_count": 1, "quality": 0.8, "usefulness": 0.7,
             "readiness": 0.66, "verdict": "research", "category": "research",
             "artifacts": ["https://arxiv.org/abs/2401.1"], "reasons": [],
             "primary": {"id": 20, "source_name": "arxiv"}},
            {"id": 3, "label": "CLI tool", "summary": "Runs locally.",
             "source_count": 1, "quality": 0.7, "usefulness": 0.65,
             "readiness": 0.64, "verdict": "research", "category": "tooling-oss",
             "artifacts": ["https://github.com/acme/cli"], "reasons": [],
             "primary": {"id": 30, "source_name": "gh"}},
        ]
        ready = [{"id": 9, "item_id": 99, "title": "Local 7B", "decision": "adopt", "readiness": 0.84}]
        frame = build_frame(stories, ready, {
            "markdown": "## The one thing\nLoud drop mattered. [S1]\n",
        })
        assert frame["lead"]["href"] == "/story/1"
        assert "[S1](/story/1)" in frame["lead"]["prose"]
        assert all(row["href"].startswith("/story/") for row in frame["also"])
        assert frame["closer"]
        assert all(row["why"] for row in frame["closer"])
        assert frame["ready"][0]["href"] == "/adapt/9#handoff"

    def test_ready_without_research_id_does_not_404(self):
        frame = build_frame([], [{"item_id": 4, "title": "Orphan", "decision": "spike"}], None)
        assert frame["ready"][0]["href"] == "/read/4"


class TestHandoff:
    def test_pack_has_experiment_and_in_app_links(self):
        md = handoff_markdown({
            "id": 5, "item_id": 8, "cluster_id": 3, "title": "vLLM 0.8",
            "url": "https://github.com/vllm-project/vllm",
            "decision": "adopt", "readiness": 0.84,
            "quality": 0.8, "practicality": 0.8, "feasibility": 0.7, "usefulness": 0.75,
            "reasons": ["names a fetchable artifact"],
            "artifacts": ["https://github.com/vllm-project/vllm"],
            "pages": [{
                "slug": "adapt",
                "markdown": render_page("adapt", {
                    "title": "vLLM 0.8",
                    "artifacts": ["https://github.com/vllm-project/vllm"],
                    "judgment": {
                        "quality": 0.8, "practicality": 0.8, "feasibility": 0.7,
                        "usefulness": 0.75, "readiness": 0.84, "verdict": "adopt",
                    },
                }, {}),
            }],
        })
        assert md.startswith("# Ready to deploy: vLLM 0.8")
        assert "/read/8" in md
        assert "/story/3" in md
        assert "## First experiment" in md
        assert "## Done looks like" in md
        assert "Copy this document into an agent" in md


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("AIR_DATA_DIR", str(data))
    monkeypatch.setenv("AIR_ACCESS_TOKEN", "")
    monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "0")
    settings = Settings(
        data_dir=data,
        access_token="",
        sources_path=Path(__file__).resolve().parents[1] / "config" / "sources.yaml",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    return app, settings


class TestPaneRoutes:
    def test_reader_and_brief_and_handoff(self, app_env):
        app, settings = app_env
        db = Database(settings.db_path)
        lead = _item(db, title="Acme 7B open weights", body="# Acme 7B\n\nWeights on Hugging Face.",
                     source="lab", category="model-release", verdict="watch", readiness=0.5,
                     url="https://example.com/acme-7b")
        paper = _item(db, title="Attention variant paper", body="A new attention paper with code.",
                      source="arxiv", category="research", verdict="research", readiness=0.7,
                      artifacts=["https://github.com/acme/attn"], reasons=["names a fetchable artifact"])
        tool = _item(db, title="Local CLI tool", body="Clone and run README.",
                     source="gh", category="tooling-oss", verdict="adopt", readiness=0.82,
                     artifacts=["https://github.com/acme/cli"])
        c1 = _cluster(db, [lead], label="Acme 7B open weights", score=0.9, ranking_why="1 source · high importance")
        _cluster(db, [paper], label="Attention variant paper", score=0.6, ranking_why="1 source · research")
        c3 = _cluster(db, [tool], label="Local CLI tool", score=0.55, ranking_why="1 source · practitioner-ready")
        rid = _research(db, tool, title="Local CLI tool", cluster_id=c3)
        db.execute(
            "INSERT INTO briefs (day, markdown, model, created_at) VALUES (?,?,?,?)",
            (local_day(),
             "## The one thing\nAcme shipped 7B GGUF weights. [S1]\n\n"
             "## Also today\n- **x** — y\n",
             "stub", iso(utcnow())),
        )
        db.close()

        with TestClient(app) as client:
            home = client.get("/").text
            assert 'href="/story/' in home
            assert "Worth a closer look" in home
            assert "under-covered" in home or "artifact" in home or "research" in home
            assert f'href="/adapt/{rid}#handoff"' in home
            assert 'target="_blank"' not in home.split("Daily Brief")[1].split("Rising")[0]
            # Story titles on the board stay in-app.
            assert f'href="/story/{c1}"' in home

            read = client.get(f"/read/{lead}")
            assert read.status_code == 200
            html = read.text
            assert "Summary of Acme 7B" in html
            assert "Original source" in html
            assert "Weights on Hugging Face" in html
            assert 'target="_blank"' in html  # original only
            assert f'href="/read/{lead}"' not in html.split("<h1>")[1].split("</h1>")[0]
            assert "<h1>Acme 7B open weights</h1>" in html

            story = client.get(f"/story/{c1}")
            assert story.status_code == 200
            assert f'href="/read/{lead}"' in story.text

            feed = client.get("/feed").text
            assert f'href="/read/{lead}"' in feed
            assert "original" in feed

            detail = client.get(f"/adapt/{rid}")
            assert detail.status_code == 200
            assert "Ready to deploy" in detail.text
            assert "Copy for agent" in detail.text
            assert f'href="/read/{tool}"' in detail.text
            assert "First experiment" in detail.text
            assert "id=\"handoff\"" in detail.text

            assert client.get("/read/999").status_code == 404
            assert client.get("/story/999").status_code == 404

    def test_javascript_original_is_not_an_href(self, app_env):
        app, settings = app_env
        db = Database(settings.db_path)
        item_id = _item(
            db, title="Hostile", body="payload", url="javascript:alert(1)",
            verdict="skip", readiness=0.1, category="opinion-analysis",
        )
        db.close()
        with TestClient(app) as client:
            html = client.get(f"/read/{item_id}").text
            assert "javascript:alert(1)" not in html
            assert "<h1>Hostile</h1>" in html
