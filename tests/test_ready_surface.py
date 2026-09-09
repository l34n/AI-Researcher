"""Bugs that only show up once the gate is wired into the dashboard.

The leftover `**` on every story card was one of these. These tests pin the
ones next to it: Decision parsing that hears 'skip' in the done-list,
a Ready chip that hides implementable stories behind louder clusters,
excerpts that still leak markdown, and briefs that go orphan after a
cluster rebuild because they lived on a non-primary member.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from ai_researcher.config import Settings
from ai_researcher.db import Database, jdump
from ai_researcher.research.fallback import render_page
from ai_researcher.research.wiki import DeepResearcher, parse_decision
from ai_researcher.util import content_hash, iso, local_day, url_hash, utcnow
from ai_researcher.web import queries as Q


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield Database(Path(tmp) / "t.db")


def seed_item(
    db,
    *,
    title,
    url=None,
    source="src",
    category="model-release",
    verdict=None,
    readiness=0.5,
    hours_ago=2,
    artifacts=None,
):
    now = utcnow()
    url = url or f"https://example.com/{source}/{title[:24]}"
    cur = db.execute(
        "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
        "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (source, f"{source}:{title[:32]}:{url}", url, url, url_hash(url),
         content_hash(title, ""), title, "", title, iso(now - timedelta(hours=hours_ago)),
         iso(now), 0, 0, "{}"),
    )
    item_id = cur.lastrowid
    db.execute(
        "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
        "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (item_id, title, category, "[]", "[]", 0.6, "", "", iso(now)),
    )
    db.execute(
        "INSERT OR IGNORE INTO sources (key, name, kind, tier, weight) VALUES (?,?,?,?,?)",
        (source, source, "rss", "news", 1.0),
    )
    if verdict is not None:
        db.execute(
            "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
            "readiness, verdict, reasons, artifacts, model, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, readiness, readiness, readiness, readiness, readiness,
             verdict, jdump([]), jdump(artifacts or []), "", iso(now)),
        )
    return item_id


def seed_cluster(db, item_id, *, label, score, day=None, primary=True):
    now = utcnow()
    day = day or local_day()
    cur = db.execute(
        "INSERT INTO clusters (day, label, summary, category, score, size, source_count, "
        "first_seen, last_seen, entities, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (day, label, label, "opinion-analysis", score, 1, 1, iso(now), iso(now), "[]", iso(now)),
    )
    cid = cur.lastrowid
    db.execute(
        "INSERT INTO cluster_items (cluster_id, item_id, is_primary) VALUES (?,?,?)",
        (cid, item_id, 1 if primary else 0),
    )
    return cid


def seed_research(db, item_id, *, title, verdict, decision, readiness, adapt_md, cluster_id=None):
    now = iso(utcnow())
    cur = db.execute(
        "INSERT INTO research (cluster_id, item_id, title, status, readiness, verdict, "
        "decision, model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cluster_id, item_id, title, "complete", readiness, verdict, decision, "", now, now),
    )
    rid = cur.lastrowid
    db.execute(
        "INSERT INTO research_pages (research_id, slug, title, markdown, turn, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (rid, "adapt", "Adapt", adapt_md, 4, now),
    )
    return rid


class TestParseDecisionHearsTheHeading:
    def test_fallback_adapt_is_not_skip(self):
        """The template says 'explicit skip' in Done looks like. That is not
        the decision — last-word-wins used to file every fallback as skip."""
        md = render_page("adapt", {
            "title": "Local 7B",
            "artifacts": ["https://github.com/acme/x"],
            "judgment": {
                "quality": 0.72, "practicality": 0.70, "feasibility": 0.68,
                "usefulness": 0.66, "readiness": 0.69, "verdict": "research",
            },
        }, {})
        assert "explicit skip" in md
        assert "file this as watch" in md
        assert parse_decision(md) == "spike"

    def test_final_verdict_still_wins_over_decision(self):
        text = "## Decision\n**adopt** — try it.\n## Final verdict\n**watch** because the repo 404s."
        assert parse_decision(text) == "watch"

    def test_body_skip_cannot_overwrite_a_bold_decision(self):
        text = (
            "## Decision\n**adopt** — pull the GGUF this week.\n"
            "## Done looks like\nEither a follow-up spike or an explicit skip."
        )
        assert parse_decision(text) == "adopt"

    def test_bold_call_without_a_heading(self):
        assert parse_decision("I would **watch** this until the repo is public.") == "watch"


class TestAdaptExcerpt:
    def test_single_hash_decision_heading(self):
        excerpt = Q.adapt_excerpt("# Decision\n**adopt** — serve the Q4 this week.")
        assert excerpt.startswith("adopt —")
        assert "*" not in excerpt

    def test_underscore_emphasis(self):
        excerpt = Q.adapt_excerpt("## Decision\n_spike_ — clone the repo.")
        assert excerpt.startswith("spike —")
        assert "_" not in excerpt

    def test_link_and_backticks(self):
        excerpt = Q.adapt_excerpt(
            "## Decision\n**spike** — run `llama-server` from [the card](https://hf.co/x)."
        )
        assert "llama-server" in excerpt
        assert "https://" not in excerpt
        assert "*" not in excerpt
        assert "`" not in excerpt

    def test_empty_and_missing_heading(self):
        assert Q.adapt_excerpt("") == ""
        assert Q.adapt_excerpt("# Adapt\n## Who\nPractitioners.") == ""


class TestReadyChipDoesNotHideTheWork:
    def test_ready_filter_survives_louder_watch_clusters(self, db):
        """Ready used to LIMIT by cluster score, then filter. Five viral
        teases would hide the one implementable story."""
        for i in range(5):
            item_id = seed_item(
                db, title=f"tease {i}", source=f"t{i}",
                verdict="watch", readiness=0.30, category="product-launch",
            )
            seed_cluster(db, item_id, label=f"tease {i}", score=0.99 - i * 0.01)
        ready_id = seed_item(
            db, title="Local 7B open weights", source="lab",
            verdict="research", readiness=0.74, category="model-release",
        )
        seed_cluster(db, ready_id, label="Local 7B open weights", score=0.10)

        shown = Q.top_stories(db, day=local_day(), limit=3, ready=True)
        titles = [s["label"] for s in shown]
        assert "Local 7B open weights" in titles
        assert all(
            s["verdict"] in ("research", "adopt") or s["research_id"]
            for s in shown
        )

    def test_ready_view_sorts_by_readiness_not_cluster_score(self, db):
        low = seed_item(db, title="quiet spike", source="a", verdict="research", readiness=0.64)
        high = seed_item(db, title="ship this", source="b", verdict="adopt", readiness=0.88)
        seed_cluster(db, low, label="quiet spike", score=0.95)
        seed_cluster(db, high, label="ship this", score=0.20)

        shown = Q.top_stories(db, day=local_day(), limit=10, ready=True)
        titles = [s["label"] for s in shown]
        assert titles.index("ship this") < titles.index("quiet spike")

    def test_ready_sidebar_omits_skip_briefs(self, db):
        skip_id = seed_item(db, title="hype", source="a", verdict="skip", readiness=0.20)
        adopt_id = seed_item(db, title="ship", source="b", verdict="adopt", readiness=0.86)
        seed_research(
            db, skip_id, title="hype", verdict="skip", decision="skip",
            readiness=0.20, adapt_md="## Decision\n**skip** — no artifact.",
        )
        seed_research(
            db, adopt_id, title="ship", verdict="adopt", decision="adopt",
            readiness=0.86, adapt_md="## Decision\n**adopt** — run it this week.",
        )
        ready = Q.ready_briefs(db, limit=8)
        titles = [b["title"] for b in ready]
        assert "ship" in titles
        assert "hype" not in titles


class TestRelinkNonPrimary:
    def test_brief_on_a_member_reattaches_after_rebuild(self, db):
        primary = seed_item(db, title="Demo reel", source="verge", verdict="watch", readiness=0.29)
        member = seed_item(
            db, title="Bytebot compose file", source="gh",
            verdict="research", readiness=0.77,
            artifacts=["https://github.com/bytebot-ai/bytebot"],
        )
        cid = seed_cluster(db, primary, label="Computer use", score=0.70)
        db.execute(
            "INSERT INTO cluster_items (cluster_id, item_id, is_primary) VALUES (?,?,0)",
            (cid, member),
        )
        rid = seed_research(
            db, member, title="Bytebot compose file", verdict="research",
            decision="spike", readiness=0.77, cluster_id=None,
            adapt_md="## Decision\n**spike** — compose up and run the Playwright task.",
        )
        n = DeepResearcher(Settings(), db, client=None).relink_clusters()
        assert n == 1
        assert db.scalar("SELECT cluster_id FROM research WHERE id=?", (rid,)) == cid

        stories = Q.top_stories(db, day=local_day(), limit=10)
        story = next(s for s in stories if s["id"] == cid)
        assert story["research_id"] == rid
        assert story["readiness"] == 0.77
        assert "compose up" in story["adapt_excerpt"]
        assert "*" not in story["adapt_excerpt"]


class TestFallbackStoresTheDecision:
    def test_templated_wiki_files_spike_not_skip(self, db):
        from ai_researcher.enrich.ollama import OllamaClient

        item_id = seed_item(
            db, title="Local 7B open weights on GitHub", source="lab",
            verdict="research", readiness=0.72,
            artifacts=["https://github.com/acme/local-7b"],
        )
        settings = Settings()
        settings.research_budget = 2
        settings.research_threshold = 0.60
        result = asyncio.run(DeepResearcher(settings, db, OllamaClient(settings)).run(limit=1))
        assert result["researched"] == 1
        row = db.one("SELECT decision, verdict FROM research WHERE item_id=?", (item_id,))
        assert row["decision"] == "spike"
        assert row["verdict"] in ("research", "adopt")
