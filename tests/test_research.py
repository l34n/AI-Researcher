"""Karpathy wiki: fallback pages, parsers, and the gated runner."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from ai_researcher.config import Settings
from ai_researcher.db import Database, jdump
from ai_researcher.enrich.judge import Judge
from ai_researcher.enrich.ollama import OllamaClient
from ai_researcher.research.fallback import render_page
from ai_researcher.research.schema import TURNS, adapt_complete, index_markdown
from ai_researcher.research.wiki import (
    DeepResearcher,
    decision_to_verdict,
    parse_decision,
    parse_scores,
)
from ai_researcher.util import content_hash, iso, url_hash, utcnow


def _candidate(**kw):
    base = {
        "title": "Open weights for a 7B local model",
        "category": "model-release",
        "entities": ["Acme"],
        "source_count": 2,
        "artifacts": ["https://github.com/acme/model"],
        "reasons": ["names a fetchable artifact", "looks runnable on commodity hardware"],
        "judgment": {
            "quality": 0.72, "practicality": 0.70, "feasibility": 0.68,
            "usefulness": 0.66, "readiness": 0.69, "verdict": "research",
            "reasons": ["names a fetchable artifact"],
        },
        "items": [{
            "title": "Open weights for a 7B local model",
            "body": "Weights and code. Runs on RTX.",
            "summary": "Acme released 7B open weights with a GitHub repo.",
            "why": "local inference just got cheaper",
            "url": "https://acme.example/blog",
            "source_name": "Acme",
        }],
    }
    base.update(kw)
    return base


class TestFallbackPages:
    @pytest.mark.parametrize("slug,heading", [
        ("source", "# Source"),
        ("claims", "# Claims"),
        ("critique", "# Critique"),
        ("adapt", "# Adapt"),
        ("lint", "# Lint"),
    ])
    def test_each_turn_has_its_heading(self, slug, heading):
        pages = {}
        md = render_page(slug, _candidate(), pages)
        assert md.startswith(heading)
        pages[slug] = md

    def test_adapt_carries_a_decision(self):
        md = render_page("adapt", _candidate(), {})
        assert "**spike**" in md or "**adopt**" in md

    def test_lint_mentions_missing_pages(self):
        md = render_page("lint", _candidate(), {})
        assert "source" in md.lower()


class TestParsers:
    def test_parse_decision_last_wins(self):
        text = "## Decision\n**adopt** — try it.\n## Final verdict\n**watch** because the repo 404s."
        assert parse_decision(text) == "watch"

    def test_parse_scores(self):
        scores = parse_scores("scores: Q=0.71 P=0.60 F=0.55 U=0.80")
        assert scores == {"quality": 0.71, "practicality": 0.60, "feasibility": 0.55, "usefulness": 0.80}

    def test_parse_scores_rejects_junk(self):
        assert parse_scores("no scores here") is None

    def test_decision_cannot_jump_skip_to_adopt(self):
        assert decision_to_verdict("adopt", "skip") == "skip"

    def test_decision_may_downgrade(self):
        assert decision_to_verdict("skip", "research") == "skip"

    def test_index_lists_filed_pages(self):
        idx = index_markdown({"source": "# Source\nFacts."})
        assert "`source`" in idx


class TestTurns:
    def test_five_turns_in_karpathy_order(self):
        assert [t["slug"] for t in TURNS] == ["source", "claims", "critique", "adapt", "lint"]


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield Database(Path(tmp) / "t.db")


def add_ready_item(db, *, title="Local 7B open weights on GitHub", verdict="research", readiness=0.72, source="lab"):
    now = utcnow()
    published = iso(now - timedelta(hours=2))
    url = "https://github.com/acme/local-7b"
    cur = db.execute(
        "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
        "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (source, f"{source}:{title[:32]}:{url}", url, url, url_hash(url), content_hash(title, "body"),
         title, "", "Open weights. pip install. Runs locally on RTX with GGUF.",
         published, iso(now), 0, 0, "{}"),
    )
    item_id = cur.lastrowid
    db.execute(
        "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
        "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (item_id, title, "model-release", jdump(["Acme"]), "[]", 0.8, "", "", iso(now)),
    )
    db.execute("INSERT OR IGNORE INTO sources (key, name, kind, tier) VALUES (?,?,?,?)",
               (source, source, "rss", "lab"))
    db.execute(
        "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
        "readiness, verdict, reasons, artifacts, model, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (item_id, 0.74, 0.70, 0.66, 0.70, readiness, verdict,
         jdump(["names a fetchable artifact"]), jdump([url]), "", iso(now)),
    )
    return item_id


class TestDeepResearcherFallback:
    def test_writes_five_pages_without_a_model(self, db):
        item_id = add_ready_item(db)
        settings = Settings()
        settings.research_budget = 2
        settings.research_threshold = 0.60
        client = OllamaClient(settings)
        # Probe was never called; chat_model is empty so generate_text is a no-op.
        researcher = DeepResearcher(settings, db, client)
        result = asyncio.run(researcher.run(limit=1))
        assert result["researched"] == 1
        assert result["fallback"] == 1
        assert db.scalar("SELECT COUNT(*) FROM research WHERE item_id=?", (item_id,)) == 1
        slugs = [r["slug"] for r in db.query(
            "SELECT slug FROM research_pages ORDER BY turn"
        )]
        assert slugs == ["source", "claims", "critique", "adapt", "lint"]
        adapt = db.scalar(
            "SELECT markdown FROM research_pages WHERE slug='adapt'"
        )
        assert adapt.startswith("# Adapt")
        from ai_researcher.web.queries import get_research
        rid = db.scalar("SELECT id FROM research WHERE item_id=?", (item_id,))
        detail = get_research(db, rid)
        assert detail["pages"][0]["slug"] == "adapt"
        assert detail["excerpt"]

    def test_skips_items_below_the_gate(self, db):
        add_ready_item(db, verdict="watch", readiness=0.40)
        settings = Settings()
        settings.research_threshold = 0.62
        researcher = DeepResearcher(settings, db, OllamaClient(settings))
        result = asyncio.run(researcher.run(limit=4))
        assert result["researched"] == 0
        assert db.scalar("SELECT COUNT(*) FROM research") == 0

    def test_does_not_rewrite_a_complete_brief(self, db):
        item_id = add_ready_item(db)
        settings = Settings()
        researcher = DeepResearcher(settings, db, OllamaClient(settings))
        first = asyncio.run(researcher.run(limit=1))
        second = asyncio.run(researcher.run(limit=1))
        assert first["researched"] == 1
        assert second["researched"] == 0
        assert db.scalar("SELECT COUNT(*) FROM research WHERE item_id=?", (item_id,)) == 1


class TestJudgePass:
    def test_heuristic_pass_scores_unenriched_shape(self, db):
        add_ready_item(db)
        # Delete the pre-seeded judgment so the heuristic pass has work to do.
        db.execute("DELETE FROM judgments")
        settings = Settings()
        judge = Judge(settings, db, OllamaClient(settings))
        result = asyncio.run(judge.run(limit=0))
        assert result["judged"] == 1
        row = db.one("SELECT verdict, readiness FROM judgments")
        assert row["readiness"] > 0
        assert row["verdict"] in ("skip", "watch", "research", "adopt")

    def test_force_rewrites_an_existing_brief(self, db):
        add_ready_item(db)
        settings = Settings()
        researcher = DeepResearcher(settings, db, OllamaClient(settings))
        asyncio.run(researcher.run(limit=1))
        first_updated = db.scalar("SELECT updated_at FROM research")
        result = asyncio.run(researcher.run(limit=1, force=True))
        assert result["researched"] == 1
        assert db.scalar("SELECT COUNT(*) FROM research") == 1
        assert db.scalar("SELECT COUNT(*) FROM research_pages") == 5
        assert db.scalar("SELECT updated_at FROM research") >= first_updated


class TestPremiumWiki:
    def test_every_turn_requests_premium_and_unslops(self, db):
        add_ready_item(db)
        settings = Settings()
        settings.research_budget = 1
        client = _PremiumChat()
        result = asyncio.run(DeepResearcher(settings, db, client).run(limit=1))
        assert result["llm"] == 1
        assert result["model"] == "stub:premium"
        assert client.text_kwargs
        assert all(kw.get("premium") is True for kw in client.text_kwargs)
        assert len(client.text_kwargs) == 5
        adapt = db.scalar("SELECT markdown FROM research_pages WHERE slug='adapt'")
        assert "—" not in adapt
        assert "pivotal" not in adapt.lower()
        assert db.scalar("SELECT model FROM research") == "stub:premium"

    def test_every_turn_is_tagged_with_the_research_role(self, db):
        """``AIR_RESEARCH_MODEL`` is only honoured when the call names its role.

        Without ``role="research"`` the router falls back to the premium band's
        cloud model and the per-role override is silently ignored.
        """
        add_ready_item(db)
        settings = Settings()
        settings.research_budget = 1
        client = _RoleChat()
        result = asyncio.run(DeepResearcher(settings, db, client).run(limit=1))
        assert result["llm"] == 1
        assert client.text_kwargs
        assert all(kw.get("role") == "research" for kw in client.text_kwargs)
        assert result["model"] == "stub:research"
        assert db.scalar("SELECT model FROM research") == "stub:research"


class _PremiumChat:
    available = True
    chat_model = "stub:workhorse"
    settings = Settings()

    def __init__(self):
        self.text_kwargs = []

    async def probe(self):
        return True

    async def generate_text(self, *args, **kwargs):
        self.text_kwargs.append(kwargs)
        return (
            "# Adapt\n"
            "## Decision\n"
            "**spike** — this is a pivotal moment for local GGUF.\n"
            "In order to run it, clone the repo and pull Q4_K_M.\n"
            + "Weights, license, and RTX notes for a practitioner. " * 6
        )

    def model_for(self, *, premium=False, role=""):
        return "stub:premium" if premium else self.chat_model


class _RoleChat(_PremiumChat):
    """Premium stub whose model name reveals which role the caller asked for."""

    def model_for(self, *, premium=False, role=""):
        if premium and role:
            return f"stub:{role}"
        return "stub:premium" if premium else self.chat_model


class TestAdaptComplete:
    def test_requires_experiment_fields(self):
        assert not adapt_complete("# Adapt\n## Decision\nadopt\n")
        assert adapt_complete(
            "# Adapt\n## Prerequisites\nx\n## First week\n1.\n"
            "## Risks\ny\n## Done looks like\nz\n"
        )

