"""Deterministic unslop pass — the safety net after the prompt rule."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from ai_researcher.config import Settings
from ai_researcher.db import Database, jdump
from ai_researcher.enrich.analyze import SYSTEM as ENRICH_SYSTEM
from ai_researcher.enrich.unslop import UNSLOP_RULE, unslop_text
from ai_researcher.research.schema import SCHEMA
from ai_researcher.trends.brief import SYSTEM as BRIEF_SYSTEM, _clean, generate_brief
from ai_researcher.util import content_hash, iso, local_day, url_hash, utcnow


class TestUnslopRule:
    def test_rule_is_wired_into_long_form_prompts(self):
        assert "em dash" in UNSLOP_RULE.lower() or "em dashes" in UNSLOP_RULE.lower()
        assert "delve" in UNSLOP_RULE
        assert UNSLOP_RULE in SCHEMA
        assert UNSLOP_RULE in BRIEF_SYSTEM
        assert UNSLOP_RULE in ENRICH_SYSTEM


class TestUnslopText:
    def test_em_dash_becomes_a_comma(self):
        assert "—" not in unslop_text("Adopt — serve the Q4 this week.")
        assert "serve the Q4" in unslop_text("Adopt — serve the Q4 this week.")

    def test_en_dash_becomes_a_hyphen(self):
        assert unslop_text("3–5B parameters") == "3-5B parameters"

    def test_curly_quotes_become_straight(self):
        assert unslop_text("“hello” and ‘there’") == '"hello" and \'there\''

    def test_chatbot_opener_is_stripped(self):
        text = unslop_text("Of course! The repo is public.\n\n## Decision")
        assert not text.lower().startswith("of course")
        assert "## Decision" in text

    def test_lets_dive_in_is_stripped(self):
        text = unslop_text("Let's dive in: clone the repo first.")
        assert "dive" not in text.lower()
        assert "clone the repo first" in text.lower()

    def test_ai_vocabulary_is_replaced(self):
        out = unslop_text("Researchers delve into the tapestry of results.")
        assert "delve" not in out.lower()
        assert "tapestry" not in out.lower()

    def test_filler_is_cut(self):
        out = unslop_text("In order to run it, clone the repo.")
        assert out.lower().startswith("to run it")

    def test_not_only_but_also_is_flattened(self):
        out = unslop_text("It not only ships weights but also a GGUF.")
        assert "not only" not in out.lower()
        assert "GGUF" in out

    def test_markdown_hr_is_preserved(self):
        text = "## Notes\n\n---\n\nFooter stays."
        assert "---" in unslop_text(text)

    def test_empty_string_is_a_noop(self):
        assert unslop_text("") == ""

    def test_clean_unslops_model_markdown(self):
        raw = (
            "## The one thing\n"
            "This is a pivotal moment — the weights are on Hugging Face.\n"
        )
        cleaned = _clean(raw)
        assert "—" not in cleaned
        assert "pivotal" not in cleaned.lower()
        assert cleaned.startswith("## The one thing")

    def test_fallback_brief_is_not_unslopped(self):
        """Templated fallback still uses em dashes; only model output is scrubbed."""
        from ai_researcher.trends.brief import _fallback_markdown
        md = _fallback_markdown(
            [{"label": "Acme 7B", "summary": "open weights", "category": "model-release",
              "source_count": 1, "sources": ["Acme"]}],
            [],
            [{"title": "Acme 7B", "decision": "spike"}],
        )
        assert "—" in md


class TestBriefUsesPremium:
    @pytest.fixture
    def db(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Database(Path(tmp) / "t.db")

    def test_generate_brief_requests_premium_and_unslops(self, db):
        now = utcnow()
        day = local_day()
        url = "https://example.com/acme-7b"
        cur = db.execute(
            "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
            "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("src", "src:acme", url, url, url_hash(url), content_hash("Acme 7B", ""),
             "Acme 7B", "", "open weights", iso(now - timedelta(hours=1)), iso(now), 0, 0, "{}"),
        )
        item_id = cur.lastrowid
        db.execute(
            "INSERT INTO clusters (day, label, summary, category, score, size, source_count, "
            "first_seen, last_seen, entities, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (day, "Acme 7B open weights", "GGUF on Hugging Face", "model-release",
             0.9, 1, 1, iso(now), iso(now), jdump(["Acme"]), iso(now)),
        )
        cid = db.scalar("SELECT id FROM clusters")
        db.execute(
            "INSERT INTO cluster_items (cluster_id, item_id, is_primary) VALUES (?,?,?)",
            (cid, item_id, 1),
        )

        sloppy = (
            "## The one thing\n"
            "This is a pivotal moment — Acme shipped 7B GGUF weights today. "
            "In order to run them, pull Q4_K_M.\n\n"
            "## Also today\n"
            "- **Weights** — Acme 7B open weights landed on Hugging Face.\n"
            "- **Quant** — Q4_K_M is the practical local default.\n"
            "- **Card** — It fits a 12GB GPU without offload.\n"
            "- **License** — Check the model card before shipping.\n\n"
            "## Worth a closer look\n"
            "- **Card** — Confirm the GGUF quant and license.\n"
            "- **Serve** — A local llama.cpp spike is the first experiment.\n"
        )
        assert len(sloppy) > 120
        client = _BriefChat(sloppy)
        result = asyncio.run(generate_brief(db, client, day=day, force=True))
        assert client.text_kwargs[0].get("premium") is True
        assert result["model"] == "stub:premium"
        stored = db.scalar("SELECT markdown FROM briefs WHERE day=?", (day,))
        assert stored.startswith("## The one thing")
        assert "—" not in stored
        assert "pivotal" not in stored.lower()
        assert "to run them" in stored.lower()


class _BriefChat:
    available = True
    chat_model = "stub:workhorse"
    settings = Settings()

    def __init__(self, text):
        self.text = text
        self.text_kwargs = []

    async def probe(self):
        return True

    async def generate_text(self, *args, **kwargs):
        self.text_kwargs.append(kwargs)
        return self.text

    def model_for(self, *, premium=False, role=""):
        return "stub:premium" if premium else self.chat_model
