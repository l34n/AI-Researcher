"""The quality-gate contract.

These tests pin the numbers that decide whether a story spends a scarce
research slot. If you change a threshold or a brake, this file should fail
in a way that names the policy you just changed — not a vague inequality.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from ai_researcher.config import Settings
from ai_researcher.db import Database, jdump
from ai_researcher.enrich.analyze import Enricher, SYSTEM as ENRICH_SYSTEM
from ai_researcher.enrich.judge import (
    ADOPT_FEASIBILITY,
    ADOPT_PRACTICALITY,
    ADOPT_READINESS,
    RESEARCH_READINESS,
    VERDICTS,
    WATCH_READINESS,
    WEIGHTS,
    Judge,
    blend,
    clamp,
    judge_text,
    readiness_of,
    verdict_of,
)
from ai_researcher.research.wiki import DeepResearcher
from ai_researcher.trends.cluster import build_clusters
from ai_researcher.util import content_hash, iso, url_hash, utcnow


# ── helpers ──────────────────────────────────────────────────────────

def heuristic(**kw):
    base = {
        "quality": 0.50, "practicality": 0.50, "feasibility": 0.50,
        "usefulness": 0.50, "readiness": 0.50, "verdict": "watch",
        "reasons": ["prior"], "artifacts": [],
    }
    base.update(kw)
    return base


class FakeChat:
    """In-process stand-in so gate tests never touch Ollama or the network."""

    available = True
    chat_model = "stub:test"
    settings = Settings()

    def __init__(self, payload=None, *, probe_ok=False, text=None):
        self.payload = payload
        self.probe_ok = probe_ok
        self.text = text
        self.calls = 0
        self.json_kwargs = []
        self.text_kwargs = []

    async def probe(self):
        return self.probe_ok

    async def generate_json(self, *args, **kwargs):
        self.calls += 1
        self.json_kwargs.append(kwargs)
        return self.payload

    async def generate_text(self, *args, **kwargs):
        self.text_kwargs.append(kwargs)
        return self.text

    def model_for(self, *, premium=False, role=""):
        return "stub:premium" if premium else self.chat_model

    async def aclose(self):
        return None


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield Database(Path(tmp) / "t.db")


def seed_item(
    db,
    *,
    title,
    body="",
    url=None,
    source="src",
    category="model-release",
    tier="news",
    importance=0.5,
    hours_ago=2,
    verdict=None,
    readiness=None,
    quality=0.70,
    practicality=0.70,
    feasibility=0.70,
    usefulness=0.70,
):
    """One stored item, optionally already judged.

    external_id includes title+url so multiple rows in one test do not collide.
    """
    now = utcnow()
    url = url or f"https://example.com/{source}/{title[:24]}"
    cur = db.execute(
        "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
        "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (source, f"{source}:{title[:32]}:{url}", url, url, url_hash(url),
         content_hash(title, body), title, "", body, iso(now - timedelta(hours=hours_ago)),
         iso(now), 0, 0, "{}"),
    )
    item_id = cur.lastrowid
    db.execute(
        "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
        "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (item_id, title, category, "[]", "[]", importance, "", "", iso(now)),
    )
    db.execute(
        "INSERT OR IGNORE INTO sources (key, name, kind, tier, weight) VALUES (?,?,?,?,?)",
        (source, source, "rss", tier, 1.0),
    )
    if verdict is not None:
        ready = readiness if readiness is not None else readiness_of(
            quality, practicality, feasibility, usefulness
        )
        db.execute(
            "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
            "readiness, verdict, reasons, artifacts, model, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, quality, practicality, feasibility, usefulness, ready,
             verdict, jdump([]), jdump([]), "", iso(now)),
        )
    return item_id


# ── the numbers themselves ───────────────────────────────────────────

class TestGateThresholds:
    """These four constants *are* the gate. Off-by-0.001 is a policy change."""

    def test_documented_priors(self):
        assert ADOPT_READINESS == 0.78
        assert RESEARCH_READINESS == 0.62
        assert WATCH_READINESS == 0.42
        assert ADOPT_PRACTICALITY == 0.58
        assert ADOPT_FEASIBILITY == 0.52

    def test_settings_threshold_matches_the_constant(self):
        """A run uses Settings.research_threshold, not the constant directly.
        Drift between them would silently research a different slice."""
        assert Settings().research_threshold == RESEARCH_READINESS

    def test_premium_readiness_matches_the_research_gate(self):
        """Premium models are spent on the same slice the wiki is gated on."""
        assert Settings().premium_readiness == RESEARCH_READINESS

    def test_verdict_ladder(self):
        assert VERDICTS == ("skip", "watch", "research", "adopt")

    @pytest.mark.parametrize("readiness,expected", [
        (0.00, "skip"),
        (WATCH_READINESS - 0.0001, "skip"),
        (WATCH_READINESS, "watch"),
        (RESEARCH_READINESS - 0.0001, "watch"),
        (RESEARCH_READINESS, "research"),
        (ADOPT_READINESS - 0.0001, "research"),
        (ADOPT_READINESS, "research"),  # adopt still needs the brakes
        (0.99, "research"),             # high readiness, no P/F → not adopt
    ])
    def test_readiness_bands_without_brakes(self, readiness, expected):
        assert verdict_of(readiness) == expected

    def test_adopt_requires_all_three(self):
        assert verdict_of(
            ADOPT_READINESS,
            practicality=ADOPT_PRACTICALITY,
            feasibility=ADOPT_FEASIBILITY,
        ) == "adopt"

    @pytest.mark.parametrize("practicality,feasibility", [
        (ADOPT_PRACTICALITY - 0.0001, ADOPT_FEASIBILITY),
        (ADOPT_PRACTICALITY, ADOPT_FEASIBILITY - 0.0001),
        (0.0, 1.0),
        (1.0, 0.0),
    ])
    def test_adopt_brakes_fire_independently(self, practicality, feasibility):
        """A beautiful paper you cannot run must not be 'adopt'."""
        assert verdict_of(
            0.99, practicality=practicality, feasibility=feasibility
        ) == "research"


class TestReadinessContract:
    def test_weights_are_the_documented_split(self):
        assert WEIGHTS == {
            "quality": 0.28, "practicality": 0.22,
            "feasibility": 0.22, "usefulness": 0.28,
        }
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12

    def test_each_axis_recovers_its_weight(self):
        assert readiness_of(1, 0, 0, 0) == WEIGHTS["quality"]
        assert readiness_of(0, 1, 0, 0) == WEIGHTS["practicality"]
        assert readiness_of(0, 0, 1, 0) == WEIGHTS["feasibility"]
        assert readiness_of(0, 0, 0, 1) == WEIGHTS["usefulness"]

    def test_rounds_to_four_decimals(self):
        # 0.28*0.3333 + 0.22*0.3333 + 0.22*0.3333 + 0.28*0.3333 = 0.3333
        assert readiness_of(0.3333, 0.3333, 0.3333, 0.3333) == 0.3333

    def test_clamp_bounds(self):
        assert clamp(-0.4) == 0.0
        assert clamp(1.7) == 1.0
        assert clamp(0.42) == 0.42


# ── heuristic scoring: one lever at a time ───────────────────────────

class TestDimensionIsolation:
    """Hold everything else still and move one signal. Ranking, not vibes."""

    def test_lab_tier_beats_news_on_quality(self):
        lab = judge_text("Model ships", "open weights", category="model-release", tier="lab")
        news = judge_text("Model ships", "open weights", category="model-release", tier="news")
        assert lab["quality"] > news["quality"]

    def test_primary_research_tier_beats_analyst(self):
        research = judge_text("A method", "we propose", category="research", tier="research")
        analyst = judge_text("A method", "we propose", category="research", tier="analyst")
        assert research["quality"] > analyst["quality"]

    def test_weak_title_is_commentary_not_a_discovery(self):
        weak = judge_text("How to use RAG: a beginner tutorial", "a weekly digest")
        strong = judge_text("Acme releases v1.2 with open weights", "release notes")
        assert weak["quality"] < strong["quality"]
        assert weak["usefulness"] < strong["usefulness"]
        assert "commentary or how-to" in " ".join(weak["reasons"])

    def test_long_body_lifts_quality(self):
        short = judge_text("Specific v1.2 release", "short")
        long = judge_text("Specific v1.2 release", "x" * 501)
        assert long["quality"] - short["quality"] == pytest.approx(0.08)

    def test_specificity_lifts_quality(self):
        vague = judge_text("Someone shipped something", "announcement")
        numbered = judge_text("Someone shipped v2.1", "announcement")
        assert numbered["quality"] > vague["quality"]

    def test_closed_access_lowers_practicality_and_feasibility(self):
        open_ = judge_text("Weights up", "open weights on huggingface")
        shut = judge_text("Weights up", "weights not released, proprietary API-only waitlist")
        assert shut["practicality"] < open_["practicality"]
        assert shut["feasibility"] < open_["feasibility"]
        assert "closed or waitlisted" in " ".join(shut["reasons"])

    def test_frontier_compute_is_penalised_unless_it_runs_locally(self):
        heavy = judge_text("New 405B model", "405B weights, cluster of H100s required")
        local = judge_text(
            "New 405B model",
            "405B weights, cluster of H100s. Also a GGUF Q4 that runs locally on an RTX.",
        )
        assert heavy["feasibility"] < local["feasibility"]
        assert "frontier-scale" in " ".join(heavy["reasons"])
        assert "frontier-scale" not in " ".join(local["reasons"])

    def test_named_repo_lifts_practicality(self):
        none = judge_text("We shipped a thing", "announcement")
        repo = judge_text(
            "We shipped a thing",
            "Code at https://github.com/acme/thing",
            url="https://github.com/acme/thing",
        )
        assert repo["practicality"] > none["practicality"]
        assert repo["artifacts"]
        assert "fetchable artifact" in " ".join(repo["reasons"])

    def test_research_paper_without_artifact_is_harder_to_adopt(self):
        paper = judge_text(
            "We propose an attention variant",
            "In this paper we introduce a method.",
            category="research",
            tier="research",
        )
        coded = judge_text(
            "We propose an attention variant",
            "In this paper we introduce a method. Code at https://github.com/lab/attn",
            category="research",
            tier="research",
            url="https://github.com/lab/attn",
        )
        assert coded["practicality"] > paper["practicality"]
        assert coded["feasibility"] > paper["feasibility"]

    def test_funding_and_policy_are_not_practical(self):
        funding = judge_text(
            "Acme raises two billion", "Series B funding round",
            category="funding-acquisition", tier="news", importance=0.8,
        )
        tool = judge_text(
            "vLLM v0.8 released",
            "pip install vllm. Code at https://github.com/vllm-project/vllm",
            category="tooling-oss", tier="vendor",
            url="https://github.com/vllm-project/vllm", importance=0.8,
        )
        assert funding["practicality"] < tool["practicality"]
        assert funding["verdict"] in ("skip", "watch")

    def test_corroboration_caps_and_is_mentioned(self):
        one = judge_text("Lab ships", "open weights", source_count=1)
        two = judge_text("Lab ships", "open weights", source_count=2)
        four = judge_text("Lab ships", "open weights", source_count=4)
        ten = judge_text("Lab ships", "open weights", source_count=10)
        assert two["quality"] - one["quality"] == pytest.approx(0.04)
        # +0.04 per extra source, capped at +0.10 — a pile-on is not more evidence.
        assert four["quality"] - one["quality"] == pytest.approx(0.10)
        assert ten["quality"] == four["quality"]
        assert "corroborated by 4 sources" in four["reasons"]

    def test_importance_feeds_usefulness_and_is_clamped(self):
        low = judge_text("Item", "body", importance=0.0)
        high = judge_text("Item", "body", importance=1.0)
        over = judge_text("Item", "body", importance=4.0)
        assert high["usefulness"] > low["usefulness"]
        assert over["usefulness"] == high["usefulness"]

    def test_empty_inputs_still_produce_a_legal_judgment(self):
        judged = judge_text("", "")
        for key in ("quality", "practicality", "feasibility", "usefulness", "readiness"):
            assert 0.0 <= judged[key] <= 1.0
        assert judged["verdict"] in VERDICTS
        assert judged["reasons"]


# ── typical items land in the intended band ──────────────────────────

class TestPersonaBands:
    """End-to-end heuristic verdicts for the four kinds of thing the gate
    was written to separate. Bands, not exact scores — the scores can drift
    a little; the band is the product promise."""

    def test_local_open_tool_is_worth_a_slot(self):
        judged = judge_text(
            "vLLM v0.8 released with OpenAI-compatible server",
            "Release notes and pip install vllm. Code at https://github.com/vllm-project/vllm "
            "runs locally with llama.cpp-style GGUF and a 4-bit quant. Faster and cheaper.",
            category="tooling-oss",
            tier="vendor",
            url="https://github.com/vllm-project/vllm",
            importance=0.75,
        )
        assert judged["verdict"] in ("research", "adopt")
        assert judged["readiness"] >= RESEARCH_READINESS
        assert judged["practicality"] >= ADOPT_PRACTICALITY

    def test_hot_take_does_not_consume_a_slot(self):
        judged = judge_text(
            "What do you think about the latest AI hype? Unpopular opinion thread",
            "Just a weekly digest hot take. How to get started as a beginner.",
            category="opinion-analysis",
            tier="community",
            importance=0.3,
        )
        assert judged["verdict"] in ("skip", "watch")
        assert judged["readiness"] < RESEARCH_READINESS

    def test_closed_frontier_model_is_not_adoptable(self):
        judged = judge_text(
            "Lab announces a closed 405B API-only model, waitlist now open",
            "Weights not released. Proprietary API-only access. Cluster of H100s. "
            "Not yet available to the public.",
            category="model-release",
            tier="lab",
            importance=0.95,
        )
        assert judged["verdict"] != "adopt"
        assert judged["feasibility"] < ADOPT_FEASIBILITY

    def test_funding_round_stays_on_the_firehose(self):
        judged = judge_text(
            "Acme raises two billion at a sixty billion valuation",
            "Series B funding round. No product shipped.",
            category="funding-acquisition",
            tier="news",
            importance=0.85,
        )
        assert judged["verdict"] in ("skip", "watch")
        assert judged["readiness"] < RESEARCH_READINESS


# ── blend: the model is a prior, not the judge ───────────────────────

class TestBlendBrakes:
    def test_blend_math_is_fifty_five_forty_five(self):
        out = blend(heuristic(quality=0.40), {"quality": 0.80})
        assert out["quality"] == round(0.55 * 0.80 + 0.45 * 0.40, 4)
        assert out["readiness"] == readiness_of(
            out["quality"], out["practicality"], out["feasibility"], out["usefulness"]
        )

    def test_out_of_range_model_score_is_clamped_before_blend(self):
        out = blend(heuristic(quality=0.40), {"quality": 4.0})
        assert out["quality"] == round(0.55 * 1.0 + 0.45 * 0.40, 4)

    def test_garbage_model_score_keeps_the_heuristic(self):
        out = blend(heuristic(quality=0.40), {"quality": "high"})
        assert out["quality"] == 0.40

    def test_model_may_nudge_one_band_not_two(self):
        # computed from 0.50s is watch; model saying research is one step.
        out = blend(heuristic(), {"verdict": "research"})
        assert out["verdict"] == "research"
        # skip → adopt is a jump; keep the computed band.
        jumped = blend(heuristic(), {
            "quality": 0.50, "practicality": 0.50, "feasibility": 0.50,
            "usefulness": 0.50, "verdict": "adopt",
        })
        assert jumped["verdict"] != "adopt"

    def test_cheerful_model_cannot_force_adopt_on_a_skip(self):
        """Regression: 0.99 model scores blend a skip prior into the research
        band (readiness ~0.62). Measuring the one-step rule from that blended
        band — not the heuristic — used to honour adopt."""
        prior = judge_text(
            "weekly digest hot take", "what do you think, beginner tutorial",
            category="opinion-analysis", tier="news", importance=0.2,
        )
        assert prior["verdict"] == "skip"
        out = blend(prior, {
            "quality": 0.99, "practicality": 0.99, "feasibility": 0.99,
            "usefulness": 0.99, "verdict": "adopt",
            "reasons": ["amazing"], "artifacts": [],
        })
        assert out["verdict"] != "adopt"
        assert out["verdict"] in ("skip", "watch", "research")

    def test_model_may_promote_research_to_adopt_when_brakes_pass(self):
        prior = heuristic(
            quality=0.80, practicality=0.80, feasibility=0.80, usefulness=0.80,
            verdict="research",
        )
        out = blend(prior, {
            "quality": 0.90, "practicality": 0.90, "feasibility": 0.90,
            "usefulness": 0.90, "verdict": "adopt",
        })
        assert out["verdict"] == "adopt"

    def test_adopt_brakes_reapplied_after_blend(self):
        # High readiness, but practicality stays under the brake.
        prior = heuristic(
            quality=0.95, practicality=0.20, feasibility=0.90, usefulness=0.95,
        )
        out = blend(prior, {
            "quality": 0.95, "practicality": 0.20, "feasibility": 0.90,
            "usefulness": 0.95, "verdict": "adopt",
        })
        assert out["practicality"] < ADOPT_PRACTICALITY
        assert out["verdict"] == "research"

    def test_reasons_and_artifacts_merge_with_caps(self):
        prior = heuristic(
            reasons=["a", "b"],
            artifacts=["https://github.com/acme/x"],
        )
        out = blend(prior, {
            "reasons": ["model-reason", "  ", ""],
            "artifacts": ["https://huggingface.co/acme/y", "https://github.com/acme/x"],
        })
        assert out["reasons"][0] == "model-reason"
        assert "a" in out["reasons"]
        assert len(out["reasons"]) <= 5
        assert out["artifacts"][0] == "https://huggingface.co/acme/y"
        assert out["artifacts"].count("https://github.com/acme/x") == 1


# ── persistence: what the pipeline actually stores ───────────────────

class TestJudgePersistence:
    def test_heuristic_pass_writes_one_row_per_item(self, db):
        seed_item(db, title="Open weights on GitHub",
                  body="pip install. https://github.com/acme/mod runs locally.",
                  url="https://github.com/acme/mod", tier="lab")
        result = asyncio.run(Judge(Settings(), db, FakeChat()).run(limit=0))
        assert result["judged"] == 1
        assert result["llm"] == 0
        row = db.one("SELECT * FROM judgments")
        assert row["verdict"] in VERDICTS
        assert 0.0 <= row["readiness"] <= 1.0
        assert row["model"] == ""

    def test_second_pass_does_not_rescore(self, db):
        seed_item(db, title="Already judged", body="open weights")
        judge = Judge(Settings(), db, FakeChat())
        first = asyncio.run(judge.run(limit=0))
        second = asyncio.run(judge.run(limit=0))
        assert first["judged"] == 1
        assert second["judged"] == 0
        assert db.scalar("SELECT COUNT(*) FROM judgments") == 1

    def test_empty_db_is_a_no_op(self, db):
        result = asyncio.run(Judge(Settings(), db, FakeChat()).run())
        assert result["judged"] == 0
        assert result["adopt"] == 0

    def test_cluster_corroboration_reaches_the_heuristic(self, db):
        """The judge reads today's source_count. Two outlets, one URL, is evidence."""
        seed_item(db, title="OpenAI ships GPT-5", url="https://o.ai/gpt5",
                  source="a", body="open weights v1.0")
        seed_item(db, title="OpenAI ships GPT-5", url="https://o.ai/gpt5?utm_source=x",
                  source="b", body="open weights v1.0")
        build_clusters(db)
        assert db.scalar("SELECT source_count FROM clusters") == 2
        asyncio.run(Judge(Settings(), db, FakeChat()).run(limit=0))
        stored = db.scalar("SELECT quality FROM judgments ORDER BY quality DESC LIMIT 1")
        lone = judge_text("OpenAI ships GPT-5", "open weights v1.0", source_count=1)
        assert stored > lone["quality"]

    def test_model_budget_is_spent_on_highest_readiness(self, db):
        seed_item(db, title="low", verdict="skip", readiness=0.20,
                  quality=0.2, practicality=0.2, feasibility=0.2, usefulness=0.2)
        seed_item(db, title="high", verdict="research", readiness=0.80,
                  quality=0.8, practicality=0.8, feasibility=0.8, usefulness=0.8,
                  hours_ago=3)
        client = FakeChat(
            {"quality": 0.81, "practicality": 0.81, "feasibility": 0.81,
             "usefulness": 0.81, "verdict": "research", "reasons": ["ok"],
             "artifacts": []},
            probe_ok=True,
        )
        result = asyncio.run(Judge(Settings(), db, client).run(limit=1))
        assert result["llm"] == 1
        assert client.calls == 1
        high = db.one(
            "SELECT j.model FROM judgments j JOIN items i ON i.id=j.item_id WHERE i.title='high'"
        )
        low = db.one(
            "SELECT j.model FROM judgments j JOIN items i ON i.id=j.item_id WHERE i.title='low'"
        )
        assert high["model"] == "stub:premium"
        assert low["model"] == ""

    def test_failed_model_call_leaves_the_heuristic(self, db):
        seed_item(db, title="stay heuristic", verdict="watch", readiness=0.50)
        client = FakeChat(None, probe_ok=True)
        result = asyncio.run(Judge(Settings(), db, client).run(limit=1))
        assert result["llm"] == 0
        assert db.scalar("SELECT model FROM judgments") == ""

    def test_high_readiness_requests_the_premium_model(self, db):
        seed_item(
            db, title="ready", verdict="research", readiness=0.80,
            quality=0.8, practicality=0.8, feasibility=0.8, usefulness=0.8,
        )
        client = FakeChat(
            {"quality": 0.81, "practicality": 0.81, "feasibility": 0.81,
             "usefulness": 0.81, "verdict": "research", "reasons": ["ok"],
             "artifacts": []},
            probe_ok=True,
        )
        asyncio.run(Judge(Settings(), db, client).run(limit=1))
        assert client.json_kwargs[0]["premium"] is True
        assert db.scalar("SELECT model FROM judgments") == "stub:premium"

    def test_watch_band_stays_on_the_workhorse(self, db):
        seed_item(
            db, title="watch", verdict="watch", readiness=0.50,
            quality=0.5, practicality=0.5, feasibility=0.5, usefulness=0.5,
        )
        client = FakeChat(
            {"quality": 0.51, "practicality": 0.51, "feasibility": 0.51,
             "usefulness": 0.51, "verdict": "watch", "reasons": ["ok"],
             "artifacts": []},
            probe_ok=True,
        )
        asyncio.run(Judge(Settings(), db, client).run(limit=1))
        assert client.json_kwargs[0]["premium"] is False
        assert db.scalar("SELECT model FROM judgments") == "stub:test"


class TestEnrichmentStaysOnWorkhorse:
    def test_model_pass_does_not_request_premium(self, db):
        seed_item(db, title="Acme 7B open weights on GitHub",
                  body="GGUF on Hugging Face. pip install acme.")
        client = FakeChat(
            {"summary": "Acme released 7B GGUF weights — delve into the card.",
             "why": "local inference got cheaper",
             "category": "model-release",
             "entities": ["Acme"],
             "importance": 0.7},
            probe_ok=True,
        )
        result = asyncio.run(Enricher(Settings(), db, client).run(limit=1))
        assert result["llm"] == 1
        assert client.json_kwargs[0].get("premium", False) is False
        summary = db.scalar("SELECT summary FROM enrichment")
        assert "—" not in summary
        assert "delve" not in summary.lower()

    def test_enrich_prompt_carries_unslop_rule(self):
        from ai_researcher.enrich.unslop import UNSLOP_RULE
        assert UNSLOP_RULE in ENRICH_SYSTEM


# ── admission to deep research ───────────────────────────────────────

class TestResearchAdmission:
    """Only research/adopt above the threshold spend a wiki slot."""

    def _candidates(self, db, **settings_kw):
        settings = Settings()
        for key, value in settings_kw.items():
            setattr(settings, key, value)
        return DeepResearcher(settings, db, FakeChat())._candidates(
            settings.research_budget, force=False
        )

    def test_watch_and_skip_are_refused_even_when_readiness_is_high(self, db):
        # A high score with the wrong verdict is a bug in the writer, not a
        # ticket into the wiki — the SQL gate checks both.
        seed_item(db, title="loud but watch", verdict="watch", readiness=0.90)
        seed_item(db, title="loud but skip", verdict="skip", readiness=0.90)
        assert self._candidates(db, research_threshold=0.62, research_budget=10) == []

    def test_research_and_adopt_above_threshold_are_admitted(self, db):
        seed_item(db, title="spike me", verdict="research", readiness=0.66)
        seed_item(db, title="ship me", verdict="adopt", readiness=0.84)
        titles = {c["title"] for c in self._candidates(db, research_budget=10)}
        assert titles == {"spike me", "ship me"}

    def test_threshold_is_inclusive(self, db):
        seed_item(db, title="on the line", verdict="research", readiness=0.62)
        seed_item(db, title="just under", verdict="research", readiness=0.6199)
        titles = [c["title"] for c in self._candidates(db, research_threshold=0.62, research_budget=10)]
        assert titles == ["on the line"]

    def test_highest_readiness_is_researched_first(self, db):
        seed_item(db, title="second", verdict="research", readiness=0.65)
        seed_item(db, title="first", verdict="adopt", readiness=0.88)
        titles = [c["title"] for c in self._candidates(db, research_budget=2)]
        assert titles == ["first", "second"]

    def test_budget_clips_the_queue(self, db):
        seed_item(db, title="a", verdict="research", readiness=0.90)
        seed_item(db, title="b", verdict="research", readiness=0.80)
        seed_item(db, title="c", verdict="research", readiness=0.70)
        got = self._candidates(db, research_budget=2)
        assert [c["title"] for c in got] == ["a", "b"]

    def test_zero_budget_admits_nothing(self, db):
        seed_item(db, title="ready", verdict="adopt", readiness=0.90)
        assert self._candidates(db, research_budget=0) == []

    def test_completed_brief_is_skipped_unless_forced(self, db):
        item_id = seed_item(db, title="already done", verdict="research", readiness=0.70)
        now = iso(utcnow())
        db.execute(
            "INSERT INTO research (item_id, title, status, readiness, verdict, decision, "
            "model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, "already done", "complete", 0.70, "research", "spike", "", now, now),
        )
        settings = Settings()
        settings.research_budget = 4
        researcher = DeepResearcher(settings, db, FakeChat())
        assert researcher._candidates(4, force=False) == []
        forced = researcher._candidates(4, force=True)
        assert len(forced) == 1
        assert forced[0]["item_id"] == item_id


class TestStoryRollup:
    """A clustered story must inherit the member you can actually implement."""

    def test_rollup_prefers_the_ready_member(self):
        from ai_researcher.web.queries import _rollup_judgment

        primary = {
            "quality": 0.4, "practicality": 0.2, "feasibility": 0.3,
            "usefulness": 0.4, "readiness": 0.34, "verdict": "watch",
            "research_id": 0, "research_decision": "", "artifacts": [],
        }
        implementable = {
            "quality": 0.7, "practicality": 0.8, "feasibility": 0.7,
            "usefulness": 0.7, "readiness": 0.72, "verdict": "research",
            "research_id": 9, "research_decision": "spike",
            "artifacts": ["https://github.com/acme/x"],
        }
        rolled = _rollup_judgment([primary, implementable])
        assert rolled["readiness"] == 0.72
        assert rolled["verdict"] == "research"
        assert rolled["research_id"] == 9
        assert "https://github.com/acme/x" in rolled["artifacts"]

    def test_adapt_excerpt_reads_the_decision(self):
        from ai_researcher.web.queries import adapt_excerpt

        md = "# Adapt\n## Decision\n**spike** — clone the repo and run the README.\n\n## Who\nML infra."
        excerpt = adapt_excerpt(md)
        assert excerpt.startswith("spike —")
        assert "clone the repo" in excerpt
        assert "*" not in excerpt

    def test_ready_filter_hides_watch_stories(self, db):
        from ai_researcher.trends.cluster import build_clusters
        from ai_researcher.web import queries as Q
        from ai_researcher.util import local_day

        seed_item(db, title="OpenAI ships GPT-5", url="https://o.ai/gpt5",
                  source="a", verdict="watch", readiness=0.40)
        seed_item(db, title="Local 7B open weights", url="https://github.com/acme/7b",
                  source="b", verdict="research", readiness=0.70)
        build_clusters(db)
        all_stories = Q.top_stories(db, day=local_day(), limit=20)
        ready = Q.top_stories(db, day=local_day(), limit=20, ready=True)
        assert len(all_stories) >= 2
        assert all(
            s["verdict"] in ("research", "adopt") or s["research_id"]
            for s in ready
        )
        assert any(s["verdict"] == "research" for s in ready)

    def test_firehose_ready_order_puts_high_readiness_first(self, db):
        from ai_researcher.web import queries as Q

        seed_item(db, title="low", verdict="watch", readiness=0.30, hours_ago=1)
        seed_item(db, title="high", verdict="adopt", readiness=0.88, hours_ago=5)
        items = Q.list_items(db, hours=48, order="ready", limit=10)
        titles = [i["title"] for i in items]
        assert titles.index("high") < titles.index("low")


class TestBriefReadySection:
    def test_fallback_brief_lists_ready_items(self):
        from ai_researcher.trends.brief import _fallback_markdown

        stories = [{
            "label": "Someone shipped a model",
            "summary": "A release.",
            "category": "model-release",
            "source_count": 2,
        }]
        ready = [{"title": "vLLM 0.8", "decision": "adopt", "readiness": 0.84}]
        md = _fallback_markdown(stories, [], ready)
        assert "## Ready to build" in md
        assert "vLLM 0.8" in md
        assert "adopt" in md

    def test_fallback_omits_ready_section_when_empty(self):
        from ai_researcher.trends.brief import _fallback_markdown

        stories = [{
            "label": "Noise", "summary": "x", "category": "opinion-analysis",
            "source_count": 1,
        }]
        md = _fallback_markdown(stories, [], [])
        assert "## Ready to build" not in md
