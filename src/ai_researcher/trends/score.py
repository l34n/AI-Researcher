"""Ranking.

Two ideas do most of the work:

* Engagement numbers are not comparable across sources. 400 points on Hacker
  News, 400 upvotes on r/LocalLLaMA, and 400 GitHub stars mean different things,
  so each is squashed through a log curve calibrated per source kind rather than
  compared raw.
* Corroboration is the strongest signal a dashboard has. One outlet reporting
  something is a claim; five independent outlets within a day is a story.
"""

from __future__ import annotations

import math
from datetime import datetime

from ..util import parse_datetime, utcnow

# Roughly "the engagement number at which this source is interesting".
# The log curve maps that value to ~0.5.
ENGAGEMENT_SCALE = {
    "hackernews": 150.0,
    "reddit": 300.0,
    "x": 400.0,
    "hf_papers": 40.0,
    "hf_models": 60.0,
    "github_trending": 800.0,
    "default": 200.0,
}

# Half-life of attention, in hours. AI news goes stale fast but not instantly.
RECENCY_HALFLIFE_H = 26.0


def _engagement_norm(value: float, source_key: str) -> float:
    if value <= 0:
        return 0.0
    scale = ENGAGEMENT_SCALE.get(source_key, ENGAGEMENT_SCALE["default"])
    return min(1.0, math.log1p(value) / math.log1p(scale * 2))


def _recency(published: datetime | None, now: datetime) -> float:
    if published is None:
        return 0.35
    age_h = max((now - published).total_seconds() / 3600.0, 0.0)
    if age_h > 24 * 30:
        return 0.02
    return 0.5 ** (age_h / RECENCY_HALFLIFE_H)


def score_item(item: dict, *, now: datetime | None = None) -> float:
    """0..~2 relevance score for a single item, before clustering."""
    now = now or utcnow()
    published = parse_datetime(item.get("published_at") or item.get("fetched_at"))

    importance = float(item.get("importance") or 0.5)
    meta = item.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    if meta.get("no_score"):
        # The source could not report engagement (e.g. Reddit's RSS fallback).
        # Scoring that as 0 would bury it beneath genuinely ignored items, so
        # treat it as unknown and let importance and trust decide.
        engagement = 0.35
    else:
        engagement = _engagement_norm(
            float(item.get("engagement") or 0.0), item.get("source_key", "")
        )

    trust = float(item.get("source_weight") or 1.0)
    # Subreddit-level weights ride along in meta so r/LocalLLaMA outranks
    # r/singularity without needing a source row of its own.
    if meta.get("sub_weight"):
        trust *= float(meta["sub_weight"])

    # Discussion volume is weak but real evidence that something landed.
    comments = min(1.0, math.log1p(float(item.get("comments") or 0)) / math.log1p(300))

    quality = 0.50 * importance + 0.32 * engagement + 0.18 * comments
    return round(quality * (trust / 1.5) * _recency(published, now) * 2.0, 4)


def score_cluster(group: list[dict], *, now: datetime | None = None) -> float:
    """Rank a story. Corroboration and source diversity dominate."""
    now = now or utcnow()
    if not group:
        return 0.0

    scores = sorted((float(it.get("score") or 0.0) for it in group), reverse=True)
    # Top item carries the story; each additional one adds a decaying share, so
    # a 30-item aggregator pile-up cannot outrank a genuine multi-source event.
    base = scores[0] + sum(s * (0.55 ** i) for i, s in enumerate(scores[1:], start=1))

    distinct_sources = len({it.get("source_key") for it in group})
    distinct_tiers = len({it.get("tier") for it in group})
    corroboration = 1.0 + 0.34 * math.log1p(distinct_sources - 1) + 0.10 * (distinct_tiers - 1)

    peak_importance = max(float(it.get("importance") or 0.5) for it in group)
    # A single very important item should still surface without corroboration.
    importance_floor = 1.0 + 0.45 * max(0.0, peak_importance - 0.7)

    freshest = max(
        (parse_datetime(it.get("published_at") or it.get("fetched_at")) for it in group),
        key=lambda d: d or datetime.min.replace(tzinfo=now.tzinfo),
        default=None,
    )
    return base * corroboration * importance_floor * (0.55 + 0.45 * _recency(freshest, now))


def explain_cluster(group: list[dict], *, now: datetime | None = None) -> str:
    """One-line ranking explanation shown on story cards."""
    now = now or utcnow()
    if not group:
        return "no items"
    distinct = len({it.get("source_key") for it in group})
    tiers = sorted({it.get("tier") or "news" for it in group})
    peak = max(float(it.get("importance") or 0.5) for it in group)
    freshest = max(
        (parse_datetime(it.get("published_at") or it.get("fetched_at")) for it in group),
        key=lambda d: d or datetime.min.replace(tzinfo=now.tzinfo),
        default=None,
    )
    age = ""
    if freshest:
        hours = max((now - freshest).total_seconds() / 3600.0, 0.0)
        if hours < 6:
            age = "very recent"
        elif hours < 24:
            age = "today"
        elif hours < 72:
            age = "this week"
        else:
            age = "aging"
    bits = [f"{distinct} source{'s' if distinct != 1 else ''}"]
    if len(tiers) > 1:
        bits.append(f"tiers {', '.join(tiers[:3])}")
    if peak >= 0.7:
        bits.append("high importance")
    if age:
        bits.append(age)
    ready = max((float(it.get("readiness") or 0) for it in group), default=0)
    if ready >= 0.62:
        bits.append("practitioner-ready")
    return " · ".join(bits)
