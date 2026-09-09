"""Conservative AI-relevance filter.

Unrelated headlines (sports, celebrity, generic HN offtopic) must not enter
the briefing. The default is *keep when unsure*: a false negative hides a
real model drop, a false positive is later downranked. Users can suppress a
source or category; those blocks are absolute.
"""

from __future__ import annotations

import re
from typing import Any

from .heuristics import (
    MODEL_FAMILIES,
    extract_entities,
    extract_tags,
)

# Source kinds that are AI-native. Their items stay unless a user muted them.
AI_NATIVE_KINDS = {
    "arxiv", "hf_papers", "hf_models", "github_releases", "github_trending",
}

_AI_WORDS = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|llm|large language model|gpt|"
    r"claude|gemini|llama|qwen|mistral|deepseek|openai|anthropic|hugging ?face|"
    r"machine learning|deep learning|neural net|transformer|diffusion|"
    r"foundation model|open[- ]weights?|inference|tokeniser|tokenizer|"
    r"fine[- ]tun(?:e|ing)|prompt|agentic|copilot|chatbot|multimodal|"
    r"gpu|h100|h200|b200|cuda|pytorch|tensorflow|mlx|ollama|vllm|"
    r"benchmark|leaderboard|sota|alignment|rlhf|rag)\b",
    re.IGNORECASE,
)

# Headlines that are almost never an AI story on their own. A genuine AI
# mention in the same title still wins (the positive regex runs first).
_OFFTOPIC = re.compile(
    r"\b(nfl|nba|mlb|nhl|premier league|world cup|super bowl|playoffs?|"
    r"box office|oscars?|grammys|celebrity|kardashian|taylor swift|"
    r"recipe|cookbook|dating app|crypto crash|bitcoin etf|"
    r"weather forecast|earthquake|horoscope)\b",
    re.IGNORECASE,
)

_WEAK_HN = re.compile(
    r"^(ask hn|tell hn|who is hiring|who wants to be hired|freelancer|"
    r"monthly|weekly who'?s hiring)\b",
    re.IGNORECASE,
)


def _user_blocks(meta: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    meta = meta or {}
    muted_sources = {str(x) for x in (meta.get("muted_sources") or []) if x}
    muted_categories = {str(x) for x in (meta.get("muted_categories") or []) if x}
    return muted_sources, muted_categories


def score_relevance(
    title: str,
    body: str = "",
    *,
    kind: str = "",
    source_key: str = "",
    category: str = "",
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    muted_sources: set[str] | None = None,
    muted_categories: set[str] | None = None,
) -> dict[str, Any]:
    """Return a keep/drop decision with a reason. Conservative: keep if unsure."""
    muted_sources = muted_sources or set()
    muted_categories = muted_categories or set()
    title = title or ""
    body = body or ""
    haystack = f"{title}\n{body[:1500]}"

    if source_key and source_key in muted_sources:
        return {
            "relevant": False, "score": 0.0, "reason": f"source {source_key} muted",
        }
    if category and category in muted_categories:
        return {
            "relevant": False, "score": 0.05, "reason": f"category {category} muted",
        }

    if kind in AI_NATIVE_KINDS:
        return {"relevant": True, "score": 0.92, "reason": f"native {kind} source"}

    entities = entities if entities is not None else extract_entities(title, body)
    tags = tags if tags is not None else extract_tags(title, body)

    score = 0.15
    reasons: list[str] = []
    if _AI_WORDS.search(title):
        score += 0.45
        reasons.append("AI term in title")
    elif _AI_WORDS.search(haystack):
        score += 0.22
        reasons.append("AI term in body")
    if entities:
        score += min(0.25, 0.08 * len(entities))
        reasons.append("named AI org or model")
    if tags:
        score += min(0.15, 0.04 * len(tags))
        reasons.append("AI technique tag")
    if any(fam in (title or "") for fam in MODEL_FAMILIES):
        score += 0.1

    offtopic = bool(_OFFTOPIC.search(title)) and not _AI_WORDS.search(title)
    if offtopic:
        score -= 0.55
        reasons.append("off-topic headline")
    weak_hn = bool(_WEAK_HN.search(title)) and not _AI_WORDS.search(haystack)
    if weak_hn:
        score -= 0.35
        reasons.append("generic HN thread")

    score = max(0.0, min(1.0, score))
    # Conservative keep: drop only a clear offtopic miss. Unknown blogs stay.
    # Generic HN threads (hiring, Ask HN with no AI term) are the exception —
    # they otherwise flood the brief as false positives.
    if (offtopic or weak_hn) and score < 0.28:
        relevant = False
        if not reasons:
            reasons.append("off-topic with no AI signal")
    elif score >= 0.28:
        relevant = True
    else:
        relevant = True
        score = max(score, 0.32)
        reasons = reasons or ["uncertain — kept"]
    return {
        "relevant": relevant,
        "score": round(score, 4),
        "reason": "; ".join(reasons[:3]),
        "entities": entities,
        "tags": tags,
    }


def load_user_blocks(db) -> tuple[set[str], set[str]]:
    """Muted sources and categories from the controls table (empty if absent)."""
    muted_sources: set[str] = set()
    muted_categories: set[str] = set()
    try:
        rows = db.query(
            "SELECT source_key, muted, category FROM source_controls"
        )
    except Exception:  # noqa: BLE001 - table may not exist on a mid-migration read
        return muted_sources, muted_categories
    for row in rows:
        if row["muted"]:
            if row["source_key"]:
                muted_sources.add(row["source_key"])
            if row["category"]:
                muted_categories.add(row["category"])
    return muted_sources, muted_categories


def apply_relevance(db, *, limit: int | None = None) -> dict[str, Any]:
    """Score items that have not been classified yet. Does not call a model."""
    muted_sources, muted_categories = load_user_blocks(db)
    sql = """
        SELECT i.id, i.title, i.body, i.source_key,
               COALESCE(s.kind, '') AS kind,
               COALESCE(e.category, '') AS category,
               COALESCE(e.entities, '[]') AS entities,
               COALESCE(e.tags, '[]') AS tags
        FROM items i
        LEFT JOIN sources s ON s.key = i.source_key
        LEFT JOIN enrichment e ON e.item_id = i.id
        WHERE COALESCE(i.relevant, -1) < 0
        ORDER BY COALESCE(i.published_at, i.fetched_at) DESC
    """
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    from ..db import jload  # local to avoid a cycle at import

    rows = db.query(sql, params)
    kept = dropped = 0
    from ..util import iso, utcnow
    now = iso(utcnow())
    with db.tx() as conn:
        for row in rows:
            judged = score_relevance(
                row["title"] or "",
                row["body"] or "",
                kind=row["kind"] or "",
                source_key=row["source_key"] or "",
                category=row["category"] or "",
                entities=jload(row["entities"], []),
                tags=jload(row["tags"], []),
                muted_sources=muted_sources,
                muted_categories=muted_categories,
            )
            conn.execute(
                "UPDATE items SET relevant=?, relevance_score=?, relevance_reason=?, "
                "relevance_at=? WHERE id=?",
                (
                    1 if judged["relevant"] else 0,
                    judged["score"],
                    judged["reason"][:240],
                    now,
                    row["id"],
                ),
            )
            if judged["relevant"]:
                kept += 1
            else:
                dropped += 1
    return {"scored": len(rows), "kept": kept, "dropped": dropped}

