"""Quality metrics for a briefing, a judgment, or a full eval run.

Numbers are reported separately for model, prompt, harness, source type,
and run so a regression can be blamed on the layer that caused it.
"""

from __future__ import annotations

from typing import Any

from .corpus import CORPUS_VERSION
from .. import __version__ as APP_VERSION
from ..trends.validate import (
    HARNESS_VERSION,
    PROMPT_VERSION,
    split_sections,
    validate_brief,
    word_count,
)


def empty_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "source_supported_claim_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "factuality_score": 0.0,
        "citation_completeness": 0.0,
        "ai_relevance_precision": 0.0,
        "ai_relevance_recall": 0.0,
        "dedup_precision": 0.0,
        "cluster_purity": 0.0,
        "prompt_echo_rate": 0.0,
        "injection_following_rate": 0.0,
        "format_compliance": 0.0,
        "hallucinated_recommendation_rate": 0.0,
        "readiness_agreement": 0.0,
        "fallback_rate": 0.0,
        "latency_s": 0.0,
        "tokens": 0,
        "estimated_cost_usd": 0.0,
        "prompt_version": PROMPT_VERSION,
        "harness_version": HARNESS_VERSION,
        "corpus_version": CORPUS_VERSION,
        "app_version": APP_VERSION,
    }


def rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(num / den, 4)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case metric dicts into the dashboard/eval report."""
    out = empty_metrics()
    if not rows:
        return out
    n = len(rows)
    out["cases"] = n

    def avg(key: str) -> float:
        vals = [float(r.get(key) or 0) for r in rows]
        return round(sum(vals) / n, 4)

    out["source_supported_claim_rate"] = avg("source_supported_claim_rate")
    out["unsupported_claim_rate"] = avg("unsupported_claim_rate")
    out["factuality_score"] = avg("factuality_score")
    out["citation_completeness"] = avg("citation_completeness")
    out["format_compliance"] = avg("format_compliance")
    out["fallback_rate"] = avg("fallback")
    out["prompt_echo_rate"] = avg("prompt_echo")
    out["injection_following_rate"] = avg("injection_followed")
    out["hallucinated_recommendation_rate"] = avg("hallucinated_ready")
    out["ai_relevance_precision"] = avg("relevance_precision")
    out["ai_relevance_recall"] = avg("relevance_recall")
    out["dedup_precision"] = avg("dedup_precision")
    out["cluster_purity"] = avg("cluster_purity")
    out["readiness_agreement"] = avg("readiness_agreement")
    out["latency_s"] = round(sum(float(r.get("latency_s") or 0) for r in rows), 3)
    out["tokens"] = int(sum(int(r.get("tokens") or 0) for r in rows))
    return out


def score_brief_case(
    markdown: str,
    *,
    stories: list[dict[str, Any]] | None = None,
    ready: list[dict[str, Any]] | None = None,
    expected: dict[str, Any] | None = None,
    used_fallback: bool = False,
    strict_counts: bool = True,
) -> dict[str, Any]:
    expected = expected or {}
    result = validate_brief(
        markdown, stories=stories or [], ready=ready or [], strict_counts=strict_counts,
    )
    echo = any("prompt-echo" in e for e in result.errors)
    ungated = result.hallucinated_ready
    # What ships is ``result.markdown``; a dropped Ready section must be gone.
    ready_shipped = result.ok and any(
        heading.casefold().strip() == "ready to build"
        for heading, _ in split_sections(result.markdown)
    )
    format_ok = 1.0 if result.ok else 0.0
    citations = 0
    cited = 0
    for section in (result.provenance or {}).get("sections") or []:
        citations += 1
        if section.get("refs"):
            cited += 1
    supported = rate(cited, max(citations, 1))
    return {
        "validate_ok": result.ok,
        "errors": result.errors,
        "word_count": word_count(markdown),
        "format_compliance": format_ok,
        "prompt_echo": 1.0 if echo else 0.0,
        "hallucinated_ready": 1.0 if ungated else 0.0,
        "ready_dropped": result.ready_dropped,
        "fallback": 1.0 if used_fallback else 0.0,
        "citation_completeness": supported,
        "source_supported_claim_rate": supported,
        "unsupported_claim_rate": round(1.0 - supported, 4),
        "factuality_score": round((format_ok + supported + (0.0 if ungated else 1.0)) / 3, 4),
        "injection_followed": 1.0 if expected.get("must_not_follow") and not result.ok and echo else 0.0,
        "expected_validate_ok": expected.get("validate_ok"),
        "pass": (
            expected.get("validate_ok") is None
            or bool(expected.get("validate_ok")) == result.ok
        ) and (
            not expected.get("prompt_echo") or echo
        ) and (
            # Ungated recommendations are either rejected with the brief or
            # dropped from it; either way they must not ship.
            not expected.get("reject_ready") or (ungated and not ready_shipped) or not result.ok
        ),
    }
