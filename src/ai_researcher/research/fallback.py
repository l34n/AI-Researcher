"""Templated wiki pages for when Ollama is down or a turn fails.

The Adapt tab must never be blank. A model-written brief is better, but a
structured digest from the judgment and the clustered sources is enough to
decide whether to come back when a model is available.
"""

from __future__ import annotations

from typing import Any

from ..util import truncate


def _bullets(values: list[str], empty: str) -> str:
    cleaned = [v.strip() for v in values if v and str(v).strip()]
    if not cleaned:
        return f"- {empty}"
    return "\n".join(f"- {truncate(v, 180)}" for v in cleaned)


def _scores(judgment: dict[str, Any]) -> str:
    return (
        f"Q={judgment.get('quality', 0):.2f} "
        f"P={judgment.get('practicality', 0):.2f} "
        f"F={judgment.get('feasibility', 0):.2f} "
        f"U={judgment.get('usefulness', 0):.2f}"
    )


def render_page(slug: str, candidate: dict[str, Any], pages: dict[str, str]) -> str:
    dispatch = {
        "source": _source,
        "claims": _claims,
        "critique": _critique,
        "adapt": _adapt,
        "lint": _lint,
    }
    writer = dispatch.get(slug, _source)
    return writer(candidate, pages)


def _source(candidate: dict[str, Any], pages: dict[str, str]) -> str:
    items = candidate.get("items") or []
    primary = items[0] if items else {}
    artifacts = candidate.get("artifacts") or []
    who = candidate.get("entities") or []
    body = (primary.get("body") or primary.get("summary") or "").strip()
    claims = []
    if primary.get("summary"):
        claims.append(primary["summary"])
    if primary.get("why"):
        claims.append(primary["why"])
    if not claims and body:
        claims.append(truncate(body, 220))
    return "\n".join([
        "# Source",
        "## What happened",
        truncate(primary.get("summary") or primary.get("title") or candidate.get("title", ""), 400)
        or "No summary stored for this item.",
        "",
        "## Who",
        _bullets(who, "Unknown — no entities extracted."),
        "",
        "## Artifacts",
        _bullets(artifacts, "None named."),
        "",
        "## Claims (as stated)",
        _bullets(claims, "The sources do not state a concrete claim."),
        "",
        "## Not in the sources",
        "- License, hardware envelope, and reproduction steps are not in the stored text.",
        "- Generated without a language model — install an Ollama chat model for a written ingest.",
    ])


def _claims(candidate: dict[str, Any], pages: dict[str, str]) -> str:
    artifacts = candidate.get("artifacts") or []
    reasons = candidate.get("reasons") or []
    demonstrated = list(artifacts) or [r for r in reasons if "artifact" in r]
    asserted = [r for r in reasons if r not in demonstrated] or [
        "The stored text is too short to separate demonstration from assertion."
    ]
    return "\n".join([
        "# Claims",
        "## Demonstrated",
        _bullets(demonstrated, "Nothing fetchable is named in the stored text."),
        "",
        "## Asserted",
        _bullets(asserted, "No unsupported assertions isolated."),
        "",
        "## Missing evidence",
        "- Independent reproduction, license, and hardware cost are not in the sources.",
        "",
        "## Testable next",
        "- Open the primary URL and confirm the artifact still exists.",
        "- If a repo is named, clone it and run its README smoke test.",
        "- If only a paper is named, look for a linked code section before scheduling more time.",
    ])


def _critique(candidate: dict[str, Any], pages: dict[str, str]) -> str:
    j = candidate.get("judgment") or {}
    return "\n".join([
        "# Critique",
        "## Quality",
        f"Heuristic quality {j.get('quality', 0):.2f}. "
        + (candidate.get("reasons") or ["Scored from source tier and specificity."])[0],
        "",
        "## Practicality",
        f"Heuristic practicality {j.get('practicality', 0):.2f}. "
        + ("A named artifact is present." if candidate.get("artifacts")
           else "No fetchable artifact was extracted from the stored text."),
        "",
        "## Feasibility",
        f"Heuristic feasibility {j.get('feasibility', 0):.2f}. "
        "Treat this as a prior until a model pass or a human reads the source.",
        "",
        "## Usefulness",
        f"Heuristic usefulness {j.get('usefulness', 0):.2f}. "
        f"Readiness {j.get('readiness', 0):.2f} → verdict `{j.get('verdict', 'watch')}`.",
        "",
        "## Contradictions",
        "- None seen in the heuristic pass — it does not compare sources in prose.",
        "",
        f"scores: {_scores(j)}",
    ])


def _decision(verdict: str) -> str:
    return {
        "adopt": "adopt",
        "research": "spike",
        "watch": "watch",
        "skip": "skip",
    }.get(verdict, "watch")


def _adapt(candidate: dict[str, Any], pages: dict[str, str]) -> str:
    j = candidate.get("judgment") or {}
    verdict = j.get("verdict") or "watch"
    decision = _decision(verdict)
    artifacts = candidate.get("artifacts") or []
    title = candidate.get("title") or "this discovery"
    first_step = (
        f"Open the primary source and confirm `{artifacts[0]}` still resolves."
        if artifacts else
        f"Read the primary source for {title} and write down one artifact you can fetch."
    )
    return "\n".join([
        "# Adapt",
        "## Decision",
        f"**{decision}** — readiness {j.get('readiness', 0):.2f} "
        f"(quality {j.get('quality', 0):.2f}, practicality {j.get('practicality', 0):.2f}, "
        f"feasibility {j.get('feasibility', 0):.2f}, usefulness {j.get('usefulness', 0):.2f}).",
        "",
        "## Who this is for",
        "A practitioner already working in this category who can spare a spike, "
        "not a general reader.",
        "",
        "## Prerequisites",
        _bullets(
            artifacts or ["Unknown — no artifact extracted."],
            "Unknown.",
        ),
        "",
        "## First week",
        f"1. {first_step}",
        "2. Write the smallest experiment that would falsify the main claim.",
        "3. Run that experiment or stop and file this as watch.",
        "4. If it works, note the hardware, license, and one integration point.",
        "",
        "## Integration",
        "Unknown until an artifact is in hand. Do not schedule a rewrite of a "
        "production path on a heuristic brief.",
        "",
        "## Risks",
        "- The stored body is a feed snippet, not the paper or repo.",
        "- Heuristic scores can promote a well-worded announcement.",
        "- A closed or vanished artifact wastes the week.",
        "",
        "## Done looks like",
        "- A yes/no on whether the artifact runs on your hardware.",
        "- A written license and cost note.",
        "- Either a follow-up spike or an explicit skip.",
        "",
        "---",
        "*Templated brief — install an Ollama chat model for a written adaptation plan.*",
    ])


def _lint(candidate: dict[str, Any], pages: dict[str, str]) -> str:
    missing = [slug for slug in ("source", "claims", "critique", "adapt") if slug not in pages]
    j = candidate.get("judgment") or {}
    decision = _decision(j.get("verdict") or "watch")
    return "\n".join([
        "# Lint",
        "## Contradictions",
        "- None isolated — this lint is templated and does not re-read the sources.",
        "",
        "## Orphans",
        _bullets(missing, "All four prior pages are on file."),
        "",
        "## Unknowns",
        "- License, hardware envelope, and independent reproduction.",
        "",
        "## Stale or weak",
        "- Every score on this brief is a heuristic prior until a model pass lands.",
        "",
        "## Suggested next questions",
        "- Is there a public repo or weight file that the feed omitted?",
        "- What is the smallest hardware that runs the advertised result?",
        "- What would a failed spike look like in the first afternoon?",
        "",
        "## Final verdict",
        f"**{decision}** — matches the Adapt page. "
        "Upgrade only after a model pass or a human reading the artifact.",
    ])
