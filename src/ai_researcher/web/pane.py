"""Display-time assembly for the daily brief, reader, and agent handoff.

The stored brief is prose. Links, metrics, and "why this is here" are
derived from clusters and research at read time so an old brief still
clicks through, and so a citation token cannot invent a destination.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import CATEGORY_LABELS
from ..db import Database
from ..trends.brief import _collect_ready
from ..trends.validate import annotate_citations, split_sections
from ..util import local_day, truncate
from . import queries as Q

_HEADING_KEY = re.compile(r"[^a-z0-9]+")
_QUIET_CATS = {"research", "tooling-oss", "benchmark-eval", "model-release"}



def _key(title: str) -> str:
    return _HEADING_KEY.sub(" ", (title or "").casefold()).strip()


def metric_line(obj: dict[str, Any]) -> str:
    """One quiet chip: `3 src · 0.81 · research`. Not four meter bars."""
    parts: list[str] = []
    n = int(obj.get("source_count") or 0)
    if n > 1:
        parts.append(f"{n} src")
    else:
        name = ""
        primary = obj.get("primary") or {}
        if isinstance(primary, dict):
            name = primary.get("source_name") or ""
        name = name or obj.get("source_name") or ""
        if name:
            parts.append(name)
        elif n == 1:
            parts.append("1 src")
    readiness = float(obj.get("readiness") or 0)
    if readiness:
        parts.append(f"{readiness:.2f}")
    verdict = (obj.get("decision") or obj.get("verdict") or "").strip()
    if verdict:
        parts.append(verdict)
    return " · ".join(parts)


def closer_look_why(story: dict[str, Any]) -> str:
    """One sentence: why this quieter item still deserves a slot."""
    n = int(story.get("source_count") or 1)
    quality = float(story.get("quality") or 0)
    usefulness = float(story.get("usefulness") or 0)
    artifacts = [a for a in (story.get("artifacts") or []) if a]
    category = story.get("category") or ""
    ranking = (story.get("ranking_why") or "").strip()
    reasons = [str(r).strip() for r in (story.get("reasons") or []) if str(r).strip()]

    bits: list[str] = []
    if artifacts:
        bits.append("it names a fetchable artifact")
    if n <= 1 and quality >= 0.55:
        bits.append("it is under-covered relative to its quality")
    elif n <= 2 and usefulness >= 0.62:
        bits.append("usefulness is high while coverage is still quiet")
    if category == "research":
        bits.append("this is primary research, not a recap")
    elif category == "tooling-oss":
        bits.append("it is runnable tooling")
    elif category == "benchmark-eval":
        bits.append("it is a measurable eval, not a take")
    for reason in reasons:
        low = reason.lower()
        if any(token in low for token in ("artifact", "corroborat", "scored from")):
            continue
        if reason.casefold() not in " ".join(bits).casefold():
            bits.append(reason)
            break
    if not bits and ranking:
        return f"On the board because {ranking}."
    if not bits:
        label = CATEGORY_LABELS.get(category, "this story")
        return f"On the board as {label.lower()}, with {n} source{'s' if n != 1 else ''}."
    head, *rest = bits[:2]
    sentence = head[0].upper() + head[1:] if head else head
    if rest:
        sentence = f"{sentence}, and {rest[0]}"
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _closer_signal(story: dict[str, Any], *, ready_ids: set[int]) -> float:
    rid = int(story.get("research_id") or 0)
    if rid and rid in ready_ids:
        return -1.0
    score = 0.0
    if story.get("artifacts"):
        score += 3.0
    if (story.get("category") or "") in _QUIET_CATS:
        score += 1.5
    n = int(story.get("source_count") or 1)
    if n <= 2 and float(story.get("quality") or 0) >= 0.55:
        score += 2.0
    if float(story.get("usefulness") or 0) >= 0.6:
        score += 1.0
    if (story.get("verdict") or "") == "skip":
        score -= 2.0
    score += float(story.get("readiness") or 0)
    return score


def _section_body(markdown: str, heading: str) -> str:
    wanted = _key(heading)
    for title, body in split_sections(markdown or ""):
        if _key(title) == wanted:
            return body
    return ""


def _story_href(story: dict[str, Any]) -> str:
    cid = int(story.get("id") or 0)
    if cid:
        return f"/story/{cid}"
    primary = story.get("primary") or {}
    iid = int((primary.get("id") if isinstance(primary, dict) else 0) or 0)
    return f"/read/{iid}" if iid else ""


def _ready_href(item: dict[str, Any]) -> str:
    rid = int(item.get("id") or item.get("research_id") or 0)
    if rid:
        return f"/adapt/{rid}#handoff"
    iid = int(item.get("item_id") or 0)
    return f"/read/{iid}" if iid else ""


def _row(
    *,
    label: str,
    href: str,
    metrics: str,
    why: str = "",
    prose: str = "",
) -> dict[str, str]:
    return {
        "label": label,
        "href": href,
        "metrics": metrics,
        "why": why,
        "prose": prose,
    }


def build_frame(
    stories: list[dict[str, Any]],
    ready: list[dict[str, Any]],
    stored: dict[str, Any] | None,
) -> dict[str, Any]:
    """Structured brief rows. Empty lists are omitted by the template."""
    markdown = (stored or {}).get("markdown") or ""
    lead_story = stories[0] if stories else None
    lead_prose = annotate_citations(_section_body(markdown, "The one thing"), stories)
    if lead_story and not lead_prose.strip():
        lead_prose = lead_story.get("summary") or lead_story.get("label") or ""

    lead = None
    if lead_story:
        lead = _row(
            label=lead_story.get("label") or "Today's lead",
            href=_story_href(lead_story),
            metrics=metric_line(lead_story),
            why=lead_story.get("ranking_why") or "",
            prose=lead_prose,
        )

    also: list[dict[str, str]] = []
    for story in stories[1:7]:
        also.append(_row(
            label=story.get("label") or "Story",
            href=_story_href(story),
            metrics=metric_line(story),
            prose=truncate(story.get("summary") or "", 140),
        ))

    ready_ids = {int(r.get("id") or 0) for r in ready if r.get("id")}
    ranked = sorted(
        (( _closer_signal(s, ready_ids=ready_ids), s) for s in stories[1:]),
        key=lambda pair: pair[0],
        reverse=True,
    )
    closer: list[dict[str, str]] = []
    seen: set[int] = set()
    for signal, story in ranked:
        if signal <= 0:
            continue
        closer.append(_row(
            label=story.get("label") or "Item",
            href=_story_href(story),
            metrics=metric_line(story),
            why=closer_look_why(story),
        ))
        seen.add(int(story.get("id") or 0))
        if len(closer) >= 3:
            break
    if len(closer) < 2:
        for story in stories[1:]:
            sid = int(story.get("id") or 0)
            if sid in seen:
                continue
            if int(story.get("research_id") or 0) in ready_ids:
                continue
            closer.append(_row(
                label=story.get("label") or "Item",
                href=_story_href(story),
                metrics=metric_line(story),
                why=closer_look_why(story),
            ))
            if len(closer) >= 2:
                break

    ready_rows: list[dict[str, str]] = []
    for item in ready[:4]:
        href = _ready_href(item)
        if not href:
            continue
        ready_rows.append(_row(
            label=item.get("title") or "Ready item",
            href=href,
            metrics=metric_line(item),
            prose=truncate(
                item.get("excerpt")
                or f"{item.get('decision') or 'spike'} — first experiment is in the build guide.",
                160,
            ),
        ))

    return {
        "lead": lead,
        "also": also,
        "closer": closer,
        "ready": ready_rows,
    }


def assemble_brief(db: Database, day: str | None = None) -> dict[str, Any] | None:
    """Stored prose plus clickable rows. Works before a brief has been written."""
    stored = Q.get_brief(db, day)
    frame_day = stored["day"] if stored else (day or local_day())
    stories = Q.top_stories(db, day=frame_day, limit=12)
    ready = _collect_ready(db)
    by_id = {b["id"]: b for b in Q.ready_briefs(db, limit=12)}
    for item in ready:
        extra = by_id.get(item.get("id"))
        if extra:
            item["excerpt"] = extra.get("excerpt") or ""
    frame = build_frame(stories, ready, stored)
    if stored:
        stored["frame"] = frame
        stored["use_markdown"] = not bool(frame.get("lead"))
        return stored
    if not stories:
        return None
    return {
        "day": frame_day,
        "markdown": "",
        "model": "",
        "age": "",
        "is_today": True,
        "fingerprint": "",
        "prompt_version": "",
        "harness_version": "",
        "validation_ok": True,
        "validation_errors": [],
        "fallback": True,
        "provenance": {},
        "stale": False,
        "invalidation_reason": "",
        "frame": frame,
        "use_markdown": False,
    }


def wiki_sections(markdown: str) -> dict[str, str]:
    return {_key(title): body for title, body in split_sections(markdown or "") if title}


def handoff_markdown(brief: dict[str, Any]) -> str:
    """One document a practitioner can paste into an agent to run the spike."""
    pages = {p.get("slug"): p.get("markdown") or "" for p in (brief.get("pages") or [])}
    adapt = pages.get("adapt") or ""
    sections = wiki_sections(adapt)
    item_id = int(brief.get("item_id") or 0)
    cluster_id = int(brief.get("cluster_id") or 0)
    rid = int(brief.get("id") or 0)
    title = brief.get("title") or "Untitled"
    decision = brief.get("decision") or brief.get("verdict") or "spike"
    original = brief.get("url") or ""
    artifacts = [str(a) for a in (brief.get("artifacts") or []) if a]
    reasons = [str(r) for r in (brief.get("reasons") or []) if r]

    in_app: list[str] = []
    if item_id:
        in_app.append(f"- Source notes — [{title}](/read/{item_id})")
    if cluster_id:
        in_app.append(f"- Cluster — [all reports](/story/{cluster_id})")
    if rid:
        in_app.append(f"- Wiki — [Adapt](/adapt/{rid})")

    metrics = (
        f"readiness {float(brief.get('readiness') or 0):.2f} · "
        f"Q {float(brief.get('quality') or 0):.2f} · "
        f"P {float(brief.get('practicality') or 0):.2f} · "
        f"F {float(brief.get('feasibility') or 0):.2f} · "
        f"U {float(brief.get('usefulness') or 0):.2f}"
    )
    why_bits = reasons[:3] or ["cleared the quality gate with a complete Adapt page"]

    lines = [
        f"# Ready to deploy: {title}",
        "",
        "Copy this document into an agent. Prefer the in-app sources below. "
        "Only fetch the original URL to confirm an artifact still resolves.",
        "",
        "## Original",
        original or "unknown",
        "",
        "## In-app sources",
        *(in_app or ["- None linked."]),
        "",
        "## Goal",
        sections.get("decision") or brief.get("excerpt") or f"**{decision}** this week.",
        "",
        "## Why this is ready",
        metrics,
        "",
        *(f"- {bit}" for bit in why_bits),
        "",
        "## Who this is for",
        sections.get("who this is for") or "A practitioner who can spare a spike this week.",
        "",
        "## Prerequisites",
        sections.get("prerequisites") or (
            "\n".join(f"- {a}" for a in artifacts)
            if artifacts else
            "- Unknown — confirm the artifact still exists."
        ),
        "",
        "## First experiment",
        sections.get("first week")
        or "Open the in-app source, confirm the artifact, run the smallest smoke test.",
        "",
        "## Done looks like",
        sections.get("done looks like") or "- A yes/no on whether the artifact runs on your hardware.",
        "",
        "## Risks",
        sections.get("risks") or "- The stored body may be a feed snippet, not the repo.",
        "",
        "## Artifacts",
    ]
    if artifacts:
        lines.extend(f"- {a}" for a in artifacts)
    else:
        lines.append("- None named.")
    lines.extend(["", "## Adapt (full)", adapt.strip() or "_No Adapt page on file._", ""])
    return "\n".join(lines)
