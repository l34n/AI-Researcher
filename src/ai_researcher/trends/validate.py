"""Treat every generated briefing as untrusted data.

The cleanup pass in ``brief._clean`` strips wrappers. This module decides
whether what remains is a *valid* briefing: the required shape, no prompt
echo, no invented Ready-to-build items, and a word budget. A failed
validation must never become the dashboard's authoritative brief.

Two repairs are applied rather than rejected, because the rest of the brief
is sound and the repaired text is what ships: trailing commentary past the
word budget is trimmed, and a ``## Ready to build`` section written on a day
with no gated items is dropped. Both are recorded as warnings, and the
dropped section also sets ``hallucinated_ready`` so the benchmark keeps
measuring the behaviour. A Ready section that names a title that is not
gated is still an error: a repair cannot tell which bullets are invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..util import normalize_text, truncate

PROMPT_VERSION = "brief-v5"
HARNESS_VERSION = "validate-v2"
WORD_LIMIT = 360

REQUIRED_HEADINGS = (
    "The one thing",
    "Also today",
    "Worth a closer look",
)
OPTIONAL_HEADINGS = ("Ready to build",)

# Instruction / reasoning leftovers that mean the model did not produce a brief.
_ECHO_MARKERS = (
    "begin untrusted",
    "end untrusted",
    "write exactly this structure",
    "you write a daily",
    "output only the finished",
    "below are today's top",
    "ready to build (already quality-gated",
    "rules: no preamble",
    "start your reply with",
    "the user wants",
    "the user asked",
    "okay, the user",
    "alright, i need",
    "let me think",
    "here is the briefing i will",
    "as an ai",
    "i cannot browse",
    "i don't have access to",
)
_REASONING_LEAD = re.compile(
    r"^\s*(okay|alright|sure|first|now|let me|i(?:'ll| will| need)|"
    r"the user|here(?:'s| is)|to summar|thinking|analysis)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_BULLET = re.compile(r"^[-*]\s+")
_NUMBERED = re.compile(r"^\d+[.)]\s+")
_BOLD_LEAD = re.compile(r"^\*\*[^*\n]+\*\*")
_BULLET_SECTIONS = {"also today", "worth a closer look", "ready to build"}
_BOLD_LABEL = re.compile(r"\*\*([^*]+)\*\*")
_WORD = re.compile(r"\S+")


@dataclass
class ValidationResult:
    ok: bool
    markdown: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    word_count: int = 0
    ready_titles: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    # The model recommended something that was never gated. True both when
    # the section was dropped (ok) and when it was rejected (not ok).
    hallucinated_ready: bool = False
    # An ungated Ready section was removed from ``markdown``.
    ready_dropped: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "word_count": self.word_count,
            "hallucinated_ready": self.hallucinated_ready,
            "ready_dropped": self.ready_dropped,
            "prompt_version": PROMPT_VERSION,
            "harness_version": HARNESS_VERSION,
        }


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def _heading_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").casefold()).strip()


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return ``[(heading, body), ...]`` in document order."""
    sections: list[tuple[str, str]] = []
    current_title = ""
    body: list[str] = []
    for line in (markdown or "").splitlines():
        match = _HEADING.match(line.strip())
        if match:
            if current_title or body:
                sections.append((current_title, "\n".join(body).strip()))
            current_title = match.group(2).strip()
            body = []
        else:
            body.append(line)
    if current_title or body:
        sections.append((current_title, "\n".join(body).strip()))
    return sections


def bullets_of(body: str) -> list[str]:
    return [ln.strip() for ln in (body or "").splitlines() if _BULLET.match(ln.strip())]


def drop_section(markdown: str, title: str) -> str:
    """Remove every section headed ``title`` (heading and body), keep the rest."""
    key = _heading_key(title)
    out: list[str] = []
    skipping = False
    for line in (markdown or "").splitlines():
        match = _HEADING.match(line.strip())
        if match:
            skipping = _heading_key(match.group(2)) == key
        if not skipping:
            out.append(line)
    text = "\n".join(out).rstrip()
    return text + ("\n" if (markdown or "").endswith("\n") else "")


def promote_bullets(markdown: str) -> str:
    """Give the bullet sections their markers back.

    The prompt asks for "**bold label** then one sentence" per bullet, and
    models regularly render exactly that, one per line, without a leading
    ``- `` (qwen3:32b does it every time). Numbered lists are the other
    common variant. The content is complete; only the marker is missing, so
    add it inside the three bullet sections and nowhere else.
    """
    out: list[str] = []
    in_bullets = False
    section_start = -1  # index in `out` of the current bullet section's heading
    section_has_bullet = False

    def _promote_plain_lines() -> None:
        # A bullet section with text but not one bullet: the model wrote the
        # content as prose (gpt-4.1-nano and solar-pro4 do this when asked
        # for "1 bullet"). Each non-empty line becomes a bullet.
        if section_start < 0 or section_has_bullet:
            return
        for i in range(section_start + 1, len(out)):
            text = out[i].strip()
            if text and not _HEADING.match(text) and not _BULLET.match(text):
                out[i] = "- " + text

    for line in (markdown or "").splitlines():
        stripped = line.strip()
        match = _HEADING.match(stripped)
        if match:
            _promote_plain_lines()
            in_bullets = _heading_key(match.group(2)) in _BULLET_SECTIONS
            section_start = len(out) if in_bullets else -1
            section_has_bullet = False
            out.append(line)
            continue
        if in_bullets and stripped and not _BULLET.match(stripped):
            if _NUMBERED.match(stripped):
                out.append("- " + _NUMBERED.sub("", stripped, count=1))
                section_has_bullet = True
                continue
            if _BOLD_LEAD.match(stripped):
                out.append("- " + stripped)
                section_has_bullet = True
                continue
        elif in_bullets and _BULLET.match(stripped):
            section_has_bullet = True
        out.append(line)
    _promote_plain_lines()
    return "\n".join(out) + ("\n" if (markdown or "").endswith("\n") else "")


def _title_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) >= 3}


def title_matches(candidate: str, allowed: Iterable[str], *, min_overlap: float = 0.55) -> bool:
    """True when ``candidate`` names one of the gated records, not a free invention."""
    cand = normalize_text(candidate)
    if not cand:
        return False
    cand_tokens = _title_tokens(candidate)
    for title in allowed:
        allowed_norm = normalize_text(title)
        if not allowed_norm:
            continue
        if allowed_norm in cand or cand in allowed_norm:
            return True
        allowed_tokens = _title_tokens(title)
        if not allowed_tokens or not cand_tokens:
            continue
        overlap = len(cand_tokens & allowed_tokens) / len(allowed_tokens)
        if overlap >= min_overlap:
            return True
    return False


def _looks_like_echo(text: str) -> list[str]:
    errors: list[str] = []
    low = (text or "").casefold()
    for marker in _ECHO_MARKERS:
        if marker in low:
            errors.append(f"prompt-echo: {marker!r}")
    if "<think>" in low or "</think>" in low:
        errors.append("prompt-echo: leaked thinking tags")
    return errors


def _trim_to_word_limit(markdown: str, limit: int = WORD_LIMIT) -> tuple[str, int, bool]:
    """Drop trailing commentary until the document fits, never the first heading."""
    text = (markdown or "").rstrip()
    count = word_count(text)
    if count <= limit:
        return text, count, False
    lines = text.splitlines()
    while lines and word_count("\n".join(lines)) > limit:
        # Prefer dropping a trailing non-heading paragraph over a required section.
        if _HEADING.match(lines[-1].strip() or ""):
            break
        lines.pop()
    trimmed = "\n".join(lines).rstrip()
    return trimmed, word_count(trimmed), True


def extract_ready_labels(body: str) -> list[str]:
    """Bold labels from Ready-to-build bullets (decision word stripped)."""
    labels: list[str] = []
    for bullet in bullets_of(body):
        bolds = _BOLD_LABEL.findall(bullet)
        if bolds:
            label = bolds[-1] if bolds[0].casefold() in {"adopt", "spike", "watch", "skip"} and len(bolds) > 1 else bolds[0]
            if label.casefold() in {"adopt", "spike", "watch", "skip"}:
                rest = _BULLET.sub("", bullet)
                rest = _BOLD_LABEL.sub("", rest, count=1).strip(" —–-")
                label = rest or label
            labels.append(label.strip())
        else:
            labels.append(_BULLET.sub("", bullet)[:140])
    return labels


def attach_provenance(
    markdown: str,
    stories: list[dict[str, Any]],
    ready: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map each section/bullet to the story or research record it most likely cites.

    Matching is conservative: a sentence that names a story label, a cluster
    id token like ``[S3]``, or a gated title is attributed; unmatched prose is
    left unattributed rather than guessed.
    """
    by_token = {f"s{i}": s for i, s in enumerate(stories, 1)}
    sections = []
    for heading, body in split_sections(markdown):
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_story(story: dict[str, Any]) -> None:
            key = f"cluster:{story.get('id')}"
            if key in seen:
                return
            seen.add(key)
            refs.append({
                "kind": "story",
                "cluster_id": story.get("id"),
                "item_ids": list(story.get("item_ids") or []),
                "label": story.get("label") or "",
            })

        def add_ready(item: dict[str, Any]) -> None:
            key = f"ready:{item.get('id') or item.get('item_id')}"
            if key in seen:
                return
            seen.add(key)
            refs.append({
                "kind": "research",
                "research_id": item.get("id"),
                "item_id": item.get("item_id"),
                "title": item.get("title") or "",
            })

        haystack = f"{heading}\n{body}".casefold()
        for token, story in by_token.items():
            if f"[{token}]" in haystack or f"(s{token[1:]})" in haystack:
                add_story(story)
        for story in stories:
            label = story.get("label") or ""
            if label and normalize_text(label)[:40] and normalize_text(label)[:40] in normalize_text(body):
                add_story(story)
        if _heading_key(heading) == "ready to build":
            for item in ready:
                title = item.get("title") or ""
                if title and title_matches(body, [title], min_overlap=0.4):
                    add_ready(item)
        sections.append({
            "heading": heading,
            "kind": "recommendation" if _heading_key(heading) == "ready to build" else "report",
            "refs": refs,
        })
    return {
        "stories": [
            {
                "token": f"S{i}",
                "cluster_id": s.get("id"),
                "item_ids": list(s.get("item_ids") or []),
                "label": s.get("label") or "",
            }
            for i, s in enumerate(stories, 1)
        ],
        "ready": [
            {
                "research_id": r.get("id"),
                "item_id": r.get("item_id"),
                "title": r.get("title") or "",
                "decision": r.get("decision") or r.get("verdict") or "",
            }
            for r in ready
        ],
        "sections": sections,
    }


def validate_brief(
    markdown: str,
    *,
    stories: list[dict[str, Any]] | None = None,
    ready: list[dict[str, Any]] | None = None,
    strict_counts: bool = True,
) -> ValidationResult:
    """Validate (and lightly repair) a model briefing.

    ``strict_counts`` is True for model output. Deterministic fallbacks may
    have fewer bullets when the day is empty; they still must start with
    ``## The one thing`` and must not invent a Ready section.
    """
    stories = stories or []
    ready = ready or []
    errors: list[str] = []
    warnings: list[str] = []
    text = promote_bullets((markdown or "").strip())
    if not text:
        return ValidationResult(False, "", errors=["empty briefing"])

    errors.extend(_looks_like_echo(text))

    sections = split_sections(text)
    if not sections or _heading_key(sections[0][0]) != "the one thing":
        errors.append("first heading must be '## The one thing'")

    by_key = {_heading_key(title): (title, body) for title, body in sections}
    for required in REQUIRED_HEADINGS:
        if _heading_key(required) not in by_key:
            errors.append(f"missing section: {required}")

    also = by_key.get("also today")
    if also:
        n = len(bullets_of(also[1]))
        if strict_counts and not (4 <= n <= 6):
            errors.append(f"Also today needs 4-6 bullets, got {n}")
        elif n < 1:
            errors.append("Also today has no bullets")

    closer = by_key.get("worth a closer look")
    if closer:
        n = len(bullets_of(closer[1]))
        if strict_counts and not (2 <= n <= 3):
            errors.append(f"Worth a closer look needs 2-3 bullets, got {n}")
        elif n < 1:
            errors.append("Worth a closer look has no bullets")

    ready_section = by_key.get("ready to build")
    ready_titles_found: list[str] = []
    hallucinated_ready = False
    ready_dropped = False
    allowed_titles = [r.get("title") or "" for r in ready if r.get("title")]
    if ready_section and not allowed_titles:
        # Nothing was gated, so every bullet is invented. The rest of the
        # brief is unaffected: drop the section, ship the rest, and report
        # the behaviour through ``hallucinated_ready``.
        ready_titles_found = extract_ready_labels(ready_section[1])
        text = drop_section(text, "Ready to build")
        hallucinated_ready = ready_dropped = True
        warnings.append(
            "dropped Ready to build: no gated items"
            + (f" ({truncate(ready_titles_found[0], 80)})" if ready_titles_found else "")
        )
    elif ready_section:
        bullets = bullets_of(ready_section[1])
        if strict_counts and not (2 <= len(bullets) <= 3) and allowed_titles:
            # One gated item may produce a single honest bullet; that is fine.
            if len(bullets) < 1:
                errors.append("Ready to build has no bullets")
            elif len(bullets) > 3:
                errors.append(f"Ready to build has {len(bullets)} bullets; max 3")
        ready_titles_found = extract_ready_labels(ready_section[1])
        for label in ready_titles_found:
            if not title_matches(label, allowed_titles) and not title_matches(
                ready_section[1], allowed_titles
            ):
                errors.append(f"ungated recommendation: {truncate(label, 80)}")
                hallucinated_ready = True
                break
        else:
            for label in ready_titles_found:
                if not title_matches(label, allowed_titles):
                    # The whole section names a gated title even if the bold
                    # label is a decision word — accept if the body matches.
                    if not title_matches(ready_section[1], allowed_titles, min_overlap=0.35):
                        errors.append(f"ungated recommendation: {truncate(label, 80)}")
                        hallucinated_ready = True
                        break

    # Lead-in narration that survived cleanup.
    lead = sections[0][1] if sections else ""
    if lead and _REASONING_LEAD.match(lead.splitlines()[0] if lead.splitlines() else ""):
        errors.append("reasoning lead-in before the briefing")

    trimmed, count, was_trimmed = _trim_to_word_limit(text)
    if was_trimmed:
        warnings.append(f"trimmed to {WORD_LIMIT} words")
        text = trimmed
        count = word_count(text)
    if count > WORD_LIMIT:
        errors.append(f"word limit {WORD_LIMIT} exceeded ({count})")

    provenance = attach_provenance(text, stories, ready)
    ok = not errors
    return ValidationResult(
        ok=ok,
        markdown=text if ok else markdown,
        errors=errors,
        warnings=warnings,
        word_count=count,
        ready_titles=ready_titles_found,
        provenance=provenance,
        hallucinated_ready=hallucinated_ready,
        ready_dropped=ready_dropped,
    )


_CITE = re.compile(r"\[S(\d+)\](?!\()")


def annotate_citations(markdown: str, stories: list[dict[str, Any]]) -> str:
    """Turn bare ``[S1]`` tokens into in-app story links."""
    if not stories or not markdown:
        return markdown
    by_n = {i: s for i, s in enumerate(stories, 1)}

    def repl(match: re.Match[str]) -> str:
        n = int(match.group(1))
        story = by_n.get(n) or {}
        cid = story.get("id")
        if not cid:
            return match.group(0)
        return f"[S{n}](/story/{int(cid)})"

    return _CITE.sub(repl, markdown)
