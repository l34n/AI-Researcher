"""Karpathy-style multi-turn deep research.

Only stories that already cleared the readiness gate get this treatment.
Each qualifying story becomes a tiny wiki: five pages, written in order,
each turn reading the index and the pages already filed.

The human role is the threshold (`AIR_RESEARCH_THRESHOLD`) and the budget.
The model role is the bookkeeping — ingest, cross-page consistency, and an
implementation brief that would otherwise vanish into a one-shot summary.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..config import Settings
from ..db import Database, jdump, jload
from ..enrich.judge import RESEARCH_READINESS, verdict_of
from ..enrich.ollama import OllamaClient
from ..progress import RunProgress
from ..sanitize import fence
from ..util import iso, local_day, truncate, utcnow
from .fallback import render_page
from .schema import SCHEMA, TURNS, adapt_complete, index_markdown

log = logging.getLogger("ai_researcher.research")

DECISION_RE = re.compile(
    r"\b(adopt|spike|watch|skip)\b", re.IGNORECASE
)
_DECISION_HEADING = re.compile(
    r"^#{1,3}\s+(?:Decision|Final verdict)\b", re.IGNORECASE
)
_NEXT_HEADING = re.compile(r"^#{1,3}\s+\S")
_BOLD_DECISION = re.compile(
    r"\*\*\s*(adopt|spike|watch|skip)\s*\*\*", re.IGNORECASE
)
SCORES_RE = re.compile(
    r"scores:\s*Q=([0-9.]+)\s*P=([0-9.]+)\s*F=([0-9.]+)\s*U=([0-9.]+)",
    re.IGNORECASE,
)


def _raw_block(candidate: dict[str, Any]) -> str:
    lines = []
    for i, item in enumerate(candidate.get("items") or [], 1):
        lines.append(
            f"{i}. [{item.get('source_name') or item.get('source_key') or '?'}]\n"
            + fence("TITLE", item.get("title") or "", limit=140) + "\n"
            + fence("URL", item.get("url") or "", limit=200) + "\n"
            + fence("BODY", item.get("summary") or item.get("body") or "", limit=280)
        )
    return "\n".join(lines) or "(no source items)"


def _prompt(turn: dict[str, Any], candidate: dict[str, Any], pages: dict[str, str]) -> str:
    judgment = candidate.get("judgment") or {}
    prior = ""
    if pages:
        prior = (
            "## Wiki so far (untrusted model text)\n"
            + "\n\n".join(
                fence(f"PAGE_{slug.upper()}", body, limit=4000)
                for slug, body in pages.items()
            )
            + "\n\n"
        )
    return (
        fence("SUBJECT", candidate.get("title") or "Untitled", limit=160) + "\n"
        f"Category: {candidate.get('category') or 'unknown'} · "
        f"sources: {candidate.get('source_count') or 1} · "
        f"heuristic verdict: {judgment.get('verdict', 'watch')} "
        f"(readiness {judgment.get('readiness', 0):.2f})\n"
        f"Artifacts already extracted: "
        f"{', '.join(candidate.get('artifacts') or []) or 'none'}\n\n"
        f"{index_markdown(pages)}\n\n"
        f"## Raw sources (immutable)\n{_raw_block(candidate)}\n\n"
        + prior
        + turn["instruction"]
    )


def parse_decision(markdown: str, default: str = "watch") -> str:
    """Last explicit call in a Decision / Final verdict block wins.

    The rest of the page talks about skip/watch as risks ('file this as
    watch', 'an explicit skip'). Those must not overwrite the heading.
    """
    text = markdown or ""
    blocks: list[str] = []
    taking = False
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _DECISION_HEADING.match(stripped):
            if taking:
                blocks.append("\n".join(buf))
            taking = True
            buf = []
            continue
        if taking and _NEXT_HEADING.match(stripped):
            blocks.append("\n".join(buf))
            taking = False
            buf = []
            continue
        if taking:
            buf.append(line)
    if taking:
        blocks.append("\n".join(buf))

    haystack = "\n".join(blocks) if blocks else text
    bold = list(_BOLD_DECISION.finditer(haystack))
    if bold:
        return bold[-1].group(1).lower()
    matches = list(DECISION_RE.finditer(haystack))
    if not matches:
        return default
    return matches[-1].group(1).lower()


def parse_scores(markdown: str) -> dict[str, float] | None:
    match = SCORES_RE.search(markdown or "")
    if not match:
        return None
    try:
        q, p, f, u = (max(0.0, min(1.0, float(x))) for x in match.groups())
    except (TypeError, ValueError):
        return None
    return {"quality": q, "practicality": p, "feasibility": f, "usefulness": u}


def decision_to_verdict(decision: str, current: str) -> str:
    """Lint may downgrade. It may not promote skip → adopt on one adjective."""
    mapped = {"adopt": "adopt", "spike": "research", "watch": "watch", "skip": "skip"}
    incoming = mapped.get(decision, current)
    order = ("skip", "watch", "research", "adopt")
    if incoming not in order or current not in order:
        return current
    # Allow a one-step move either way; block a jump from skip to adopt.
    if abs(order.index(incoming) - order.index(current)) <= 1:
        return incoming
    return current if order.index(incoming) > order.index(current) else incoming


class DeepResearcher:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        client: OllamaClient,
        progress: RunProgress | None = None,
    ):
        self.settings = settings
        self.db = db
        self.client = client
        self.progress = progress

    async def run(self, limit: int | None = None, *, force: bool = False) -> dict[str, Any]:
        if self.progress:
            self.progress.update(
                stage="research",
                detail="Selecting stories that cleared the readiness gate",
                current="",
                done=0,
                total=0,
                active=[],
            )
        budget = limit if limit is not None else self.settings.research_budget
        candidates = self._candidates(budget, force=force)
        if not candidates:
            return {"researched": 0, "llm": 0, "fallback": 0, "skipped": 0}

        llm_ready = await self.client.probe()
        time_budget = float(self.settings.research_time_budget)
        started = time.monotonic()
        researched = llm_pages = fallback_pages = 0

        for i, candidate in enumerate(candidates, 1):
            if time.monotonic() - started > time_budget:
                log.warning(
                    "deep research hit its %.0fs time budget after %s stories",
                    time_budget, researched,
                )
                break
            title = truncate(candidate.get("title") or "", 80)
            if self.progress:
                self.progress.update(
                    stage="research",
                    detail=f"Deep research · {i - 1}/{len(candidates)}",
                    current=title,
                    done=i - 1,
                    total=len(candidates),
                    active=[title],
                )
            used_llm = await self._research(candidate, llm_ready=llm_ready)
            researched += 1
            if used_llm:
                llm_pages += 1
            else:
                fallback_pages += 1

        elapsed = time.monotonic() - started
        log.info(
            "deep research completed %s stories (%s model, %s fallback) in %.0fs",
            researched, llm_pages, fallback_pages, elapsed,
        )
        return {
            "researched": researched,
            "llm": llm_pages,
            "fallback": fallback_pages,
            "candidates": len(candidates),
            "model": self.client.model_for(premium=True, role="research") if llm_ready else "",
            "model_seconds": round(elapsed, 1),
        }

    def relink_clusters(self, day: str | None = None) -> int:
        """Point research rows at today's rebuilt clusters.

        Clusters are deleted and rewritten every run; research is keyed on
        the primary item so the brief survives. This just restores the join.
        """
        day = day or local_day()
        # Prefer a primary membership, but a brief often lives on the quieter
        # implementable member. Clusters are rebuilt every run and SET NULL
        # the old cluster_id, so skipping non-primaries orphaned those briefs.
        rows = self.db.query(
            """
            SELECT r.id,
                   (
                       SELECT ci.cluster_id
                       FROM cluster_items ci
                       JOIN clusters c ON c.id = ci.cluster_id AND c.day = ?
                       WHERE ci.item_id = r.item_id
                       ORDER BY ci.is_primary DESC
                       LIMIT 1
                   ) AS cluster_id
            FROM research r
            """,
            (day,),
        )
        linked = [(row["cluster_id"], row["id"]) for row in rows if row["cluster_id"]]
        with self.db.tx() as conn:
            for cluster_id, research_id in linked:
                conn.execute(
                    "UPDATE research SET cluster_id=? WHERE id=?",
                    (cluster_id, research_id),
                )
        return len(linked)

    def _candidates(self, limit: int, *, force: bool) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        threshold = float(self.settings.research_threshold)
        # Prefer today's clustered stories. An unclustered high-readiness
        # item can still qualify via the UNION so a first install is not stuck.
        existing = "" if force else "AND i.id NOT IN (SELECT item_id FROM research WHERE status='complete')"
        rows = self.db.query(
            f"""
            SELECT i.id AS item_id, i.title, i.body, i.url,
                   COALESCE(e.summary, '') AS summary,
                   COALESCE(e.category, '') AS category,
                   COALESCE(e.entities, '[]') AS entities,
                   COALESCE(e.why, '') AS why,
                   COALESCE(s.name, i.source_key) AS source_name,
                   COALESCE(s.tier, 'news') AS tier,
                   j.quality, j.practicality, j.feasibility, j.usefulness,
                   j.readiness, j.verdict, j.reasons, j.artifacts,
                   (
                       SELECT c.id FROM cluster_items ci
                       JOIN clusters c ON c.id = ci.cluster_id
                       WHERE ci.item_id = i.id AND ci.is_primary = 1
                       ORDER BY c.day DESC LIMIT 1
                   ) AS cluster_id,
                   (
                       SELECT c.source_count FROM cluster_items ci
                       JOIN clusters c ON c.id = ci.cluster_id
                       WHERE ci.item_id = i.id
                       ORDER BY c.day DESC LIMIT 1
                   ) AS source_count
            FROM judgments j
            JOIN items i ON i.id = j.item_id
            LEFT JOIN enrichment e ON e.item_id = i.id
            LEFT JOIN sources s ON s.key = i.source_key
            WHERE j.readiness >= ? AND j.verdict IN ('research', 'adopt')
              {existing}
            ORDER BY j.readiness DESC, COALESCE(i.published_at, i.fetched_at) DESC
            LIMIT ?
            """,
            (threshold, limit),
        )
        return [self._hydrate(row) for row in rows]

    def _hydrate(self, row) -> dict[str, Any]:
        cluster_id = row["cluster_id"]
        if cluster_id:
            members = self.db.query(
                """
                SELECT i.id, i.title, i.body, i.url,
                       COALESCE(e.summary, i.body) AS summary,
                       COALESCE(e.why, '') AS why,
                       COALESCE(s.name, i.source_key) AS source_name,
                       i.source_key
                FROM cluster_items ci
                JOIN items i ON i.id = ci.item_id
                LEFT JOIN enrichment e ON e.item_id = i.id
                LEFT JOIN sources s ON s.key = i.source_key
                WHERE ci.cluster_id = ?
                ORDER BY ci.is_primary DESC, i.engagement DESC
                LIMIT 8
                """,
                (cluster_id,),
            )
            items = [dict(m) for m in members]
        else:
            items = [{
                "id": row["item_id"],
                "title": row["title"],
                "body": row["body"],
                "url": row["url"],
                "summary": row["summary"],
                "why": row["why"],
                "source_name": row["source_name"],
                "source_key": "",
            }]
        artifacts = jload(row["artifacts"], [])
        judgment = {
            "quality": float(row["quality"]),
            "practicality": float(row["practicality"]),
            "feasibility": float(row["feasibility"]),
            "usefulness": float(row["usefulness"]),
            "readiness": float(row["readiness"]),
            "verdict": row["verdict"],
            "reasons": jload(row["reasons"], []),
        }
        return {
            "item_id": row["item_id"],
            "cluster_id": cluster_id,
            "title": row["title"],
            "category": row["category"],
            "entities": jload(row["entities"], []),
            "source_count": int(row["source_count"] or 1),
            "artifacts": artifacts,
            "reasons": judgment["reasons"],
            "judgment": judgment,
            "items": items,
        }

    async def _research(self, candidate: dict[str, Any], *, llm_ready: bool) -> bool:
        now = iso(utcnow())
        existing = self.db.one(
            "SELECT id FROM research WHERE item_id=?",
            (candidate["item_id"],),
        )
        if existing:
            research_id = existing["id"]
            self.db.execute(
                "UPDATE research SET status='running', cluster_id=?, updated_at=? WHERE id=?",
                (candidate.get("cluster_id"), now, research_id),
            )
            self.db.execute("DELETE FROM research_pages WHERE research_id=?", (research_id,))
        else:
            cur = self.db.execute(
                """
                INSERT INTO research (
                    cluster_id, item_id, title, status, readiness, verdict,
                    decision, model, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    candidate.get("cluster_id"),
                    candidate["item_id"],
                    candidate["title"] or "Untitled",
                    "running",
                    candidate["judgment"]["readiness"],
                    candidate["judgment"]["verdict"],
                    "",
                    "",
                    now,
                    now,
                ),
            )
            research_id = cur.lastrowid

        pages: dict[str, str] = {}
        used_llm = False
        model_name = ""

        for turn_i, turn in enumerate(TURNS, 1):
            markdown = ""
            if llm_ready:
                text = await self.client.generate_text(
                    _prompt(turn, candidate, pages),
                    system=SCHEMA,
                    num_predict=turn["num_predict"],
                    temperature=0.25,
                    timeout=max(240.0, float(self.client.settings.ollama_timeout)),
                    premium=True,
                    role="research",
                )
                if text and len(text) > 80:
                    from ..trends.brief import _clean

                    markdown = _clean(text)
                    if markdown:
                        used_llm = True
                        model_name = self.client.model_for(premium=True, role="research")
            if not markdown:
                markdown = render_page(turn["slug"], candidate, pages)

            pages[turn["slug"]] = markdown
            self.db.execute(
                """
                INSERT INTO research_pages (research_id, slug, title, markdown, turn, created_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(research_id, slug) DO UPDATE SET
                    markdown=excluded.markdown, turn=excluded.turn, created_at=excluded.created_at
                """,
                (research_id, turn["slug"], turn["title"], markdown, turn_i, iso(utcnow())),
            )

        decision = parse_decision(pages.get("adapt", ""), default="spike")
        lint_decision = parse_decision(pages.get("lint", ""), default=decision)
        if lint_decision != decision:
            # Lint is the last word, but only for a downgrade or a one-step move.
            decision = lint_decision
        if decision == "adopt" and not adapt_complete(pages.get("adapt", "")):
            decision = "watch"

        verdict = decision_to_verdict(decision, candidate["judgment"]["verdict"])
        scores = parse_scores(pages.get("critique", ""))
        if scores:
            readiness = round(
                0.28 * scores["quality"]
                + 0.22 * scores["practicality"]
                + 0.22 * scores["feasibility"]
                + 0.28 * scores["usefulness"],
                4,
            )
            # A critique score cannot promote a skip past the research gate
            # on its own; it can refine a story already inside the gate.
            if candidate["judgment"]["readiness"] >= RESEARCH_READINESS:
                self.db.execute(
                    """
                    UPDATE judgments SET
                        quality=?, practicality=?, feasibility=?, usefulness=?,
                        readiness=?, verdict=?
                    WHERE item_id=?
                    """,
                    (
                        scores["quality"], scores["practicality"],
                        scores["feasibility"], scores["usefulness"],
                        readiness, verdict_of(
                            readiness,
                            practicality=scores["practicality"],
                            feasibility=scores["feasibility"],
                        ),
                        candidate["item_id"],
                    ),
                )
        else:
            readiness = candidate["judgment"]["readiness"]

        self.db.execute(
            """
            UPDATE research SET
                status='complete', readiness=?, verdict=?, decision=?,
                model=?, updated_at=?
            WHERE id=?
            """,
            (readiness, verdict, decision, model_name if used_llm else "", iso(utcnow()), research_id),
        )
        return used_llm
