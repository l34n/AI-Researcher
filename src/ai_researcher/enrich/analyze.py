"""Per-item enrichment: summary, category, entities, and an importance score.

Enrichment runs in two passes, because local inference is the scarcest resource
in the whole pipeline and spending it uniformly is waste:

1. **Heuristic pass** — every unenriched item is classified, entity-tagged and
   scored by rule. Thousands per second, no model involved. This guarantees the
   dashboard is never blank and never blocked on a queue.
2. **Model pass** — only the highest-priority items, the ones that will actually
   surface on the dashboard, are re-enriched by the language model for a real
   written summary. Bounded by both an item count and a wall-clock budget, so a
   run finishes in predictable time no matter how slow the host is.

An item promoted by the model keeps `model` set, so later runs skip it and
spend their budget on newly-arrived items instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..config import CATEGORIES, Settings
from ..db import Database, jdump
from ..progress import RunProgress
from ..sanitize import UNTRUSTED_RULE, fence
from ..util import iso, truncate, utcnow
from . import heuristics as H
from .ollama import OllamaClient
from .unslop import UNSLOP_RULE, unslop_text

log = logging.getLogger("ai_researcher.enrich")

SYSTEM = (
    "You are an analyst tracking AI research, products, and industry moves. "
    "You are terse and concrete, and you never claim anything the text does not "
    "say. Reply with JSON only. "
    + UNTRUSTED_RULE + " "
    + UNSLOP_RULE
)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "why": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "entities": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number"},
    },
    "required": ["summary", "category", "importance"],
}

# Deliberately compact: on a CPU-bound host, prompt tokens cost as much as
# generated ones, and a 2000-token prompt doubles the time per item.
PROMPT = """{title}

{body}

(source: {source})

JSON keys:
"summary": ONE sentence, UNDER 25 WORDS. State what happened. Do not explain
what a technique is, do not add background, do not restate the title.
"why": why a practitioner cares, under 12 words, "" if routine.
"category": one of {categories}
"entities": up to 4 proper names (orgs, models, products).
"importance": 0.0-1.0. 0.9+ only for a frontier model release, major
acquisition, or a result that changes what is possible. 0.2-0.4 for tutorials
and opinion."""


class Enricher:
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
        # Ollama serialises generation per model, so a wide pool just builds a
        # queue of requests that each time out. Two keeps the runner fed.
        self._gate = asyncio.Semaphore(2)

    async def run(self, limit: int | None = None) -> dict[str, Any]:
        if self.progress:
            self.progress.update(
                stage="enrich",
                detail="Heuristic pass — classifying every new item",
                current="",
                done=0,
                total=0,
                active=[],
            )
        heuristic_count = self._heuristic_pass()

        llm_ready = await self.client.probe()
        promoted = 0
        elapsed = 0.0
        if llm_ready:
            promoted, elapsed = await self._model_pass(
                limit if limit is not None else self.settings.enrich_budget
            )

        pending = self.db.scalar(
            "SELECT COUNT(*) FROM enrichment WHERE model=''", default=0
        )
        return {
            "enriched": heuristic_count,
            "llm": promoted,
            "heuristic": heuristic_count,
            "awaiting_model": pending,
            "pending": self.db.scalar(
                "SELECT COUNT(*) FROM items WHERE id NOT IN (SELECT item_id FROM enrichment)",
                default=0,
            ),
            "model": self.client.chat_model if llm_ready else "",
            "model_seconds": round(elapsed, 1),
        }

    # ── pass 1: rules over everything ────────────────────────────────
    def _heuristic_pass(self) -> int:
        rows = self.db.query(
            """
            SELECT i.id, i.title, i.body, i.meta, COALESCE(s.tier, 'news') AS tier
            FROM items i
            LEFT JOIN sources s ON s.key = i.source_key
            WHERE i.id NOT IN (SELECT item_id FROM enrichment)
            ORDER BY COALESCE(i.published_at, i.fetched_at) DESC
            """
        )
        if not rows:
            return 0

        now = iso(utcnow())
        payload = []
        for row in rows:
            title, body = row["title"] or "", row["body"] or ""
            kind_hint = _kind_hint(row["meta"] or "{}")
            category, _ = H.classify(title, body, kind=kind_hint)
            entities = H.extract_entities(title, body)
            tags = H.extract_tags(title, body)
            importance = H.heuristic_importance(
                title, body, category=category, tier=row["tier"] or "news"
            )
            summary = truncate(body.strip() or title, 260)
            payload.append(
                (row["id"], summary, category, jdump(entities), jdump(tags),
                 importance, "", "", now)
            )

        with self.db.tx() as conn:
            conn.executemany(
                """
                INSERT INTO enrichment (item_id, summary, category, entities, tags,
                                        importance, why, model, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(item_id) DO NOTHING
                """,
                payload,
            )
        for row, rec in zip(rows, payload):
            self.db.index_item(row["id"], row["title"] or "", row["body"] or "", rec[1], rec[3])

        log.info("heuristic pass enriched %s items", len(payload))
        return len(payload)

    # ── pass 2: model over the items that will be seen ───────────────
    async def _model_pass(self, budget: int) -> tuple[int, float]:
        if budget <= 0:
            return 0, 0.0

        # Priority = what the dashboard would rank highest. Enriching in this
        # order means the visible top of the page is model-written even when
        # the budget covers only a fraction of the backlog.
        rows = self.db.query(
            """
            SELECT i.id, i.title, i.body, i.meta,
                   COALESCE(s.tier, 'news') AS tier,
                   COALESCE(s.name, i.source_key) AS source_name
            FROM items i
            JOIN enrichment e ON e.item_id = i.id
            LEFT JOIN sources s ON s.key = i.source_key
            WHERE e.model = ''
            ORDER BY (e.importance * COALESCE(s.weight, 1.0)) DESC,
                     COALESCE(i.published_at, i.fetched_at) DESC
            LIMIT ?
            """,
            (budget,),
        )
        if not rows:
            return 0, 0.0

        time_budget = float(self.settings.enrich_time_budget)
        started = time.monotonic()
        promoted = 0
        total = len(rows)
        if self.progress:
            self.progress.update(
                stage="enrich",
                detail=f"Model pass · 0/{total}",
                current="",
                done=0,
                total=total,
                active=[],
            )

        # Sequential-ish with a small pool. Each result is committed as it
        # lands, so hitting the time budget still keeps everything done so far.
        queue = asyncio.Queue()
        for row in rows:
            queue.put_nowait(row)

        async def worker():
            nonlocal promoted
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if time.monotonic() - started > time_budget:
                    return
                title = (row["title"] or "")[:80]
                if self.progress:
                    self.progress.update(
                        stage="enrich",
                        detail=f"Model pass · {promoted}/{total}",
                        current=title,
                        done=promoted,
                        total=total,
                    )
                try:
                    if await self._promote(row):
                        promoted += 1
                        if self.progress:
                            self.progress.update(
                                stage="enrich",
                                detail=f"Model pass · {promoted}/{total}",
                                current=title,
                                done=promoted,
                                total=total,
                            )
                except Exception as exc:  # noqa: BLE001
                    log.debug("model enrichment failed for %s: %s", row["id"], exc)

        await asyncio.gather(*(worker() for _ in range(2)))

        elapsed = time.monotonic() - started
        remaining = self.db.scalar("SELECT COUNT(*) FROM enrichment WHERE model=''", default=0)
        log.info(
            "model pass promoted %s/%s items in %.0fs (%s still heuristic-only)",
            promoted, len(rows), elapsed, remaining,
        )
        if elapsed >= time_budget:
            log.warning(
                "model pass hit its %.0fs time budget; raise AIR_ENRICH_TIME_BUDGET "
                "or use a smaller model", time_budget,
            )
        return promoted, elapsed

    async def _promote(self, row) -> bool:
        title = row["title"] or ""
        body = row["body"] or ""
        prompt = PROMPT.format(
            title=fence("TITLE", title, limit=250),
            body=fence("BODY", body, limit=900),
            source=fence("SOURCE", row["source_name"] or "", limit=80),
            categories=", ".join(CATEGORIES),
        )
        async with self._gate:
            payload = await self.client.generate_json(
                prompt, system=SYSTEM, schema=SCHEMA, num_predict=140
            )
        if not payload:
            return False

        summary = unslop_text(truncate(str(payload.get("summary") or "").strip(), 320))
        if not summary:
            return False
        why = unslop_text(truncate(str(payload.get("why") or "").strip(), 140))

        current = self.db.one(
            "SELECT category, entities, importance FROM enrichment WHERE item_id=?",
            (row["id"],),
        )
        h_category = current["category"] if current else ""
        h_importance = float(current["importance"]) if current else 0.5

        # A strong keyword signal outranks a small model's guess.
        _, h_strength = H.classify(title, body, kind=_kind_hint(row["meta"] or "{}"))
        llm_category = str(payload.get("category") or "").strip()
        category = llm_category if (llm_category in CATEGORIES and h_strength < 3) else h_category

        llm_entities = [
            str(e).strip() for e in (payload.get("entities") or [])
            if isinstance(e, (str, int)) and str(e).strip()
        ]
        entities = H.clean_entities(
            H.extract_entities(title, body) + llm_entities
        )[:8]

        try:
            llm_importance = max(0.0, min(1.0, float(payload.get("importance", h_importance))))
        except (TypeError, ValueError):
            llm_importance = h_importance
        # Blend rather than trust: small models cluster their scores near 0.7.
        importance = round(0.55 * llm_importance + 0.45 * h_importance, 4)

        self.db.execute(
            "UPDATE enrichment SET summary=?, category=?, entities=?, importance=?, "
            "why=?, model=?, created_at=? WHERE item_id=?",
            (summary, category, jdump(entities), importance, why,
             self.client.chat_model, iso(utcnow()), row["id"]),
        )
        self.db.index_item(row["id"], title, body, summary, " ".join(entities))
        return True


def _kind_hint(meta: str) -> str:
    """Recover the structural category from the stored meta blob."""
    if '"is_model_release"' in meta:
        return "hf_models"
    if '"is_release"' in meta:
        return "github_releases"
    if '"is_new_project"' in meta:
        return "github_trending"
    if '"arxiv_id"' in meta:
        return "arxiv"
    return ""
