"""Read-side queries for the dashboard. No writes happen here."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from ..config import CATEGORY_LABELS
from ..db import Database, jload
from ..sanitize import href, sanitize_artifacts
from ..util import domain_of, humanize_age, local_day, parse_datetime, truncate, utcnow

_VERDICT_RANK = {"adopt": 3, "research": 2, "watch": 1, "skip": 0, "": 0}
_PAGE_PRIORITY = {"adapt": 0, "lint": 1, "source": 2, "claims": 3, "critique": 4}
_DECISION_HEADING = re.compile(r"^#{1,3}\s+Decision\b", re.IGNORECASE)

ITEM_SELECT = """
    SELECT i.id, i.title, i.url, i.author, i.published_at, i.fetched_at,
           i.engagement, i.comments, i.source_key, i.meta, i.body,
           COALESCE(i.freshness_status, 'fresh') AS freshness_status,
           COALESCE(i.relevant, 1) AS relevant,
           COALESCE(e.summary, '')     AS summary,
           COALESCE(e.category, '')    AS category,
           COALESCE(e.entities, '[]')  AS entities,
           COALESCE(e.tags, '[]')      AS tags,
           COALESCE(e.importance, 0.5) AS importance,
           COALESCE(e.why, '')         AS why,
           COALESCE(j.quality, 0)      AS quality,
           COALESCE(j.practicality, 0) AS practicality,
           COALESCE(j.feasibility, 0)  AS feasibility,
           COALESCE(j.usefulness, 0)   AS usefulness,
           COALESCE(j.readiness, 0)    AS readiness,
           COALESCE(j.verdict, '')     AS verdict,
           COALESCE(j.artifacts, '[]') AS artifacts,
           COALESCE(j.reasons, '[]')   AS reasons,
           COALESCE(r.id, 0)           AS research_id,
           COALESCE(r.decision, '')    AS research_decision,
           COALESCE(s.name, i.source_key) AS source_name,
           COALESCE(s.tier, 'news')    AS tier,
           COALESCE(s.weight, 1.0)     AS source_weight,
           (sv.item_id IS NOT NULL)    AS is_saved
    FROM items i
    LEFT JOIN enrichment e ON e.item_id = i.id
    LEFT JOIN judgments j  ON j.item_id = i.id
    LEFT JOIN research r   ON r.item_id = i.id AND r.status = 'complete'
    LEFT JOIN sources s    ON s.key = i.source_key
    LEFT JOIN saved sv     ON sv.item_id = i.id
"""


def _shape_item(row, now=None) -> dict[str, Any]:
    now = now or utcnow()
    published = parse_datetime(row["published_at"] or row["fetched_at"])
    meta = jload(row["meta"], {})
    item = {
        "id": row["id"],
        "title": row["title"] or "(untitled)",
        "url": href(row["url"]),
        "author": row["author"],
        "summary": row["summary"],
        "why": row["why"],
        "category": row["category"] or "opinion-analysis",
        "category_label": CATEGORY_LABELS.get(row["category"], "Other"),
        "entities": jload(row["entities"], []),
        "tags": jload(row["tags"], []),
        "importance": round(float(row["importance"]), 2),
        "quality": round(float(row["quality"] or 0), 2),
        "practicality": round(float(row["practicality"] or 0), 2),
        "feasibility": round(float(row["feasibility"] or 0), 2),
        "usefulness": round(float(row["usefulness"] or 0), 2),
        "readiness": round(float(row["readiness"] or 0), 2),
        "verdict": row["verdict"] or "",
        "artifacts": sanitize_artifacts(jload(row["artifacts"], [])),
        "reasons": jload(row["reasons"], []) if "reasons" in row.keys() else [],
        "research_id": int(row["research_id"] or 0),
        "research_decision": row["research_decision"] or "",
        "source_key": row["source_key"],
        "source_name": row["source_name"],
        "tier": row["tier"],
        "engagement": int(row["engagement"] or 0),
        "comments": int(row["comments"] or 0),
        "published_at": row["published_at"],
        "age": humanize_age(published, now=now),
        "freshness_status": row["freshness_status"] if "freshness_status" in row.keys() else "fresh",
        "relevant": int(row["relevant"]) if "relevant" in row.keys() and row["relevant"] is not None else 1,
        # Google News links are redirect wrappers; show who actually
        # published the piece, not news.google.com.
        "domain": meta.get("display_domain") or domain_of(row["url"]),
        "is_saved": bool(row["is_saved"]),
        "meta": meta,
        # Community items have a discussion URL distinct from the link target.
        "discussion_url": href(
            meta.get("permalink") or meta.get("hn_url") or meta.get("tweet_url") or ""
        ),
    }
    if not item["summary"]:
        body = (row["body"] or "").strip()
        item["summary"] = (body[:220] + "…") if len(body) > 220 else body
    return item


def adapt_excerpt(markdown: str, *, limit: int = 220) -> str:
    """The Adapt page's Decision paragraph — what a practitioner reads first."""
    if not markdown:
        return ""
    taking = False
    collected: list[str] = []
    for line in markdown.splitlines():
        if _DECISION_HEADING.match(line.strip()):
            taking = True
            continue
        if taking and line.startswith("#"):
            break
        if taking and line.strip():
            collected.append(line.strip())
            if len(collected) >= 2:
                break
    text = re.sub(r"\s+", " ", " ".join(collected)).strip()
    return truncate(_plain_excerpt(text), limit)


def _plain_excerpt(text: str) -> str:
    """Decision lines are markdown; story cards are not."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    return re.sub(r"[*_`]+", "", text).strip()


def _rollup_judgment(items: list[dict[str, Any]]) -> dict[str, Any]:
    """A story inherits the best member judgment, not just the primary's.

    The primary is chosen for attention (lab post, high engagement). The
    member that named a repo can easily be a quieter source — using only
    the primary would hide the thing you can actually implement.
    """
    if not items:
        return {
            "quality": 0.0, "practicality": 0.0, "feasibility": 0.0,
            "usefulness": 0.0, "readiness": 0.0, "verdict": "",
            "research_id": 0, "research_decision": "", "artifacts": [],
            "reasons": [],
        }
    scored = [i for i in items if i.get("readiness")]
    best = max(
        scored or items,
        key=lambda i: (float(i.get("readiness") or 0), _VERDICT_RANK.get(i.get("verdict") or "", 0)),
    )
    researched = [i for i in items if i.get("research_id")]
    artifacts: list[str] = []
    for item in items:
        for art in item.get("artifacts") or []:
            if art not in artifacts:
                artifacts.append(art)
    carrier = max(
        researched,
        key=lambda i: (
            float(i.get("readiness") or 0),
            _VERDICT_RANK.get(i.get("verdict") or "", 0),
        ),
    ) if researched else best
    return {
        "quality": best["quality"],
        "practicality": best["practicality"],
        "feasibility": best["feasibility"],
        "usefulness": best["usefulness"],
        "readiness": best["readiness"],
        "verdict": best["verdict"],
        "research_id": carrier["research_id"],
        "research_decision": carrier["research_decision"],
        "artifacts": artifacts[:8],
        "reasons": list(best.get("reasons") or [])[:4],
    }


def body_as_markdown(body: str) -> str:
    """Turn a stored feed body into markdown the reader can render cleanly."""
    text = (body or "").strip()
    if not text:
        return ""
    if re.search(r"(?m)^#{1,3} |\n[-*] |\n\d+\. ", text):
        return text
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(part.split()) for part in paragraphs if part.strip())


CLUSTER_SELECT = """
    SELECT c.id, c.label, c.summary, c.category, c.score, c.size,
           c.source_count, c.entities, c.first_seen, c.last_seen,
           COALESCE(c.freshness_status, 'fresh') AS freshness_status,
           COALESCE(c.stale, 0) AS stale,
           COALESCE(c.ranking_why, '') AS ranking_why,
           COALESCE(c.confidence, 0) AS confidence
    FROM clusters c
"""


def _assemble_story(db: Database, row, now) -> dict[str, Any] | None:
    item_rows = db.query(
        ITEM_SELECT + """
        JOIN cluster_items ci ON ci.item_id = i.id
        WHERE ci.cluster_id = ?
        ORDER BY ci.is_primary DESC, i.engagement DESC,
                 COALESCE(i.published_at, i.fetched_at) DESC
        """,
        (row["id"],),
    )
    items = [_shape_item(r, now) for r in item_rows]
    if not items:
        return None
    judged = _rollup_judgment(items)
    return {
        "id": row["id"],
        "label": row["label"],
        "summary": row["summary"] or items[0]["summary"],
        "category": row["category"] or "opinion-analysis",
        "category_label": CATEGORY_LABELS.get(row["category"], "Other"),
        "score": round(float(row["score"]), 3),
        "size": row["size"],
        "source_count": row["source_count"],
        "entities": jload(row["entities"], []),
        "age": humanize_age(parse_datetime(row["last_seen"]), now=now),
        "freshness_status": row["freshness_status"] or "fresh",
        "stale": bool(row["stale"]),
        "ranking_why": row["ranking_why"] or "",
        "confidence": round(float(row["confidence"] or 0), 2),
        "primary": items[0],
        "items": items,
        "others": items[1:],
        "sources": sorted({i["source_name"] for i in items}),
        "item_ids": [i["id"] for i in items],
        **judged,
        "adapt_excerpt": "",
    }


def top_stories(
    db: Database, *, day: str | None = None, limit: int = 30,
    category: str | None = None, min_sources: int = 0, ready: bool = False,
) -> list[dict[str, Any]]:
    day = day or local_day()
    now = utcnow()

    where = ["c.day = ?"]
    params: list[Any] = [day]
    if category:
        where.append("c.category = ?")
        params.append(category)
    if min_sources > 1:
        where.append("c.source_count >= ?")
        params.append(min_sources)

    # Ready is a filter, not a prefix of the scoreboard. Fetch a wider
    # window so five viral teases cannot hide the one implementable story,
    # then sort and clip after the gate.
    fetch_limit = limit if not ready else max(limit * 5, 80)
    rows = db.query(
        CLUSTER_SELECT + f"""
        WHERE {' AND '.join(where)}
        ORDER BY c.score DESC
        LIMIT ?
        """,
        tuple(params + [fetch_limit]),
    )

    stories = []
    for row in rows:
        story = _assemble_story(db, row, now)
        if not story:
            continue
        if ready and story["verdict"] not in ("research", "adopt") and not story["research_id"]:
            continue
        stories.append(story)
    if ready:
        stories.sort(
            key=lambda s: (
                _VERDICT_RANK.get(s.get("verdict") or "", 0),
                float(s.get("readiness") or 0),
                1 if s.get("research_id") else 0,
            ),
            reverse=True,
        )
        stories = stories[:limit]
    _attach_adapt_excerpts(db, stories)
    return stories


def _attach_adapt_excerpts(db: Database, stories: list[dict[str, Any]]) -> None:
    ids = [s["research_id"] for s in stories if s.get("research_id")]
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    rows = db.query(
        f"SELECT research_id, markdown FROM research_pages "
        f"WHERE slug='adapt' AND research_id IN ({placeholders})",
        tuple(ids),
    )
    excerpts = {r["research_id"]: adapt_excerpt(r["markdown"]) for r in rows}
    for story in stories:
        story["adapt_excerpt"] = excerpts.get(story.get("research_id") or 0, "")


def get_item(db: Database, item_id: int) -> dict[str, Any] | None:
    """One scraped item for the in-app reader. Body is included here only."""
    row = db.one(ITEM_SELECT + " WHERE i.id = ?", (item_id,))
    if row is None:
        return None
    now = utcnow()
    item = _shape_item(row, now)
    raw = (row["body"] or "").strip()
    item["body"] = raw
    item["body_markdown"] = body_as_markdown(raw)
    cluster = db.one(
        CLUSTER_SELECT + """
        JOIN cluster_items ci ON ci.cluster_id = c.id
        WHERE ci.item_id = ?
        ORDER BY c.day DESC, ci.is_primary DESC
        LIMIT 1
        """,
        (item_id,),
    )
    if cluster:
        item["cluster_id"] = cluster["id"]
        item["cluster_label"] = cluster["label"]
        story = _assemble_story(db, cluster, now)
        item["related"] = [
            r for r in (story["items"] if story else []) if r["id"] != item_id
        ][:8]
    else:
        item["cluster_id"] = 0
        item["cluster_label"] = ""
        item["related"] = []
    return item


def get_story(db: Database, cluster_id: int) -> dict[str, Any] | None:
    row = db.one(CLUSTER_SELECT + " WHERE c.id = ?", (cluster_id,))
    if row is None:
        return None
    story = _assemble_story(db, row, utcnow())
    if story:
        _attach_adapt_excerpts(db, [story])
    return story


def list_items(
    db: Database, *, hours: int = 48, category: str | None = None,
    source_key: str | None = None, tier: str | None = None,
    saved_only: bool = False, ids: list[int] | None = None,
    order: str = "recent", limit: int = 120, offset: int = 0,
) -> list[dict[str, Any]]:
    now = utcnow()
    where: list[str] = []
    params: list[Any] = []

    if ids is not None:
        if not ids:
            return []
        where.append(f"i.id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    else:
        where.append("COALESCE(i.published_at, i.fetched_at) >= ?")
        params.append((now - timedelta(hours=hours)).isoformat(timespec="seconds"))

    if category:
        where.append("e.category = ?")
        params.append(category)
    if source_key:
        where.append("i.source_key = ?")
        params.append(source_key)
    if tier:
        where.append("s.tier = ?")
        params.append(tier)
    if saved_only:
        where.append("sv.item_id IS NOT NULL")

    order_sql = {
        "recent": "COALESCE(i.published_at, i.fetched_at) DESC",
        "important": "e.importance DESC, COALESCE(i.published_at, i.fetched_at) DESC",
        "engagement": "i.engagement DESC",
        "ready": "COALESCE(j.readiness, 0) DESC, COALESCE(i.published_at, i.fetched_at) DESC",
    }.get(order, "COALESCE(i.published_at, i.fetched_at) DESC")

    sql = ITEM_SELECT
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_sql} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    return [_shape_item(r, now) for r in db.query(sql, tuple(params))]


def search_items(db: Database, text: str, limit: int = 100) -> list[dict[str, Any]]:
    ids = db.search_ids(text, limit=limit)
    if not ids:
        return []
    items = list_items(db, ids=ids, limit=limit)
    # Preserve relevance order from the FTS ranking.
    position = {item_id: i for i, item_id in enumerate(ids)}
    items.sort(key=lambda it: position.get(it["id"], 10**6))
    return items


def model_drops(db: Database, *, days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    """Recent model releases, newest first — the panel people check first."""
    now = utcnow()
    rows = db.query(
        ITEM_SELECT + """
        WHERE e.category = 'model-release'
          AND COALESCE(i.published_at, i.fetched_at) >= ?
        ORDER BY COALESCE(i.published_at, i.fetched_at) DESC
        LIMIT ?
        """,
        ((now - timedelta(days=days)).isoformat(timespec="seconds"), limit),
    )
    return [_shape_item(r, now) for r in rows]


def category_counts(db: Database, *, hours: int = 24) -> list[dict[str, Any]]:
    since = (utcnow() - timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = db.query(
        """
        SELECT COALESCE(e.category, 'unclassified') AS category, COUNT(*) AS n
        FROM items i
        LEFT JOIN enrichment e ON e.item_id = i.id
        WHERE COALESCE(i.published_at, i.fetched_at) >= ?
        GROUP BY category
        ORDER BY n DESC
        """,
        (since,),
    )
    return [
        {
            "category": r["category"],
            "label": CATEGORY_LABELS.get(r["category"], r["category"].replace("-", " ").title()),
            "count": r["n"],
        }
        for r in rows
    ]


def source_health(db: Database) -> list[dict[str, Any]]:
    now = utcnow()
    rows = db.query(
        """
        SELECT s.*,
               (SELECT COUNT(*) FROM items i WHERE i.source_key = s.key) AS total_items,
               (SELECT COUNT(*) FROM items i WHERE i.source_key = s.key
                  AND COALESCE(i.published_at, i.fetched_at) >= datetime('now', '-7 days')) AS week_items
        FROM sources s
        WHERE s.enabled = 1 OR COALESCE(s.last_status, '') != 'retired'
        ORDER BY
            CASE s.last_status WHEN 'error' THEN 0 WHEN 'disabled' THEN 1 ELSE 2 END,
            s.name
        """
    )
    out = []
    for r in rows:
        out.append({
            "key": r["key"],
            "name": r["name"],
            "kind": r["kind"],
            "tier": r["tier"],
            "weight": r["weight"],
            "enabled": bool(r["enabled"]),
            "url": r["url"],
            "status": r["last_status"] or "never run",
            "error": r["last_error"] or "",
            "last_fetch": humanize_age(parse_datetime(r["last_fetch_at"]), now=now)
                          if r["last_fetch_at"] else "never",
            "last_new_items": r["last_new_items"],
            "consecutive_failures": r["consecutive_failures"],
            "total_items": r["total_items"],
            "week_items": r["week_items"],
            "request_count": r["request_count"] if "request_count" in r.keys() else 0,
            "success_count": r["success_count"] if "success_count" in r.keys() else 0,
            "success_rate": (
                round(r["success_count"] / r["request_count"], 3)
                if "request_count" in r.keys() and r["request_count"]
                else None
            ),
            "timeout_count": r["timeout_count"] if "timeout_count" in r.keys() else 0,
            "retry_count": r["retry_count"] if "retry_count" in r.keys() else 0,
            "latency_ms_avg": (
                round(r["latency_ms_sum"] / r["latency_count"], 1)
                if "latency_count" in r.keys() and r["latency_count"]
                else None
            ),
            "items_returned": r["items_returned"] if "items_returned" in r.keys() else 0,
            "items_retained": r["items_retained"] if "items_retained" in r.keys() else 0,
            "last_content_change": r["last_content_change"] if "last_content_change" in r.keys() else None,
            "rate_limited_until": r["rate_limited_until"] if "rate_limited_until" in r.keys() else None,
            "status_counts": jload(r["status_counts"], {}) if "status_counts" in r.keys() else {},
        })
    return out


def recent_runs(db: Database, limit: int = 20) -> list[dict[str, Any]]:
    now = utcnow()
    rows = db.query(
        "SELECT id, started_at, finished_at, status, stats FROM runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    out = []
    for r in rows:
        stats = jload(r["stats"], {})
        ingest = stats.get("ingest") or {}
        out.append({
            "id": r["id"],
            "status": r["status"],
            "started": humanize_age(parse_datetime(r["started_at"]), now=now),
            "started_at": r["started_at"],
            "elapsed_s": stats.get("elapsed_s"),
            "new_items": ingest.get("new_items", 0),
            "sources_ok": ingest.get("ok", 0),
            "sources_failed": ingest.get("failed", 0),
            "enriched": (stats.get("enrich") or {}).get("enriched", 0),
            "clusters": (stats.get("cluster") or {}).get("clusters", 0),
            "judged": (stats.get("judge") or {}).get("judged", 0),
            "research_briefs": (stats.get("research") or {}).get("researched", 0),
            "brief_fallback": (stats.get("brief") or {}).get("fallback", False),
            "brief_validation_ok": (stats.get("brief") or {}).get("validation_ok"),
            "coverage": ingest.get("coverage") or "",
            "errors": (ingest.get("errors") or [])[:8],
            "error": stats.get("error", ""),
        })
    return out


def dashboard_stats(db: Database) -> dict[str, Any]:
    now = utcnow()
    day = local_day(now)
    last_run = db.one("SELECT started_at, status FROM runs ORDER BY id DESC LIMIT 1")
    return {
        "items_24h": db.scalar(
            "SELECT COUNT(*) FROM items WHERE COALESCE(published_at, fetched_at) >= ?",
            ((now - timedelta(hours=24)).isoformat(timespec="seconds"),), default=0,
        ),
        "items_total": db.scalar("SELECT COUNT(*) FROM items", default=0),
        "stories_today": db.scalar("SELECT COUNT(*) FROM clusters WHERE day=?", (day,), default=0),
        "sources_ok": db.scalar(
            "SELECT COUNT(*) FROM sources WHERE enabled=1 AND last_status IN ('ok','not-modified')",
            default=0,
        ),
        "sources_total": db.scalar("SELECT COUNT(*) FROM sources WHERE enabled=1", default=0),
        "sources_failing": db.scalar(
            "SELECT COUNT(*) FROM sources WHERE enabled=1 AND last_status='error'", default=0,
        ),
        "saved": db.scalar("SELECT COUNT(*) FROM saved", default=0),
        "pending_enrich": db.scalar(
            "SELECT COUNT(*) FROM items WHERE id NOT IN (SELECT item_id FROM enrichment)",
            default=0,
        ),
        "judged": db.scalar("SELECT COUNT(*) FROM judgments", default=0),
        "adopt": db.scalar("SELECT COUNT(*) FROM judgments WHERE verdict='adopt'", default=0),
        "research_ready": db.scalar(
            "SELECT COUNT(*) FROM judgments WHERE verdict IN ('research','adopt')", default=0,
        ),
        "research_briefs": db.scalar(
            "SELECT COUNT(*) FROM research WHERE status='complete'", default=0,
        ),
        "last_run": humanize_age(parse_datetime(last_run["started_at"]), now=now) if last_run else "never",
        "last_run_status": last_run["status"] if last_run else "none",
    }


def get_brief(db: Database, day: str | None = None) -> dict[str, Any] | None:
    day = day or local_day()
    row = db.one(
        "SELECT day, markdown, model, created_at, fingerprint, prompt_version, "
        "harness_version, validation_ok, validation_errors, fallback, provenance, "
        "stale, invalidation_reason FROM briefs WHERE day=?",
        (day,),
    )
    if row is None:
        row = db.one(
            "SELECT day, markdown, model, created_at, fingerprint, prompt_version, "
            "harness_version, validation_ok, validation_errors, fallback, provenance, "
            "stale, invalidation_reason FROM briefs ORDER BY day DESC LIMIT 1"
        )
    if row is None:
        return None
    from ..trends.brief import current_fingerprint

    current = ""
    try:
        current = current_fingerprint(db, day=row["day"], model=row["model"] or "")
    except Exception:  # noqa: BLE001
        current = row["fingerprint"] or ""
    stale = bool(row["stale"]) or (
        bool(row["fingerprint"]) and current and row["fingerprint"] != current
    )
    return {
        "day": row["day"],
        "markdown": row["markdown"],
        "model": row["model"],
        "age": humanize_age(parse_datetime(row["created_at"])),
        "is_today": row["day"] == (day or local_day()),
        "fingerprint": row["fingerprint"] or "",
        "prompt_version": row["prompt_version"] or "",
        "harness_version": row["harness_version"] or "",
        "validation_ok": bool(row["validation_ok"]),
        "validation_errors": jload(row["validation_errors"], []),
        "fallback": bool(row["fallback"]),
        "provenance": jload(row["provenance"], {}),
        "stale": stale,
        "invalidation_reason": row["invalidation_reason"] or "",
    }


def source_options(db: Database) -> list[dict[str, str]]:
    rows = db.query(
        "SELECT key, name FROM sources WHERE enabled=1 ORDER BY name"
    )
    return [{"key": r["key"], "name": r["name"]} for r in rows]


def list_research(
    db: Database, *, verdict: str | None = None, limit: int = 40,
) -> list[dict[str, Any]]:
    now = utcnow()
    where = ["r.status = 'complete'"]
    params: list[Any] = []
    if verdict:
        where.append("r.verdict = ?")
        params.append(verdict)
    rows = db.query(
        f"""
        SELECT r.id, r.item_id, r.cluster_id, r.title, r.readiness, r.verdict,
               r.decision, r.model, r.created_at, r.updated_at,
               COALESCE(e.category, '') AS category,
               COALESCE(e.summary, '') AS summary,
               COALESCE(j.quality, 0) AS quality,
               COALESCE(j.practicality, 0) AS practicality,
               COALESCE(j.feasibility, 0) AS feasibility,
               COALESCE(j.usefulness, 0) AS usefulness,
               COALESCE(j.artifacts, '[]') AS artifacts,
               COALESCE(p.markdown, '') AS adapt_markdown,
               i.url
        FROM research r
        JOIN items i ON i.id = r.item_id
        LEFT JOIN enrichment e ON e.item_id = i.id
        LEFT JOIN judgments j ON j.item_id = i.id
        LEFT JOIN research_pages p ON p.research_id = r.id AND p.slug = 'adapt'
        WHERE {' AND '.join(where)}
        ORDER BY CASE r.decision
                    WHEN 'adopt' THEN 0
                    WHEN 'spike' THEN 1
                    WHEN 'watch' THEN 2
                    WHEN 'skip' THEN 3
                    ELSE 4 END,
                 r.readiness DESC, r.updated_at DESC
        LIMIT ?
        """,
        tuple(params + [limit]),
    )
    out = []
    for row in rows:
        out.append({
            "id": row["id"],
            "item_id": row["item_id"],
            "cluster_id": row["cluster_id"],
            "title": row["title"],
            "url": href(row["url"]),
            "summary": row["summary"],
            "excerpt": adapt_excerpt(row["adapt_markdown"]),
            "artifacts": sanitize_artifacts(jload(row["artifacts"], [])),
            "category": row["category"] or "opinion-analysis",
            "category_label": CATEGORY_LABELS.get(row["category"], "Other"),
            "readiness": round(float(row["readiness"] or 0), 2),
            "verdict": row["verdict"] or "",
            "decision": row["decision"] or "",
            "model": row["model"] or "",
            "quality": round(float(row["quality"] or 0), 2),
            "practicality": round(float(row["practicality"] or 0), 2),
            "feasibility": round(float(row["feasibility"] or 0), 2),
            "usefulness": round(float(row["usefulness"] or 0), 2),
            "age": humanize_age(parse_datetime(row["updated_at"]), now=now),
        })
    return out


def get_research(db: Database, research_id: int) -> dict[str, Any] | None:
    row = db.one(
        """
        SELECT r.id, r.item_id, r.cluster_id, r.title, r.readiness, r.verdict,
               r.decision, r.model, r.created_at, r.updated_at, r.status,
               COALESCE(e.category, '') AS category,
               COALESCE(e.summary, '') AS summary,
               COALESCE(j.quality, 0) AS quality,
               COALESCE(j.practicality, 0) AS practicality,
               COALESCE(j.feasibility, 0) AS feasibility,
               COALESCE(j.usefulness, 0) AS usefulness,
               COALESCE(j.reasons, '[]') AS reasons,
               COALESCE(j.artifacts, '[]') AS artifacts,
               i.url
        FROM research r
        JOIN items i ON i.id = r.item_id
        LEFT JOIN enrichment e ON e.item_id = i.id
        LEFT JOIN judgments j ON j.item_id = i.id
        WHERE r.id = ?
        """,
        (research_id,),
    )
    if row is None:
        return None
    pages = db.query(
        """
        SELECT slug, title, markdown, turn
        FROM research_pages
        WHERE research_id = ?
        ORDER BY turn ASC
        """,
        (research_id,),
    )
    pages_out = sorted(
        (
            {"slug": p["slug"], "title": p["title"], "markdown": p["markdown"], "turn": p["turn"]}
            for p in pages
        ),
        key=lambda p: _PAGE_PRIORITY.get(p["slug"], 9),
    )
    return {
        "id": row["id"],
        "item_id": row["item_id"],
        "cluster_id": row["cluster_id"],
        "title": row["title"],
        "url": href(row["url"]),
        "summary": row["summary"],
        "category": row["category"] or "opinion-analysis",
        "category_label": CATEGORY_LABELS.get(row["category"], "Other"),
        "status": row["status"],
        "readiness": round(float(row["readiness"] or 0), 2),
        "verdict": row["verdict"] or "",
        "decision": row["decision"] or "",
        "model": row["model"] or "",
        "quality": round(float(row["quality"] or 0), 2),
        "practicality": round(float(row["practicality"] or 0), 2),
        "feasibility": round(float(row["feasibility"] or 0), 2),
        "usefulness": round(float(row["usefulness"] or 0), 2),
        "reasons": jload(row["reasons"], []),
        "artifacts": sanitize_artifacts(jload(row["artifacts"], [])),
        "age": humanize_age(parse_datetime(row["updated_at"])),
        "excerpt": adapt_excerpt(
            next((p["markdown"] for p in pages_out if p["slug"] == "adapt"), "")
        ),
        "pages": pages_out,
    }


def ready_briefs(db: Database, *, limit: int = 8) -> list[dict[str, Any]]:
    """Stories a practitioner should look at first — adopt, then spike."""
    briefs = list_research(db, limit=max(limit * 3, 24))
    return [
        b for b in briefs
        if b["decision"] in ("adopt", "spike")
        or b["verdict"] in ("research", "adopt")
    ][:limit]
