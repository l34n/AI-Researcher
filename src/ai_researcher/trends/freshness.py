"""Freshness: when a story is current, aging, stale, or superseded.

Every item and clustered story carries published / first-seen / last-fetched
times. A stale story must not silently appear as current on the dashboard.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..db import Database, jdump
from ..util import iso, parse_datetime, utcnow

# (fresh_hours, aging_hours) by source kind / tier. After aging_hours → stale.
WINDOWS: dict[str, tuple[float, float]] = {
    "arxiv": (72, 240),
    "hf_papers": (72, 240),
    "hf_models": (48, 168),
    "github_releases": (48, 168),
    "github_trending": (36, 96),
    "hackernews": (18, 48),
    "reddit": (18, 48),
    "rss": (24, 72),
    "gnews": (18, 48),
    "x": (12, 36),
    "lab": (36, 96),
    "vendor": (36, 96),
    "research": (72, 240),
    "news": (18, 48),
    "community": (12, 36),
    "analyst": (24, 72),
    "infra": (36, 96),
}
DEFAULT_WINDOW = (24.0, 72.0)

STATES = ("fresh", "aging", "stale", "revalidated", "superseded")


def window_for(*, kind: str = "", tier: str = "") -> tuple[float, float]:
    if kind and kind in WINDOWS:
        return WINDOWS[kind]
    if tier and tier in WINDOWS:
        return WINDOWS[tier]
    return DEFAULT_WINDOW


def classify_freshness(
    *,
    published_at: datetime | None,
    fetched_at: datetime | None,
    last_revalidated_at: datetime | None = None,
    kind: str = "",
    tier: str = "",
    superseded: bool = False,
    now: datetime | None = None,
) -> str:
    if superseded:
        return "superseded"
    now = now or utcnow()
    fresh_h, aging_h = window_for(kind=kind, tier=tier)
    anchor = published_at or fetched_at
    if anchor is None:
        return "aging"
    age_h = max((now - anchor).total_seconds() / 3600.0, 0.0)
    if last_revalidated_at is not None:
        reval_h = max((now - last_revalidated_at).total_seconds() / 3600.0, 0.0)
        if reval_h <= fresh_h and age_h <= aging_h:
            return "revalidated"
    if age_h <= fresh_h:
        return "fresh"
    if age_h <= aging_h:
        return "aging"
    return "stale"


def apply_item_freshness(db: Database, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or utcnow()
    rows = db.query(
        """
        SELECT i.id, i.published_at, i.fetched_at, i.last_revalidated_at,
               i.superseded_by, COALESCE(s.kind, '') AS kind, COALESCE(s.tier, 'news') AS tier
        FROM items i
        LEFT JOIN sources s ON s.key = i.source_key
        """
    )
    counts: dict[str, int] = {s: 0 for s in STATES}
    with db.tx() as conn:
        for row in rows:
            state = classify_freshness(
                published_at=parse_datetime(row["published_at"]),
                fetched_at=parse_datetime(row["fetched_at"]),
                last_revalidated_at=parse_datetime(row["last_revalidated_at"]),
                kind=row["kind"] or "",
                tier=row["tier"] or "",
                superseded=bool(row["superseded_by"]),
                now=now,
            )
            conn.execute(
                "UPDATE items SET freshness_status=? WHERE id=?",
                (state, row["id"]),
            )
            counts[state] = counts.get(state, 0) + 1
    return {"items": sum(counts.values()), "by_state": counts}


def apply_cluster_freshness(db: Database, *, day: str | None = None) -> dict[str, Any]:
    """A story is as stale as its freshest member — a new corroborating
    report can revive it; a pile of old copies cannot."""
    from ..util import local_day

    day = day or local_day()
    rows = db.query(
        """
        SELECT c.id,
               MIN(COALESCE(i.published_at, i.fetched_at)) AS first_seen,
               MAX(COALESCE(i.published_at, i.fetched_at)) AS last_seen,
               MIN(i.freshness_status) AS worst,
               MAX(CASE i.freshness_status
                     WHEN 'fresh' THEN 4 WHEN 'revalidated' THEN 3
                     WHEN 'aging' THEN 2 WHEN 'stale' THEN 1
                     ELSE 0 END) AS best_rank
        FROM clusters c
        JOIN cluster_items ci ON ci.cluster_id = c.id
        JOIN items i ON i.id = ci.item_id
        WHERE c.day = ?
        GROUP BY c.id
        """,
        (day,),
    )
    rank_to_state = {4: "fresh", 3: "revalidated", 2: "aging", 1: "stale", 0: "stale"}
    n = 0
    with db.tx() as conn:
        for row in rows:
            state = rank_to_state.get(int(row["best_rank"] or 0), "aging")
            conn.execute(
                "UPDATE clusters SET freshness_status=?, stale=? WHERE id=?",
                (state, 1 if state == "stale" else 0, row["id"]),
            )
            n += 1
    return {"clusters": n, "day": day}


def invalidate_outputs(db: Database, *, reason: str, cluster_ids: list[int] | None = None) -> int:
    """Mark dependent briefs/research for review when source data changed."""
    now = iso(utcnow())
    marked = 0
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        with db.tx() as conn:
            conn.execute(
                f"UPDATE clusters SET invalidated_reason=?, stale=1 WHERE id IN ({placeholders})",
                tuple([reason[:300], *cluster_ids]),
            )
            marked = len(cluster_ids)
    # The daily brief is keyed by calendar day; a material change stamps it stale.
    from ..util import local_day

    db.execute(
        "UPDATE briefs SET stale=1, invalidation_reason=? WHERE day=? AND stale=0",
        (reason[:300], local_day()),
    )
    db.execute(
        "INSERT INTO output_invalidations (created_at, reason, target) VALUES (?,?,?)",
        (now, reason[:300], jdump({"clusters": cluster_ids or []})),
    )
    return marked


def detect_supersessions(db: Database) -> int:
    """Same canonical URL with a newer published time supersedes the older row."""
    rows = db.query(
        """
        SELECT url_hash, id, COALESCE(published_at, fetched_at) AS ts
        FROM items
        WHERE url_hash != ''
        ORDER BY url_hash, ts DESC
        """
    )
    latest_by_hash: dict[str, int] = {}
    updates: list[tuple[int, int]] = []
    for row in rows:
        uh = row["url_hash"]
        if uh not in latest_by_hash:
            latest_by_hash[uh] = row["id"]
            continue
        if row["id"] != latest_by_hash[uh]:
            updates.append((latest_by_hash[uh], row["id"]))
    if not updates:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "UPDATE items SET superseded_by=?, freshness_status='superseded' "
            "WHERE id=? AND COALESCE(superseded_by, 0)=0",
            [(newer, older) for newer, older in updates],
        )
    return len(updates)
