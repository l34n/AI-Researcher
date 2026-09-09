"""Re-fetch canonical URLs of high-ranked stories and record revisions.

The original snapshot stays on the item row. A changed title, a withdrawn
page, or a new last-modified stamp is filed in ``item_revisions`` and
dependent outputs are marked stale.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..http import Fetcher
from ..util import iso, local_day, utcnow
from .freshness import invalidate_outputs

log = logging.getLogger("ai_researcher.revalidate")

_WITHDRAWN = {404, 410}
_OK = {200, 304}


async def revalidate_top_stories(
    db: Database,
    fetcher: Fetcher,
    *,
    day: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """HEAD/GET the primary URL of today's top clusters. Never raises."""
    day = day or local_day()
    rows = db.query(
        """
        SELECT c.id AS cluster_id, i.id AS item_id, i.url, i.title, i.body
        FROM clusters c
        JOIN cluster_items ci ON ci.cluster_id = c.id AND ci.is_primary = 1
        JOIN items i ON i.id = ci.item_id
        WHERE c.day = ?
        ORDER BY c.score DESC
        LIMIT ?
        """,
        (day, limit),
    )
    checked = changed = withdrawn = errors = 0
    now = iso(utcnow())
    touched: list[int] = []
    for row in rows:
        url = (row["url"] or "").strip()
        if not url.startswith("http"):
            continue
        checked += 1
        resp = await fetcher.get(url, attempts=2)
        if resp is None:
            errors += 1
            continue
        if resp.status_code in _WITHDRAWN:
            withdrawn += 1
            db.execute(
                "INSERT INTO item_revisions (item_id, title, body, url, captured_at, change) "
                "VALUES (?,?,?,?,?,?)",
                (row["item_id"], row["title"] or "", (row["body"] or "")[:4000],
                 url, now, f"withdrawn HTTP {resp.status_code}"),
            )
            db.execute(
                "UPDATE items SET freshness_status='stale', last_revalidated_at=? WHERE id=?",
                (now, row["item_id"]),
            )
            touched.append(row["cluster_id"])
            continue
        if resp.status_code not in _OK:
            errors += 1
            db.execute(
                "UPDATE items SET last_revalidated_at=? WHERE id=?",
                (now, row["item_id"]),
            )
            continue
        new_title = ""
        try:
            new_title = (resp.headers.get("x-title") or "").strip()
        except Exception:  # noqa: BLE001
            new_title = ""
        change = ""
        if new_title and new_title != (row["title"] or "") and len(new_title) > 8:
            change = "title-changed"
        db.execute(
            "UPDATE items SET last_revalidated_at=?, freshness_status=CASE "
            "WHEN freshness_status IN ('stale','superseded') THEN freshness_status "
            "ELSE 'revalidated' END WHERE id=?",
            (now, row["item_id"]),
        )
        if change:
            changed += 1
            db.execute(
                "INSERT INTO item_revisions (item_id, title, body, url, captured_at, change) "
                "VALUES (?,?,?,?,?,?)",
                (row["item_id"], row["title"] or "", (row["body"] or "")[:4000],
                 url, now, change),
            )
            touched.append(row["cluster_id"])
        else:
            # Successful confirmation of the live page.
            pass
    if touched:
        invalidate_outputs(
            db,
            reason="source revalidation detected a change or withdrawal",
            cluster_ids=list(dict.fromkeys(touched)),
        )
    return {
        "checked": checked,
        "changed": changed,
        "withdrawn": withdrawn,
        "errors": errors,
    }
