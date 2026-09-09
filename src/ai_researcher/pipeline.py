"""The ingest pipeline: fetch → store → enrich → embed → cluster → judge → research → brief.

One `run()` is a complete refresh. It is safe to call concurrently-ish (the web
UI guards with a lock) and safe to interrupt: every stage commits as it goes, so
a killed run leaves the database consistent, just less current.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import Settings, Source, load_sources
from .connectors import build_registry
from .connectors.article import hydrate_items
from .db import Database, jdump, jload
from .enrich import ChatRouter, Embedder, Enricher, Judge
from .enrich.relevance import apply_relevance
from .http import Fetcher
from .progress import RunProgress
from .research import DeepResearcher
from .trends import build_clusters, compute_daily_topics, generate_brief
from .trends.freshness import apply_cluster_freshness, apply_item_freshness, detect_supersessions
from .trends.revalidate import revalidate_top_stories
from .util import iso, utcnow

log = logging.getLogger("ai_researcher.pipeline")


class RunLockBusy(RuntimeError):
    """Another ingest run already holds the lock."""


@contextmanager
def run_lock(data_dir: Path):
    """Exclusive, cross-process ingest lock.

    Two concurrent runs are not merely wasteful: both drive Ollama at once, and
    under that contention the embedding endpoint was observed returning one
    batch's vectors for unrelated inputs, silently corrupting the index that
    clustering depends on. The systemd timer can easily fire while a manual run
    or a dashboard-triggered refresh is in flight, so the guard is a real file
    lock rather than an in-process one.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "ingest.lock"
    handle = open(path, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RunLockBusy(
                    f"another ingest run is already active (lock: {path})"
                ) from exc
            raise
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def sync_sources(db: Database, sources: list[Source]) -> None:
    """Mirror the YAML catalog into the DB, preserving per-source fetch state."""
    with db.tx() as conn:
        for src in sources:
            conn.execute(
                """
                INSERT INTO sources (key, name, kind, tier, weight, enabled, url)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                    name=excluded.name, kind=excluded.kind, tier=excluded.tier,
                    weight=excluded.weight, enabled=excluded.enabled, url=excluded.url
                """,
                (src.key, src.name, src.kind, src.tier, src.weight,
                 1 if src.enabled else 0, src.url),
            )
        # A source deleted from YAML stops being fetched but keeps its items.
        # It is marked 'retired' rather than left holding its last error, which
        # would otherwise show up forever as a failing source.
        keys = {s.key for s in sources}
        for row in conn.execute("SELECT key FROM sources").fetchall():
            if row["key"] not in keys:
                conn.execute(
                    "UPDATE sources SET enabled=0, last_status='retired', "
                    "last_error='', consecutive_failures=0 WHERE key=?",
                    (row["key"],),
                )


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        progress: RunProgress | None = None,
    ):
        self.settings = settings
        self.db = db
        self.sources = load_sources(settings)
        self.progress = progress or RunProgress(settings.data_dir / "ingest.progress.json")

    # ── stage 1: ingest ──────────────────────────────────────────────
    async def ingest(self, only: list[str] | None = None) -> dict[str, Any]:
        sync_sources(self.db, self.sources)
        paused = {
            r["source_key"]
            for r in self.db.query(
                "SELECT source_key FROM source_controls WHERE paused=1 AND source_key != ''"
            )
        }
        muted = {
            r["source_key"]
            for r in self.db.query(
                "SELECT source_key FROM source_controls WHERE muted=1 AND source_key != ''"
            )
        }
        blocked = paused | muted
        targets = [
            s for s in self.sources
            if s.enabled and (not only or s.key in only) and s.key not in blocked
        ]
        self.progress.begin_sources(len(targets))

        fetcher = Fetcher(
            self.settings.user_agent,
            concurrency=self.settings.fetch_concurrency,
        )
        registry = build_registry(self.settings, fetcher)
        try:
            results = await asyncio.gather(
                *(self._ingest_source(src, registry, fetcher) for src in targets),
                return_exceptions=True,
            )
        finally:
            await fetcher.aclose()

        stats = {"sources": len(targets), "new_items": 0, "ok": 0, "failed": 0,
                 "skipped": 0, "rate_limited": 0, "errors": [],
                 "coverage": "complete"}
        for src, result in zip(targets, results):
            if isinstance(result, BaseException):
                stats["failed"] += 1
                stats["errors"].append(f"{src.key}: {type(result).__name__}: {result}")
                self._record_source_status(src.key, "error", str(result)[:300], 0)
                continue
            stats["new_items"] += result["new"]
            if result["status"] == "ok":
                stats["ok"] += 1
            elif result["status"] in ("disabled", "not-modified"):
                stats["skipped"] += 1
            elif result["status"] == "rate-limited":
                stats["rate_limited"] += 1
                stats["failed"] += 1
                if result.get("error"):
                    stats["errors"].append(f"{src.key}: {result['error']}")
            else:
                stats["failed"] += 1
                if result.get("error"):
                    stats["errors"].append(f"{src.key}: {result['error']}")
        if stats["failed"] and stats["ok"]:
            stats["coverage"] = "partial"
        elif stats["failed"] and not stats["ok"]:
            stats["coverage"] = "error"
        elif stats["skipped"] == stats["sources"] and stats["sources"]:
            stats["coverage"] = "partial"
        return stats

    async def _ingest_source(self, src: Source, registry, fetcher) -> dict[str, Any]:
        self.progress.source_start(src.key, src.name)
        connector = registry.get(src.kind)
        if connector is None:
            self._record_source_status(src.key, "error", f"unknown kind '{src.kind}'", 0)
            self.progress.source_done(src.key, src.name, status="error", new_items=0)
            return {"status": "error", "new": 0, "error": f"unknown kind '{src.kind}'"}

        ok, reason = connector.available(src)
        if not ok:
            self._record_source_status(src.key, "disabled", reason, 0)
            self.progress.source_done(src.key, src.name, status="disabled", new_items=0)
            return {"status": "disabled", "new": 0, "error": reason}

        row = self.db.one("SELECT etag, last_modified FROM sources WHERE key=?", (src.key,))
        state = {
            "etag": row["etag"] if row else "",
            "last_modified": row["last_modified"] if row else "",
            # Rotation position for connectors that cannot sweep every target
            # in one run (Reddit's rate limit forces this).
            "cursor": self.db.get_kv(f"cursor:{src.key}"),
        }

        started = time.monotonic()
        events_before = len(getattr(fetcher, "events", []) or [])
        try:
            result = await connector.fetch(src, state)
        except Exception as exc:  # noqa: BLE001 - a broken feed must not sink the run
            log.warning("source %s raised: %s", src.key, exc)
            latency = (time.monotonic() - started) * 1000
            self._record_source_status(
                src.key, "error", f"{type(exc).__name__}: {exc}"[:300], 0,
                latency_ms=latency, requests=1,
            )
            self.progress.source_done(src.key, src.name, status="error", new_items=0)
            return {"status": "error", "new": 0, "error": str(exc)[:200]}

        status = result.status
        if result.status == "error" and (
            "429" in (result.error or "") or "rate" in (result.error or "").lower()
        ):
            status = "rate-limited"

        if status not in ("error", "rate-limited") and result.items:
            try:
                await hydrate_items(fetcher, result.items, kind=src.kind)
            except Exception as exc:  # noqa: BLE001
                log.info("article hydrate skipped for %s: %s", src.key, exc)

        latency = (time.monotonic() - started) * 1000
        events = (getattr(fetcher, "events", []) or [])[events_before:]
        http_stats = _summarise_http(events)

        if result.cursor:
            self.db.set_kv(f"cursor:{src.key}", result.cursor)

        if status in ("error", "rate-limited"):
            self._record_source_status(
                src.key, status, result.error, 0,
                latency_ms=latency, http=http_stats,
                items_returned=len(result.items),
            )
            self.progress.source_done(src.key, src.name, status=status, new_items=0)
            return {"status": status, "new": 0, "error": result.error}

        new_count = self._store(src, result.items) if result.items else 0
        self._record_source_status(
            src.key, status, result.error, new_count,
            etag=result.etag, last_modified=result.last_modified,
            latency_ms=latency, http=http_stats,
            items_returned=len(result.items), items_retained=new_count,
            content_changed=new_count > 0,
        )
        log.info(
            "%-18s %-14s %3d new / %3d fetched  (%.1fs)",
            src.key, status, new_count, len(result.items), time.monotonic() - started,
        )
        self.progress.source_done(
            src.key, src.name, status=status, new_items=new_count,
        )
        return {"status": status, "new": new_count, "error": result.error}

    def _store(self, src: Source, items) -> int:
        horizon = utcnow() - timedelta(days=self.settings.item_max_age_days)
        now_iso = iso(utcnow())
        new_count = 0

        with self.db.tx() as conn:
            for item in items:
                # A title is what the dashboard actually renders. An item
                # without one is unreadable in every view and unrankable by the
                # classifier, so it is dropped rather than shown as "(untitled)".
                if not item.title.strip() or not item.url:
                    continue
                published = item.published_at
                # Undated items are treated as "now" rather than dropped; feeds
                # that omit dates are common and usually still current.
                if published and published < horizon:
                    continue
                existing = conn.execute(
                    "SELECT id FROM items WHERE source_key=? AND external_id=?",
                    (src.key, item.external_id),
                ).fetchone()

                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO items (source_key, external_id, url, canonical_url,
                                           url_hash, content_hash, title, author, body,
                                           published_at, fetched_at, last_fetched_at,
                                           engagement, comments, meta, relevant)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,-1)
                        """,
                        (
                            src.key, item.external_id, item.url, item.url, item.uhash,
                            item.chash, item.title, item.author, item.body,
                            iso(published) if published else None, now_iso, now_iso,
                            item.engagement, item.comments, jdump(item.meta),
                        ),
                    )
                    new_count += 1
                else:
                    # Re-seeing an item is normal; only its engagement moves, and
                    # only upward — a cached listing must not erase a real count.
                    conn.execute(
                        "UPDATE items SET engagement=MAX(engagement, ?), "
                        "comments=MAX(comments, ?), last_fetched_at=? WHERE id=?",
                        (item.engagement, item.comments, now_iso, existing["id"]),
                    )
        return new_count

    def _record_source_status(
        self, key: str, status: str, error: str, new_items: int,
        *, etag: str = "", last_modified: str = "",
        latency_ms: float = 0, requests: int = 1,
        http: dict | None = None, items_returned: int = 0,
        items_retained: int = 0, content_changed: bool = False,
    ) -> None:
        failed = status in ("error", "rate-limited")
        http = http or {}
        row = self.db.one("SELECT status_counts FROM sources WHERE key=?", (key,))
        counts = jload(row["status_counts"], {}) if row else {}
        for code, n in (http.get("status") or {}).items():
            counts[str(code)] = int(counts.get(str(code), 0)) + int(n)
        timeouts = int(http.get("timeouts") or 0)
        retries = int(http.get("retries") or 0)
        reqs = max(requests, int(http.get("requests") or 0), 1 if status != "disabled" else 0)
        self.db.execute(
            """
            UPDATE sources SET
                last_fetch_at = ?,
                last_status = ?,
                last_error = ?,
                last_new_items = ?,
                etag = CASE WHEN ? != '' THEN ? ELSE etag END,
                last_modified = CASE WHEN ? != '' THEN ? ELSE last_modified END,
                consecutive_failures = CASE WHEN ? THEN consecutive_failures + 1 ELSE 0 END,
                request_count = request_count + ?,
                success_count = success_count + ?,
                timeout_count = timeout_count + ?,
                retry_count = retry_count + ?,
                latency_ms_sum = latency_ms_sum + ?,
                latency_count = latency_count + ?,
                items_returned = items_returned + ?,
                items_retained = items_retained + ?,
                last_content_change = CASE WHEN ? THEN ? ELSE last_content_change END,
                rate_limited_until = CASE WHEN ? THEN ? ELSE rate_limited_until END,
                status_counts = ?
            WHERE key = ?
            """,
            (
                iso(utcnow()), status, error[:500], new_items,
                etag, etag, last_modified, last_modified, 1 if failed else 0,
                reqs,
                1 if status in ("ok", "not-modified") else 0,
                timeouts,
                retries,
                float(latency_ms),
                1 if latency_ms else 0,
                items_returned,
                items_retained,
                1 if content_changed else 0, iso(utcnow()),
                1 if status == "rate-limited" else 0,
                iso(utcnow()) if status == "rate-limited" else "",
                jdump(counts),
                key,
            ),
        )

    # ── stage 2+: analysis ───────────────────────────────────────────
    async def analyse(self, *, brief: bool = True, force_brief: bool = False) -> dict[str, Any]:
        client = ChatRouter(self.settings)
        try:
            self.progress.update(
                stage="enrich", detail="Probing chat backends", current="", done=0, total=0, active=[],
            )
            await client.probe()
            enrich_stats = await Enricher(
                self.settings, self.db, client, progress=self.progress,
            ).run()
            embed_stats = await Embedder(
                self.db, client, progress=self.progress,
            ).run()
            if embed_stats.get("model"):
                self.db.set_kv("embed_model", str(embed_stats["model"]))
            self.progress.update(
                stage="relevance", detail="Filtering off-topic items",
                current="", done=0, total=0, active=[],
            )
            relevance_stats = apply_relevance(self.db)
            detect_supersessions(self.db)
            apply_item_freshness(self.db)
            self.progress.update(
                stage="cluster", detail="Clustering items into stories",
                current="", done=0, total=0, active=[],
            )
            cluster_stats = build_clusters(self.db)
            apply_cluster_freshness(self.db)
            self.progress.update(
                stage="topics", detail="Computing topic history",
                current="", done=0, total=0, active=[],
            )
            topic_stats = compute_daily_topics(self.db)
            judge_stats = await Judge(
                self.settings, self.db, client, progress=self.progress,
            ).run()
            researcher = DeepResearcher(
                self.settings, self.db, client, progress=self.progress,
            )
            research_stats = await researcher.run()
            researcher.relink_clusters()
            self.progress.update(
                stage="revalidate", detail="Re-checking top story sources",
                current="", done=0, total=0, active=[],
            )
            revalidate_stats: dict[str, Any] = {}
            fetcher = Fetcher(self.settings.user_agent, concurrency=4)
            try:
                revalidate_stats = await revalidate_top_stories(self.db, fetcher)
            except Exception as exc:  # noqa: BLE001
                log.info("revalidate skipped: %s", exc)
                revalidate_stats = {"error": str(exc)[:200]}
            finally:
                await fetcher.aclose()
            if brief:
                self.progress.update(
                    stage="brief", detail="Writing daily brief",
                    current="", done=0, total=0, active=[],
                )
            brief_stats = (
                await generate_brief(self.db, client, force=force_brief) if brief else {}
            )
        finally:
            await client.aclose()

        return {
            "enrich": enrich_stats,
            "embed": embed_stats,
            "relevance": relevance_stats,
            "cluster": cluster_stats,
            "topics": topic_stats,
            "judge": judge_stats,
            "research": research_stats,
            "revalidate": revalidate_stats,
            "brief": brief_stats,
            "ollama": {
                "available": client.available,
                "chat_model": client.chat_model,
                "embed_model": client.embed_model,
                "error": client.last_error,
            },
            "chat": client.describe(),
        }

    # ── full run ─────────────────────────────────────────────────────
    async def run(
        self, *, only: list[str] | None = None, skip_ingest: bool = False,
        force_brief: bool = False,
    ) -> dict[str, Any]:
        try:
            with run_lock(self.settings.data_dir):
                return await self._run(
                    only=only, skip_ingest=skip_ingest, force_brief=force_brief
                )
        except RunLockBusy as exc:
            log.warning("%s", exc)
            return {"status": "busy", "error": str(exc)}

    async def _run(
        self, *, only: list[str] | None = None, skip_ingest: bool = False,
        force_brief: bool = False,
    ) -> dict[str, Any]:
        started = utcnow()
        self.progress.start()
        cur = self.db.execute(
            "INSERT INTO runs (started_at, status) VALUES (?, 'running')", (iso(started),)
        )
        run_id = cur.lastrowid

        stats: dict[str, Any] = {}
        status = "ok"
        try:
            if skip_ingest:
                stats["ingest"] = {}
                self.progress.update(
                    stage="enrich", detail="Skipping fetch — re-analysing stored items",
                    current="", done=0, total=0, active=[],
                )
            else:
                stats["ingest"] = await self.ingest(only=only)
            stats.update(await self.analyse(force_brief=force_brief))
            self.progress.update(
                stage="prune", detail="Pruning old items", current="", done=0, total=0, active=[],
            )
            self.prune()
            ingest = stats.get("ingest") or {}
            coverage = ingest.get("coverage")
            if coverage == "partial" or (ingest.get("failed") and ingest.get("ok")):
                status = "partial"
            elif coverage == "error" and ingest.get("failed") and not ingest.get("ok"):
                status = "partial" if ingest.get("skipped") else "error"
        except Exception as exc:  # noqa: BLE001 - record the failure, don't hide it
            status = "error"
            stats["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("run failed")
        finally:
            elapsed = (utcnow() - started).total_seconds()
            stats["elapsed_s"] = round(elapsed, 1)
            self.db.execute(
                "UPDATE runs SET finished_at=?, status=?, stats=? WHERE id=?",
                (iso(utcnow()), status, jdump(stats), run_id),
            )
            self.progress.finish(ok=status in ("ok", "partial"))
        stats["run_id"] = run_id
        stats["status"] = status
        return stats

    def prune(self) -> int:
        """Drop items past the retention horizon, keeping anything starred.

        Deletes are chunked: a first prune after a long outage can match tens of
        thousands of rows, and SQLite caps the number of bound parameters in a
        single statement.
        """
        cutoff = iso(utcnow() - timedelta(days=self.settings.retention_days))
        with self.db.tx() as conn:
            ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM items WHERE COALESCE(published_at, fetched_at) < ? "
                    "AND id NOT IN (SELECT item_id FROM saved) "
                    "AND id NOT IN (SELECT item_id FROM research)",
                    (cutoff,),
                ).fetchall()
            ]
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                placeholders = ",".join("?" * len(chunk))
                # enrichment, embeddings and cluster_items cascade from here.
                conn.execute(f"DELETE FROM items WHERE id IN ({placeholders})", tuple(chunk))
                if self.db.fts_enabled:
                    conn.executemany(
                        "DELETE FROM items_fts WHERE rowid=?", [(i,) for i in chunk]
                    )
            conn.execute(
                "DELETE FROM runs WHERE id NOT IN "
                "(SELECT id FROM runs ORDER BY id DESC LIMIT 200)"
            )
        if ids:
            log.info("pruned %s items older than %s days", len(ids), self.settings.retention_days)
        return len(ids)


def _summarise_http(events: list[dict]) -> dict:
    status: dict[str, int] = {}
    timeouts = 0
    retries = 0
    for ev in events:
        code = str(ev.get("status") or 0)
        status[code] = status.get(code, 0) + 1
        if ev.get("timeout"):
            timeouts += 1
        if int(ev.get("attempt") or 1) > 1:
            retries += 1
    return {
        "status": status,
        "timeouts": timeouts,
        "retries": retries,
        "requests": len(events),
    }
