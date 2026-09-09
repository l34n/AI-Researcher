"""SQLite storage layer.

One file, WAL mode, no ORM. The schema is small enough to read in one sitting
and the dashboard's queries are all single-table or one join deep.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    key            TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    tier           TEXT NOT NULL DEFAULT 'news',
    weight         REAL NOT NULL DEFAULT 1.0,
    enabled        INTEGER NOT NULL DEFAULT 1,
    url            TEXT NOT NULL DEFAULT '',
    last_fetch_at  TEXT,
    last_status    TEXT,
    last_error     TEXT,
    last_new_items INTEGER NOT NULL DEFAULT 0,
    etag           TEXT,
    last_modified  TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    request_count  INTEGER NOT NULL DEFAULT 0,
    success_count  INTEGER NOT NULL DEFAULT 0,
    timeout_count  INTEGER NOT NULL DEFAULT 0,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    latency_ms_sum REAL NOT NULL DEFAULT 0,
    latency_count  INTEGER NOT NULL DEFAULT 0,
    items_returned INTEGER NOT NULL DEFAULT 0,
    items_retained INTEGER NOT NULL DEFAULT 0,
    last_content_change TEXT,
    rate_limited_until TEXT,
    status_counts  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key    TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    url           TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    url_hash      TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    author        TEXT NOT NULL DEFAULT '',
    body          TEXT NOT NULL DEFAULT '',
    published_at  TEXT,
    fetched_at    TEXT NOT NULL,
    last_fetched_at TEXT,
    last_revalidated_at TEXT,
    freshness_status TEXT NOT NULL DEFAULT 'fresh',
    relevant      INTEGER NOT NULL DEFAULT -1,
    relevance_score REAL NOT NULL DEFAULT 0,
    relevance_reason TEXT NOT NULL DEFAULT '',
    relevance_at  TEXT,
    superseded_by INTEGER,
    engagement    REAL NOT NULL DEFAULT 0,
    comments      INTEGER NOT NULL DEFAULT 0,
    meta          TEXT NOT NULL DEFAULT '{}',
    UNIQUE (source_key, external_id)
);
CREATE INDEX IF NOT EXISTS idx_items_published ON items (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_urlhash   ON items (url_hash);
CREATE INDEX IF NOT EXISTS idx_items_chash     ON items (content_hash);
CREATE INDEX IF NOT EXISTS idx_items_source    ON items (source_key);

CREATE TABLE IF NOT EXISTS enrichment (
    item_id    INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    summary    TEXT NOT NULL DEFAULT '',
    category   TEXT NOT NULL DEFAULT '',
    entities   TEXT NOT NULL DEFAULT '[]',
    tags       TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5,
    why        TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_enrich_category ON enrichment (category);

CREATE TABLE IF NOT EXISTS embeddings (
    item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vec     BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    day        TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    category   TEXT NOT NULL DEFAULT '',
    score      REAL NOT NULL DEFAULT 0,
    size       INTEGER NOT NULL DEFAULT 0,
    source_count INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT,
    last_seen  TEXT,
    entities   TEXT NOT NULL DEFAULT '[]',
    freshness_status TEXT NOT NULL DEFAULT 'fresh',
    stale      INTEGER NOT NULL DEFAULT 0,
    invalidated_reason TEXT NOT NULL DEFAULT '',
    ranking_why TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clusters_day ON clusters (day, score DESC);

CREATE TABLE IF NOT EXISTS cluster_items (
    cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    is_primary INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_cluster_items_item ON cluster_items (item_id);

CREATE TABLE IF NOT EXISTS briefs (
    day        TEXT PRIMARY KEY,
    markdown   TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    harness_version TEXT NOT NULL DEFAULT '',
    validation_ok INTEGER NOT NULL DEFAULT 1,
    validation_errors TEXT NOT NULL DEFAULT '[]',
    fallback   INTEGER NOT NULL DEFAULT 0,
    provenance TEXT NOT NULL DEFAULT '{}',
    stale      INTEGER NOT NULL DEFAULT 0,
    invalidation_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS topic_daily (
    day    TEXT NOT NULL,
    term   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 0,
    weight REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, term)
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    stats       TEXT NOT NULL DEFAULT '{}',
    log         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS saved (
    item_id   INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    saved_at  TEXT NOT NULL,
    note      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seen (
    item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judgments (
    item_id      INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    quality      REAL NOT NULL DEFAULT 0,
    practicality REAL NOT NULL DEFAULT 0,
    feasibility  REAL NOT NULL DEFAULT 0,
    usefulness   REAL NOT NULL DEFAULT 0,
    readiness    REAL NOT NULL DEFAULT 0,
    verdict      TEXT NOT NULL DEFAULT 'skip',
    reasons      TEXT NOT NULL DEFAULT '[]',
    artifacts    TEXT NOT NULL DEFAULT '[]',
    model        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judgments_readiness ON judgments (readiness DESC);
CREATE INDEX IF NOT EXISTS idx_judgments_verdict ON judgments (verdict);

CREATE TABLE IF NOT EXISTS research (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    item_id    INTEGER NOT NULL UNIQUE REFERENCES items(id) ON DELETE CASCADE,
    title      TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'running',
    readiness  REAL NOT NULL DEFAULT 0,
    verdict    TEXT NOT NULL DEFAULT '',
    decision   TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_status ON research (status, readiness DESC);

CREATE TABLE IF NOT EXISTS research_pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    research_id INTEGER NOT NULL REFERENCES research(id) ON DELETE CASCADE,
    slug        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    markdown    TEXT NOT NULL DEFAULT '',
    turn        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    UNIQUE (research_id, slug)
);

CREATE TABLE IF NOT EXISTS source_controls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key     TEXT NOT NULL DEFAULT '',
    category       TEXT NOT NULL DEFAULT '',
    muted          INTEGER NOT NULL DEFAULT 0,
    paused         INTEGER NOT NULL DEFAULT 0,
    weight_override REAL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_controls_key ON source_controls (source_key);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    INTEGER REFERENCES items(id) ON DELETE CASCADE,
    cluster_id INTEGER,
    kind       TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_item ON feedback (item_id);

CREATE TABLE IF NOT EXISTS output_invalidations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    reason     TEXT NOT NULL,
    target     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS item_revisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '',
    body        TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    captured_at TEXT NOT NULL,
    change      TEXT NOT NULL DEFAULT ''
);
"""

# NOTE: deliberately NOT `content=''`. A contentless FTS5 table cannot be
# DELETEd from, which breaks both re-indexing an item after enrichment and
# pruning old rows. Storing our own copy of the (already truncated) text costs
# a few tens of MB and makes the index behave like an ordinary table.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    title, body, summary, entities,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class Database:
    """Thread-confined SQLite connections behind a tiny helper API."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.fts_enabled = False
        self._init()

    # ── connection handling ──────────────────────────────────────────
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── schema ───────────────────────────────────────────────────────
    def _init(self) -> None:
        with self.tx() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            self._init_fts(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        if self.fts_enabled and self._needs_reindex:
            self.reindex_all()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive SQLite migrations. CREATE IF NOT EXISTS does not alter old tables."""
        self._add_columns(conn, "sources", (
            ("request_count", "INTEGER NOT NULL DEFAULT 0"),
            ("success_count", "INTEGER NOT NULL DEFAULT 0"),
            ("timeout_count", "INTEGER NOT NULL DEFAULT 0"),
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ("latency_ms_sum", "REAL NOT NULL DEFAULT 0"),
            ("latency_count", "INTEGER NOT NULL DEFAULT 0"),
            ("items_returned", "INTEGER NOT NULL DEFAULT 0"),
            ("items_retained", "INTEGER NOT NULL DEFAULT 0"),
            ("last_content_change", "TEXT"),
            ("rate_limited_until", "TEXT"),
            ("status_counts", "TEXT NOT NULL DEFAULT '{}'"),
        ))
        self._add_columns(conn, "items", (
            ("last_fetched_at", "TEXT"),
            ("last_revalidated_at", "TEXT"),
            ("freshness_status", "TEXT NOT NULL DEFAULT 'fresh'"),
            ("relevant", "INTEGER NOT NULL DEFAULT -1"),
            ("relevance_score", "REAL NOT NULL DEFAULT 0"),
            ("relevance_reason", "TEXT NOT NULL DEFAULT ''"),
            ("relevance_at", "TEXT"),
            ("superseded_by", "INTEGER"),
        ))
        self._add_columns(conn, "clusters", (
            ("freshness_status", "TEXT NOT NULL DEFAULT 'fresh'"),
            ("stale", "INTEGER NOT NULL DEFAULT 0"),
            ("invalidated_reason", "TEXT NOT NULL DEFAULT ''"),
            ("ranking_why", "TEXT NOT NULL DEFAULT ''"),
            ("confidence", "REAL NOT NULL DEFAULT 0"),
        ))
        self._add_columns(conn, "briefs", (
            ("fingerprint", "TEXT NOT NULL DEFAULT ''"),
            ("prompt_version", "TEXT NOT NULL DEFAULT ''"),
            ("harness_version", "TEXT NOT NULL DEFAULT ''"),
            ("validation_ok", "INTEGER NOT NULL DEFAULT 1"),
            ("validation_errors", "TEXT NOT NULL DEFAULT '[]'"),
            ("fallback", "INTEGER NOT NULL DEFAULT 0"),
            ("provenance", "TEXT NOT NULL DEFAULT '{}'"),
            ("stale", "INTEGER NOT NULL DEFAULT 0"),
            ("invalidation_reason", "TEXT NOT NULL DEFAULT ''"),
        ))

    @staticmethod
    def _add_columns(conn: sqlite3.Connection, table: str, columns: tuple[tuple[str, str], ...]) -> None:
        existing = {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, spec in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")

    def _init_fts(self, conn: sqlite3.Connection) -> None:
        self._needs_reindex = False
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='items_fts'"
        ).fetchone()
        if existing and "content=''" in (existing["sql"] or ""):
            # Migrate off the original contentless table, which could not be
            # deleted from. Rebuilt below from `items`, so nothing is lost.
            conn.execute("DROP TABLE items_fts")
            self._needs_reindex = True
        try:
            conn.executescript(_FTS_SCHEMA)
            self.fts_enabled = True
        except sqlite3.OperationalError:
            # SQLite built without FTS5; search degrades to LIKE.
            self.fts_enabled = False
            self._needs_reindex = False

    def reindex_all(self) -> int:
        """Rebuild the search index from items + enrichment."""
        if not self.fts_enabled:
            return 0
        rows = self.conn.execute(
            """
            SELECT i.id, i.title, i.body,
                   COALESCE(e.summary, '')  AS summary,
                   COALESCE(e.entities, '') AS entities
            FROM items i LEFT JOIN enrichment e ON e.item_id = i.id
            """
        ).fetchall()
        with self.tx() as conn:
            conn.execute("DELETE FROM items_fts")
            conn.executemany(
                "INSERT INTO items_fts (rowid, title, body, summary, entities) "
                "VALUES (?,?,?,?,?)",
                [(r["id"], r["title"], (r["body"] or "")[:4000], r["summary"],
                  r["entities"]) for r in rows],
            )
        return len(rows)

    # ── generic helpers ──────────────────────────────────────────────
    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: tuple | dict = (), default: Any = None) -> Any:
        row = self.one(sql, params)
        return row[0] if row is not None else default

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self.tx() as conn:
            return conn.execute(sql, params)

    # ── small key/value state (rotation cursors, etc.) ───────────────
    def get_kv(self, key: str, default: str = "") -> str:
        row = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, now),
        )

    # ── FTS maintenance ──────────────────────────────────────────────
    def index_item(self, item_id: int, title: str, body: str, summary: str, entities: str) -> None:
        if not self.fts_enabled:
            return
        with self.tx() as conn:
            conn.execute("DELETE FROM items_fts WHERE rowid=?", (item_id,))
            conn.execute(
                "INSERT INTO items_fts (rowid, title, body, summary, entities) VALUES (?,?,?,?,?)",
                (item_id, title, body[:4000], summary, entities),
            )

    def search_ids(self, text: str, limit: int = 200) -> list[int]:
        text = (text or "").strip()
        if not text:
            return []
        if self.fts_enabled:
            # Quote each term so user punctuation can't break FTS syntax.
            terms = [t for t in text.replace('"', " ").split() if t]
            if not terms:
                return []
            match = " ".join(f'"{t}"*' for t in terms)
            try:
                rows = self.query(
                    "SELECT rowid FROM items_fts WHERE items_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (match, limit),
                )
                return [r[0] for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{text}%"
        rows = self.query(
            "SELECT id FROM items WHERE title LIKE ? OR body LIKE ? "
            "ORDER BY published_at DESC LIMIT ?",
            (like, like, limit),
        )
        return [r[0] for r in rows]


def jload(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
