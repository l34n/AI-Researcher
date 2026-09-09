"""SQLite backup and restore. Saved research is never part of retention pruning."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .db import Database


def backup_path(data_dir: Path, when: datetime | None = None) -> Path:
    when = when or datetime.now(timezone.utc)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return Path(data_dir) / "backups" / f"airesearch-{stamp}.db"


def sqlite_backup(src: Path, dest: Path) -> Path:
    """Online backup via SQLite's backup API so WAL writers are not copied torn."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    return dest


def integrity_check(path: Path) -> tuple[bool, str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        msg = row[0] if row else "no result"
        return msg == "ok", str(msg)
    finally:
        conn.close()


def backup_database(db: Database, dest: Path | None = None) -> dict[str, str | bool]:
    dest = dest or backup_path(db.path.parent)
    sqlite_backup(db.path, dest)
    ok, msg = integrity_check(dest)
    if not ok:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"backup failed integrity check: {msg}")
    return {"path": str(dest), "ok": True, "integrity": msg}


def restore_database(src: Path, dest: Path) -> dict[str, str | bool]:
    ok, msg = integrity_check(src)
    if not ok:
        raise RuntimeError(f"refusing to restore a corrupt backup: {msg}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sqlite_backup(src, dest)
    return {"path": str(dest), "ok": True, "integrity": "ok"}
