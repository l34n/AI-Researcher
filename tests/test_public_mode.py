"""AIR_PUBLIC_MODE: a read-only public instance of the same dashboard.

One codebase, two processes, one SQLite file. The private instance keeps its
token, its controls and its ingest; the public one serves Dashboard, Firehose,
Search and Adapt to anyone and can write nothing.

Every public assertion here has a private counterpart beside it. A check that
"the save button is absent" proves nothing on its own — it would also pass if
the button had been deleted for everybody — so each one is paired with the
private-mode reading that shows the affordance is still there by default.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_researcher.config import Settings
from ai_researcher.db import Database
from ai_researcher.util import content_hash, iso, local_day, url_hash, utcnow
from ai_researcher.web import app as app_module
from ai_researcher.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "sources.yaml"
APP_JS = ROOT / "src" / "ai_researcher" / "web" / "static" / "app.js"

# The private instance's own pages and its every write endpoint. The public
# instance registers none of them.
PRIVATE_PAGES = ["/saved", "/sources", "/runs", "/health"]
WRITE_ENDPOINTS = [
    "/api/refresh",
    "/api/save/1",
    "/api/brief/regenerate",
    "/api/feedback/1",
    "/api/sources/seed-src/mute",
    "/refresh",
]
# Everything the public instance must still serve, minus the three id routes
# that need a row to exist.
PUBLIC_SURFACE = [
    "/",
    "/feed",
    "/search",
    "/adapt",
    "/healthz",
    "/readyz",
    "/api/status",
    "/api/stories",
    "/api/rising",
    "/api/research",
    "/static/app.js",
]
# Tables a request could plausibly write. Counted before and after a read
# sweep to show the public instance leaves the shared file alone.
COUNTED_TABLES = ("items", "saved", "feedback", "source_controls", "runs", "sources", "briefs")


def _seed(data_dir: Path) -> int:
    """One source, one item and today's brief — enough for every affordance."""
    db = Database(data_dir / "airesearch.db")
    now = utcnow()
    url = "https://example.com/seed"
    db.execute(
        "INSERT INTO sources (key, name, kind, tier, weight) VALUES (?,?,?,?,?)",
        ("seed-src", "Seed Source", "rss", "news", 1.0),
    )
    cur = db.execute(
        "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
        "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("seed-src", "seed:1", url, url, url_hash(url), content_hash("Seed item", ""),
         "Seed item", "", "Seed body", iso(now - timedelta(hours=2)), iso(now), 0, 0, "{}"),
    )
    item_id = int(cur.lastrowid)
    db.execute(
        "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
        "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (item_id, "Seed summary", "model-release", "[]", "[]", 0.6, "", "", iso(now)),
    )
    db.execute(
        "INSERT INTO briefs (day, markdown, model, created_at) VALUES (?,?,?,?)",
        (local_day(), "- Seed brief bullet", "test", iso(now)),
    )
    db.close()
    return item_id


def _row_counts(data_dir: Path) -> dict[str, int]:
    """Row counts read on a throwaway read-only connection."""
    conn = sqlite3.connect(f"file:{data_dir / 'airesearch.db'}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in COUNTED_TABLES}
    finally:
        conn.close()


def _settings(data_dir: Path, *, public: bool, token: str = "") -> Settings:
    return Settings(
        data_dir=data_dir,
        access_token=token,
        sources_path=SOURCES,
        public_mode=public,
    )


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A seeded database both instances open, as they do in deployment."""
    monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "0")
    monkeypatch.setenv("AIR_PUBLIC_MODE", "")
    data = tmp_path / "data"
    data.mkdir()
    _seed(data)
    return data


@pytest.fixture
def item_id(data_dir: Path) -> int:
    conn = sqlite3.connect(f"file:{data_dir / 'airesearch.db'}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT id FROM items LIMIT 1").fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def public_client(data_dir: Path):
    # A token is set on purpose: the public instance must ignore it.
    with TestClient(create_app(_settings(data_dir, public=True, token="s3cret"))) as c:
        yield c


@pytest.fixture
def private_client(data_dir: Path):
    with TestClient(create_app(_settings(data_dir, public=False))) as c:
        yield c


class TestFlag:
    """`Settings.public_mode`, off unless the environment says otherwise."""

    def test_default_is_off(self):
        assert Settings().public_mode is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_environment_turns_it_on(self, raw, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_PUBLIC_MODE", raw)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe"])
    def test_everything_else_leaves_it_off(self, raw, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_PUBLIC_MODE", raw)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is False

    def test_absent_environment_leaves_it_off(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AIR_PUBLIC_MODE", raising=False)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is False


class TestAccess:
    """Clause 1: the token guard is skipped, even when a token is configured."""

    def test_public_needs_no_token(self, public_client: TestClient):
        for path in ("/", "/feed", "/search", "/adapt", "/api/status"):
            assert public_client.get(path).status_code == 200, path

    def test_private_still_enforces_its_token(self, data_dir: Path):
        with TestClient(create_app(_settings(data_dir, public=False, token="s3cret"))) as c:
            assert c.get("/").status_code == 401
            assert c.get("/", params={"k": "s3cret"}).status_code == 200


class TestReadOnlyMethods:
    """Clause 2: anything but GET or HEAD is 403, in middleware, on every path."""

    @pytest.mark.parametrize(
        "path",
        ["/", "/feed", "/api/status", "/api/save/1", "/refresh", "/static/app.js",
         "/healthz", "/no-such-path"],
    )
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_public_refuses_every_write_method(self, public_client: TestClient, method, path):
        assert public_client.request(method, path).status_code == 403

    def test_public_lets_get_and_head_through(
        self, public_client: TestClient, private_client: TestClient
    ):
        # HEAD is not rejected by the read-only middleware. FastAPI does not
        # register HEAD for a @app.get route, so it answers 405 — the same
        # answer the private instance gives, which is the point: the middleware
        # changed nothing for a non-write method.
        assert public_client.get("/").status_code == 200
        head = public_client.head("/").status_code
        assert head != 403
        assert head == private_client.head("/").status_code

    def test_private_still_accepts_a_post(self, private_client: TestClient, item_id: int):
        assert private_client.post(f"/api/save/{item_id}").json() == {"saved": True}
        assert private_client.post(f"/api/save/{item_id}").json() == {"saved": False}


class TestRouteSurface:
    """Clauses 3 and 4: which routes exist at all in each mode."""

    def test_public_does_not_serve_the_private_pages(self, public_client: TestClient):
        for path in PRIVATE_PAGES:
            assert public_client.get(path).status_code == 404, path

    def test_public_does_not_register_the_write_endpoints(self, public_client: TestClient):
        # A GET on a registered POST-only route is 405; on an unregistered path
        # it is 404. That difference is how "not registered at all" is read.
        for path in WRITE_ENDPOINTS:
            assert public_client.get(path).status_code == 404, path

    def test_private_still_serves_all_of_them(self, private_client: TestClient):
        for path in PRIVATE_PAGES:
            assert private_client.get(path).status_code == 200, path
        for path in WRITE_ENDPOINTS:
            assert private_client.get(path).status_code == 405, path

    def test_public_serves_the_read_surface(self, public_client: TestClient, item_id: int):
        for path in PUBLIC_SURFACE:
            assert public_client.get(path).status_code == 200, path
        assert public_client.get(f"/read/{item_id}").status_code == 200

    @pytest.mark.parametrize("path", ["/read/not-an-int", "/story/not-an-int", "/adapt/not-an-int"])
    def test_public_keeps_the_id_routes(self, public_client: TestClient, path):
        # 422 means the route matched and rejected the id; 404 would mean the
        # route is gone. Both read as "not 200", so only the code separates them.
        assert public_client.get(path).status_code == 422


class TestNoStartupWrites:
    """Clause 5: nothing is written at startup or on a read sweep."""

    def test_public_does_not_sync_sources(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        calls: list[object] = []
        monkeypatch.setattr(app_module, "sync_sources", lambda *a, **k: calls.append(a))
        create_app(_settings(data_dir, public=True))
        assert calls == []

    def test_private_still_syncs_sources(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch):
        calls: list[object] = []
        monkeypatch.setattr(app_module, "sync_sources", lambda *a, **k: calls.append(a))
        create_app(_settings(data_dir, public=False))
        assert len(calls) == 1

    def test_public_startup_and_reads_change_no_rows(self, data_dir: Path):
        before = _row_counts(data_dir)
        with TestClient(create_app(_settings(data_dir, public=True))) as c:
            for path in PUBLIC_SURFACE:
                c.get(path)
        assert _row_counts(data_dir) == before

    def test_private_startup_writes_the_source_catalog(self, data_dir: Path):
        before = _row_counts(data_dir)
        with TestClient(create_app(_settings(data_dir, public=False))):
            pass
        assert _row_counts(data_dir)["sources"] > before["sources"]

    def test_public_never_schedules_the_refresh_loop(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "5")
        app = create_app(_settings(data_dir, public=True))
        with TestClient(app):
            assert app.state.refresh_task is None

    def test_private_still_schedules_the_refresh_loop(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "5")
        app = create_app(_settings(data_dir, public=False))
        with TestClient(app):
            assert app.state.refresh_task is not None
