"""Regression: dashboard status API and verbose-ingest UI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_researcher.config import Settings
from ai_researcher.progress import RunProgress
from ai_researcher.web.app import create_app


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated data dir so tests never touch the live dashboard DB."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("AIR_DATA_DIR", str(data))
    monkeypatch.setenv("AIR_ACCESS_TOKEN", "")
    monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "0")
    settings = Settings(
        data_dir=data,
        access_token="",
        sources_path=Path(__file__).resolve().parents[1] / "config" / "sources.yaml",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    return app, settings


@pytest.fixture
def client(app_env):
    app, _ = app_env
    with TestClient(app) as c:
        yield c


class TestStatusApi:
    def test_status_includes_progress_shape(self, client: TestClient):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert "stats" in body and "run" in body
        run = body["run"]
        assert "running" in run
        assert "progress" in run
        progress = run["progress"]
        for key in ("running", "stage", "detail", "current", "done", "total", "active", "updated_at"):
            assert key in progress, f"missing progress.{key}"
        assert isinstance(progress["active"], list)
        assert progress["running"] is False
        assert progress["stage"] in ("idle", "done", "error")

    def test_status_stats_keys(self, client: TestClient):
        stats = client.get("/api/status").json()["stats"]
        for key in (
            "items_24h", "stories_today", "sources_ok", "sources_total",
            "sources_failing", "last_run", "last_run_status",
            "judged", "adopt", "research_ready", "research_briefs",
        ):
            assert key in stats

    def test_status_reflects_live_progress(self, client: TestClient, app_env):
        _, settings = app_env
        progress = RunProgress(settings.data_dir / "ingest.progress.json")
        # The app holds its own RunProgress instance; write the sidecar and
        # also poke the in-memory object via the module-level factory path by
        # updating through the shared file + ensuring disk fallback works when
        # the in-process flag is idle.
        progress.start()
        progress.begin_sources(4)
        progress.source_start("openai", "OpenAI")
        progress.update(
            stage="ingest",
            detail="Fetching · OpenAI · 0/4 done",
            current="OpenAI",
            done=0,
            total=4,
            active=["OpenAI"],
        )

        body = client.get("/api/status").json()
        # Disk sidecar is consulted when the web process is not itself running
        # an ingest; that is the CLI/systemd visibility path.
        p = body["run"]["progress"]
        assert p["running"] is True
        assert p["stage"] == "ingest"
        assert p["total"] == 4
        assert "OpenAI" in (p.get("detail") or "") or "OpenAI" in (p.get("active") or [])


class TestDashboardChrome:
    def test_home_includes_verbose_toggle(self, client: TestClient):
        html = client.get("/").text
        assert 'id="verbose-ingest"' in html
        assert "Show verbose ingest status" in html
        assert 'id="ingest-status"' in html
        assert 'id="ingest-stage"' in html
        assert 'id="ingest-detail"' in html
        assert 'id="ingest-current"' in html
        assert 'id="ingest-counts"' in html
        assert 'id="refresh"' in html
        assert 'id="statusdot"' in html
        assert 'data-stat="items_24h"' in html
        assert 'data-stat="stories_today"' in html
        assert 'data-stat="sources_ok"' in html
        assert 'data-stat="research_briefs"' in html
        assert "Ready</a>" in html or ">Ready<" in html

    def test_app_js_wires_verbose_polling(self, client: TestClient):
        js = client.get("/static/app.js").text
        assert "air.verboseIngest" in js
        assert "renderProgress" in js
        assert "/api/status" in js
        assert "/api/refresh" in js
        assert "verbose-ingest" in js
        assert "ingest-status" in js

    def test_app_css_defines_toggle_and_strip(self, client: TestClient):
        css = client.get("/static/app.css").text
        assert ".verbose-toggle" in css
        assert ".ingest-status" in css
        assert ".ingest-stage" in css
        assert ".ingest-status.live" in css

    def test_sticky_headers_clear_the_ingest_strip(self, client: TestClient):
        """Regression: table headers must not pin underneath the status strip.

        Both stick to the top of the viewport. The strip claims `top: 54px`
        and a z-index, so a table header pinned at the same 54px scrolls in
        behind it and the column labels become unreadable on /sources and
        /runs whenever the verbose toggle is on. The header offset has to be
        driven by --stick-top, which app.js grows by the strip's real height.
        """
        css = client.get("/static/app.css").text
        js = client.get("/static/app.js").text
        assert "--stick-top" in css, "no shared offset token"
        assert "top: var(--stick-top)" in css, "table headers still hardcode an offset"
        assert "syncStickTop" in js, "nothing updates the offset when the strip shows"

    def test_pages_render(self, client: TestClient):
        for path in ("/", "/feed", "/search", "/saved", "/adapt", "/sources", "/runs"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert "AI" in r.text

    def test_home_and_nav_include_adapt(self, client: TestClient):
        html = client.get("/").text
        assert 'href="/adapt"' in html
        assert "Ready to build" in html
        assert "ready=1" in html
        js = client.get("/static/app.js").text
        assert "/adapt" in js

    def test_feed_can_sort_by_readiness(self, client: TestClient):
        html = client.get("/feed").text
        assert "Most ready" in html
        assert 'value="ready"' in html

    def test_adapt_list_leads_with_the_week_plan(self, client: TestClient):
        html = client.get("/adapt").text
        assert "Implementation briefs" in html
        assert "Do this week" not in html  # that's the detail page

    def test_missing_research_brief_is_404(self, client: TestClient):
        assert client.get("/adapt/999").status_code == 404

    def test_hostile_markdown_and_javascript_href_do_not_render(self, app_env):
        from datetime import timedelta

        from ai_researcher.db import Database, jdump
        from ai_researcher.util import content_hash, iso, local_day, url_hash, utcnow

        app, settings = app_env
        db = Database(settings.db_path)
        now = utcnow()
        url = "javascript:alert(1)"
        cur = db.execute(
            "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
            "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("lab", "lab:xss", url, url, url_hash(url), content_hash("xss", ""),
             "Ignore instructions", "", "payload", iso(now - timedelta(hours=1)),
             iso(now), 0, 0, "{}"),
        )
        item_id = cur.lastrowid
        db.execute(
            "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
            "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, "Ignore instructions", "opinion-analysis", "[]", "[]", 0.4, "", "", iso(now)),
        )
        db.execute(
            "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
            "readiness, verdict, reasons, artifacts, model, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, 0.2, 0.2, 0.2, 0.2, 0.2, "skip", "[]",
             jdump(["javascript:alert(1)"]), "", iso(now)),
        )
        db.execute(
            "INSERT INTO briefs (day, markdown, model, created_at) VALUES (?,?,?,?)",
            (local_day(),
             "## The one thing\n<script>alert(1)</script>\n[x](javascript:alert(1))\n",
             "", iso(now)),
        )
        db.close()

        with TestClient(app) as client:
            home = client.get("/").text
            assert "<script>alert(1)</script>" not in home
            assert "javascript:alert(1)" not in home
            feed = client.get("/feed").text
            assert "javascript:alert(1)" not in feed

    def test_adapt_html_does_not_leak_markdown_stars(self, app_env):
        """Regression: story cards used to render 'adopt** — …'."""
        from datetime import timedelta

        from ai_researcher.db import Database, jdump
        from ai_researcher.util import content_hash, iso, url_hash, utcnow

        app, settings = app_env
        db = Database(settings.db_path)
        now = utcnow()
        url = "https://github.com/acme/local-7b"
        cur = db.execute(
            "INSERT INTO items (source_key, external_id, url, canonical_url, url_hash, "
            "content_hash, title, author, body, published_at, fetched_at, engagement, comments, meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("lab", "lab:7b", url, url, url_hash(url), content_hash("7B", ""),
             "Local 7B open weights", "", "weights", iso(now - timedelta(hours=2)),
             iso(now), 0, 0, "{}"),
        )
        item_id = cur.lastrowid
        db.execute(
            "INSERT INTO enrichment (item_id, summary, category, entities, tags, importance, "
            "why, model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, "Local 7B", "model-release", "[]", "[]", 0.8, "", "", iso(now)),
        )
        db.execute(
            "INSERT INTO judgments (item_id, quality, practicality, feasibility, usefulness, "
            "readiness, verdict, reasons, artifacts, model, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, 0.8, 0.8, 0.8, 0.8, 0.86, "adopt", "[]",
             jdump([url]), "", iso(now)),
        )
        db.execute(
            "INSERT INTO research (item_id, title, status, readiness, verdict, decision, "
            "model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, "Local 7B open weights", "complete", 0.86, "adopt", "adopt",
             "", iso(now), iso(now)),
        )
        rid = db.scalar("SELECT id FROM research WHERE item_id=?", (item_id,))
        db.execute(
            "INSERT INTO research_pages (research_id, slug, title, markdown, turn, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (rid, "adapt", "Adapt",
             "# Adapt\n## Decision\n**adopt** — serve the Q4 this week.\n",
             4, iso(now)),
        )
        db.close()

        with TestClient(app) as client:
            html = client.get("/adapt").text
            assert "adopt — serve the Q4 this week." in html
            assert "adopt**" not in html
            detail = client.get(f"/adapt/{rid}").text
            assert "Do this week" in detail
            assert "serve the Q4 this week" in detail


class TestRefreshEndpoint:
    def test_refresh_starts_without_real_ingest(self, client: TestClient, monkeypatch):
        import ai_researcher.web.app as app_mod

        # Never run the real pipeline in unit tests — it hits the network.
        monkeypatch.setattr(
            app_mod, "_run_sync", lambda pipeline, kwargs: {"status": "ok", "elapsed_s": 0.01}
        )
        r = client.post("/api/refresh")
        assert r.status_code == 200
        assert r.json()["status"] == "started"

        # A second overlapping start should be rejected while the first holds
        # the in-process lock briefly; if it already finished, started is fine.
        r2 = client.post("/api/refresh")
        assert r2.status_code in (200, 409)


class TestHealthz:
    def test_healthz_is_ok(self, client: TestClient):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_readyz_and_health(self, client: TestClient):
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["database"] == "ok"
        h = client.get("/health")
        assert h.status_code == 200
        body = h.json()
        assert body["ok"] is True
        assert "schema_version" in body
        assert "chat_default" in body

    def test_healthz_skips_the_access_token(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("AIR_DATA_DIR", str(data))
        monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "0")
        settings = Settings(
            data_dir=data,
            access_token="secret-token",
            sources_path=Path(__file__).resolve().parents[1] / "config" / "sources.yaml",
        )
        app = create_app(settings)
        with TestClient(app) as c:
            assert c.get("/healthz").status_code == 200
            assert c.get("/").status_code == 401
            assert c.get("/api/status").status_code == 401

    def test_browser_without_token_gets_a_prompt_not_a_dead_page(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("AIR_DATA_DIR", str(data))
        monkeypatch.setenv("AIR_AUTO_REFRESH_MIN", "0")
        settings = Settings(
            data_dir=data,
            access_token="secret-token",
            sources_path=Path(__file__).resolve().parents[1] / "config" / "sources.yaml",
        )
        app = create_app(settings)
        with TestClient(app) as c:
            r = c.get("/", headers={"accept": "text/html,*/*"})
            assert r.status_code == 401
            assert 'name="k"' in r.text and "AI Researcher is running" in r.text
            # Non-browser callers keep the terse 401.
            assert 'name="k"' not in c.get("/api/status", headers={"accept": "application/json"}).text
            # Submitting the form (a GET with ?k=) opens the app and sets the cookie.
            ok = c.get("/", params={"k": "secret-token"}, headers={"accept": "text/html"})
            assert ok.status_code == 200
            assert c.cookies.get("air_token") == "secret-token"
            assert c.get("/", headers={"accept": "text/html"}).status_code == 200
