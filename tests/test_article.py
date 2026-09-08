"""Article hydration: fill thin feed teasers from the linked HTML page."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ai_researcher.config import Settings, Source
from ai_researcher.connectors.article import extract_article, hydrate_items, should_hydrate
from ai_researcher.connectors.base import FetchResult, RawItem
from ai_researcher.db import Database
from ai_researcher.http import Fetcher
from ai_researcher.pipeline import Pipeline


ARTICLE_HTML = """
<html><head><title>Acme 7B</title></head>
<body>
<nav>Home About Careers</nav>
<article>
  <h1>Acme releases 7B GGUF weights</h1>
  <p>Open weights, Apache-2.0, Q4 GGUF on Hugging Face. The card lists VRAM
  notes for a 12 GB GPU and a one-line llama.cpp command.</p>
  <p>A second paragraph so the extracted body is clearly longer than a teaser.</p>
</article>
<footer>Copyright</footer>
</body></html>
"""


def test_extract_prefers_article_over_chrome():
    text = extract_article(ARTICLE_HTML)
    assert "Acme releases 7B" in text
    assert "Careers" not in text
    assert "Copyright" not in text


def test_skips_native_kinds_and_hn_threads():
    thin = RawItem(external_id="1", title="x", url="https://openai.com/blog/x", body="hi")
    assert should_hydrate(thin, kind="rss") is True
    assert should_hydrate(thin, kind="arxiv") is False
    hn = RawItem(
        external_id="2", title="Ask HN",
        url="https://news.ycombinator.com/item?id=1", body="",
    )
    assert should_hydrate(hn, kind="hackernews") is False
    long = RawItem(
        external_id="3", title="x", url="https://openai.com/blog/x",
        body="x" * 400,
    )
    assert should_hydrate(long, kind="rss") is False


def test_hydrate_fills_thin_body():
    item = RawItem(
        external_id="1", title="Acme 7B",
        url="https://acme.example/7b", body="teaser",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE_HTML)

    fetcher = Fetcher("test-agent", concurrency=1)
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    n = asyncio.run(hydrate_items(fetcher, [item], kind="rss"))
    asyncio.run(fetcher.aclose())
    assert n == 1
    assert "Apache-2.0" in item.body
    assert item.meta.get("hydrated") is True


def test_ingest_source_receives_fetcher_and_stores_hydrated_body(tmp_path: Path):
    """Regression: health stats referenced `fetcher` without passing it in,
    so every source raised NameError and nothing was stored."""
    db = Database(tmp_path / "t.db")
    catalog = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
    settings = Settings(data_dir=tmp_path, sources_path=catalog)
    pipeline = Pipeline(settings, db)
    db.execute(
        "INSERT INTO sources (key, name, kind) VALUES (?,?,?)",
        ("blog", "Blog", "rss"),
    )

    class Conn:
        def available(self, src):
            return True, ""

        async def fetch(self, src, state):
            return FetchResult(items=[
                RawItem(
                    external_id="post-1",
                    title="Acme 7B",
                    url="https://acme.example/7b",
                    body="teaser",
                ).normalized()
            ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE_HTML)

    fetcher = Fetcher("test-agent", concurrency=1)
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    src = Source(key="blog", name="Blog", kind="rss")
    result = asyncio.run(pipeline._ingest_source(src, {"rss": Conn()}, fetcher))
    asyncio.run(fetcher.aclose())
    assert result["status"] == "ok"
    assert result["new"] == 1
    body = db.scalar("SELECT body FROM items WHERE external_id='post-1'")
    assert "Apache-2.0" in body
