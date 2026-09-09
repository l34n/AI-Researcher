"""Optional live-source contract checks. Never part of the default suite.

Run with: pytest -m network tests/test_live_sources.py
External failures must not break CI; the marker is not selected by default.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.network

pytest.importorskip("httpx")


@pytest.mark.skipif(
    os.environ.get("AIR_LIVE_SOURCES") != "1",
    reason="set AIR_LIVE_SOURCES=1 to hit real feeds",
)
def test_hackernews_feed_still_lists_hits():
    import httpx

    resp = httpx.get("https://hnrss.org/frontpage", timeout=15.0)
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "<item" in body or "<entry" in body
