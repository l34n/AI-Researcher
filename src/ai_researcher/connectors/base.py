"""Connector contract shared by every source kind."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import Settings, Source
from ..http import Fetcher
from ..util import canonical_url, content_hash, strip_html, url_hash, utcnow


@dataclass
class RawItem:
    """One fetched thing, before dedup and enrichment."""

    external_id: str
    title: str
    url: str = ""
    body: str = ""
    author: str = ""
    published_at: datetime | None = None
    engagement: float = 0.0          # upvotes, likes, stars — raw, per-source scale
    comments: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "RawItem":
        self.title = strip_html(self.title, 400) or self.title.strip()
        self.body = strip_html(self.body, 4000)
        self.author = strip_html(self.author, 80)
        self.url = canonical_url(self.url)
        return self

    @property
    def uhash(self) -> str:
        return url_hash(self.url) if self.url else ""

    @property
    def chash(self) -> str:
        return content_hash(self.title, self.body)


@dataclass
class FetchResult:
    items: list[RawItem] = field(default_factory=list)
    status: str = "ok"               # ok | not-modified | error | disabled
    error: str = ""
    etag: str = ""
    last_modified: str = ""
    # Opaque per-source state persisted between runs. Used by connectors that
    # must rotate through their targets across runs rather than sweep them all.
    cursor: str = ""


class Connector:
    """Base class. Subclasses implement `fetch`."""

    kind = "base"

    def __init__(self, settings: Settings, fetcher: Fetcher):
        self.settings = settings
        self.fetcher = fetcher

    async def fetch(self, source: Source, state: dict[str, Any]) -> FetchResult:
        raise NotImplementedError

    def available(self, source: Source) -> tuple[bool, str]:
        """Whether this connector can run at all (credentials, config)."""
        return True, ""

    @staticmethod
    def now() -> datetime:
        return utcnow()
