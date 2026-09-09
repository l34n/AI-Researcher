"""Connector registry — maps a source's `kind` to its implementation."""

from __future__ import annotations

from ..config import Settings
from ..http import Fetcher
from .arxiv import ArxivConnector
from .base import Connector, FetchResult, RawItem
from .article import extract_article, hydrate_items, should_hydrate
from .github import GitHubReleasesConnector, GitHubTrendingConnector
from .gnews import GoogleNewsConnector
from .hackernews import HackerNewsConnector
from .huggingface import HFModelsConnector, HFPapersConnector
from .reddit import RedditConnector
from .rss import RSSConnector
from .x import XConnector

CONNECTOR_CLASSES: dict[str, type[Connector]] = {
    cls.kind: cls
    for cls in (
        RSSConnector,
        RedditConnector,
        HackerNewsConnector,
        ArxivConnector,
        HFPapersConnector,
        HFModelsConnector,
        GitHubReleasesConnector,
        GitHubTrendingConnector,
        GoogleNewsConnector,
        XConnector,
    )
}


def build_registry(settings: Settings, fetcher: Fetcher) -> dict[str, Connector]:
    """One instance per kind, shared across sources (they cache tokens)."""
    return {kind: cls(settings, fetcher) for kind, cls in CONNECTOR_CLASSES.items()}


__all__ = [
    "CONNECTOR_CLASSES",
    "Connector",
    "FetchResult",
    "RawItem",
    "build_registry",
    "extract_article",
    "hydrate_items",
    "should_hydrate",
]
