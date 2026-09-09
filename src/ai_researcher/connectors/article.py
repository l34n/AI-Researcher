"""Pull the linked page when a feed only gave a title or a teaser.

Connectors store whatever the API or RSS entry carried. For HN link posts,
Google News, and most vendor RSS that is a headline plus a sentence — not
the article. The in-app reader (`/read/{id}`) is labelled "scraped"; this
module is what actually fetches and strips the page.

Failures are silent: a 403, a paywall shell, or a PDF must not fail the
source. Native APIs (arXiv abstracts, GitHub release notes, HF cards) are
left alone.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable
from urllib.parse import urlsplit

from ..http import Fetcher
from ..util import domain_of, strip_html
from .base import RawItem

log = logging.getLogger("ai_researcher.article")

# Below this, the stored body is a teaser and the linked page is worth a GET.
TEASER_CHARS = 280
ARTICLE_CHARS = 8000
PER_SOURCE_CAP = 12

# Hosts that are indexes, threads, or wrappers — not the article.
_SKIP_HOSTS = {
    "news.ycombinator.com",
    "news.google.com",
    "google.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "old.reddit.com",
    "www.reddit.com",
    "youtube.com",
    "youtu.be",
    "linkedin.com",
}
_SKIP_SUFFIX = (
    ".pdf", ".zip", ".gz", ".tgz", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".mp4", ".mp3", ".wav", ".svg",
)
# These kinds already ship a usable body (abstract, README, release notes).
_NATIVE_KINDS = {
    "arxiv", "hf_papers", "hf_models", "github_releases", "github_trending", "x",
}


def should_hydrate(item: RawItem, *, kind: str = "") -> bool:
    if kind in _NATIVE_KINDS:
        return False
    if len((item.body or "").strip()) >= TEASER_CHARS:
        return False
    url = (item.url or "").strip()
    if not url.startswith("http"):
        return False
    host = domain_of(url)
    if not host or host in _SKIP_HOSTS or any(host.endswith("." + h) for h in _SKIP_HOSTS):
        return False
    path = (urlsplit(url).path or "").lower()
    if any(path.endswith(suf) for suf in _SKIP_SUFFIX):
        return False
    return True


def extract_article(html: str, *, limit: int = ARTICLE_CHARS) -> str:
    """Prefer <article>/<main>; fall back to stripped body text."""
    raw = html or ""
    if not raw.strip():
        return ""
    try:
        from selectolax.parser import HTMLParser
        tree = HTMLParser(raw)
    except Exception:
        return strip_html(raw, limit)
    for tag in tree.css("script, style, noscript, nav, footer, form, aside, iframe"):
        tag.decompose()
    node = None
    for sel in ("article", "main", "[role=main]", ".post-content", ".entry-content"):
        found = tree.css_first(sel)
        if found and (found.text(separator=" ", strip=True) or "").strip():
            node = found
            break
    text = ""
    if node is not None:
        text = node.text(separator=" ", strip=True) or ""
    if len(text) < TEASER_CHARS:
        text = tree.text(separator=" ", strip=True) if tree.body else text
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


async def hydrate_items(
    fetcher: Fetcher,
    items: Iterable[RawItem],
    *,
    kind: str = "",
    limit: int = PER_SOURCE_CAP,
) -> int:
    """Fill thin bodies from the linked page. Returns how many were upgraded."""
    targets = [it for it in items if should_hydrate(it, kind=kind)][: max(0, limit)]
    if not targets:
        return 0
    upgraded = 0
    for item in targets:
        resp = await fetcher.get(item.url, attempts=2)
        if resp is None or resp.status_code >= 400:
            continue
        ctype = (resp.headers.get("content-type") or "").lower()
        if "html" not in ctype and "xml" not in ctype and ctype:
            continue
        text = extract_article(resp.text or "")
        if len(text) <= len((item.body or "").strip()) + 40:
            continue
        item.body = text
        meta = dict(item.meta or {})
        meta["hydrated"] = True
        item.meta = meta
        upgraded += 1
    if upgraded:
        log.info("hydrated %s/%s thin %s items from the linked page", upgraded, len(targets), kind or "feed")
    return upgraded
