"""URL canonicalisation, hashing, time handling, and text cleanup.

Deduplication quality lives or dies here: the same story reaches us from six
sources with six different tracking-parameter tails, and a canonical URL is what
collapses them into one.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dateutil import parser as date_parser
from selectolax.parser import HTMLParser

# Tracking noise that never changes what a URL points at.
_JUNK_PARAM_PREFIXES = ("utm_", "mc_", "pk_", "hsa_", "vero_", "_hs")
_JUNK_PARAMS = {
    "ref", "referrer", "source", "src", "fbclid", "gclid", "igshid", "mkt_tok",
    "spm", "cmpid", "ncid", "sh", "share", "s", "t", "at_medium", "at_campaign",
    "__twitter_impression", "guccounter", "amp", "cmp", "smid",
}
# Params that genuinely select content and must survive canonicalisation.
_KEEP_PARAMS = {"v", "id", "p", "q", "page", "story", "article", "paper", "model"}

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\+#]{1,}")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "by", "at", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "we", "you", "they",
    "he", "she", "i", "our", "your", "their", "his", "her", "not", "no", "can",
    "will", "would", "should", "could", "may", "might", "must", "have", "has",
    "had", "do", "does", "did", "so", "than", "then", "there", "here", "what",
    "which", "who", "whom", "how", "why", "when", "where", "all", "any", "some",
    "more", "most", "other", "into", "over", "after", "before", "about", "up",
    "down", "out", "off", "again", "new", "now", "just", "only", "also", "very",
    "using", "use", "used", "via", "vs", "得", "one", "two", "get", "make",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def local_day(when: datetime | None = None) -> str:
    """The calendar date bucket a dashboard day is filed under, in LOCAL time.

    Every stored timestamp stays UTC — only this bucket label is local. Keying
    it on the UTC date instead means a user west of Greenwich watches "today"
    reset in the middle of their afternoon: at UTC-6 the dashboard emptied at
    18:00 local and stayed empty until the next ingest run re-clustered.
    """
    when = when or datetime.now(timezone.utc)
    return when.astimezone().date().isoformat()


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value) -> datetime | None:
    """Best-effort parse of anything a feed might call a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = date_parser.parse(text)
    except (ValueError, OverflowError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def canonical_url(url: str) -> str:
    """Strip tracking cruft, normalise host and path, drop fragments."""
    if not url:
        return ""
    url = url.strip()
    if any(ch in url for ch in "<>\"'\\"):
        return ""
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in ("http", "https"):
        return ""
    host = (parts.hostname or "").lower()
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and host.count(".") >= 2:
        host = host[2:]
    # Google News and similar wrappers hide the real destination; keep as-is
    # rather than guess, but never let the wrapper's params inflate the hash.
    netloc = host
    if parts.port and parts.port not in (80, 443):
        netloc = f"{host}:{parts.port}"

    kept = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        low = key.lower()
        if low in _KEEP_PARAMS:
            kept.append((key, value))
            continue
        if low in _JUNK_PARAMS or any(low.startswith(p) for p in _JUNK_PARAM_PREFIXES):
            continue
        kept.append((key, value))
    query = urlencode(sorted(kept))

    path = re.sub(r"/+", "/", parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if path.endswith("/amp"):
        path = path[:-4] or "/"

    return urlunsplit((parts.scheme, netloc, path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:32]


def content_hash(title: str, body: str = "") -> str:
    """Hash of normalised text, so a re-titled repost still collides."""
    norm = normalize_text(f"{title} {body}")[:2000]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return _WS_RE.sub(" ", text).strip()


def strip_html(html: str, limit: int = 4000) -> str:
    """Turn feed-supplied HTML into readable plain text."""
    if not html:
        return ""
    html = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", html)
    if "<" not in html:
        return _WS_RE.sub(" ", html).strip()[:limit]
    try:
        tree = HTMLParser(html)
    except Exception:
        return _WS_RE.sub(" ", re.sub(r"<[^>]+>", " ", html)).strip()[:limit]
    for tag in tree.css("script, style, noscript, nav, footer, form"):
        tag.decompose()
    text = tree.text(separator=" ", strip=True) if tree.body else ""
    return _WS_RE.sub(" ", text).strip()[:limit]


def tokens(text: str, min_len: int = 3) -> list[str]:
    """Lowercase content tokens with stopwords removed."""
    lowered = unicodedata.normalize("NFKC", text or "").casefold()
    return [
        t for t in _TOKEN_RE.findall(lowered)
        if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()
    ]


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip() + "…"


def domain_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def humanize_age(dt: datetime | None, *, now: datetime | None = None) -> str:
    if dt is None:
        return "unknown"
    now = now or utcnow()
    delta = now - dt
    if delta < timedelta(0):
        return "just now"
    secs = int(delta.total_seconds())
    if secs < 3600:
        return f"{max(secs // 60, 1)}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    days = secs // 86400
    return f"{days}d ago" if days < 30 else f"{days // 30}mo ago"
