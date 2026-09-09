"""URL canonicalisation and text normalisation — the basis of deduplication."""

import re
from datetime import datetime, timedelta, timezone

from ai_researcher.util import (
    canonical_url, content_hash, domain_of, humanize_age, local_day, tokens,
    truncate, url_hash, utcnow,
)


class TestCanonicalUrl:
    def test_strips_tracking_params(self):
        a = canonical_url("https://example.com/post?utm_source=x&utm_medium=y")
        b = canonical_url("https://example.com/post")
        assert a == b

    def test_strips_known_junk_params(self):
        assert canonical_url("https://a.com/p?fbclid=123") == "https://a.com/p"
        assert canonical_url("https://a.com/p?ref=hn") == "https://a.com/p"

    def test_keeps_content_selecting_params(self):
        # ?v= identifies the video; dropping it would merge unrelated pages.
        assert "v=abc" in canonical_url("https://youtube.com/watch?v=abc&utm_source=x")

    def test_normalises_host_and_www(self):
        assert canonical_url("https://WWW.Example.COM/p") == "https://example.com/p"

    def test_strips_fragment_and_trailing_slash(self):
        assert canonical_url("https://a.com/p/#section") == "https://a.com/p"

    def test_strips_amp_suffix(self):
        assert canonical_url("https://a.com/story/amp") == "https://a.com/story"

    def test_param_order_does_not_matter(self):
        assert canonical_url("https://a.com/p?b=2&a=1") == canonical_url("https://a.com/p?a=1&b=2")

    def test_non_http_is_dropped(self):
        assert canonical_url("mailto:x@y.com") == ""
        assert canonical_url("javascript:alert(1)") == ""
        assert canonical_url("data:text/html,x") == ""

    def test_empty_is_safe(self):
        assert canonical_url("") == ""

    def test_same_story_two_sources_collapses(self):
        """The whole point: one story, six referrers, one hash."""
        variants = [
            "https://openai.com/index/gpt-5?utm_source=techmeme",
            "https://www.openai.com/index/gpt-5/",
            "https://openai.com/index/gpt-5#top",
            "https://openai.com/index/gpt-5?ref=reddit&fbclid=99",
        ]
        assert len({url_hash(v) for v in variants}) == 1


class TestContentHash:
    def test_ignores_case_and_punctuation(self):
        assert content_hash("GPT-5 Released!") == content_hash("gpt 5 released")

    def test_differs_on_real_change(self):
        assert content_hash("GPT-5 released") != content_hash("Claude 5 released")


class TestTokens:
    def test_drops_stopwords_and_short_words(self):
        assert "the" not in tokens("the model is a big one")
        assert "model" in tokens("the model is a big one")

    def test_keeps_versioned_names(self):
        assert "gpt-5" in tokens("GPT-5 is here")


def test_truncate_adds_ellipsis_and_respects_limit():
    out = truncate("word " * 60, 40)
    assert len(out) <= 41 and out.endswith("…")
    assert truncate("short", 40) == "short"


def test_domain_of_strips_www():
    assert domain_of("https://www.example.com/x") == "example.com"


def test_humanize_age():
    now = utcnow()
    assert humanize_age(now - timedelta(minutes=5), now=now) == "5m ago"
    assert humanize_age(now - timedelta(hours=3), now=now) == "3h ago"
    assert humanize_age(now - timedelta(days=2), now=now) == "2d ago"
    assert humanize_age(None) == "unknown"


class TestLocalDay:
    """Regression guard: day buckets must follow the viewer's calendar.

    Keying them on the UTC date emptied the dashboard every afternoon for any
    host west of Greenwich — at UTC-6 "today" rolled over at 18:00 local.
    """

    def test_uses_local_calendar_not_utc(self):
        # 01:30 UTC on the 22nd is still the 21st for anyone at UTC-6.
        aware = datetime(2026, 8, 22, 1, 30, tzinfo=timezone.utc)
        expected = aware.astimezone().date().isoformat()
        assert local_day(aware) == expected

    def test_defaults_to_now(self):
        assert local_day() == datetime.now(timezone.utc).astimezone().date().isoformat()

    def test_returns_iso_date(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", local_day())
