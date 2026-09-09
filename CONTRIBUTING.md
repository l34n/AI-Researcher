# Contributing

Thanks for looking. This is a self-hosted tool, so the most useful
contributions are usually small and concrete: a source worth tracking, a
connector that stopped working, a classifier rule that mislabels something.

## Getting set up

```bash
git clone https://github.com/l34n/AI-Researcher.git
cd AI-Researcher
uv venv && uv pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m pytest        # well under a second; no network, no GPU
```

Docker is the other supported way to run it (`docker compose up -d --build`).
You do **not** need Ollama, API keys, or a GPU to develop or to run the tests.
Every layer degrades on purpose: no Ollama falls back to rule-based enrichment
and TF-IDF clustering, no embedding model falls back to hashed TF-IDF, no
GitHub token just means a lower rate limit. If a change only works with the
optional pieces installed, that is a bug in the change.

To see it running, `.venv/bin/ai-researcher run` once, then
`.venv/bin/ai-researcher serve`.

## The rule that matters most

**Measure before you assert.** Most of the non-obvious code here exists because
a reasonable assumption turned out to be wrong when tested against the live
service — Reddit's rate limits, embedding-similarity thresholds, which feed
URLs actually resolve, whether a task prefix improves separation. Several
comments in the source record a measurement precisely so the next person does
not redo it.

If you are changing a threshold, a limit, or a fetch cadence, say in the PR
what you measured and how. "Seems better" is not enough for a number that
someone else will have to trust at 3am.

## Adding a source

This is the most common contribution and usually needs no Python at all.

If the site has an RSS or Atom feed, add a block to `config/sources.yaml`:

```yaml
  - key: my-source            # unique, stable, used as the DB key
    name: My Source           # shown in the UI
    kind: rss
    tier: lab                 # lab | vendor | research | news | analyst | community | infra
    weight: 1.8               # trust multiplier; see below
    category_hint: model-release   # optional nudge for the classifier
    config:
      url: https://example.com/feed.xml
```

Then verify it actually works before opening the PR:

```bash
.venv/bin/ai-researcher run
.venv/bin/ai-researcher sources     # your key should read `ok`, not `error`
```

`weight` scales how much the ranker trusts the source. Roughly: 2.0–2.5 for a
primary lab announcement, 1.5–2.0 for a strong vendor or research blog, 1.0 for
general news, below 1.0 for high-volume commentary. Please do not weight your
own blog at 2.5.

If the site has **no** feed, prefer `kind: gnews` with a path-scoped query
rather than writing a scraper:

```yaml
  - key: gn-example
    name: Example
    kind: gnews
    tier: vendor
    weight: 1.6
    config:
      query: "site:example.com/blog"    # scope to the path, not the domain
      days: 21
```

Scope it to the blog path. A bare `site:example.com` pulls in careers pages,
API docs, and pricing pages, which then cluster into a convincing-looking story
about nothing. There is a junk-title filter in `connectors/gnews.py`, but it is
a backstop, not a substitute for a narrow query.

## Adding a connector kind

Only needed for a source with real API semantics. Subclass `Connector` in
`src/ai_researcher/connectors/`, set `kind`, implement `fetch`, and add the
class to `CONNECTOR_CLASSES` in `connectors/__init__.py`.

```python
class MyConnector(Connector):
    kind = "mykind"

    async def fetch(self, source: Source, state: dict) -> FetchResult:
        payload = await self.fetcher.get_json("https://api.example.com/things")
        if payload is None:
            return FetchResult(status="error", error="unreachable")
        return FetchResult(items=[
            RawItem(
                external_id=str(t["id"]),
                title=t["title"],
                url=t["url"],
                published_at=parse_datetime(t["created_at"]),
                engagement=float(t.get("score", 0)),
            ).normalized()
            for t in payload["items"]
        ])
```

Requirements:

- **Always go through `self.fetcher`.** It carries the shared retry, backoff,
  `Retry-After` handling, and the global and per-host concurrency caps. A bare
  `httpx` call bypasses all of it and will earn a rate limit for everyone.
- **Call `.normalized()`** on every `RawItem`. URL canonicalisation is what lets
  the same story arriving from five sources dedup into one.
- **Never raise.** Return `FetchResult(status="error", error=...)`. One broken
  source must not end a run.
- **Set `engagement` on the source's own scale.** Do not pre-normalise; the
  ranker calibrates per source kind in `trends/score.py`.
- If the source cannot report engagement, set `meta["no_score"] = True` so the
  ranker treats it as unknown rather than as zero interest.
- If the source is rate-limited hard enough that one run cannot sweep it, use
  the `cursor` field on `FetchResult` to rotate across runs. `connectors/
  reddit.py` is the worked example.

## Tests

`pytest` for anything with logic. The existing tests avoid network and avoid a
live database where they can — see `tests/test_connectors.py` for how a
connector is tested against a recorded payload.

Please add a regression test when you fix a bug. Several tests here exist
because something broke in production once; the comment above each says what.

## Style

Match what is already there rather than importing a house style:

- Comments explain *why*, not *what*. If a line needs a comment to say what it
  does, rename something instead.
- Keep the docstring at the top of a module honest about its constraints. The
  ones in `connectors/reddit.py` and `pipeline.py` are the model.
- No new runtime dependencies without a reason in the PR. The dependency list
  is deliberately short.

## Pull requests

Small and single-purpose. Say what you changed, why, and what you ran to check
it. Screenshots for anything that alters the dashboard.

Reporting a bug is a contribution too — a source that silently stopped
returning items is genuinely useful to hear about, and `ai-researcher doctor`
output is the fastest way to show it.
