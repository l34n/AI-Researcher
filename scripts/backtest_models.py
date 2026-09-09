"""L34N historical backtest — APE-712.

Runs candidate models against real, past-dated items pulled from L34N's
SQLite store, one dated corpus per day, so we can compare model output
consistency **over time** on real-world inputs.

Each backtest date is one call to the brief pipeline per model. Metrics
that need ground-truth labels (relevance precision/recall, readiness
agreement) are dropped; the remaining schema-layer metrics
(format_compliance, citation_completeness, factuality_score,
hallucinated_recommendation_rate, prompt_echo_rate, fallback_rate) are
what we score consistency on.

Usage:
    OPENROUTER_API_KEY=... python scripts/backtest_models.py \\
        --profile paid --dates 2026-08-22,2026-08-26,2026-08-30,2026-09-01,2026-09-03
    python scripts/backtest_models.py --profile local --items-per-date 8

Outputs:
    data/backtest-results/<slug>__<date>.json    per-model per-date detail
    data/backtest-results/index.json             sweep summary + consistency table
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

L34N_ROOT = Path("/home/ebg/l34n")
sys.path.insert(0, str(L34N_ROOT / "src"))
sys.path.insert(0, str(L34N_ROOT / "scripts"))

# Reuse the APE-711 harness: model catalog + client + rubric scoring.
from benchmark_models import (  # noqa: E402
    ModelSpec, ProviderClient, _estimate_cost, load_matrix, sweep_settings,
)
from ai_researcher.config import Settings  # noqa: E402
from ai_researcher.eval.harness import run_case  # noqa: E402
from ai_researcher.eval.metrics import summarise  # noqa: E402

DB_PATH = L34N_ROOT / "data" / "airesearch.db"

# Metrics that survive without ground-truth labels — the ones a
# consistency analysis can legitimately report on.
CONSISTENCY_METRICS = (
    "format_compliance",
    "citation_completeness",
    "factuality_score",
    "prompt_echo_rate",
    "hallucinated_recommendation_rate",
    "fallback_rate",
)


# ── Corpus construction from the DB ────────────────────────────────────────────

def _fetch_items_for_date(
    conn: sqlite3.Connection, day: str, *, limit: int
) -> list[dict[str, Any]]:
    """Return items published on ``day`` (YYYY-MM-DD), ordered by relevance.

    Prefers items already tagged ``relevant=1`` so the corpus looks like a
    realistic post-classifier brief input rather than raw feed noise.
    """
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, source_key, title, url, canonical_url, body,
               published_at, relevance_score
          FROM items
         WHERE substr(published_at, 1, 10) = ?
           AND relevant = 1
         ORDER BY relevance_score DESC, id DESC
         LIMIT ?
        """,
        (day, limit),
    ).fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _kind_from_source(source_key: str) -> str:
    # Mirror the classifier's coarse source_kind for prompt shaping.
    if source_key.startswith("hf-"):
        return "hf_models" if "model" in source_key else "hf_papers"
    if source_key == "arxiv-core":
        return "arxiv"
    if source_key.startswith("gh-"):
        return "gh"
    return "rss"


def _tier_from_source(source_key: str) -> str:
    lab = {"openai-news", "openai-blog", "anthropic-news", "google-deepmind",
           "meta-ai", "nvidia-blog", "nvidia-dev"}
    vendor = {"hf-blog", "hf-models", "hf-papers", "aws-ml", "databricks",
              "langchain", "vercel-ai"}
    research = {"arxiv-core", "mit-news-ai"}
    if source_key in lab:
        return "lab"
    if source_key in vendor:
        return "vendor"
    if source_key in research:
        return "research"
    return "news"


def _row_to_story(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    """Turn a DB row into the story shape L34N's brief prompt wants."""
    title = (row.get("title") or "item").strip()
    body = (row.get("body") or "").strip()
    url = row.get("canonical_url") or row.get("url") or ""
    source_key = row.get("source_key") or "rss"
    return {
        "id": idx,
        "label": title[:140],
        "summary": body[:220],
        "category": "opinion-analysis",
        "source_count": 1,
        "sources": [source_key],
        "item_ids": [row["id"]],
        "freshness_status": "fresh",
        "_url": url,
    }


def _build_case_for_date(day: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one eval case = one daily brief input covering N stories.

    We also stash a per-story primary item so per-item relevance scoring
    could reuse the harness if we ever get ground-truth labels; for now
    we skip relevance/readiness (no labels) and let those metrics fall
    to zero — they're excluded from the consistency scoring below.
    """
    stories = [_row_to_story(i + 1, r) for i, r in enumerate(rows)]
    # No `item` / `expected` — the harness will skip relevance/readiness.
    return {
        "id": f"backtest-{day}",
        "family": "backtest-historical",
        "stories": stories,
        "ready": [],
        "expected": {},
        "_pub_date": day,
        "_item_count": len(rows),
    }


# ── Runner ─────────────────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@dataclass
class BacktestPoint:
    date: str
    item_count: int
    wall_s: float
    validate_ok: bool
    format_compliance: float
    citation_completeness: float
    factuality_score: float
    prompt_echo: float
    hallucinated_ready: float
    fallback: float
    word_count: int
    errors: list[str] = field(default_factory=list)
    call_input_tokens: int = 0
    call_output_tokens: int = 0
    call_cost_usd: float = 0.0


def run_one_model_one_date(
    spec: ModelSpec, client: ProviderClient, case: dict[str, Any],
) -> BacktestPoint:
    """Generate one brief with ``client``, then score it with the standard harness."""
    baseline = asdict(client.stats)  # snapshot so per-call token counts are attributable
    started = time.monotonic()
    row = run_case(case, layer="schema", generate=client.generate)
    wall = time.monotonic() - started
    delta_in = client.stats.input_tokens - baseline["input_tokens"]
    delta_out = client.stats.output_tokens - baseline["output_tokens"]
    call_cost = 0.0
    if spec.input_per_m is not None and spec.output_per_m is not None:
        call_cost = (delta_in / 1_000_000 * spec.input_per_m
                     + delta_out / 1_000_000 * spec.output_per_m)
    return BacktestPoint(
        date=case["_pub_date"],
        item_count=case["_item_count"],
        wall_s=round(wall, 2),
        validate_ok=bool(row.get("validate_ok")),
        format_compliance=float(row.get("format_compliance") or 0.0),
        citation_completeness=float(row.get("citation_completeness") or 0.0),
        factuality_score=float(row.get("factuality_score") or 0.0),
        prompt_echo=float(row.get("prompt_echo") or 0.0),
        hallucinated_ready=float(row.get("hallucinated_ready") or 0.0),
        fallback=float(row.get("fallback") or 0.0),
        word_count=int(row.get("word_count") or 0),
        errors=list(row.get("errors") or []),
        call_input_tokens=delta_in,
        call_output_tokens=delta_out,
        call_cost_usd=round(call_cost, 6),
    )


def summarise_consistency(points: list[BacktestPoint]) -> dict[str, Any]:
    """Compute mean + population std of the label-free metrics across dates."""
    if not points:
        return {"n": 0}
    def vec(attr: str) -> list[float]:
        return [float(getattr(p, attr)) for p in points]

    out: dict[str, Any] = {"n": len(points)}
    key_by_metric = {
        "format_compliance": "format_compliance",
        "citation_completeness": "citation_completeness",
        "factuality_score": "factuality_score",
        "prompt_echo_rate": "prompt_echo",
        "hallucinated_recommendation_rate": "hallucinated_ready",
        "fallback_rate": "fallback",
    }
    for metric, attr in key_by_metric.items():
        vals = vec(attr)
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
    # Overall consistency score: 1 - avg(std across the label-free metrics),
    # excluding negatives-only signals (prompt_echo / hallucinated_ready /
    # fallback) — for those, higher variance is *worse* but so is a high mean;
    # we still expose them so the report can inspect regressions per-day.
    positive_stds = [
        out[m]["std"]
        for m in ("format_compliance", "citation_completeness", "factuality_score")
    ]
    out["consistency_score"] = round(1.0 - statistics.fmean(positive_stds), 4)
    out["wall_s"] = round(sum(p.wall_s for p in points), 2)
    out["cost_usd"] = round(sum(p.call_cost_usd for p in points), 6)
    return out


def sweep(
    specs: list[ModelSpec], settings: Settings, out_dir: Path,
    *, dates: list[str], items_per_date: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Build one case per date, shared across all models.
    cases: list[dict[str, Any]] = []
    for day in dates:
        rows = _fetch_items_for_date(conn, day, limit=items_per_date)
        if not rows:
            print(f"  skip {day}: no items in DB", file=sys.stderr)
            continue
        case = _build_case_for_date(day, rows)
        cases.append(case)
        print(f"  corpus {day}: {len(rows)} items")

    if not cases:
        raise SystemExit("no dated corpora built — DB is empty for those dates")

    all_summaries: list[dict[str, Any]] = []
    for spec in specs:
        print(f"→ {spec.slug} ({spec.provider} / {spec.model}, {spec.tier}) "
              f"over {len(cases)} date(s)")
        try:
            client = ProviderClient(spec, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! client init failed: {exc}", file=sys.stderr)
            all_summaries.append({
                "slug": spec.slug, "model": spec.model, "tier": spec.tier,
                "provider": spec.provider, "failed": True, "error": str(exc)[:400],
            })
            continue

        points: list[BacktestPoint] = []
        errors: list[dict[str, str]] = []
        try:
            for case in cases:
                try:
                    pt = run_one_model_one_date(spec, client, case)
                    points.append(pt)
                    print(f"    {pt.date}: wall={pt.wall_s}s fmt={pt.format_compliance:.2f} "
                          f"cite={pt.citation_completeness:.2f} fact={pt.factuality_score:.2f} "
                          f"fallback={pt.fallback:.2f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! {case['_pub_date']} failed: {exc}", file=sys.stderr)
                    errors.append({"date": case["_pub_date"], "error": str(exc)[:400]})
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

        stats = asdict(client.stats)
        cost = _estimate_cost(spec, client.stats)
        consistency = summarise_consistency(points)

        doc = {
            "model_slug": spec.slug,
            "model_id": spec.model,
            "provider": spec.provider,
            "tier": spec.tier,
            "backend": "local_ollama" if spec.provider == "ollama" else "openrouter",
            "dates": [c["_pub_date"] for c in cases],
            "items_per_date": items_per_date,
            "call_stats": stats,
            "cost_usd_estimate": round(cost, 6),
            "points": [asdict(p) for p in points],
            "consistency": consistency,
            "errors": errors,
        }
        (out_dir / f"{spec.slug}.json").write_text(
            json.dumps(doc, indent=2, default=str), encoding="utf-8",
        )
        all_summaries.append({
            "slug": spec.slug, "model": spec.model, "tier": spec.tier,
            "provider": spec.provider,
            "consistency_score": consistency.get("consistency_score"),
            "avg_format_compliance": consistency.get("format_compliance", {}).get("mean"),
            "std_format_compliance": consistency.get("format_compliance", {}).get("std"),
            "avg_factuality": consistency.get("factuality_score", {}).get("mean"),
            "std_factuality": consistency.get("factuality_score", {}).get("std"),
            "avg_citation": consistency.get("citation_completeness", {}).get("mean"),
            "std_citation": consistency.get("citation_completeness", {}).get("std"),
            "wall_s": consistency.get("wall_s"),
            "cost_usd": consistency.get("cost_usd"),
            "cases": consistency.get("n"),
            "errors": len(errors),
        })
        print(f"  ✓ wrote {spec.slug}.json  consistency={consistency.get('consistency_score')}")

    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dates": [c["_pub_date"] for c in cases],
        "items_per_date": items_per_date,
        "models": all_summaries,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def parse_dates(spec_str: str, conn: sqlite3.Connection) -> list[str]:
    if spec_str == "auto":
        # Pick 5 evenly-spaced days from the observed distribution.
        cur = conn.cursor()
        days = [r[0] for r in cur.execute(
            "SELECT DISTINCT substr(published_at,1,10) d "
            "FROM items WHERE published_at IS NOT NULL ORDER BY d"
        )]
        if not days:
            return []
        if len(days) <= 5:
            return days
        # Even sample: pick 5 indices spanning [0, len-1].
        step = (len(days) - 1) / 4
        idx = sorted({int(round(i * step)) for i in range(5)})
        return [days[i] for i in idx][:5]
    return [d.strip() for d in spec_str.split(",") if d.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="APE-712 L34N historical backtest sweep.")
    ap.add_argument("--profile", choices=["free", "local", "paid", "all"],
                    default="paid")
    ap.add_argument("--model", action="append", default=None,
                    help="run only this model slug (repeatable)")
    ap.add_argument("--dates", default="auto",
                    help="comma-sep YYYY-MM-DD list, or 'auto' to auto-pick 5 dates")
    ap.add_argument("--items-per-date", type=int, default=8,
                    help="max items per dated corpus (default 8)")
    ap.add_argument("--out", default=str(L34N_ROOT / "data" / "backtest-results"))
    ap.add_argument("--matrix", default=None,
                    help="path to a YAML model matrix (default: scripts/benchmark_matrix.yaml)")
    args = ap.parse_args(argv)
    # The matrix is loaded lazily by benchmark_models since APE-713; importing
    # MODEL_MATRIX gave an empty list and every backtest ran zero models.
    MODEL_MATRIX = load_matrix(Path(args.matrix) if args.matrix else None)

    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}", file=sys.stderr)
        return 2

    # Sweep-private data dir, daily cap lifted (APE-727). The corpus is still
    # read straight from the production SQLite file via DB_PATH.
    settings = sweep_settings()

    if args.model:
        specs = [s for s in MODEL_MATRIX if s.slug in args.model or s.model in args.model]
        if not specs:
            print(f"no matching models for {args.model}", file=sys.stderr)
            return 2
    else:
        if args.profile == "all":
            specs = list(MODEL_MATRIX)
        else:
            specs = [s for s in MODEL_MATRIX if s.tier == args.profile]

    with sqlite3.connect(DB_PATH) as conn:
        dates = parse_dates(args.dates, conn)
    if not dates:
        print("no dates resolved", file=sys.stderr)
        return 2
    print(f"Backtest: {len(specs)} model(s) × {len(dates)} date(s), "
          f"items/date={args.items_per_date}\n  dates: {dates}")
    sweep(specs, settings, Path(args.out), dates=dates,
          items_per_date=args.items_per_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
