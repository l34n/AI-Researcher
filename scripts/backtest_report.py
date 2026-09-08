"""Render the APE-712 backtest markdown report from per-tier index files.

Reads:
    data/backtest-results/<tier>/<slug>.json   per-model per-date detail
    data/backtest-results/<tier>/index.json    sweep summary

Writes:
    docs/backtest-results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

L34N_ROOT = Path("/home/ebg/l34n")
DEFAULT_TIERS = ("paid", "local", "free")


def load_tier(root: Path, tier: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    tier_dir = root / tier
    if not tier_dir.exists():
        return docs
    for path in sorted(tier_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def _fmt_pair(m: dict[str, Any] | None) -> str:
    if not m:
        return "—"
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


def render(docs: list[dict[str, Any]], *, dates: list[str]) -> str:
    lines: list[str] = []
    lines.append("# L34N Historical Backtest — APE-712")
    lines.append("")
    lines.append("**Deliverable of:** [APE-712](/APE/issues/APE-712) "
                 "· **Parent:** [APE-709](/APE/issues/APE-709) "
                 "· **Rubric:** [APE-710](/APE/issues/APE-710) v1.0 "
                 "· **Harness:** [APE-711](/APE/issues/APE-711) reuse")
    lines.append("")
    lines.append(f"**Backtest dates:** {', '.join(dates)}")
    lines.append("")
    lines.append("**Corpus source:** `/home/ebg/l34n/data/airesearch.db` — up to 8 "
                 "relevance-ranked items per date, one brief per model per date.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "This is a **consistency backtest**, not a labelled evaluation. Historical "
        "items pulled from the L34N SQLite store have no ground-truth `expected` "
        "labels, so the harness metrics that need labels — relevance precision/"
        "recall, readiness_agreement, injection_following_rate — are excluded. "
        "What we do score, per date, is what the schema validator + citation + "
        "echo detector return on the model's actual brief output:"
    )
    lines.append("")
    lines.append("- `format_compliance` — brief passed the L34N schema validator")
    lines.append("- `citation_completeness` — fraction of brief sections that cited a source")
    lines.append("- `factuality_score` — derived from format + citation + non-hallucinated ready")
    lines.append("- `prompt_echo_rate` — model regurgitated the prompt")
    lines.append("- `hallucinated_recommendation_rate` — model wrote an ungated adopt/ready "
                 "(dropped before shipping when nothing was gated; fatal otherwise)")
    lines.append("- `fallback_rate` — L34N fell back to the deterministic brief")
    lines.append("")
    lines.append(
        "Consistency across dates is the coefficient of interest: **mean** shows "
        "how well the model performed on average, **std** shows how much that "
        "swung day-to-day. A single-number `consistency_score = 1 − avg(std of "
        "format/citation/factuality)` is provided as a shorthand — but read it "
        "alongside the mean, because a model that is uniformly bad will score "
        "as very consistent."
    )
    lines.append("")

    # ── Per-model consistency table ────────────────────────────────────────────
    lines.append("## Consistency Across Dates")
    lines.append("")
    lines.append("Sorted by (mean factuality × consistency_score) descending — "
                 "we want models that are both good on average AND stable.")
    lines.append("")
    lines.append("| Model | Tier | Cases | Consistency | Format (μ±σ) | "
                 "Citation (μ±σ) | Factuality (μ±σ) | Fallback (μ±σ) | "
                 "Wall (s) | Cost ($) |")
    lines.append("|---|---|---:|---:|---|---|---|---|---:|---:|")

    rows: list[tuple[float, str]] = []
    for d in docs:
        if not d.get("consistency") or d.get("failed"):
            note = d.get("error", "no data")
            rows.append((-1.0, f"| `{d.get('model_id', d.get('model'))}` | "
                              f"{d.get('tier','?')} | 0 | — | — | — | — | — | — | — |  "
                              f"<!-- {note} -->"))
            continue
        c = d["consistency"]
        rank_key = c["factuality_score"]["mean"] * c["consistency_score"]
        row = (
            f"| `{d['model_id']}` | {d['tier']} | {c['n']} | "
            f"**{c['consistency_score']:.3f}** | "
            f"{_fmt_pair(c.get('format_compliance'))} | "
            f"{_fmt_pair(c.get('citation_completeness'))} | "
            f"{_fmt_pair(c.get('factuality_score'))} | "
            f"{_fmt_pair(c.get('fallback_rate'))} | "
            f"{c['wall_s']:.1f} | {c['cost_usd']:.4f} |"
        )
        rows.append((rank_key, row))
    for _, row in sorted(rows, key=lambda x: x[0], reverse=True):
        lines.append(row)
    lines.append("")

    # ── Dates × models grid ────────────────────────────────────────────────────
    lines.append("## Format Compliance — Dates × Models")
    lines.append("")
    header = ["Date"] + [d["model_id"] for d in docs if d.get("points")]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    date_index: dict[str, dict[str, float]] = {}
    for d in docs:
        for p in d.get("points") or []:
            date_index.setdefault(p["date"], {})[d["model_id"]] = p["format_compliance"]
    for day in sorted(date_index.keys()):
        row = [day]
        for d in docs:
            if not d.get("points"):
                continue
            v = date_index[day].get(d["model_id"])
            row.append("—" if v is None else f"{v:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Factuality — Dates × Models")
    lines.append("")
    fact_index: dict[str, dict[str, float]] = {}
    for d in docs:
        for p in d.get("points") or []:
            fact_index.setdefault(p["date"], {})[d["model_id"]] = p["factuality_score"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for day in sorted(fact_index.keys()):
        row = [day]
        for d in docs:
            if not d.get("points"):
                continue
            v = fact_index[day].get(d["model_id"])
            row.append("—" if v is None else f"{v:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Analysis ──────────────────────────────────────────────────────────────
    scored = [d for d in docs if d.get("consistency") and not d.get("failed")]
    if scored:
        lines.append("## Analysis")
        lines.append("")
        # Best-mean-factuality
        best_fact = max(scored, key=lambda d: d["consistency"]["factuality_score"]["mean"])
        # Most consistent (but with fact > 0.5 floor to filter uniformly-bad-and-flat)
        stable_and_good = [
            d for d in scored
            if d["consistency"]["factuality_score"]["mean"] >= 0.5
        ]
        most_stable = max(
            stable_and_good or scored,
            key=lambda d: d["consistency"]["consistency_score"],
        )
        cheapest_paid = [d for d in scored if d["tier"] == "paid"]
        cheapest_paid = sorted(cheapest_paid, key=lambda d: d["consistency"]["cost_usd"])

        lines.append(f"- **Highest average factuality**: `{best_fact['model_id']}` "
                     f"({best_fact['tier']}) — μ="
                     f"{best_fact['consistency']['factuality_score']['mean']:.2f}, "
                     f"σ={best_fact['consistency']['factuality_score']['std']:.2f}")
        lines.append(f"- **Most consistent (with factuality ≥ 0.5)**: "
                     f"`{most_stable['model_id']}` "
                     f"(consistency_score={most_stable['consistency']['consistency_score']:.3f})")
        if cheapest_paid:
            cp = cheapest_paid[0]
            lines.append(f"- **Cheapest paid model this sweep**: `{cp['model_id']}` — "
                         f"${cp['consistency']['cost_usd']:.4f} for {cp['consistency']['n']} briefs")

        # Total spend
        total_cost = sum(d["consistency"]["cost_usd"] for d in scored)
        total_wall = sum(d["consistency"]["wall_s"] for d in scored)
        lines.append(f"- **Total sweep cost**: ${total_cost:.4f} across "
                     f"{sum(d['consistency']['n'] for d in scored)} model-date runs, "
                     f"{total_wall:.1f}s wall time")
        lines.append("")

        # Consistency-vs-quality gotchas
        gotchas = []
        for d in scored:
            c = d["consistency"]
            if c["fallback_rate"]["mean"] >= 0.8:
                gotchas.append(
                    f"- `{d['model_id']}`: fallback fired on **every date** — its "
                    f"apparent consistency reflects the deterministic fallback, "
                    f"not the model. Treat brief outputs as unusable."
                )
            elif c["consistency_score"] >= 0.95 and c["factuality_score"]["mean"] < 0.4:
                gotchas.append(
                    f"- `{d['model_id']}`: perfectly consistent but uniformly bad "
                    f"(factuality μ={c['factuality_score']['mean']:.2f}). Do not confuse "
                    f"low variance with quality."
                )
        if gotchas:
            lines.append("### Consistency gotchas")
            lines.append("")
            lines.extend(gotchas)
            lines.append("")

    # ── Recommendation ────────────────────────────────────────────────────────
    if scored:
        prod_candidates = [
            d for d in scored
            if d["consistency"]["factuality_score"]["mean"] >= 0.5
            and d["consistency"]["fallback_rate"]["mean"] < 0.5
        ]
        prod_candidates.sort(
            key=lambda d: (
                d["consistency"]["factuality_score"]["mean"]
                * d["consistency"]["consistency_score"]
            ),
            reverse=True,
        )
        lines.append("## Recommendation")
        lines.append("")
        if not prod_candidates:
            lines.append(
                "**No model in this sweep meets the production bar** "
                "(factuality μ ≥ 0.5 and fallback μ < 0.5). The APE-710 schema-layer "
                "validator remains stricter than any candidate satisfies on live "
                "corpora — the same DQ pattern seen in [APE-711](/APE/issues/APE-711). "
                "Follow-up work (own issue) needs to either relax the validator or "
                "extend prompts to satisfy the strictest checks."
            )
        else:
            top = prod_candidates[0]
            local_top = next(
                (d for d in prod_candidates if d["tier"] == "local"), None,
            )
            lines.append(
                f"- **Primary production pick**: `{top['model_id']}` ({top['tier']}). "
                f"Highest mean factuality × consistency, cost "
                f"${top['consistency']['cost_usd']:.4f} for a 5-day rolling sweep."
            )
            if local_top and local_top is not top:
                lines.append(
                    f"- **Local fallback (zero-cost)**: `{local_top['model_id']}` — "
                    f"factuality μ="
                    f"{local_top['consistency']['factuality_score']['mean']:.2f}, "
                    f"consistency_score="
                    f"{local_top['consistency']['consistency_score']:.3f}."
                )
            lines.append(
                "- Re-run this backtest weekly (routine or manual trigger) so any "
                "silent quality drift shows up as a delta in `consistency_score` "
                "or `factuality_score` mean."
            )
        lines.append("")

    # ── Method caveats ────────────────────────────────────────────────────────
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Historical corpus items were **fetched today** (2026-09-03) — "
        "`published_at` gives the source-of-record publication date, but there "
        "is no true archival replay of the RSS feeds themselves. Publisher "
        "backdating and later edits are not captured."
    )
    lines.append(
        "- Items per date are top-N by classifier `relevance_score`, so days with "
        "very few relevant items (see 2026-08-22 with 5) give a smaller corpus and "
        "noisier scores."
    )
    lines.append(
        "- Ollama speeds are LAN-only against the local backend; do not compare "
        "against OpenRouter wall times."
    )
    lines.append(
        "- `consistency_score` weights the three positive-signal metrics equally; "
        "if you care most about schema compliance in production, prefer the "
        "Format-Compliance grid over the single-number ranking."
    )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(L34N_ROOT / "data" / "backtest-results"))
    ap.add_argument("--out", default="/home/ebg/research-pipeline/docs/backtest-results.md")
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    args = ap.parse_args(argv)

    root = Path(args.root)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    docs: list[dict[str, Any]] = []
    dates_seen: set[str] = set()
    for tier in tiers:
        for doc in load_tier(root, tier):
            docs.append(doc)
            dates_seen.update(doc.get("dates") or [])
    if not docs:
        print(f"no docs under {root}", file=sys.stderr)
        return 2
    md = render(docs, dates=sorted(dates_seen))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}  ({len(md)} bytes, {len(docs)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
