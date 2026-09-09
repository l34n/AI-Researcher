"""Render the APE-711 benchmark report from per-model JSON outputs.

Reads `data/benchmark-results/*.json` (and optional subdirs), renders a
Markdown table + dimension breakdown, and writes it to the target path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_all(dirs: list[Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.json")):
            if path.name == "index.json":
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            slug = doc.get("model_slug") or path.stem
            if slug in seen:
                continue
            seen.add(slug)
            docs.append(doc)
    return docs


def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _call_failure_rate(doc: dict[str, Any]) -> float | None:
    rate = doc.get("call_failure_rate")
    if rate is not None:
        return float(rate)
    stats = doc.get("call_stats") or {}
    calls = stats.get("calls") or 0
    return (stats.get("failures") or 0) / calls if calls else None


def _invalid(doc: dict[str, Any]) -> bool:
    """A run whose provider calls mostly failed scored the deterministic
    fallback, not the model. Rank it last and say so."""
    rate = _call_failure_rate(doc)
    return rate is not None and rate > 0.5


def _dq_badge(doc: dict[str, Any]) -> str:
    if doc.get("failed"):
        return "**FAILED**"
    if _invalid(doc):
        return "**INVALID**"
    r = doc.get("rubric") or {}
    if r.get("disqualified"):
        return "**DQ**"
    c = float(r.get("composite") or 0)
    if c >= 80:
        return "Excellent"
    if c >= 65:
        return "Pass"
    if c >= 50:
        return "Marginal"
    return "Fail"


def render(docs: list[dict[str, Any]]) -> str:
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            999 if d.get("failed") else
            (500 if _invalid(d) else 0) - (d.get("rubric") or {}).get("composite", -1)
        ),
    )
    total_cost = sum(d.get("cost_usd_estimate", 0) or 0 for d in docs if not d.get("failed"))
    total_calls = sum((d.get("call_stats") or {}).get("calls", 0) for d in docs if not d.get("failed"))

    lines: list[str] = []
    lines.append("# L34N Model Benchmark — APE-711")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    corpus_v = next((d.get("corpus_version") for d in docs if d.get("corpus_version")), "?")
    prompt_v = next((d.get("prompt_version") for d in docs if d.get("prompt_version")), "?")
    harness_v = next((d.get("harness_version") for d in docs if d.get("harness_version")), "?")
    lines.append(f"**Corpus:** v{corpus_v} · **Prompt:** {prompt_v} · **Harness:** {harness_v}")
    rubric_v = next((d.get("rubric_version") for d in docs if d.get("rubric_version")), "1.0")
    lines.append(f"**Rubric:** [APE-710](/APE/issues/APE-710) v{rubric_v} · **Scoring layer:** `schema`")
    lines.append(f"**Sweep totals:** {len(docs)} models, {total_calls} generations, ${total_cost:.4f} spent")
    lines.append("")
    lines.append("## Composite Ranking")
    lines.append("")
    lines.append("| Rank | Model | Tier | Composite | Halluc. −pts | Verdict | Calls failed | Wall (s) | Cost ($) | DQ |")
    lines.append("|---:|---|---|---:|---:|---|---:|---:|---:|---|")
    rank = 0
    for d in docs_sorted:
        if d.get("failed"):
            continue
        rank += 1
        r = d["rubric"]
        stats = d.get("call_stats") or {}
        failed = f"{stats.get('failures', 0)}/{stats.get('calls', 0)}"
        lines.append(
            f"| {rank} | `{d['model_id']}` | {d['tier']} "
            f"| {r['composite']:.2f} | {r.get('hallucination_deduction', 0.0):.1f} | {_dq_badge(d)} "
            f"| {failed} "
            f"| {d['wall_clock_s']:.1f} "
            f"| {d['cost_usd_estimate']:.4f} "
            f"| {'yes' if r['disqualified'] else '—'} |"
        )
    lines.append("")
    invalid = [d for d in docs_sorted if _invalid(d) and not d.get("failed")]
    if invalid:
        lines.append("**INVALID** rows: more than half of the provider calls failed "
                     "(404/400 slug, empty reasoning-only output, timeouts), so the "
                     "scores reflect the deterministic fallback brief, not the model. "
                     "They are ranked last and must not be read as model quality.")
        lines.append("")

    lines.append("## Per-Dimension Scores")
    lines.append("")
    lines.append("| Model | Relevance | Accuracy | Depth | Actionability | Cost | Speed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in docs_sorted:
        if d.get("failed"):
            continue
        r = d["rubric"]
        lines.append(
            f"| `{d['model_id']}` "
            f"| {r['relevance']:.1f} | {r['accuracy']:.1f} | {r['depth']:.1f} "
            f"| {r['actionability']:.1f} | {r['cost']:.1f} | {r['speed']:.1f} |"
        )
    lines.append("")

    lines.append("## Full-Fidelity Judge Metrics")
    lines.append("")
    lines.append("Depth and Actionability scores use these numbers when the enrichment "
                 "pass ran. Each research-tier case's items are scored by the candidate "
                 "model with `enrich/judge.py::SYSTEM` + `PROMPT` + `SCHEMA`, then blended "
                 "with the deterministic `judge_text()` heuristic prior via `blend()`. "
                 "`Verdict Agreement` is the fraction of cases whose blended verdict landed "
                 "in `expected.verdict_in`; `Judge JSON` is the model's JSON parse rate.")
    lines.append("")
    lines.append("| Model | Full-Fidelity | Cases | Judge Calls | Judge JSON | Avg Q | Avg U | Avg Readiness | Verdict Agreement | Adapt-Complete Rate |")
    lines.append("|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in docs_sorted:
        if d.get("failed"):
            continue
        enr = d.get("enrichment") or {}
        cases_n = enr.get("cases", 0) if enr else 0
        calls = enr.get("judge_calls", 0) if enr else 0
        parse = enr.get("judge_json_parse_rate", 0) if enr else 0
        q = enr.get("avg_quality") if enr else None
        u = enr.get("avg_usefulness") if enr else None
        rdy = enr.get("avg_readiness") if enr else None
        agree = enr.get("verdict_agreement") if enr else None
        ar = enr.get("adapt_complete_rate", 0) if enr else 0
        lines.append(
            f"| `{d['model_id']}` "
            f"| {'yes' if d.get('full_fidelity') else 'no'} "
            f"| {cases_n} | {calls} | {_fmt(parse, 3)} "
            f"| {_fmt(q, 3)} | {_fmt(u, 3)} | {_fmt(rdy, 3)} "
            f"| {_fmt(agree, 3)} | {_fmt(ar, 3)} |"
        )
    lines.append("")

    # Per-case verdict rollup — show which cases each model got right/wrong
    # so a reviewer can see whether a low verdict_agreement is a systematic
    # failure or noise on one hostile case.
    per_case_docs = [d for d in docs_sorted
                     if not d.get("failed") and d.get("enrichment_details")]
    if per_case_docs:
        lines.append("## Per-Case Judge Verdicts")
        lines.append("")
        # Union of case ids seen across all models, preserving corpus order.
        seen_cases: list[str] = []
        seen_set: set[str] = set()
        for d in per_case_docs:
            for row in d["enrichment_details"]:
                cid = row.get("case_id")
                if cid and cid not in seen_set:
                    seen_set.add(cid)
                    seen_cases.append(cid)
        header = "| Model | " + " | ".join(seen_cases) + " |"
        sep = "|---|" + ":-:|" * len(seen_cases)
        lines.append(header)
        lines.append(sep)
        for d in per_case_docs:
            by_id = {row.get("case_id"): row for row in d["enrichment_details"]}
            cells = []
            for cid in seen_cases:
                row = by_id.get(cid)
                if not row:
                    cells.append("—")
                    continue
                verdict = row.get("verdict") or "—"
                expected = row.get("verdict_expected") or []
                if not expected:
                    marker = "·"
                elif row.get("verdict_matches"):
                    marker = "✓"
                else:
                    marker = "✗"
                cells.append(f"{marker} {verdict}")
            lines.append(f"| `{d['model_id']}` | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Legend: ✓ = blended verdict ∈ `expected.verdict_in`; "
                     "✗ = miss; · = no expected verdict on this case.")
        lines.append("")

    lines.append("## Raw Harness Metrics (schema layer)")
    lines.append("")
    keys = [
        "format_compliance", "citation_completeness", "factuality_score",
        "ai_relevance_precision", "ai_relevance_recall",
        "dedup_precision", "cluster_purity",
        "readiness_agreement", "prompt_echo_rate",
        "hallucinated_recommendation_rate", "injection_following_rate",
        "fallback_rate",
    ]
    lines.append("| Model | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---:|" * len(keys))
    for d in docs_sorted:
        if d.get("failed") or "metrics_by_layer" not in d:
            continue
        m = d["metrics_by_layer"].get("schema") or {}
        row = " | ".join(_fmt(m.get(k, 0), 3) for k in keys)
        lines.append(f"| `{d['model_id']}` | {row} |")
    lines.append("")

    fails = [d for d in docs if d.get("failed")]
    if fails:
        lines.append("## Failures")
        lines.append("")
        for d in fails:
            lines.append(f"- `{d['model_id']}` ({d['tier']}, {d['provider']}) — {d.get('error','?')}")
        lines.append("")

    dq_notes = [d for d in docs
                if not d.get("failed") and (d.get("rubric") or {}).get("disqualifiers")]
    if dq_notes:
        lines.append("## Disqualifier Details")
        lines.append("")
        for d in dq_notes:
            lines.append(f"- `{d['model_id']}`: {'; '.join(d['rubric']['disqualifiers'])}")
        lines.append("")

    lines.append("## Notes & Caveats")
    lines.append("")
    lines.append("- **Judge wiring (full-fidelity)** — the candidate model is called with "
                 "`enrich/judge.py::SYSTEM` + `PROMPT` + `SCHEMA` for every source item in "
                 "each research-tier case. The model's JSON output is blended with the "
                 "deterministic `judge_text()` heuristic prior via `blend()` (0.55 model + "
                 "0.45 heuristic on Q/P/F/U, verdict clamped to ≤1 step from the heuristic "
                 "band with adopt brakes re-applied). This is the same code path that scores "
                 "items in production.")
    lines.append("- **Depth (full-fidelity)** — `Q × 0.40 + U × 0.35 + adapt_complete_rate × 0.15 "
                 "+ (1 − prompt_echo_rate) × 0.10`, where Q and U are the blended judge scores. "
                 "Models with `full-fidelity = no` fall back to a proxy "
                 "(format + citation + non-echo) — those depth scores are not directly comparable "
                 "to full-fidelity depth scores.")
    lines.append("- **Actionability (full-fidelity)** — `verdict_agreement × 0.60 "
                 "+ adapt_complete_rate × 0.40` per rubric §4, where `verdict_agreement` is "
                 "the fraction of cases whose *blended* judge verdict lands in "
                 "`expected.verdict_in`. Fall-back to the harness's heuristic-only "
                 "`readiness_agreement` if the enrichment pass did not run — those actionability "
                 "scores are not directly comparable to full-fidelity ones.")
    lines.append("- **Accuracy** — per rubric §2: `factuality × 0.40 + citation × 0.25 "
                 "+ (1 − hallucinated) × 0.25 + (1 − injection) × 0.10`.")
    lines.append("- **Cost score** — free/local tier anchored at 100 (zero cost). Paid models "
                 "are quality-per-dollar `(accuracy + relevance) / cost`, normalised so the "
                 "best-viable paid model in the sweep scores 100. Non-viable (disqualified) "
                 "and zero-cost models are excluded from the baseline.")
    lines.append("- **Speed comparability** — OpenRouter latency includes network hops; Ollama "
                 "latency is LAN-only. Do not cross-compare speed across backends.")
    lines.append("- **Hallucinated-recommendation deduction (rubric v1.1, "
                 "[APE-724](/APE/issues/APE-724))** — `hallucinated_recommendation_rate × 20` "
                 "points are subtracted from the weighted composite (`Halluc. −pts`). Since "
                 "harness validate-v2 an ungated Ready section is dropped before the brief "
                 "ships, so the rate measures behaviour quality, not a safety risk; it no "
                 "longer disqualifies.")
    lines.append("- **Band cutoffs** — Composite ≥ 80 = Excellent, ≥ 65 = Pass, ≥ 50 = Marginal, "
                 "< 50 = Fail. Hard disqualifiers: injection_following_rate > 0.05, "
                 "fallback_rate > 0.40, format_compliance < 0.50.")
    lines.append("- **Enrichment corpus subset** — full-fidelity Depth/Actionability run on the "
                 "5 corpus cases whose `expected.verdict_in` includes adopt/research/spike "
                 "(the 'research-tier' filter per rubric §3): `sum-single-hf`, `model-release-version`, "
                 "`paper-with-code`, `inject-indirect`, `ready-valid-adopt`.")
    lines.append("- Cases: 21 in corpus v1.0.0. Reference: `ai_researcher/eval/corpus.py`.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render APE-711 benchmark report.")
    ap.add_argument("--in", dest="inputs", action="append",
                    default=None, metavar="DIR",
                    help="input dir with per-model JSONs (repeatable)")
    # Derive defaults from repo layout instead of hard-coding an operator path.
    import os
    l34n_root = Path(os.environ.get("L34N_ROOT",
                                    Path(__file__).resolve().parent.parent))
    default_out = l34n_root / "docs" / "benchmark-results.md"
    default_in = l34n_root / "data" / "benchmark-results"
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args(argv)

    dirs = [Path(p) for p in (args.inputs or [
        str(default_in),
        str(default_in / "free"),
        str(default_in / "local"),
        str(default_in / "paid"),
    ])]
    docs = _load_all(dirs)
    if not docs:
        print("no benchmark JSONs found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(docs), encoding="utf-8")
    print(f"wrote {out_path} ({len(docs)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
