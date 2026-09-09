# L34N Historical Backtest — APE-712

**Deliverable of:** [APE-712](/APE/issues/APE-712) · **Parent:** [APE-709](/APE/issues/APE-709) · **Rubric:** [APE-710](/APE/issues/APE-710) v1.0 · **Harness:** [APE-711](/APE/issues/APE-711) reuse

**Backtest dates:** 2026-08-22, 2026-08-26, 2026-08-30, 2026-09-01, 2026-09-03

**Corpus source:** `/home/ebg/l34n/data/airesearch.db` — up to 8 relevance-ranked items per date, one brief per model per date.

## Method

This is a **consistency backtest**, not a labelled evaluation. Historical items pulled from the L34N SQLite store have no ground-truth `expected` labels, so the harness metrics that need labels — relevance precision/recall, readiness_agreement, injection_following_rate — are excluded. What we do score, per date, is what the schema validator + citation + echo detector return on the model's actual brief output:

- `format_compliance` — brief passed the L34N schema validator
- `citation_completeness` — fraction of brief sections that cited a source
- `factuality_score` — derived from format + citation + non-hallucinated ready
- `prompt_echo_rate` — model regurgitated the prompt
- `hallucinated_recommendation_rate` — model wrote an ungated adopt/ready (dropped before shipping when nothing was gated; fatal otherwise)
- `fallback_rate` — L34N fell back to the deterministic brief

Consistency across dates is the coefficient of interest: **mean** shows how well the model performed on average, **std** shows how much that swung day-to-day. A single-number `consistency_score = 1 − avg(std of format/citation/factuality)` is provided as a shorthand — but read it alongside the mean, because a model that is uniformly bad will score as very consistent.

## Consistency Across Dates

Sorted by (mean factuality × consistency_score) descending — we want models that are both good on average AND stable.

| Model | Tier | Cases | Consistency | Format (μ±σ) | Citation (μ±σ) | Factuality (μ±σ) | Fallback (μ±σ) | Wall (s) | Cost ($) |
|---|---|---:|---:|---|---|---|---|---:|---:|
| `deepseek/deepseek-chat` | paid | 5 | **1.000** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 47.5 | 0.0015 |
| `upstage/solar-pro4` | paid | 5 | **1.000** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 23.1 | 0.0008 |
| `llama3.1:8b` | local | 5 | **1.000** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 28.8 | 0.0000 |
| `nvidia/nemotron-3-super-120b-a12b:free` | free | 5 | **1.000** | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 27.9 | 0.0000 |
| `google/gemini-2.5-flash` | paid | 5 | **0.941** | 1.00 ± 0.00 | 0.93 ± 0.13 | 0.98 ± 0.04 | 0.00 ± 0.00 | 13.2 | 0.0068 |
| `gemma3:27b` | local | 5 | **0.927** | 1.00 ± 0.00 | 0.87 ± 0.16 | 0.96 ± 0.05 | 0.00 ± 0.00 | 116.1 | 0.0000 |
| `openai/gpt-4.1-nano` | paid | 5 | **0.822** | 0.80 ± 0.40 | 1.00 ± 0.00 | 0.93 ± 0.13 | 0.00 ± 0.00 | 18.6 | 0.0016 |
| `qwen/qwen3.7-flash` | paid | 5 | **0.822** | 0.80 ± 0.40 | 1.00 ± 0.00 | 0.93 ± 0.13 | 0.00 ± 0.00 | 16.3 | 0.0007 |
| `nvidia/nemotron-3.5-lightning:free` | free | 5 | **0.822** | 0.80 ± 0.40 | 1.00 ± 0.00 | 0.93 ± 0.13 | 0.00 ± 0.00 | 103.4 | 0.0000 |
| `qwen3:32b` | local | 5 | **0.782** | 0.60 ± 0.49 | 1.00 ± 0.00 | 0.87 ± 0.16 | 0.00 ± 0.00 | 92.7 | 0.0000 |

## Format Compliance — Dates × Models

| Date | deepseek/deepseek-chat | google/gemini-2.5-flash | openai/gpt-4.1-nano | qwen/qwen3.7-flash | upstage/solar-pro4 | gemma3:27b | llama3.1:8b | qwen3:32b | nvidia/nemotron-3.5-lightning:free | nvidia/nemotron-3-super-120b-a12b:free |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-22 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 2026-08-26 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2026-08-30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| 2026-09-01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 2026-09-03 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Factuality — Dates × Models

| Date | deepseek/deepseek-chat | google/gemini-2.5-flash | openai/gpt-4.1-nano | qwen/qwen3.7-flash | upstage/solar-pro4 | gemma3:27b | llama3.1:8b | qwen3:32b | nvidia/nemotron-3.5-lightning:free | nvidia/nemotron-3-super-120b-a12b:free |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-22 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 |
| 2026-08-26 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2026-08-30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 |
| 2026-09-01 | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 0.89 | 1.00 | 0.67 | 1.00 | 1.00 |
| 2026-09-03 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Analysis

- **Highest average factuality**: `deepseek/deepseek-chat` (paid) — μ=1.00, σ=0.00
- **Most consistent (with factuality ≥ 0.5)**: `deepseek/deepseek-chat` (consistency_score=1.000)
- **Cheapest paid model this sweep**: `qwen/qwen3.7-flash` — $0.0007 for 5 briefs
- **Total sweep cost**: $0.0115 across 50 model-date runs, 487.5s wall time

## Recommendation

- **Primary production pick**: `deepseek/deepseek-chat` (paid). Highest mean factuality × consistency, cost $0.0015 for a 5-day rolling sweep.
- **Local fallback (zero-cost)**: `llama3.1:8b` — factuality μ=1.00, consistency_score=1.000.
- Re-run this backtest weekly (routine or manual trigger) so any silent quality drift shows up as a delta in `consistency_score` or `factuality_score` mean.

## Caveats

- Historical corpus items were **fetched today** (2026-09-03) — `published_at` gives the source-of-record publication date, but there is no true archival replay of the RSS feeds themselves. Publisher backdating and later edits are not captured.
- Items per date are top-N by classifier `relevance_score`, so days with very few relevant items (see 2026-08-22 with 5) give a smaller corpus and noisier scores.
- Ollama speeds are LAN-only against the local backend; do not compare against OpenRouter wall times.
- `consistency_score` weights the three positive-signal metrics equally; if you care most about schema compliance in production, prefer the Format-Compliance grid over the single-number ranking.
