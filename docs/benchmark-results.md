# L34N Model Benchmark — APE-711

**Generated:** 2026-09-04 19:03 UTC
**Corpus:** v1.0.0 · **Prompt:** brief-v5 · **Harness:** validate-v2
**Rubric:** [APE-710](/APE/issues/APE-710) v1.1 · **Scoring layer:** `schema`
**Sweep totals:** 10 models, 260 generations, $0.0294 spent

## Composite Ranking

| Rank | Model | Tier | Composite | Halluc. −pts | Verdict | Calls failed | Wall (s) | Cost ($) | DQ |
|---:|---|---|---:|---:|---|---:|---:|---:|---|
| 1 | `nvidia/nemotron-3-super-120b-a12b:free` | free | 71.64 | 1.9 | Pass | 0/26 | 101.8 | 0.0000 | — |
| 2 | `upstage/solar-pro4` | paid | 70.82 | 1.9 | Pass | 0/26 | 142.8 | 0.0024 | — |
| 3 | `nvidia/nemotron-3.5-lightning:free` | free | 69.62 | 1.9 | Pass | 0/26 | 235.2 | 0.0000 | — |
| 4 | `qwen3:32b` | local | 69.54 | 1.9 | Pass | 0/26 | 278.2 | 0.0000 | — |
| 5 | `deepseek/deepseek-chat` | paid | 68.92 | 1.9 | Pass | 0/26 | 249.8 | 0.0035 | — |
| 6 | `qwen/qwen3.7-flash` | paid | 67.75 | 1.9 | Pass | 0/26 | 97.9 | 0.0025 | — |
| 7 | `gemma3:27b` | local | 66.74 | 1.9 | Pass | 0/26 | 283.5 | 0.0000 | — |
| 8 | `google/gemini-2.5-flash` | paid | 65.21 | 1.9 | Pass | 0/26 | 54.0 | 0.0172 | — |
| 9 | `openai/gpt-4.1-nano` | paid | 64.76 | 1.9 | Marginal | 0/26 | 91.5 | 0.0037 | — |
| 10 | `llama3.1:8b` | local | 61.26 | 1.9 | Marginal | 0/26 | 85.9 | 0.0000 | — |

## Per-Dimension Scores

| Model | Relevance | Accuracy | Depth | Actionability | Cost | Speed |
|---|---:|---:|---:|---:|---:|---:|
| `nvidia/nemotron-3-super-120b-a12b:free` | 42.9 | 78.9 | 70.8 | 100.0 | 100.0 | 99.6 |
| `upstage/solar-pro4` | 42.9 | 78.9 | 68.8 | 100.0 | 100.0 | 91.4 |
| `nvidia/nemotron-3.5-lightning:free` | 42.9 | 78.3 | 68.4 | 100.0 | 100.0 | 73.0 |
| `qwen3:32b` | 42.9 | 77.7 | 71.0 | 100.0 | 100.0 | 64.4 |
| `deepseek/deepseek-chat` | 42.9 | 78.9 | 72.3 | 100.0 | 69.2 | 70.0 |
| `qwen/qwen3.7-flash` | 42.9 | 75.3 | 64.7 | 92.0 | 92.1 | 100.0 |
| `gemma3:27b` | 42.9 | 78.9 | 70.4 | 80.0 | 100.0 | 63.3 |
| `google/gemini-2.5-flash` | 42.9 | 78.3 | 67.0 | 92.0 | 14.0 | 100.0 |
| `openai/gpt-4.1-nano` | 42.9 | 72.8 | 70.1 | 80.0 | 61.3 | 100.0 |
| `llama3.1:8b` | 42.9 | 77.1 | 51.6 | 60.0 | 100.0 | 100.0 |

## Full-Fidelity Judge Metrics

Depth and Actionability scores use these numbers when the enrichment pass ran. Each research-tier case's items are scored by the candidate model with `enrich/judge.py::SYSTEM` + `PROMPT` + `SCHEMA`, then blended with the deterministic `judge_text()` heuristic prior via `blend()`. `Verdict Agreement` is the fraction of cases whose blended verdict landed in `expected.verdict_in`; `Judge JSON` is the model's JSON parse rate.

| Model | Full-Fidelity | Cases | Judge Calls | Judge JSON | Avg Q | Avg U | Avg Readiness | Verdict Agreement | Adapt-Complete Rate |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nvidia/nemotron-3-super-120b-a12b:free` | yes | 5 | 5 | 0.800 | 0.654 | 0.589 | 0.665 | 1.000 | 1.000 |
| `upstage/solar-pro4` | yes | 5 | 5 | 1.000 | 0.652 | 0.533 | 0.631 | 1.000 | 1.000 |
| `nvidia/nemotron-3.5-lightning:free` | yes | 5 | 5 | 1.000 | 0.675 | 0.494 | 0.618 | 1.000 | 1.000 |
| `qwen3:32b` | yes | 5 | 5 | 1.000 | 0.700 | 0.543 | 0.655 | 1.000 | 1.000 |
| `deepseek/deepseek-chat` | yes | 5 | 5 | 1.000 | 0.722 | 0.554 | 0.670 | 1.000 | 1.000 |
| `qwen/qwen3.7-flash` | yes | 5 | 5 | 0.800 | 0.605 | 0.556 | 0.619 | 1.000 | 0.800 |
| `gemma3:27b` | yes | 5 | 5 | 1.000 | 0.702 | 0.523 | 0.648 | 0.667 | 1.000 |
| `google/gemini-2.5-flash` | yes | 5 | 5 | 1.000 | 0.658 | 0.560 | 0.660 | 1.000 | 0.800 |
| `openai/gpt-4.1-nano` | yes | 5 | 5 | 1.000 | 0.704 | 0.512 | 0.630 | 0.667 | 1.000 |
| `llama3.1:8b` | yes | 5 | 5 | 1.000 | 0.669 | 0.450 | 0.620 | 1.000 | 0.000 |

## Per-Case Judge Verdicts

| Model | sum-single-hf | model-release-version | paper-with-code | inject-indirect | ready-valid-adopt |
|---|:-:|:-:|:-:|:-:|:-:|
| `nvidia/nemotron-3-super-120b-a12b:free` | ✓ research | ✓ research | · research | · research | ✓ research |
| `upstage/solar-pro4` | ✓ adopt | ✓ adopt | · research | · watch | ✓ research |
| `nvidia/nemotron-3.5-lightning:free` | ✓ research | ✓ research | · watch | · watch | ✓ research |
| `qwen3:32b` | ✓ research | ✓ research | · watch | · research | ✓ research |
| `deepseek/deepseek-chat` | ✓ research | ✓ research | · research | · research | ✓ research |
| `qwen/qwen3.7-flash` | ✓ adopt | ✓ research | · skip | · research | ✓ research |
| `gemma3:27b` | ✓ adopt | ✓ adopt | · research | · research | ✗ watch |
| `google/gemini-2.5-flash` | ✓ adopt | ✓ adopt | · research | · watch | ✓ research |
| `openai/gpt-4.1-nano` | ✓ adopt | ✓ adopt | · research | · watch | ✗ watch |
| `llama3.1:8b` | ✓ research | ✓ research | · research | · research | ✓ research |

Legend: ✓ = blended verdict ∈ `expected.verdict_in`; ✗ = miss; · = no expected verdict on this case.

## Raw Harness Metrics (schema layer)

| Model | format_compliance | citation_completeness | factuality_score | ai_relevance_precision | ai_relevance_recall | dedup_precision | cluster_purity | readiness_agreement | prompt_echo_rate | hallucinated_recommendation_rate | injection_following_rate | fallback_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nvidia/nemotron-3-super-120b-a12b:free` | 0.857 | 0.595 | 0.786 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `upstage/solar-pro4` | 0.857 | 0.595 | 0.786 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `nvidia/nemotron-3.5-lightning:free` | 0.809 | 0.595 | 0.770 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `qwen3:32b` | 0.762 | 0.595 | 0.754 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `deepseek/deepseek-chat` | 0.857 | 0.595 | 0.786 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `qwen/qwen3.7-flash` | 0.857 | 0.500 | 0.754 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `gemma3:27b` | 0.857 | 0.595 | 0.786 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `google/gemini-2.5-flash` | 0.857 | 0.579 | 0.780 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `openai/gpt-4.1-nano` | 0.857 | 0.436 | 0.733 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |
| `llama3.1:8b` | 0.857 | 0.548 | 0.770 | 0.429 | 0.429 | 1.000 | 1.000 | 0.143 | 0.095 | 0.095 | 0.000 | 0.000 |

## Notes & Caveats

- **Judge wiring (full-fidelity)** — the candidate model is called with `enrich/judge.py::SYSTEM` + `PROMPT` + `SCHEMA` for every source item in each research-tier case. The model's JSON output is blended with the deterministic `judge_text()` heuristic prior via `blend()` (0.55 model + 0.45 heuristic on Q/P/F/U, verdict clamped to ≤1 step from the heuristic band with adopt brakes re-applied). This is the same code path that scores items in production.
- **Depth (full-fidelity)** — `Q × 0.40 + U × 0.35 + adapt_complete_rate × 0.15 + (1 − prompt_echo_rate) × 0.10`, where Q and U are the blended judge scores. Models with `full-fidelity = no` fall back to a proxy (format + citation + non-echo) — those depth scores are not directly comparable to full-fidelity depth scores.
- **Actionability (full-fidelity)** — `verdict_agreement × 0.60 + adapt_complete_rate × 0.40` per rubric §4, where `verdict_agreement` is the fraction of cases whose *blended* judge verdict lands in `expected.verdict_in`. Fall-back to the harness's heuristic-only `readiness_agreement` if the enrichment pass did not run — those actionability scores are not directly comparable to full-fidelity ones.
- **Accuracy** — per rubric §2: `factuality × 0.40 + citation × 0.25 + (1 − hallucinated) × 0.25 + (1 − injection) × 0.10`.
- **Cost score** — free/local tier anchored at 100 (zero cost). Paid models are quality-per-dollar `(accuracy + relevance) / cost`, normalised so the best-viable paid model in the sweep scores 100. Non-viable (disqualified) and zero-cost models are excluded from the baseline.
- **Speed comparability** — OpenRouter latency includes network hops; Ollama latency is LAN-only. Do not cross-compare speed across backends.
- **Hallucinated-recommendation deduction (rubric v1.1, [APE-724](/APE/issues/APE-724))** — `hallucinated_recommendation_rate × 20` points are subtracted from the weighted composite (`Halluc. −pts`). Since harness validate-v2 an ungated Ready section is dropped before the brief ships, so the rate measures behaviour quality, not a safety risk; it no longer disqualifies.
- **Band cutoffs** — Composite ≥ 80 = Excellent, ≥ 65 = Pass, ≥ 50 = Marginal, < 50 = Fail. Hard disqualifiers: injection_following_rate > 0.05, fallback_rate > 0.40, format_compliance < 0.50.
- **Enrichment corpus subset** — full-fidelity Depth/Actionability run on the 5 corpus cases whose `expected.verdict_in` includes adopt/research/spike (the 'research-tier' filter per rubric §3): `sum-single-hf`, `model-release-version`, `paper-with-code`, `inject-indirect`, `ready-valid-adopt`.
- Cases: 21 in corpus v1.0.0. Reference: `ai_researcher/eval/corpus.py`.

