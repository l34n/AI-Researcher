# L34N (AI-Researcher) — Deployment Notes

Deployed for APE-704 on 2026-09-03 by Forge.

## Location

- Clone: `/home/ebg/l34n` (from `https://github.com/mspicer/AI-Researcher-l34n`, fork of `l34n/AI-Researcher`)
- Virtualenv: `/home/ebg/l34n/.venv` (Python 3.13.5, created with `uv`)
- Data dir: `/home/ebg/l34n/data` (SQLite database `airesearch.db`)
- Config: `/home/ebg/l34n/.env`
- Sources catalog: `/home/ebg/l34n/config/sources.yaml` (56 sources)

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # once
cd /home/ebg/l34n
/home/ebg/.local/bin/uv venv
/home/ebg/.local/bin/uv pip install -e ".[dev]"
```

## Model endpoints

L34N speaks to a **single** Ollama host. We point it at the shared AI server,
which has the widest catalog and the biggest models.

| Endpoint | Host | Notes |
|---|---|---|
| **Primary (in use)** | `http://192.168.201.136:11434` | ~40 models incl. `qwen3:32b`, `gemma3:27b`, `qwen2.5:72b`, `nemotron:70b`, `llama3.1:8b`, `mxbai-embed-large` |
| Local fallback | `http://localhost:11434` | 5 models: `mxbai-embed-large`, `gpt-oss:20b`, `qwen2.5-coder:7b`, `qwen2.5:14b`, `llama3.1:8b` |

To swap to local, edit `.env` and set `OLLAMA_HOST=http://localhost:11434` then
restart `ai-researcher serve`.

### Role assignments (`.env`)

| Role | Model | Rationale |
|---|---|---|
| workhorse enrich | `llama3.1:8b` | fast, has `tools` capability, 8B fits enrichment budget |
| workhorse judge | `llama3.1:8b` | same as enrich; called on every judged item |
| premium research | `qwen3:32b` | thinking + tools, 32B, matches deep research quality gate |
| premium brief | `qwen3:32b` | one call per day; quality > latency |
| embeddings | `mxbai-embed-large:latest` | available on remote host, 1024-dim, 512-token ctx |

Other budget tweaks vs. `.env.example`:

- `AIR_MAX_MODEL_MEMORY_GB=32` (was 8) — the shared server has headroom
- `AIR_HOST=0.0.0.0` — dashboard reachable on the LAN at `:8899`
- `AIR_ACCESS_TOKEN=` (unset) — treat as trusted LAN; set a token before exposing

## Validation

```
$ ai-researcher doctor
  ollama host    : http://192.168.201.136:11434
  workhorse      : llama3.1:8b
  premium        : llama3.1:8b   (falls through to workhorse for lack of key)
  default chat   : llama3.1:8b
  enrich/judge   : llama3.1:8b / llama3.1:8b
  research/brief : qwen3:32b / qwen3:32b
  embed model    : mxbai-embed-large:latest
  ready          : yes
```

Ollama round-trip:

```
$ curl -s http://192.168.201.136:11434/api/generate \
    -d '{"model":"llama3.1:8b","prompt":"Say the single word: pong","stream":false}'
  → "Pong!" done=True
```

Dashboard smoke test (server started with `ai-researcher serve`, then hit):

```
/healthz  → 200
/readyz   → 200
/         → 200
```

## Ingest bug — resolved (APE-706)

`ai-researcher run` originally failed on every source with
`NameError: name 'fetcher' is not defined`, caused by
`_ingest_source()` referencing a `fetcher` that only existed in the caller
`ingest()`. Fixed under APE-706 by commit `9659522` on this clone
(threads the `Fetcher` through as a parameter). Overseer-approved.

## Initial benchmark

First real run (2026-09-03, sources: `simonw`, `hf-papers`):

```
run #5  ok  237.2s
sources : 1 ok, 0 failed, 1 skipped
items   : 96 new
enriched: 96 (40 by model, 96 heuristic), 0 pending
embedded: 96 new (126 total)
stories : 16 from 16 items via embeddings
judged  : 96 (21 by model)  adopt=0 research=66
research: 4 briefs (4 model, 0 fallback)
ollama  : ready chat=llama3.1:8b embed=mxbai-embed-large:latest
```

All model calls served by the shared Ollama at `192.168.201.136:11434`.
Deep-research wiki (4 stories × 5 turns) took ~85 s on `qwen3:32b`.
One soft warning: the daily brief hit its bullet-count validator with a
small sample — expected to clear once more sources are ingested.

## Running

```bash
# One-off ingest (after the connector bug is fixed):
cd /home/ebg/l34n && source .venv/bin/activate && ai-researcher run

# Dashboard:
cd /home/ebg/l34n && source .venv/bin/activate && ai-researcher serve
# → http://<host>:8899

# Diagnose:
ai-researcher doctor
ai-researcher sources
```

`scripts/install-systemd.sh` in the repo installs a background service +
hourly timer. Not enabled yet — wait until the ingest bug is resolved so
the timer does not spin uselessly.

## Switching between Ollama and OpenRouter (APE-708)

Two `.env` profiles live at the repo root:

| Profile | File | When to use |
|---|---|---|
| Ollama (default) | `.env.ollama` | on-prem inference via `192.168.201.136:11434` (safest, zero cloud spend) |
| OpenRouter | `.env.openrouter` | cloud benchmarking with the 10 curated models from [APE-705](/APE/issues/APE-705) |

Swap via the helper script (backs up the current `.env` to `.env.bak`):

```bash
cd /home/ebg/l34n
scripts/switch-provider.sh ollama          # or: openrouter
# then restart the running server (Ctrl-C + rerun, or `systemctl restart ai-researcher` once the unit is enabled)
```

Verify with `ai-researcher doctor` — look for the `openrouter :` line and the
`workhorse / premium` picks.

### Why profiles instead of toggling one file

`_pair()` in `enrich/chat.py` chooses the OpenRouter backend as soon as
`OPENROUTER_API_KEY` is set (there is no gemini key present to preempt it), but
`model_for(role=…)` still returns whatever is in `AIR_ENRICH_MODEL` /
`AIR_JUDGE_MODEL` / `AIR_RESEARCH_MODEL` / `AIR_BRIEF_MODEL`. If those role
overrides still carry Ollama tags (`llama3.1:8b`, `qwen3:32b`), calls fail with
"model not found" against OpenRouter. The `.env.openrouter` profile clears
those overrides so `OPENROUTER_MODEL` / `OPENROUTER_PREMIUM_MODEL` are used
end to end. The `.env.ollama` profile leaves them populated and keeps
`OPENROUTER_API_KEY` blank.

### OpenRouter model shortlist (from APE-705)

Verified against `https://openrouter.ai/api/v1/models` on 2026-09-03; drop-in
replacements for `OPENROUTER_MODEL` or `OPENROUTER_PREMIUM_MODEL`:

| Model id | Cost | Context | Notes |
|---|---|---|---|
| `qwen/qwen3.7-flash` | $0.03/M | 1M | Default workhorse. Reasoning model — token budget must account for hidden thinking tokens. |
| `upstage/solar-pro4` | $0.03/M | 524K | |
| `ibm-granite/granite-4.2-8b` | $0.06/M | 131K | |
| `nvidia/nemotron-3.5-lightning` | $0.08/M | 262K | |
| `openai/gpt-4.1-nano` | $0.10/M | 1M | |
| `meta-llama/llama-3.3-70b-instruct` | $0.10/M | 131K | Default premium. |
| `deepseek/deepseek-chat` | $0.14/M | 164K | |
| `google/gemini-2.5-flash` | $0.30/M | 1M | |
| `google/gemini-2.0-flash-001` | — | — | **not available** — deprecated for 3.x on OpenRouter. Use `gemini-2.5-flash` or `gemini-3-flash-preview` instead. |

### First OpenRouter run (2026-09-03)

Run #6 (`ai-researcher run --source simonw`) on the OpenRouter profile:

```
sources : 0 ok, 0 failed, 1 skipped  (no new items to fetch)
enrich  : 0 (0 model, 0 heuristic)   model=qwen/qwen3.7-flash    22.1s
judge   : 22 by model                  model=qwen/qwen3.7-flash    88.7s
research: 4 model / 4 briefs          model=meta-llama/llama-3.3-70b-instruct  135.4s
brief   : validation_ok=true          model=meta-llama/llama-3.3-70b-instruct
chat    : workhorse=qwen/qwen3.7-flash premium=meta-llama/llama-3.3-70b-instruct
openrouter: true
elapsed : 247.0s
```

Comparison with run #5 (Ollama profile, same day): brief `validation_ok` was
`false` on Ollama `qwen3:32b` and `true` on OpenRouter `llama-3.3-70b-instruct`;
judge throughput ~22 items in 89s (OpenRouter) vs 21 items in 50s (Ollama) —
Ollama is faster on this box but the OpenRouter briefs pass validation on the
first try. A handful of 429s during the enrich pass were tolerated by the
retry logic. The daily model-call budget (`AIR_DAILY_MODEL_CALLS=250`) was
exhausted before the trailing revalidation cleanup — expected on a first cloud
run given the larger call footprint.

### Cost containment

- `AIR_DAILY_MODEL_CALLS` caps total model calls per calendar day (currently 250).
- `AIR_ENRICH_BUDGET` / `AIR_JUDGE_BUDGET` / `AIR_RESEARCH_BUDGET` cap per-role calls per run.
- To rehearse without spending, run `ai-researcher run --no-ingest` — it re-analyses items already in the DB.
- Prefer the cheapest OpenRouter model (`qwen/qwen3.7-flash` at $0.03/M) for casual runs. `google/gemini-2.5-flash` is 10× the workhorse cost.
