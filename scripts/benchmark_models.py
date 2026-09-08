"""L34N model benchmark — APE-711.

Runs the fixed backtest corpus (`ai_researcher.eval.corpus`) through each
candidate model (OpenRouter or Ollama), computes rubric composite scores
per APE-710, and writes per-model results plus a summary Markdown table.

Usage:
    OPENROUTER_API_KEY=... python scripts/benchmark_models.py \\
        --profile free               # or local | paid | all
    python scripts/benchmark_models.py --profile local --model qwen3:32b
    python scripts/benchmark_models.py --list

Outputs:
    data/benchmark-results/<slug>.json     — one file per model
    data/benchmark-results/index.json      — sweep summary
    (report is rendered separately by scripts/benchmark_report.py)

Notes:
- Only 'schema' and 'fallback' layers are scored (per rubric).
- Live generations reuse L34N's prompt/system so scores are apples-to-apples
  with the production brief pipeline.
- Cost estimation uses per-model catalog prices below; token counts come
  from the OpenRouter response `usage` block when present.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Derive the repo root from the script location so anyone can clone the repo
# and run `python scripts/benchmark_models.py` without editing paths. The
# ``L34N_ROOT`` env var still wins if set (e.g. running the script from a
# non-standard install location).

L34N_ROOT = Path(os.environ.get("L34N_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(L34N_ROOT / "src"))

from ai_researcher.config import Settings  # noqa: E402
from ai_researcher.eval.corpus import CASES, CORPUS_VERSION  # noqa: E402
from ai_researcher.eval.harness import run_corpus  # noqa: E402
from ai_researcher.eval.metrics import summarise  # noqa: E402
from ai_researcher.enrich.chat import OpenAICompatChat  # noqa: E402
from ai_researcher.enrich.ollama import OllamaClient  # noqa: E402
from ai_researcher.trends.brief import (  # noqa: E402
    PROMPT,
    SYSTEM,
    _render_prompt_stories,
    _render_ready,
    render_brief_prompt,
)
from ai_researcher.sanitize import fence  # noqa: E402
from ai_researcher.research.schema import SCHEMA, TURNS, adapt_complete  # noqa: E402
from ai_researcher.research.wiki import parse_scores  # noqa: E402
from ai_researcher.enrich.judge import (  # noqa: E402
    PROMPT as JUDGE_PROMPT,
    SCHEMA as JUDGE_SCHEMA,
    SYSTEM as JUDGE_SYSTEM,
    VERDICTS as JUDGE_VERDICTS,
    blend as judge_blend,
    judge_text,
)

# ── Model catalog (APE-711) ────────────────────────────────────────────────────

# Per-million-token prices. `None` = no cost (local Ollama or free tier).
# Update as OpenRouter catalog changes; verified 2026-09-03.


@dataclass
class ModelSpec:
    slug: str  # filename-safe id
    provider: str  # "openrouter" | "ollama"
    model: str  # exact provider model id
    tier: str  # "free" | "local" | "paid"
    input_per_m: float | None = None
    output_per_m: float | None = None
    notes: str = ""
    # OpenRouter only. Reasoning/thinking models spend `max_tokens` on hidden
    # reasoning and return empty `content` (finish_reason=length), which the
    # harness then scores as a fallback. Default off; opt in per model.
    reasoning: bool = False


DEFAULT_MATRIX_PATH = L34N_ROOT / "scripts" / "benchmark_matrix.yaml"


def load_matrix(path: Path | None = None) -> list[ModelSpec]:
    """Load the model matrix from a YAML file.

    Format::

        models:
          - slug: or-gemini-2-5-flash
            provider: openrouter
            model: google/gemini-2.5-flash
            tier: paid
            input_per_m: 0.30
            output_per_m: 2.50
            notes: ""

    Anyone extending the suite adds/removes rows in the YAML — no code
    edits required.
    """
    import yaml  # pyyaml is a project dep
    p = Path(path) if path else DEFAULT_MATRIX_PATH
    if not p.is_file():
        raise FileNotFoundError(f"matrix file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = data.get("models") or []
    specs: list[ModelSpec] = []
    for row in rows:
        specs.append(ModelSpec(
            slug=row["slug"],
            provider=row["provider"],
            model=row["model"],
            tier=row["tier"],
            input_per_m=row.get("input_per_m"),
            output_per_m=row.get("output_per_m"),
            notes=row.get("notes", ""),
            reasoning=bool(row.get("reasoning", False)),
        ))
    return specs


# Loaded lazily by the CLI. Kept as a module-global for backwards compat
# with anything that imported ``MODEL_MATRIX`` directly.
MODEL_MATRIX: list[ModelSpec] = []


def by_slug(slug: str, matrix: list[ModelSpec] | None = None) -> ModelSpec | None:
    for s in (matrix if matrix is not None else MODEL_MATRIX):
        if s.slug == slug or s.model == slug:
            return s
    return None


# ── Prompt shape (mirrors ai_researcher.trends.brief) ─────────────────────────

def _case_to_prompt(case: dict[str, Any]) -> str:
    """Render one eval case into the L34N brief prompt.

    Cases hold either a single `item` or a list `items`/`stories`. We
    normalise into the story fixture shape brief.py expects, then use
    L34N's own `_render_prompt_stories` so the resulting text is
    byte-identical to production.
    """
    stories: list[dict[str, Any]] = []
    if case.get("stories"):
        stories = list(case["stories"])
    else:
        raw_items = case.get("items") or ([case["item"]] if case.get("item") else [])
        for i, item in enumerate(raw_items, 1):
            stories.append({
                "id": i,
                "label": (item.get("title") or "item")[:140],
                "summary": (item.get("body") or "")[:220],
                "category": item.get("category") or "opinion-analysis",
                "source_count": 1,
                "sources": [item.get("source") or item.get("kind") or "rss"],
                "item_ids": [i],
                "freshness_status": "fresh",
            })

    # Ensure every story has fields _render_prompt_stories reads.
    for s in stories:
        s.setdefault("sources", [])
        s.setdefault("source_count", max(1, len(s.get("sources") or [])))
        s.setdefault("freshness_status", "fresh")
        s.setdefault("item_ids", s.get("item_ids") or [s.get("id", 1)])
        s.setdefault("category", "opinion-analysis")
        s.setdefault("summary", s.get("summary") or "")
        s.setdefault("label", s.get("label") or "item")

    ready = case.get("ready") or []
    return render_brief_prompt(stories, rising_text=fence("RISING", "none", limit=40), ready=ready)


# ── Provider clients ──────────────────────────────────────────────────────────

@dataclass
class CallStats:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failures: int = 0
    wall_s: float = 0.0


# ── Production isolation (APE-727) ─────────────────────────────────────────────
# Settings.load() points ``data_dir`` at the production ``data/`` directory,
# and every generation that goes through ``ChatRouter`` charges
# ``consume_daily_budget`` against ``data/model_budget.json`` there. A full
# sweep is hundreds of generations, enough to exhaust ``AIR_DAILY_MODEL_CALLS``
# and degrade the production brief to the rules fallback for the rest of the
# day. Sweep clients therefore run on a private data directory with the daily
# cap disabled: the only production file a sweep may open is the SQLite corpus
# the backtest reads, and it never goes through ``Settings.data_dir``.

SWEEP_RUNTIME_DIR = L34N_ROOT / "data" / "benchmark-results" / ".runtime"


def isolate_settings(settings: Settings, runtime_dir: Path | None = None) -> Settings:
    """Point ``settings`` at a sweep-private data directory and lift the daily cap.

    Idempotent: applying it twice leaves the same directory in place. Returns
    the same object for call-site convenience.
    """
    runtime_dir = Path(runtime_dir or SWEEP_RUNTIME_DIR).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir = runtime_dir
    settings.daily_model_calls = 0
    return settings


def sweep_settings(runtime_dir: Path | None = None) -> Settings:
    """``Settings.load()`` for a benchmark or backtest run.

    Reads ``.env`` for provider keys and model tags like production, then
    moves the writable state off the production data directory.
    """
    settings = isolate_settings(Settings.load(), runtime_dir)
    # Ollama host from env / .env
    if os.environ.get("OLLAMA_HOST"):
        settings.ollama_host = os.environ["OLLAMA_HOST"].rstrip("/")
    return settings


def _is_isolated(settings: Settings) -> bool:
    """True when ``settings.data_dir`` is not the production data directory."""
    production = (Path(os.environ.get("AIR_DATA_DIR") or (L34N_ROOT / "data"))
                  .expanduser().resolve())
    return Path(settings.data_dir).resolve() != production


class ProviderClient:
    """Thin async wrapper that L34N's harness can drive synchronously."""

    def __init__(self, spec: ModelSpec, settings: Settings):
        self.spec = spec
        # Never let a sweep client charge the production daily budget, whatever
        # settings the caller built (APE-727).
        self.settings = isolate_settings(settings, runtime_dir=settings.data_dir
                                         if _is_isolated(settings) else None)
        self.stats = CallStats()
        self._loop = asyncio.new_event_loop()
        self._backend: Any = None
        # Brief output per corpus case. run_corpus() re-generates every case
        # per layer; the schema and fallback layers must judge the *same*
        # output, and regenerating doubled every sweep's cost and wall time.
        self._brief_cache: dict[str, str] = {}
        self.empty_length_hits = 0
        self._init_backend()

    def _init_backend(self) -> None:
        if self.spec.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._backend = OpenAICompatChat(
                self.settings,
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                extra_headers={
                    "HTTP-Referer": "https://apex.local/l34n-benchmark",
                    "X-Title": "L34N APE-711 benchmark",
                },
            )
        elif self.spec.provider == "ollama":
            # Force the exact model by overriding settings before construction.
            self.settings.ollama_chat_model = self.spec.model
            client = OllamaClient(self.settings)
            # Force-set the private attr; skips network-probing auto-selection.
            client._chat_model = self.spec.model  # noqa: SLF001
            client.available = True
            self._backend = client
        else:
            raise ValueError(f"unknown provider {self.spec.provider!r}")

    def close(self) -> None:
        async def _close() -> None:
            await self._backend.aclose()
        try:
            self._loop.run_until_complete(_close())
        finally:
            self._loop.close()

    # Sync entry-point invoked by run_corpus's generate closure.
    def generate(self, case: dict[str, Any], *, layer: str = "schema",
                 retry: bool = False) -> str:
        key = str(case.get("id"))
        if not retry and key in self._brief_cache:
            return self._brief_cache[key]
        out = self.generate_prompt(
            _case_to_prompt(case), system=SYSTEM, num_predict=900,
            temperature=0.35, tag=f"brief/{case.get('id')}",
        )
        if not retry:
            self._brief_cache[key] = out
        return out

    def generate_prompt(
        self, prompt: str, *, system: str = SYSTEM, num_predict: int = 900,
        temperature: float = 0.35, timeout: float | None = None,
        tag: str = "",
    ) -> str:
        """Synchronous wrapper around the backend call for one arbitrary prompt."""
        started = time.monotonic()
        # Larger local models need much more than the settings default (180s).
        # Bump to 600s for the enrichment turns and any big-model brief run.
        or_timeout = timeout if timeout is not None else 120.0
        ollama_timeout = timeout if timeout is not None else 600.0
        try:
            if self.spec.provider == "openrouter":
                out = self._loop.run_until_complete(
                    self._or_call(prompt, system=system,
                                  max_tokens=num_predict,
                                  temperature=temperature,
                                  timeout=or_timeout)
                )
            else:
                out = self._loop.run_until_complete(
                    self._backend.generate_text(
                        prompt, system=system, num_predict=num_predict,
                        temperature=temperature, timeout=ollama_timeout,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.stats.failures += 1
            self.stats.wall_s += time.monotonic() - started
            self.stats.calls += 1
            print(f"    ! call failed ({self.spec.slug}, {tag}): {exc}",
                  file=sys.stderr)
            return ""
        self.stats.wall_s += time.monotonic() - started
        self.stats.calls += 1
        if not out:
            self.stats.failures += 1
            return ""
        return out

    async def _or_call(self, prompt: str, *, system: str = SYSTEM,
                       max_tokens: int = 900, temperature: float = 0.35,
                       timeout: float) -> str:
        # OpenAICompatChat.complete returns str|None and doesn't surface usage.
        # We reimplement the POST here so we can capture token usage and to add
        # a 429/5xx backoff since parallel sweeps against OpenRouter otherwise
        # trip rate limits within seconds.
        import httpx
        body = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not self.spec.reasoning:
            body["reasoning"] = {"enabled": False}
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://apex.local/l34n-benchmark",
            "X-Title": "L34N APE-711 benchmark",
        }
        backoffs = [2.0, 5.0, 10.0, 20.0]
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as c:
            for attempt, wait_s in enumerate([0.0] + backoffs):
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                r = await c.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body, headers=headers,
                )
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    # honour Retry-After if present, else fall through to next backoff
                    retry_after = r.headers.get("retry-after")
                    if retry_after:
                        try:
                            await asyncio.sleep(min(30.0, float(retry_after)))
                        except ValueError:
                            pass
                    if attempt < len(backoffs):
                        continue
                r.raise_for_status()
                data = r.json()
                break
            else:  # pragma: no cover — final attempt already raised
                return ""
        usage = data.get("usage") or {}
        self.stats.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.stats.output_tokens += int(usage.get("completion_tokens") or 0)
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        if not content and choices[0].get("finish_reason") == "length":
            # The whole budget went to hidden reasoning; surface it so the
            # run is not silently scored as a fallback.
            self.empty_length_hits += 1
            reasoning_tokens = ((usage.get("completion_tokens_details") or {})
                                .get("reasoning_tokens"))
            print(f"    ! empty content, finish_reason=length "
                  f"({self.spec.slug}; reasoning_tokens={reasoning_tokens}) — "
                  f"set `reasoning: false` or raise max_tokens", file=sys.stderr)
        return content


# ── Enrichment pass (full-fidelity Depth + Actionability) ─────────────────────

TURN_BY_SLUG = {t["slug"]: t for t in TURNS}


def _research_tier_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cases whose expected verdict includes research/adopt/spike.

    Per rubric: "each corpus item that reaches the research tier". We proxy
    that with the expected verdict — the cases the model *should* enrich.
    """
    out = []
    for c in cases:
        exp = c.get("expected") or {}
        verdicts = exp.get("verdict_in") or []
        if any(v in ("adopt", "research", "spike") for v in verdicts) \
                or exp.get("has_artifact"):
            out.append(c)
    return out


def _candidate_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Adapt a corpus case to the wiki._prompt candidate shape."""
    items_raw = case.get("items") or ([case["item"]] if case.get("item") else [])
    items = []
    for it in items_raw:
        items.append({
            "source_name": it.get("source") or it.get("kind") or "rss",
            "source_key": it.get("source") or "",
            "title": it.get("title") or "",
            "url": it.get("url") or "",
            "summary": (it.get("body") or "")[:400],
            "body": it.get("body") or "",
        })
    lead = items_raw[0] if items_raw else {}
    return {
        "title": lead.get("title") or case.get("id") or "Untitled",
        "category": (case.get("expected") or {}).get("category")
                    or lead.get("category") or "opinion-analysis",
        "source_count": len(items) or 1,
        "artifacts": [],
        "judgment": {"verdict": "watch", "readiness": 0.5},
        "items": items,
    }


def _enrichment_prompt(turn_slug: str, case: dict[str, Any],
                        pages: dict[str, str]) -> str:
    """Compose an enrichment-turn prompt using L34N's own wiki._prompt shape."""
    # We inline a minimal port of wiki._prompt so we don't depend on Database.
    turn = TURN_BY_SLUG[turn_slug]
    candidate = _candidate_from_case(case)
    prior = ""
    if pages:
        prior = (
            "## Wiki so far (untrusted model text)\n"
            + "\n\n".join(
                fence(f"PAGE_{slug.upper()}", body, limit=4000)
                for slug, body in pages.items()
            )
            + "\n\n"
        )
    return (
        fence("SUBJECT", candidate["title"], limit=160) + "\n"
        f"Category: {candidate['category']} · "
        f"sources: {candidate['source_count']} · "
        f"heuristic verdict: watch (readiness 0.50)\n"
        f"Artifacts already extracted: none\n\n"
        f"## Raw sources (immutable)\n"
        + "\n".join(
            f"{i}. [{it['source_name']}]\n"
            + fence("TITLE", it['title'], limit=140) + "\n"
            + fence("URL", it['url'], limit=200) + "\n"
            + fence("BODY", it['summary'], limit=280)
            for i, it in enumerate(candidate["items"], 1)
        )
        + "\n\n"
        + prior
        + turn["instruction"]
    )


@dataclass
class EnrichmentCaseResult:
    case_id: str
    # Blended judge scores (heuristic prior + candidate-model output, per
    # enrich/judge.py::blend). One row per corpus case; when a case has
    # multiple source items we average the blended scores.
    quality: float | None = None
    usefulness: float | None = None
    practicality: float | None = None
    feasibility: float | None = None
    readiness: float | None = None
    verdict: str = ""                    # blended verdict (skip|watch|research|adopt)
    heuristic_verdict: str = ""          # rule-only verdict for reference
    model_verdict: str = ""              # raw model verdict before blending
    verdict_expected: list = field(default_factory=list)
    verdict_matches: bool = False        # blended verdict ∈ expected.verdict_in
    judge_calls: int = 0                 # number of candidate-model judge calls
    judge_json_ok: int = 0               # successful JSON parses from model
    critique_ok: bool = False            # kept for backwards compatibility
    adapt_complete: bool = False
    adapt_word_count: int = 0
    error: str = ""


_JUDGE_JSON_RE = re.compile(r"\{[\s\S]*?\}")


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model reply. Tolerant of fenced
    code blocks and leading prose so a small model that wraps its JSON in
    markdown does not fail the whole judge pass."""
    if not raw:
        return None
    text = raw.strip()
    # Strip leading ```json fences the model often adds.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # First, try the whole thing.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, json.JSONDecodeError):
        pass
    # Fallback: scan for the first {…} block that parses.
    for match in _JUDGE_JSON_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def run_enrichment_pass(client: "ProviderClient",
                        cases: list[dict[str, Any]]) -> list[EnrichmentCaseResult]:
    """Full-featured Judge + Adapt pass.

    For each research-tier case we:

    1. Compute the heuristic ``judge_text`` prior for every source item.
    2. Call the candidate model with ``enrich.judge.SYSTEM`` +
       ``PROMPT`` + ``SCHEMA`` for that item (this is exactly what
       ``Judge._promote`` does in production).
    3. Blend the model JSON with the heuristic prior via
       ``enrich.judge.blend`` — same weights, same verdict brakes.
    4. Aggregate per-case (average scores, most-severe verdict for
       multi-source cases) and compare against ``expected.verdict_in``.
    5. Run the Adapt turn and record ``adapt_complete`` for the
       actionability metric.
    """
    results: list[EnrichmentCaseResult] = []
    for case in cases:
        r = EnrichmentCaseResult(case_id=case["id"])
        expected = case.get("expected") or {}
        r.verdict_expected = list(expected.get("verdict_in") or [])

        # Materialise source items — a case may hold `item` or `items`.
        items_raw = case.get("items") or ([case["item"]] if case.get("item") else [])
        if not items_raw:
            r.error = "no items on case"
            results.append(r)
            continue

        per_item_blended: list[dict[str, Any]] = []
        per_item_heuristic: list[dict[str, Any]] = []
        per_item_model_verdict: list[str] = []

        # Per-source Judge (heuristic + model + blend).
        for item in items_raw:
            title = item.get("title") or ""
            body = item.get("body") or ""
            source_name = item.get("source") or item.get("kind") or "rss"
            tier = item.get("tier") or "news"
            category = (item.get("category")
                        or (case.get("expected") or {}).get("category")
                        or "unknown")
            url = item.get("url") or ""

            heuristic = judge_text(
                title, body,
                category=category, tier=tier, url=url,
                importance=0.5, source_count=len(items_raw),
            )
            per_item_heuristic.append(heuristic)

            prompt = JUDGE_PROMPT.format(
                title=fence("TITLE", title, limit=250),
                body=fence("BODY", body, limit=900),
                source=fence("SOURCE", source_name, limit=80),
                tier=tier,
                category=category,
                q=heuristic["quality"], p=heuristic["practicality"],
                f=heuristic["feasibility"], u=heuristic["usefulness"],
                r=heuristic["readiness"],
            )
            raw = client.generate_prompt(
                prompt, system=JUDGE_SYSTEM, num_predict=220,
                temperature=0.2, tag=f"judge/{case['id']}",
            )
            r.judge_calls += 1
            model_payload = _parse_judge_json(raw)
            if model_payload:
                r.judge_json_ok += 1
                per_item_model_verdict.append(str(model_payload.get("verdict") or ""))
                blended = judge_blend(heuristic, model_payload)
            else:
                # Model failed to return usable JSON — fall back to heuristic.
                blended = dict(heuristic)
                per_item_model_verdict.append("")

            per_item_blended.append(blended)

        # Aggregate across items: mean scores, worst-case verdict resolution.
        # For multi-source cases we take the max-severity verdict (adopt >
        # research > watch > skip) so an "adopt" signal on one item is not
        # diluted; this matches how the pipeline surfaces the strongest lead.
        r.quality = _mean([b["quality"] for b in per_item_blended])
        r.usefulness = _mean([b["usefulness"] for b in per_item_blended])
        r.practicality = _mean([b["practicality"] for b in per_item_blended])
        r.feasibility = _mean([b["feasibility"] for b in per_item_blended])
        r.readiness = _mean([b["readiness"] for b in per_item_blended])

        verdicts = [b["verdict"] for b in per_item_blended if b.get("verdict")]
        if verdicts:
            # Highest-severity verdict wins.
            r.verdict = max(verdicts, key=lambda v: JUDGE_VERDICTS.index(v)
                             if v in JUDGE_VERDICTS else -1)
        r.heuristic_verdict = max(
            (h["verdict"] for h in per_item_heuristic),
            key=lambda v: JUDGE_VERDICTS.index(v) if v in JUDGE_VERDICTS else -1,
            default="",
        )
        model_verdicts = [v for v in per_item_model_verdict if v]
        if model_verdicts:
            r.model_verdict = max(
                model_verdicts,
                key=lambda v: JUDGE_VERDICTS.index(v) if v in JUDGE_VERDICTS else -1,
            )

        # verdict_matches only against cases that supply an allow-list.
        if r.verdict_expected:
            r.verdict_matches = r.verdict in r.verdict_expected

        # critique_ok kept for backwards compat with older reports — mark
        # True when we produced numeric scores for the case at all.
        r.critique_ok = r.quality is not None

        # Turn 2: Adapt — check adapt_complete for the actionability metric.
        adapt_prompt = _enrichment_prompt("adapt", case, {})
        adapt_md = client.generate_prompt(
            adapt_prompt, system=SCHEMA,
            num_predict=TURN_BY_SLUG["adapt"]["num_predict"],
            temperature=0.2, tag=f"adapt/{case['id']}",
        )
        if adapt_md:
            r.adapt_complete = adapt_complete(adapt_md)
            r.adapt_word_count = len(adapt_md.split())
        else:
            r.error = (r.error + "; " if r.error else "") + "adapt generation failed"
        results.append(r)
    return results


def summarise_enrichment(results: list[EnrichmentCaseResult]) -> dict[str, Any]:
    if not results:
        return {
            "cases": 0, "avg_quality": None, "avg_usefulness": None,
            "avg_practicality": None, "avg_feasibility": None,
            "avg_readiness": None,
            "adapt_complete_rate": 0.0, "critique_parse_rate": 0.0,
            "judge_json_parse_rate": 0.0,
            "judge_calls": 0,
            "verdict_agreement": None,
            "verdict_agreement_cases": 0,
        }
    scored = [r for r in results if r.quality is not None]
    judge_calls_total = sum(r.judge_calls for r in results)
    judge_ok_total = sum(r.judge_json_ok for r in results)

    # verdict_agreement — fraction of cases with an expected.verdict_in
    # whose blended verdict landed in that allow-list.
    verdict_cases = [r for r in results if r.verdict_expected and r.verdict]
    if verdict_cases:
        agreement = round(
            sum(1 for r in verdict_cases if r.verdict_matches) / len(verdict_cases), 4
        )
    else:
        agreement = None
    return {
        "cases": len(results),
        "avg_quality": (
            round(statistics.mean(r.quality for r in scored), 4) if scored else None
        ),
        "avg_usefulness": (
            round(statistics.mean(r.usefulness for r in scored), 4) if scored else None
        ),
        "avg_practicality": (
            round(statistics.mean(r.practicality for r in scored), 4) if scored else None
        ),
        "avg_feasibility": (
            round(statistics.mean(r.feasibility for r in scored), 4) if scored else None
        ),
        "avg_readiness": (
            round(statistics.mean(r.readiness for r in scored), 4) if scored else None
        ),
        "adapt_complete_rate": round(
            sum(1 for r in results if r.adapt_complete) / len(results), 4
        ),
        "critique_parse_rate": round(
            sum(1 for r in results if r.critique_ok) / len(results), 4
        ),
        "judge_calls": judge_calls_total,
        "judge_json_parse_rate": (
            round(judge_ok_total / judge_calls_total, 4) if judge_calls_total else 0.0
        ),
        "verdict_agreement": agreement,
        "verdict_agreement_cases": len(verdict_cases),
    }


# ── Rubric composite scoring (APE-710) ─────────────────────────────────────────

@dataclass
class RubricScores:
    relevance: float = 0.0
    accuracy: float = 0.0
    depth: float = 0.0
    actionability: float = 0.0
    cost: float = 0.0
    speed: float = 0.0
    composite: float = 0.0
    # Weighted sum before the soft deduction (what rubric v1.0 reported).
    composite_raw: float = 0.0
    # Rubric v1.1 (APE-724): hallucinated_recommendation_rate × 20 points.
    hallucination_deduction: float = 0.0
    disqualified: bool = False
    disqualifiers: list[str] = field(default_factory=list)


RUBRIC_VERSION = "1.1"

WEIGHTS = {
    "relevance": 0.25, "accuracy": 0.30, "depth": 0.20,
    "actionability": 0.15, "cost": 0.05, "speed": 0.05,
}

# APE-724: since harness validate-v2 an ungated Ready section is dropped before
# the brief ships, so writing one is a behaviour-quality signal rather than a
# safety risk. It is scored proportionally instead of disqualifying the model.
HALLUCINATION_DEDUCTION_PTS = 20.0


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) <= 0 else 2 * p * r / (p + r)


def rubric_score(
    metrics: dict[str, Any],
    *,
    wall_s: float,
    cost_usd: float,
    baseline_qpd: float | None = None,
    enrichment: dict[str, Any] | None = None,
) -> RubricScores:
    """Compute the APE-710 v1.1 composite from a `summarise()` metrics dict.

    When ``enrichment`` is provided (from ``summarise_enrichment``), full-fidelity
    Depth and Actionability are computed per rubric §3 and §4. Otherwise falls
    back to the proxy scoring documented in the report.
    """
    rs = RubricScores()

    # 1. Relevance & Recall
    p = metrics.get("ai_relevance_precision", 0.0)
    r = metrics.get("ai_relevance_recall", 0.0)
    rs.relevance = _f1(p, r) * 100

    # 2. Accuracy — composite per rubric
    factuality = metrics.get("factuality_score", 0.0)
    citation = metrics.get("citation_completeness", 0.0)
    hallucinated = metrics.get("hallucinated_recommendation_rate", 0.0)
    injection = metrics.get("injection_following_rate", 0.0)
    rs.accuracy = (
        factuality * 0.40
        + citation * 0.25
        + (1 - hallucinated) * 0.25
        + (1 - injection) * 0.10
    ) * 100

    # 3. Depth — full-fidelity when enrichment scores are present, else proxy.
    format_ok = metrics.get("format_compliance", 0.0)
    prompt_echo = metrics.get("prompt_echo_rate", 0.0)
    if enrichment and enrichment.get("cases", 0) > 0 \
            and enrichment.get("avg_quality") is not None:
        q = enrichment["avg_quality"]
        u = enrichment["avg_usefulness"] or 0.0
        adapt_rate = enrichment.get("adapt_complete_rate", 0.0)
        rs.depth = (
            q * 0.40
            + u * 0.35
            + adapt_rate * 0.15
            + (1 - prompt_echo) * 0.10
        ) * 100
    else:
        # Proxy: format-compliance + citation + non-echo
        rs.depth = (
            format_ok * 0.55
            + citation * 0.30
            + (1 - prompt_echo) * 0.15
        ) * 100

    # 4. Actionability — prefer the *blended* judge verdict from the
    # candidate model (per rubric §4). Falls back to the harness's
    # heuristic-only readiness_agreement when full-fidelity is off.
    heuristic_readiness = metrics.get("readiness_agreement", 0.0)
    if enrichment and enrichment.get("cases", 0) > 0:
        blended_verdict_agreement = enrichment.get("verdict_agreement")
        readiness = (
            blended_verdict_agreement
            if blended_verdict_agreement is not None
            else heuristic_readiness
        )
        adapt_rate = enrichment.get("adapt_complete_rate", 0.0)
        rs.actionability = (readiness * 0.60 + adapt_rate * 0.40) * 100
    else:
        rs.actionability = heuristic_readiness * 100  # proxy

    # 5. Cost — normalized against `baseline_qpd` (best qpd across sweep)
    qpd = 0.0 if cost_usd <= 0 else (rs.accuracy + rs.relevance) / cost_usd
    if cost_usd <= 0:
        rs.cost = 100.0  # zero-cost (local/free) tier
    elif baseline_qpd and baseline_qpd > 0 and qpd > 0:
        ratio = qpd / baseline_qpd
        rs.cost = max(0.0, min(100.0, ratio * 100.0))
    else:
        rs.cost = 50.0  # unknown baseline; neutral

    # 6. Speed — 100 at ≤100s, 0 at ≥600s (linear)
    rs.speed = max(0.0, min(100.0, (600.0 - wall_s) / 5.0))

    # Composite
    rs.composite_raw = (
        rs.relevance * WEIGHTS["relevance"]
        + rs.accuracy * WEIGHTS["accuracy"]
        + rs.depth * WEIGHTS["depth"]
        + rs.actionability * WEIGHTS["actionability"]
        + rs.cost * WEIGHTS["cost"]
        + rs.speed * WEIGHTS["speed"]
    )

    # Soft deduction (rubric v1.1): proportional, no cliff edge.
    rs.hallucination_deduction = round(hallucinated * HALLUCINATION_DEDUCTION_PTS, 2)
    rs.composite = max(0.0, rs.composite_raw - rs.hallucination_deduction)

    # Hard disqualifiers
    if injection > 0.05:
        rs.disqualifiers.append(f"injection_following_rate={injection:.2f}>0.05")
    if metrics.get("fallback_rate", 0.0) > 0.40:
        rs.disqualifiers.append(f"fallback_rate={metrics['fallback_rate']:.2f}>0.40")
    if format_ok < 0.50:
        rs.disqualifiers.append(f"format_compliance={format_ok:.2f}<0.50")
    rs.disqualified = bool(rs.disqualifiers)

    return rs


# ── Runner ─────────────────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _estimate_cost(spec: ModelSpec, stats: CallStats) -> float:
    if spec.input_per_m is None or spec.output_per_m is None:
        return 0.0
    return (
        stats.input_tokens / 1_000_000 * spec.input_per_m
        + stats.output_tokens / 1_000_000 * spec.output_per_m
    )


def run_one_model(spec: ModelSpec, settings: Settings, *,
                  case_ids: list[str] | None = None,
                  layers: tuple[str, ...] = ("schema", "fallback"),
                  full_fidelity: bool = True) -> dict[str, Any]:
    print(f"→ {spec.slug} ({spec.provider} / {spec.model}, {spec.tier})"
          f"{' [full-fidelity]' if full_fidelity else ''}")
    client = ProviderClient(spec, settings)
    started = time.monotonic()
    enrichment_results: list[EnrichmentCaseResult] = []
    try:
        result = run_corpus(
            generate=lambda case, layer=None, **kw: client.generate(case, layer=layer or "schema"),
            layers=layers, case_ids=case_ids,
        )
        if full_fidelity:
            cases = [c for c in CASES if not case_ids or c["id"] in set(case_ids)]
            research_cases = _research_tier_cases(cases)
            if research_cases:
                print(f"  enrichment pass: {len(research_cases)} research-tier case(s)")
                enrichment_results = run_enrichment_pass(client, research_cases)
    finally:
        wall_s = time.monotonic() - started
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    stats = client.stats
    cost_usd = _estimate_cost(spec, stats)

    # Score using primary layer (schema) per rubric
    primary_layer = result["layers"].get("schema") or next(iter(result["layers"].values()))
    metrics = primary_layer["metrics"]

    enrichment_summary = summarise_enrichment(enrichment_results) if enrichment_results else None

    rs = rubric_score(metrics, wall_s=wall_s, cost_usd=cost_usd,
                      enrichment=enrichment_summary)

    doc = {
        "model_slug": spec.slug,
        "model_id": spec.model,
        "provider": spec.provider,
        "tier": spec.tier,
        "backend": "local_ollama" if spec.provider == "ollama" else "openrouter",
        "full_fidelity": full_fidelity and bool(enrichment_results),
        "corpus_version": result["corpus_version"],
        "prompt_version": result["prompt_version"],
        "harness_version": result["harness_version"],
        "app_version": result["app_version"],
        "layer_primary": "schema",
        "cases": len(primary_layer["cases"]),
        "wall_clock_s": round(wall_s, 2),
        "call_stats": asdict(stats),
        "call_failure_rate": round(stats.failures / stats.calls, 4) if stats.calls else None,
        "empty_length_hits": client.empty_length_hits,
        "cost_usd_estimate": round(cost_usd, 6),
        "metrics_by_layer": {
            layer: entry["metrics"] for layer, entry in result["layers"].items()
        },
        "case_details": primary_layer["cases"],
        "enrichment": enrichment_summary,
        "enrichment_details": [asdict(r) for r in enrichment_results] if enrichment_results else [],
        "rubric": asdict(rs),
        "rubric_version": RUBRIC_VERSION,
        "notes": spec.notes,
    }
    return doc


def sweep(specs: list[ModelSpec], settings: Settings, out_dir: Path,
          *, case_ids: list[str] | None = None,
          full_fidelity: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = []
    for spec in specs:
        try:
            doc = run_one_model(spec, settings, case_ids=case_ids,
                                 full_fidelity=full_fidelity)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {spec.slug} failed: {exc}", file=sys.stderr)
            doc = {
                "model_slug": spec.slug, "model_id": spec.model,
                "provider": spec.provider, "tier": spec.tier,
                "failed": True, "error": str(exc)[:400],
            }
        docs.append(doc)
        # Write per-model file eagerly so a crash mid-sweep still leaves data.
        (out_dir / f"{spec.slug}.json").write_text(
            json.dumps(doc, indent=2, default=str), encoding="utf-8",
        )
        print(f"  ✓ wrote {spec.slug}.json")

    return rescore(docs, out_dir)


def rescore(docs: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Recompute every rubric from the stored metrics and rewrite the files.

    Second pass of a sweep, and the whole of ``--rescore``: rubric §5 Cost is
    normalised against the best qpd ("quality per dollar") observed among
    *viable* paid models, so every row must be scored once to find the
    baseline and once more against it. No model calls are made.
    """
    def score(doc: dict[str, Any], baseline_qpd: float | None) -> None:
        metrics = (doc["metrics_by_layer"].get("schema")
                   or next(iter(doc["metrics_by_layer"].values())))
        rs = rubric_score(metrics, wall_s=doc["wall_clock_s"],
                          cost_usd=doc["cost_usd_estimate"],
                          baseline_qpd=baseline_qpd,
                          enrichment=doc.get("enrichment"))
        doc["rubric"] = asdict(rs)
        doc["rubric_version"] = RUBRIC_VERSION

    live = [doc for doc in docs if not doc.get("failed")]
    for doc in live:
        score(doc, None)
    baseline_qpd = _best_viable_qpd(docs)
    if baseline_qpd and baseline_qpd > 0:
        for doc in live:
            score(doc, baseline_qpd)
    for doc in live:
        (out_dir / f"{doc['model_slug']}.json").write_text(
            json.dumps(doc, indent=2, default=str), encoding="utf-8",
        )

    summary_entries = [_summary_row(doc) for doc in docs]
    index = {
        "corpus_version": CORPUS_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_qpd": baseline_qpd,
        "models": summary_entries,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def load_results(out_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def _best_viable_qpd(docs: list[dict[str, Any]]) -> float | None:
    best = 0.0
    for doc in docs:
        if doc.get("failed"):
            continue
        r = doc.get("rubric") or {}
        cost = doc.get("cost_usd_estimate", 0.0)
        if r.get("disqualified") or cost <= 0:
            continue
        qpd = (r.get("accuracy", 0.0) + r.get("relevance", 0.0)) / cost
        if qpd > best:
            best = qpd
    return best or None


def _summary_row(doc: dict[str, Any]) -> dict[str, Any]:
    if doc.get("failed"):
        return {
            "slug": doc["model_slug"], "model": doc["model_id"],
            "tier": doc["tier"], "provider": doc["provider"],
            "failed": True, "error": doc.get("error"),
        }
    r = doc["rubric"]
    enrichment = doc.get("enrichment") or {}
    return {
        "slug": doc["model_slug"], "model": doc["model_id"],
        "tier": doc["tier"], "provider": doc["provider"],
        "composite": round(r["composite"], 2),
        "composite_raw": round(r.get("composite_raw", r["composite"]), 2),
        "hallucination_deduction": round(r.get("hallucination_deduction", 0.0), 2),
        "relevance": round(r["relevance"], 2),
        "accuracy": round(r["accuracy"], 2),
        "depth": round(r["depth"], 2),
        "actionability": round(r["actionability"], 2),
        "cost_score": round(r["cost"], 2),
        "speed_score": round(r["speed"], 2),
        "wall_s": doc["wall_clock_s"],
        "cost_usd": doc["cost_usd_estimate"],
        "disqualified": r["disqualified"],
        "disqualifiers": r["disqualifiers"],
        "cases": doc["cases"],
        "full_fidelity": doc.get("full_fidelity", False),
        "judge_calls": enrichment.get("judge_calls", 0),
        "judge_json_parse_rate": enrichment.get("judge_json_parse_rate", 0.0),
        "verdict_agreement": enrichment.get("verdict_agreement"),
        "adapt_complete_rate": enrichment.get("adapt_complete_rate", 0.0),
        "avg_judge_quality": enrichment.get("avg_quality"),
        "avg_judge_usefulness": enrichment.get("avg_usefulness"),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="APE-711 L34N model benchmark sweep.")
    ap.add_argument("--profile", choices=["free", "local", "paid", "all"],
                    default="free", help="which tier to run")
    ap.add_argument("--model", action="append", default=None,
                    help="run only this model slug (repeatable). Overrides --profile.")
    ap.add_argument("--cases", default=None,
                    help="comma-sep list of case ids to run (default: all)")
    ap.add_argument("--out", default=str(L34N_ROOT / "data" / "benchmark-results"),
                    help="output directory")
    ap.add_argument("--brief-only", action="store_true",
                    help="skip the enrichment pass (Critique + Adapt turns); "
                         "Depth/Actionability fall back to proxy scoring")
    ap.add_argument("--matrix", default=None,
                    help="path to a YAML matrix file "
                         f"(default: {DEFAULT_MATRIX_PATH})")
    ap.add_argument("--list", action="store_true", help="list matrix and exit")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the rubric for the JSON already in --out "
                         "(no model calls) and rewrite files + index")
    args = ap.parse_args(argv)

    if args.rescore:
        out_dir = Path(args.out)
        docs = load_results(out_dir)
        if not docs:
            print(f"no results in {out_dir}", file=sys.stderr)
            return 2
        index = rescore(docs, out_dir)
        for row in index["models"]:
            if row.get("failed"):
                continue
            print(f"  {row['slug']:35s} composite {row['composite_raw']:6.2f} "
                  f"- {row['hallucination_deduction']:5.2f} = {row['composite']:6.2f}"
                  f"{'  DQ: ' + '; '.join(row['disqualifiers']) if row['disqualified'] else ''}")
        print(f"rescored {len(docs)} model(s) under rubric v{RUBRIC_VERSION} → {out_dir}")
        return 0

    global MODEL_MATRIX
    MODEL_MATRIX = load_matrix(Path(args.matrix) if args.matrix else None)

    if args.list:
        for s in MODEL_MATRIX:
            price = f"${s.input_per_m}/M in, ${s.output_per_m}/M out" \
                if s.input_per_m is not None else "free"
            print(f"  {s.slug:35s} {s.tier:5s} {s.provider:10s} {s.model:60s} {price}")
        return 0

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

    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None

    print(f"Sweeping {len(specs)} model(s), corpus v{CORPUS_VERSION}, "
          f"cases={len(case_ids) if case_ids else len(CASES)}, "
          f"mode={'brief-only' if args.brief_only else 'full-fidelity'}")
    sweep(specs, settings, Path(args.out), case_ids=case_ids,
          full_fidelity=not args.brief_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
