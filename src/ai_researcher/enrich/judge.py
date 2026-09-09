"""Quality, practicality, feasibility, and usefulness.

The dashboard already ranks *attention*: corroboration, engagement, recency.
That is the wrong axis for "should I try this". A viral thread about a closed
API and a quiet arXiv paper with a reproducible repo score the same on
attention and opposite on adoptability.

This module scores every item on four practitioner questions:

* **quality** — is the signal specific and evidenced, or noise?
* **practicality** — can someone actually use the thing (code, weights, API)?
* **feasibility** — with realistic compute, licenses, and skills?
* **usefulness** — would adopting it change what you build or run?

`readiness` is the composite that gates Karpathy-style deep research. The
thresholds are priors, not measurements — they exist so a run spends its
scarce model budget on a handful of stories rather than on everything.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from ..config import Settings
from ..db import Database, jdump, jload
from ..progress import RunProgress
from ..sanitize import UNTRUSTED_RULE, fence, sanitize_artifacts
from ..util import iso, utcnow
from .ollama import OllamaClient

log = logging.getLogger("ai_researcher.judge")

VERDICTS = ("skip", "watch", "research", "adopt")

# Composite weights: quality and usefulness dominate because a beautiful but
# empty paper, or a trivial but well-packaged tool, should not consume a
# research slot. Practicality and feasibility are the brakes.
WEIGHTS = {
    "quality": 0.28,
    "practicality": 0.22,
    "feasibility": 0.22,
    "usefulness": 0.28,
}

# Priors. "Adopt" should be rare — things you would actually try this week.
# "Research" is the deep-research gate. "Watch" stays on the firehose.
ADOPT_READINESS = 0.78
RESEARCH_READINESS = 0.62
WATCH_READINESS = 0.42
ADOPT_PRACTICALITY = 0.58
ADOPT_FEASIBILITY = 0.52

_ARTIFACT = re.compile(
    r"github\.com|huggingface\.co|\bhf\.co\b|arxiv\.org|gitlab\.com|"
    r"open[- ]?weights?|model card|checkpoint|weights are|gguf|"
    r"pip install|uv add|npm i(?:nstall)?|pypi\.org",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"\b(code|repo(?:sitory)?|sdk|library|framework|cli|api|notebook|"
    r"colab|docker|reproducible|implementation|source available)\b",
    re.IGNORECASE,
)
_CLOSED = re.compile(
    r"\b(closed[- ]?source|api[- ]?only|waitlist|not (?:yet )?available|"
    r"proprietary|internal only|no (?:code|weights)|weights? not released)\b",
    re.IGNORECASE,
)
_HEAVY = re.compile(
    r"\b(\d{2,3}\s*b(?:illion)?\s*(?:param|parameter)|405b|1t parameter|"
    r"cluster of|thousands of (?:h100|h200|b200)|multi[- ]billion)\b",
    re.IGNORECASE,
)
_LOCAL = re.compile(
    r"\b(local|self[- ]?host|on[- ]?prem|consumer gpu|rtx|mlx|"
    r"ollama|llama\.cpp|gguf|4-?bit|q4|q5|q8|quantiz\w+)\b",
    re.IGNORECASE,
)
_USEFUL = re.compile(
    r"\b(sota|state[- ]of[- ]the[- ]art|beats?|outperform\w*|faster|cheaper|"
    r"open alternative|reproducible|drops? (?:cost|latency)|usable today)\b",
    re.IGNORECASE,
)
_WEAK = re.compile(
    r"\b(how to|tutorial|beginner|listicle|weekly digest|roundup|"
    r"what do you think|hot take|unpopular opinion|rant|meme)\b",
    re.IGNORECASE,
)
_SPECIFIC = re.compile(
    r"\b(v?\d+\.\d+|arxiv:\d{4}\.\d{4,5}|https?://|"
    r"gpt-|claude |llama\s?-?\d|qwen\s?-?\d|deepseek)\b",
    re.IGNORECASE,
)
_GITHUB = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+", re.IGNORECASE)
_HF = re.compile(r"https?://(?:www\.)?huggingface\.co/[\w.-]+/[\w.-]+", re.IGNORECASE)
_ARXIV = re.compile(r"(?:arxiv\.org/abs/|arxiv:)\s*(\d{4}\.\d{4,5})", re.IGNORECASE)

SYSTEM = (
    "You judge AI discoveries for a practitioner who will implement things, "
    "not collect bookmarks. You never invent an artifact, license, or "
    "benchmark the text does not state. Reply with JSON only. "
    + UNTRUSTED_RULE
)

SCHEMA = {
    "type": "object",
    "properties": {
        "quality": {"type": "number"},
        "practicality": {"type": "number"},
        "feasibility": {"type": "number"},
        "usefulness": {"type": "number"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["quality", "practicality", "feasibility", "usefulness"],
}

PROMPT = """{title}

{body}

(source: {source} · tier: {tier} · category: {category})
heuristic prior: Q={q:.2f} P={p:.2f} F={f:.2f} U={u:.2f} readiness={r:.2f}

Score 0.0-1.0 using these anchors. 0.8+ needs concrete evidence in the text.
Calibration: open weights or a pip-installable repo named in the text = practicality >= 0.8. Permissive license (Apache/MIT) and consumer-GPU size = feasibility >= 0.8. Funding, policy, opinion, or a digest with nothing to run = practicality <= 0.2.
Verdict rule: adopt = you would install it this week; research = worth an afternoon spike; watch = interesting, nothing to run yet; skip = noise.
"quality": specific, evidenced, primary — not a recap or listicle.
"practicality": a practitioner can touch it (code, weights, API, dataset).
"feasibility": realistic compute, license, skill. Closed 405B is low.
"usefulness": would change what they build or run this month.
"verdict": skip | watch | research | adopt
"reasons": up to 4 short facts from the text, not slogans.
"artifacts": URLs or names of code, weights, papers, APIs actually named."""


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def readiness_of(quality: float, practicality: float, feasibility: float, usefulness: float) -> float:
    return round(
        WEIGHTS["quality"] * quality
        + WEIGHTS["practicality"] * practicality
        + WEIGHTS["feasibility"] * feasibility
        + WEIGHTS["usefulness"] * usefulness,
        4,
    )


def verdict_of(
    readiness: float, *, practicality: float = 0.0, feasibility: float = 0.0
) -> str:
    if (
        readiness >= ADOPT_READINESS
        and practicality >= ADOPT_PRACTICALITY
        and feasibility >= ADOPT_FEASIBILITY
    ):
        return "adopt"
    if readiness >= RESEARCH_READINESS:
        return "research"
    if readiness >= WATCH_READINESS:
        return "watch"
    return "skip"


def extract_artifacts(title: str, body: str, url: str = "") -> list[str]:
    """Named, concrete things a practitioner could fetch. Deduped, short list."""
    haystack = f"{title}\n{body}\n{url}"
    found: list[str] = []
    for pattern in (_GITHUB, _HF):
        found.extend(m.group(0).rstrip(").,;") for m in pattern.finditer(haystack))
    for match in _ARXIV.finditer(haystack):
        found.append(f"arxiv:{match.group(1)}")
    if url and any(host in url for host in ("github.com", "huggingface.co", "arxiv.org")):
        found.append(url)
    # Stable order, drop fragments that are just the item's own URL twice.
    return sanitize_artifacts(list(dict.fromkeys(found)))[:6]


def judge_text(
    title: str,
    body: str,
    *,
    category: str = "",
    tier: str = "news",
    url: str = "",
    importance: float = 0.5,
    source_count: int = 1,
) -> dict[str, Any]:
    """Rule-based judgment. Always available; the model later blends, not replaces."""
    haystack = f"{title}\n{body[:2000]}"
    artifacts = extract_artifacts(title, body, url)
    has_artifact = bool(artifacts) or bool(_ARTIFACT.search(haystack))
    has_code = bool(_CODE.search(haystack))
    is_closed = bool(_CLOSED.search(haystack))
    is_heavy = bool(_HEAVY.search(haystack))
    is_local = bool(_LOCAL.search(haystack))
    is_useful = bool(_USEFUL.search(haystack))
    is_weak = bool(_WEAK.search(title)) or bool(_WEAK.search(haystack[:400]))
    is_specific = bool(_SPECIFIC.search(haystack))

    quality = 0.38
    quality += {"lab": 0.18, "vendor": 0.10, "research": 0.14, "analyst": 0.06}.get(tier, 0.0)
    quality += 0.10 if is_specific else 0.0
    quality += 0.08 if len(body) > 500 else 0.0
    quality += 0.06 if has_artifact else 0.0
    quality -= 0.22 if is_weak else 0.0
    quality += {
        "model-release": 0.08,
        "research": 0.06,
        "tooling-oss": 0.06,
        "opinion-analysis": -0.10,
        "funding-acquisition": -0.04,
    }.get(category, 0.0)
    # Independent outlets agreeing is evidence, not just attention.
    quality += min(0.10, 0.04 * max(0, source_count - 1))

    practicality = 0.28
    practicality += 0.22 if has_artifact else 0.0
    practicality += 0.14 if has_code else 0.0
    practicality += {
        "tooling-oss": 0.18,
        "product-launch": 0.12,
        "model-release": 0.10,
        "research": 0.06,
        "opinion-analysis": -0.12,
        "funding-acquisition": -0.14,
        "policy-regulation": -0.10,
    }.get(category, 0.0)
    practicality -= 0.18 if is_closed else 0.0
    practicality += 0.06 if is_local else 0.0

    feasibility = 0.40
    feasibility += 0.16 if is_local else 0.0
    feasibility += 0.10 if has_artifact and not is_closed else 0.0
    feasibility -= 0.22 if is_closed else 0.0
    feasibility -= 0.16 if is_heavy and not is_local else 0.0
    feasibility += {"lab": 0.04, "research": 0.04, "vendor": 0.02}.get(tier, 0.0)
    # A paper with no artifact is interesting and hard to adopt.
    if category == "research" and not has_artifact:
        feasibility -= 0.08
        practicality -= 0.06

    usefulness = 0.32 + 0.28 * clamp(importance)
    usefulness += 0.12 if is_useful else 0.0
    usefulness -= 0.16 if is_weak else 0.0
    usefulness += {
        "model-release": 0.10,
        "tooling-oss": 0.08,
        "research": 0.06,
        "product-launch": 0.06,
        "opinion-analysis": -0.10,
        "funding-acquisition": -0.08,
    }.get(category, 0.0)
    usefulness += min(0.08, 0.03 * max(0, source_count - 1))

    quality = clamp(quality)
    practicality = clamp(practicality)
    feasibility = clamp(feasibility)
    usefulness = clamp(usefulness)
    readiness = readiness_of(quality, practicality, feasibility, usefulness)
    verdict = verdict_of(readiness, practicality=practicality, feasibility=feasibility)

    reasons: list[str] = []
    if has_artifact:
        reasons.append("names a fetchable artifact")
    if is_local:
        reasons.append("looks runnable on commodity hardware")
    if is_closed:
        reasons.append("access looks closed or waitlisted")
    if is_heavy and not is_local:
        reasons.append("compute footprint looks frontier-scale")
    if is_weak:
        reasons.append("reads as commentary or how-to, not a discovery")
    if source_count >= 3:
        reasons.append(f"corroborated by {source_count} sources")
    if not reasons:
        reasons.append("scored from source tier, category, and specificity")

    return {
        "quality": round(quality, 4),
        "practicality": round(practicality, 4),
        "feasibility": round(feasibility, 4),
        "usefulness": round(usefulness, 4),
        "readiness": readiness,
        "verdict": verdict,
        "reasons": reasons[:4],
        "artifacts": artifacts,
    }


def blend(heuristic: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Small models cluster near 0.7; keep the rules as a prior."""
    out = dict(heuristic)
    for key in ("quality", "practicality", "feasibility", "usefulness"):
        try:
            raw = clamp(float(model.get(key, heuristic[key])))
        except (TypeError, ValueError):
            raw = heuristic[key]
        out[key] = round(0.55 * raw + 0.45 * heuristic[key], 4)
    out["readiness"] = readiness_of(
        out["quality"], out["practicality"], out["feasibility"], out["usefulness"]
    )
    llm_verdict = str(model.get("verdict") or "").strip()
    computed = verdict_of(
        out["readiness"],
        practicality=out["practicality"],
        feasibility=out["feasibility"],
    )
    # The blended scores decide the band. The model's verdict word may lift
    # the band by one step, never lower it: small local models (llama3.1:8b,
    # qwen3:32b, gemma3:27b) score an open-weights release at readiness
    # 0.70 and still write "watch", and honouring the word over the numbers
    # is why every local judge in the APE-703 sweep scored 0/3 on verdicts.
    # The step is still measured from the *heuristic* band as well, not
    # from the blended scores: blending 0.99 into a skip-band prior can
    # land in research, and allowing a further step to adopt is how a
    # cheerful 7B used to mint "adopt" on a weekly digest.
    prior_verdict = heuristic.get("verdict") if heuristic.get("verdict") in VERDICTS else computed
    out["verdict"] = computed
    if (
        llm_verdict in VERDICTS
        and VERDICTS.index(llm_verdict) == VERDICTS.index(computed) + 1
        and abs(VERDICTS.index(llm_verdict) - VERDICTS.index(prior_verdict)) <= 1
    ):
        out["verdict"] = llm_verdict
    # Re-apply the adopt brakes after blending.
    if out["verdict"] == "adopt" and (
        out["practicality"] < ADOPT_PRACTICALITY or out["feasibility"] < ADOPT_FEASIBILITY
    ):
        out["verdict"] = "research"

    reasons = [str(r).strip() for r in (model.get("reasons") or []) if str(r).strip()]
    out["reasons"] = (reasons + heuristic["reasons"])[:5]
    artifacts = [str(a).strip() for a in (model.get("artifacts") or []) if str(a).strip()]
    out["artifacts"] = sanitize_artifacts(artifacts + heuristic["artifacts"])
    return out


class Judge:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        client: OllamaClient,
        progress: RunProgress | None = None,
    ):
        self.settings = settings
        self.db = db
        self.client = client
        self.progress = progress
        self._gate = asyncio.Semaphore(2)

    async def run(self, limit: int | None = None) -> dict[str, Any]:
        if self.progress:
            self.progress.update(
                stage="judge",
                detail="Scoring quality, practicality, feasibility, usefulness",
                current="",
                done=0,
                total=0,
                active=[],
            )
        heuristic_count = self._heuristic_pass()

        promoted = 0
        elapsed = 0.0
        if await self.client.probe():
            promoted, elapsed = await self._model_pass(
                limit if limit is not None else self.settings.judge_budget
            )

        pending = self.db.scalar(
            "SELECT COUNT(*) FROM judgments WHERE model=''", default=0
        )
        return {
            "judged": heuristic_count,
            "llm": promoted,
            "heuristic": heuristic_count,
            "awaiting_model": pending,
            "adopt": self.db.scalar(
                "SELECT COUNT(*) FROM judgments WHERE verdict='adopt'", default=0
            ),
            "research": self.db.scalar(
                "SELECT COUNT(*) FROM judgments WHERE verdict='research'", default=0
            ),
            "model": self.client.chat_model if self.client.available else "",
            "model_seconds": round(elapsed, 1),
        }

    def _heuristic_pass(self) -> int:
        rows = self.db.query(
            """
            SELECT i.id, i.title, i.body, i.url,
                   COALESCE(e.category, '') AS category,
                   COALESCE(e.importance, 0.5) AS importance,
                   COALESCE(s.tier, 'news') AS tier,
                   COALESCE((
                       SELECT c.source_count
                       FROM cluster_items ci
                       JOIN clusters c ON c.id = ci.cluster_id
                       WHERE ci.item_id = i.id
                       ORDER BY c.day DESC LIMIT 1
                   ), 1) AS source_count
            FROM items i
            LEFT JOIN enrichment e ON e.item_id = i.id
            LEFT JOIN sources s ON s.key = i.source_key
            WHERE i.id NOT IN (SELECT item_id FROM judgments)
            ORDER BY COALESCE(i.published_at, i.fetched_at) DESC
            """
        )
        if not rows:
            return 0

        now = iso(utcnow())
        payload = []
        for row in rows:
            judged = judge_text(
                row["title"] or "",
                row["body"] or "",
                category=row["category"] or "",
                tier=row["tier"] or "news",
                url=row["url"] or "",
                importance=float(row["importance"] or 0.5),
                source_count=int(row["source_count"] or 1),
            )
            payload.append(
                (
                    row["id"],
                    judged["quality"],
                    judged["practicality"],
                    judged["feasibility"],
                    judged["usefulness"],
                    judged["readiness"],
                    judged["verdict"],
                    jdump(judged["reasons"]),
                    jdump(judged["artifacts"]),
                    "",
                    now,
                )
            )

        with self.db.tx() as conn:
            conn.executemany(
                """
                INSERT INTO judgments (
                    item_id, quality, practicality, feasibility, usefulness,
                    readiness, verdict, reasons, artifacts, model, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(item_id) DO NOTHING
                """,
                payload,
            )
        log.info("heuristic judgment scored %s items", len(payload))
        return len(payload)

    async def _model_pass(self, budget: int) -> tuple[int, float]:
        if budget <= 0:
            return 0, 0.0

        rows = self.db.query(
            """
            SELECT i.id, i.title, i.body, i.url,
                   COALESCE(e.category, '') AS category,
                   COALESCE(e.importance, 0.5) AS importance,
                   COALESCE(s.tier, 'news') AS tier,
                   COALESCE(s.name, i.source_key) AS source_name,
                   j.quality, j.practicality, j.feasibility, j.usefulness,
                   j.readiness, j.verdict, j.reasons, j.artifacts,
                   COALESCE((
                       SELECT c.source_count
                       FROM cluster_items ci
                       JOIN clusters c ON c.id = ci.cluster_id
                       WHERE ci.item_id = i.id
                       ORDER BY c.day DESC LIMIT 1
                   ), 1) AS source_count
            FROM judgments j
            JOIN items i ON i.id = j.item_id
            LEFT JOIN enrichment e ON e.item_id = i.id
            LEFT JOIN sources s ON s.key = i.source_key
            WHERE j.model = ''
            ORDER BY j.readiness DESC, COALESCE(i.published_at, i.fetched_at) DESC
            LIMIT ?
            """,
            (budget,),
        )
        if not rows:
            return 0, 0.0

        time_budget = float(self.settings.judge_time_budget)
        started = time.monotonic()
        promoted = 0
        total = len(rows)
        if self.progress:
            self.progress.update(
                stage="judge",
                detail=f"Model judgment · 0/{total}",
                current="",
                done=0,
                total=total,
                active=[],
            )

        queue = asyncio.Queue()
        for row in rows:
            queue.put_nowait(row)

        async def worker():
            nonlocal promoted
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if time.monotonic() - started > time_budget:
                    return
                title = (row["title"] or "")[:80]
                if self.progress:
                    self.progress.update(
                        stage="judge",
                        detail=f"Model judgment · {promoted}/{total}",
                        current=title,
                        done=promoted,
                        total=total,
                    )
                try:
                    if await self._promote(row):
                        promoted += 1
                except Exception as exc:  # noqa: BLE001
                    log.debug("model judgment failed for %s: %s", row["id"], exc)

        await asyncio.gather(*(worker() for _ in range(2)))
        elapsed = time.monotonic() - started
        log.info("model judgment promoted %s/%s items in %.0fs", promoted, total, elapsed)
        return promoted, elapsed

    async def _promote(self, row) -> bool:
        heuristic = {
            "quality": float(row["quality"]),
            "practicality": float(row["practicality"]),
            "feasibility": float(row["feasibility"]),
            "usefulness": float(row["usefulness"]),
            "readiness": float(row["readiness"]),
            "verdict": row["verdict"],
            "reasons": jload(row["reasons"], []),
            "artifacts": jload(row["artifacts"], []),
        }
        prompt = PROMPT.format(
            title=fence("TITLE", row["title"] or "", limit=250),
            body=fence("BODY", row["body"] or "", limit=900),
            source=fence("SOURCE", row["source_name"] or "", limit=80),
            tier=row["tier"] or "news",
            category=row["category"] or "unknown",
            q=heuristic["quality"],
            p=heuristic["practicality"],
            f=heuristic["feasibility"],
            u=heuristic["usefulness"],
            r=heuristic["readiness"],
        )
        async with self._gate:
            premium = float(row["readiness"] or 0) >= self.settings.premium_readiness
            payload = await self.client.generate_json(
                prompt, system=SYSTEM, schema=SCHEMA, num_predict=180,
                premium=premium,
            )
        if not payload:
            return False

        blended = blend(heuristic, payload)
        self.db.execute(
            """
            UPDATE judgments SET
                quality=?, practicality=?, feasibility=?, usefulness=?,
                readiness=?, verdict=?, reasons=?, artifacts=?,
                model=?, created_at=?
            WHERE item_id=?
            """,
            (
                blended["quality"],
                blended["practicality"],
                blended["feasibility"],
                blended["usefulness"],
                blended["readiness"],
                blended["verdict"],
                jdump(blended["reasons"]),
                jdump(blended["artifacts"]),
                self.client.model_for(premium=premium),
                iso(utcnow()),
                row["id"],
            ),
        )
        return True
