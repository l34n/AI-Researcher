"""Chat backends: local Ollama, cheap Gemini, optional OpenRouter premium.

Embeddings stay on Ollama (or TF-IDF). Chat is routed by *content quality*:

* **Workhorse** — bulk enrichment and low-readiness judgment. Gemini Flash
  if a key is set, else the cheap OpenRouter model, else local Ollama.
* **Premium** — deep-research wiki, the daily brief, and judgment when the
  heuristic readiness is already at the research gate. OpenRouter's high-end
  model if a key is set, else Gemini Pro, else the workhorse.

No keys required. Missing backends drop out; the run still finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings
from ..util import local_day
from .ollama import OllamaClient, _parse_json_object

log = logging.getLogger("ai_researcher.chat")

_UNSET = object()

_SECRET = re.compile(
    r"(key=)[^&\s]+|(Bearer\s+)\S+|((?:sk-|or-)[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """Keep vendor error URLs out of logs and doctor output."""
    if not text:
        return text
    return _SECRET.sub(lambda m: (m.group(1) or m.group(2) or "") + "REDACTED", text)


def consume_daily_budget(data_dir: Path, *, limit: int, day: str | None = None) -> bool:
    """True when a model call is allowed. Increments the on-disk counter.

    A limit of 0 or less means unlimited. The file is best-effort: two
    processes can race, which is acceptable for a soft daily cap.
    """
    if limit <= 0:
        return True
    day = day or local_day()
    path = Path(data_dir) / "model_budget.json"
    payload = {"day": day, "count": 0}
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8")) or payload
    except (OSError, ValueError):
        payload = {"day": day, "count": 0}
    if payload.get("day") != day:
        payload = {"day": day, "count": 0}
    if int(payload.get("count") or 0) >= limit:
        return False
    payload["count"] = int(payload.get("count") or 0) + 1
    try:
        if not Path(data_dir).is_dir():
            return True
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
    return True


class OpenAICompatChat:
    """OpenRouter (and any OpenAI-compatible chat completions endpoint)."""

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str,
        base_url: str,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout, connect=8.0)
        )
        self.last_error = ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str | None:
        if not model or not self.api_key:
            return None
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._headers(),
                **kwargs,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.last_error = redact_secrets(str(exc) or type(exc).__name__)[:200]
            log.warning("OpenRouter generate failed: %s", self.last_error)
            return None
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            self.last_error = "empty OpenRouter response"
            return None
        message = (choices[0] or {}).get("message") or {}
        text = (message.get("content") or "").strip()
        if not text and (choices[0] or {}).get("finish_reason") == "length":
            # Reasoning models spend max_tokens on hidden thinking and return
            # nothing; say so instead of silently handing back a fallback.
            self.last_error = f"{model}: empty content, finish_reason=length (reasoning budget?)"
            log.warning("OpenRouter generate returned no content: %s", self.last_error)
        return text or None


class GeminiChat:
    """Google Gemini generateContent. Cheap workhorse when a key is set."""

    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout, connect=8.0)
        )
        self.last_error = ""

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str | None:
        if not model or not self.api_key:
            return None
        gen: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
        }
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.post(
                f"{self.BASE}/models/{model}:generateContent",
                params={"key": self.api_key},
                json=body,
                **kwargs,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.last_error = redact_secrets(str(exc) or type(exc).__name__)[:200]
            log.warning("Gemini generate failed: %s", self.last_error)
            return None
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            self.last_error = "empty Gemini response"
            return None
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
        return text or None


class ChatRouter:
    """Duck-types :class:`OllamaClient` so enrich / judge / wiki / brief stay unchanged."""

    def __init__(
        self,
        settings: Settings,
        *,
        ollama: OllamaClient | Any | None = None,
        openrouter: Any = _UNSET,
        gemini: Any = _UNSET,
    ):
        self.settings = settings
        self.ollama = ollama if ollama is not None else OllamaClient(settings)
        self._openrouter = openrouter
        self._gemini = gemini
        self.available = False
        self.last_error = ""
        self._gen_gate = asyncio.Semaphore(max(1, int(settings.max_concurrent_generations or 1)))

    def _openrouter_backend(self) -> Any:
        if self._openrouter is _UNSET:
            key = self.settings.openrouter_api_key
            if key:
                self._openrouter = OpenAICompatChat(
                    self.settings,
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                    extra_headers={
                        "HTTP-Referer": f"http://localhost:{self.settings.port}",
                        "X-Title": "AI Researcher",
                    },
                )
            else:
                self._openrouter = None
        return self._openrouter

    def _gemini_backend(self) -> Any:
        if self._gemini is _UNSET:
            key = self.settings.gemini_api_key
            self._gemini = GeminiChat(self.settings, api_key=key) if key else None
        return self._gemini

    def _role_override(self, role: str) -> str:
        s = self.settings
        return {
            "enrich": s.ollama_enrich_model,
            "judge": s.ollama_judge_model,
            "research": s.ollama_research_model,
            "brief": s.ollama_brief_model,
        }.get(role, "") or ""

    def _ollama_serves(self, tag: str) -> bool:
        installed = set(getattr(self.ollama, "installed", []) or [])
        return bool(tag) and (tag in installed or f"{tag}:latest" in installed)

    def _pair(self, premium: bool, role: str = "") -> tuple[Any, str]:
        """Return ``(backend, model)`` for this band and role.

        A per-role override (``AIR_BRIEF_MODEL`` etc.) names a model, not a
        backend. If that name is an installed Ollama tag it must go to Ollama
        even when a cloud key is set; sending ``qwen3:32b`` to OpenRouter is a
        400 and a silent fallback brief. A cloud slug in the override rides the
        band's cloud backend as before.
        """
        backend, model = self._band_pair(premium)
        override = self._role_override(role)
        if not override:
            return backend, model
        if self._ollama_serves(override) and backend is not self.ollama:
            return self.ollama, override
        if backend is None and (getattr(self.ollama, "chat_model", "") or ""):
            return self.ollama, override
        return backend, override

    def _band_pair(self, premium: bool) -> tuple[Any, str]:
        """Return ``(backend, model)`` for this quality band. Deterministic; no last-used state."""
        s = self.settings
        gemini = self._gemini_backend()
        openrouter = self._openrouter_backend()
        ollama_model = getattr(self.ollama, "chat_model", "") or ""
        if premium:
            if openrouter:
                return openrouter, s.openrouter_premium_model
            if gemini:
                return gemini, s.gemini_premium_model
            if ollama_model:
                return self.ollama, ollama_model
            return None, ""
        if gemini:
            return gemini, s.gemini_model
        if openrouter:
            return openrouter, s.openrouter_model
        if ollama_model:
            return self.ollama, ollama_model
        return None, ""

    def model_for(self, *, premium: bool = False, role: str = "") -> str:
        return self._pair(premium, role)[1]

    @property
    def chat_model(self) -> str:
        return self.model_for(premium=False)

    @property
    def embed_model(self) -> str:
        return getattr(self.ollama, "embed_model", "") or ""

    @property
    def installed(self) -> list[str]:
        return list(getattr(self.ollama, "installed", []) or [])

    def describe(self) -> dict[str, Any]:
        chat = self.model_for(premium=False)
        warn = ""
        from .ollama import model_over_budget, param_billions
        size = param_billions(self.ollama.chat_model if self.ollama.chat_model else chat)
        if model_over_budget(self.ollama.chat_model or chat, max_gb=self.settings.max_model_memory_gb):
            warn = (
                f"configured model looks larger than AIR_MAX_MODEL_MEMORY_GB="
                f"{self.settings.max_model_memory_gb}"
            )
        return {
            "available": self.available,
            "workhorse": chat,
            "premium": self.model_for(premium=True),
            "ollama": getattr(self.ollama, "chat_model", "") or "",
            "embed": self.embed_model,
            "enrich": self.model_for(role="enrich"),
            "judge": self.model_for(role="judge"),
            "research": self.model_for(premium=True, role="research"),
            "brief": self.model_for(premium=True, role="brief"),
            "gemini": bool(self.settings.gemini_api_key),
            "openrouter": bool(self.settings.openrouter_api_key),
            "premium_readiness": self.settings.premium_readiness,
            "default_chat": self.settings.ollama_default_chat_model,
            "param_b": size,
            "resource_warning": warn,
            "error": self.last_error or getattr(self.ollama, "last_error", "") or "",
        }

    async def probe(self) -> bool:
        """Discover Ollama models; cloud keys count as available without a round-trip."""
        try:
            await self.ollama.probe()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Ollama probe failed: %s", exc)
        self.available = bool(self.model_for(premium=False))
        if not self.available:
            self.last_error = (
                self.last_error
                or getattr(self.ollama, "last_error", "")
                or "no chat backend configured"
            )
        return self.available

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        schema: dict[str, Any] | None = None,
        num_predict: int = 512,
        temperature: float = 0.1,
        premium: bool = False,
        model: str | None = None,
        role: str = "",
    ) -> dict[str, Any] | None:
        backend, use_model = self._pair(premium, role)
        use_model = model or use_model
        if backend is None or not use_model:
            return None
        async with self._gen_gate:
            # The daily cap exists to bound cloud spend. Local Ollama calls
            # cost nothing, and charging them meant an hourly ingest with
            # enrich/judge/brief on Ollama burned the whole cap by evening and
            # every later brief was the fallback (APE-703).
            if backend is not self.ollama and not consume_daily_budget(
                self.settings.data_dir, limit=int(self.settings.daily_model_calls or 0)
            ):
                self.last_error = "daily model-call budget exhausted"
                log.warning("%s — degrading to rules", self.last_error)
                return None
            return await self._generate_json_unlocked(
                backend, use_model, prompt,
                system=system, schema=schema, num_predict=num_predict,
                temperature=temperature, premium=premium,
            )

    async def _generate_json_unlocked(
        self,
        backend: Any,
        use_model: str,
        prompt: str,
        *,
        system: str,
        schema: dict[str, Any] | None,
        num_predict: int,
        temperature: float,
        premium: bool,
    ) -> dict[str, Any] | None:
        if backend is self.ollama:
            return await self.ollama.generate_json(
                prompt,
                system=system,
                schema=schema,
                num_predict=num_predict,
                temperature=temperature,
                premium=premium,
                model=use_model,
            )
        text = await backend.complete(
            model=use_model,
            prompt=prompt,
            system=system,
            max_tokens=num_predict,
            temperature=temperature,
            json_mode=True,
        )
        if getattr(backend, "last_error", ""):
            self.last_error = backend.last_error
        if not text:
            return None
        return _parse_json_object(text)

    async def generate_text(
        self,
        prompt: str,
        *,
        system: str = "",
        num_predict: int = 900,
        temperature: float = 0.3,
        timeout: float | None = None,
        premium: bool = False,
        model: str | None = None,
        role: str = "",
    ) -> str | None:
        backend, use_model = self._pair(premium, role)
        use_model = model or use_model
        if backend is None or not use_model:
            return None
        async with self._gen_gate:
            # The daily cap exists to bound cloud spend. Local Ollama calls
            # cost nothing, and charging them meant an hourly ingest with
            # enrich/judge/brief on Ollama burned the whole cap by evening and
            # every later brief was the fallback (APE-703).
            if backend is not self.ollama and not consume_daily_budget(
                self.settings.data_dir, limit=int(self.settings.daily_model_calls or 0)
            ):
                self.last_error = "daily model-call budget exhausted"
                log.warning("%s — degrading to rules", self.last_error)
                return None
            return await self._generate_text_unlocked(
                backend, use_model, prompt,
                system=system, num_predict=num_predict,
                temperature=temperature, timeout=timeout, premium=premium,
            )

    async def _generate_text_unlocked(
        self,
        backend: Any,
        use_model: str,
        prompt: str,
        *,
        system: str,
        num_predict: int,
        temperature: float,
        timeout: float | None,
        premium: bool,
    ) -> str | None:
        if backend is self.ollama:
            return await self.ollama.generate_text(
                prompt,
                system=system,
                num_predict=num_predict,
                temperature=temperature,
                timeout=timeout,
                premium=premium,
                model=use_model,
            )
        text = await backend.complete(
            model=use_model,
            prompt=prompt,
            system=system,
            max_tokens=num_predict,
            temperature=temperature,
            json_mode=False,
            timeout=timeout,
        )
        if getattr(backend, "last_error", ""):
            self.last_error = backend.last_error
        return text

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        return await self.ollama.embed(texts)

    async def aclose(self) -> None:
        await self.ollama.aclose()
        for backend in (self._openrouter, self._gemini):
            if backend is _UNSET or backend is None:
                continue
            close = getattr(backend, "aclose", None)
            if close is not None:
                await close()
