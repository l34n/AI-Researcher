"""Configuration: .env loading, typed settings, and the source catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent.parent


def _default_sources_path() -> Path:
    """Repo checkout, packaged wheel, or the Docker image copy — first hit wins."""
    for candidate in (
        PROJECT_ROOT / "config" / "sources.yaml",
        PKG_ROOT / "packaged_sources.yaml",
        Path("/app/config/sources.yaml"),
    ):
        if candidate.is_file():
            return candidate
    return PROJECT_ROOT / "config" / "sources.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader. Real environment variables always win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8899
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    access_token: str = ""

    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = ""
    ollama_embed_model: str = ""
    ollama_timeout: int = 180
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.5-flash"
    openrouter_premium_model: str = "anthropic/claude-sonnet-4"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_premium_model: str = "gemini-2.5-pro"
    premium_readiness: float = 0.62
    enrich_budget: int = 40
    enrich_time_budget: int = 900
    judge_budget: int = 24
    judge_time_budget: int = 400
    research_budget: int = 4
    research_time_budget: int = 900
    research_threshold: float = 0.62

    # Conservative local default. Empty still auto-picks, but auto-pick
    # refuses to load a 30B+ model just because it is installed.
    ollama_default_chat_model: str = "gemma3:4b"
    ollama_enrich_model: str = ""
    ollama_judge_model: str = ""
    ollama_research_model: str = ""
    ollama_brief_model: str = ""
    max_model_memory_gb: float = 8.0
    max_context: int = 8192
    max_concurrent_generations: int = 2
    daily_model_calls: int = 250
    prefer_device: str = "auto"  # auto | cpu | gpu

    user_agent: str = "ai-researcher/0.1 (+local dashboard)"
    fetch_concurrency: int = 8
    item_max_age_days: int = 14
    retention_days: int = 120

    x_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    github_token: str = ""

    sources_path: Path = field(default_factory=_default_sources_path)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "airesearch.db"

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv(PROJECT_ROOT / ".env")
        s = cls(
            host=_env("AIR_HOST", "0.0.0.0"),
            port=_env_int("AIR_PORT", 8899),
            data_dir=Path(_env("AIR_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser(),
            access_token=_env("AIR_ACCESS_TOKEN"),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            ollama_chat_model=_env("OLLAMA_CHAT_MODEL"),
            ollama_embed_model=_env("OLLAMA_EMBED_MODEL"),
            ollama_timeout=_env_int("OLLAMA_TIMEOUT", 180),
            openrouter_api_key=_env("OPENROUTER_API_KEY"),
            openrouter_model=_env("OPENROUTER_MODEL") or "google/gemini-2.5-flash",
            openrouter_premium_model=_env("OPENROUTER_PREMIUM_MODEL") or "anthropic/claude-sonnet-4",
            gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            gemini_model=_env("GEMINI_MODEL") or "gemini-2.5-flash",
            gemini_premium_model=_env("GEMINI_PREMIUM_MODEL") or "gemini-2.5-pro",
            premium_readiness=_env_float("AIR_PREMIUM_READINESS", 0.62),
            enrich_budget=_env_int("AIR_ENRICH_BUDGET", 40),
            enrich_time_budget=_env_int("AIR_ENRICH_TIME_BUDGET", 900),
            judge_budget=_env_int("AIR_JUDGE_BUDGET", 24),
            judge_time_budget=_env_int("AIR_JUDGE_TIME_BUDGET", 400),
            research_budget=_env_int("AIR_RESEARCH_BUDGET", 4),
            research_time_budget=_env_int("AIR_RESEARCH_TIME_BUDGET", 900),
            research_threshold=_env_float("AIR_RESEARCH_THRESHOLD", 0.62),
            ollama_default_chat_model=_env("OLLAMA_DEFAULT_CHAT_MODEL", "gemma3:4b") or "gemma3:4b",
            ollama_enrich_model=_env("AIR_ENRICH_MODEL") or _env("OLLAMA_ENRICH_MODEL"),
            ollama_judge_model=_env("AIR_JUDGE_MODEL") or _env("OLLAMA_JUDGE_MODEL"),
            ollama_research_model=_env("AIR_RESEARCH_MODEL") or _env("OLLAMA_RESEARCH_MODEL"),
            ollama_brief_model=_env("AIR_BRIEF_MODEL") or _env("OLLAMA_BRIEF_MODEL"),
            max_model_memory_gb=_env_float("AIR_MAX_MODEL_MEMORY_GB", 8.0),
            max_context=_env_int("AIR_MAX_CONTEXT", 8192),
            max_concurrent_generations=_env_int("AIR_MAX_CONCURRENT_GEN", 2),
            daily_model_calls=_env_int("AIR_DAILY_MODEL_CALLS", 250),
            prefer_device=_env("AIR_PREFER_DEVICE", "auto") or "auto",
            user_agent=_env("AIR_USER_AGENT", "ai-researcher/0.1 (+local dashboard)"),
            fetch_concurrency=_env_int("AIR_FETCH_CONCURRENCY", 8),
            item_max_age_days=_env_int("AIR_ITEM_MAX_AGE_DAYS", 14),
            retention_days=_env_int("AIR_RETENTION_DAYS", 120),
            x_bearer_token=_env("X_BEARER_TOKEN"),
            reddit_client_id=_env("REDDIT_CLIENT_ID"),
            reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
            github_token=_env("GITHUB_TOKEN"),
        )
        if _env("AIR_SOURCES_PATH"):
            s.sources_path = Path(_env("AIR_SOURCES_PATH")).expanduser()
            if not s.sources_path.is_absolute():
                s.sources_path = (PROJECT_ROOT / s.sources_path).resolve()
        if not s.data_dir.is_absolute():
            s.data_dir = (PROJECT_ROOT / s.data_dir).resolve()
        s.data_dir.mkdir(parents=True, exist_ok=True)
        return s


@dataclass
class Source:
    key: str
    name: str
    kind: str
    tier: str = "news"
    weight: float = 1.0
    enabled: bool = True
    url: str = ""
    category_hint: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def load_sources(settings: Settings) -> list[Source]:
    raw = yaml.safe_load(settings.sources_path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    out: list[Source] = []
    for entry in raw.get("sources") or []:
        merged = {**defaults, **entry}
        out.append(
            Source(
                key=merged["key"],
                name=merged.get("name", merged["key"]),
                kind=merged["kind"],
                tier=merged.get("tier", "news"),
                weight=float(merged.get("weight", 1.0)),
                enabled=bool(merged.get("enabled", True)),
                url=merged.get("url", ""),
                category_hint=merged.get("category_hint", ""),
                config=merged.get("config") or {},
            )
        )
    return out


# Categories the classifier may assign. Kept small on purpose: a taxonomy the
# eye can scan in one pass is worth more than a precise one nobody filters by.
CATEGORIES = [
    "model-release",
    "research",
    "product-launch",
    "tooling-oss",
    "funding-acquisition",
    "infrastructure-compute",
    "benchmark-eval",
    "policy-regulation",
    "safety-incident",
    "opinion-analysis",
]

CATEGORY_LABELS = {
    "model-release": "Model Drops",
    "research": "Research",
    "product-launch": "Product Launches",
    "tooling-oss": "Tooling & OSS",
    "funding-acquisition": "Funding & M&A",
    "infrastructure-compute": "Infra & Compute",
    "benchmark-eval": "Benchmarks & Evals",
    "policy-regulation": "Policy & Regulation",
    "safety-incident": "Safety & Incidents",
    "opinion-analysis": "Analysis",
}
