"""Benchmark and backtest sweeps must never touch the production daily budget.

Regression for APE-727: the runners built their ``Settings`` from the
production ``.env``, so a sweep's generations counted against
``data/model_budget.json`` and could exhaust ``AIR_DAILY_MODEL_CALLS`` for
the live brief. The sweep now runs on a private data directory with the cap
disabled. These tests stand up a fake "production" data dir holding a
counter and check nothing moves it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_models  # noqa: E402
from benchmark_models import (  # noqa: E402
    ModelSpec,
    ProviderClient,
    isolate_settings,
    run_one_model,
    sweep_settings,
)
from ai_researcher.config import Settings  # noqa: E402
from ai_researcher.enrich.chat import ChatRouter, consume_daily_budget  # noqa: E402
from ai_researcher.util import local_day  # noqa: E402

SEED_COUNT = 5


def _read_count(data_dir: Path) -> int:
    return int(json.loads((data_dir / "model_budget.json").read_text())["count"])


@pytest.fixture
def production(tmp_path, monkeypatch):
    """A fake production data dir with a live counter, wired as the default."""
    prod = tmp_path / "prod"
    prod.mkdir()
    (prod / "model_budget.json").write_text(
        json.dumps({"day": local_day(), "count": SEED_COUNT}), encoding="utf-8"
    )
    monkeypatch.setenv("AIR_DATA_DIR", str(prod))
    monkeypatch.setattr(benchmark_models, "SWEEP_RUNTIME_DIR", tmp_path / "runtime")
    return prod


def _prod_settings(prod: Path) -> Settings:
    return Settings(data_dir=prod, daily_model_calls=SEED_COUNT + 10)


class FakeOllama:
    chat_model = "gemma3:27b"
    available = True
    last_error = ""
    installed = ["gemma3:27b"]

    async def probe(self):
        return True

    async def generate_text(self, prompt, **kwargs):
        return "ollama-text"

    async def generate_json(self, prompt, **kwargs):
        return {"ok": True}

    async def aclose(self):
        return None

    def model_for(self, *, premium=False, role=""):
        return self.chat_model


class TestIsolateSettings:
    def test_moves_data_dir_and_lifts_cap(self, production, tmp_path):
        s = isolate_settings(_prod_settings(production))
        assert s.data_dir == (tmp_path / "runtime").resolve()
        assert s.data_dir.is_dir()
        assert s.daily_model_calls == 0

    def test_is_idempotent(self, production):
        s = isolate_settings(_prod_settings(production))
        first = s.data_dir
        assert isolate_settings(s).data_dir == first

    def test_sweep_settings_ignores_air_data_dir(self, production):
        # Settings.load() honours AIR_DATA_DIR; the sweep loader must not.
        s = sweep_settings()
        assert s.data_dir != production.resolve()
        assert s.daily_model_calls == 0
        assert _read_count(production) == SEED_COUNT

    def test_router_on_isolated_settings_never_charges_production(self, production):
        s = isolate_settings(_prod_settings(production))
        router = ChatRouter(s, ollama=FakeOllama(), openrouter=None, gemini=None)
        for _ in range(SEED_COUNT):
            assert asyncio.run(router.generate_text("hi")) == "ollama-text"
        assert _read_count(production) == SEED_COUNT
        # Sanity: the same router on production settings does charge it for a
        # cloud backend (local Ollama calls are free of the cap since APE-703).
        class FakeCloud:
            last_error = ""

            async def complete(self, **kwargs):
                return "cloud-text"

            async def aclose(self):
                return None

        hot = ChatRouter(_prod_settings(production), ollama=FakeOllama(),
                         openrouter=FakeCloud(), gemini=None)
        hot.settings.openrouter_api_key = "k"
        asyncio.run(hot.generate_text("hi", premium=True))
        assert _read_count(production) == SEED_COUNT + 1


class TestRunnerIsolation:
    @pytest.fixture
    def stub_client(self, monkeypatch):
        """Keep the runner offline: a backend that answers like a budgeted router."""
        calls: list[str] = []

        def init_backend(self):
            self._backend = ChatRouter(self.settings, ollama=FakeOllama(),
                                       openrouter=None, gemini=None)

        def generate_prompt(self, prompt, *, system="", num_predict=900,
                            temperature=0.35, timeout=None, tag=""):
            calls.append(tag)
            out = self._loop.run_until_complete(self._backend.generate_text(prompt))
            self.stats.calls += 1
            return out or ""

        monkeypatch.setattr(ProviderClient, "_init_backend", init_backend)
        monkeypatch.setattr(ProviderClient, "generate_prompt", generate_prompt)
        return calls

    def test_provider_client_isolates_production_settings(self, production):
        spec = ModelSpec(slug="ollama-gemma3-27b", provider="ollama",
                         model="gemma3:27b", tier="local")
        client = ProviderClient(spec, _prod_settings(production))
        try:
            assert client.settings.data_dir != production.resolve()
            assert client.settings.daily_model_calls == 0
        finally:
            client.close()

    def test_one_case_leaves_production_counter_unchanged(self, production, stub_client):
        spec = ModelSpec(slug="ollama-gemma3-27b", provider="ollama",
                         model="gemma3:27b", tier="local")
        doc = run_one_model(spec, _prod_settings(production),
                            case_ids=["sum-single-hf"], full_fidelity=True)
        assert doc["cases"] == 1
        assert stub_client, "the stub backend was never called"
        assert _read_count(production) == SEED_COUNT
        # And the private counter is not silently capping the sweep either.
        assert not (benchmark_models.SWEEP_RUNTIME_DIR / "model_budget.json").exists()
        assert consume_daily_budget(benchmark_models.SWEEP_RUNTIME_DIR, limit=0) is True
