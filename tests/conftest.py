"""Shared test setup.

The default suite is offline (`-m 'not network'`). `Settings()` still points
at http://localhost:11434, so a developer box with Ollama installed makes the
"without a model" tests take the live-model path and fail. Send every real
OllamaClient at a closed port unless a test opts back in with
AIR_TEST_LIVE_OLLAMA=1.

`Settings()` also defaults `data_dir` to the repo's `data/`, which on the
deployment box is production: the daily model-call counter in
`data/model_budget.json` made the router tests fail (and each run added to
the production count). Any Settings left on that default is moved to a
per-test temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_researcher.config import PROJECT_ROOT, Settings
from ai_researcher.enrich.ollama import OllamaClient

_DEAD_OLLAMA = "http://127.0.0.1:9"
_PRODUCTION_DATA_DIR = (PROJECT_ROOT / "data").resolve()


@pytest.fixture(autouse=True)
def _no_production_data_dir(monkeypatch, tmp_path):
    original_init = Settings.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if Path(self.data_dir).resolve() == _PRODUCTION_DATA_DIR:
            self.data_dir = tmp_path / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Settings, "__init__", init)
    yield


@pytest.fixture(autouse=True)
def _no_live_ollama(monkeypatch):
    if os.environ.get("AIR_TEST_LIVE_OLLAMA"):
        yield
        return
    original_init = OllamaClient.__init__

    def init(self, settings, *args, **kwargs):
        settings.ollama_host = _DEAD_OLLAMA
        original_init(self, settings, *args, **kwargs)

    monkeypatch.setattr(OllamaClient, "__init__", init)
    yield
