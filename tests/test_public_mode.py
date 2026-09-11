"""AIR_PUBLIC_MODE: a read-only public instance of the same dashboard.

One codebase, two processes, one SQLite file. The private instance keeps its
token, its controls and its ingest; the public one serves Dashboard, Firehose,
Search and Adapt to anyone and can write nothing.

Every public assertion here has a private counterpart beside it. A check that
"the save button is absent" proves nothing on its own — it would also pass if
the button had been deleted for everybody — so each one is paired with the
private-mode reading that shows the affordance is still there by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_researcher.config import Settings

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "sources.yaml"


class TestFlag:
    """`Settings.public_mode`, off unless the environment says otherwise."""

    def test_default_is_off(self):
        assert Settings().public_mode is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_environment_turns_it_on(self, raw, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_PUBLIC_MODE", raw)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe"])
    def test_everything_else_leaves_it_off(self, raw, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_PUBLIC_MODE", raw)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is False

    def test_absent_environment_leaves_it_off(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AIR_PUBLIC_MODE", raising=False)
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        assert Settings.load().public_mode is False
