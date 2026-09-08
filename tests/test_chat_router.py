"""Chat router: OpenRouter + Gemini + quality-gated model selection.

No live vendor calls. Backends are fakes or httpx.MockTransport.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from ai_researcher.config import Settings
from ai_researcher.enrich.chat import (
    ChatRouter,
    GeminiChat,
    OpenAICompatChat,
    redact_secrets,
)


class FakeOllama:
    chat_model = "qwen3:4b"
    embed_model = "nomic-embed-text"
    available = True
    last_error = ""
    installed = ["qwen3:4b"]

    def __init__(self):
        self.json_calls = []
        self.text_calls = []

    async def probe(self):
        return True

    async def generate_json(self, prompt, **kwargs):
        self.json_calls.append(kwargs)
        return {"from": "ollama", "model": kwargs.get("model")}

    async def generate_text(self, prompt, **kwargs):
        self.text_calls.append(kwargs)
        return "ollama-text"

    async def embed(self, texts):
        return None

    async def aclose(self):
        return None

    def model_for(self, *, premium=False, role=""):
        return self.chat_model


class RecBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.last_error = ""

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("json_mode"):
            return json.dumps({"from": self.name, "model": kwargs.get("model")})
        return f"{self.name}-text"

    async def aclose(self):
        return None


def _settings(**kw) -> Settings:
    defaults = dict(
        gemini_model="gemini-2.5-flash",
        gemini_premium_model="gemini-2.5-pro",
        openrouter_model="google/gemini-2.5-flash",
        openrouter_premium_model="anthropic/claude-sonnet-4",
        premium_readiness=0.62,
    )
    defaults.update(kw)
    return Settings(**defaults)


class TestRouting:
    def test_local_only_uses_ollama_for_both_bands(self):
        ollama = FakeOllama()
        router = ChatRouter(_settings(), ollama=ollama)
        asyncio.run(router.probe())
        assert router.model_for(premium=False) == "qwen3:4b"
        assert router.model_for(premium=True) == "qwen3:4b"
        assert router.available

    def test_gemini_is_the_cheap_workhorse(self):
        gemini = RecBackend("gemini")
        router = ChatRouter(
            _settings(gemini_api_key="g"),
            ollama=FakeOllama(),
            gemini=gemini,
        )
        asyncio.run(router.probe())
        assert router.model_for(premium=False) == "gemini-2.5-flash"
        assert router.model_for(premium=True) == "gemini-2.5-pro"

    def test_openrouter_takes_premium_when_both_keys_are_set(self):
        gemini = RecBackend("gemini")
        openrouter = RecBackend("openrouter")
        router = ChatRouter(
            _settings(gemini_api_key="g", openrouter_api_key="o"),
            ollama=FakeOllama(),
            gemini=gemini,
            openrouter=openrouter,
        )
        asyncio.run(router.probe())
        assert router.chat_model == "gemini-2.5-flash"
        assert router.model_for(premium=True) == "anthropic/claude-sonnet-4"

        asyncio.run(router.generate_json("hi", premium=False))
        asyncio.run(router.generate_text("hi", premium=True))
        assert gemini.calls[0]["model"] == "gemini-2.5-flash"
        assert gemini.calls[0]["json_mode"] is True
        assert openrouter.calls[0]["model"] == "anthropic/claude-sonnet-4"
        assert openrouter.calls[0]["json_mode"] is False

    def test_openrouter_only_uses_cheap_then_premium_models(self):
        openrouter = RecBackend("openrouter")
        router = ChatRouter(
            _settings(openrouter_api_key="o"),
            ollama=FakeOllama(),
            openrouter=openrouter,
        )
        assert router.model_for(premium=False) == "google/gemini-2.5-flash"
        assert router.model_for(premium=True) == "anthropic/claude-sonnet-4"

    def test_describe_never_includes_keys(self):
        router = ChatRouter(
            _settings(gemini_api_key="super-secret", openrouter_api_key="also-secret"),
            ollama=FakeOllama(),
            gemini=RecBackend("g"),
            openrouter=RecBackend("o"),
        )
        blob = json.dumps(router.describe())
        assert "super-secret" not in blob
        assert "also-secret" not in blob
        assert router.describe()["gemini"] is True
        assert router.describe()["openrouter"] is True


class TestOpenRouterShape:
    def test_json_mode_sends_response_format_and_bearer(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAICompatChat(
            Settings(),
            api_key="sk-test-secret",
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"X-Title": "AI Researcher", "HTTP-Referer": "http://localhost:8899"},
            client=http,
        )
        text = asyncio.run(
            backend.complete(
                model="anthropic/claude-sonnet-4",
                prompt="score this",
                system="json only",
                json_mode=True,
                max_tokens=180,
            )
        )
        asyncio.run(http.aclose())
        assert json.loads(text) == {"ok": True}
        req = captured[0]
        assert str(req.url).endswith("/chat/completions")
        assert req.headers["Authorization"] == "Bearer sk-test-secret"
        assert req.headers["X-Title"] == "AI Researcher"
        body = json.loads(req.content)
        assert body["model"] == "anthropic/claude-sonnet-4"
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["content"] == "score this"


class TestGeminiShape:
    def test_json_mode_sets_mime_type_and_system_instruction(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]},
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = GeminiChat(Settings(), api_key="AIza-test-secret", client=http)
        text = asyncio.run(
            backend.complete(
                model="gemini-2.5-flash",
                prompt="score this",
                system="json only",
                json_mode=True,
            )
        )
        asyncio.run(http.aclose())
        assert json.loads(text) == {"ok": True}
        req = captured[0]
        assert req.url.params["key"] == "AIza-test-secret"
        assert ":generateContent" in str(req.url)
        body = json.loads(req.content)
        assert body["system_instruction"]["parts"][0]["text"] == "json only"
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["contents"][0]["parts"][0]["text"] == "score this"

    def test_failed_call_redacts_the_api_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="nope")

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = GeminiChat(Settings(), api_key="AIza-test-secret", client=http)
        text = asyncio.run(
            backend.complete(model="gemini-2.5-flash", prompt="x")
        )
        asyncio.run(http.aclose())
        assert text is None
        assert "AIza-test-secret" not in backend.last_error
        assert "REDACTED" in backend.last_error or "401" in backend.last_error


class TestRedact:
    def test_query_key_and_bearer_are_stripped(self):
        url = "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=AIzaSecret"
        assert "AIzaSecret" not in redact_secrets(url)
        assert "Bearer REDACTED" in redact_secrets("401 Bearer sk-live-abcdef")


class TestChatRouterPlumbing:
    def test_no_keys_does_not_construct_cloud_backends(self):
        router = ChatRouter(_settings(), ollama=FakeOllama())
        assert router._gemini_backend() is None
        assert router._openrouter_backend() is None

    def test_embed_delegates_to_ollama(self):
        ollama = FakeOllama()
        ollama.embed_called = []

        async def embed(texts):
            ollama.embed_called.append(texts)
            return [[0.1, 0.2]]

        ollama.embed = embed
        router = ChatRouter(_settings(), ollama=ollama)
        vectors = asyncio.run(router.embed(["hello"]))
        assert vectors == [[0.1, 0.2]]
        assert ollama.embed_called == [["hello"]]

    def test_aclose_closes_injected_backends(self):
        gemini = RecBackend("gemini")
        gemini.closed = False

        async def aclose():
            gemini.closed = True

        gemini.aclose = aclose
        router = ChatRouter(
            _settings(gemini_api_key="g"),
            ollama=FakeOllama(),
            gemini=gemini,
        )
        asyncio.run(router.aclose())
        assert gemini.closed is True

    def test_wrapped_json_from_cloud_is_parsed(self):
        class Prose(RecBackend):
            async def complete(self, **kwargs):
                self.calls.append(kwargs)
                return "Sure, here you go:\n```json\n{\"ok\": true}\n```"

        backend = Prose("gemini")
        router = ChatRouter(
            _settings(gemini_api_key="g"),
            ollama=FakeOllama(),
            gemini=backend,
        )
        payload = asyncio.run(router.generate_json("hi"))
        assert payload == {"ok": True}

    def test_empty_cloud_response_returns_none(self):
        class Empty(RecBackend):
            async def complete(self, **kwargs):
                return None

        router = ChatRouter(
            _settings(gemini_api_key="g"),
            ollama=FakeOllama(),
            gemini=Empty("gemini"),
        )
        assert asyncio.run(router.generate_json("hi")) is None
        assert asyncio.run(router.generate_text("hi")) is None

    def test_workhorse_json_does_not_hit_premium_backend(self):
        gemini = RecBackend("gemini")
        openrouter = RecBackend("openrouter")
        router = ChatRouter(
            _settings(gemini_api_key="g", openrouter_api_key="o"),
            ollama=FakeOllama(),
            gemini=gemini,
            openrouter=openrouter,
        )
        asyncio.run(router.generate_json("enrich me", premium=False))
        assert gemini.calls
        assert not openrouter.calls


class TestSettingsEnv:
    def test_load_reads_cloud_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("AIR_PREMIUM_READINESS", "0.70")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        monkeypatch.setenv("OPENROUTER_PREMIUM_MODEL", "anthropic/claude-opus-4")
        s = Settings.load()
        assert s.gemini_api_key == "g-key"
        assert s.openrouter_api_key == "or-key"
        assert s.premium_readiness == 0.70
        assert s.gemini_model == "gemini-2.5-flash-lite"
        assert s.openrouter_premium_model == "anthropic/claude-opus-4"

    def test_role_models_and_default_chat(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("OLLAMA_DEFAULT_CHAT_MODEL", "gemma3:4b")
        monkeypatch.setenv("AIR_ENRICH_MODEL", "qwen3:4b")
        monkeypatch.setenv("AIR_BRIEF_MODEL", "qwen3:8b")
        s = Settings.load()
        assert s.ollama_default_chat_model == "gemma3:4b"
        router = ChatRouter(s, ollama=FakeOllama())
        assert router.model_for(role="enrich") == "qwen3:4b"
        assert router.model_for(premium=True, role="brief") == "qwen3:8b"

    def test_google_api_key_fills_in_for_gemini(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        s = Settings.load()
        assert s.gemini_api_key == "google-key"

    def test_sources_path_can_be_overridden(self, monkeypatch, tmp_path):
        catalog = tmp_path / "sources.yaml"
        catalog.write_text("sources: []\n", encoding="utf-8")
        monkeypatch.setenv("AIR_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("AIR_SOURCES_PATH", str(catalog))
        s = Settings.load()
        assert s.sources_path == catalog


class TestGeminiMultipart:
    def test_joins_all_text_parts(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [
                    {"text": "hello "},
                    {"text": "world"},
                ]}}]},
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = GeminiChat(Settings(), api_key="k", client=http)
        text = asyncio.run(backend.complete(model="gemini-2.5-flash", prompt="x"))
        asyncio.run(http.aclose())
        assert text == "hello world"


class TestOpenRouterEmpty:
    def test_empty_choices_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAICompatChat(
            Settings(), api_key="sk-test", base_url="https://openrouter.ai/api/v1", client=http,
        )
        text = asyncio.run(backend.complete(model="x", prompt="hi"))
        asyncio.run(http.aclose())
        assert text is None
        assert backend.last_error == "empty OpenRouter response"


class TestRoleOverrideBackend:
    """A role override names a model; the backend follows the model."""

    def test_installed_ollama_tag_stays_on_ollama_when_a_cloud_key_is_set(self, tmp_path):
        ollama = FakeOllama()
        openrouter = RecBackend("openrouter")
        router = ChatRouter(
            _settings(openrouter_api_key="o", ollama_brief_model="qwen3:4b",
                      ollama_judge_model="qwen3:4b", data_dir=tmp_path),
            ollama=ollama,
            openrouter=openrouter,
        )
        asyncio.run(router.probe())
        assert router.model_for(premium=True, role="brief") == "qwen3:4b"
        asyncio.run(router.generate_text("brief me", premium=True, role="brief"))
        asyncio.run(router.generate_json("judge me", premium=False, role="judge"))
        assert openrouter.calls == []
        assert ollama.text_calls[0]["model"] == "qwen3:4b"
        assert ollama.json_calls[0]["model"] == "qwen3:4b"
        # Roles without an override still ride the cloud band.
        asyncio.run(router.generate_text("research", premium=True, role="research"))
        assert openrouter.calls[0]["model"] == "anthropic/claude-sonnet-4"

    def test_cloud_slug_override_rides_the_cloud_backend(self, tmp_path):
        ollama = FakeOllama()
        openrouter = RecBackend("openrouter")
        router = ChatRouter(
            _settings(openrouter_api_key="o", ollama_brief_model="qwen/qwen3.7-flash",
                      data_dir=tmp_path),
            ollama=ollama,
            openrouter=openrouter,
        )
        asyncio.run(router.probe())
        asyncio.run(router.generate_text("brief me", premium=True, role="brief"))
        assert ollama.text_calls == []
        assert openrouter.calls[0]["model"] == "qwen/qwen3.7-flash"


class TestDailyBudgetIsCloudOnly:
    def test_ollama_calls_are_not_charged_and_still_run_when_cap_is_hit(self, tmp_path):
        from ai_researcher.enrich.chat import consume_daily_budget

        ollama = FakeOllama()
        router = ChatRouter(_settings(data_dir=tmp_path, daily_model_calls=1), ollama=ollama)
        asyncio.run(router.probe())
        assert consume_daily_budget(tmp_path, limit=1) is True   # cap now reached
        assert asyncio.run(router.generate_text("hi")) == "ollama-text"
        assert asyncio.run(router.generate_json("hi"))["from"] == "ollama"
        assert consume_daily_budget(tmp_path, limit=1) is False  # counter untouched by Ollama calls

    def test_cloud_calls_are_still_capped(self, tmp_path):
        openrouter = RecBackend("openrouter")
        router = ChatRouter(_settings(openrouter_api_key="o", data_dir=tmp_path, daily_model_calls=1),
                            ollama=FakeOllama(), openrouter=openrouter)
        asyncio.run(router.probe())
        assert asyncio.run(router.generate_text("one", premium=True)) == "openrouter-text"
        assert asyncio.run(router.generate_text("two", premium=True)) is None
        assert router.last_error == "daily model-call budget exhausted"
        assert len(openrouter.calls) == 1
