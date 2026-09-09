"""Docker packaging: secrets stay out of the image, Compose is deployable."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TestDockerfile:
    def test_runs_as_non_root_and_healthchecks(self):
        text = ROOT.joinpath("Dockerfile").read_text(encoding="utf-8")
        assert "USER app" in text
        assert "HEALTHCHECK" in text
        assert "/healthz" in text
        assert "AIR_DATA_DIR=/data" in text
        assert "AIR_AUTO_REFRESH_MIN=60" in text
        assert "COPY .env" not in text
        assert "COPY data" not in text

    def test_installs_non_editable_with_packaged_sources(self):
        """Runtime image is a regular install; catalog is copied and packaged."""
        text = ROOT.joinpath("Dockerfile").read_text(encoding="utf-8")
        assert "pip install" in text
        assert "pip install --no-cache-dir ." in text
        assert "-e ." not in text
        assert "AIR_SOURCES_PATH=/app/config/sources.yaml" in text
        assert "ai.researcher.schema-version" in text
        assert "OLLAMA_DEFAULT_CHAT_MODEL=gemma3:4b" in text


class TestDockerignore:
    def test_excludes_secrets_and_state(self):
        text = ROOT.joinpath(".dockerignore").read_text(encoding="utf-8")
        for line in (".env", "data", ".git"):
            assert line in text.splitlines() or any(
                ln.strip() == line for ln in text.splitlines()
            )


class TestCompose:
    def test_persists_data_and_reaches_host_ollama(self):
        compose = yaml.safe_load(ROOT.joinpath("docker-compose.yml").read_text(encoding="utf-8"))
        app = compose["services"]["ai-researcher"]
        volumes = [str(v) for v in app["volumes"]]
        assert any(v.endswith(":/data") for v in volumes)
        assert "host.docker.internal:host-gateway" in app["extra_hosts"]
        assert "AIR_AUTO_REFRESH_MIN" in app["environment"]
        assert app["environment"]["AIR_DATA_DIR"] == "/data"
        assert "ollama" in compose["services"]
        assert "ollama" in compose["services"]["ollama"].get("profiles", [])
        worker = compose["services"]["worker"]
        assert "worker" in worker.get("profiles", [])
        assert any(str(v).endswith(":/data") for v in worker["volumes"])
