# syntax=docker/dockerfile:1
# Dashboard image. SQLite lives in /data. Ingest can run in-process or in a
# sibling worker container that mounts the same volume.

FROM python:3.12-slim-bookworm

ARG APP_VERSION=0.1.0
ARG SCHEMA_VERSION=4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    AIR_HOST=0.0.0.0 \
    AIR_PORT=8899 \
    AIR_DATA_DIR=/data \
    AIR_AUTO_REFRESH_MIN=60 \
    AIR_SOURCES_PATH=/app/config/sources.yaml \
    OLLAMA_DEFAULT_CHAT_MODEL=gemma3:4b

LABEL org.opencontainers.image.title="AI Researcher" \
      org.opencontainers.image.description="Local AI research dashboard" \
      org.opencontainers.image.version="${APP_VERSION}" \
      ai.researcher.schema-version="${SCHEMA_VERSION}" \
      ai.researcher.app-version="${APP_VERSION}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

# Hatchling needs the README/LICENSE next to pyproject at build time.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config

# Regular (non-editable) install. sources.yaml is packaged into the wheel and
# also copied to /app/config for AIR_SOURCES_PATH / bind-mount overrides.
RUN pip install --no-cache-dir . \
    && chown -R app:app /app

USER app

EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('AIR_PORT', '8899'), timeout=4)"

ENTRYPOINT ["ai-researcher"]
CMD ["serve"]
