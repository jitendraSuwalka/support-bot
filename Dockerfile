# The HTTP service (api.py) in a container.
#
#   docker build -t support-bot .
#   docker run --env-file .env -p 127.0.0.1:8000:8000 support-bot
#
# The embedding model and the document index are built into the image, so a
# container is ready the moment it starts and makes no download at runtime:
# its only outbound calls are to the Anthropic API. It runs as an unprivileged
# user, logs one JSON line per turn to stderr, and reports its health to
# Docker through /health. CI builds this image and smoke-tests it on every
# push (scripts/container_smoke_test.sh).

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# An unprivileged user to run as. Nothing in the image needs root once built.
# The code stays owned by root, so the service cannot rewrite itself; the user
# owns only the two directories it writes: the model cache (at build time)
# and the index (Qdrant keeps a lock file there at runtime).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app/.models /app/.index \
    && chown app:app /app/.models /app/.index

# Dependencies first, in their own layer, so a code change does not reinstall
# them. Runtime dependencies only (requirements.txt), at the exact versions
# the lockfile pins for everything they pull in; the test tools stay out.
COPY requirements.txt requirements.lock ./
RUN pip install -r requirements.txt -c requirements.lock

COPY agent.py agent_graph.py api.py app.py config.py observability.py pricing.py ./
COPY rag/ rag/
COPY corpus/ corpus/

USER app

ENV SUPPORT_BOT_MODEL_CACHE=/app/.models \
    SUPPORT_BOT_INDEX_PATH=/app/.index \
    SUPPORT_BOT_LOG_FORMAT=json

# Download the embedding model and build the index now, at build time.
RUN python -m rag.index

# From here on the model is read from the cache only: a container never
# reaches out to the model hub, and would fail loudly rather than quietly
# download if the cache were missing.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]

# --no-access-log: the service writes its own JSON line per request, so
# uvicorn's plain-text access lines would only break the stream's format.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
