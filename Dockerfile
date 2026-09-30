# CodeWeaver — deployable container image.
#
# Runs the FastAPI backend (which also serves the dashboard) on $PORT.
# Works on Render, Railway, Fly.io, Koyeb, or any Docker host.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# git: cloning repositories; pytest: so the agent can run ingested repos' tests.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (better layer caching).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r backend/requirements.txt pytest

# Application code.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY app/ ./app/
COPY run_server.py ./

# Writable runtime state (databases, workspaces, repo snapshots).
# On hosts with an ephemeral disk this resets on redeploy — attach a
# persistent volume here if you want state to survive restarts.
RUN mkdir -p /app/data

EXPOSE 8600

# Run from backend/ so `app.main:app` resolves to the real package.
# ${PORT:-8600} respects the host-provided port (Render/Railway/Fly).
WORKDIR /app/backend
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8600}
