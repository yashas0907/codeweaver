# ⚒ CodeWeaver

**CodeWeaver — Autonomous AI Engineering Dashboard**

CodeWeaver is a self-contained, autonomous AI engineering system: it ingests a
codebase, builds a retrieval index over it, plans and executes engineering
tasks through a typed phase pipeline, runs the tests, reviews its own changes,
and produces a structured engineering report — all behind a clean HTTP API and
a built-in web dashboard.

It ships with a **deterministic, rule-based reasoning engine** out of the box,
so the full pipeline runs with no API key and no external services. When you
are ready for a real model, point it at any HTTP chat-completions endpoint —
such as Groq, vLLM, LM Studio, or Ollama (with the `/v1` adapter).

## Highlights

- **Repository ingestion** — index a local path or a public GitHub repository
  (unauthenticated for public repos; higher rate limits via a token env var).
- **Code intelligence** — language-aware parsing (Python + a generic parser),
  chunking, and a lightweight retrieval index with code-aware text utilities.
- **Agent pipeline** — typed phases with structured state: plan, engineer,
  execute, repair (bounded iterations), review, security scan, report.
- **Workspace sandboxing** — `READ_ONLY` or `WORKSPACE_EDIT` agent modes,
  protected path patterns, allow-listed tools, and optional network tool
  restrictions.
- **Live observability** — Server-Sent Events stream of run events, plus full
  plan / changes / tests / review / report artifacts per run.
- **Evaluation harness** — structured run artifacts suitable for regression
  testing agent behavior.

## Requirements

- Python 3.12+ (runtime tested against Python 3.14 on Windows)
- No database server required — SQLite via `aiosqlite` by default

## Quick start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. (Optional) configure — copy the example and adjust
cp .env.example .env

# 3. Launch the backend + dashboard
python run_server.py
```

Then open the dashboard at **http://127.0.0.1:8600**.

## Configuration

All settings are environment variables prefixed with `CODEWEAVER_` (optionally
loaded from a `.env` file in the project root — see `.env.example`). Secrets
are never stored in config: API keys are referenced by *env var name* and
resolved lazily at call time.

| Variable | Purpose |
| --- | --- |
| `CODEWEAVER_HOST` / `CODEWEAVER_PORT` | HTTP bind address (default `127.0.0.1:8600`) |
| `CODEWEAVER_DATA_DIR` | Where databases and run artifacts live |
| `CODEWEAVER_WORKSPACE_ROOT` | Root for checked-out agent workspaces |
| `CODEWEAVER_DB_URL` | Database URL (default: local SQLite) |
| `CODEWEAVER_AGENT_MODE` | `READ_ONLY` or `WORKSPACE_EDIT` |
| `CODEWEAVER_MAX_REPAIR_ITERATIONS` | Bound on automated test-failure repair loops |
| `CODEWEAVER_LLM_PROVIDER` | `deterministic`, `http_compatible`, or `none` |
| `CODEWEAVER_LLM_BASE_URL` | Chat-completions endpoint for `http_compatible` |
| `CODEWEAVER_LLM_API_KEY_ENV` | Name of the env var holding the API key |
| `CODEWEAVER_GITHUB_TOKEN_ENV` | Name of the env var holding a GitHub token |
| `CODEWEAVER_ALLOW_NETWORK_TOOLS` | Whether network-touching tools may run |

## HTTP API

A FastAPI app serves the dashboard and a JSON API, including:

- `GET  /api/health` — liveness
- `POST/GET/DELETE /api/repositories` — register and manage repositories
- `POST /api/repositories/{id}/index` — (re)build the retrieval index
- `GET  /api/repositories/{id}/search` — semantic-ish code search
- `POST /api/agent-runs` / `POST /api/tasks` — kick off an engineering run
- `GET  /api/agent-runs/{id}/stream` — live SSE event stream
- `GET  /api/agent-runs/{id}/plan|changes|tests|review|report|tools` — artifacts
- `POST /api/agent-runs/{id}/cancel` — cancel a running job

Interactive docs are served at `/docs` while the server runs.

## Project layout

```
backend/app/          Core package: agents, api, config, db, ingestion,
                      intelligence, llm, retrieval, runner, review, security,
                      tools, failure, observability, evaluation
app/                  Convenience re-export layer for backend/app/
frontend/index.html   The bundled dashboard UI
sample_repos/         Small demo repositories used as agent targets
data/                 Runtime data (databases, workspaces) — gitignored
run_server.py         One-command launcher for the backend + dashboard
```

## Demo repositories

Two tiny, intentionally imperfect sample projects are included under
`sample_repos/` (`auth_portal` and `notes_service`). Register one via the API
or the dashboard, index it, and launch an agent run to watch the full
plan → edit → test → review → report cycle.

## Deployment

CodeWeaver ships as a single container that serves both the API and the
dashboard. A `Dockerfile`, `.dockerignore`, and a `render.yaml` blueprint are
included.

### Render (free, recommended)

1. Push this repository to GitHub (it already is, if you cloned it from there).
2. Render Dashboard → **New** → **Blueprint** → select the repository.
3. Render reads `render.yaml` and configures the service automatically.
4. When prompted, paste the two secrets:
   - `CODEWEAVER_LLM_API_KEY` — your Groq key (`gsk_...`)
   - `CODEWEAVER_GITHUB_TOKEN` — your GitHub token (`ghp_...`)
5. Deploy. The dashboard is served at the service URL; health at `/api/health`.

The frontend talks to whatever origin serves it, so no URL configuration is
needed after deploy.

### Any Docker host

```bash
docker build -t codeweaver .
docker run -p 8600:8600 \
  -e CODEWEAVER_LLM_PROVIDER=http_compatible \
  -e CODEWEAVER_LLM_MODEL=qwen/qwen3.8-27b \
  -e CODEWEAVER_LLM_BASE_URL=https://api.groq.com/openai/v1 \
  -e CODEWEAVER_LLM_API_KEY=gsk_... \
  -e CODEWEAVER_GITHUB_TOKEN=ghp_... \
  -v codeweaver-data:/app/data \
  codeweaver
```

### Storage note

The agent writes databases, workspaces and repo snapshots under `/app/data`.
On hosts with an **ephemeral disk** (free tiers) that state resets on
restart/redeploy — attach a persistent volume at `/app/data` to keep it.

## License

All rights reserved. See repository history for provenance.
