"""CodeWeaver HTTP API.

Clean REST surface over the ingestion pipeline, retrieval, and the agent
run manager. Long-running agent work is always async (background tasks +
SSE streaming); nothing blocks a request for the duration of a run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from app.config import PROJECT_ROOT, get_settings
from app.db import (
    AgentRunRow, ChunkRow, Database, RepositoryRow, RunEventRow, parse_json,
)
from app.ingestion.indexer import IngestionService
from app.ingestion.sources import SourceError
from app.retrieval.index import RetrievalService
from app.runner.runner import RunManager, RunValidationError
from app.schemas import (
    AgentPhase, FinalReport, Plan, RepoGraph, RepoSummary, RunStatus,
    loads_json, utcnow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("codeweaver.api")


class AppState:
    db: Database
    ingestion: IngestionService
    retrieval: RetrievalService
    runs: RunManager


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    state.db = Database()
    await state.db.init()
    state.ingestion = IngestionService(state.db)
    state.retrieval = RetrievalService(state.db)
    state.runs = RunManager(state.db, state.retrieval)
    logger.info("CodeWeaver backend ready (provider=%s)", settings.llm_provider)
    yield
    await state.db.close()


app = FastAPI(
    title="CodeWeaver",
    description="Autonomous AI Software Engineering & GitHub Engineering Agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------------- #

class RepositoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(default="local", pattern="^(local|git|github)$")
    path: str = ""
    url: str = ""
    branch: str = ""


class TaskCreate(BaseModel):
    repo_id: str = Field(min_length=1)
    task: str = Field(min_length=3, max_length=4000)
    mode: str = Field(default="", pattern="^(|READ_ONLY|WORKSPACE_EDIT)$")


def repo_to_dict(row: RepositoryRow) -> dict:
    summary = loads_json(RepoSummary, row.summary_json)
    return {
        "id": row.id,
        "name": row.name,
        "source": row.source,
        "url": row.url,
        "branch": row.branch,
        "status": row.status,
        "last_commit": row.last_commit,
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "summary": summary.model_dump(mode="json") if summary else None,
    }


def run_to_dict(row: AgentRunRow) -> dict:
    return {
        "id": row.id,
        "run_id": row.id,
        "repo_id": row.repo_id,
        "task": row.task,
        "mode": row.mode,
        "status": row.status,
        "phase": row.phase,
        "current_phase": row.phase,
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


_PHASE_WEIGHTS = {
    "QUEUED": 2, "ANALYZING": 15, "RETRIEVING": 28, "PLANNING": 42,
    "IMPLEMENTING": 60, "TESTING": 80, "DEBUGGING": 88, "REVIEWING": 94,
    "COMPLETED": 100, "FAILED": 100, "CANCELLED": 100,
}


def _progress_for(row: AgentRunRow) -> int:
    return _PHASE_WEIGHTS.get((row.phase or "QUEUED").upper(), 5)


async def _repo_name(repo_id: str) -> str:
    if not repo_id:
        return ""
    async with state.db.session() as session:
        row = await session.get(RepositoryRow, repo_id)
    return row.name if row else ""


async def _get_repo(repo_id: str) -> RepositoryRow:
    async with state.db.session() as session:
        row = await session.get(RepositoryRow, repo_id)
    if row is None:
        raise HTTPException(404, f"repository not found: {repo_id}")
    return row


async def _get_run(run_id: str) -> AgentRunRow:
    async with state.db.session() as session:
        row = await session.get(AgentRunRow, run_id)
    if row is None:
        raise HTTPException(404, f"agent run not found: {run_id}")
    return row


# --------------------------------------------------------------------------- #
# Health & repositories
# --------------------------------------------------------------------------- #

@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": "1.0.0",
        "agent_mode": settings.agent_mode,
        "llm_provider": settings.llm_provider,
        "llm_available": settings.llm_provider != "none",
        "time": utcnow().isoformat(),
    }


@app.post("/api/repositories", status_code=201)
async def create_repository(payload: RepositoryCreate) -> dict:
    """Ingest + index a repository. Synchronous because indexing is fast and
    callers need the id immediately; source acquisition errors become 400s.
    Returns existing repository if one with the same name is already indexed."""
    # Return existing repo if same name already exists
    async with state.db.session() as session:
        existing = (await session.execute(
            select(RepositoryRow).where(RepositoryRow.name == payload.name)
        )).scalars().first()
    if existing:
        return repo_to_dict(existing)

    events: list[tuple[str, str]] = []

    def progress(stage: str, message: str) -> None:
        events.append((stage, message))
        logger.info("[ingest %s] %s: %s", payload.name, stage, message)

    try:
        repo_id = await state.ingestion.ingest(
            payload.name,
            {"source": payload.source, "path": payload.path, "url": payload.url, "branch": payload.branch},
            progress=progress,
        )
    except SourceError as exc:
        raise HTTPException(400, str(exc)) from exc
    row = await _get_repo(repo_id)
    return repo_to_dict(row)


@app.get("/api/repositories")
async def list_repositories() -> list[dict]:
    async with state.db.session() as session:
        rows = (await session.execute(select(RepositoryRow).order_by(RepositoryRow.created_at.desc()))).scalars().all()
    return [repo_to_dict(r) for r in rows]


@app.get("/api/repositories/{repo_id}")
async def get_repository(repo_id: str) -> dict:
    return repo_to_dict(await _get_repo(repo_id))


@app.delete("/api/repositories/{repo_id}")
async def delete_repository(repo_id: str) -> dict:
    ok = await state.ingestion.delete_repository(repo_id)
    state.retrieval.invalidate(repo_id)
    if not ok:
        raise HTTPException(404, f"repository not found: {repo_id}")
    return {"deleted": repo_id}


@app.post("/api/repositories/{repo_id}/index")
async def reindex_repository(repo_id: str) -> dict:
    await _get_repo(repo_id)
    try:
        await state.ingestion.reindex(repo_id)
    except SourceError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.retrieval.invalidate(repo_id)
    return repo_to_dict(await _get_repo(repo_id))


@app.get("/api/repositories/{repo_id}/map")
async def repository_map(repo_id: str) -> dict:
    row = await _get_repo(repo_id)
    graph = loads_json(RepoGraph, row.graph_json)
    if graph is None:
        raise HTTPException(400, "repository has no map (index it first)")
    return graph.model_dump(mode="json")


@app.get("/api/repositories/{repo_id}/search")
async def repository_search(repo_id: str, q: str = Query(min_length=2), top_k: int = 10) -> dict:
    row = await _get_repo(repo_id)
    graph = loads_json(RepoGraph, row.graph_json)
    context = await state.retrieval.retrieve(repo_id, q, graph=graph, top_k=top_k)
    return {
        "queries": context.queries,
        "results": [
            {
                "chunk_id": c.chunk_id, "path": c.path, "language": c.language,
                "chunk_type": c.chunk_type, "line_start": c.line_start,
                "line_end": c.line_end, "symbol": c.symbol, "text": c.text[:1500],
                "provenance": p.model_dump(),
            }
            for c, p in zip(context.items, context.provenance)
        ],
    }


# --------------------------------------------------------------------------- #
# Agent runs
# --------------------------------------------------------------------------- #

@app.post("/api/agent-runs", status_code=202)
async def create_agent_run(payload: TaskCreate) -> dict:
    try:
        return await state.runs.start_run(payload.repo_id, payload.task, payload.mode)
    except RunValidationError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/tasks", status_code=202)
async def create_task(payload: TaskCreate) -> dict:
    """Alias of POST /api/agent-runs (spec endpoint name)."""
    return await create_agent_run(payload)


@app.get("/api/agent-runs")
async def list_agent_runs(repo_id: str = "") -> list[dict]:
    async with state.db.session() as session:
        stmt = select(AgentRunRow).order_by(AgentRunRow.created_at.desc()).limit(100)
        if repo_id:
            stmt = stmt.where(AgentRunRow.repo_id == repo_id)
        rows = (await session.execute(stmt)).scalars().all()
        repo_rows = {}
        if rows:
            rid = [r.repo_id for r in rows if r.repo_id]
            if rid:
                rr = (await session.execute(
                    select(RepositoryRow).where(RepositoryRow.id.in_(rid))
                )).scalars().all()
                repo_rows = {r.id: r.name for r in rr}
    out = []
    for r in rows:
        d = run_to_dict(r)
        d["repo_name"] = repo_rows.get(r.repo_id, "")
        d["progress"] = _progress_for(r)
        out.append(d)
    return out


@app.get("/api/agent-runs/{run_id}")
async def get_agent_run(run_id: str) -> dict:
    row = await _get_run(run_id)
    data = run_to_dict(row)
    plan = loads_json(Plan, row.plan_json)
    data["plan_available"] = plan is not None
    data["report_available"] = bool(row.report_json)
    data["repo_name"] = await _repo_name(row.repo_id)
    data["progress"] = _progress_for(row)
    return data


@app.get("/api/tasks/{run_id}")
async def get_task(run_id: str) -> dict:
    return await get_agent_run(run_id)


@app.get("/api/agent-runs/{run_id}/events")
async def get_run_events(run_id: str) -> list[dict]:
    await _get_run(run_id)
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(RunEventRow).where(RunEventRow.run_id == run_id).order_by(RunEventRow.id)
            )
        ).scalars().all()
    return [
        {
            "ts": r.ts.isoformat() if r.ts else None,
            "phase": r.phase,
            "level": r.level,
            "message": r.message,
            "data": parse_json(r.data_json),
        }
        for r in rows
    ]


@app.get("/api/agent-runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE stream of live agent events; replays history first."""
    row = await _get_run(run_id)
    start_index = 0

    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(RunEventRow).where(RunEventRow.run_id == run_id).order_by(RunEventRow.id)
            )
        ).scalars().all()
    history = [
        {
            "ts": r.ts.isoformat() if r.ts else None, "phase": r.phase,
            "level": r.level, "message": r.message, "data": parse_json(r.data_json),
        }
        for r in rows
    ]
    start_index = len(history)
    terminal = row.status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value)

    async def event_stream() -> AsyncIterator[str]:
        for item in history:
            yield f"data: {json.dumps(item)}\n\n"
        if terminal:
            yield "event: done\ndata: {}\n\n"
            return
        queue = state.runs.subscribe(run_id)
        try:
            idle = 0.0
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    idle = 0.0
                    yield f"data: {json.dumps(event.model_dump(mode='json', exclude_none=True))}\n\n"
                    if event.phase in (AgentPhase.COMPLETED, AgentPhase.FAILED):
                        yield "event: done\ndata: {}\n\n"
                        return
                except asyncio.TimeoutError:
                    idle += 1.0
                    if idle % 15 == 0:
                        yield ": keep-alive\n\n"
                    # Poll DB as fallback in case events bypass the queue.
                    async with state.db.session() as session:
                        fresh = await session.get(AgentRunRow, run_id)
                    if fresh is not None and fresh.status in (
                        RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value,
                    ):
                        yield "event: done\ndata: {}\n\n"
                        return
        finally:
            state.runs.unsubscribe(run_id, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/agent-runs/{run_id}/plan")
async def get_run_plan(run_id: str) -> dict:
    row = await _get_run(run_id)
    plan = loads_json(Plan, row.plan_json)
    if plan is None:
        raise HTTPException(404, "plan not generated yet")
    return plan.model_dump(mode="json")


@app.get("/api/agent-runs/{run_id}/changes")
async def get_run_changes(run_id: str) -> list[dict]:
    from app.db import FileChangeRow

    await _get_run(run_id)
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(FileChangeRow).where(FileChangeRow.run_id == run_id).order_by(FileChangeRow.id)
            )
        ).scalars().all()
    return [
        {
            "path": r.path, "change_type": r.change_type, "diff": r.diff,
            "description": r.description,
            "before_text": r.before_text, "after_text": r.after_text,
        }
        for r in rows
    ]


@app.get("/api/agent-runs/{run_id}/tests")
async def get_run_tests(run_id: str) -> list[dict]:
    from app.db import TestRunRow

    await _get_run(run_id)
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(TestRunRow).where(TestRunRow.run_id == run_id).order_by(TestRunRow.id)
            )
        ).scalars().all()
    return [
        {
            "command": r.command, "exit_code": r.exit_code, "passed": r.passed,
            "failed": r.failed, "errors": r.errors, "skipped": r.skipped,
            "all_passed": r.all_passed, "cases": parse_json(r.cases_json) or [],
            "raw_output_tail": r.raw_output_tail, "duration_ms": r.duration_ms,
            "ran_at": r.ran_at.isoformat() if r.ran_at else None,
        }
        for r in rows
    ]


@app.get("/api/agent-runs/{run_id}/review")
async def get_run_review(run_id: str) -> dict:
    row = await _get_run(run_id)
    report = loads_json(FinalReport, row.report_json)
    if report is None or (report.review is None and report.security is None):
        raise HTTPException(404, "review not available for this run")
    return {
        "review": report.review.model_dump(mode="json") if report.review else None,
        "security": report.security.model_dump(mode="json") if report.security else None,
    }


@app.get("/api/agent-runs/{run_id}/report")
async def get_run_report(run_id: str, format: str = "json") -> Any:
    row = await _get_run(run_id)
    report = loads_json(FinalReport, row.report_json)
    if report is None:
        raise HTTPException(404, "report not generated yet")
    if format == "markdown":
        return JSONResponse(
            content={"markdown": report.to_markdown()},
            headers={"Content-Type": "application/json"},
        )
    return report.model_dump(mode="json")


@app.get("/api/agent-runs/{run_id}/tools")
async def get_run_tools(run_id: str) -> list[dict]:
    from app.db import ToolExecutionRow

    await _get_run(run_id)
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(ToolExecutionRow).where(ToolExecutionRow.run_id == run_id).order_by(ToolExecutionRow.id)
            )
        ).scalars().all()
    return [
        {
            "name": r.name, "ok": r.ok, "arguments": parse_json(r.arguments_json),
            "output": r.output[:4000], "error": r.error, "duration_ms": r.duration_ms,
            "started_at": r.started_at.isoformat() if r.started_at else None,
        }
        for r in rows
    ]


@app.post("/api/agent-runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    await _get_run(run_id)
    ok = await state.runs.cancel(run_id)
    if not ok:
        raise HTTPException(409, "run is not active")
    return {"cancelled": run_id}


class GitHubPushRequest(BaseModel):
    repo_url: str = Field(description="GitHub repo URL e.g. https://github.com/user/repo or owner/repo")
    commit_message: str = Field(default="fix: applied by CodeWeaver autonomous agent")
    pr_title: str = Field(default="")
    pr_description: str = Field(default="")
    base_branch: str = Field(default="main")


@app.post("/api/agent-runs/{run_id}/push", status_code=202)
async def push_run_to_github(run_id: str, body: GitHubPushRequest) -> dict:
    """Push the workspace changes from this run to GitHub as a Pull Request.

    Requires CODEWEAVER_GITHUB_TOKEN env var to be set (ghp_... token).
    The token needs repo scope for private repos, public_repo scope for public.
    """
    import os

    from app.integration.github import GitHubPushError, push_workspace_to_pr

    await _get_run(run_id)
    settings = get_settings()
    workspace_path = Path(settings.workspace_root) / run_id
    if not workspace_path.is_dir():
        raise HTTPException(
            400, "no workspace path for this run — run the agent first in WORKSPACE_EDIT mode"
        )

    # Resolve the token from the configured env var / .env file at call time.
    token_name = settings.github_token_env
    token = settings.resolve_github_token() or ""
    if not token:
        raise HTTPException(
            400,
            f"GitHub token not set. Set the {token_name} environment variable "
            "with a GitHub personal access token (ghp_...).",
        )

    # Only the files this run actually changed are pushed, and each edit is
    # re-applied onto the current remote file — never a tree-wide sync (a
    # workspace snapshot can legitimately differ from the target branch, and a
    # blind sync would rewrite or delete real code).
    from app.db import FileChangeRow

    async with state.db.session() as session:
        change_rows = (
            await session.execute(select(FileChangeRow).where(FileChangeRow.run_id == run_id))
        ).scalars().all()
    change_records = [
        {
            "path": r.path,
            "change_type": r.change_type,
            "before_text": r.before_text or "",
            "after_text": r.after_text or "",
        }
        for r in change_rows
    ]
    if not change_records:
        raise HTTPException(
            400,
            "this run made no file changes — nothing to push. Run the agent with a task "
            "that edits files (WORKSPACE_EDIT mode) first.",
        )

    # Blocking HTTP work runs in a thread so the event loop stays responsive.
    try:
        result = await asyncio.to_thread(
            push_workspace_to_pr,
            workspace_path=workspace_path,
            repo_url=body.repo_url,
            token=token,
            commit_message=body.commit_message,
            pr_title=body.pr_title or f"CodeWeaver: {body.commit_message}",
            pr_body=body.pr_description or f"Applied by CodeWeaver autonomous agent.\n\nRun ID: {run_id}",
            base_branch=body.base_branch,
            change_records=change_records,
        )
        return result
    except GitHubPushError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"GitHub push failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if (FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

if (FRONTEND_DIR / "index.html").is_file():
    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.exception_handler(SourceError)
async def source_error_handler(_, exc: SourceError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
