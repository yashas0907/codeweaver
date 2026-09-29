"""Async agent-run manager.

Runs execute as background asyncio tasks (HTTP requests return immediately).
Live updates flow through per-run subscriber queues consumed by the SSE
endpoint; late subscribers replay persisted events from the database first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.pipeline import CodeWeaverPipeline
from app.config import get_settings, new_run_id
from app.db import AgentRunRow, Database
from app.schemas import AgentEvent, AgentMode, AgentPhase, RunStatus, utcnow

logger = logging.getLogger("codeweaver.runner")


class RunValidationError(RuntimeError):
    pass


class RunManager:
    def __init__(self, db: Database, retrieval) -> None:
        self.db = db
        self.retrieval = retrieval
        self.settings = get_settings()
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ #

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        queues = self._subscribers.get(run_id)
        if queues:
            queues.discard(queue)
            if not queues:
                self._subscribers.pop(run_id, None)

    async def _broadcast(self, event: AgentEvent) -> None:
        for queue in list(self._subscribers.get(event.run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE queue overflow for run %s", event.run_id)

    # ------------------------------------------------------------------ #

    async def start_run(self, repo_id: str, task: str, mode: str = "") -> dict[str, Any]:
        task = (task or "").strip()
        if not task:
            raise RunValidationError("task must not be empty")

        from app.db import RepositoryRow

        async with self.db.session() as session:
            repo = await session.get(RepositoryRow, repo_id)
        if repo is None:
            raise RunValidationError(f"unknown repository: {repo_id}")
        if repo.status != "indexed":
            raise RunValidationError(f"repository {repo_id} is not indexed (status={repo.status})")

        try:
            agent_mode = AgentMode((mode or self.settings.agent_mode).upper())
        except ValueError:
            raise RunValidationError(f"invalid mode: {mode!r} (use READ_ONLY or WORKSPACE_EDIT)")

        run_id = new_run_id("run")
        async with self.db.session() as session:
            session.add(AgentRunRow(
                id=run_id, repo_id=repo_id, task=task, mode=agent_mode.value,
                status=RunStatus.PENDING.value, phase=AgentPhase.PENDING.value,
            ))
            await session.commit()

        coro = self._execute(run_id, repo_id, repo.name, repo.local_path, task, agent_mode)
        self._tasks[run_id] = asyncio.create_task(coro, name=run_id)
        return {"run_id": run_id, "repo_id": repo_id, "task": task, "mode": agent_mode.value}

    async def _execute(self, run_id: str, repo_id: str, repo_name: str, local_path: str, task: str, mode: AgentMode) -> None:
        try:
            from app.db import RepositoryRow

            async with self.db.session() as session:
                repo_row = await session.get(RepositoryRow, repo_id)
            pipeline = CodeWeaverPipeline(self.db, self.retrieval)
            await pipeline.execute(run_id, repo_row, task, mode, event_sink=self._broadcast)
        except Exception:
            logger.exception("background run %s crashed", run_id)
        finally:
            self._tasks.pop(run_id, None)

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    async def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        async with self.db.session() as session:
            row = await session.get(AgentRunRow, run_id)
            if row is not None:
                row.status = RunStatus.CANCELLED.value
                row.finished_at = utcnow()
                await session.commit()
        return True
