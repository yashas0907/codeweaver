"""Async persistence layer for CodeWeaver.

SQLite (via aiosqlite) by default; the URL is configurable so PostgreSQL can
be swapped in later without touching call sites. All JSON payloads stored here
are pydantic-validated on write (`.model_dump_json()`) and on read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings
from app.schemas import loads_json


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RepositoryRow(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    source: Mapped[str] = mapped_column(String(32))          # local | git | github
    url: Mapped[str] = mapped_column(Text, default="")
    local_path: Mapped[str] = mapped_column(Text, default="")
    branch: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|indexed|failed
    summary_json: Mapped[str] = mapped_column(Text, default="")
    graph_json: Mapped[str] = mapped_column(Text, default="")
    last_commit: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    repo_id: Mapped[str] = mapped_column(String(32), ForeignKey("repositories.id"), index=True)
    task: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(20), default="WORKSPACE_EDIT")
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    phase: Mapped[str] = mapped_column(String(20), default="PENDING")
    error: Mapped[str] = mapped_column(Text, default="")
    report_json: Mapped[str] = mapped_column(Text, default="")
    plan_json: Mapped[str] = mapped_column(Text, default="")
    observability_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RunEventRow(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("agent_runs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    phase: Mapped[str] = mapped_column(String(20), default="PENDING")
    level: Mapped[str] = mapped_column(String(10), default="info")
    message: Mapped[str] = mapped_column(Text)
    data_json: Mapped[str] = mapped_column(Text, default="")


class FileChangeRow(Base):
    __tablename__ = "file_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("agent_runs.id"), index=True)
    path: Mapped[str] = mapped_column(Text)
    change_type: Mapped[str] = mapped_column(String(16))
    diff: Mapped[str] = mapped_column(Text, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    before_text: Mapped[str] = mapped_column(Text, default="")
    after_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TestRunRow(Base):
    __tablename__ = "test_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("agent_runs.id"), index=True)
    command: Mapped[str] = mapped_column(Text)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    all_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    cases_json: Mapped[str] = mapped_column(Text, default="")
    raw_output_tail: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    ran_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ToolExecutionRow(Base):
    __tablename__ = "tool_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("agent_runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    phase: Mapped[str] = mapped_column(String(20), default="IMPLEMENTING")
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    arguments_json: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String(32), ForeignKey("repositories.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(Text, index=True)
    language: Mapped[str] = mapped_column(String(32), default="unknown")
    chunk_type: Mapped[str] = mapped_column(String(20), default="source")
    line_start: Mapped[int] = mapped_column(Integer, default=0)
    line_end: Mapped[int] = mapped_column(Integer, default=0)
    symbol: Mapped[str] = mapped_column(String(255), default="", index=True)
    text: Mapped[str] = mapped_column(Text)
    tokens_json: Mapped[str] = mapped_column(Text, default="")


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    suite: Mapped[str] = mapped_column(String(64), default="")
    name: Mapped[str] = mapped_column(String(255))
    task_type: Mapped[str] = mapped_column(String(32))
    repo_id: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[str] = mapped_column(String(32), default="")
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Database:
    """Owns the async engine and session factory."""

    def __init__(self, url: str | None = None) -> None:
        settings = get_settings()
        self.url = url or settings.db_url
        self._ensure_sqlite_dir()
        self._engine: AsyncEngine = create_async_engine(self.url, echo=False, future=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    def _ensure_sqlite_dir(self) -> None:
        prefix = "sqlite+aiosqlite:///"
        if self.url.startswith(prefix):
            raw = self.url[len(prefix):]
            path = Path(raw if len(raw) > 2 and raw[1:2] == ":" else f"./{raw}")
            parent = path.parent if str(path.parent) not in ("", ".") else Path(".")
            parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    def session(self) -> AsyncSession:
        return self._session_factory()

    async def dispose(self) -> None:
        await self.close()


def row_to_dict(row: Any) -> dict[str, Any]:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def parse_json(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


__all__ = [
    "Base", "Database", "RepositoryRow", "AgentRunRow", "RunEventRow",
    "FileChangeRow", "TestRunRow", "ToolExecutionRow", "ChunkRow",
    "EvaluationRow", "row_to_dict", "parse_json", "loads_json", "select",
]
