"""Repository ingestion & indexing pipeline.

Pipeline stages:
  1. acquire a snapshot (local copy / git clone / GitHub tarball)
  2. walk + classify files
  3. build repository intelligence (AST symbols, dependency graph, summary)
  4. build retrieval chunks (AST-aware for Python, windowed for others)
  5. persist everything to the database

Each stage emits progress via an optional callback so the API layer can log it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import get_settings, new_run_id
from app.db import ChunkRow, Database, RepositoryRow
from app.ingestion.sources import RepoSource, SourceError, create_source
from app.ingestion.walker import walk_repository
from app.intelligence.analyzer import RepositoryAnalyzer
from app.intelligence.chunks import build_chunks
from app.schemas import RepoGraph, RepoSummary, utcnow

ProgressFn = Callable[[str, str], None]  # (stage, message)


class IngestionService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.settings = get_settings()
        self.store_root = self.settings.data_dir / "repo_store"
        self.analyzer = RepositoryAnalyzer()

    def snapshot_path(self, repo_id: str) -> Path:
        return self.store_root / repo_id

    async def ingest(
        self,
        name: str,
        spec: dict,
        progress: ProgressFn | None = None,
    ) -> str:
        """Ingest a repository; returns the new repo id.

        Raises SourceError when acquisition fails (caller maps to HTTP 400).
        """
        emit = progress or (lambda stage, msg: None)
        repo_id = new_run_id("repo")
        emit("acquire", f"resolving source: {spec.get('source', 'local')}")

        source: RepoSource = create_source(spec, self.store_root)
        snapshot = source.acquire(repo_id)
        emit("acquire", f"snapshot ready at {snapshot.name} ({source.describe()})")

        emit("walk", "scanning repository files")
        files, notes = walk_repository(snapshot)
        if not files:
            raise SourceError("repository appears empty (no indexable files)")
        emit("walk", f"indexed {len(files)} files ({len(notes)} skipped)")

        emit("analyze", "building repository graph (AST analysis)")
        graph, summary = self.analyzer.analyze(snapshot, name, files, notes)
        emit("analyze", (
            f"{graph.stats.get('total_symbols', 0)} symbols, "
            f"{len(graph.edges)} dependency edges, "
            f"languages: {', '.join(sorted(graph.languages))}"
        ))

        emit("index", "building retrieval chunks")
        chunks = build_chunks(repo_id, snapshot, files, graph)
        emit("index", f"created {len(chunks)} retrieval chunks")

        last_commit = _head_commit(snapshot)
        async with self.db.session() as session:
            session.add(
                RepositoryRow(
                    id=repo_id,
                    name=name,
                    source=spec.get("source", "local"),
                    url=spec.get("url", "") or spec.get("path", ""),
                    local_path=str(snapshot),
                    branch=spec.get("branch", ""),
                    status="indexed",
                    summary_json=summary.model_dump_json(),
                    graph_json=graph.model_dump_json(),
                    last_commit=last_commit,
                    error="",
                )
            )
            for batch in _batched(chunks, 200):
                session.add_all(
                    [
                        ChunkRow(
                            repo_id=repo_id,
                            chunk_id=c.chunk_id,
                            path=c.path,
                            language=c.language,
                            chunk_type=c.chunk_type,
                            line_start=c.line_start,
                            line_end=c.line_end,
                            symbol=c.symbol,
                            text=c.text,
                            tokens_json="",
                        )
                        for c in batch
                    ]
                )
            await session.commit()
        emit("done", "repository ingestion complete")
        return repo_id

    async def reindex(self, repo_id: str, progress: ProgressFn | None = None) -> None:
        """Re-run walking/analysis/chunking for an existing repository."""
        row = await self.get_repo_row(repo_id)
        if row is None:
            raise LookupError(f"unknown repository: {repo_id}")
        snapshot = Path(row.local_path)
        if not snapshot.is_dir():
            raise SourceError(f"repository snapshot missing on disk: {snapshot}")
        emit = progress or (lambda stage, msg: None)

        emit("walk", "re-scanning repository files")
        files, notes = walk_repository(snapshot)
        if not files:
            raise SourceError("repository appears empty")

        graph, summary = self.analyzer.analyze(snapshot, row.name, files, notes)
        chunks = build_chunks(repo_id, snapshot, files, graph)

        async with self.db.session() as session:
            fresh = await session.get(RepositoryRow, repo_id)
            if fresh is not None:
                fresh.status = "indexed"
                fresh.summary_json = summary.model_dump_json()
                fresh.graph_json = graph.model_dump_json()
                fresh.updated_at = datetime.now(timezone.utc)
                fresh.last_commit = _head_commit(snapshot)
            # Replace old chunks
            from sqlalchemy import delete

            await session.execute(delete(ChunkRow).where(ChunkRow.repo_id == repo_id))
            for batch in _batched(chunks, 200):
                session.add_all(
                    [
                        ChunkRow(
                            repo_id=repo_id, chunk_id=c.chunk_id, path=c.path,
                            language=c.language, chunk_type=c.chunk_type,
                            line_start=c.line_start, line_end=c.line_end,
                            symbol=c.symbol, text=c.text, tokens_json="",
                        )
                        for c in batch
                    ]
                )
            await session.commit()
        emit("done", "re-index complete")

    async def get_repo_row(self, repo_id: str) -> RepositoryRow | None:
        async with self.db.session() as session:
            return await session.get(RepositoryRow, repo_id)

    async def delete_repository(self, repo_id: str) -> bool:
        from sqlalchemy import delete

        async with self.db.session() as session:
            row = await session.get(RepositoryRow, repo_id)
            if row is None:
                return False
            await session.execute(delete(ChunkRow).where(ChunkRow.repo_id == repo_id))
            await session.delete(row)
            await session.commit()
        snapshot = self.snapshot_path(repo_id)
        if snapshot.is_dir():
            import shutil

            shutil.rmtree(snapshot, ignore_errors=True)
        return True


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _head_commit(snapshot: Path) -> str:
    import subprocess

    git_dir = snapshot / ".git"
    if not git_dir.exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=snapshot,
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()[:12]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""
