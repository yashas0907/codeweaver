"""Builds retrieval chunks from a repository snapshot.

Python files are chunked per AST symbol (function/class/method bodies keep
exact line ranges); other text files use overlapping line windows. Every
chunk embeds its path/symbol header so lexical search benefits from
provenance-aware text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import SOURCE_EXTENSIONS
from app.ingestion.walker import RepoFile
from app.intelligence.python_parser import PythonFileAnalyzer
from app.schemas import Chunk, RepoGraph

MAX_CHUNK_CHARS = 1_800
WINDOW_LINES = 60
OVERLAP_LINES = 10


def build_chunks(repo_id: str, snapshot: Path, files: list[RepoFile], graph: RepoGraph) -> list[Chunk]:
    chunks: list[Chunk] = []
    py_analyzer = PythonFileAnalyzer()
    seq = 0

    def add(path: str, language: str, chunk_type: str, line_start: int, line_end: int, symbol: str, text: str) -> None:
        nonlocal seq
        text = text[:MAX_CHUNK_CHARS * 2]
        if not text.strip():
            return
        header = f"# file: {path}"
        if symbol:
            header += f"  symbol: {symbol}"
        if line_start:
            header += f"  lines: {line_start}-{line_end}"
        body = f"{header}\n{text}"
        cid = f"{repo_id[:8]}-{hashlib.sha1(f'{path}:{symbol}:{line_start}'.encode()).hexdigest()[:10]}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                repo_id=repo_id,
                path=path,
                language=language,
                chunk_type=chunk_type,
                line_start=line_start,
                line_end=line_end,
                symbol=symbol,
                text=body,
            )
        )
        seq += 1

    for f in files:
        full = snapshot / f.path
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()

        if f.language == "python" and f.role in ("source", "test"):
            facts = py_analyzer.analyze(f.path, text)
            if facts.symbols and not facts.syntax_error:
                covered: list[tuple[int, int]] = []
                for ps in facts.symbols:
                    start = max(1, ps.line_start)
                    end = min(len(lines), ps.line_end or ps.line_start)
                    body = "\n".join(lines[start - 1:end])
                    kind = "test" if ps.kind == "test" else "symbol"
                    qualname = f"{ps.parent}.{ps.name}" if ps.parent else ps.name
                    add(f.path, "python", kind, start, end, qualname, body)
                    covered.append((start, end))
                # Module-level preamble (imports, constants) if meaningful.
                if covered:
                    first = min(c[0] for c in covered)
                    preamble_end = max(1, first - 1)
                    if preamble_end > 1:
                        add(f.path, "python", "source", 1, preamble_end, "", "\n".join(lines[:preamble_end]))
            else:
                _windowed(add, f.path, f.language, lines, "source")
        elif f.role in ("doc", "readme") or f.path.lower().startswith("readme"):
            _windowed(add, f.path, f.language, lines, "doc", window=120, overlap=20)
        elif f.role == "test":
            _windowed(add, f.path, f.language, lines, "test")
        elif f.role in ("config", "manifest", "ci") and len(lines) <= 200:
            add(f.path, f.language, "config", 1, len(lines), "", text)
        elif f.role == "source" and f.extension.lower() in SOURCE_EXTENSIONS:
            _windowed(add, f.path, f.language, lines, "source")
        elif f.role == "doc":
            _windowed(add, f.path, f.language, lines, "doc", window=120, overlap=20)

    return chunks


def _windowed(add, path: str, language: str, lines: list[str], chunk_type: str, window: int = WINDOW_LINES, overlap: int = OVERLAP_LINES) -> None:
    if len(lines) <= window:
        add(path, language, chunk_type, 1, len(lines), "", "\n".join(lines))
        return
    step = max(1, window - overlap)
    idx = 0
    while idx < len(lines):
        end = min(idx + window, len(lines))
        add(path, language, chunk_type, idx + 1, end, "", "\n".join(lines[idx:end]))
        if end >= len(lines):
            break
        idx += step
