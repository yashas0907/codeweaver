"""Hybrid code retrieval: BM25 + TF-IDF cosine + symbol/path boosting +
repository-graph expansion, with full provenance on every item.

Design goals (per spec): never dump whole files into a prompt; return the
smallest set of high-precision chunks, each annotated with FILE, LINE RANGE,
SYMBOL, and WHY it was retrieved.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from app.retrieval.textutils import keywords, looks_like_path, tokenize
from app.schemas import Chunk, Provenance, RepoGraph, RetrievedContext

K1 = 1.4
B = 0.72
TOP_K = 22
MAX_PER_FILE = 3
GRAPH_EXPAND_LIMIT = 6


def _norm(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 1.0 if value > 0 else 0.0
    return (value - lo) / (hi - lo)


class RepoIndex:
    """In-memory hybrid index over the chunks of one repository."""

    def __init__(self, repo_id: str, chunks: list[Chunk]) -> None:
        self.repo_id = repo_id
        self.chunks = chunks
        self._tokens: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._symbol_map: dict[str, list[int]] = defaultdict(list)
        self._path_tokens: list[set[str]] = []
        self._avg_len = 0.0
        self._build()

    def _build(self) -> None:
        for i, chunk in enumerate(self.chunks):
            toks = tokenize(chunk.text)
            self._tokens.append(toks)
            tf = Counter(toks)
            self._tf.append(tf)
            self._doc_len.append(len(toks))
            for term in set(toks):
                self._df[term] += 1
            # symbol indexing: full qualname + simple name
            if chunk.symbol:
                self._symbol_map[chunk.symbol.lower()].append(i)
                simple = chunk.symbol.split("::")[-1].split(".")[-1].lower()
                self._symbol_map[simple].append(i)
            path = PurePosixPath(chunk.path)
            self._path_tokens.append({t.lower() for t in path.parts if t != path.name} | {path.name.lower(), path.suffix.lower()})

        total = sum(self._doc_len)
        self._avg_len = (total / len(self.chunks)) if self.chunks else 0.0

    # ------------------------------------------------------------------ #

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        if n == 0 or df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.chunks)
        for i, tf in enumerate(self._tf):
            dl = self._doc_len[i] or 1
            for term in query_tokens:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf(term)
                denom = f + K1 * (1 - B + B * dl / (self._avg_len or 1))
                scores[i] += idf * (f * (K1 + 1)) / denom
        return scores

    def _vector(self, i: int) -> dict[str, float]:
        n = len(self.chunks) or 1
        vec: dict[str, float] = {}
        for term, f in self._tf[i].items():
            idf = self._idf(term)
            if idf <= 0:
                continue
            vec[term] = (1 + math.log(f)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _vector_scores(self, query_tokens: list[str]) -> list[float]:
        q_counter = Counter(query_tokens)
        q_vec: dict[str, float] = {}
        for term, f in q_counter.items():
            idf = self._idf(term)
            if idf <= 0:
                continue
            q_vec[term] = (1 + math.log(f)) * idf
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / norm for t, v in q_vec.items()}
        if not q_vec:
            return [0.0] * len(self.chunks)
        scores: list[float] = []
        for i in range(len(self.chunks)):
            v = self._vector(i)
            small, big = (q_vec, v) if len(q_vec) < len(v) else (v, q_vec)
            dot = sum(val * big.get(t, 0.0) for t, val in small.items())
            scores.append(dot)
        return scores

    def _symbol_scores(self, query_tokens: list[str], raw_query: str) -> list[float]:
        scores = [0.0] * len(self.chunks)
        raw_lower = raw_query.lower()
        for sym, idxs in self._symbol_map.items():
            hit = sym in raw_lower or sym in query_tokens
            if not hit:
                sym_kws = set(keywords(sym.replace("::", " ").replace(".", " "), limit=6))
                if len(sym_kws) >= 2 and sym_kws.issubset(set(query_tokens)):
                    hit = True
            if hit:
                for i in idxs:
                    scores[i] = max(scores[i], 1.0)
        return scores

    def _path_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.chunks)
        for i, path_tokens in enumerate(self._path_tokens):
            overlap = len(set(query_tokens) & path_tokens)
            if overlap:
                scores[i] = min(1.0, 0.4 + 0.2 * overlap)
        return scores

    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float, str]]:
        q_tokens = tokenize(query) + keywords(query, limit=14)
        if not q_tokens:
            return []
        bm25 = self._bm25_scores(q_tokens)
        vec = self._vector_scores(q_tokens)
        sym = self._symbol_scores(q_tokens, query)
        path = self._path_scores(q_tokens)

        bm25_n = _norm_list(bm25)
        vec_n = _norm_list(vec)
        fused = [
            0.42 * bm25_n[i] + 0.23 * vec_n[i] + 0.20 * sym[i] + 0.15 * path[i]
            for i in range(len(self.chunks))
        ]
        ranked = sorted(range(len(self.chunks)), key=lambda i: -fused[i])

        results: list[tuple[Chunk, float, str]] = []
        per_file: Counter[str] = Counter()
        for i in ranked:
            if len(results) >= top_k:
                break
            chunk = self.chunks[i]
            if fused[i] <= 0.02:
                break
            if per_file[chunk.path] >= MAX_PER_FILE:
                continue
            reasons = []
            if sym[i] > 0:
                reasons.append("symbol match")
            if path[i] > 0:
                reasons.append("path relevance")
            if bm25_n[i] >= 0.5:
                reasons.append("lexical match")
            elif vec_n[i] >= 0.4:
                reasons.append("term similarity")
            if not reasons:
                reasons.append("combined score")
            per_file[chunk.path] += 1
            results.append((chunk, fused[i], " + ".join(reasons)))
        return results

    def by_symbol(self, name: str) -> list[tuple[Chunk, float, str]]:
        name = name.lower()
        out: list[tuple[Chunk, float, str]] = []
        for sym, idxs in self._symbol_map.items():
            if name in sym:
                for i in idxs:
                    out.append((self.chunks[i], 1.0, f"symbol lookup: {self.chunks[i].symbol}"))
        return out

    def by_path(self, path: str) -> list[tuple[Chunk, float, str]]:
        out: list[tuple[Chunk, float, str]] = []
        for i, chunk in enumerate(self.chunks):
            if chunk.path == path or chunk.path.startswith(path):
                out.append((chunk, 1.0, "path lookup"))
        return out

    def file_chunks(self, path: str) -> list[Chunk]:
        return [c for c in self.chunks if c.path == path]


def _norm_list(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - lo) / (hi - lo) for v in values]


_PATH_TOKEN_RE = re.compile(r"[\w./-]+")


def extract_path_hints(task: str) -> list[str]:
    hints: list[str] = []
    for m in _PATH_TOKEN_RE.finditer(task):
        token = m.group(0)
        if looks_like_path(token) or token.endswith((".py", ".ts", ".js", ".go", ".md")):
            hints.append(token.strip("`'\""))
    return hints


class RetrievalService:
    """DB-backed retrieval with per-repository index caching."""

    def __init__(self, db) -> None:
        self.db = db
        self._cache: dict[str, RepoIndex] = {}

    def invalidate(self, repo_id: str) -> None:
        self._cache.pop(repo_id, None)

    async def get_index(self, repo_id: str) -> RepoIndex:
        if repo_id in self._cache:
            return self._cache[repo_id]
        from sqlalchemy import select

        from app.db import ChunkRow

        async with self.db.session() as session:
            rows = (await session.execute(select(ChunkRow).where(ChunkRow.repo_id == repo_id))).scalars().all()
        chunks = [
            Chunk(
                chunk_id=r.chunk_id, repo_id=r.repo_id, path=r.path,
                language=r.language, chunk_type=r.chunk_type,
                line_start=r.line_start, line_end=r.line_end,
                symbol=r.symbol, text=r.text,
            )
            for r in rows
        ]
        index = RepoIndex(repo_id, chunks)
        self._cache[repo_id] = index
        return index

    async def retrieve(
        self,
        repo_id: str,
        task: str,
        graph: RepoGraph | None = None,
        extra_queries: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> RetrievedContext:
        index = await self.get_index(repo_id)

        queries = [task] + (extra_queries or [])
        merged: dict[str, tuple[Chunk, float, str]] = {}
        for q in queries:
            for chunk, score, reason in index.search(q, top_k=top_k):
                key = chunk.chunk_id
                if key not in merged or merged[key][1] < score:
                    merged[key] = (chunk, score, reason)

        # path hints from the task get direct lookups
        for hint in extract_path_hints(task):
            for chunk, score, reason in index.by_path(hint):
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = (chunk, max(score, 0.9), reason)

        items = sorted(merged.values(), key=lambda t: -t[1])[:top_k]

        # Graph expansion: imports of the top files, and their tests.
        if graph is not None and items:
            top_paths = {c.path for c, _, _ in items[:GRAPH_EXPAND_LIMIT]}
            related: set[str] = set()
            for path in top_paths:
                related.update(graph.imports_of(path))
                related.update(graph.tests_for(path))
                for edge in graph.edges:
                    if edge.dst == path and edge.kind == "imports":
                        related.add(edge.src)
            related -= top_paths
            additions = 0
            for rel in sorted(related):
                if additions >= 6:
                    break
                for chunk in index.file_chunks(rel)[:2]:
                    key = chunk.chunk_id
                    if key not in merged:
                        merged[key] = (chunk, 0.35, "repository graph: related via imports/tests")
                        additions += 1

        items = sorted(merged.values(), key=lambda t: -t[1])[:top_k]
        context = RetrievedContext(queries=queries)
        for chunk, score, reason in items:
            context.items.append(chunk)
            context.provenance.append(chunk.provenance(reason, score))
            context.total_chars += len(chunk.text)
        return context
