"""RepositoryAnalyzer: builds the repository graph and engineering summary.

Combines Python AST facts with generic per-language parsing into:
  - RepoGraph: files, symbols, import/test/call edges
  - RepoSummary: entry points, routes, test frameworks, build commands,
    observations — consumed by the planner and analyst prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import SOURCE_EXTENSIONS
from app.ingestion.walker import RepoFile
from app.intelligence.generic_parser import parse_generic
from app.intelligence.python_parser import PythonFileAnalyzer
from app.schemas import DependencyEdge, RepoFile, RepoGraph, RepoSummary, Symbol


class RepositoryAnalyzer:
    """Stateful per-repository analyzer (one instance per ingestion)."""

    def analyze(
        self,
        snapshot: Path,
        name: str,
        files: list[RepoFile],
        walker_notes: list[str],
    ) -> tuple[RepoGraph, RepoSummary]:
        graph = RepoGraph(root_name=name)
        graph.files = files
        langs: dict[str, int] = {}
        for f in files:
            langs[f.language] = langs.get(f.language, 0) + 1
        graph.languages = {k: v for k, v in sorted(langs.items(), key=lambda kv: -kv[1])}

        path_set = {f.path for f in files}
        py_analyzer = PythonFileAnalyzer()

        module_index: dict[str, str] = {}   # dotted module -> file path
        symbol_index: dict[str, Symbol] = {}
        source_lines: dict[str, list[str]] = {}

        # ---- pass 1: parse every source/test file -------------------------
        for f in files:
            if f.language == "unknown" or f.role == "other":
                continue
            full = snapshot / f.path
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source_lines[f.path] = text.splitlines()

            if f.language == "python":
                facts = py_analyzer.analyze(f.path, text)
                if facts.syntax_error:
                    graph.stats.setdefault("syntax_errors", []).append({"path": f.path, "error": facts.syntax_error})
                module = self._python_module_name(f.path)
                if module:
                    module_index[module] = f.path
                for ps in facts.symbols:
                    sym = Symbol(
                        name=ps.name,
                        kind=ps.kind if ps.kind in ("function", "class", "method", "route", "test") else "function",
                        path=f.path,
                        line_start=ps.line_start,
                        line_end=ps.line_end,
                        parent=ps.parent,
                        signature=ps.signature,
                        doc=ps.doc[:300],
                        is_exported=ps.is_exported,
                        language="python",
                    )
                    graph.symbols.append(sym)
                    symbol_index[f"{f.path}::{ps.name}"] = sym
                for imp in facts.imports:
                    dst = self._resolve_python_import(imp, f.path, module_index, path_set)
                    if dst and dst != f.path:
                        graph.edges.append(DependencyEdge(src=f.path, dst=dst, kind="imports"))
                # Route facts
                for ps in facts.symbols:
                    if ps.route:
                        graph.stats.setdefault("routes", []).append(
                            {"method": ps.route_method, "path": ps.route, "file": f.path, "symbol": ps.name, "line": ps.line_start}
                        )
            else:
                gfacts = parse_generic(f.path, text, f.language)
                if gfacts is None:
                    continue
                self._index_generic_module(gfacts.language, f.path, module_index)
                for gs in gfacts.symbols:
                    kind = gs.kind if gs.kind in ("function", "class", "method", "route", "test") else "function"
                    sym = Symbol(
                        name=gs.name, kind=kind, path=f.path,
                        line_start=gs.line_start, line_end=gs.line_end,
                        parent=gs.parent, signature=gs.signature,
                        language=gfacts.language,
                    )
                    graph.symbols.append(sym)
                    symbol_index[f"{f.path}::{gs.name}"] = sym
                for imp in gfacts.imports:
                    dst = self._resolve_generic_import(imp, f.path, path_set)
                    if dst and dst != f.path:
                        graph.edges.append(DependencyEdge(src=f.path, dst=dst, kind="imports"))

        # ---- pass 2: tested_by edges ---------------------------------------
        for edge in list(graph.edges):
            if edge.kind != "imports":
                continue
            src_role = next((f.role for f in files if f.path == edge.src), "source")
            dst_role = next((f.role for f in files if f.path == edge.dst), "source")
            if src_role == "test" and dst_role in ("source", "config"):
                graph.edges.append(DependencyEdge(src=edge.src, dst=edge.dst, kind="tested_by"))
            elif dst_role == "test" and src_role == "source":
                graph.edges.append(DependencyEdge(src=edge.src, dst=edge.dst, kind="tests"))

        # ---- pass 3: function call edges within the repo --------------------
        defined = {s.name for s in graph.symbols if s.kind in ("function", "method")}
        for f in files:
            if f.language != "python" or f.role not in ("source", "test"):
                continue
            full = snapshot / f.path
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            facts = py_analyzer.analyze(f.path, text)
            for ps in facts.symbols:
                for call in ps.calls:
                    if call in defined:
                        target = next(
                            (s for s in graph.symbols if s.name == call and s.path != f.path),
                            None,
                        )
                        if target is not None:
                            graph.edges.append(DependencyEdge(
                                src=f"{f.path}::{ps.name}", dst=f"{target.path}::{target.name}",
                                kind="calls", weight=0.5,
                            ))

        graph.stats["total_symbols"] = len(graph.symbols)
        graph.stats["total_files"] = len(files)
        summary = self._build_summary(snapshot, name, graph, files, walker_notes)
        return graph, summary

    # ------------------------------------------------------------------ #

    def _python_module_name(self, rel_path: str) -> str:
        p = Path(rel_path)
        parts = list(p.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _index_generic_module(self, language: str, rel_path: str, module_index: dict[str, str]) -> None:
        p = Path(rel_path)
        module_index.setdefault(f"{language}:{p.as_posix()}", rel_path)
        module_index.setdefault(f"{language}:{p.stem}", rel_path)

    def _resolve_python_import(
        self, imp, src_path: str, module_index: dict[str, str], path_set: set[str]
    ) -> str | None:
        src_mod = self._python_module_name(src_path)
        if imp.module:
            candidates = [imp.module]
            if imp.level:
                base_parts = src_mod.split(".")
                for _ in range(imp.level - 1):
                    base_parts = base_parts[:-1] if base_parts else base_parts
                prefix = ".".join(base_parts)
                candidates = [f"{prefix}.{imp.module}" if prefix else imp.module]
            elif imp.names:
                candidates = [f"{imp.module}.{imp.names[0]}"] + candidates
            for cand in candidates:
                if cand in module_index:
                    return module_index[cand]
            # Partial match: package __init__
            parts = imp.module.split(".")
            while parts:
                partial = ".".join(parts)
                if partial in module_index:
                    return module_index[partial]
                parts.pop()
        elif imp.level and imp.names:
            base_parts = src_mod.split(".")
            for _ in range(imp.level - 1):
                base_parts = base_parts[:-1] if base_parts else base_parts
            prefix = ".".join(base_parts)
            cand = f"{prefix}.{imp.names[0]}" if prefix else imp.names[0]
            if cand in module_index:
                return module_index[cand]
            parts = list(base_parts)
            while parts:
                partial = ".".join(parts + imp.names[:1])
                if partial in module_index:
                    return module_index[partial]
                parts.pop()
        return None

    def _resolve_generic_import(self, imp: str, src_path: str, path_set: set[str]) -> str | None:
        base = imp.replace("./", "").replace("../", "")
        base = base.split("?")[0]
        candidates = [base, f"{base}.ts", f"{base}.tsx", f"{base}.js", f"{base}.mjs",
                      f"{base}.go", f"{base}/index.ts", f"{base}/index.js", f"{base}.py"]
        for cand in candidates:
            if cand in path_set:
                return cand
        stem = Path(base).stem
        for path in path_set:
            if Path(path).stem == stem and Path(path).suffix.lower() in SOURCE_EXTENSIONS:
                return path
        return None

    def _build_summary(
        self, snapshot: Path, name: str, graph: RepoGraph, files: list[RepoFile], walker_notes: list[str]
    ) -> RepoSummary:
        summary = RepoSummary(name=name)
        summary.languages = graph.languages
        summary.total_files = len(files)
        summary.total_symbols = len(graph.symbols)

        readme = next((f.path for f in files if f.path.lower() in ("readme.md", "readme.rst", "readme.txt")), "")
        if readme:
            summary.notable_files.append(readme)
            try:
                text = (snapshot / readme).read_text(encoding="utf-8", errors="replace")
                first_para = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
                summary.description = (first_para[0].strip()[:300] if first_para else "")
            except OSError:
                pass

        entry_names = {"main.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "server.py",
                       "index.js", "index.ts", "server.js", "main.go", "main.ts"}
        for f in files:
            if Path(f.path).name.lower() in entry_names and f.role == "source":
                summary.entry_points.append(f.path)
            if f.role == "manifest":
                summary.package_manifests.append(f.path)
        summary.entry_points = summary.entry_points[:8]
        summary.package_manifests = summary.package_manifests[:8]

        routes = graph.stats.get("routes", [])
        summary.api_routes = [f"{r['method']} {r['path']} ({r['file']}:{r['line']})" for r in routes[:25]]

        frameworks = self._detect_test_frameworks(snapshot, files)
        summary.test_frameworks = frameworks

        summary.build_commands = self._detect_build_commands(snapshot, files)

        test_files = [f.path for f in files if f.role == "test"]
        source_files = [f.path for f in files if f.role == "source"]
        observations: list[str] = []
        if not test_files:
            observations.append("No test files detected — validation will rely on build/compile checks.")
        else:
            observations.append(f"{len(test_files)} test file(s) and {len(source_files)} source file(s) detected.")
        syntax_errors = graph.stats.get("syntax_errors", [])
        if syntax_errors:
            observations.append(f"{len(syntax_errors)} Python file(s) currently have syntax errors: " +
                                ", ".join(e["path"] for e in syntax_errors[:5]))
        if routes:
            observations.append(f"{len(routes)} HTTP route(s) detected.")
        for note in walker_notes[:3]:
            observations.append(note)
        if graph.languages.get("python") and graph.languages.get("javascript"):
            observations.append("Mixed Python/JavaScript codebase.")
        summary.observations = observations
        return summary

    def _detect_test_frameworks(self, snapshot: Path, files: list[RepoFile]) -> list[str]:
        frameworks: list[str] = []
        for f in files:
            if f.role == "manifest":
                try:
                    text = (snapshot / f.path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                low = text.lower()
                for fw, markers in (
                    ("pytest", ("pytest",)),
                    ("unittest", ("unittest",)),
                    ("jest", ("\"jest\"", "'jest'")),
                    ("vitest", ("vitest",)),
                    ("mocha", ("mocha",)),
                    ("go test", ("go.mod",)),
                ):
                    if any(m in low for m in markers) and fw not in frameworks:
                        frameworks.append(fw)
        for f in files:
            if f.role == "test" and f.language == "python" and "pytest" not in frameworks:
                frameworks.append("pytest")
                break
        return frameworks

    def _detect_build_commands(self, snapshot: Path, files: list[RepoFile]) -> list[str]:
        commands: list[str] = []
        names = {Path(f.path).name.lower() for f in files}
        if "pyproject.toml" in names or "requirements.txt" in names:
            commands.append("python -m pytest")
        if "package.json" in names:
            try:
                pkg = next(f.path for f in files if Path(f.path).name.lower() == "package.json")
                data = json.loads((snapshot / pkg).read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {})
                for script in ("test", "build", "lint"):
                    if script in scripts:
                        runner = "npm" if "package-lock.json" not in names else "npm"
                        commands.append(f"{runner} run {script}")
            except (OSError, json.JSONDecodeError):
                commands.append("npm test")
        if "go.mod" in names:
            commands.append("go test ./...")
        if "cargo.toml" in names:
            commands.append("cargo test")
        if "makefile" in names:
            commands.append("make test")
        return commands
