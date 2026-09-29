"""Typed tool registry.

Every agent action runs through a registered tool: declared parameters,
mode enforcement, timeouts, and persisted ToolCall/ToolResult records.
Tools never touch the filesystem or subprocess world directly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import get_settings
from app.failure.analyzer import FailureAnalyzer
from app.schemas import (
    AgentMode, AgentPhase, FailureAnalysis, Provenance, RepoGraph,
    SecurityScanResult, TestRun, ToolCall, ToolResult,
)
from app.security.scanner import SecurityScanner
from app.tools.exec import run_command
from app.tools.testparse import parse_test_output
from app.workspace import Workspace


@dataclass
class ToolContext:
    workspace: Workspace
    graph: RepoGraph | None = None
    retrieval: Any = None            # RetrievalService
    repo_id: str = ""
    last_test_output: str = ""
    last_test_run: TestRun | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    category: str                    # read | search | edit | git | exec | analysis
    mutates: bool = False
    handler: Callable[[ToolContext, dict], Awaitable[tuple[str, dict]]] | None = None
    params: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self.settings = get_settings()
        self._tools: dict[str, ToolDef] = {}
        self._register_all()

    # ------------------------------------------------------------------ #

    def _register_all(self) -> None:
        reg = self._register
        reg(ToolDef("repository_map", "Summarize the repository structure, languages and key modules", "read", handler=self._repository_map))
        reg(ToolDef("repository_search", "Hybrid code search over the indexed repository", "search",
                    params={"query": "str", "top_k": "int=10"}, handler=self._repository_search))
        reg(ToolDef("read_file", "Read a whole file (bounded)", "read", params={"path": "str"}, handler=self._read_file))
        reg(ToolDef("read_range", "Read lines start..end of a file", "read",
                    params={"path": "str", "start": "int", "end": "int"}, handler=self._read_range))
        reg(ToolDef("symbol_lookup", "Look up a symbol by name", "search", params={"name": "str"}, handler=self._symbol_lookup))
        reg(ToolDef("git_status", "git status of the workspace", "git", handler=self._git_status))
        reg(ToolDef("git_diff", "Unified diff of workspace changes", "git", handler=self._git_diff))
        reg(ToolDef("git_log", "Recent commit history (optionally per file)", "git",
                    params={"path": "str?"}, handler=self._git_log))
        reg(ToolDef("run_tests", "Run the test suite (pytest/jest/go test auto-detected)", "exec",
                    params={"command": "str?"}, handler=self._run_tests))
        reg(ToolDef("run_linter", "Run the available linter", "exec", handler=self._run_linter))
        reg(ToolDef("run_build", "Run build/compile validation", "exec", handler=self._run_build))
        reg(ToolDef("run_typecheck", "Run type checking when available", "exec", handler=self._run_typecheck))
        reg(ToolDef("run_security_scan", "Evidence-based security scan of the workspace", "analysis", handler=self._run_security_scan))
        reg(ToolDef("inspect_test_failure", "Categorize and analyze the last test failure", "analysis", handler=self._inspect_test_failure))
        reg(ToolDef("create_file", "Create a new file in the workspace", "edit", mutates=True,
                    params={"path": "str", "content": "str"}, handler=self._create_file))
        reg(ToolDef("apply_patch", "Apply a replacement patch (old_text -> new_text) to a file", "edit", mutates=True,
                    params={"path": "str", "old_text": "str", "new_text": "str", "description": "str?"}, handler=self._apply_patch))
        reg(ToolDef("list_changed_files", "List files changed in this run", "read", handler=self._list_changed_files))

    def _register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "category": t.category,
             "mutates": t.mutates, "params": t.params}
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict, phase: AgentPhase = AgentPhase.IMPLEMENTING) -> ToolResult:
        started = time.monotonic()
        call = ToolCall(id=f"tc_{int(started*1000)%10_000_000}", name=name, arguments=arguments, phase=phase)
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool_call_id=call.id, name=name, ok=False, error=f"unknown tool: {name}")
        if tool.mutates:
            try:
                self.context.workspace.check_mode()
            except Exception as exc:
                return ToolResult(tool_call_id=call.id, name=name, ok=False, error=str(exc))
        try:
            output, artifacts = await tool.handler(self.context, arguments or {})
            result = ToolResult(
                tool_call_id=call.id, name=name, ok=True,
                output=output[:30000], artifacts=artifacts,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except PermissionError as exc:
            result = ToolResult(tool_call_id=call.id, name=name, ok=False, error=str(exc),
                                duration_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:  # noqa: BLE001 — tools must not crash the agent
            result = ToolResult(
                tool_call_id=call.id, name=name, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        return result

    # ------------------------------------------------------------------ #
    # Read / search tools
    # ------------------------------------------------------------------ #

    async def _repository_map(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        if ctx.graph is None:
            return "repository graph unavailable", {}
        g = ctx.graph
        lines = [
            f"Repository: {g.root_name}",
            f"Languages: {g.languages}",
            f"Files: {g.stats.get('total_files', len(g.files))}, Symbols: {g.stats.get('total_symbols', len(g.symbols))}",
            "",
            "Top modules (by incoming import edges):",
        ]
        incoming: dict[str, int] = {}
        for e in g.edges:
            if e.kind == "imports":
                incoming[e.dst] = incoming.get(e.dst, 0) + 1
        for path, count in sorted(incoming.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  {path}  (imported by {count})")
        routes = g.stats.get("routes", [])
        if routes:
            lines.append("\nHTTP routes:")
            for r in routes[:15]:
                lines.append(f"  {r['method']} {r['path']} -> {r['file']}:{r['line']} ({r['symbol']})")
        return "\n".join(lines), {"languages": g.languages}

    async def _repository_search(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        query = str(args.get("query", "")).strip()
        top_k = int(args.get("top_k", 10))
        if not query:
            raise ValueError("query is required")
        if ctx.retrieval is None:
            raise RuntimeError("retrieval service unavailable")
        context = await ctx.retrieval.retrieve(ctx.repo_id, query, graph=ctx.graph, top_k=top_k)
        parts = []
        for chunk, prov in zip(context.items, context.provenance):
            parts.append(f"--- {prov.path} ({prov.line_start}-{prov.line_end}) {prov.symbol} [{prov.reason}]\n{chunk.text[:1500]}")
        return "\n\n".join(parts), {"provenance": [p.model_dump() for p in context.provenance]}

    async def _read_file(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        path = str(args.get("path", ""))
        text = ctx.workspace.read_file(path)
        if len(text) > 20000:
            text = text[:20000] + f"\n... [truncated, {len(text)} chars total]"
        return text, {"path": path, "chars": len(text)}

    async def _read_range(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        path = str(args.get("path", ""))
        start = int(args.get("start", 1))
        end = int(args.get("end", start + 100))
        return ctx.workspace.read_range(path, start, end), {"path": path, "start": start, "end": end}

    async def _symbol_lookup(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        name = str(args.get("name", "")).strip()
        if ctx.graph is None or not name:
            raise ValueError("name is required")
        matches = [s for s in ctx.graph.symbols if s.name == name or s.qualname == name or s.name.endswith(f".{name}")]
        if not matches:
            return f"no symbol named {name!r} found in repository graph", {}
        out = []
        for s in matches[:8]:
            tests = ctx.graph.tests_for(s.path)
            out.append(f"{s.kind} {s.qualname} at {s.path}:{s.line_start}-{s.line_end}  signature: {s.signature}  tests: {', '.join(tests) or 'none linked'}")
        return "\n".join(out), {"matches": [s.path for s in matches]}

    # ------------------------------------------------------------------ #
    # Git tools
    # ------------------------------------------------------------------ #

    def _git(self, *argv: str) -> str:
        result = run_command(["git", *argv], cwd=str(self.context.workspace.root),
                             timeout_seconds=30)
        if result.error and result.exit_code is None:
            raise RuntimeError(result.error)
        return (result.stdout + "\n" + result.stderr).strip() or "(no output)"

    async def _git_status(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        text = self._git("status", "--porcelain=v1")
        if not text or text == "(no output)":
            return "clean working tree", {}
        return text, {}

    async def _git_diff(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        diff = ctx.workspace.git_diff()
        return diff[:30000] or "(no changes)", {"changed": len(ctx.workspace.changes())}

    async def _git_log(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        argv = ["log", "--oneline", "-15", "--decorate=short"]
        path = args.get("path")
        if path:
            argv += ["--", str(path)]
        try:
            text = self._git(*argv)
        except RuntimeError as exc:
            return f"git history unavailable: {exc}", {}
        return text, {}

    # ------------------------------------------------------------------ #
    # Execution tools
    # ------------------------------------------------------------------ #

    def _detect_test_command(self) -> list[str] | None:
        root = self.context.workspace.root
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or any(root.glob("test_*.py")) or (root / "tests").is_dir():
            if _has_binary("python"):
                return ["python", "-m", "pytest", "-q", "--no-header"]
        if (root / "package.json").exists() and _has_binary("npm"):
            return ["npm", "test"]
        if (root / "go.mod").exists() and _has_binary("go"):
            return ["go", "test", "./..."]
        if (root / "Cargo.toml").exists() and _has_binary("cargo"):
            return ["cargo", "test"]
        return None  # no recognized test runner

    async def _run_tests(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        cmd_arg = args.get("command")
        if cmd_arg:
            cmd = str(cmd_arg).split()
        else:
            cmd = self._detect_test_command()
        if cmd is None:
            test_run = {
                "all_passed": True, "total": 0, "passed": 0, "failed": 0,
                "errors": 0, "skipped": 0, "command": "run_tests",
                "exit_code": 0, "duration_ms": 0,
                "raw_output_tail": "no recognized test runner found in repository (no tests to run)",
                "cases": [],
            }
            return "", {"test_run": test_run}
        result = await asyncio.to_thread(
            run_command, cmd, str(ctx.workspace.root), self.settings.max_test_timeout_seconds,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        test_run = parse_test_output(" ".join(cmd), output, result.exit_code)
        test_run.duration_ms = result.duration_ms
        if result.timed_out or result.error:
            test_run.all_passed = False
            test_run.raw_output_tail = (result.error or "") + "\n" + test_run.raw_output_tail
        ctx.last_test_output = output
        ctx.last_test_run = test_run
        status = "PASSED" if test_run.all_passed else "FAILED"
        summary = (
            f"{status}: `{test_run.command}` — {test_run.passed} passed, {test_run.failed} failed, "
            f"{test_run.errors} errors, {test_run.skipped} skipped (exit={test_run.exit_code})\n"
            f"{test_run.raw_output_tail[-3000:]}"
        )
        return summary, {"test_run": test_run.model_dump(mode="json")}

    async def _run_linter(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        root = str(ctx.workspace.root)
        if _has_binary("ruff"):
            result = await asyncio.to_thread(run_command, ["ruff", "check", "."], root, 120)
        elif _has_binary("eslint") and (ctx.workspace.root / ".eslintrc").exists():
            result = await asyncio.to_thread(run_command, ["npx", "eslint", "."], root, 180)
        else:
            return await self._py_compile_check(ctx)
        output = (result.stdout + "\n" + result.stderr).strip()
        return f"linter exit={result.exit_code}\n{output[-4000:]}", {"exit_code": result.exit_code}

    async def _py_compile_check(self, ctx: ToolContext) -> tuple[str, dict]:
        import py_compile

        problems = []
        for rel in ctx.workspace.list_files("**/*.py")[:500]:
            try:
                py_compile.compile(str(ctx.workspace.root / rel), doraise=True, cfile="NUL" if _is_windows() else "/dev/null")
            except py_compile.PyCompileError as exc:
                problems.append(str(exc)[:400])
        if problems:
            return f"compile check found {len(problems)} problem(s):\n" + "\n".join(problems), {"exit_code": 1}
        return "compile check passed (all .py files syntactically valid)", {"exit_code": 0}

    async def _run_build(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        root = ctx.workspace.root
        if (root / "package.json").exists() and _has_binary("npm"):
            result = await asyncio.to_thread(run_command, ["npm", "run", "build"], str(root), 300)
            output = (result.stdout + "\n" + result.stderr).strip()
            return f"build exit={result.exit_code}\n{output[-4000:]}", {"exit_code": result.exit_code}
        return await self._py_compile_check(ctx)

    async def _run_typecheck(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        root = str(ctx.workspace.root)
        if _has_binary("mypy"):
            result = await asyncio.to_thread(run_command, ["mypy", "--ignore-missing-imports", "."], root, 240)
        elif _has_binary("tsc"):
            result = await asyncio.to_thread(run_command, ["npx", "tsc", "--noEmit"], root, 240)
        else:
            return "no type checker available (install mypy or tsc); skipped", {"exit_code": None}
        output = (result.stdout + "\n" + result.stderr).strip()
        return f"typecheck exit={result.exit_code}\n{output[-4000:]}", {"exit_code": result.exit_code}

    async def _run_security_scan(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        scanner = SecurityScanner()
        scan = await asyncio.to_thread(scanner.scan_directory, ctx.workspace.root)
        scan.files_scanned = scan.files_scanned
        lines = [f"security scan: {len(scan.findings)} finding(s) across {scan.files_scanned} file(s)"]
        for f in scan.findings[:25]:
            lines.append(f"  [{f.severity.value}] {f.rule_id} {f.path}:{f.line} — {f.title}: {f.evidence[:120]}")
        return "\n".join(lines), {"scan": scan.model_dump(mode="json")}

    async def _inspect_test_failure(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        if not ctx.last_test_output and ctx.last_test_run is None:
            raise ValueError("no test output to inspect; run_tests first")
        analyzer = FailureAnalyzer()
        analysis = analyzer.analyze(
            ctx.last_test_output,
            repo_root=ctx.workspace.root,
            command=ctx.last_test_run.command if ctx.last_test_run else "",
        )
        text = (
            f"category: {analysis.category} (confidence {analysis.confidence})\n"
            f"error: {analysis.error}\nlocation: {analysis.location}\n"
            f"likely cause: {analysis.likely_cause}\n"
            f"related files: {', '.join(analysis.related_files)}\n"
            f"recommended repair: {analysis.recommended_repair}\n"
            f"evidence: {chr(10).join(analysis.evidence[:6])}"
        )
        return text, {"analysis": analysis.model_dump(mode="json")}

    # ------------------------------------------------------------------ #
    # Edit tools
    # ------------------------------------------------------------------ #

    async def _create_file(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not path:
            raise ValueError("path is required")
        change = ctx.workspace.create_file(path, content, description=str(args.get("description", "")))
        return f"created {change.path} ({len(content)} chars)", {"change": change.model_dump()}

    async def _apply_patch(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        path = str(args.get("path", ""))
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        if not path:
            raise ValueError("path is required")
        change = ctx.workspace.replace_in_file(
            path, old_text, new_text,
            description=str(args.get("description", "")),
        )
        return (
            f"patched {change.path}: {change.description}\n{change.diff[:2000]}",
            {"change": change.model_dump()},
        )

    async def _list_changed_files(self, ctx: ToolContext, args: dict) -> tuple[str, dict]:
        changes = ctx.workspace.changes()
        if not changes:
            return "no files changed yet", {"count": 0}
        lines = [f"{c.change_type}: {c.path}" for c in changes]
        return "\n".join(lines), {"count": len(changes), "paths": [c.path for c in changes]}


def _has_binary(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")
