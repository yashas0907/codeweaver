"""Static code reviewer producing evidence-backed findings.

Philosophy (per spec): no vague "best practice" criticism. Every finding
points at a concrete file:line with the offending text as evidence.
Used both for reviewing agent-produced changes and the dedicated
repository review mode.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from app.schemas import RepoGraph, ReviewCategory, ReviewFinding, ReviewResult, Severity

FUNC_DEF = re.compile(r"^(\s*)def\s+\w+|^(\s*)async def\s+\w+")
BARE_EXCEPT = re.compile(r"except\s*:")
EXCEPT_PASS = re.compile(r"except[^:]*:\s*$")
BROAD_EXCEPT = re.compile(r"except\s+Exception\s*:")
NONE_COMPARE = re.compile(r"==\s*None|!=\s*None")
MUTABLE_DEFAULT = re.compile(r"def\s+\w+\([^)]*=\s*(\[\]|\{\}|set\(\))")
TODO_MARK = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b")
PRINT_DEBUG = re.compile(r"^\s*print\(")
OPEN_NO_WITH = re.compile(r"^\s*\w+\s*=\s*open\(")
LONG_LINE = re.compile(r"^.{121,}$")
STR_CONCAT_LOOP = re.compile(r'^\s*\w+\s*\+=\s*[^"\']*["\']')
SQL_FORMAT = re.compile(r"""(execute|executemany)\s*\(\s*f?["'].*(SELECT|INSERT|UPDATE|DELETE)""", re.IGNORECASE)
RETURN_MISSING_PAREN = re.compile(r"^\s*(if|while|for)\s+.*:\s*$")
MAGIC_COMPARE = re.compile(r"(?:status|code|level|type)\s*==\s*['\"]\w+['\"]")

MAX_FUNC_LINES = 80
MAX_FILE_LINES = 600
MAX_LINE_LEN = 120


class CodeReviewer:
    """Reviews a set of files (changed files in a run, or the whole repo)."""

    def review_files(
        self,
        root: Path,
        paths: list[str],
        graph: RepoGraph | None = None,
        include_structure: bool = True,
    ) -> ReviewResult:
        findings: list[ReviewFinding] = []
        files_reviewed: list[str] = []
        func_bodies: dict[str, list[str]] = defaultdict(list)

        for rel in paths:
            path = root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files_reviewed.append(rel)
            findings.extend(self._scan_file(rel, text, graph))
            if rel.endswith(".py"):
                for name, body in self._extract_functions(text).items():
                    func_bodies[_normalize_body(body)].append(f"{rel}::{name}")

        # Duplication detection across files (evidence-based: identical bodies)
        for body, locations in func_bodies.items():
            if len(locations) > 1 and body.count("\n") >= 3:
                for loc in locations[1:]:
                    file_part = loc.split("::")[0]
                    line = _find_line(root, file_part, locations[0].split("::")[-1])
                    findings.append(ReviewFinding(
                        id=_fid("DUP", loc, line),
                        category="duplication", severity=Severity.MEDIUM,
                        path=file_part, line=line,
                        evidence=f"duplicate of {locations[0]} ({body.count(chr(10))} identical lines)",
                        explanation="this function body is identical to another function in the codebase",
                        recommended_fix=f"extract the shared logic used by {locations[0]} and {loc} into a single function",
                    ))

        if include_structure and graph is not None:
            findings.extend(self._structure_findings(graph, root))

        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4}
        findings.sort(key=lambda f: (severity_order[f.severity], f.path, f.line))
        result = ReviewResult(findings=findings[:80], files_reviewed=files_reviewed)
        result.summary = (
            f"{len(findings)} finding(s) across {len(files_reviewed)} file(s); "
            f"by severity: {result.counts_by_severity()}"
        )
        return result

    # ------------------------------------------------------------------ #

    def _scan_file(self, rel: str, text: str, graph: RepoGraph | None) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        lines = text.splitlines()

        if len(lines) > MAX_FILE_LINES and rel.endswith(".py"):
            findings.append(ReviewFinding(
                id=_fid("BIGFILE", rel, 1), category="maintainability", severity=Severity.LOW,
                path=rel, line=1, evidence=f"file has {len(lines)} lines",
                explanation="large files are harder to navigate and modify safely",
                recommended_fix="consider splitting this file by responsibility",
            ))

        current_func = ""
        current_func_start = 0
        current_func_indent = 0
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if m := FUNC_DEF.match(line):
                current_func = re.sub(r"^\s*(async )?def\s+", "", m.group(0)).split("(")[0]
                current_func_start = lineno
                current_func_indent = len(line) - len(line.lstrip())
                if MUTABLE_DEFAULT.search(line):
                    findings.append(ReviewFinding(
                        id=_fid("MUTDEF", rel, lineno), category="correctness", severity=Severity.MEDIUM,
                        path=rel, line=lineno, symbol=current_func,
                        evidence=stripped[:200],
                        explanation="mutable default arguments are shared across all calls, which usually causes accumulation bugs",
                        recommended_fix="use None as the default and create the container inside the function",
                    ))
            elif current_func and line.strip() and not line.startswith(" " * (current_func_indent + 1)) and lineno > current_func_start:
                func_len = lineno - current_func_start
                if func_len > MAX_FUNC_LINES:
                    findings.append(ReviewFinding(
                        id=_fid("LONGFUNC", rel, current_func_start), category="maintainability", severity=Severity.LOW,
                        path=rel, line=current_func_start, symbol=current_func,
                        evidence=f"function {current_func} spans {func_len} lines",
                        explanation=f"functions longer than {MAX_FUNC_LINES} lines are hard to test and review",
                        recommended_fix=f"split {current_func} into smaller units",
                    ))
                current_func = ""

            if BARE_EXCEPT.search(line):
                findings.append(ReviewFinding(
                    id=_fid("BAREEXC", rel, lineno), category="error_handling", severity=Severity.HIGH,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="a bare `except:` also catches KeyboardInterrupt/SystemExit and hides real errors",
                    recommended_fix="catch the specific exception types you can handle",
                ))
            elif BROAD_EXCEPT.search(line):
                findings.append(ReviewFinding(
                    id=_fid("BRDEXC", rel, lineno), category="error_handling", severity=Severity.LOW,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="catching Exception broadly can silently swallow unrelated failures",
                    recommended_fix="narrow the exception type or re-raise after logging",
                ))
            if NONE_COMPARE.search(line):
                findings.append(ReviewFinding(
                    id=_fid("NONECMP", rel, lineno), category="correctness", severity=Severity.LOW,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="comparisons to None should use `is`/`is not` (PEP 8; avoids overloaded __eq__ bugs)",
                    recommended_fix="replace `== None` with `is None` and `!= None` with `is not None`",
                ))
            if TODO_MARK.search(line):
                findings.append(ReviewFinding(
                    id=_fid("TODO", rel, lineno), category="maintainability", severity=Severity.INFO,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="unresolved TODO/FIXME marker",
                    recommended_fix="resolve the TODO or track it in an issue",
                ))
            if PRINT_DEBUG.search(line) and rel.endswith(".py") and "tests/" not in rel and "logging" not in line:
                findings.append(ReviewFinding(
                    id=_fid("PRINT", rel, lineno), category="maintainability", severity=Severity.INFO,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="print() used for output; structured logging is preferable in library/production code",
                    recommended_fix="use the logging module",
                ))
            if OPEN_NO_WITH.search(line):
                findings.append(ReviewFinding(
                    id=_fid("NOWITH", rel, lineno), category="error_handling", severity=Severity.MEDIUM,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="file opened without a context manager leaks the handle on exceptions",
                    recommended_fix="use `with open(...) as fh:`",
                ))
            if STR_CONCAT_LOOP.search(line) and rel.endswith(".py"):
                findings.append(ReviewFinding(
                    id=_fid("STRCAT", rel, lineno), category="performance", severity=Severity.LOW,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="string concatenation in a loop is O(n^2); parts should be collected and joined",
                    recommended_fix="accumulate parts in a list and use ''.join(parts)",
                ))
            if SQL_FORMAT.search(line):
                findings.append(ReviewFinding(
                    id=_fid("SQLSTR", rel, lineno), category="security", severity=Severity.HIGH,
                    path=rel, line=lineno, symbol=current_func or "",
                    evidence=stripped[:200],
                    explanation="SQL statement appears to be built via string formatting",
                    recommended_fix="use parameterized queries",
                ))
        return findings

    def _extract_functions(self, text: str) -> dict[str, str]:
        """Fallback line-based function body extraction for duplication checks."""
        out: dict[str, str] = {}
        lines = text.splitlines()
        current: list[str] = []
        name = ""
        base_indent = 0
        for line in lines:
            m = re.match(r"^(\s*)(?:async )?def\s+(\w+)", line)
            if m:
                if name and current:
                    out[name] = "\n".join(current)
                name, base_indent, current = m.group(2), len(m.group(1)), [line]
            elif name:
                if line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                    out[name] = "\n".join(current)
                    name, current = "", []
                else:
                    current.append(line)
        if name and current:
            out[name] = "\n".join(current)
        return out

    def _structure_findings(self, graph: RepoGraph, root: Path) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        source_paths = {f.path for f in graph.files if f.role == "source"}
        tested = {e.dst for e in graph.edges if e.kind == "tested_by"}
        untested = sorted(source_paths - tested)
        # Report only a bounded sample, with concrete file evidence.
        for path in untested[:10]:
            findings.append(ReviewFinding(
                id=_fid("NOTEST", path, 0), category="testing", severity=Severity.LOW,
                path=path, line=0,
                evidence=f"no test file imports {path} (repository graph: tested_by edges)",
                explanation="source module has no directly associated test file",
                recommended_fix=f"add tests importing {path} (they will link automatically via the repository graph)",
            ))
        return findings


def _normalize_body(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip()[:2000]


def _find_line(root: Path, rel_path: str, func_name: str) -> int:
    try:
        text = (root / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if re.search(rf"(?:async )?def\s+{re.escape(func_name)}\b", line):
            return lineno
    return 0


def _fid(rule: str, path: str, line: int) -> str:
    digest = hashlib.sha1(f"{rule}:{path}:{line}".encode()).hexdigest()[:8]
    return f"{rule}-{digest}"
