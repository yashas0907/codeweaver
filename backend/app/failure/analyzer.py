"""Failure analyzer: turns raw test/build output into a categorized,
evidence-linked FailureAnalysis.

Categories follow the spec: syntax, type, dependency, import, runtime,
test_assertion, environment, configuration, api_contract, database,
security, flaky/unknown. Traceback frames are mapped back to repository
files whenever possible.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import FailureAnalysis, FailureCategory

PY_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
PY_SHORT_FRAME = re.compile(r"^([\w./:\\~-]+\.py):(\d+): in (\S+)", re.MULTILINE)
PY_EXC = re.compile(r"^([\w.]+(?:Error|Exception|Interrupt|Warning|Exit))(?::\s*(.*))?$", re.MULTILINE)
ASSERT_LINE = re.compile(r"^[Ee]\s+.*(assert|AssertionError|expected|!=|==)", re.MULTILINE)
MODULE_NOT_FOUND = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
IMPORT_ERROR = re.compile(r"ImportError: (?:cannot import name )?'?([\w.]+)'?")
SYNTAX_ERROR = re.compile(r"SyntaxError: \((.*?)\)?(.*)")
TYPE_ERROR = re.compile(r"TypeError: (.*)")
NPM_MISSING = re.compile(r"Cannot find module '([^']+)'")
GO_UNDEFINED = re.compile(r"undefined: (\S+)")
PIP_ERROR = re.compile(r"(ERROR: (?:Could not find|No matching distribution).*)")

CATEGORY_RULES: tuple[tuple[FailureCategory, re.Pattern[str]], ...] = (
    ("syntax", re.compile(r"SyntaxError|IndentationError|TabError", re.IGNORECASE)),
    ("import", re.compile(r"ModuleNotFoundError|ImportError|Cannot find module|undefined: \w+", re.IGNORECASE)),
    ("dependency", re.compile(r"No matching distribution|pip install|npm ERR|missing dependencies|unresolved import", re.IGNORECASE)),
    ("type", re.compile(r"TypeError|mypy|AttributeError: '.*' object has no attribute", re.IGNORECASE)),
    ("test_assertion", re.compile(r"AssertionError|assert .* ==|Expected:|Falsifying example|did not raise", re.IGNORECASE)),
    ("database", re.compile(r"OperationalError|IntegrityError|no such table|no such column|UniqueViolation|sqlite3\.", re.IGNORECASE)),
    ("security", re.compile(r"PermissionError|AccessDenied|Unauthorized|Forbidden|401|403", re.IGNORECASE)),
    ("environment", re.compile(r"command not found|is not recognized|No such file or directory|binary not found", re.IGNORECASE)),
    ("configuration", re.compile(r"KeyError: |MissingError|invalid configuration|ConfigError|missing environment variable", re.IGNORECASE)),
    ("api_contract", re.compile(r"HTTPStatusError|status_code=5\d\d|404 Client Error|malformed response|KeyError: 'results'", re.IGNORECASE)),
    ("runtime", re.compile(r"RuntimeError|ValueError|ZeroDivisionError|KeyError|IndexError|OSError|ConnectionError|TimeoutError", re.IGNORECASE)),
)


class FailureAnalyzer:
    """Categorizes a failure from raw output; optional repo root maps frames."""

    def analyze(self, raw_output: str, repo_root: Path | None = None, command: str = "") -> FailureAnalysis:
        analysis = FailureAnalysis(error=self._headline(raw_output))
        analysis.category = self._categorize(raw_output)
        analysis.evidence = self._evidence_lines(raw_output)

        frames = PY_FRAME.findall(raw_output)
        short_frames = PY_SHORT_FRAME.findall(raw_output)
        repo_files: list[str] = []
        last_frame: tuple[str, str, str] | None = None
        for file_ref, lineno, func in frames + short_frames:
            rel = self._to_repo_path(file_ref, repo_root)
            if rel:
                repo_files.append(f"{rel}:{lineno}")
                last_frame = (rel, lineno, func)
        analysis.related_files = list(dict.fromkeys(repo_files))[:8]
        if last_frame:
            analysis.location = f"{last_frame[0]}:{last_frame[1]} in {last_frame[2]}"

        analysis.likely_cause, analysis.recommended_repair = self._cause_and_repair(
            analysis.category, raw_output, analysis.location
        )
        analysis.confidence = self._confidence(analysis.category, bool(frames), raw_output)
        return analysis

    # ------------------------------------------------------------------ #

    def _categorize(self, output: str) -> FailureCategory:
        for category, pattern in CATEGORY_RULES:
            if pattern.search(output):
                return category
        if "FAILED" in output or "failed" in output:
            return "test_assertion"
        return "unknown"

    def _headline(self, output: str) -> str:
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("E ") or re.match(PY_EXC, line):
                return line[:300]
        exc = PY_EXC.search(output)
        if exc:
            return f"{exc.group(1)}: {exc.group(2) or ''}"[:300]
        for line in output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                return line.strip()[:300]
        return (output.strip().splitlines() or ["unknown failure"])[-1][:300]

    def _evidence_lines(self, output: str) -> list[str]:
        evidence: list[str] = []
        for line in output.splitlines():
            s = line.rstrip()
            if s.startswith("E ") or PY_FRAME.search(s) or re.match(PY_EXC, s.strip()):
                evidence.append(s[:250])
        return evidence[-8:]

    def _to_repo_path(self, file_ref: str, repo_root: Path | None) -> str | None:
        if not repo_root:
            return None
        p = Path(file_ref)
        if not p.is_absolute():
            p = repo_root / p
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
            if str(rel).startswith(".."):
                return None
            return rel.as_posix()
        except (ValueError, OSError):
            return None

    def _cause_and_repair(self, category: FailureCategory, output: str, location: str) -> tuple[str, str]:
        if m := MODULE_NOT_FOUND.search(output):
            mod = m.group(1)
            return (
                f"the module `{mod}` is not importable — it may be missing from the project or the environment",
                f"verify the module path `{mod}`; if it is a project package ensure __init__.py exists; if it is third-party, add it to requirements and install",
            )
        if m := NPM_MISSING.search(output):
            mod = m.group(1)
            return (f"node module '{mod}' is missing", f"add '{mod}' to package.json and run npm install")
        if m := SYNTAX_ERROR.search(output):
            return ("Python syntax error", "re-read the failing file and fix the reported line")
        if category == "type":
            m = TYPE_ERROR.search(output)
            detail = m.group(1) if m else "type mismatch"
            return (
                f"a type error occurred: {detail[:200]}",
                "inspect the failing line for wrong argument types or None values; add explicit checks",
            )
        if category == "test_assertion":
            detail = self._assertion_detail(output)
            return (
                f"a test assertion failed{(' — ' + detail) if detail else ''}",
                "compare the expected vs actual values in the failing test; the recent change likely altered behavior",
            )
        if category == "dependency":
            return (
                "a dependency could not be resolved or installed",
                "check the package manifest for the missing dependency and the install command output",
            )
        if category == "environment":
            return (
                "the validation command or environment is unavailable",
                "check that the required tool is installed and the working directory is correct",
            )
        if category == "database":
            return (
                "a database operation failed (schema or connection)",
                "verify migrations/schema and the database connection configuration",
            )
        if category == "configuration":
            return (
                "a required configuration key is missing",
                "trace the KeyError to its source and provide the configuration or a safe default",
            )
        if category == "import":
            return (
                "an import statement references a name that does not exist",
                "check the imported symbol exists in the source module (or fix the import path)",
            )
        if category == "runtime":
            return (
                "an unhandled runtime error occurred",
                "inspect the traceback frame and add handling for the failing condition",
            )
        return (
            "the failure could not be classified with confidence",
            "re-run the failing command and inspect the full output",
        )

    def _assertion_detail(self, output: str) -> str:
        for m in ASSERT_LINE.finditer(output):
            text = m.group(0).strip()
            if len(text) > 8:
                return text[2:200]
        return ""

    def _confidence(self, category: FailureCategory, has_frames: bool, output: str) -> float:
        base = {
            "syntax": 0.95, "import": 0.9, "dependency": 0.85, "type": 0.8,
            "test_assertion": 0.75, "environment": 0.85, "configuration": 0.7,
            "database": 0.75, "security": 0.6, "api_contract": 0.55,
            "runtime": 0.7, "flaky": 0.3, "unknown": 0.3,
        }.get(category, 0.4)
        if has_frames:
            base = min(0.98, base + 0.05)
        if len(output) < 40:
            base = min(base, 0.5)
        return round(base, 2)
