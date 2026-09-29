"""Typed schemas for every artifact CodeWeaver produces.

All agent state crossing stage boundaries, the database, or the API is a
pydantic model from this module. LLM output is parsed into these schemas and
rejected/repaired when malformed — free-form text is never trusted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentPhase(str, Enum):
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    DEBUGGING = "DEBUGGING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMode(str, Enum):
    READ_ONLY = "READ_ONLY"
    WORKSPACE_EDIT = "WORKSPACE_EDIT"


FailureCategory = Literal[
    "syntax", "type", "dependency", "import", "runtime", "test_assertion",
    "environment", "configuration", "api_contract", "database", "security",
    "flaky", "unknown",
]

ReviewCategory = Literal[
    "correctness", "maintainability", "architecture", "duplication",
    "error_handling", "security", "performance", "testing", "dependency_risk",
    "api_design",
]

ChangeType = Literal["created", "modified", "deleted"]


# --------------------------------------------------------------------------- #
# Repository intelligence
# --------------------------------------------------------------------------- #

class Symbol(BaseModel):
    """A parsed code symbol (function, class, route, test...)."""

    name: str
    kind: Literal["function", "class", "method", "route", "test", "module", "interface", "variable"]
    path: str
    line_start: int = 0
    line_end: int = 0
    parent: str | None = None
    signature: str = ""
    doc: str = ""
    is_exported: bool = False
    language: str = "unknown"

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


class DependencyEdge(BaseModel):
    """Directed edge in the repository graph."""

    src: str          # path or symbol qualifier
    dst: str
    kind: Literal["imports", "calls", "tested_by", "tests", "contains", "uses_config"]
    weight: float = 1.0


class RepoFile(BaseModel):
    path: str
    language: str
    extension: str
    size_bytes: int
    num_lines: int
    role: Literal["source", "test", "doc", "config", "ci", "manifest", "other"]
    sha1: str = ""


class RepoGraph(BaseModel):
    """Repository map: files + symbols + relationship edges."""

    root_name: str
    files: list[RepoFile] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)
    languages: dict[str, int] = Field(default_factory=dict)  # lang -> file count
    stats: dict[str, Any] = Field(default_factory=dict)

    def outgoing(self, src: str) -> list[DependencyEdge]:
        return [e for e in self.edges if e.src == src]

    def incoming(self, dst: str) -> list[DependencyEdge]:
        return [e for e in self.edges if e.dst == dst]

    def tests_for(self, source_path: str) -> list[str]:
        return [e.src for e in self.edges if e.kind == "tested_by" and e.dst == source_path]

    def imports_of(self, path: str) -> list[str]:
        return [e.dst for e in self.edges if e.kind == "imports" and e.src == path]


class RepoSummary(BaseModel):
    """Natural-language + structured digest of the repository used in prompts."""

    name: str
    description: str = ""
    languages: dict[str, int] = Field(default_factory=dict)
    entry_points: list[str] = Field(default_factory=list)
    api_routes: list[str] = Field(default_factory=list)
    test_frameworks: list[str] = Field(default_factory=list)
    build_commands: list[str] = Field(default_factory=list)
    package_manifests: list[str] = Field(default_factory=list)
    notable_files: list[str] = Field(default_factory=list)
    total_files: int = 0
    total_symbols: int = 0
    observations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

class Chunk(BaseModel):
    """Indexable unit of repository content with provenance."""

    chunk_id: str
    repo_id: str
    path: str
    language: str
    chunk_type: Literal["source", "symbol", "doc", "config", "test", "readme", "git_meta"]
    line_start: int = 0
    line_end: int = 0
    symbol: str = ""
    text: str
    tokens: list[str] = Field(default_factory=list)

    def provenance(self, reason: str, score: float) -> "Provenance":
        return Provenance(
            path=self.path, line_start=self.line_start, line_end=self.line_end,
            symbol=self.symbol, reason=reason, score=round(score, 4),
            chunk_id=self.chunk_id,
        )


class Provenance(BaseModel):
    """Why a piece of context was retrieved — every retrieved item carries one."""

    path: str
    line_start: int = 0
    line_end: int = 0
    symbol: str = ""
    reason: str
    score: float = 0.0
    chunk_id: str = ""


class RetrievedContext(BaseModel):
    items: list[Chunk] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    total_chars: int = 0
    queries: list[str] = Field(default_factory=list)

    def render(self, max_chars: int) -> str:
        """Render retrieved chunks with file/line headers, respecting budget."""
        parts: list[str] = []
        used = 0
        for chunk in self.items:
            header = f"# {chunk.path}" + (f"  lines {chunk.line_start}-{chunk.line_end}" if chunk.line_start else "")
            if chunk.symbol:
                header += f"  ({chunk.symbol})"
            body = f"{header}\n{chunk.text}"
            if used + len(body) > max_chars and parts:
                parts.append(f"# [context truncated at budget: {len(parts)} of {len(self.items)} chunks shown]")
                break
            parts.append(body)
            used += len(body)
        return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

class PlanStep(BaseModel):
    id: str
    title: str
    description: str = ""
    files: list[str] = Field(default_factory=list)
    kind: Literal["modify", "create", "delete", "test", "validate", "research"] = "modify"
    validation: str = ""


class Plan(BaseModel):
    task: str
    task_type: Literal["feature", "bugfix", "refactor", "review", "investigation", "security", "explain", "mixed"] = "feature"
    summary: str = ""
    affected_files: list[str] = Field(default_factory=list)
    dependency_impact: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    tests_required: list[str] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    generated_by: Literal["llm", "deterministic"] = "deterministic"
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Tools / execution
# --------------------------------------------------------------------------- #

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    phase: AgentPhase = AgentPhase.IMPLEMENTING
    started_at: datetime = Field(default_factory=utcnow)


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0
    artifacts: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Changes
# --------------------------------------------------------------------------- #

class FileChange(BaseModel):
    path: str
    change_type: ChangeType
    before_text: str = ""
    after_text: str = ""
    diff: str = ""
    description: str = ""


# --------------------------------------------------------------------------- #
# Tests / failures
# --------------------------------------------------------------------------- #

class TestCaseResult(BaseModel):
    name: str
    status: Literal["passed", "failed", "error", "skipped", "unknown"]
    message: str = ""
    file: str = ""
    duration_ms: int = 0


class TestRun(BaseModel):
    command: str
    exit_code: int | None = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    all_passed: bool = False
    cases: list[TestCaseResult] = Field(default_factory=list)
    raw_output_tail: str = ""
    duration_ms: int = 0
    ran_at: datetime = Field(default_factory=utcnow)


class FailureAnalysis(BaseModel):
    category: FailureCategory = "unknown"
    error: str = ""
    location: str = ""
    likely_cause: str = ""
    related_files: list[str] = Field(default_factory=list)
    recommended_repair: str = ""
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Review / security
# --------------------------------------------------------------------------- #

class ReviewFinding(BaseModel):
    id: str
    category: ReviewCategory
    severity: Severity = Severity.LOW
    path: str
    line: int = 0
    end_line: int = 0
    symbol: str = ""
    evidence: str
    explanation: str
    recommended_fix: str = ""
    source: Literal["static", "llm"] = "static"


class ReviewResult(BaseModel):
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = ""
    files_reviewed: list[str] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=utcnow)

    def counts_by_severity(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out


class SecurityFinding(BaseModel):
    id: str
    rule_id: str
    title: str
    severity: Severity
    path: str
    line: int = 0
    evidence: str
    explanation: str
    recommended_fix: str = ""


class SecurityScanResult(BaseModel):
    findings: list[SecurityFinding] = Field(default_factory=list)
    files_scanned: int = 0
    scanned_at: datetime = Field(default_factory=utcnow)

    def counts_by_severity(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out


# --------------------------------------------------------------------------- #
# Agent runs / events / reports
# --------------------------------------------------------------------------- #

class AgentEvent(BaseModel):
    ts: datetime = Field(default_factory=utcnow)
    run_id: str = ""
    phase: AgentPhase = AgentPhase.PENDING
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class StageObservability(BaseModel):
    stage: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    errors: int = 0


class RunObservability(BaseModel):
    run_id: str
    stages: list[StageObservability] = Field(default_factory=list)
    model_calls_total: int = 0
    tool_calls_total: int = 0
    repair_iterations: int = 0
    failures: int = 0
    retries: int = 0
    llm_input_chars: int = 0
    llm_output_chars: int = 0


class RepairCycle(BaseModel):
    iteration: int
    failure: FailureAnalysis
    attempted_files: list[str] = Field(default_factory=list)
    applied_changes: int = 0
    resolved: bool = False


class FinalReport(BaseModel):
    run_id: str
    task: str
    repo_name: str
    repo_id: str
    status: RunStatus
    summary: str = ""
    plan: Plan | None = None
    files_changed: list[FileChange] = Field(default_factory=list)
    tests_run: list[TestRun] = Field(default_factory=list)
    test_results_summary: str = ""
    failures_encountered: list[FailureAnalysis] = Field(default_factory=list)
    repairs_performed: list[RepairCycle] = Field(default_factory=list)
    review: ReviewResult | None = None
    security: SecurityScanResult | None = None
    remaining_risks: list[str] = Field(default_factory=list)
    validation_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    context_provenance: list[Provenance] = Field(default_factory=list)
    observability: RunObservability | None = None
    generated_at: datetime = Field(default_factory=utcnow)
    workspace_path: str = ""
    verified: bool = False

    def to_markdown(self) -> str:
        """Render the report as markdown for humans."""
        lines: list[str] = [
            f"# CodeWeaver Engineering Report",
            f"",
            f"- **Run:** `{self.run_id}`",
            f"- **Task:** {self.task}",
            f"- **Repository:** {self.repo_name} (`{self.repo_id}`)",
            f"- **Status:** {self.status.value.upper()}",
            f"- **Verified:** {'yes — validation passed' if self.verified else 'no — success NOT claimed'}",
            f"",
            f"## Summary",
            self.summary or "_(no summary)_",
        ]
        if self.plan:
            lines += ["", "## Plan", f"*Strategy: {self.plan.generated_by}*"]
            for step in self.plan.steps:
                files = f" — `{', '.join(step.files)}`" if step.files else ""
                lines.append(f"{step.id}. **{step.title}**{files}")
                if step.description:
                    lines.append(f"   {step.description}")
        if self.files_changed:
            lines += ["", "## Files Changed"]
            for ch in self.files_changed:
                lines.append(f"- `{ch.path}` ({ch.change_type}): {ch.description or 'no description'}")
        if self.tests_run:
            lines += ["", "## Tests"]
            for tr in self.tests_run:
                status = "PASSED" if tr.all_passed else "FAILED"
                lines.append(f"- `{tr.command}` → {status} ({tr.passed} passed, {tr.failed} failed, {tr.errors} errors)")
        if self.failures_encountered:
            lines += ["", "## Failures Encountered"]
            for fa in self.failures_encountered:
                lines.append(f"- [{fa.category}] {fa.error[:200]} — {fa.likely_cause[:200]}")
        if self.repairs_performed:
            lines += ["", "## Repairs Performed"]
            for rc in self.repairs_performed:
                resolved = "resolved" if rc.resolved else "not resolved"
                lines.append(f"- Iteration {rc.iteration}: {rc.failure.category} → {resolved} ({rc.applied_changes} change(s))")
        if self.review and self.review.findings:
            lines += ["", "## Review Findings"]
            for f in self.review.findings:
                lines.append(f"- [{f.severity.value}] `{f.path}:{f.line}` ({f.category}): {f.explanation[:200]}")
        if self.security and self.security.findings:
            lines += ["", "## Security Findings"]
            for f in self.security.findings:
                lines.append(f"- [{f.severity.value}] `{f.path}:{f.line}` {f.rule_id}: {f.title} — {f.explanation[:160]}")
        if self.remaining_risks:
            lines += ["", "## Remaining Risks"]
            lines += [f"- {r}" for r in self.remaining_risks]
        if self.validation_evidence:
            lines += ["", "## Validation Evidence"]
            lines += [f"- {v}" for v in self.validation_evidence]
        if self.unsupported_claims:
            lines += ["", "## Unsupported Claims Blocked"]
            lines += [f"- {u}" for u in self.unsupported_claims]
        lines.append("")
        return "\n".join(lines)


def dumps_json(model: BaseModel) -> str:
    return model.model_dump_json()


def loads_json(model_cls: type[BaseModel], raw: str | bytes | None) -> BaseModel | None:
    if not raw:
        return None
    try:
        return model_cls.model_validate_json(raw)
    except Exception:
        try:
            return model_cls.model_validate(json.loads(raw))
        except Exception:
            return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from LLM text; tolerate code fences."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None
