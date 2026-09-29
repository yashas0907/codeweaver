"""Deterministic reasoning engine.

When no LLM provider is configured, CodeWeaver still performs real engineering
work with rule-based analysis: task classification, plan construction from
repository-graph facts, and mechanical code transformations (e.g. AST-based
pagination). Nothing here fabricates success: when a transformation is not
possible it reports that honestly.
"""

from __future__ import annotations

import re

from app.schemas import Plan, PlanStep, RepoGraph, RepoSummary

TASK_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("security", re.compile(r"\b(security|vulnerab|injection|secret)\b", re.I)),
    ("review", re.compile(r"\b(review|audit|inspect|assess)\b", re.I)),
    ("bugfix", re.compile(r"\b(fix(?:es|ed|ing)?|repair(?:s|ed|ing)?|resolve|patch|amend|correct|rectify|debug)\b", re.I)),
    ("investigation", re.compile(r"\b(why|failing|fails|broken|regression|root cause|diagnos)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|duplicat|clean ?up|tidy|reorganize|extract)\b", re.I)),
    ("explain", re.compile(r"\b(explain|how does|what does|walk me|architecture|overview|document)\b", re.I)),
    ("feature", re.compile(r"\b(add|implement|support|introduce|create|enable|pagination|feature)\b", re.I)),
)

PAGINATION_TASK = re.compile(r"\bpagination\b|\bpaginate\b|\bpage(?:s| size)?\b", re.I)
FEATURE_TASK = re.compile(r"\badd\b|\bimplement\b|\bsupport\b", re.I)


def classify_task(task: str) -> str:
    for task_type, pattern in TASK_TYPE_PATTERNS:
        if pattern.search(task):
            return task_type
    return "feature"


def wants_pagination(task: str) -> bool:
    return bool(PAGINATION_TASK.search(task)) and bool(FEATURE_TASK.search(task) or "pagination" in task.lower())


def deterministic_plan(
    task: str,
    summary: RepoSummary,
    graph: RepoGraph,
    candidate_paths: list[str],
    task_type: str | None = None,
) -> Plan:
    """Build an explicit engineering plan from repository facts + retrieval."""
    task_type = task_type or classify_task(task)
    plan = Plan(task=task, task_type=task_type)  # type: ignore[arg-type]
    plan.generated_by = "deterministic"

    affected = [p for p in candidate_paths if p.endswith(".py")][:8]
    plan.affected_files = list(affected)

    tests: list[str] = []
    for path in affected:
        tests.extend(graph.tests_for(path))
    plan.tests_required = sorted(set(tests))[:6]

    plan.validation_commands = list(summary.build_commands) or ["python -m pytest"]

    # ---- risk assessment from the graph -------------------------------
    for path in affected[:4]:
        importers = [e.src for e in graph.edges if e.kind == "imports" and e.dst == path]
        if importers:
            plan.dependency_impact.append(f"{path} is imported by {len(importers)} module(s): {', '.join(importers[:5])}")
            plan.risks.append(f"changing {path} may affect its importers: {', '.join(importers[:3])}")
    if not plan.tests_required:
        plan.risks.append("no test file is linked to the affected modules; rely on compile/type validation")

    step_no = 1

    def add(title: str, files: list[str], kind: str, description: str = "", validation: str = "") -> None:
        nonlocal step_no
        plan.steps.append(PlanStep(
            id=f"S{step_no}", title=title, files=files, kind=kind,  # type: ignore[arg-type]
            description=description, validation=validation,
        ))
        step_no += 1

    if task_type in ("review", "security", "explain", "investigation"):
        add("Retrieve and study relevant code", affected[:5], "research",
            "read the affected files and their callers from the repository graph")
        if task_type in ("review", "security"):
            add("Run static review + security scan", [], "validate",
                "evidence-based review findings and security rules over the affected area")
        if task_type == "investigation":
            add("Run tests and analyze failures", [], "validate",
                "execute the test suite, parse failures, correlate with git history")
        add("Produce engineering report", [], "validate", "report findings with file/line evidence")
        return plan

    if task_type == "refactor":
        add("Identify duplicated/brittle regions", affected[:5], "research",
            "duplication scan + structure findings from the repository graph")
        add("Apply safe refactor", affected, "modify",
            "mechanical, behavior-preserving edits where safely derivable")
        add("Run tests", plan.tests_required, "test", "validate behavior is preserved")
        return plan

    # feature / bugfix / mixed
    if wants_pagination(task):
        targets = [p for p in affected if "route" in str(graph.stats.get("routes", [])) or p.endswith(("api.py", "routes.py", "views.py", "service.py"))] or affected
        add("Study list-returning endpoints", targets[:4], "research",
            "identify the functions returning collections that need pagination")
        add("Add pagination parameters and slicing", targets[:3], "modify",
            "add limit/offset parameters (AST-guided), slice returned collections")
    else:
        add("Study affected modules", affected[:4], "research",
            "read the relevant code and its tests before editing")
        add("Implement the change", affected, "modify",
            "apply minimal, focused edits")
    if plan.tests_required:
        add("Run affected tests", plan.tests_required, "test", "verify the change")
    else:
        add("Run full test suite", [], "test", "no module-specific tests found; run the suite")
    add("Review changes and finalize report", [], "validate",
        "static review of the diff; verify before claiming success")
    return plan
