"""CodeWeaver agent pipeline.

Orchestrates the engineering workflow as typed phases with structured state:

  ANALYZING → RETRIEVING → PLANNING → IMPLEMENTING → TESTING ⇄ DEBUGGING
  → REVIEWING → COMPLETED/FAILED → FinalReport

Every phase emits real AgentEvents (persisted + streamed over SSE), every
tool call is recorded, and the run can never claim success without actual
validation evidence.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from app.agents.engineer import (
    TransformError, find_list_returning_functions, fix_module_import_path,
    paginate_function, rewrite_import_to_symbols,
)
from app.config import get_settings
from app.db import Database, AgentRunRow, FileChangeRow, RunEventRow, TestRunRow, ToolExecutionRow
from app.failure.analyzer import FailureAnalyzer
from app.llm.deterministic import classify_task, deterministic_plan, wants_pagination
from app.llm.service import LLMService
from app.review.reviewer import CodeReviewer
from app.schemas import (
    AgentEvent, AgentMode, AgentPhase, Chunk, FailureAnalysis, FinalReport,
    Plan, Provenance, RepairCycle, RepoGraph, RepoSummary, ReviewFinding,
    RunObservability, RunStatus, SecurityScanResult, StageObservability,
    TestRun, ToolResult, loads_json, utcnow,
)
from app.security.scanner import SecurityScanner
from app.tools.registry import ToolContext, ToolRegistry
from app.workspace import Workspace

logger = logging.getLogger("codeweaver.pipeline")

EventSink = Callable[[AgentEvent], Awaitable[None]]

ANALYST_FILES_MAX = 6


class CodeWeaverPipeline:
    def __init__(self, db: Database, retrieval) -> None:
        self.db = db
        self.retrieval = retrieval
        self.settings = get_settings()
        self.failure_analyzer = FailureAnalyzer()
        self.reviewer = CodeReviewer()
        self.security_scanner = SecurityScanner()

    # ------------------------------------------------------------------ #

    async def execute(
        self,
        run_id: str,
        repo_row: AgentRunRow | object,
        task: str,
        mode: AgentMode,
        event_sink: EventSink | None = None,
    ) -> FinalReport:
        """Run the full pipeline. Never raises: failures produce FAILED runs."""
        self.event_sink = event_sink
        obs = RunObservability(run_id=run_id)
        llm = LLMService()
        started = utcnow()

        graph = loads_json(RepoGraph, getattr(repo_row, "graph_json", ""))
        summary = loads_json(RepoSummary, getattr(repo_row, "summary_json", ""))
        repo_id = getattr(repo_row, "id", "")
        repo_name = getattr(repo_row, "name", "unknown")
        snapshot = Path(getattr(repo_row, "local_path", ""))

        context = None
        plan: Plan | None = None
        workspace: Workspace | None = None
        test_runs: list[TestRun] = []
        repairs: list[RepairCycle] = []
        failures: list[FailureAnalysis] = []
        security: SecurityScanResult | None = None
        review = None
        unsupported: list[str] = []
        validation_evidence: list[str] = []
        stage_start = utcnow()

        def stage(name: str) -> StageObservability:
            nonlocal stage_start
            stage_start = utcnow()
            return StageObservability(stage=name, started_at=stage_start)

        def end_stage(st: StageObservability, model_calls: int = 0, tool_calls: int = 0, errors: int = 0) -> None:
            st.finished_at = utcnow()
            st.duration_ms = int((st.finished_at - st.started_at).total_seconds() * 1000)
            st.model_calls, st.tool_calls, st.errors = model_calls, tool_calls, errors
            obs.stages.append(st)

        try:
            # ---------------- ANALYZING ---------------- #
            st = stage("ANALYZING")
            await self._set_phase(run_id, AgentPhase.ANALYZING, RunStatus.RUNNING)
            if graph is None or summary is None:
                raise RuntimeError("repository is not indexed; run ingestion first")
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.ANALYZING,
                message=f"Repository analyzed: {summary.total_files} files, {summary.total_symbols} symbols, languages {summary.languages}",
                data={"observations": summary.observations, "entry_points": summary.entry_points[:5]},
            ))
            end_stage(st)

            # ---------------- RETRIEVING ---------------- #
            st = stage("RETRIEVING")
            context = await self.retrieval.retrieve(repo_id, task, graph=graph)
            files_touched = sorted({p.path for p in context.provenance})
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.RETRIEVING,
                message=f"Retrieved {len(context.items)} chunks across {len(files_touched)} files (hybrid: BM25 + vectors + graph)",
                data={"provenance": [p.model_dump() for p in context.provenance[:25]]},
            ))
            end_stage(st)

            # ---------------- PLANNING ---------------- #
            st = stage("PLANNING")
            await self._set_phase(run_id, AgentPhase.PLANNING)
            candidate_paths = list(dict.fromkeys([p.path for p in context.provenance]))
            plan = deterministic_plan(task, summary, graph, candidate_paths)
            if llm.llm_available:
                improved = await llm.plan_with_llm(
                    task, summary.model_dump_json(), context.render(self.settings.max_context_chars), plan,
                )
                if improved is not None:
                    plan = improved
            await self._persist_plan(run_id, plan)
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.PLANNING,
                message=f"Plan generated ({plan.generated_by}): {len(plan.steps)} steps, {len(plan.affected_files)} affected file(s)",
                data={"plan": plan.model_dump(mode="json")},
            ))
            end_stage(st, model_calls=llm.model_calls)

            task_type = plan.task_type
            edit_capable = mode == AgentMode.WORKSPACE_EDIT and task_type in ("feature", "bugfix", "refactor", "mixed")

            # ---------------- IMPLEMENTING ---------------- #
            tool_calls = 0
            if edit_capable:
                st = stage("IMPLEMENTING")
                await self._set_phase(run_id, AgentPhase.IMPLEMENTING)
                workspace = Workspace(run_id, snapshot, mode)
                registry = self._make_registry(workspace, graph, repo_id)
                n_changes = await self._implement(
                    run_id, task, plan, context, workspace, registry, llm, graph, obs, unsupported,
                )
                tool_calls += n_changes
                await self.emit(AgentEvent(
                    run_id=run_id, phase=AgentPhase.IMPLEMENTING,
                    message=f"Implementation finished: {len(workspace.changes())} file change(s)",
                    data={"files": [c.path for c in workspace.changes()]},
                ))
                end_stage(st, model_calls=llm.model_calls, tool_calls=n_changes)
            elif mode == AgentMode.READ_ONLY:
                await self.emit(AgentEvent(
                    run_id=run_id, phase=AgentPhase.IMPLEMENTING, level="warning",
                    message="READ_ONLY mode: analysis will be produced without modifications",
                ))

            # ---------------- TESTING ---------------- #
            st = stage("TESTING")
            await self._set_phase(run_id, AgentPhase.TESTING)
            workspace = workspace or Workspace(run_id, snapshot, AgentMode.READ_ONLY)
            registry = self._make_registry(workspace, graph, repo_id)
            tests_ok, runs = await self._run_validation(
                run_id, task, task_type, registry, llm, obs,
            )
            test_runs.extend(runs)
            end_stage(st, tool_calls=len(runs))

            # ---------------- DEBUGGING / REPAIR LOOP ---------------- #
            last_run = test_runs[-1] if test_runs else None
            if edit_capable and last_run is not None and not last_run.all_passed:
                st = stage("DEBUGGING")
                await self._set_phase(run_id, AgentPhase.DEBUGGING)
                ok, repair_list, failure_list, n_tools = await self._repair_loop(
                    run_id, task, context, workspace, registry, llm, graph, obs, test_runs,
                )
                repairs.extend(repair_list)
                failures.extend(failure_list)
                obs.repair_iterations = len(repair_list)
                tool_calls += n_tools
                end_stage(st, model_calls=llm.model_calls, tool_calls=n_tools)
            elif last_run is not None and not last_run.all_passed and task_type == "investigation":
                st = stage("DEBUGGING")
                await self._set_phase(run_id, AgentPhase.DEBUGGING)
                analysis = self.failure_analyzer.analyze(last_run.raw_output_tail, repo_root=workspace.root)
                refined = await llm.analyze_failure_llm(task, last_run.raw_output_tail, "", analysis)
                final_analysis = refined or analysis
                failures.append(final_analysis)
                await self.emit(AgentEvent(
                    run_id=run_id, phase=AgentPhase.DEBUGGING,
                    message=f"Failure analyzed: [{final_analysis.category}] {final_analysis.error[:120]}",
                    data=final_analysis.model_dump(mode="json"),
                ))
                end_stage(st, model_calls=llm.model_calls)

            # ---------------- REVIEWING ---------------- #
            st = stage("REVIEWING")
            await self._set_phase(run_id, AgentPhase.REVIEWING)
            review, security = await self._review_phase(
                run_id, task, task_type, workspace, graph, registry, llm, obs,
            )
            end_stage(st, model_calls=llm.model_calls, tool_calls=1)

            # ---------------- FINALIZE ---------------- #
            changed = workspace.changes() if workspace else []
            # Verification reflects the CURRENT state: the last test execution
            # (post-repair) is what proves or disproves the final changes.
            tests_all_passed = bool(test_runs) and test_runs[-1].all_passed
            has_runner = bool(test_runs)

            if edit_capable:
                if tests_all_passed and has_runner:
                    status, verified = RunStatus.COMPLETED, True
                    if changed:
                        validation_evidence.append(f"changes applied — test suite passed ({test_runs[-1].passed} passed / {test_runs[-1].failed} failed)")
                    else:
                        validation_evidence.append("tests already passing — no changes needed")
                elif changed and not has_runner:
                    status, verified = RunStatus.COMPLETED, False
                    unsupported.append("no test runner detected; success is not claimed without test verification")
                    validation_evidence.append("no test runner available — changes unverified by tests")
                elif not has_runner:
                    status, verified = RunStatus.COMPLETED, False
                    unsupported.append("no test runner detected")
                    validation_evidence.append("no test runner available — success cannot be verified")
                else:
                    status, verified = RunStatus.FAILED, False
                    validation_evidence.append(f"tests still failing after {obs.repair_iterations} repair iteration(s)")
            else:
                # Analysis / review / investigation run
                status, verified = RunStatus.COMPLETED, True
                if review is not None:
                    validation_evidence.append(f"static review produced {len(review.findings)} evidence-backed finding(s)")
                if security is not None:
                    validation_evidence.append(f"security scan produced {len(security.findings)} finding(s) over {security.files_scanned} files")
                if test_runs:
                    validation_evidence.append(
                        f"test execution evidence: exit={test_runs[-1].exit_code}, {test_runs[-1].passed} passed / {test_runs[-1].failed} failed"
                    )
                if context is not None:
                    validation_evidence.append(f"context provenance recorded for {len(context.provenance)} retrieved chunk(s)")

            if edit_capable and not changed and task_type in ("feature", "bugfix", "refactor"):
                unsupported.append("no code change could be applied safely by the available engines (deterministic/LLM)")

            obs.model_calls_total = llm.model_calls
            obs.tool_calls_total = tool_calls + len(test_runs)
            obs.llm_input_chars = llm.input_chars
            obs.llm_output_chars = llm.output_chars
            obs.failures = len(failures)

            if llm.llm_available:
                facts = (
                    f"changed files: {[c.path for c in changed]}; tests: "
                    f"{[(t.passed, t.failed) for t in test_runs]}; review findings: "
                    f"{len(review.findings) if review else 0}; security: {len(security.findings) if security else 0}"
                )
                llm_summary = await llm.summarize(task, facts)
            else:
                llm_summary = ""

            report = FinalReport(
                run_id=run_id, task=task, repo_name=repo_name, repo_id=repo_id,
                status=status,
                summary=llm_summary or self._deterministic_summary(
                    task, task_type, changed, test_runs, review, security, repairs,
                ),
                plan=plan, files_changed=changed, tests_run=test_runs,
                test_results_summary=self._test_summary(test_runs),
                failures_encountered=failures, repairs_performed=repairs,
                review=review, security=security,
                remaining_risks=plan.risks if plan else [],
                validation_evidence=validation_evidence,
                unsupported_claims=unsupported,
                context_provenance=context.provenance if context else [],
                observability=obs,
                workspace_path=str(workspace.root) if workspace else "",
                verified=verified,
            )
            await self._persist_report(run_id, report)
            await self._set_phase(run_id, AgentPhase.COMPLETED if status == RunStatus.COMPLETED else AgentPhase.FAILED, status)
            await self.emit(AgentEvent(
                run_id=run_id,
                phase=AgentPhase.COMPLETED if status == RunStatus.COMPLETED else AgentPhase.FAILED,
                message=f"Run {status.value.upper()} — verified={verified}; {len(changed)} change(s), {len(test_runs)} test execution(s)",
                data={"verified": verified},
            ))
            return report

        except Exception as exc:
            logger.exception("agent run %s failed", run_id)
            obs.model_calls_total = llm.model_calls
            status = RunStatus.FAILED
            report = FinalReport(
                run_id=run_id, task=task, repo_name=repo_name, repo_id=repo_id,
                status=status, summary=f"Run failed: {type(exc).__name__}: {exc}",
                plan=plan, tests_run=test_runs, failures_encountered=failures,
                repairs_performed=repairs, unsupported_claims=unsupported,
                context_provenance=context.provenance if context else [],
                observability=obs, verified=False,
            )
            await self._persist_report(run_id, report)
            await self._set_phase(run_id, AgentPhase.FAILED, status, error=f"{type(exc).__name__}: {exc}")
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.FAILED, level="error",
                message=f"Run failed: {type(exc).__name__}: {exc}",
            ))
            return report
        finally:
            await llm.aclose()

    # ------------------------------------------------------------------ #
    # Implementation strategies
    # ------------------------------------------------------------------ #

    async def _implement(
        self, run_id, task, plan, context, workspace, registry, llm, graph, obs, unsupported,
    ) -> int:
        tool_count = 0
        applied_any = False

        if wants_pagination(task):
            applied_any = await self._implement_pagination(
                run_id, task, plan, workspace, registry, graph, obs,
            )
            tool_count += 1 if applied_any else 0

        if not applied_any and llm.llm_available:
            for step in plan.steps:
                if step.kind != "modify":
                    continue
                step_context = context.render(self.settings.max_context_chars)
                proposal = await llm.propose_patch(task, f"{step.id}. {step.title}: {step.description}", step_context)
                if proposal is None:
                    continue
                result = await registry.execute(
                    "apply_patch",
                    {"path": proposal["file"], "old_text": proposal["old_text"],
                     "new_text": proposal["new_text"], "description": proposal.get("description", "")},
                    phase=AgentPhase.IMPLEMENTING,
                )
                await self._persist_tool(run_id, "apply_patch", {"path": proposal["file"]}, result)
                tool_count += 1
                if result.ok:
                    applied_any = True
                    await self.emit(AgentEvent(
                        run_id=run_id, phase=AgentPhase.IMPLEMENTING,
                        message=f"Patch applied to {proposal['file']} for step {step.id}",
                    ))
                else:
                    await self.emit(AgentEvent(
                        run_id=run_id, phase=AgentPhase.IMPLEMENTING, level="warning",
                        message=f"Patch rejected for {proposal['file']}: {result.error[:160]}",
                    ))

        if not applied_any and not wants_pagination(task) and not llm.llm_available:
            unsupported.append(
                "deterministic engine has no safe transformation for this task and no LLM provider is configured"
            )
        return tool_count

    async def _implement_pagination(self, run_id, task, plan, workspace, registry, graph, obs) -> bool:
        """AST-guided pagination over list-returning functions in affected files."""
        ordered = self._order_pagination_files(plan.affected_files, graph)
        applied = False
        for rel in ordered[:3]:
            if not workspace.file_exists(rel):
                continue
            source = workspace.read_file(rel)
            targets = find_list_returning_functions(source)
            if not targets:
                continue
            # Prefer endpoint-shaped names; never touch trivial helpers.
            endpoint_kws = ("list", "all", "search", "find", "browse", "index",
                            "users", "items", "notes", "feed", "catalog", "page")
            candidates = [
                t for t in targets
                if any(k in t["name"].lower() for k in endpoint_kws)
                and not t["name"].startswith("_")
            ]
            for target in candidates[:2]:
                try:
                    old_text, new_text = paginate_function(source, target["name"])
                except TransformError as exc:
                    await self.emit(AgentEvent(
                        run_id=run_id, phase=AgentPhase.IMPLEMENTING, level="warning",
                        message=f"pagination skipped for {rel}::{target['name']}: {exc}",
                    ))
                    continue
                result = await registry.execute(
                    "apply_patch",
                    {
                        "path": rel, "old_text": old_text, "new_text": new_text,
                        "description": f"add limit/offset pagination to {target['name']}()",
                    },
                    phase=AgentPhase.IMPLEMENTING,
                )
                await self._persist_tool(run_id, "apply_patch", {"path": rel, "function": target["name"]}, result)
                if result.ok:
                    applied = True
                    await self.emit(AgentEvent(
                        run_id=run_id, phase=AgentPhase.IMPLEMENTING,
                        message=f"Pagination added to {rel}::{target['name']}() (limit/offset parameters + result slicing)",
                    ))
                    source = workspace.read_file(rel)
        return applied

    def _order_pagination_files(self, files: list[str], graph: RepoGraph) -> list[str]:
        routes = graph.stats.get("routes", []) if graph else []
        route_files = {r["file"] for r in routes}
        def score(path: str) -> int:
            s = 0
            name = path.lower()
            if path in route_files:
                s += 10
            if any(k in name for k in ("api", "route", "view", "handler", "endpoint", "resource")):
                s += 5
            if any(k in name for k in ("user", "item", "list")):
                s += 2
            return -s
        return sorted(files, key=score)

    # ------------------------------------------------------------------ #
    # Validation & repair
    # ------------------------------------------------------------------ #

    async def _run_validation(self, run_id, task, task_type, registry, llm, obs) -> tuple[bool, list[TestRun]]:
        """Runs tests for code-change and investigation tasks; skips pure review."""
        if task_type in ("review", "security", "explain"):
            return False, []
        try:
            result = await registry.execute("run_tests", {}, phase=AgentPhase.TESTING)
        except Exception as exc:
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.TESTING, level="warning",
                message=f"test execution unavailable: {exc}",
            ))
            return False, []
        await self._persist_tool(run_id, "run_tests", {}, result)
        test_run = TestRun(**result.artifacts["test_run"]) if result.ok and "test_run" in result.artifacts else TestRun(command="run_tests", all_passed=False, raw_output_tail=result.output[-4000:])
        if not result.ok and "test_run" not in result.artifacts:
            test_run.raw_output_tail = (result.error or result.output)[-4000:]
        await self._persist_test_run(run_id, test_run)
        await self.emit(AgentEvent(
            run_id=run_id, phase=AgentPhase.TESTING,
            level="info" if test_run.all_passed else "warning",
            message=(
                f"Tests {'passed' if test_run.all_passed else 'FAILED'}: {test_run.passed} passed, "
                f"{test_run.failed} failed, {test_run.errors} errors (exit={test_run.exit_code})"
            ),
            data={"command": test_run.command, "exit_code": test_run.exit_code},
        ))
        return test_run.all_passed, [test_run]

    async def _repair_loop(self, run_id, task, context, workspace, registry, llm, graph, obs, test_runs):
        repairs: list[RepairCycle] = []
        failures: list[FailureAnalysis] = []
        tool_count = 0
        max_iter = self.settings.max_repair_iterations

        for iteration in range(1, max_iter + 1):
            last = test_runs[-1]
            analysis = self.failure_analyzer.analyze(
                last.raw_output_tail or (last.cases[0].message if last.cases else ""),
                repo_root=workspace.root,
            )
            if llm.llm_available:
                refined = await llm.analyze_failure_llm(task, last.raw_output_tail, context.render(12000), analysis)
                if refined is not None:
                    analysis = refined
            failures.append(analysis)
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.DEBUGGING,
                message=f"Iteration {iteration}: failure analyzed [{analysis.category}] — {analysis.likely_cause[:140]}",
                data={"analysis": analysis.model_dump(mode="json")},
            ))

            applied = await self._apply_repairs(
                run_id, analysis, workspace, registry, llm, graph, context, task,
            )
            tool_count += applied

            result = await registry.execute("run_tests", {}, phase=AgentPhase.DEBUGGING)
            await self._persist_tool(run_id, "run_tests", {}, result)
            test_run = TestRun(**result.artifacts["test_run"]) if result.ok and "test_run" in result.artifacts else TestRun(command="run_tests", all_passed=False)
            await self._persist_test_run(run_id, test_run)
            test_runs.append(test_run)
            obs.retries += 1

            cycle = RepairCycle(
                iteration=iteration, failure=analysis,
                attempted_files=analysis.related_files[:5],
                applied_changes=applied, resolved=test_run.all_passed,
            )
            repairs.append(cycle)

            if test_run.all_passed:
                await self.emit(AgentEvent(
                    run_id=run_id, phase=AgentPhase.DEBUGGING,
                    message=f"Repair iteration {iteration} succeeded: tests now pass ({test_run.passed} passed)",
                ))
                break
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.DEBUGGING, level="warning",
                message=f"Repair iteration {iteration} did not resolve the failure ({test_run.failed + test_run.errors} failing)",
            ))
        return test_runs[-1].all_passed, repairs, failures, tool_count

    async def _apply_repairs(self, run_id, analysis, workspace, registry, llm, graph, context, task) -> int:
        applied = 0
        # 1) Deterministic import repairs
        if analysis.category in ("import", "dependency") and graph is not None:
            module_index = self._module_index(graph)
            symbol_map = {s.name: s.path for s in graph.symbols}
            missing_refs = self._extract_missing_modules(
                analysis.error + "\n" + "\n".join(analysis.evidence)
            )
            candidate_files = [rel.split(":")[0] for rel in analysis.related_files]
            if not candidate_files and missing_refs:
                # No usable traceback frames: find files importing the missing module.
                for rel in workspace.list_files("**/*.py")[:500]:
                    try:
                        head = workspace.read_file(rel)[:8000]
                    except Exception:
                        continue
                    if any(f"from {m} import" in head or f"import {m}" in head for m in missing_refs):
                        candidate_files.append(rel)
            for rel_path in candidate_files:
                if not workspace.file_exists(rel_path):
                    continue
                source = workspace.read_file(rel_path)
                for missing in missing_refs:
                    fix = fix_module_import_path(source, missing, module_index)
                    if fix:
                        old, new = fix
                        result = await registry.execute(
                            "apply_patch",
                            {"path": rel_path, "old_text": old, "new_text": new,
                             "description": f"repair import of {missing}"},
                            phase=AgentPhase.DEBUGGING,
                        )
                        await self._persist_tool(run_id, "apply_patch", {"path": rel_path, "repair": missing}, result)
                        if result.ok:
                            applied += 1
                            await self.emit(AgentEvent(
                                run_id=run_id, phase=AgentPhase.DEBUGGING,
                                message=f"Import repair applied in {rel_path}: {missing} -> {new.split('import')[0].strip()}",
                            ))
                            source = workspace.read_file(rel_path)
                # Symbol-level repair: the module is gone but its symbols moved.
                for missing in missing_refs:
                    fix = rewrite_import_to_symbols(source, missing, symbol_map)
                    if fix:
                        old, new = fix
                        result = await registry.execute(
                            "apply_patch",
                            {"path": rel_path, "old_text": old, "new_text": new,
                             "description": f"re-point import of moved symbols from {missing}"},
                            phase=AgentPhase.DEBUGGING,
                        )
                        await self._persist_tool(run_id, "apply_patch", {"path": rel_path, "repair": f"symbols from {missing}"}, result)
                        if result.ok:
                            applied += 1
                            await self.emit(AgentEvent(
                                run_id=run_id, phase=AgentPhase.DEBUGGING,
                                message=f"Symbol import repair applied in {rel_path}: {new.strip()[:100]}",
                            ))
                            source = workspace.read_file(rel_path)
        # 2) LLM-guided repair
        if llm.llm_available:
            failure_context = context.render(self.settings.max_context_chars // 2)
            proposal = await llm.propose_patch(
                task, f"Fix this failure: {analysis.error[:300]}. Cause: {analysis.likely_cause[:300]}. Suggested: {analysis.recommended_repair[:300]}",
                failure_context,
            )
            if proposal:
                result = await registry.execute(
                    "apply_patch",
                    {"path": proposal["file"], "old_text": proposal["old_text"],
                     "new_text": proposal["new_text"], "description": "repair iteration"},
                    phase=AgentPhase.DEBUGGING,
                )
                await self._persist_tool(run_id, "apply_patch", {"path": proposal["file"], "repair": "llm"}, result)
                if result.ok:
                    applied += 1
                    await self.emit(AgentEvent(
                        run_id=run_id, phase=AgentPhase.DEBUGGING,
                        message=f"LLM repair patch applied to {proposal['file']}",
                    ))
        if applied == 0:
            await self.emit(AgentEvent(
                run_id=run_id, phase=AgentPhase.DEBUGGING, level="warning",
                message="no safe automatic repair available for this failure",
            ))
        return applied

    def _module_index(self, graph: RepoGraph) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in graph.files:
            p = f.path
            mod = p.replace("/", ".").removesuffix(".py").removesuffix(".__init__")
            out[mod] = p
            out[p.split("/")[-1].removesuffix(".py")] = p
        return out

    def _extract_missing_modules(self, text: str) -> list[str]:
        import re

        found = []
        for m in re.finditer(r"No module named '([\w.]+)'", text):
            found.append(m.group(1))
        for m in re.finditer(r"ImportError: cannot import name '(\w+)' from '([\w.]+)'", text):
            found.append(m.group(2))
        for m in re.finditer(r"Cannot find module '([^']+)'", text):
            found.append(m.group(1))
        return list(dict.fromkeys(found))[:5]

    # ------------------------------------------------------------------ #
    # Review
    # ------------------------------------------------------------------ #

    async def _review_phase(self, run_id, task, task_type, workspace, graph, registry, llm, obs):
        review_paths = [c.path for c in workspace.changes()]
        if task_type in ("review", "security") or not review_paths:
            review_paths = [f.path for f in (graph.files if graph else []) if f.role in ("source", "test")][:60]
        review = self.reviewer.review_files(workspace.root, review_paths, graph=graph)
        if llm.llm_available:
            diff_text = workspace.git_diff()[:30000]
            llm_findings = await llm.review_llm(diff_text, review.summary)
            review.findings.extend(llm_findings)
        security_result = await registry.execute("run_security_scan", {}, phase=AgentPhase.REVIEWING)
        await self._persist_tool(run_id, "run_security_scan", {}, security_result)
        security = None
        if security_result.ok and "scan" in security_result.artifacts:
            security = SecurityScanResult(**security_result.artifacts["scan"])
        else:
            security = await asyncio.to_thread(self.security_scanner.scan_directory, workspace.root)
        await self.emit(AgentEvent(
            run_id=run_id, phase=AgentPhase.REVIEWING,
            message=(
                f"Review complete: {len(review.findings)} finding(s); security: {len(security.findings)} finding(s) "
                f"over {security.files_scanned} file(s)"
            ),
            data={"review_counts": review.counts_by_severity(), "security_counts": security.counts_by_severity()},
        ))
        return review, security

    # ------------------------------------------------------------------ #
    # Helpers: events, persistence
    # ------------------------------------------------------------------ #

    async def emit(self, event: AgentEvent) -> None:
        async with self.db.session() as session:
            session.add(RunEventRow(
                run_id=event.run_id, ts=event.ts, phase=event.phase.value,
                level=event.level, message=event.message,
                data_json=event.model_dump_json(include={"data"}),
            ))
            await session.commit()
        if self.event_sink is not None:
            try:
                await self.event_sink(event)
            except Exception:  # noqa: BLE001 — a broken SSE client must not kill the run
                logger.debug("event sink failed", exc_info=True)

    async def _set_phase(self, run_id, phase, status=None, error: str = "") -> None:
        async with self.db.session() as session:
            row = await session.get(AgentRunRow, run_id)
            if row is not None:
                row.phase = phase.value
                if status is not None:
                    row.status = status.value
                    if status in (RunStatus.RUNNING,):
                        row.started_at = row.started_at or utcnow()
                    if status in (RunStatus.COMPLETED, RunStatus.FAILED):
                        row.finished_at = utcnow()
                if error:
                    row.error = error
                await session.commit()

    async def _persist_plan(self, run_id, plan: Plan) -> None:
        async with self.db.session() as session:
            row = await session.get(AgentRunRow, run_id)
            if row is not None:
                row.plan_json = plan.model_dump_json()
                await session.commit()

    async def _persist_tool(self, run_id, name, args, result: ToolResult) -> None:
        async with self.db.session() as session:
            session.add(ToolExecutionRow(
                run_id=run_id, name=name, phase="IMPLEMENTING", ok=result.ok,
                arguments_json=str(args)[:2000], output=result.output[:8000],
                error=result.error[:4000], duration_ms=result.duration_ms,
            ))
            await session.commit()

    async def _persist_test_run(self, run_id, tr: TestRun) -> None:
        async with self.db.session() as session:
            session.add(TestRunRow(
                run_id=run_id, command=tr.command, exit_code=tr.exit_code,
                passed=tr.passed, failed=tr.failed, errors=tr.errors,
                skipped=tr.skipped, all_passed=tr.all_passed,
                cases_json=tr.model_dump_json(include={"cases"}),
                raw_output_tail=tr.raw_output_tail[-8000:], duration_ms=tr.duration_ms,
            ))
            await session.commit()

    async def _persist_report(self, run_id, report: FinalReport) -> None:
        async with self.db.session() as session:
            row = await session.get(AgentRunRow, run_id)
            if row is not None:
                row.report_json = report.model_dump_json()
                row.observability_json = (report.observability or RunObservability(run_id=run_id)).model_dump_json()
                await session.commit()
            # file changes
            for ch in report.files_changed:
                session.add(FileChangeRow(
                    run_id=run_id, path=ch.path, change_type=ch.change_type,
                    diff=ch.diff, description=ch.description,
                    before_text=ch.before_text[-20000:], after_text=ch.after_text[-20000:],
                ))
            await session.commit()

    def _make_registry(self, workspace: Workspace, graph, repo_id: str) -> ToolRegistry:
        ctx = ToolContext(workspace=workspace, graph=graph, retrieval=self.retrieval, repo_id=repo_id)
        return ToolRegistry(ctx)

    def _test_summary(self, runs: list[TestRun]) -> str:
        if not runs:
            return "no test execution performed"
        last = runs[-1]
        return (
            f"{len(runs)} execution(s); last: {last.passed} passed, {last.failed} failed, "
            f"{last.errors} errors, {last.skipped} skipped — {'ALL PASSED' if last.all_passed else 'NOT PASSING'}"
        )

    def _deterministic_summary(self, task, task_type, changed, test_runs, review, security, repairs) -> str:
        parts = [f"Task type: {task_type}."]
        if changed:
            parts.append(f"Modified {len(changed)} file(s): {', '.join(c.path for c in changed[:6])}.")
        else:
            parts.append("No files were modified (analysis-only run).")
        if test_runs:
            last = test_runs[-1]
            parts.append(
                f"Validation: {last.passed} passed / {last.failed} failed ({'passing' if last.all_passed else 'still failing'})."
            )
        if repairs:
            parts.append(f"{len(repairs)} repair iteration(s) performed.")
        if review is not None and review.findings:
            parts.append(f"Review: {len(review.findings)} finding(s).")
        if security is not None and security.findings:
            parts.append(f"Security: {len(security.findings)} finding(s).")
        return " ".join(parts)
