"""LLMService: the only door stages use to reach a model.

- Resolves the configured provider once (deterministic / http_compatible / none).
- Tracks observability counters (calls, chars) that the runner persists.
- Every high-level operation returns structured, schema-validated output:
  when the model's JSON is malformed, we repair once, then fall back to the
  deterministic engine — never to unstructured guessing.
"""

from __future__ import annotations

import json
import logging
from pydantic import ValidationError

from app.config import get_settings
from app.llm.base import LLMResponse, ProviderError
from app.schemas import (
    FailureAnalysis, Plan, PlanStep, ReviewFinding, extract_json_object,
)

log = logging.getLogger("codeweaver.llm")


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = None
        self.provider_name = self.settings.llm_provider
        self.model_calls = 0
        self.model_errors = 0
        self.input_chars = 0
        self.output_chars = 0
        self._init_provider()

    def _init_provider(self) -> None:
        if self.provider_name == "http_compatible":
            key = self.settings.resolve_api_key()
            if not key:
                log.warning(
                    "llm_provider=http_compatible but %s is not set; "
                    "falling back to the deterministic engine",
                    self.settings.llm_api_key_env,
                )
                self.provider_name = "deterministic"
                return
            from app.llm.http_compat import HTTPCompatibleProvider

            self.provider = HTTPCompatibleProvider(
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                api_key=key,
                timeout_seconds=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )

    @property
    def llm_available(self) -> bool:
        return self.provider is not None

    async def aclose(self) -> None:
        if self.provider is not None:
            await self.provider.aclose()

    # ------------------------------------------------------------------ #
    # Low-level completion with counters
    # ------------------------------------------------------------------ #

    async def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        if self.provider is None:
            raise ProviderError("no LLM provider configured")
        self.model_calls += 1
        resp = await self.provider.complete(
            system=system, user=user, json_mode=json_mode,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        self.input_chars += resp.input_chars
        self.output_chars += resp.output_chars
        if not resp.ok:
            self.model_errors += 1
        return resp

    def _structured_call(self, system: str, user: str) -> dict | None:
        """Sync-style structured call used inside async wrappers."""
        raise NotImplementedError  # replaced by async path below

    # ------------------------------------------------------------------ #
    # High-level structured operations
    # ------------------------------------------------------------------ #

    async def plan_with_llm(self, task: str, repo_summary_json: str, context: str, base_plan: Plan) -> Plan | None:
        """Try to improve the deterministic plan with an LLM; None on failure."""
        if not self.llm_available:
            return None
        system = (
            "You are a staff software engineer producing a precise engineering plan. "
            "Respond with ONLY a JSON object matching: "
            '{"summary": str, "affected_files": [str], "steps": [{"id": str, "title": str, '
            '"description": str, "files": [str], "kind": "modify|create|delete|test|validate|research", '
            '"validation": str}], "tests_required": [str], "validation_commands": [str], "risks": [str], '
            '"dependency_impact": [str]}. '
            "Use ONLY files that appear in the provided repository context. Never invent paths."
        )
        user_prompt = (
            f"TASK: {task}\n\nREPOSITORY SUMMARY:\n{repo_summary_json}\n\n"
            f"RETRIEVED CONTEXT:\n{context[:self.settings.max_context_chars]}\n\n"
            f"BASE PLAN (improve it, keep it factual):\n{base_plan.model_dump_json()[:6000]}"
        )
        try:
            resp = await self.complete(system, user_prompt, json_mode=True)
            data = extract_json_object(resp.text)
            if not data:
                return None
            data.setdefault("task", task)
            data["generated_by"] = "llm"
            # Validate file references against the base plan + context paths
            known = set(base_plan.affected_files)
            data["affected_files"] = [p for p in data.get("affected_files", []) if p in known] or base_plan.affected_files
            try:
                # Rebuild through the Plan model so nested steps are coerced to PlanStep
                merged = Plan(**data)
            except ValidationError as exc:
                log.warning("LLM plan failed validation, using deterministic plan: %s", exc)
                self.model_errors += 1
                return None
            merged.steps = merged.steps or base_plan.steps
            return merged
        except ProviderError as exc:
            log.warning("LLM plan failed, using deterministic plan: %s", exc)
            self.model_errors += 1
            return None

    async def propose_patch(self, task: str, step_title: str, context: str) -> dict | None:
        """Ask the LLM for a patch proposal.

        Returns {"file": str, "old_text": str, "new_text": str, "description": str}
        or None. The agent applies it through the workspace sandbox which
        verifies the old block exists — malformed patches are rejected there.
        """
        if not self.llm_available:
            return None
        system = (
            "You are a careful software engineer. Make the smallest correct change. "
            "Respond with ONLY JSON: "
            '{"file": str, "old_text": str (exact current lines to replace, copy them verbatim '
            "from the context including indentation), "
            '"new_text": str (replacement lines), "description": str}. '
            "old_text MUST be copied character-for-character from the file content in the context."
        )
        user_prompt = f"TASK: {task}\nCURRENT STEP: {step_title}\n\nCODE CONTEXT:\n{context[:self.settings.max_context_chars]}"
        try:
            resp = await self.complete(system, user_prompt, json_mode=True)
            data = extract_json_object(resp.text)
            if not data:
                return None
            if not all(k in data and isinstance(data[k], str) for k in ("file", "old_text", "new_text")):
                return None
            return data
        except ProviderError as exc:
            log.warning("LLM patch proposal failed: %s", exc)
            self.model_errors += 1
            return None

    async def analyze_failure_llm(self, task: str, failure_output: str, context: str, base: FailureAnalysis) -> FailureAnalysis | None:
        if not self.llm_available:
            return None
        system = (
            "You are a debugging expert. Respond with ONLY JSON matching: "
            '{"category": "syntax|type|dependency|import|runtime|test_assertion|environment|'
            'configuration|api_contract|database|security|flaky|unknown", "error": str, '
            '"location": str, "likely_cause": str, "related_files": [str], '
            '"recommended_repair": str, "confidence": float}. '
            "Base everything on the provided failure output; do not speculate beyond it."
        )
        user_prompt = f"TASK: {task}\n\nFAILURE OUTPUT:\n{failure_output[:12000]}\n\nRELEVANT CODE:\n{context[:12000]}"
        try:
            resp = await self.complete(system, user_prompt, json_mode=True)
            data = extract_json_object(resp.text)
            if not data:
                return None
            data.pop("evidence", None)
            merged = base.model_copy(update=data)
            return merged
        except ProviderError as exc:
            log.warning("LLM failure analysis failed: %s", exc)
            self.model_errors += 1
            return None

    async def review_llm(self, diff: str, context: str) -> list[ReviewFinding]:
        if not self.llm_available:
            return []
        system = (
            "You are a principal engineer reviewing a diff. Respond with ONLY JSON: "
            '{"findings": [{"category": "correctness|maintainability|architecture|duplication|'
            'error_handling|security|performance|testing|dependency_risk|api_design", '
            '"severity": "info|low|medium|high|critical", "path": str, "line": int, '
            '"evidence": str (quote the actual diff lines), "explanation": str, '
            '"recommended_fix": str}]}. '
            "Only report findings you can support with quoted evidence from the diff."
        )
        user_prompt = f"DIFF:\n{diff[:30000]}\n\nCONTEXT:\n{context[:12000]}"
        try:
            resp = await self.complete(system, user_prompt, json_mode=True)
            data = extract_json_object(resp.text)
            if not data:
                return []
            findings = []
            for i, f in enumerate(data.get("findings", [])[:20]):
                try:
                    f.setdefault("id", f"LLM-{i}")
                    findings.append(ReviewFinding.model_validate(f))
                except Exception:
                    continue
            return findings
        except ProviderError as exc:
            log.warning("LLM review failed: %s", exc)
            self.model_errors += 1
            return []

    async def summarize(self, task: str, facts: str) -> str:
        if not self.llm_available:
            return ""
        system = (
            "You summarize engineering work in 2-4 sentences, first person plural, factual, "
            "no marketing language, no claims of success that are not in the facts."
        )
        try:
            resp = await self.complete(system, f"TASK: {task}\n\nFACTS:\n{facts[:12000]}")
            return resp.text.strip()[:2000]
        except ProviderError:
            return ""

    def snapshot_counters(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "model_errors": self.model_errors,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "provider": self.provider_name,
        }
