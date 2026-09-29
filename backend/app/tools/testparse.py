"""Parses test-runner output (pytest, jest, go test, cargo) into TestRun.

Deterministic text parsing — no LLM involved — so test results are always
trustworthy regardless of model quality.
"""

from __future__ import annotations

import re

from app.schemas import TestCaseResult, TestRun

PYTEST_SUMMARY = re.compile(
    r"=+\s*(?:(\d+)\s+passed.*?)?(?:.*?(\d+)\s+failed.*?)?(?:.*?(\d+)\s+error.*?)?"
    r"(?:.*?(\d+)\s+skipped.*?)?\s*(?:in\s+([\d.]+)s(?:\s+.*)?)?\s*=+\s*$"
)
PYTEST_FAILURE_LINE = re.compile(r"^(?:FAILED|ERROR)\s+([\w./:\\-]+?::[\w:]+)(?:\s+-\s+(.*))?$")
PYTEST_SECTION = re.compile(r"^(?:=+)\s+(FAILURES|ERRORS)\s+=+")
JEST_SUMMARY = re.compile(r"(?:Tests|Test Suites):\s*(\d+)\s+failed.*?(\d+)\s+passed.*?(\d+)\s+total")
GO_TEST = re.compile(r"^(?:---\s+(FAIL|PASS|SKIP):\s+(\S+))|(?:^(ok|FAIL)\s+(\S+))")
CARGO_RESULT = re.compile(r"test result:\s*(\w+)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;\s*(\d+)\s+ignored")


def parse_pytest(output: str, exit_code: int | None) -> TestRun:
    run = TestRun(command="pytest")
    lines = output.splitlines()

    cases: list[TestCaseResult] = []
    in_failures = False

    for line in lines:
        m = PYTEST_FAILURE_LINE.match(line.strip())
        if m:
            node = m.group(1)
            file_part = node.split("::")[0]
            cases.append(TestCaseResult(
                name=node, status="failed" if line.strip().startswith("FAILED") else "error",
                message=(m.group(2) or "")[:300], file=file_part,
            ))
        elif PYTEST_SECTION.match(line.strip()):
            in_failures = True
        elif in_failures and line.startswith("=") and "short test summary" not in line:
            in_failures = False

    summary_line = ""
    for line in reversed(lines):
        line_s = line.strip().strip("=").strip()
        if ("passed" in line_s or "failed" in line_s or "error" in line_s or "no tests ran" in line_s) and (
            re.search(r"\d+\s+(passed|failed|error)", line_s) or "no tests ran" in line_s
        ):
            summary_line = line_s
            break
    if summary_line:
        passed = re.search(r"(\d+)\s+passed", summary_line)
        failed = re.search(r"(\d+)\s+failed", summary_line)
        errors = re.search(r"(\d+)\s+error", summary_line)
        skipped = re.search(r"(\d+)\s+skipped", summary_line)
        dur = re.search(r"in\s+([\d.]+)s", summary_line)
        run.passed = int(passed.group(1)) if passed else 0
        run.failed = int(failed.group(1)) if failed else 0
        run.errors = int(errors.group(1)) if errors else 0
        run.skipped = int(skipped.group(1)) if skipped else 0
        if dur:
            run.duration_ms = int(float(dur.group(1)) * 1000)

    # Collection/interrupt failures ("Interrupted: 1 error during collection")
    interrupted = re.search(r"Interrupted:\s*(\d+)\s+error", output)
    if interrupted:
        run.errors = max(run.errors, int(interrupted.group(1)))

    if not cases:
        # Derive case names from summary lines when the sections were absent.
        for line in lines:
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                node = line.split(None, 1)[1].split(" - ")[0].strip()
                status = "failed" if line.startswith("FAILED ") else "error"
                cases.append(TestCaseResult(name=node, status=status, file=node.split("::")[0]))

    # Ensure counts reflect the derived cases when the summary was missing.
    derived_failed = sum(1 for c in cases if c.status == "failed")
    derived_errors = sum(1 for c in cases if c.status == "error")
    run.failed = max(run.failed, derived_failed)
    run.errors = max(run.errors, derived_errors)

    run.cases = cases
    run.exit_code = exit_code
    run.total = run.passed + run.failed + run.errors + run.skipped
    run.all_passed = exit_code == 0 and run.failed == 0 and run.errors == 0
    if exit_code == 5:  # pytest: no tests collected
        run.all_passed = False
    run.raw_output_tail = output[-8000:]
    return run


def parse_generic_output(command: str, output: str, exit_code: int | None) -> TestRun:
    run = TestRun(command=command)
    run.raw_output_tail = output[-8000:]
    if m := CARGO_RESULT.search(output):
        run.passed, run.failed, run.skipped = int(m.group(2)), int(m.group(3)), int(m.group(4))
        run.total = run.passed + run.failed + run.skipped
        run.all_passed = exit_code == 0 and run.failed == 0
        return run
    if m := JEST_SUMMARY.search(output):
        run.failed, run.passed, run.total = int(m.group(1)), int(m.group(2)), int(m.group(3))
        run.all_passed = exit_code == 0 and run.failed == 0
        return run
    if "go test" in command or re.search(r"^(ok|FAIL)\s+\S+", output, re.MULTILINE):
        fails = 0
        for m in GO_TEST.finditer(output):
            if m.group(1) == "FAIL":
                fails += 1
                run.cases.append(TestCaseResult(name=m.group(2), status="failed", file=m.group(2).split(":")[0]))
        run.failed = fails
        run.all_passed = exit_code == 0
        run.total = fails if fails else 0
        return run
    run.all_passed = exit_code == 0
    return run


def parse_test_output(command: str, output: str, exit_code: int | None) -> TestRun:
    run: TestRun
    if "pytest" in command or ("FAILED" in output and "passed" in output):
        run = parse_pytest(output, exit_code)
    else:
        run = parse_generic_output(command, output, exit_code)
    run.command = command
    run.exit_code = exit_code
    return run
