"""Hardened subprocess execution for tools.

- Args are always passed as a list (never through a shell), so shell
  metacharacters in arguments cannot inject commands.
- A binary allowlist gates what can run.
- Output is captured with hard byte caps; wall-clock timeouts kill the
  process tree.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

MAX_OUTPUT_BYTES = 200_000

# Only these binaries may be executed by agent tools. Everything else is
# refused regardless of task text.
ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        "python", "python3", "pytest", "pip", "node", "npm", "npx",
        "go", "cargo", "git", "make", "ruff", "eslint", "mypy", "tsc",
        "black", "flake8", "gofmt", "gofumpt", "javac", "java", "dotnet",
    }
)

# Substrings that are always refused (defense in depth on top of the
# allowlist + no-shell guarantees).
DENY_PATTERNS: tuple[str, ...] = (
    "rm -rf", "rmdir /s", "del /f", "rd /s", "format ", "mkfs",
    "shutdown", "reboot", ":(){", "fork bomb", "> /dev/", "curl http",
    "wget http", "invoke-webrequest", "invoke-expression", "iex ",
    "registry delete", "reg delete", "taskkill", "kill -9", "chmod 777 /",
)


@dataclass
class ExecResult:
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.error and self.exit_code == 0

    @property
    def combined_tail(self) -> str:
        tail = (self.stdout or "") + "\n" + (self.stderr or "")
        return tail[-8000:]


def validate_command(cmd: list[str]) -> None:
    if not cmd or not isinstance(cmd, list):
        raise ValueError("command must be a non-empty list of arguments")
    binary = os.path.basename(cmd[0]).lower().replace(".exe", "")
    if binary not in ALLOWED_BINARIES:
        raise ValueError(f"binary {binary!r} is not in the tool allowlist")
    joined = " ".join(cmd).lower()
    for pattern in DENY_PATTERNS:
        if pattern in joined:
            raise ValueError(f"command refused by safety policy: contains {pattern!r}")


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    timeout_seconds: int = 120,
    extra_env: dict[str, str] | None = None,
) -> ExecResult:
    import time

    validate_command(cmd)
    env = _clean_env()
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            shell=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        return ExecResult(
            command=cmd,
            exit_code=proc.returncode,
            stdout=proc.stdout[:MAX_OUTPUT_BYTES] if proc.stdout else "",
            stderr=proc.stderr[:MAX_OUTPUT_BYTES] if proc.stderr else "",
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - started) * 1000)
        return ExecResult(
            command=cmd, exit_code=None, stdout="", stderr="",
            duration_ms=duration_ms, timed_out=True,
            error=f"command timed out after {timeout_seconds}s",
        )
    except FileNotFoundError:
        return ExecResult(
            command=cmd, exit_code=None, stdout="", stderr="",
            duration_ms=0, error=f"binary not found: {cmd[0]}",
        )
    except OSError as exc:
        return ExecResult(
            command=cmd, exit_code=None, stdout="", stderr="",
            duration_ms=0, error=f"execution failed: {exc}",
        )


def _clean_env() -> dict[str, str]:
    keep = (
        "PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME",
        "USERPROFILE", "APPDATA", "PROGRAMFILES", "PYTHONPATH",
        "SYSTEMDRIVE", "WINDIR", "PATHEXT", "LIB", "INCLUDE",
        "GOMODCACHE", "GOPATH", "CARGO_HOME", "NVM_HOME", "NODE_PATH",
    )
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    # Strip any proxy variables that could exfiltrate or break local runs.
    for k in list(env):
        if k.lower().endswith("_proxy"):
            env.pop(k)
    return env
