"""Rule-based security scanner with line-level evidence.

Every finding quotes the exact offending line — the scanner never invents
vulnerabilities without textual evidence. Designed for Python first (the
spec's priority language) with a few cross-language checks.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.schemas import SecurityFinding, SecurityScanResult, Severity

SECRET_PATTERNS: tuple[tuple[str, str, Severity], ...] = (
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), Severity.CRITICAL),
    ("private key block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"), Severity.CRITICAL),
    (
        "credential assignment",
        re.compile(
            r"""(?i)\b(?:api[_-]?key|apikey|secret|token|password|passwd|pwd|access[_-]?key|auth[_-]?token|client[_-]?secret)\w*\s*[:=]\s*['"][^'"\s]{8,}['"]"""
        ),
        Severity.HIGH,
    ),
    ("bearer token literal", re.compile(r"""(?i)bearer\s+['"][A-Za-z0-9_\-\.=]{16,}['"]"""), Severity.HIGH),
    ("database url with password", re.compile(r"""(?i)(postgres|mysql|mongodb(\+srv)?|amqp)://[^:\s"']+:[^@\s"']+@"""), Severity.CRITICAL),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), Severity.CRITICAL),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), Severity.CRITICAL),
)

SECRET_ALLOWLIST = re.compile(
    r"""(?ix)
    ^\s*#.* |                                   # comments
    os\.environ | getenv | settings\. | config\. |
    example | sample | placeholder | xxx+ | <[^>]*> | dummy |
    \{\{|\}\} | \{[a-z_]+\} | format | input\( | your[-_]?key | todo
    """
)

DANGEROUS_CALLS: tuple[tuple[str, str, str, Severity], ...] = (
    ("FGE001", "eval/exec usage", re.compile(r"(?<![\w.])(eval|exec)\s*\("), Severity.HIGH,
     "eval/exec executes arbitrary Python; prefer explicit parsing or a safe evaluator"),
    ("FGE002", "shell=True subprocess", re.compile(r"""subprocess\.\w+\([^)]*shell\s*=\s*True"""), Severity.HIGH,
     "shell=True allows command injection; pass an argument list and shell=False"),
    ("FGE003", "os.system call", re.compile(r"(?<![\w.])os\.system\s*\("), Severity.HIGH,
     "os.system runs a shell string; use subprocess with an argument list"),
    ("FGE004", "pickle load", re.compile(r"(?<![\w.])pickle\.loads?\s*\("), Severity.HIGH,
     "pickle deserializes arbitrary objects; use JSON or validate the source"),
    ("FGE005", "unsafe yaml load", re.compile(r"(?<![\w.])yaml\.load\s*\((?![^)]*Loader\s*=\s*(?:yaml\.)?SafeLoader)"), Severity.MEDIUM,
     "yaml.load without SafeLoader can construct arbitrary objects"),
    ("FGE006", "md5/sha1 for passwords", re.compile(r"""(?i)(md5|sha1)\s*\(.*passw"""), Severity.MEDIUM,
     "fast hashes are unsuitable for password storage; use bcrypt/scrypt/argon2"),
    ("FGE007", "random for secrets", re.compile(r"""(?i)(random\.(choice|randint|randrange|random)\s*\(.*)"""), Severity.LOW,
     "the random module is not cryptographically secure; use secrets/token_bytes for tokens"),
    ("FGE008", "flask debug mode", re.compile(r"(?i)app\.run\([^)]*debug\s*=\s*True"), Severity.MEDIUM,
     "debug mode exposes an interactive debugger; disable in deployed code"),
    ("FGE009", "verify=False TLS", re.compile(r"(?i)verify\s*=\s*False"), Severity.MEDIUM,
     "TLS verification disabled enables man-in-the-middle attacks"),
    ("FGE010", "SQL built via f-string/format", re.compile(r"""(?i)(execute|executemany|executescript)\s*\(\s*(f["']|["']\s*%|["']\s*\.\s*format|\w+\s*\+)"""), Severity.HIGH,
     "SQL assembled from strings enables injection; use parameterized queries"),
    ("FGE011", "assert used for control flow", re.compile(r"^\s*assert\s+"), Severity.INFO,
     "asserts are stripped under -O; use explicit checks and exceptions"),
    ("FGE012", "temporary file with predictable name", re.compile(r"""(?i)tempfile|open\(["']\/tmp"""), Severity.INFO, ""),
)


class SecurityScanner:
    """Scans a directory tree (workspace or snapshot) for security issues."""

    def scan_directory(self, root: Path, max_files: int = 3000) -> SecurityScanResult:
        result = SecurityScanResult()
        if not root.is_dir():
            return result
        scanned = 0
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
        for path in sorted(root.rglob("*")):
            if scanned >= max_files:
                break
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix.lower() not in (".py", ".js", ".ts", ".jsx", ".tsx", ".env", ".yaml", ".yml", ".json", ".sh", ".go", ".rb", ".php", ".java"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            rel = path.relative_to(root).as_posix()
            result.findings.extend(self._scan_text(rel, text))
        result.files_scanned = scanned
        return result

    def scan_text(self, rel_path: str, text: str) -> list[SecurityFinding]:
        return self._scan_text(rel_path, text)

    # ------------------------------------------------------------------ #

    def _scan_text(self, rel_path: str, text: str) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        lines = text.splitlines()
        doc_position = self._docstring_end(lines)
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            in_docstring = lineno <= doc_position
            for rule_id, title, pattern, severity, explanation in DANGEROUS_CALLS:
                if rule_id == "FGE011" and in_docstring:
                    continue
                if m := pattern.search(line):
                    if rule_id == "FGE007":
                        # Only flag `random` when clearly security-adjacent.
                        if not re.search(r"(?i)token|password|secret|otp|nonce|csrf", line):
                            continue
                    if rule_id == "FGE012" and "tempfile" in line and "NamedTemporaryFile" in line:
                        continue
                    if rule_id == "FGE006" and "hashlib" not in line:
                        continue
                    findings.append(self._finding(rule_id, title, severity, rel_path, lineno, line, explanation or title))
            for title, pattern, severity in SECRET_PATTERNS:
                if m := pattern.search(line):
                    if SECRET_ALLOWLIST.search(line) or in_docstring:
                        continue
                    match_text = m.group(0)
                    # Redact the secret in evidence but prove presence.
                    evidence = line.replace(match_text, match_text[:4] + "***REDACTED***")[:200]
                    findings.append(
                        SecurityFinding(
                            id=self._fid(rule_id="SEC", path=rel_path, line=lineno),
                            rule_id="SEC-HARDCODED",
                            title=f"possible hardcoded {title}",
                            severity=severity,
                            path=rel_path,
                            line=lineno,
                            evidence=evidence,
                            explanation=f"line contains what looks like a hardcoded {title}; secrets belong in environment variables or a secret manager",
                            recommended_fix="move the value to an environment variable and load it at startup",
                        )
                    )
        # .env-style files: any assignment is a finding
        if Path(rel_path).name in (".env", ".env.local", ".env.production") or rel_path.endswith(".env"):
            for lineno, line in enumerate(lines, start=1):
                if line.strip() and not line.strip().startswith("#") and "=" in line:
                    key = line.split("=", 1)[0].strip()
                    findings.append(
                        SecurityFinding(
                            id=self._fid("ENV", rel_path, lineno),
                            rule_id="SEC-ENV-FILE",
                            title=f".env file contains assignment for {key}",
                            severity=Severity.MEDIUM,
                            path=rel_path, line=lineno,
                            evidence=line[:120],
                            explanation=".env files with real values must not be committed to version control",
                            recommended_fix="add to .gitignore and use .env.example with placeholder values",
                        )
                    )
        return findings

    def _finding(self, rule_id: str, title: str, severity: Severity, path: str, line: int, text: str, explanation: str) -> SecurityFinding:
        fixes = {
            "FGE001": "replace eval/exec with explicit parsing (ast.literal_eval, json.loads)",
            "FGE002": "use subprocess.run([...], shell=False)",
            "FGE003": "use subprocess.run with an argument list",
            "FGE004": "use JSON or a validated schema-bound deserializer",
            "FGE005": "use yaml.safe_load",
            "FGE008": "set debug=False outside development",
            "FGE009": "keep TLS verification enabled (verify=True)",
            "FGE010": "use parameterized queries (?, %s placeholders)",
            "FGE011": "replace assert with an explicit if/raise",
        }
        return SecurityFinding(
            id=self._fid(rule_id, path, line), rule_id=rule_id, title=title,
            severity=severity, path=path, line=line, evidence=text.strip()[:200],
            explanation=explanation, recommended_fix=fixes.get(rule_id, "review and harden this construct"),
        )

    def _fid(self, rule_id: str, path: str, line: int) -> str:
        digest = hashlib.sha1(f"{rule_id}:{path}:{line}".encode()).hexdigest()[:8]
        return f"{rule_id}-{digest}"

    def _docstring_end(self, lines: list[str]) -> int:
        """Line number where the module docstring ends (0 if none)."""
        if not lines:
            return 0
        first = lines[0].strip()
        if first.startswith(('"""', "'''")):
            quote = first[:3]
            if len(first) >= 6 and first.endswith(quote) and len(first) > 3:
                return 1
            for lineno, line in enumerate(lines[1:], start=2):
                if quote in line:
                    return lineno
        return 0
