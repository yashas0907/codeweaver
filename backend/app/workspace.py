"""Isolated agent workspace with controlled, fully-tracked modifications.

Guarantees:
  - The original repository snapshot is never touched; each run gets a copy
    under data/workspaces/<run_id>/.
  - Every write goes through this module: path traversal, protected files
    (.env, keys, etc.) and overwrites of unrelated files are refused.
  - READ_ONLY mode refuses all mutations.
  - Every change records before/after text and a unified diff.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
import shutil
from pathlib import Path

from app.config import PROTECTED_PATH_PATTERNS, get_settings
from app.schemas import AgentMode, FileChange

MAX_WRITE_BYTES = 1_000_000


class WorkspaceError(RuntimeError):
    """Refused or failed workspace operation."""


class PatchFormatError(WorkspaceError):
    """Raised when a patch cannot be parsed or located."""


class Workspace:
    def __init__(self, run_id: str, snapshot: Path, mode: AgentMode | None = None) -> None:
        self.settings = get_settings()
        self.run_id = run_id
        self.mode = mode or AgentMode(self.settings.agent_mode.upper())
        self.snapshot = snapshot.resolve()
        self.root = (self.settings.workspace_root / run_id).resolve()
        self._changed: dict[str, FileChange] = {}
        self._ensure()

    # ------------------------------------------------------------------ #

    def _ensure(self) -> None:
        if self.root.exists():
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.snapshot.is_dir():
            shutil.copytree(
                self.snapshot, self.root,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "node_modules", ".venv"),
                dirs_exist_ok=True,
            )
        else:
            self.root.mkdir(parents=True, exist_ok=True)

    # --------------------------- path safety --------------------------- #

    def resolve_path(self, rel_path: str, for_write: bool = False) -> Path:
        if not rel_path or "\x00" in rel_path:
            raise WorkspaceError("empty or invalid path")
        rel_path = rel_path.strip().replace("\\", "/").lstrip("/")
        if any(part in ("..",) for part in rel_path.split("/")):
            raise WorkspaceError(f"path traversal refused: {rel_path}")
        if rel_path.startswith(".git/") or rel_path == ".git":
            raise WorkspaceError("refusing to touch .git internals")
        candidate = (self.root / rel_path).resolve()
        root_str = str(self.root)
        if not str(candidate).startswith(root_str + "\\") and not str(candidate).startswith(root_str + "/") and str(candidate) != root_str:
            raise WorkspaceError(f"path escapes workspace: {rel_path}")
        if for_write and self._is_protected(rel_path):
            raise WorkspaceError(
                f"refusing to modify protected file: {rel_path} (secrets/credentials are off-limits)"
            )
        return candidate

    def _is_protected(self, rel_path: str) -> bool:
        lowered = rel_path.lower()
        for pattern in PROTECTED_PATH_PATTERNS:
            if fnmatch.fnmatch(lowered, pattern) or fnmatch.fnmatch(Path(lowered).name, pattern):
                return True
        return False

    def check_mode(self) -> None:
        if self.mode == AgentMode.READ_ONLY:
            raise WorkspaceError("agent is in READ_ONLY mode; modifications are disabled")

    # ------------------------------ reads ------------------------------ #

    def read_file(self, rel_path: str) -> str:
        path = self.resolve_path(rel_path)
        if not path.is_file():
            raise WorkspaceError(f"file not found: {rel_path}")
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise WorkspaceError(f"cannot read {rel_path}: {exc}") from exc

    def read_range(self, rel_path: str, start: int, end: int) -> str:
        lines = self.read_file(rel_path).splitlines()
        start = max(1, start)
        end = min(len(lines), max(end, start))
        selected = lines[start - 1:end]
        return "\n".join(f"{n:5d} | {line}" for n, line in enumerate(selected, start=start))

    def list_files(self, pattern: str = "**/*") -> list[str]:
        results: list[str] = []
        for path in sorted(self.root.glob(pattern)):
            if path.is_file() and ".git" not in path.parts:
                results.append(path.relative_to(self.root).as_posix())
        return results[:2000]

    def file_exists(self, rel_path: str) -> bool:
        try:
            return self.resolve_path(rel_path).is_file()
        except WorkspaceError:
            return False

    # ------------------------------ writes ----------------------------- #

    def create_file(self, rel_path: str, content: str, description: str = "") -> FileChange:
        self.check_mode()
        path = self.resolve_path(rel_path, for_write=True)
        if path.exists():
            raise WorkspaceError(f"file already exists (use apply_patch to modify): {rel_path}")
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise WorkspaceError("content exceeds size budget")
        self._validate_syntax(rel_path, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        change = FileChange(
            path=rel_path,
            change_type="created",
            before_text="",
            after_text=content,
            diff=self._diff_text("", content),
            description=description or f"created {rel_path}",
        )
        self._record(change)
        return change

    def delete_file(self, rel_path: str, description: str = "") -> FileChange:
        self.check_mode()
        path = self.resolve_path(rel_path, for_write=True)
        if not path.is_file():
            raise WorkspaceError(f"file not found: {rel_path}")
        before = path.read_text(encoding="utf-8", errors="replace")
        path.unlink()
        change = FileChange(
            path=rel_path, change_type="deleted", before_text=before,
            after_text="", diff=self._diff_text(before, ""),
            description=description or f"deleted {rel_path}",
        )
        self._record(change)
        return change

    def replace_in_file(
        self, rel_path: str, old_text: str, new_text: str,
        description: str = "", occurrence: int = 1,
    ) -> FileChange:
        """Exact-block replacement; the safest primitive for LLM patches.

        Refuses when the old block is absent or ambiguous (count != 1 by
        default), so the model cannot silently clobber wrong regions.
        """
        self.check_mode()
        path = self.resolve_path(rel_path, for_write=True)
        if not path.is_file():
            raise WorkspaceError(f"file not found: {rel_path}")
        before = path.read_text(encoding="utf-8", errors="replace")
        if not old_text.strip():
            raise PatchFormatError("empty search block")
        if old_text not in before:
            # Tolerate trailing-whitespace drift line-by-line.
            patched = self._fuzzy_replace(before, old_text, new_text)
            if patched is None:
                raise PatchFormatError(
                    f"search block not found in {rel_path}; the file may have changed — re-read it and retry"
                )
            new_content = patched
        else:
            count = before.count(old_text)
            if count > 1 and occurrence == 1:
                raise PatchFormatError(
                    f"search block appears {count} times in {rel_path}; include more surrounding context to disambiguate"
                )
            new_content = before.replace(old_text, new_text, occurrence)
        self._validate_syntax(rel_path, new_content)
        path.write_text(new_content, encoding="utf-8")
        change = FileChange(
            path=rel_path, change_type="modified", before_text=before,
            after_text=new_content, diff=self._diff_text(before, new_content),
            description=description or f"modified {rel_path}",
        )
        self._record(change)
        return change

    def apply_patch(self, rel_path: str, patch: str, description: str = "") -> list[FileChange]:
        """Apply a unified diff (```diff blocks with ---/+++ accepted)."""
        self.check_mode()
        hunks = self._parse_unified_patch(patch)
        if not hunks:
            raise PatchFormatError("no valid hunks found in patch")
        changes: list[FileChange] = []
        for hunk_path, old_block, new_block in hunks:
            target = hunk_path or rel_path
            if not target:
                raise PatchFormatError("patch hunks must reference a file")
            if self.file_exists(target):
                changes.append(self.replace_in_file(target, old_block, new_block, description))
            else:
                normalized = self._strip_leading_plus(new_block)
                changes.append(self.create_file(target, normalized, description))
        return changes

    # ----------------------------- internals --------------------------- #

    @staticmethod
    def _validate_syntax(rel_path: str, content: str) -> None:
        """Refuse to write a Python file that no longer parses.

        This is the safety net that keeps an imperfect LLM patch from
        corrupting a working module: if the patched content has a syntax
        error the write is rejected and the original file stays intact.
        """
        if not rel_path.lower().endswith(".py"):
            return
        import ast

        try:
            ast.parse(content)
        except SyntaxError as exc:
            raise PatchFormatError(
                f"patch rejected: it would introduce a syntax error in {rel_path} "
                f"(line {exc.lineno}: {exc.msg}) — the file was left unchanged"
            ) from exc

    def _fuzzy_replace(self, content: str, old_text: str, new_text: str) -> str | None:
        old_lines = old_text.splitlines()
        content_lines = content.splitlines()
        if not old_lines:
            return None
        stripped = [ln.rstrip() for ln in old_lines]
        for i in range(len(content_lines) - len(stripped) + 1):
            window = [ln.rstrip() for ln in content_lines[i:i + len(stripped)]]
            if window == stripped:
                new_lines = new_text.splitlines()
                content_lines[i:i + len(stripped)] = new_lines
                return "\n".join(content_lines) + ("\n" if content.endswith("\n") else "")
        return None

    def _strip_leading_plus(self, block: str) -> str:
        return "\n".join(ln[1:] if ln.startswith("+") else ln for ln in block.splitlines())

    def _parse_unified_patch(self, patch: str) -> list[tuple[str, str, str]]:
        """Parse `diff --git`/`---`/`+++`/@@ hunks into (path, old, new) triples."""
        patch = patch.strip().removeprefix("```diff").removeprefix("```").removesuffix("```")
        lines = patch.splitlines()
        hunks: list[tuple[str, str, str]] = []
        current_path = ""
        old: list[str] = []
        new: list[str] = []
        in_hunk = False

        def flush():
            if in_hunk and (old or new):
                hunks.append((current_path, "\n".join(old), "\n".join(new)))

        for line in lines:
            if line.startswith("diff --git"):
                flush()
                in_hunk = False
                old, new = [], []
                m = re.search(r" b/(.+)$", line)
                current_path = m.group(1) if m else ""
            elif line.startswith("--- "):
                continue
            elif line.startswith("+++ "):
                m = re.search(r"\+\+\+ b?/?(\S+)", line)
                if m and not current_path:
                    current_path = m.group(1)
            elif line.startswith("@@"):
                flush()
                old, new = [], []
                in_hunk = True
            elif in_hunk:
                if line.startswith("+"):
                    new.append(line[1:])
                elif line.startswith("-"):
                    old.append(line[1:])
                elif line.startswith(" "):
                    old.append(line[1:])
                    new.append(line[1:])
                elif line.strip() == "":
                    old.append("")
                    new.append("")
        flush()
        return [h for h in hunks if h[1].strip() or h[2].strip()]

    def _diff_text(self, before: str, after: str) -> str:
        diff = difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile="before", tofile="after", lineterm="", n=3,
        )
        return "\n".join(diff)[:20000]

    def _record(self, change: FileChange) -> None:
        existing = self._changed.get(change.path)
        if existing and existing.change_type == "created" and change.change_type == "modified":
            change.change_type = "created"
            change.before_text = ""
            change.diff = self._diff_text("", change.after_text)
        self._changed[change.path] = change

    # ------------------------------ exports ---------------------------- #

    def changes(self) -> list[FileChange]:
        return list(self._changed.values())

    def git_diff(self) -> str:
        """Unified diff of the whole workspace vs HEAD, if git metadata exists."""
        import subprocess

        try:
            proc = subprocess.run(
                ["git", "diff"], cwd=self.root, capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout[:50000]
        except (OSError, subprocess.TimeoutExpired):
            pass
        return "\n\n".join(c.diff for c in self._changed.values())

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
