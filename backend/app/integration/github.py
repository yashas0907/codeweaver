"""GitHub integration: push an agent workspace to a repository as a Pull Request.

Flow:
  1. Every workspace file's git blob SHA is computed locally
     (``sha1("blob <len>\\0" + bytes)``), so unchanged files are detected
     without downloading anything.
  2. A unique working branch ``codeweaver/<run_id>/<timestamp>`` is cut from
     the base branch.
  3. Only files that differ from the base branch (or are new) are committed —
     as a single commit via the Git data API (tree + commit + ref update).
     Files deleted from the workspace are removed from the tree when the base
     tree could be listed recursively. Secrets and credential files are
     never pushed.
  4. A pull request is opened from the working branch into the base branch.

Requires a GitHub personal access token with ``repo`` scope (private repos)
or ``public_repo`` (public repos). The token is passed in by the caller and
is never logged or persisted.
"""

from __future__ import annotations

import base64
import difflib
import fnmatch
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.config import PROTECTED_PATH_PATTERNS, SKIP_DIRECTORIES

GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "CodeWeaver/1.0"
API_VERSION = "2022-11-28"
REQUEST_TIMEOUT_SECONDS = 30.0

# Files above this size are never pushed (matches the workspace write budget).
MAX_FILE_BYTES = 1_000_000
# Safety valve: refuse to open a PR with an absurd number of changed files.
MAX_CHANGED_FILES = 500


class GitHubError(RuntimeError):
    """A GitHub API call failed."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"GitHub API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class GitHubPushError(RuntimeError):
    """The push cannot proceed (bad input, missing branch, no changes, ...)."""


def parse_repo_slug(repo_url: str) -> str:
    """Normalize a repo URL (or ``owner/repo``) into an ``owner/repo`` slug."""
    raw = (repo_url or "").strip()
    if not raw:
        raise GitHubPushError("no repository URL provided")
    slug = re.sub(r"^https?://[^/]+/", "", raw, flags=re.IGNORECASE)  # https://github.com/
    slug = re.sub(r"^git@[^:]+:", "", slug)  # git@github.com:
    slug = slug.strip("/")
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]
    parts = [p for p in slug.split("/") if p]
    if len(parts) != 2:
        raise GitHubPushError(
            f"cannot parse a GitHub repository from {repo_url!r} — "
            "expected 'owner/repo' or a https://github.com/owner/repo URL"
        )
    return f"{parts[0]}/{parts[1]}"


def git_blob_sha(data: bytes) -> str:
    """Compute the git blob object SHA for raw bytes (no git binary needed)."""
    header = f"blob {len(data)}".encode("ascii") + b"\x00"
    return hashlib.sha1(header + data).hexdigest()


def _decode_text(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_protected(rel_path: str) -> bool:
    """Refuse to push secrets/credential files to a remote."""
    lowered = rel_path.lower()
    for pattern in PROTECTED_PATH_PATTERNS:
        if fnmatch.fnmatch(lowered, pattern) or fnmatch.fnmatch(Path(lowered).name, pattern):
            return True
    return False


def _workspace_files(workspace_path: Path) -> tuple[list[tuple[str, bytes]], set[str]]:
    """Walk the workspace recursively.

    Returns (committable files as (posix rel path, bytes), paths that exist
    but are skipped because they are protected, oversized or unreadable).
    """
    files: list[tuple[str, bytes]] = []
    skipped: set[str] = set()
    for path in sorted(workspace_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(workspace_path).as_posix()
        if any(part in SKIP_DIRECTORIES for part in rel.split("/")):
            continue
        if _is_protected(rel):
            skipped.add(rel)
            continue
        try:
            data = path.read_bytes()
        except OSError:
            skipped.add(rel)
            continue
        if len(data) > MAX_FILE_BYTES:
            skipped.add(rel)
            continue
        files.append((rel, data))
    return files, skipped


class GitHubClient:
    """Thin synchronous GitHub REST API client backed by ``httpx``."""

    def __init__(self, token: str) -> None:
        token = (token or "").strip()
        if not token:
            raise GitHubPushError("GitHub token is empty")
        self._http = httpx.Client(
            base_url=GITHUB_API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": API_VERSION,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------ #

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        ok: tuple[int, ...] = (200, 201, 204),
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise GitHubError(0, f"network error calling {method} {url}: {exc}") from exc
        if response.status_code not in ok:
            message = response.reason_phrase
            try:
                body = response.json()
                if isinstance(body, dict) and body.get("message"):
                    message = str(body["message"])
            except ValueError:
                message = message or response.text[:300]
            raise GitHubError(response.status_code, message)
        if response.status_code == 204 or not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise GitHubError(response.status_code, "invalid JSON in GitHub response") from exc
        return data if isinstance(data, dict) else {"_list": data}

    # ------------------------------ refs ------------------------------- #

    def get_branch_sha(self, repo: str, branch: str) -> str | None:
        """HEAD commit SHA of ``branch``, or None when the branch is absent."""
        data = self._request(
            "GET", f"/repos/{repo}/git/ref/{quote('heads/' + branch, safe='')}", ok=(200, 404)
        )
        obj = data.get("object") if isinstance(data, dict) else None
        return obj.get("sha") if isinstance(obj, dict) else None

    def create_branch(self, repo: str, base_branch: str, new_branch: str) -> str:
        """Create ``new_branch`` at the current HEAD of ``base_branch``.

        Returns the SHA of the base branch's HEAD.
        """
        base_sha = self.get_branch_sha(repo, base_branch)
        if not base_sha:
            raise GitHubPushError(f"base branch {base_branch!r} not found in {repo}")
        self._request(
            "POST",
            f"/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
            ok=(201,),
        )
        return base_sha

    def update_ref(self, repo: str, branch: str, sha: str, force: bool = False) -> None:
        """Point ``branch`` at ``sha`` (fast-forward unless ``force``)."""
        self._request(
            "PATCH",
            f"/repos/{repo}/git/refs/heads/{quote(branch, safe='')}",
            json={"sha": sha, "force": force},
            ok=(200,),
        )

    # --------------------------- contents API --------------------------- #

    def get_file_sha(self, repo: str, branch: str, path: str) -> str | None:
        """Blob SHA of ``path`` on ``branch``, or None when it does not exist."""
        data = self._request(
            "GET",
            f"/repos/{repo}/contents/{quote(path)}",
            params={"ref": branch},
            ok=(200, 404),
        )
        if not isinstance(data, dict) or "_list" in data:
            return None  # a directory listing / missing file
        return data.get("sha")

    def get_file_content(self, repo: str, branch: str, path: str) -> str | None:
        """Decoded text of ``path`` on ``branch``, or None when absent/binary."""
        data = self._request(
            "GET",
            f"/repos/{repo}/contents/{quote(path)}",
            params={"ref": branch},
            ok=(200, 404),
        )
        if not isinstance(data, dict) or "_list" in data:
            return None
        raw = data.get("content")
        if not raw:
            return None
        try:
            if data.get("encoding") == "base64":
                return base64.b64decode(raw).decode("utf-8")
            return str(raw)
        except (UnicodeDecodeError, ValueError):
            return None

    def update_file(
        self, repo: str, branch: str, path: str, content: str, message: str, sha: str | None
    ) -> str:
        """Create or update one file via the Contents API; returns the new commit SHA."""
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        data = self._request("PUT", f"/repos/{repo}/contents/{quote(path)}", json=payload, ok=(200, 201))
        commit = data.get("commit") or {}
        return str(commit.get("sha") or "")

    # --------------------------- git data API --------------------------- #

    def create_blob(self, repo: str, data: bytes) -> str:
        """Upload raw bytes as a blob (used for binary files); returns blob SHA."""
        out = self._request(
            "POST",
            f"/repos/{repo}/git/blobs",
            json={"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"},
            ok=(201,),
        )
        return str(out.get("sha") or "")

    def get_tree(self, repo: str, tree_sha: str) -> tuple[dict[str, str], bool]:
        """Recursively list ``tree_sha`` as {path: blob sha}; second value is the
        GitHub ``truncated`` flag (True for very large trees)."""
        out = self._request(
            "GET", f"/repos/{repo}/git/trees/{tree_sha}", params={"recursive": "1"}, ok=(200, 404)
        )
        if not isinstance(out, dict) or "tree" not in out:
            raise GitHubPushError(
                f"cannot read the repository tree of {repo} — check that the repo exists "
                "and the token has access to it"
            )
        mapping = {
            entry["path"]: entry["sha"]
            for entry in out.get("tree", [])
            if entry.get("type") == "blob" and entry.get("path") and entry.get("sha")
        }
        return mapping, bool(out.get("truncated"))

    def get_commit_tree_sha(self, repo: str, commit_sha: str) -> str:
        """Tree SHA behind a commit.

        Reads of ``/git/trees`` accept a commit SHA, but ``base_tree`` on a
        tree *write* requires a real tree object SHA — passing a commit SHA
        there makes GitHub return 404.
        """
        out = self._request("GET", f"/repos/{repo}/git/commits/{commit_sha}", ok=(200,))
        tree = out.get("tree") if isinstance(out, dict) else None
        sha = tree.get("sha") if isinstance(tree, dict) else None
        if not sha:
            raise GitHubPushError(f"cannot resolve the tree of commit {commit_sha} in {repo}")
        return str(sha)

    def create_tree(self, repo: str, tree: list[dict[str, Any]], base_tree: str | None = None) -> str:
        payload: dict[str, Any] = {"tree": tree}
        if base_tree:
            payload["base_tree"] = base_tree
        out = self._request("POST", f"/repos/{repo}/git/trees", json=payload, ok=(201,))
        return str(out.get("sha") or "")

    def create_commit(self, repo: str, message: str, tree_sha: str, parents: list[str]) -> str:
        out = self._request(
            "POST",
            f"/repos/{repo}/git/commits",
            json={"message": message, "tree": tree_sha, "parents": parents},
            ok=(201,),
        )
        return str(out.get("sha") or "")

    # ----------------------------- pull requests ------------------------ #

    def create_pull_request(self, repo: str, title: str, body: str, head: str, base: str) -> dict:
        """Open a PR; returns {number, url, html_url, state}."""
        out = self._request(
            "POST",
            f"/repos/{repo}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
            ok=(201,),
        )
        return {
            "number": int(out.get("number") or 0),
            "url": str(out.get("url") or ""),
            "html_url": str(out.get("html_url") or ""),
            "state": str(out.get("state") or "open"),
        }


def _find_block(haystack: list[str], needle: list[str]) -> int | None:
    """Index of the first exact occurrence of ``needle`` in ``haystack``."""
    if not needle:
        return None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _apply_change_onto(remote: str, before: str, after: str) -> str | None:
    """Apply a before→after edit onto ``remote`` text.

    Returns the merged text, or None when the edit cannot be applied cleanly
    (the remote drifted too far from what the agent edited). Each hunk of the
    before→after diff is matched by its surrounding context on the remote, so
    only the agent's actual change is carried over — never a whole-file
    overwrite.
    """
    if before == remote:
        return after  # no drift: the agent's result applies directly
    if before == after:
        return remote  # no actual change

    remote_lines = remote.splitlines()
    hunks: list[tuple[list[str], list[str]]] = []
    cur_old: list[str] = []
    cur_new: list[str] = []
    in_hunk = False
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), n=3, lineterm=""):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            if in_hunk:
                hunks.append((cur_old, cur_new))
            cur_old, cur_new, in_hunk = [], [], True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            cur_old.append(line[1:])
        elif line.startswith("+"):
            cur_new.append(line[1:])
        elif line.startswith(" "):
            cur_old.append(line[1:])
            cur_new.append(line[1:])
    if in_hunk:
        hunks.append((cur_old, cur_new))

    if not hunks:
        return None

    result = list(remote_lines)
    for old_block, new_block in hunks:
        idx = _find_block(result, old_block)
        if idx is None:
            return None  # conflict: remote drifted too far from the agent's base
        result = result[:idx] + new_block + result[idx + len(old_block) :]

    text = "\n".join(result)
    if remote.endswith("\n"):
        text += "\n"
    return text


def _commit_all(
    gh: GitHubClient,
    repo: str,
    branch: str,
    base_sha: str,
    base_tree_sha: str,
    commit_message: str,
    changed: list[tuple[str, bytes]],
    deleted: list[str],
) -> str:
    """Commit every change as exactly one commit on ``branch`` via the Git data API.

    ``base_sha`` is the parent *commit*; ``base_tree_sha`` is that commit's
    *tree* object (the Git data API rejects a commit SHA as ``base_tree``).
    """
    tree_entries: list[dict[str, Any]] = []
    for rel, data in changed:
        text = _decode_text(data)
        if text is not None:
            tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "content": text})
        else:
            tree_entries.append(
                {"path": rel, "mode": "100644", "type": "blob", "sha": gh.create_blob(repo, data)}
            )
    for rel in deleted:
        tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": None})
    tree_sha = gh.create_tree(repo, tree_entries, base_tree=base_tree_sha)
    commit_sha = gh.create_commit(repo, commit_message, tree_sha, [base_sha])
    gh.update_ref(repo, branch, commit_sha)
    return commit_sha


def push_workspace_to_pr(
    workspace_path: Path,
    repo_url: str,
    token: str,
    commit_message: str,
    pr_title: str,
    pr_body: str,
    base_branch: str = "main",
    change_records: list[dict] | None = None,
) -> dict:
    """Push the workspace changes to GitHub and open a pull request.

    ``change_records`` is the agent's recorded edits (``path``, ``change_type``,
    ``before_text``, ``after_text``). When supplied — the normal path — each
    edit is re-applied onto the *current* remote file, so the pull request
    carries only the agent's change. A snapshot that drifted from the target
    branch (renames, skipped files, stale mirrors) can therefore never rewrite
    or delete real code.

    Without ``change_records`` the whole workspace is diffed against the base
    branch — new, modified and deleted files — in a single commit on a fresh
    branch ``codeweaver/<workspace name>/<UTC timestamp>``.

    Returns ``{success, pr_url, branch, files_updated, pr_number, skipped}``.
    Raises :class:`GitHubPushError` for user-fixable problems and
    :class:`GitHubError` for API failures.
    """
    repo = parse_repo_slug(repo_url)
    workspace_path = Path(workspace_path)
    if not workspace_path.is_dir():
        raise GitHubPushError(f"workspace directory not found: {workspace_path}")

    run_id = workspace_path.name or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch = f"codeweaver/{run_id}/{timestamp}"

    files, skipped = _workspace_files(workspace_path)
    files_map = dict(files)
    if not files:
        raise GitHubPushError("the workspace contains no pushable files — nothing to do")

    with GitHubClient(token) as gh:
        base_sha = gh.get_branch_sha(repo, base_branch)
        if not base_sha:
            raise GitHubPushError(
                f"base branch {base_branch!r} not found in {repo} — check the repo URL and branch name"
            )

        # Diff the workspace against the base branch locally (blob SHAs).
        remote_shas, truncated = gh.get_tree(repo, base_sha)
        changed: list[tuple[str, bytes, str | None]] = []  # (path, bytes, remote sha or None)
        deleted: list[str] = []
        conflicts: list[str] = []

        if change_records is not None:
            # Safe mode: carry over exactly the agent's edits onto the remote.
            for rec in change_records:
                rel = (rec.get("path") or "").strip()
                ctype = (rec.get("change_type") or "modified").lower()
                if not rel:
                    continue
                if ctype == "deleted":
                    if rel in remote_shas:
                        deleted.append(rel)
                    continue
                data = files_map.get(rel)
                if data is None:
                    conflicts.append(f"{rel} (not found in workspace)")
                    continue
                remote = remote_shas.get(rel)
                if ctype == "created" or remote is None:
                    if remote is not None:
                        conflicts.append(f"{rel} (already exists on {base_branch})")
                        continue
                    changed.append((rel, data, None))
                    continue
                # Modified: merge the agent's edit onto the current remote file.
                before_text = rec.get("before_text") or ""
                after_text = rec.get("after_text") or ""
                remote_text = gh.get_file_content(repo, base_branch, rel)
                if remote_text is None:
                    conflicts.append(f"{rel} (remote content unreadable)")
                    continue
                merged = _apply_change_onto(remote_text, before_text, after_text)
                if merged is None:
                    conflicts.append(f"{rel} (differs from the agent's base — edit not applicable)")
                    continue
                merged_bytes = merged.encode("utf-8")
                if git_blob_sha(merged_bytes) == remote:
                    continue  # already identical on the remote
                changed.append((rel, merged_bytes, remote))
        elif truncated:
            # Very large repo: fall back to one contents-API lookup per file.
            for rel, data in files:
                remote = gh.get_file_sha(repo, base_branch, rel)
                if remote != git_blob_sha(data):
                    changed.append((rel, data, remote))
            deleted = []  # deletions cannot be detected cheaply here
        else:
            local_present = set(skipped)
            for rel, data in files:
                remote = remote_shas.get(rel)
                if remote != git_blob_sha(data):
                    changed.append((rel, data, remote))
                local_present.add(rel)
            deleted = sorted(p for p in remote_shas if p not in local_present)

        if not changed and not deleted:
            detail = "; ".join(conflicts[:4])
            raise GitHubPushError(
                f"nothing to push — the agent's changes could not be applied to {repo}@{base_branch}"
                + (f" ({detail})" if detail else "")
            )
        if len(changed) > MAX_CHANGED_FILES:
            raise GitHubPushError(
                f"{len(changed)} changed files exceed the {MAX_CHANGED_FILES}-file push budget"
            )

        gh.create_branch(repo, base_branch, branch)

        # The Git data API needs the base commit's *tree* object for base_tree
        # (a commit SHA there returns 404).
        base_tree_sha = gh.get_commit_tree_sha(repo, base_sha)

        # Single modified/created text file: the Contents API is the cheapest
        # path (and matches update_file's contract). Everything else becomes
        # exactly one commit via the Git data API.
        if len(changed) == 1 and not deleted and _decode_text(changed[0][1]) is not None:
            rel, data, remote = changed[0]
            gh.update_file(repo, branch, rel, _decode_text(data) or "", commit_message, remote)
        else:
            _commit_all(
                gh, repo, branch, base_sha, base_tree_sha, commit_message,
                [(rel, data) for rel, data, _ in changed], deleted,
            )

        pr = gh.create_pull_request(
            repo,
            pr_title or f"CodeWeaver: {commit_message}",
            pr_body or commit_message,
            head=branch,
            base=base_branch,
        )
        return {
            "success": True,
            "pr_url": pr["html_url"],
            "branch": branch,
            "files_updated": len(changed) + len(deleted),
            "pr_number": pr["number"],
            "skipped": conflicts,
        }
