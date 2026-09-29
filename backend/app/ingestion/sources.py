"""Repository source adapters: local path, git URL, GitHub URL.

The adapters produce a *snapshot* of a repository inside CodeWeaver's own store
(`data/repo_store/<repo_id>`). Agent workspaces branch from snapshots, never
from the user's original checkout, so originals are never modified.

GitHub works unauthenticated for public repositories (codeload tarball). When
a token is configured (env var name in settings) it is sent as a Bearer
header — the value itself is never logged.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings


class SourceError(RuntimeError):
    """Raised when a repository source cannot be acquired."""


def _safe_target(base: Path, repo_id: str) -> Path:
    target = (base / repo_id).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise SourceError("unsafe repository store path")
    return target


class RepoSource(ABC):
    """Acquires a repository snapshot into `store_root/<repo_id>`."""

    def __init__(self, store_root: Path) -> None:
        self.store_root = store_root
        self.store_root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def acquire(self, repo_id: str) -> Path:
        """Materialize the repo at store_root/<repo_id> and return the path."""

    @abstractmethod
    def describe(self) -> str:
        ...

    def finish(self, repo_id: str) -> Path:
        return _safe_target(self.store_root, repo_id)


class LocalRepoSource(RepoSource):
    """Copies a local directory (or clones a local git path) into the store."""

    def __init__(self, path: str, store_root: Path) -> None:
        super().__init__(store_root)
        p = Path(path).expanduser()
        if not p.is_absolute():
            # Resolve relative paths from the project root, not from the server CWD.
            from app.config import PROJECT_ROOT
            p = PROJECT_ROOT / p
        self.path = p.resolve()
        if not self.path.is_dir():
            raise SourceError(f"local repository path does not exist or is not a directory: {path}")

    def describe(self) -> str:
        return f"local:{self.path}"

    def acquire(self, repo_id: str) -> Path:
        target = _safe_target(self.store_root, repo_id)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        git_dir = self.path / ".git"
        if git_dir.exists():
            # Clone to preserve history for git intelligence; fall back to copy.
            if not _git_clone(self.path, target):
                shutil.copytree(self.path, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "node_modules", ".venv"))
        else:
            shutil.copytree(
                self.path, target,
                ignore=shutil.ignore_patterns("__pycache__", "node_modules", ".venv", ".git"),
            )
        return target


class GitUrlSource(RepoSource):
    """Clones any git-accessible URL (https/ssh/local file paths)."""

    def __init__(self, url: str, store_root: Path, branch: str = "") -> None:
        super().__init__(store_root)
        self.url = url
        self.branch = branch

    def describe(self) -> str:
        return f"git:{self.url}"

    def acquire(self, repo_id: str) -> Path:
        target = _safe_target(self.store_root, repo_id)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        ok = _git_clone(self.url, target, self.branch)
        if not ok:
            raise SourceError(f"git clone failed for {self.url!r} (is git installed and the URL reachable?)")
        return target


class GitHubSource(RepoSource):
    """Downloads a GitHub repository (public, unauthenticated; token optional)."""

    _PATTERNS = (
        re.compile(r"^https?://github\.com/([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$", re.IGNORECASE),
        re.compile(r"^([\w.\-]+)/([\w.\-]+)$"),
    )

    def __init__(self, url: str, store_root: Path, branch: str = "") -> None:
        super().__init__(store_root)
        self.raw = url.strip()
        self.branch = branch
        self.owner = ""
        self.repo = ""
        self._parse()

    def _parse(self) -> None:
        for pattern in self._PATTERNS:
            m = pattern.match(self.raw)
            if m:
                self.owner, self.repo = m.group(1), m.group(2)
                return
        raise SourceError(f"cannot parse GitHub reference: {self.raw!r}")

    def describe(self) -> str:
        return f"github:{self.owner}/{self.repo}"

    def acquire(self, repo_id: str) -> Path:
        import httpx  # local import keeps startup light

        target = _safe_target(self.store_root, repo_id)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        branch = self.branch or _github_default_branch(self.owner, self.repo, settings)
        archive_url = f"https://codeload.github.com/{self.owner}/{self.repo}/tar.gz/refs/heads/{branch}"
        headers = {"User-Agent": "CodeWeaver/1.0"}
        token = settings.resolve_github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "repo.tar.gz"
            try:
                with httpx.Client(timeout=60, follow_redirects=True) as client:
                    with client.stream("GET", archive_url, headers=headers) as resp:
                        if resp.status_code == 404:
                            raise SourceError(
                                f"GitHub repository {self.owner}/{self.repo} (branch {branch}) not found "
                                "or is private; set CODEWEAVER_GITHUB_TOKEN for private repos"
                            )
                        resp.raise_for_status()
                        with archive_path.open("wb") as fh:
                            for chunk in resp.iter_bytes():
                                fh.write(chunk)
            except httpx.HTTPError as exc:
                raise SourceError(f"GitHub download failed: {exc}") from exc
            self._extract(archive_path, target)
        return target

    def _extract(self, archive_path: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                if members:
                    root_prefix = members[0].name.split("/")[0]
                    for member in members:
                        # Strip the top-level directory GitHub adds.
                        parts = member.name.split("/", 1)
                        if len(parts) == 1:
                            continue
                        member.name = parts[1]
                        target_child = (target / member.name).resolve()
                        if not str(target_child).startswith(str(target.resolve())):
                            continue  # refuse path traversal in archives
                        tar.extract(member, target, filter="data")
        except tarfile.TarError as exc:
            raise SourceError(f"failed to extract GitHub archive: {exc}") from exc


def _github_default_branch(owner: str, repo: str, settings) -> str:
    import httpx

    headers = {"User-Agent": "CodeWeaver/1.0"}
    token = settings.resolve_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers, timeout=20,
        )
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
    except httpx.HTTPError:
        pass
    return "main"


def _git_clone(source: str, target: Path, branch: str = "") -> bool:
    """Best-effort `git clone`; returns False instead of raising."""
    import subprocess

    cmd = ["git", "clone", "--quiet", "--depth", "50"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [source, str(target)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={"GIT_TERMINAL_PROMPT": "0", **_minimal_env()},
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _minimal_env() -> dict[str, str]:
    import os

    keep = ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE")
    return {k: os.environ[k] for k in keep if k in os.environ}


def create_source(spec: dict, store_root: Path) -> RepoSource:
    """Factory: build the right source adapter from an API request payload."""
    source_type = (spec.get("source") or "local").lower()
    if source_type == "local":
        path = spec.get("path") or ""
        if not path:
            raise SourceError("local source requires 'path'")
        return LocalRepoSource(path, store_root)
    if source_type == "git":
        url = spec.get("url") or ""
        if not url:
            raise SourceError("git source requires 'url'")
        return GitUrlSource(url, store_root, branch=spec.get("branch", ""))
    if source_type == "github":
        url = spec.get("url") or spec.get("slug") or ""
        if not url:
            raise SourceError("github source requires 'url' (e.g. https://github.com/owner/repo or owner/repo)")
        return GitHubSource(url, store_root, branch=spec.get("branch", ""))
    raise SourceError(f"unknown repository source type: {source_type!r}")
