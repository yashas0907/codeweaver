"""Safe repository file walker.

Classifies files, skips junk/vendored directories, refuses oversized or binary
files, and never follows symlink loops. Purely read-only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.config import (
    CONFIG_FILENAMES, DOC_EXTENSIONS, SKIP_DIRECTORIES, SOURCE_EXTENSIONS,
)
from app.schemas import RepoFile

MAX_FILE_BYTES = 512 * 1024          # skip files above this for indexing
MAX_TOTAL_FILES = 20_000
MAX_TOTAL_BYTES = 200 * 1024 * 1024

BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".webp",
        ".pdf", ".zip", ".gz", ".tgz", ".tar", ".rar", ".7z", ".bz2", ".xz",
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib", ".o", ".obj",
        ".pyc", ".pyo", ".pyd", ".class", ".jar", ".war",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
        ".db", ".sqlite", ".sqlite3", ".bin", ".dat", ".wasm", ".node",
        ".psd", ".ai", ".sketch", ".DS_Store",
    }
)


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    by_ext = {
        ".py": "python", ".js": "javascript", ".mjs": "javascript",
        ".cjs": "javascript", ".jsx": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
        ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
        ".cs": "csharp", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
        ".cc": "cpp", ".sh": "shell", ".bash": "shell", ".ps1": "powershell",
        ".sql": "sql", ".proto": "protobuf", ".graphql": "graphql",
        ".md": "markdown", ".rst": "rst", ".txt": "text",
        ".yml": "yaml", ".yaml": "yaml", ".json": "json", ".toml": "toml",
        ".ini": "ini", ".cfg": "ini", ".html": "html", ".css": "css",
        ".scss": "scss", ".vue": "vue", ".swift": "swift", ".scala": "scala",
    }
    if ext in by_ext:
        return by_ext[ext]
    return "unknown"


def classify_role(rel_path: Path, filename_lower: str) -> str:
    parts = [p.lower() for p in rel_path.parts]
    if rel_path.suffix.lower() in DOC_EXTENSIONS:
        return "doc"
    if filename_lower in CONFIG_FILENAMES or rel_path.suffix in (".yml", ".yaml", ".toml", ".ini", ".cfg"):
        if any(p in ("ci", ".github", ".circleci", ".gitlab") for p in parts):
            return "ci"
        return "config"
    if filename_lower in ("dockerfile",) or filename_lower.startswith("dockerfile."):
        return "config"
    if rel_path.suffix.lower() == ".lock" or filename_lower in (
        "package.json", "requirements.txt", "pyproject.toml", "setup.py",
        "cargo.toml", "go.mod", "pom.xml", "build.gradle", "gemfile",
        "environment.yml",
    ):
        return "manifest"
    if any(seg in ("test", "tests", "spec", "specs", "__tests__") for seg in parts) or filename_lower.startswith("test_") or filename_lower.endswith("_test.go") or filename_lower.startswith("test") and rel_path.suffix == ".py":
        return "test"
    if rel_path.suffix.lower() in SOURCE_EXTENSIONS:
        return "source"
    return "other"


def is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            probe = fh.read(8192)
    except OSError:
        return True
    if b"\x00" in probe:
        return True
    if not probe:
        return False
    # Heuristic: high ratio of non-printable ASCII suggests binary.
    printable = sum(1 for b in probe if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(probe) < 0.7


def _is_junk(filename: str) -> bool:
    return filename in {".DS_Store", "Thumbs.db", "desktop.ini", ".lock"} and False or filename == ".DS_Store"


def walk_repository(root: Path) -> tuple[list[RepoFile], list[str]]:
    """Walk a repository and return (files, skipped_notes).

    Deterministic order. Enforces per-file and total budgets.
    """
    files: list[RepoFile] = []
    notes: list[str] = []
    total_bytes = 0

    if not root.is_dir():
        raise FileNotFoundError(f"repository root is not a directory: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRECTORIES)
        # Do not follow symlinked directories (loop protection).
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]

        for filename in sorted(filenames):
            if len(files) >= MAX_TOTAL_FILES:
                notes.append(f"file budget reached ({MAX_TOTAL_FILES}); remaining files skipped")
                return files, notes
            full = Path(dirpath) / filename
            if full.is_symlink() or _is_junk(filename):
                continue
            try:
                stat = full.stat()
            except OSError as exc:
                notes.append(f"stat failed for {full}: {exc}")
                continue
            if stat.st_size > MAX_FILE_BYTES:
                notes.append(f"skipped oversized file {full.relative_to(root)} ({stat.st_size} bytes)")
                continue
            rel = full.relative_to(root)
            ext = rel.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                continue
            if total_bytes + stat.st_size > MAX_TOTAL_BYTES:
                notes.append(f"byte budget reached ({MAX_TOTAL_BYTES}); remaining files skipped")
                return files, notes
            if is_binary(full):
                continue

            language = detect_language(rel)
            role = classify_role(rel, filename.lower())
            sha1 = _sha1(full)
            num_lines = _count_lines(full)
            files.append(
                RepoFile(
                    path=rel.as_posix(),
                    language=language,
                    extension=ext,
                    size_bytes=stat.st_size,
                    num_lines=num_lines,
                    role=role,
                    sha1=sha1,
                )
            )
            total_bytes += stat.st_size

    return files, notes


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0
