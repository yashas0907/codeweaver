"""Central typed configuration for CodeWeaver.

All values come from environment variables (optionally via a .env file in the
project root). Nothing here ever contains secrets: API keys are referenced by
*env var name* and resolved lazily at call time.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_MODES = ("READ_ONLY", "WORKSPACE_EDIT")
LLM_PROVIDERS = ("deterministic", "http_compatible", "none")

# Paths the agent must never modify, regardless of task.
PROTECTED_PATH_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa*",
    "id_ed25519*",
    "credentials.json",
    "secrets.*",
    ".git/config",
    "forgeai.db",
)

# File extensions the indexer treats as source code (vs data/assets).
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
        ".java", ".kt", ".go", ".rs", ".rb", ".php", ".cs",
        ".c", ".h", ".cpp", ".hpp", ".cc",
        ".sh", ".bash", ".ps1",
        ".sql", ".proto", ".graphql",
    }
)

DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".rst", ".txt"})

CONFIG_FILENAMES: frozenset[str] = frozenset(
    {
        "package.json", "requirements.txt", "pyproject.toml", "setup.py",
        "setup.cfg", "Pipfile", "poetry.lock", "Cargo.toml", "go.mod",
        "pom.xml", "build.gradle", "Gemfile", "composer.json",
        "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "makefile", "cmakelists.txt", "tsconfig.json", "tox.ini",
        ".github-workflows", "environment.yml",
    }
)

SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
        "venv", "env", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", "target", ".next", ".nuxt", "coverage",
        ".idea", ".vscode", ".eggs", "site-packages", ".tox", ".cache",
    }
)


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_prefix="CODEWEAVER_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    host: str = "127.0.0.1"
    port: int = 8600
    data_dir: Path = PROJECT_ROOT / "data"
    workspace_root: Path = PROJECT_ROOT / "data" / "workspaces"
    db_url: str = f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'forgeai.db').as_posix()}"

    # Agent behaviour
    agent_mode: str = "WORKSPACE_EDIT"
    max_repair_iterations: int = Field(default=3, ge=0, le=10)
    max_test_timeout_seconds: int = Field(default=300, ge=5, le=3600)
    max_tool_timeout_seconds: int = Field(default=120, ge=5, le=3600)
    max_context_chars: int = Field(default=48_000, ge=2_000, le=400_000)

    # LLM
    llm_provider: str = "deterministic"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""  # captured from CODEWEAVER_LLM_API_KEY (.env or environment)
    llm_api_key_env: str = "CODEWEAVER_LLM_API_KEY"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2

    # Integrations / security
    github_token: str = ""  # captured from CODEWEAVER_GITHUB_TOKEN (.env or environment)
    github_token_env: str = "CODEWEAVER_GITHUB_TOKEN"
    allow_network_tools: bool = True

    @field_validator("agent_mode")
    @classmethod
    def _valid_agent_mode(cls, v: str) -> str:
        v = v.upper()
        if v not in AGENT_MODES:
            raise ValueError(f"agent_mode must be one of {AGENT_MODES}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = v.lower()
        if v not in LLM_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {LLM_PROVIDERS}")
        return v

    def resolve_api_key(self) -> str | None:
        """Resolve the LLM API key from the configured env var at call time.

        Checks the process environment first, then falls back to the value
        captured from the .env file. The key is never logged. Returns None
        when unset — callers must degrade gracefully instead of fabricating
        credentials.
        """
        value = os.environ.get(self.llm_api_key_env, "") or self.llm_api_key
        return value or None

    def resolve_github_token(self) -> str | None:
        """Resolve the GitHub token from the env var, falling back to the
        value captured from the .env file. Never logged."""
        value = os.environ.get(self.github_token_env, "") or self.github_token
        return value or None

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"
