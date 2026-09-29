"""LLM abstraction: provider protocol + response model.

CodeWeaver never calls a vendor SDK directly. Stages go through LLMService,
which routes to whichever provider is configured and degrades to the built-in
deterministic engine when no provider is available. API keys are resolved at
call time from environment variables (names configured in Settings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    model: str = ""
    error: str = ""
    input_chars: int = 0
    output_chars: int = 0
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when the provider is unavailable or fails after retries."""


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    async def aclose(self) -> None:
        ...
