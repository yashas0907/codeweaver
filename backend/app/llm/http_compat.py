"""HTTP chat-completions provider.

Connects to any endpoint exposing POST {base_url}/chat/completions. Retries
transient failures with exponential backoff; never logs the API key.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from app.llm.base import LLMResponse, ProviderError


class HTTPCompatibleProvider:
    name = "http_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int = 120,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json", "User-Agent": "CodeWeaver/1.0"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self._client

    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else 0.2,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                client = self._ensure_client()
                resp = await client.post(url, json=payload)
                if resp.status_code == 413:
                    # Payload too large for the provider: shrink the user content
                    # (context) and retry instead of failing the whole step.
                    user_content = payload["messages"][1]["content"]
                    if len(user_content) > 4000:
                        payload["messages"][1]["content"] = user_content[: max(2000, len(user_content) // 2)]
                        last_error = "payload too large; retrying with truncated context"
                        continue
                    last_error = "provider rejected the request as too large (HTTP 413)"
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"provider returned HTTP {resp.status_code}"
                    if resp.status_code == 429:
                        # Free tiers throttle on tokens-per-minute; honour the
                        # provider's retry-after hint and back off generously.
                        retry_after = 0.0
                        try:
                            retry_after = float(resp.headers.get("retry-after", "0") or 0)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                        wait = max(retry_after, min(4 * (attempt + 1), 45))
                    else:
                        wait = min(2 ** attempt, 8)
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in (401, 403):
                    raise ProviderError(
                        "provider rejected credentials (HTTP "
                        f"{resp.status_code}); check CODEWEAVER_LLM_API_KEY"
                    )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                duration_ms = int((time.monotonic() - started) * 1000)
                usage = data.get("usage", {})
                return LLMResponse(
                    ok=True, text=text, model=data.get("model", self.model),
                    input_chars=len(system) + len(user), output_chars=len(text),
                    duration_ms=duration_ms,
                    meta={"usage": usage, "endpoint": self.base_url},
                )
            except ProviderError:
                raise
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = str(exc)[:300]
                await asyncio.sleep(min(2 ** attempt, 8))
        raise ProviderError(f"LLM request failed after retries: {last_error}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
