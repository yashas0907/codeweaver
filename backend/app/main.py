"""CodeWeaver backend entrypoint: `python -m app.main` or uvicorn app.main:app."""

from __future__ import annotations

import uvicorn

from app.api.main import app
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
