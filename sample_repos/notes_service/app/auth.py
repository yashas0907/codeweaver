"""Token authentication helpers for the notes service."""

from __future__ import annotations

from app.config import API_TOKEN


def validate_token(token: str) -> bool:
    """Validate a bearer token against the configured API token."""
    if not token:
        return False
    return token == API_TOKEN


def require_admin(token: str, action: str) -> bool:
    if not validate_token(token):
        return False
    return action in ("read", "write", "delete")
