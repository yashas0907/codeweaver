"""Authentication service."""

from __future__ import annotations

from app.passwords import hash_password, verify_password

SESSION_TTL_MINUTES = 60


class AuthError(Exception):
    pass


def register_user(username: str, password: str, store: dict) -> dict:
    if username in store:
        raise AuthError("user already exists")
    if len(password) < 8:
        raise AuthError("password too short")
    store[username] = {"password": hash_password(password), "roles": ["user"]}
    return {"username": username, "roles": ["user"]}


def login(username: str, password: str, store: dict) -> dict:
    record = store.get(username)
    if record is None:
        raise AuthError("unknown user")
    if not verify_password(password, record["password"]):
        raise AuthError("bad credentials")
    return {"username": username, "session_ttl": SESSION_TTL_MINUTES}
