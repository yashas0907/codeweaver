"""Tests for registration and login."""

from __future__ import annotations

import pytest

from app.auth_service import AuthError, login, register_user


@pytest.fixture()
def store():
    out = {}
    register_user("alice", "sup3rsecret!", out)
    return out


def test_register(store):
    assert "alice" in store
    assert store["alice"]["roles"] == ["user"]


def test_register_duplicate_raises(store):
    with pytest.raises(AuthError):
        register_user("alice", "another-pass1", store)


def test_login_ok(store):
    session = login("alice", "sup3rsecret!", store)
    assert session["username"] == "alice"


def test_login_bad_password(store):
    with pytest.raises(AuthError):
        login("alice", "wrong-password", store)


def test_login_unknown_user(store):
    with pytest.raises(AuthError):
        login("bob", "whatever123", store)
