"""Tests for the in-memory storage layer."""

from __future__ import annotations

import pytest

from app.models import Note
from app.storage import NoteNotFoundError, NoteStorage


def test_create_increments_ids():
    storage = NoteStorage()
    a = storage.create("A", "body a")
    b = storage.create("B", "body b")
    assert a.id == 1 and b.id == 2


def test_get_missing_raises():
    storage = NoteStorage()
    with pytest.raises(NoteNotFoundError):
        storage.get(999)


def test_search_matches_title_and_body():
    storage = NoteStorage()
    storage.create("Groceries", "milk eggs")
    storage.create("Work log", "standup notes")
    assert len(storage.search("milk")) == 1
    assert len(storage.search("log")) == 1
    assert storage.search("zzz") == []


def test_search_survives_odd_notes():
    storage = NoteStorage()
    storage.create("Normal", "fine")
    # A note with unusual content must not break search.
    storage._notes[99] = Note(id=99, title="x", body="y")
    assert isinstance(storage.search("fine"), list)


def test_count():
    storage = NoteStorage()
    assert storage.count() == 0
    storage.create("A", "b")
    assert storage.count() == 1
