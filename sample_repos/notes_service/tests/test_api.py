"""Tests for the notes HTTP handlers."""

from __future__ import annotations

import pytest

from app.api import NotesApi
from app.storage import NoteStorage


@pytest.fixture()
def api():
    storage = NoteStorage()
    storage.create("First note", "alpha body", tags=["work"])
    storage.create("Second note", "beta body", tags=["home"])
    storage.create("Third note", "gamma body", tags=["work"])
    return NotesApi(storage=storage)


def test_create_and_get(api):
    created = api.create_note_route({"title": "  New   Note  ", "body": "hello"})
    fetched = api.get_note_route(created["id"])
    assert fetched["title"] == "new note"
    assert fetched["body"] == "hello"


def test_list_sorted_newest_first(api):
    listing = api.list_notes_route()
    ids = [n["id"] for n in listing]
    # Newest notes have the highest ids; newest must come first.
    assert ids == sorted(ids, reverse=True), f"expected newest-first, got {ids}"


def test_list_with_tag_filter(api):
    listing = api.list_notes_route(tag="work")
    assert len(listing) == 2
    assert all("work" in n["tags"] for n in listing)


def test_health(api):
    assert api.health_route() == {"status": "ok", "notes": 3}


def test_delete(api):
    api.delete_note_route(1)
    with pytest.raises(KeyError):
        api.get_note_route(1)
