"""HTTP-facing handlers for the notes service.

Route functions follow the FastAPI-style decorator convention that CodeWeaver's
repository graph detects. `list_notes_route` returns a list — the natural
target for a "add pagination" task.
"""

from __future__ import annotations

from app.models import utc_timestamp
from app.storage import NoteStorage, _normalize_title_storage


def _normalize_title_api(title: str) -> str:
    # NOTE: duplicated on purpose from app/storage.py (see README)
    cleaned = title.strip().lower()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        cleaned = "untitled"
    return cleaned


class NotesApi:
    def __init__(self, storage: NoteStorage | None = None) -> None:
        self.storage = storage or NoteStorage()

    # -- routes ---------------------------------------------------------- #

    def list_notes_route(self, tag: str = None):
        """GET /notes — return notes as JSON dicts."""
        notes = self.storage.list_notes(tag=tag)
        return [n.to_dict() for n in notes]

    def get_note_route(self, note_id: int):
        """GET /notes/{id}"""
        return self.storage.get(note_id).to_dict()

    def create_note_route(self, payload: dict):
        """POST /notes"""
        title = _normalize_title_api(payload.get("title", ""))
        note = self.storage.create(
            title=title,
            body=payload.get("body", ""),
            tags=payload.get("tags", []),
        )
        return note.to_dict()

    def delete_note_route(self, note_id: int):
        """DELETE /notes/{id}"""
        self.storage.delete(note_id)
        return {"deleted": note_id}

    def health_route(self):
        """GET /health"""
        return {"status": "ok", "notes": self.storage.count()}
