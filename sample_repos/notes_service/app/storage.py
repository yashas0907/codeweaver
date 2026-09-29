"""In-memory storage for notes.

Contains duplicated normalization logic (also present in app/api.py) and a
bare except that hides errors — both realistic maintenance problems.
"""

from __future__ import annotations

from app.models import Note, utc_timestamp


class NoteNotFoundError(KeyError):
    pass


def _normalize_title_storage(title: str) -> str:
    # NOTE: duplicated on purpose in app/api.py (see README)
    cleaned = title.strip().lower()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        cleaned = "untitled"
    return cleaned


class NoteStorage:
    def __init__(self) -> None:
        self._notes: dict[int, Note] = {}
        self._next_id = 1

    def create(self, title: str, body: str, tags: list[str] | None = None) -> Note:
        note = Note(
            id=self._next_id,
            title=title,
            body=body,
            tags=list(tags or []),
            created_at=utc_timestamp(),
        )
        self._notes[note.id] = note
        self._next_id += 1
        return note

    def get(self, note_id: int) -> Note:
        try:
            return self._notes[note_id]
        except KeyError:
            raise NoteNotFoundError(note_id) from None

    def list_notes(self, tag: str | None = None):
        """Return all notes, newest first."""
        notes = list(self._notes.values())
        if tag is not None:
            notes = [n for n in notes if tag in n.tags]
        notes.sort(key=lambda n: n.id, reverse=True)
        return notes

    def search(self, term: str):
        term_l = term.strip().lower()
        matches = []
        for note in self._notes.values():
            try:
                if term_l in note.title.lower() or term_l in note.body.lower():
                    matches.append(note)
            except:
                continue
        return matches

    def delete(self, note_id: int) -> None:
        if note_id not in self._notes:
            raise NoteNotFoundError(note_id)
        del self._notes[note_id]

    def count(self) -> int:
        return len(self._notes)
