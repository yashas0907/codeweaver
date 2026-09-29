"""Domain model for notes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Note:
    id: int
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
