"""CodeWeaver – re-export layer for backend/app/."""
from __future__ import annotations
import sys as _sys
_BACKEND_MODS = {
    "agents":     "backend.app.agents",
    "api":        "backend.app.api",
    "config":     "backend.app.config",
    "db":         "backend.app.db",
    "schemas":    "backend.app.schemas",
    "workspace":  "backend.app.workspace",
    "ingestion":  "backend.app.ingestion",
    "intelligence": "backend.app.intelligence",
    "llm":        "backend.app.llm",
    "retrieval":  "backend.app.retrieval",
    "runner":     "backend.app.runner",
    "security":   "backend.app.security",
    "failure":    "backend.app.failure",
    "review":     "backend.app.review",
    "observability": "backend.app.observability",
    "evaluation": "backend.app.evaluation",
}
_cache = {}
def __getattr__(name):
    if name in _cache:
        return _cache[name]
    if name not in _BACKEND_MODS:
        raise AttributeError(f"module has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(_BACKEND_MODS[name])
    _cache[name] = mod
    return mod

def __dir__():
    return list(_BACKEND_MODS.keys())
