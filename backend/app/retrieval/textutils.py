"""Tokenization utilities for lexical retrieval.

Language-aware token splitting (camelCase, snake_case, dotted paths), a
compact stopword list, and light suffix stripping. No external NLP deps.
"""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being but by can could did do does doing
    done for from get gets got had has have having he her here hers him his
    how i if in into is it its just like me more most my no nor not of on
    onto or our out over own she should so some such than that the their
    them then there these they this those through to too under until up us
    very was we were what when where which while who whom why will with
    would you your add new make change update use using used want need
    please""".split()
)

_SPLIT_RE = re.compile(r"[^A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ies", "ed", "es", "s", "ly", "ment", "ness")


def _strip_suffix(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def split_identifier(token: str) -> list[str]:
    """`getUserById` / `get_user_by_id` / `user.get` -> [get, user, by, id]."""
    parts: list[str] = []
    for chunk in _SPLIT_RE.split(token):
        if not chunk:
            continue
        for piece in _CAMEL_RE.split(chunk):
            if piece:
                parts.append(piece.lower())
    return parts


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _SPLIT_RE.split(text.replace("_", " ").replace(".", " ")):
        if not raw or raw.lower() in STOPWORDS or raw.isdigit() and len(raw) > 4:
            continue
        for piece in _CAMEL_RE.split(raw):
            piece = piece.lower().strip()
            if not piece or piece in STOPWORDS or len(piece) < 2:
                continue
            tokens.append(_strip_suffix(piece))
    return tokens


def keywords(text: str, limit: int = 12) -> list[str]:
    """Distinctive keywords of a phrase, preserving order, no stopwords."""
    seen: list[str] = []
    for raw in _SPLIT_RE.split(text.replace("_", " ").replace(".", " ")):
        if not raw or raw.lower() in STOPWORDS:
            continue
        for piece in _CAMEL_RE.split(raw):
            piece = piece.lower()
            if piece and piece not in seen and len(piece) >= 2 and not piece.isdigit():
                seen.append(piece)
    return seen[:limit]


def looks_like_path(token: str) -> bool:
    return ("/" in token or token.endswith((".py", ".js", ".ts", ".go", ".java", ".rb", ".md", ".yml", ".yaml", ".json", ".tsx", ".jsx")))


def looks_like_symbol(token: str) -> bool:
    return "_" in token or (token[:1].islower() and any(c.isupper() for c in token)) or token in ("pagination",) and False
