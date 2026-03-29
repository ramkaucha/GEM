"""Text normalization helpers for ECG evaluation."""

from __future__ import annotations

import re
from typing import Iterable, List

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([,.;:!?])\1+")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize_text(text: object) -> str:
    """Return a lightweight normalized text representation."""
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = _REPEATED_PUNCT_RE.sub(r"\1", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value


def tokenize(text: object) -> List[str]:
    """Tokenize normalized text into alphanumeric tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    cleaned = _NON_WORD_RE.sub(" ", normalized)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.split(" ") if cleaned else []


def unique_tokens(text: object) -> set[str]:
    """Return the unique token set for a text."""
    return set(tokenize(text))


def deduplicate_ordered(items: Iterable[str]) -> List[str]:
    """Keep the first occurrence order while removing duplicates."""
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
