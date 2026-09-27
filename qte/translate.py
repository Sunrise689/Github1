"""QTE plain-language translation helpers.

The translator only adds explanations. It never changes numbers or turns a
descriptive result into a recommendation.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

LEXICON = Path(__file__).resolve().parent / "translation_lexicon.json"


@lru_cache(maxsize=1)
def load_lexicon() -> list[dict[str, Any]]:
    return json.loads(LEXICON.read_text(encoding="utf-8"))["terms"]


def _contains(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if phrase.isascii() and phrase.replace("-", "").replace(" ", "").isalnum():
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
    return phrase.casefold() in text.casefold()


def explain_terms(text: str, *, max_terms: int = 8) -> list[dict[str, str]]:
    """Return de-duplicated explanations for terms that occur in ``text``."""
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in load_lexicon():
        phrases = [str(item["term"]), *(str(alias) for alias in item.get("alias", []))]
        if any(_contains(text, phrase) for phrase in phrases):
            key = str(item["term"]).casefold()
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "term": str(item["term"]),
                "plain": str(item["plain"]),
                "usage_note": str(item.get("usage_note", "")),
            })
            if len(matches) >= max_terms:
                break
    return matches


def translate_text(text: str, *, max_terms: int = 8) -> str:
    """Append a human-readable glossary while preserving the input verbatim."""
    explanations = explain_terms(text, max_terms=max_terms)
    if not explanations:
        return text
    lines = [text, "", "大白话翻译："]
    for item in explanations:
        suffix = f"（注意：{item['usage_note']}）" if item["usage_note"] else ""
        lines.append(f"- {item['term']}：{item['plain']}{suffix}")
    return "\n".join(lines)
