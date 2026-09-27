"""QTE: descriptive quantitative evidence translation, never trading decisions."""

from __future__ import annotations

from typing import Any


def search(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Lazy package-level search wrapper; avoids import cycles in CLI builds."""
    from .retrieval import search as _search

    return _search(query=query, top_k=top_k, filters=filters)


def translate_text(text: str, *, max_terms: int = 8) -> str:
    """Lazy package-level plain-language translation wrapper."""
    from .translate import translate_text as _translate_text

    return _translate_text(text, max_terms=max_terms)


__all__ = ["search", "translate_text"]
