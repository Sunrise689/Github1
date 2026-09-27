"""In-memory BM25 retrieval over QTE structured knowledge.

Backend retrieval includes every eligible market, including China.  A caller
may request ``frontend_visible=True`` when it is preparing a presentation;
that opt-in filter, not the compiler, enforces display policy.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .index_builder import INDEX, build, tokenize


RECORD_TYPE_WEIGHT = {
    "official_data_evidence": 1.22,
    "paper_evidence": 1.18,
    "evidence_pending": 1.2,
    "factor_definition": 1.14,
    "legacy_factor_definition": 0.95,
    "factor_evidence_link": 1.12,
    "framework_specification": 1.1,
    "official_dataset": 1.08,
    "translation_term": 1.08,
    "knowledge": 1.0,
}


class SparseBackend:
    """BM25 implementation and seam for a future dense retrieval backend."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.avgdl = sum(len(doc["tokens"]) for doc in docs) / max(len(docs), 1)
        self.df: dict[str, int] = {}
        for doc in docs:
            for token in set(doc["tokens"]):
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, query: str, doc: dict[str, Any]) -> float:
        query_terms = tokenize(query)
        counts: dict[str, int] = {}
        for token in doc["tokens"]:
            counts[token] = counts.get(token, 0) + 1
        k1, b = 1.5, 0.75
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            document_frequency = self.df.get(term, 0)
            inverse_frequency = math.log(
                1.0 + (len(self.docs) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * len(doc["tokens"]) / max(self.avgdl, 1.0))
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        return score


@lru_cache(maxsize=1)
def _load_docs() -> list[dict[str, Any]]:
    path = INDEX / "docs.json"
    if not path.exists():
        build()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["docs"]


def reload_index() -> None:
    """Clear the process cache after an offline index rebuild."""
    _load_docs.cache_clear()


def _values(record: dict[str, Any], key: str) -> list[Any]:
    if key == "factor_id":
        value = record.get("factor_ids") or record.get("factor_id")
    elif key == "topic_id":
        value = record.get("topic_ids") or record.get("topic_id")
    elif key == "category":
        value = record.get("factor_categories") or record.get("category")
    else:
        value = record.get(key)
    return value if isinstance(value, list) else [value]


def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "frontend_visible":
            if not expected:
                continue
            policy = record.get("display_policy") or {}
            if policy.get("frontend_default", "visible") != "visible":
                return False
            continue
        actual_values = _values(record, key)
        expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(value in expected_values for value in actual_values):
            return False
    return True


def _rank_weight(record: dict[str, Any]) -> float:
    """Prefer primary/official evidence without excluding supplementary work."""
    weight = RECORD_TYPE_WEIGHT.get(str(record.get("record_type", "")), 1.0)
    metadata = record.get("metadata") or {}
    if metadata.get("authority_level") == "supplementary_research":
        weight *= 0.72
    quality = record.get("quality", record.get("quality_score", 1.0))
    try:
        # Quality changes ordering modestly; it never replaces source filters.
        weight *= 0.85 + 0.15 * min(1.0, max(0.0, float(quality)))
    except (TypeError, ValueError):
        pass
    return weight


def search(query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return top-k structured records.

    Useful filters include ``factor_id``, ``topic_id``, ``market``,
    ``record_type``, ``source_id`` and ``frontend_visible``.  Omitting
    ``frontend_visible`` means an internal/backend search and does not hide CN.
    """
    if not query.strip() or top_k <= 0:
        return []
    docs = _load_docs()
    backend = SparseBackend(docs)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        record = doc["record"]
        if not _matches(record, filters or {}):
            continue
        score = backend.score(query, doc) * _rank_weight(record)
        if score > 0:
            ranked.append((score, record))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("doc_id", ""))))
    results: list[dict[str, Any]] = []
    seen_content: set[tuple[str, str, str]] = set()
    for score, record in ranked:
        content = str(
            record.get("plain_summary")
            or record.get("claim_plain")
            or record.get("plain_definition")
            or record.get("plain")
            or record.get("doc_id", "")
        )
        dedupe_key = (str(record.get("record_type", "")), str(record.get("source_id", "")), content)
        if dedupe_key in seen_content:
            continue
        seen_content.add(dedupe_key)
        results.append({**record, "_score": round(score, 6)})
        if len(results) >= top_k:
            break
    return results
