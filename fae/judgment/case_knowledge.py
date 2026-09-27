"""Structured textbook-case retrieval for the FAE judgment layer.

The library stores machine-usable decisions, not OCR prose. Each case records
patterns, context, evidence, the resolved direction/state, invalidation clues,
and an auditable book/page source.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pattern_registry import canonicalize_pattern_id, legacy_alias_map, variant_map
from .schemas import DetectedSignal, Resolution, SignalState


@dataclass(frozen=True)
class CompiledCase:
    """One textbook example compiled into FAE decision features."""

    case_id: str
    source_book: str
    source_file: str
    pdf_pages: list[int]
    source_label: str
    patterns: list[str]
    pattern_family: str
    trend_context: str
    position_context: str
    timeframe: str
    evidence_tags: list[str]
    conflict_tags: list[str]
    correct_direction: str
    correct_state: str
    interpretation_code: str
    priority_tier: str
    invalidation_tags: list[str]
    quality_score: float
    compiler_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable case record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompiledCase:
        """Build a validated record from JSON data."""
        return cls(**dict(value))


@dataclass(frozen=True)
class CaseMatch:
    """Similarity result used by the judgment engine."""

    case: CompiledCase
    similarity: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the compact result exposed by ``JudgmentEngine``."""
        return {
            "case_id": self.case.case_id,
            "similarity": self.similarity,
            "source_book": self.case.source_book,
            "pdf_pages": self.case.pdf_pages,
            "source_label": self.case.source_label,
            "patterns": self.case.patterns,
            "correct_direction": self.case.correct_direction,
            "correct_state": self.case.correct_state,
            "interpretation_code": self.case.interpretation_code,
            "quality_score": self.case.quality_score,
            "reasons": self.reasons,
        }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_PATTERN_COMPAT_ALIASES = legacy_alias_map()
_PATTERN_COMPAT_ALIASES.update(variant_map())


def _normalize_pattern_id(pattern: str) -> str:
    """Normalize renamed pattern IDs while reading legacy compiled cases."""
    return canonicalize_pattern_id(_PATTERN_COMPAT_ALIASES.get(pattern, pattern))


class CaseKnowledgeBase:
    """Load and retrieve structured textbook examples without a network."""

    def __init__(self, path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parent
        self.path = Path(path or root / "cases" / "compiled_cases.jsonl")
        self.cases = self._load()

    def _load(self) -> list[CompiledCase]:
        if not self.path.exists():
            return []
        cases: list[CompiledCase] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                cases.append(CompiledCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"invalid compiled case at {self.path}:{line_number}"
                ) from exc
        return cases

    @staticmethod
    def _query_tags(
        signals: Sequence[DetectedSignal],
        evidence: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> tuple[set[str], set[str], str, str, str]:
        patterns = {_normalize_pattern_id(signal.pattern) for signal in signals}
        tags: set[str] = set()
        if int(evidence.get("gap_count", 0)) > 0:
            tags.add("gap")
        if int(evidence.get("gap_count", 0)) >= 2:
            tags.add("multiple_gaps")
        if evidence.get("last_gap_filled"):
            tags.add("gap_filled")
        if int(evidence.get("days_below_neckline", 0)) >= 2:
            tags.add("neckline_break")
        if evidence.get("reclaim_neckline"):
            tags.add("neckline_reclaimed")
        if evidence.get("breakout_confirmed"):
            tags.add("breakout_confirmed")
        if evidence.get("false_break"):
            tags.add("false_break")
        volume_ratio = float(evidence.get("volume_ratio", 1.0))
        if volume_ratio >= 1.2:
            tags.add("volume_expansion")
        elif volume_ratio <= 0.8:
            tags.add("volume_contraction")
        trend = str(context.get("trend", evidence.get("trend", "unknown")))
        position_value = float(
            context.get("position_pct", evidence.get("position_pct", 0.5))
        )
        position = (
            "high"
            if position_value >= 0.7
            else "low"
            if position_value <= 0.3
            else "mid"
        )
        timeframe = str(context.get("timeframe", "daily"))
        return patterns, tags, trend, position, timeframe

    def search(
        self,
        signals: Sequence[DetectedSignal],
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        *,
        top_k: int = 8,
        min_similarity: float = 0.30,
    ) -> list[CaseMatch]:
        """Return structurally similar textbook decisions."""
        query_patterns, query_tags, trend, position, timeframe = self._query_tags(
            signals, evidence, context or {}
        )
        results: list[CaseMatch] = []
        for case in self.cases:
            if case.quality_score < 0.65:
                continue
            case_patterns = {
                _normalize_pattern_id(pattern) for pattern in case.patterns
            }
            pattern_score = _jaccard(query_patterns, case_patterns)
            if pattern_score == 0:
                continue
            evidence_score = _jaccard(query_tags, set(case.evidence_tags))
            context_hits = 0
            context_total = 0
            reasons = [f"形态重合={pattern_score:.2f}"]
            for actual, expected, label in (
                (trend, case.trend_context, "趋势"),
                (position, case.position_context, "位置"),
                (timeframe, case.timeframe, "周期"),
            ):
                if expected not in {"unknown", "any", ""}:
                    context_total += 1
                    if actual == expected:
                        context_hits += 1
                        reasons.append(f"{label}一致")
            context_score = context_hits / context_total if context_total else 0.5
            quality = max(0.0, min(1.0, case.quality_score))
            similarity = (
                0.52 * pattern_score
                + 0.20 * evidence_score
                + 0.18 * context_score
                + 0.10 * quality
            )
            if similarity >= min_similarity:
                if evidence_score:
                    reasons.append(f"证据重合={evidence_score:.2f}")
                results.append(CaseMatch(case, similarity, reasons))
        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[: max(0, top_k)]

    @staticmethod
    def consensus(matches: Sequence[CaseMatch]) -> dict[str, Any]:
        """Calculate weighted case consensus without overriding hard rules."""
        usable = [match for match in matches if match.similarity >= 0.45]
        if not usable:
            return {
                "direction": "neutral",
                "strength": 0.0,
                "supporting_cases": 0,
                "policy": "no_case_adjustment",
            }
        weights: Counter[str] = Counter()
        for match in usable:
            direction = match.case.correct_direction
            if direction in {"bullish", "bearish", "neutral"}:
                weights[direction] += match.similarity * match.case.quality_score
        total = sum(weights.values())
        direction, value = weights.most_common(1)[0] if weights else ("neutral", 0.0)
        strength = value / total if total else 0.0
        return {
            "direction": direction,
            "strength": strength,
            "supporting_cases": len(usable),
            "weights": dict(weights),
            "policy": "case_consensus_is_bounded_and_cannot_bypass_quant_gate",
        }

    @staticmethod
    def apply_consensus(
        resolution: Resolution,
        consensus: Mapping[str, Any],
    ) -> Resolution:
        """Bound case influence to confidence/degradation, never hard override."""
        if resolution.winner is None:
            return resolution
        direction = str(consensus.get("direction", "neutral"))
        strength = float(consensus.get("strength", 0.0))
        count = int(consensus.get("supporting_cases", 0))
        if count < 3 or strength < 0.60 or direction == "neutral":
            return resolution
        winner_direction = resolution.winner.direction.value
        if direction == winner_direction:
            resolution.confidence = min(
                1.0, resolution.confidence + min(0.10, strength * 0.10)
            )
            resolution.reasons.append(
                f"{count}个相似教材案例形成同向共识（{strength:.0%}）"
            )
        elif not resolution.winner.confirmed:
            resolution.confidence = max(
                0.0, resolution.confidence - min(0.12, strength * 0.12)
            )
            resolution.final_state = SignalState.DEGRADED
            resolution.reasons.append(
                f"{count}个相似教材案例与候选方向冲突，信号降级（{strength:.0%}）"
            )
        else:
            resolution.reasons.append(
                "教材案例共识与已确认高优先级信号冲突，仅记录而不覆盖"
            )
        return resolution
