"""Typed data models shared by the FAE judgment layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .pattern_registry import canonicalize_pattern_id, variant_for_pattern


class SignalState(str, Enum):
    """Lifecycle state of an observed technical signal."""

    DETECTED = "detected"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    DEGRADED = "degraded"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class Direction(str, Enum):
    """Directional interpretation without implying a trading order."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternKind(str, Enum):
    """Pattern hierarchy used by the priority resolver."""

    SIMPLE = "simple"
    COMBINATION = "combination"
    TREND = "trend"
    CHART = "chart"


class Timeframe(str, Enum):
    """Supported bar timeframes in descending structural importance."""

    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    INTRADAY = "intraday"


@dataclass
class LineGeometry:
    """A support, resistance, neckline, or trend line to be drawn."""

    role: str
    x1: int
    y1: float
    x2: int
    y2: float
    style: str = "solid"


@dataclass
class ArcGeometry:
    """Quadratic arc used to draw rounding tops and bottoms."""

    role: str
    start: int
    end: int
    coefficients: tuple[float, float, float]


@dataclass
class PatternGeometry:
    """Serializable adaptive drawing instructions for one chart pattern."""

    lines: list[LineGeometry] = field(default_factory=list)
    arcs: list[ArcGeometry] = field(default_factory=list)
    pivots: list[tuple[int, float, str]] = field(default_factory=list)


@dataclass
class DetectedSignal:
    """Objective pattern observation before contextual interpretation."""

    pattern: str
    direction: Direction
    confidence: float
    start: int
    end: int
    kind: PatternKind
    timeframe: Timeframe = Timeframe.DAILY
    state: SignalState = SignalState.DETECTED
    confirmed: bool = False
    geometry: PatternGeometry | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.end < self.start:
            raise ValueError("signal end must be greater than or equal to start")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        value = asdict(self)
        value["direction"] = self.direction.value
        value["kind"] = self.kind.value
        value["timeframe"] = self.timeframe.value
        value["state"] = self.state.value
        # Keep ``pattern`` for backward compatibility while exposing the
        # canonical ID/variant expected by the frontend contract.
        raw_pattern = self.metadata.get("canonical_pattern", self.pattern)
        value["pattern_id"] = canonicalize_pattern_id(str(raw_pattern))
        value["variant"] = (
            self.metadata.get("canonical_variant")
            or variant_for_pattern(self.pattern)
            or None
        )
        # Optional presentation metadata is flattened for API consumers.  The
        # authoritative decision remains ``state``/``resolution``; these
        # fields only tell a renderer which layer to show by default.
        for field_name in (
            "display_role",
            "display_priority",
            "display_reason",
            "display_group_id",
            "display_group_size",
            "display_group_representative",
            "merged_spans",
        ):
            if field_name in self.metadata:
                value[field_name] = self.metadata[field_name]
        return value


@dataclass
class RuleMatch:
    """A rule selected by the engine together with its match score."""

    rule_id: str
    priority: int
    score: float
    rule: dict[str, Any]
    reasons: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    """Explainable result of resolving simultaneous signals."""

    winner: DetectedSignal | None
    losers: list[DetectedSignal]
    final_state: SignalState
    confidence: float
    applied_rules: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    invalidation_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result for reports and feedback."""
        return {
            "winner": self.winner.to_dict() if self.winner else None,
            "losers": [signal.to_dict() for signal in self.losers],
            "final_state": self.final_state.value,
            "confidence": self.confidence,
            "applied_rules": self.applied_rules,
            "reasons": self.reasons,
            "invalidation_conditions": self.invalidation_conditions,
        }
