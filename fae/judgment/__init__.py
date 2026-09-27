"""FAE expert judgment, chart-pattern, and conflict-resolution layer."""

from .adapters import from_v7_events
from .case_knowledge import CaseKnowledgeBase, CaseMatch, CompiledCase
from .chart_patterns import ChartPatternDetector, plot_pattern_geometry
from .conflict_resolver import ConflictResolver
from .context_interpreter import ContextInterpreter
from .context_validators import filter_contextual_candlestick_signals
from .evidence_extractor import EvidenceExtractor, QuantitativeGate
from .feedback_store import FeedbackStore
from .human_calibration import HumanCalibrationStore
from .judgment_engine import JudgmentEngine
from .rule_engine import RuleEngine
from .pattern_registry import (
    canonical_ids,
    canonicalize_pattern_id,
    get_pattern,
    load_registry,
    variant_for_pattern,
)
from .schemas import (
    DetectedSignal,
    Direction,
    PatternKind,
    Resolution,
    SignalState,
    Timeframe,
)
from .signal_lifecycle import SignalLifecycle
from .timeframe_policy import (
    TimeframeProfile,
    build_display_view,
    coerce_timeframe,
    profile_for,
)

__all__ = [
    "CaseKnowledgeBase",
    "CaseMatch",
    "ChartPatternDetector",
    "CompiledCase",
    "ConflictResolver",
    "ContextInterpreter",
    "filter_contextual_candlestick_signals",
    "DetectedSignal",
    "Direction",
    "EvidenceExtractor",
    "FeedbackStore",
    "HumanCalibrationStore",
    "JudgmentEngine",
    "PatternKind",
    "QuantitativeGate",
    "Resolution",
    "RuleEngine",
    "SignalLifecycle",
    "SignalState",
    "Timeframe",
    "TimeframeProfile",
    "build_display_view",
    "canonical_ids",
    "canonicalize_pattern_id",
    "coerce_timeframe",
    "from_v7_events",
    "get_pattern",
    "load_registry",
    "plot_pattern_geometry",
    "profile_for",
    "variant_for_pattern",
]
