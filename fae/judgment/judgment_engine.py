"""End-to-end orchestration for the FAE expert judgment layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pandas as pd

from .case_knowledge import CaseKnowledgeBase
from .chart_patterns import ChartPatternDetector
from .conflict_resolver import ConflictResolver
from .context_validators import filter_contextual_candlestick_signals
from .context_interpreter import ContextInterpreter
from .evidence_extractor import EvidenceExtractor, QuantitativeGate
from .human_calibration import HumanCalibrationStore
from .pattern_registry import canonicalize_pattern_id
from .rule_engine import RuleEngine
from .schemas import DetectedSignal, Timeframe
from .signal_lifecycle import SignalLifecycle
from .timeframe_policy import build_display_view, coerce_timeframe, profile_for


class JudgmentEngine:
    """Combine quantitative evidence, sparse charts, rules, and resolution."""

    def __init__(
        self,
        *,
        rule_engine: RuleEngine | None = None,
        chart_detector: ChartPatternDetector | None = None,
        case_knowledge: CaseKnowledgeBase | None = None,
        human_calibration: HumanCalibrationStore | None = None,
    ) -> None:
        self.rule_engine = rule_engine or RuleEngine()
        self.chart_detector = chart_detector or ChartPatternDetector()
        self.extractor = EvidenceExtractor()
        self.quantitative_gate = QuantitativeGate()
        self.lifecycle = SignalLifecycle()
        self.interpreter = ContextInterpreter()
        self.resolver = ConflictResolver(self.rule_engine.priority_matrix)
        self.case_knowledge = case_knowledge or CaseKnowledgeBase()
        self.human_calibration = human_calibration or HumanCalibrationStore()

    def evaluate(
        self,
        data: pd.DataFrame,
        candlestick_signals: Sequence[DetectedSignal] = (),
        *,
        reference_levels: Mapping[str, float] | None = None,
        context: Mapping[str, Any] | None = None,
        timeframe: Timeframe | str | None = None,
        display_mode: str = "adaptive",
        enabled_kinds: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run the complete offline judgment pipeline on one OHLCV window.

        ``timeframe`` is optional for backward compatibility.  When omitted,
        the first supplied candlestick signal or the configured chart detector
        determines it.  Callers evaluating monthly/weekly/daily windows should
        pass it explicitly so the chart detector and display policy use the
        same structural scale.
        """
        inferred_timeframe = (
            timeframe
            if timeframe is not None
            else (
                candlestick_signals[0].timeframe
                if candlestick_signals
                else self.chart_detector.timeframe
            )
        )
        resolved_timeframe = coerce_timeframe(inferred_timeframe)
        chart_signals = self._detect_chart_signals(data, resolved_timeframe)
        aligned_candlestick_signals = []
        for signal in candlestick_signals:
            aligned = deepcopy(signal)
            if aligned.timeframe != resolved_timeframe:
                aligned.metadata.setdefault("source_timeframe", aligned.timeframe.value)
                aligned.timeframe = resolved_timeframe
            aligned_candlestick_signals.append(aligned)
        input_signal_count = len(aligned_candlestick_signals) + len(chart_signals)
        calibration_context = {**dict(context or {}), "timeframe": resolved_timeframe.value}
        preflight_calibrations = self.human_calibration.matching(
            context=calibration_context,
            timeframe=resolved_timeframe.value,
            signals=[*aligned_candlestick_signals, *chart_signals],
        )
        human_preapproved = {
            canonicalize_pattern_id(str(record.get("winner_pattern_id")))
            for record in preflight_calibrations
            if record.get("winner_pattern_id")
        }
        validated_candlestick_signals, context_rejections = (
            filter_contextual_candlestick_signals(data, aligned_candlestick_signals)
        )
        validated_keys = {
            (canonicalize_pattern_id(signal.pattern), signal.start, signal.end)
            for signal in validated_candlestick_signals
        }
        # A reviewed human correction may deliberately preserve a candidate
        # that the generic contextual veto would otherwise reject.  This is a
        # narrow, record-scoped bypass, not a global weakening of the gate.
        for signal in aligned_candlestick_signals:
            key = (canonicalize_pattern_id(signal.pattern), signal.start, signal.end)
            if canonicalize_pattern_id(signal.pattern) in human_preapproved and key not in validated_keys:
                signal.metadata["human_preapproved"] = True
                validated_candlestick_signals.append(signal)
                validated_keys.add(key)
        raw_signals = [*validated_candlestick_signals, *chart_signals]
        evidence = self.extractor.extract(
            data,
            reference_levels=reference_levels,
            detected_patterns=[signal.pattern for signal in raw_signals],
            signal_windows=[
                {
                    "pattern": signal.pattern,
                    "start": signal.start,
                    "end": signal.end,
                    "breakout_level": signal.metadata.get("breakout_level"),
                    "confirmation_start": signal.metadata.get("confirmation_start"),
                    "timeframe": resolved_timeframe.value,
                }
                for signal in raw_signals
            ],
        )
        evidence.update(
            {
                "breakout_confirmed": any(signal.confirmed for signal in chart_signals),
                "false_break": any(
                    bool(signal.metadata.get("false_break")) for signal in chart_signals
                ),
                "gap_pair_confirmed": any(
                    bool(signal.metadata.get("gap_pair_confirmed"))
                    for signal in chart_signals
                ),
            }
        )

        # FAE 三层架构的第一道门：量化层先淘汰不可能或明显违背
        # 基础统计约束的候选；案例库只能看到通过这一层的信号。
        signals: list[DetectedSignal] = []
        quantitative_rejections: list[dict[str, Any]] = [*context_rejections]
        for signal in raw_signals:
            accepted, reasons = self.quantitative_gate.validate_signal(signal, evidence)
            force_human = canonicalize_pattern_id(signal.pattern) in human_preapproved
            if accepted or force_human:
                if force_human and not accepted:
                    signal.metadata["human_preapproved"] = True
                signals.append(signal)
            else:
                quantitative_rejections.append(
                    {"pattern": signal.pattern, "reasons": reasons}
                )

        accepted_chart_signals = [
            signal for signal in signals if signal.kind.value == "chart"
        ]
        evidence.update(
            {
                "breakout_confirmed": any(
                    signal.confirmed for signal in accepted_chart_signals
                ),
                "false_break": any(
                    bool(signal.metadata.get("false_break"))
                    for signal in accepted_chart_signals
                ),
                "gap_pair_confirmed": any(
                    bool(signal.metadata.get("gap_pair_confirmed"))
                    for signal in accepted_chart_signals
                ),
            }
        )

        merged_context = {
            **evidence,
            "timeframe": resolved_timeframe.value,
            "timeframe_profile": profile_for(resolved_timeframe).to_dict(),
            **(context or {}),
        }
        matches = self.rule_engine.match_rules(signals, evidence, merged_context)
        updated: list[DetectedSignal] = []
        lifecycle_reasons: list[str] = []
        for signal in signals:
            applicable = [
                match
                for match in matches
                if signal.pattern in match.rule.get("applies_to", [])
            ]
            local_evidence = self._signal_local_evidence(evidence, signal)
            signal_evidence = {
                **merged_context,
                **local_evidence,
                "breakout_confirmed": bool(
                    local_evidence.get("breakout_confirmed", signal.confirmed)
                ),
                "false_break": bool(signal.metadata.get("false_break")),
                "gap_pair_confirmed": bool(signal.metadata.get("gap_pair_confirmed")),
            }
            new_signal, reasons = self.lifecycle.update(
                signal, signal_evidence, applicable
            )
            if local_evidence:
                # Expose the confirmation contract on each serialized signal;
                # the frontend should not need to reconstruct it from the
                # aggregate evidence map.
                for field in (
                    "breakout_level",
                    "breakout_confirmed",
                    "breakout_confirmation_mode",
                    "confirmation_streak_bars",
                    "confirmation_index",
                ):
                    if field in local_evidence:
                        new_signal.metadata[field] = local_evidence[field]
            updated.append(new_signal)
            lifecycle_reasons.extend(reasons)
        # Signals passed here have already passed the quantitative gate above;
        # avoid running the same veto checks a second time inside the resolver.
        resolution = self.resolver.resolve(
            updated,
            matches,
            evidence,
            prevalidated=True,
        )
        case_matches = self.case_knowledge.search(updated, evidence, merged_context)
        case_consensus = self.case_knowledge.consensus(case_matches)
        resolution = self.case_knowledge.apply_consensus(
            resolution,
            case_consensus,
        )
        merged_context_for_calibration = {
            **merged_context,
            **dict(context or {}),
        }
        updated, resolution, human_calibration = self.human_calibration.apply(
            updated,
            resolution,
            context=merged_context_for_calibration,
            timeframe=resolved_timeframe.value,
        )
        display_view = build_display_view(
            updated,
            resolution,
            timeframe=resolved_timeframe,
            display_mode=display_mode,
            enabled_kinds=enabled_kinds,
        )
        interpretations = [
            self.interpreter.interpret(
                signal,
                trend=str(merged_context.get("trend", "range")),
                position_pct=float(merged_context.get("position_pct", 0.5)),
                ma_state=str(merged_context.get("ma_state", "unknown")),
            )
            for signal in updated
        ]
        return {
            "timeframe": resolved_timeframe.value,
            "timeframe_policy": profile_for(resolved_timeframe).to_dict(),
            "evidence": evidence,
            "quantitative_filter": {
                "input_count": input_signal_count,
                "accepted_count": len(signals),
                "rejected_count": len(quantitative_rejections),
                "rejections": quantitative_rejections,
            },
            "signals": display_view["all"],
            "display": display_view,
            "matched_rules": [
                {
                    "rule_id": match.rule_id,
                    "priority": match.priority,
                    "score": match.score,
                    "reasons": match.reasons,
                }
                for match in matches
            ],
            "lifecycle_reasons": lifecycle_reasons,
            "interpretations": interpretations,
            "case_matches": [match.to_dict() for match in case_matches],
            "case_consensus": case_consensus,
            "human_calibration": human_calibration,
            "resolution": resolution.to_dict(),
        }

    def _detect_chart_signals(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe,
    ) -> list[DetectedSignal]:
        """Run the configured chart detector at the requested timeframe."""
        if self.chart_detector.timeframe == timeframe:
            return self.chart_detector.detect(data)
        # The detector stores timeframe as immutable signal metadata.  Create a
        # matching detector rather than mutating a shared instance, which keeps
        # one JudgmentEngine safe to reuse for M/W/D previews in one process.
        detector = ChartPatternDetector(
            max_patterns_per_100=self.chart_detector.gate.max_chart_patterns_per_100,
            min_window=self.chart_detector.min_window,
            max_window=self.chart_detector.max_window,
            timeframe=timeframe,
        )
        return detector.detect(data)

    @staticmethod
    def _signal_local_evidence(
        evidence: Mapping[str, Any],
        signal: DetectedSignal,
    ) -> Mapping[str, Any]:
        """Look up extractor evidence for one signal's positional window."""
        payloads = evidence.get("signal_evidence", {})
        if not isinstance(payloads, Mapping):
            return {}
        key = f"{signal.pattern}:{signal.start}:{signal.end}"
        payload = payloads.get(key)
        return payload if isinstance(payload, Mapping) else {}
