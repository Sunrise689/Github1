"""Resolve simultaneous FAE signals with rules, timeframe, and confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .evidence_extractor import QuantitativeGate
from .pattern_registry import canonicalize_pattern_id
from .rule_engine import compare
from .schemas import (
    DetectedSignal,
    PatternKind,
    Resolution,
    RuleMatch,
    SignalState,
)


class ConflictResolver:
    """Apply quantitative vetoes before expert and structural priorities."""

    def __init__(
        self,
        priority_matrix: Mapping[str, Any],
        quantitative_gate: QuantitativeGate | None = None,
        conflict_rules_path: str | Path | None = None,
    ) -> None:
        self.matrix = dict(priority_matrix)
        self.gate = quantitative_gate or QuantitativeGate()
        root = Path(__file__).resolve().parent
        self.conflict_rules_path = Path(
            conflict_rules_path or root / "conflict_rules.json"
        )
        self.conflict_rules = self._load_conflict_rules()

    def _load_conflict_rules(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.conflict_rules_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"FAE conflict rule file not found: {self.conflict_rules_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {self.conflict_rules_path}: {exc}"
            ) from exc
        rules = payload.get("rules", []) if isinstance(payload, Mapping) else payload
        if not isinstance(rules, list):
            raise TypeError("conflict_rules.json must contain a rules list")
        seen: set[str] = set()
        for rule in rules:
            required = {
                "rule_id",
                "patterns",
                "condition",
                "winner_rule",
                "reject_reason",
                "requires_confirmation",
            }
            missing = required - set(rule)
            if missing:
                raise ValueError(
                    f"conflict rule missing fields {sorted(missing)}: {rule}"
                )
            rule_id = str(rule["rule_id"])
            if rule_id in seen:
                raise ValueError(f"duplicate conflict rule_id: {rule_id}")
            seen.add(rule_id)
            rule["patterns"] = [
                canonicalize_pattern_id(str(item)) for item in rule["patterns"]
            ]
            if rule.get("winner_pattern"):
                rule["winner_pattern"] = canonicalize_pattern_id(
                    str(rule["winner_pattern"])
                )
        return rules

    def _base_score(self, signal: DetectedSignal) -> float:
        kind = self.matrix.get("kind_weight", {})
        timeframe = self.matrix.get("timeframe_weight", {})
        state = self.matrix.get("state_weight", {})
        score = signal.confidence * 100.0
        score += float(kind.get(signal.kind.value, 0))
        score += float(timeframe.get(signal.timeframe.value, 0))
        score += float(state.get(signal.state.value, 0))
        if signal.confirmed:
            score += float(self.matrix.get("confirmed_bonus", 15))
        return score

    @staticmethod
    def _same_window(
        left: DetectedSignal,
        right: DetectedSignal,
        max_separation: int = 1,
    ) -> bool:
        if max(left.start, right.start) <= min(left.end, right.end):
            return True
        distance = min(abs(left.start - right.end), abs(right.start - left.end))
        return distance <= max_separation

    @staticmethod
    def _local_evidence(
        signal: DetectedSignal,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return extractor evidence scoped to one signal, when available."""
        payloads = evidence.get("signal_evidence", {})
        if not isinstance(payloads, Mapping):
            return {}
        canonical = canonicalize_pattern_id(signal.pattern)
        for pattern in (signal.pattern, canonical):
            key = f"{pattern}:{signal.start}:{signal.end}"
            payload = payloads.get(key)
            if isinstance(payload, Mapping):
                return payload
        return {}

    @staticmethod
    def _condition_matches(
        rule: Mapping[str, Any],
        left: DetectedSignal,
        right: DetectedSignal,
        evidence: Mapping[str, Any],
    ) -> bool:
        condition = rule.get("condition", {})
        if not isinstance(condition, Mapping):
            return True
        if condition.get("same_window") and not ConflictResolver._same_window(
            left,
            right,
            int(condition.get("max_separation_bars", 1)),
        ):
            return False
        nested = condition.get("evidence", {})
        if isinstance(nested, Mapping):
            left_local = ConflictResolver._local_evidence(left, evidence)
            right_local = ConflictResolver._local_evidence(right, evidence)
            for field, expected in nested.items():
                # Prefer the signal's own span.  Pair rules may use either
                # side's local value; global evidence remains a safe fallback
                # for direct resolver callers and backwards compatibility.
                value = left_local.get(field)
                if value is None:
                    value = right_local.get(field)
                if value is None:
                    value = evidence.get(field)
                if not compare(value, expected):
                    return False
        return True

    @staticmethod
    def _trend_winner(
        rule: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> str | None:
        trend = str(evidence.get(str(rule.get("context_field", "trend")), "")).lower()
        values = rule.get("context_values", {})
        if isinstance(values, Mapping):
            value = values.get(trend)
            if value:
                return canonicalize_pattern_id(str(value))
        return None

    def _apply_conflict_rules(
        self,
        valid: list[DetectedSignal],
        scores: dict[int, float],
        evidence: Mapping[str, Any],
    ) -> tuple[set[int], list[str], list[str], list[str], bool]:
        """Apply declarative pair/global conflicts once per resolution pass."""

        suppressed: set[int] = set()
        applied: list[str] = []
        reasons: list[str] = []
        invalidation: list[str] = []
        force_degraded = False
        for rule in self.conflict_rules:
            patterns = set(rule.get("patterns", []))
            winner_rule = str(rule.get("winner_rule", ""))
            if not patterns:
                matched = False
                if winner_rule == "confirmed_chart_over_unconfirmed_candlestick":
                    for chart in valid:
                        if chart.kind != PatternKind.CHART or not chart.confirmed:
                            continue
                        for other in valid:
                            if (
                                other.direction != chart.direction
                                and other.kind in {PatternKind.SIMPLE, PatternKind.COMBINATION}
                                and not other.confirmed
                                and self._same_window(chart, other)
                            ):
                                scores[id(chart)] += 20
                                scores[id(other)] -= 20
                                matched = True
                elif winner_rule == "higher_timeframe_chart":
                    for chart in valid:
                        if chart.kind != PatternKind.CHART or not chart.confirmed:
                            continue
                        chart_weight = float(
                            self.matrix.get("timeframe_weight", {}).get(
                                chart.timeframe.value, 0
                            )
                        )
                        for other in valid:
                            if (
                                other.direction == chart.direction
                                or other.kind not in {PatternKind.SIMPLE, PatternKind.COMBINATION}
                                or not self._same_window(chart, other)
                            ):
                                continue
                            other_weight = float(
                                self.matrix.get("timeframe_weight", {}).get(
                                    other.timeframe.value, 0
                                )
                            )
                            if chart_weight > other_weight:
                                scores[id(chart)] += 18
                                scores[id(other)] -= 18
                                matched = True
                if matched:
                    applied.append(str(rule["rule_id"]))
                    reasons.append(str(rule["reject_reason"]))
                continue

            if len(patterns) == 1:
                pairs = [
                    (signal, signal)
                    for signal in valid
                    if canonicalize_pattern_id(signal.pattern) in patterns
                    and self._condition_matches(rule, signal, signal, evidence)
                ]
            else:
                pairs = [
                    (left, right)
                    for index, left in enumerate(valid)
                    for right in valid[index + 1 :]
                    if canonicalize_pattern_id(left.pattern) in patterns
                    and canonicalize_pattern_id(right.pattern) in patterns
                    and self._condition_matches(rule, left, right, evidence)
                ]
            for left, right in pairs:
                left_id = canonicalize_pattern_id(left.pattern)
                right_id = canonicalize_pattern_id(right.pattern)
                preferred = rule.get("winner_pattern")
                if winner_rule == "contextual_trend":
                    preferred = self._trend_winner(rule, evidence)
                elif winner_rule == "confirmed_directional_pattern":
                    preferred = None
                    if left.confirmed and not right.confirmed:
                        preferred = left_id
                    elif right.confirmed and not left.confirmed:
                        preferred = right_id
                    else:
                        trend = str(evidence.get("trend", "")).lower()
                        aligned = [
                            item
                            for item in (left, right)
                            if (trend == "up" and item.direction.value == "bullish")
                            or (trend == "down" and item.direction.value == "bearish")
                        ]
                        if len(aligned) == 1:
                            preferred = canonicalize_pattern_id(aligned[0].pattern)
                elif winner_rule in {"quantitative_trend_strength", "same_structure_dedup"}:
                    preferred = left_id if scores[id(left)] >= scores[id(right)] else right_id

                if winner_rule == "degrade_opposite_reversals":
                    scores[id(left)] -= 16
                    scores[id(right)] -= 16
                    force_degraded = True
                    applied.append(str(rule["rule_id"]))
                    reasons.append(str(rule["reject_reason"]))
                    continue
                if winner_rule == "invalidate_pattern":
                    target = left
                    target.state = SignalState.INVALIDATED
                    target.metadata["conflict_suppressed"] = str(rule["rule_id"])
                    suppressed.add(id(target))
                    scores[id(target)] -= 100
                    applied.append(str(rule["rule_id"]))
                    reasons.append(str(rule["reject_reason"]))
                    invalidation.append(str(rule["reject_reason"]))
                    continue
                if preferred not in {left_id, right_id}:
                    continue
                preferred_signal = left if left_id == preferred else right
                other_signal = right if preferred_signal is left else left
                if rule.get("requires_confirmation") and not preferred_signal.confirmed:
                    continue
                bonus = max(10.0, float(rule.get("priority", 50)) * 0.25)
                scores[id(preferred_signal)] += bonus
                scores[id(other_signal)] -= bonus
                other_signal.metadata["conflict_suppressed_by"] = str(rule["rule_id"])
                suppressed.add(id(other_signal))
                applied.append(str(rule["rule_id"]))
                reasons.append(str(rule["reject_reason"]))
        return (
            suppressed,
            list(dict.fromkeys(applied)),
            reasons,
            invalidation,
            force_degraded,
        )

    def resolve(
        self,
        signals: Sequence[DetectedSignal],
        matched_rules: Sequence[RuleMatch],
        evidence: Mapping[str, Any],
        *,
        prevalidated: bool = False,
    ) -> Resolution:
        """Return an explainable winner after gate, rule, and hierarchy checks."""
        valid: list[DetectedSignal] = []
        reasons: list[str] = []
        if prevalidated:
            valid = [deepcopy(signal) for signal in signals]
        else:
            for signal in signals:
                accepted, rejected = self.gate.validate_signal(signal, evidence)
                if accepted:
                    valid.append(deepcopy(signal))
                else:
                    reasons.append(
                        f"{signal.pattern} 被量化门控否决: {'；'.join(rejected)}"
                    )
        if not valid:
            return Resolution(
                winner=None,
                losers=[],
                final_state=SignalState.INVALIDATED,
                confidence=0.0,
                reasons=reasons or ["没有有效信号"],
            )

        scores = {id(signal): self._base_score(signal) for signal in valid}
        applied: list[str] = []
        invalidation: list[str] = []
        for match in matched_rules:
            resolution = match.rule.get("resolution", {})
            winner_name = resolution.get("winner")
            loser_name = resolution.get("loser")
            adjustment = float(resolution.get("confidence_adjustment", 0.0)) * 100
            for signal in valid:
                if canonicalize_pattern_id(signal.pattern) == canonicalize_pattern_id(str(winner_name)) or (
                    match.rule_id.startswith("chart_")
                    and canonicalize_pattern_id(signal.pattern)
                    in {
                        canonicalize_pattern_id(str(item))
                        for item in match.rule.get("applies_to", [])
                    }
                ):
                    scores[id(signal)] += adjustment + match.priority * 0.2
                if loser_name and canonicalize_pattern_id(signal.pattern) == canonicalize_pattern_id(str(loser_name)):
                    scores[id(signal)] -= abs(adjustment) + match.priority * 0.1
            if winner_name or loser_name:
                applied.append(match.rule_id)
                reason = resolution.get("suppression_reason")
                if reason:
                    reasons.append(str(reason))
                description = match.rule.get("invalidation", {}).get("description")
                if description:
                    invalidation.append(str(description))

        # All explicit conflicts and global hierarchy rules are applied from
        # one declarative table. Measurement targets never enter this score.
        suppressed, conflict_applied, conflict_reasons, conflict_invalidation, force_degraded = self._apply_conflict_rules(
            valid, scores, evidence
        )
        applied.extend(conflict_applied)
        reasons.extend(conflict_reasons)
        invalidation.extend(conflict_invalidation)
        valid = [signal for signal in valid if signal.state != SignalState.INVALIDATED]
        if not valid:
            return Resolution(
                winner=None,
                losers=[],
                final_state=SignalState.INVALIDATED,
                confidence=0.0,
                applied_rules=list(dict.fromkeys(applied)),
                reasons=list(dict.fromkeys(reasons)) or ["冲突规则使全部信号失效"],
                invalidation_conditions=list(dict.fromkeys(invalidation)),
            )

        ordered = sorted(valid, key=lambda item: scores[id(item)], reverse=True)
        winner = ordered[0]
        winner_family = winner.metadata.get("family")
        losers = [
            item
            for item in ordered[1:]
            if id(item) in suppressed
            or item.direction != winner.direction
            or (
                winner_family
                and item.metadata.get("family")
                and item.metadata.get("family") == winner_family
            )
        ]
        confidence = max(0.0, min(1.0, scores[id(winner)] / 130.0))
        final_state = winner.state
        if force_degraded:
            final_state = SignalState.DEGRADED
            reasons.append("相反方向反转形态同时出现，按冲突规则降级等待确认")
        elif (
            any(item.direction != winner.direction for item in ordered[1:])
            and confidence < 0.58
        ):
            final_state = SignalState.DEGRADED
            reasons.append("相反方向证据接近，结论降级等待确认")
        return Resolution(
            winner=winner,
            losers=losers,
            final_state=final_state,
            confidence=confidence,
            applied_rules=applied,
            reasons=reasons,
            invalidation_conditions=list(dict.fromkeys(invalidation)),
        )
