"""State-machine management for FAE signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .rule_engine import compare
from .schemas import DetectedSignal, RuleMatch, SignalState

_ALLOWED = {
    SignalState.DETECTED: {
        SignalState.CANDIDATE,
        SignalState.CONFIRMED,
        SignalState.DEGRADED,
        SignalState.INVALIDATED,
        SignalState.EXPIRED,
    },
    SignalState.CANDIDATE: {
        SignalState.CONFIRMED,
        SignalState.DEGRADED,
        SignalState.INVALIDATED,
        SignalState.EXPIRED,
    },
    SignalState.CONFIRMED: {
        SignalState.DEGRADED,
        SignalState.INVALIDATED,
        SignalState.EXPIRED,
    },
    SignalState.DEGRADED: {
        SignalState.CONFIRMED,
        SignalState.INVALIDATED,
        SignalState.EXPIRED,
    },
    SignalState.INVALIDATED: set(),
    SignalState.EXPIRED: set(),
}


class SignalLifecycle:
    """Apply rule state machines and prevent impossible state transitions."""

    @staticmethod
    def transition(signal: DetectedSignal, target: SignalState) -> DetectedSignal:
        """Return a copied signal in the target state after validation."""
        if target == signal.state:
            return deepcopy(signal)
        if target not in _ALLOWED[signal.state]:
            raise ValueError(
                f"invalid transition {signal.state.value} -> {target.value}"
            )
        result = deepcopy(signal)
        result.state = target
        result.confirmed = target == SignalState.CONFIRMED
        return result

    def update(
        self,
        signal: DetectedSignal,
        evidence: Mapping[str, Any],
        matched_rules: Sequence[RuleMatch] = (),
    ) -> tuple[DetectedSignal, list[str]]:
        """Apply the first matching transition across prioritized rules."""
        current = deepcopy(signal)
        reasons: list[str] = []
        for match in matched_rules:
            machine = match.rule.get("state_machine", {})
            for transition in machine.get("transitions", []):
                if transition.get("from") != current.state.value:
                    continue
                conditions = transition.get("when", {})
                if all(
                    compare(evidence.get(key), value)
                    for key, value in conditions.items()
                ):
                    target = SignalState(transition["to"])
                    current = self.transition(current, target)
                    reasons.append(
                        f"{match.rule_id}: {transition['from']} -> {transition['to']}"
                    )
                    return current, reasons
        if current.state == SignalState.DETECTED:
            current = self.transition(current, SignalState.CANDIDATE)
            reasons.append("无否决证据，进入候选状态")
        return current, reasons
