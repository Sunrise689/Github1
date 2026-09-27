"""Load, validate, and match declarative FAE expert rules."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schemas import DetectedSignal, RuleMatch


def compare(value: Any, condition: Any) -> bool:
    """Evaluate a compact declarative condition."""
    if not isinstance(condition, Mapping):
        return value == condition
    for operator, expected in condition.items():
        if operator == "gte" and not (value is not None and value >= expected):
            return False
        if operator == "gt" and not (value is not None and value > expected):
            return False
        if operator == "lte" and not (value is not None and value <= expected):
            return False
        if operator == "lt" and not (value is not None and value < expected):
            return False
        if operator == "eq" and value != expected:
            return False
        if operator == "ne" and value == expected:
            return False
        if operator == "in" and value not in expected:
            return False
        if operator == "contains" and (
            not isinstance(value, (list, tuple, set, str)) or expected not in value
        ):
            return False
        if operator == "exists" and bool(value is not None) != bool(expected):
            return False
        if operator == "any_of" and not any(compare(value, item) for item in expected):
            return False
    return True


class RuleEngine:
    """Match objective signals and evidence to validated JSON rules."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        priority_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent
        self.rules_path = Path(rules_path or root / "expert_rules.json")
        self.priority_path = Path(priority_path or root / "priority_matrix.json")
        self.rules = self._load_rules()
        self.priority_matrix = self._load_json(self.priority_path)

    @staticmethod
    def _load_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"FAE rule file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    def _load_rules(self) -> list[dict[str, Any]]:
        payload = self._load_json(self.rules_path)
        rules = payload["rules"] if isinstance(payload, Mapping) else payload
        if not isinstance(rules, list):
            raise TypeError("expert_rules.json must contain a rules list")
        seen: set[str] = set()
        for rule in rules:
            required = {
                "rule_id",
                "name",
                "applies_to",
                "priority",
                "source",
                "version",
            }
            missing = required - set(rule)
            if missing:
                raise ValueError(f"rule missing fields {sorted(missing)}: {rule}")
            if rule["rule_id"] in seen:
                raise ValueError(f"duplicate rule_id: {rule['rule_id']}")
            seen.add(rule["rule_id"])
            if not 0 <= int(rule["priority"]) <= 100:
                raise ValueError(f"priority out of range: {rule['rule_id']}")
        return rules

    @staticmethod
    def _conditions_match(
        conditions: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for field, condition in conditions.items():
            value = values.get(field)
            if not compare(value, condition):
                return False, []
            reasons.append(f"{field}={value!r} 满足 {condition!r}")
        return True, reasons

    def match_rules(
        self,
        signals: Sequence[DetectedSignal],
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> list[RuleMatch]:
        """Return applicable rules sorted by priority and condition specificity."""
        names = {signal.pattern for signal in signals}
        context_values = {**evidence, **(context or {})}
        matched: list[RuleMatch] = []
        for rule in self.rules:
            applies = set(rule.get("applies_to", []))
            if applies and not (names & applies):
                continue
            ok_evidence, evidence_reasons = self._conditions_match(
                rule.get("required_evidence", {}), evidence
            )
            if not ok_evidence:
                continue
            ok_context, context_reasons = self._conditions_match(
                rule.get("context_filters", {}), context_values
            )
            if not ok_context:
                continue
            specificity = len(evidence_reasons) + len(context_reasons)
            matched.append(
                RuleMatch(
                    rule_id=rule["rule_id"],
                    priority=int(rule["priority"]),
                    score=min(1.0, 0.6 + specificity * 0.08),
                    rule=rule,
                    reasons=evidence_reasons + context_reasons,
                )
            )
        return sorted(
            matched,
            key=lambda item: (item.priority, item.score),
            reverse=True,
        )
