"""Append-only user feedback storage and conservative preference statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class FeedbackStore:
    """Record user overrides without silently rewriting expert rules."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        system_decision: str,
        user_decision: str,
        rule_ids: list[str],
        context: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        """Append one auditable judgment comparison as JSON Lines."""
        item = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "system_decision": system_decision,
            "user_decision": user_decision,
            "accepted": system_decision == user_decision,
            "rule_ids": rule_ids,
            "context": context or {},
            "note": note,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        """Load valid feedback rows, skipping malformed lines."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def statistics(self) -> dict[str, Any]:
        """Summarize acceptance and rejection rates by rule."""
        rows = self.load()
        totals: Counter[str] = Counter()
        accepted: Counter[str] = Counter()
        decisions: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            for rule_id in row.get("rule_ids", []):
                totals[rule_id] += 1
                if row.get("accepted"):
                    accepted[rule_id] += 1
                decisions[rule_id][row.get("user_decision", "")] += 1
        return {
            "feedback_count": len(rows),
            "by_rule": {
                rule_id: {
                    "count": count,
                    "acceptance_rate": accepted[rule_id] / count,
                    "user_decisions": dict(decisions[rule_id]),
                    "review_needed": count >= 10 and accepted[rule_id] / count < 0.6,
                }
                for rule_id, count in totals.items()
            },
            "policy": "仅建议人工复核阈值，不自动改写核心专家规则",
        }
