"""Human calibration overrides for the FAE judgment layer.

The FAE rules remain the machine baseline, but a reviewed human correction is
an explicit higher-priority decision.  Records are append-only JSON Lines and
are matched by asset/symbol, timeframe, and (optionally) a bar span.  A human
record never silently rewrites the detector or conflict table; it only
overrides the result for the matching observation.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .pattern_registry import canonicalize_pattern_id
from .schemas import DetectedSignal, Resolution, SignalState


def _normalise_asset_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return text.zfill(3)
    return text


class HumanCalibrationStore:
    """Load and apply explicit human corrections without changing FAE rules."""

    def __init__(self, path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parent
        self.path = Path(
            path or root / "cases" / "human_calibration_cases.jsonl"
        )

    def load(self) -> list[dict[str, Any]]:
        """Return valid calibration records, skipping schema/template rows."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid human calibration JSON at {self.path}:{line_number}"
                ) from exc
            if not isinstance(value, dict) or value.get("record_type") == "schema":
                continue
            records.append(value)
        return records

    @staticmethod
    def _matches_scope(
        record: dict[str, Any],
        *,
        context: dict[str, Any],
        timeframe: str,
        signals: list[DetectedSignal],
    ) -> bool:
        record_timeframe = record.get("timeframe", "")
        if isinstance(record_timeframe, (list, tuple, set)):
            if timeframe not in {str(item) for item in record_timeframe}:
                return False
        elif record_timeframe and str(record_timeframe) != timeframe:
            return False
        record_asset = record.get("asset_id")
        if record_asset not in (None, ""):
            if _normalise_asset_id(record_asset) != _normalise_asset_id(
                context.get("asset_id")
            ):
                return False
        record_symbol = record.get("symbol")
        if record_symbol not in (None, ""):
            if str(record_symbol).strip().lower() != str(
                context.get("symbol", "")
            ).strip().lower():
                return False
        scope = record.get("scope") if isinstance(record.get("scope"), dict) else record
        start = scope.get("start")
        end = scope.get("end")
        if start is not None or end is not None:
            start_value = int(start if start is not None else end)
            end_value = int(end if end is not None else start)
            if not any(
                signal.start <= end_value and signal.end >= start_value
                for signal in signals
            ):
                return False
        return True

    def matching(
        self,
        *,
        context: dict[str, Any],
        timeframe: str,
        signals: list[DetectedSignal],
    ) -> list[dict[str, Any]]:
        """Return records applicable to one evaluated asset/timeframe."""
        records = [
            record
            for record in self.load()
            if self._matches_scope(
                record,
                context=context,
                timeframe=timeframe,
                signals=signals,
            )
        ]
        return sorted(records, key=lambda item: str(item.get("timestamp", "")))

    def apply(
        self,
        signals: list[DetectedSignal],
        resolution: Resolution,
        *,
        context: dict[str, Any],
        timeframe: str,
    ) -> tuple[list[DetectedSignal], Resolution, dict[str, Any]]:
        """Apply the latest matching human decision as the highest priority."""
        records = self.matching(
            context=context,
            timeframe=timeframe,
            signals=signals,
        )
        if not records:
            return signals, resolution, {
                "matched": False,
                "policy": "human_calibration_overrides_machine_only_when_record_matches",
                "path": str(self.path),
            }

        current_signals = [deepcopy(signal) for signal in signals]
        current_resolution = deepcopy(resolution)
        applied_ids: list[str] = []
        unmatched: list[str] = []
        for record in records:
            record_id = str(record.get("case_id") or record.get("calibration_id") or "manual")
            winner_name = record.get("winner_pattern_id") or record.get("manual_primary_pattern_id")
            winner_id = (
                canonicalize_pattern_id(str(winner_name)) if winner_name else None
            )
            remove_ids = {
                canonicalize_pattern_id(str(item))
                for item in record.get("remove_pattern_ids", record.get("manual_remove_pattern_ids", []))
            }
            keep_ids = {
                canonicalize_pattern_id(str(item))
                for item in record.get("keep_pattern_ids", record.get("manual_keep_pattern_ids", []))
            }
            if winner_id is None and record.get("manual_state") == "no_pattern":
                for signal in current_signals:
                    signal.state = SignalState.INVALIDATED
                    signal.confirmed = False
                    signal.metadata["human_override"] = True
                    signal.metadata["human_calibration_id"] = record_id
                current_resolution = Resolution(
                    winner=None,
                    losers=current_signals,
                    final_state=SignalState.INVALIDATED,
                    confidence=1.0,
                    applied_rules=[*current_resolution.applied_rules, f"human:{record_id}"],
                    reasons=[*current_resolution.reasons, "人工校准明确判定当前窗口无正式形态"],
                    invalidation_conditions=current_resolution.invalidation_conditions,
                )
                applied_ids.append(record_id)
                continue

            winner = next(
                (
                    signal
                    for signal in current_signals
                    if canonicalize_pattern_id(signal.pattern) == winner_id
                    and signal.state not in {SignalState.INVALIDATED, SignalState.EXPIRED}
                ),
                None,
            )
            if winner is None and winner_id:
                unmatched.append(record_id)
                continue

            if winner is not None:
                winner.state = SignalState.CONFIRMED
                winner.confirmed = True
                winner.confidence = 1.0
                winner.metadata["human_override"] = True
                winner.metadata["human_calibration_id"] = record_id
                winner.metadata["human_override_reason"] = record.get("reason", "")
                losers: list[DetectedSignal] = []
                for signal in current_signals:
                    if signal is winner:
                        continue
                    pattern_id = canonicalize_pattern_id(signal.pattern)
                    if pattern_id in remove_ids or (keep_ids and pattern_id not in keep_ids):
                        signal.state = SignalState.INVALIDATED
                    signal.confirmed = False
                    signal.metadata["human_override_suppressed_by"] = record_id
                    losers.append(signal)
                current_resolution = Resolution(
                    winner=deepcopy(winner),
                    losers=losers,
                    final_state=SignalState.CONFIRMED,
                    confidence=1.0,
                    applied_rules=[*current_resolution.applied_rules, f"human:{record_id}"],
                    reasons=[
                        *current_resolution.reasons,
                        f"人工校准优先于 FAE 自动裁决：{record.get('reason', '未填写原因')}",
                    ],
                    invalidation_conditions=current_resolution.invalidation_conditions,
                )
                applied_ids.append(record_id)

        return current_signals, current_resolution, {
            "matched": bool(records),
            "applied": applied_ids,
            "unmatched": unmatched,
            "record_count": len(records),
            "path": str(self.path),
            "policy": "人工校准优先于 FAE 自动裁决，不修改核心检测器和专家规则",
        }

    def append(self, record: dict[str, Any]) -> None:
        """Append one validated-by-convention human record."""
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
        payload.setdefault("schema_version", 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["HumanCalibrationStore"]
