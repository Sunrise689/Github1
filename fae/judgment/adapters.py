"""Adapters from the existing pattern_core_v7 event format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import (
    DetectedSignal,
    Direction,
    PatternKind,
    SignalState,
    Timeframe,
)
from .pattern_registry import (
    canonicalize_pattern_id,
    registry_alias_map,
    variant_for_pattern,
)

_ALIASES = registry_alias_map()

def _canonical_name(event: Mapping[str, Any], original_name: str) -> str:
    base_name = original_name.split("·", 1)[0].strip()
    if str(event.get("category", "")).lower() == "gap":
        gap_kind = str(event.get("gap_kind", "")).strip()
        if gap_kind:
            return canonicalize_pattern_id(gap_kind)
        return (
            "rising_gap"
            if str(event.get("gap_direction", "")).lower() == "up"
            else "falling_gap"
        )
    return canonicalize_pattern_id(base_name)


def from_v7_events(
    events: Sequence[Mapping[str, Any]],
    *,
    timeframe: Timeframe = Timeframe.DAILY,
) -> list[DetectedSignal]:
    """Convert ``pattern_core_v7.detect_all`` results into judgment signals."""
    directions = {
        "bull": Direction.BULLISH,
        "bullish": Direction.BULLISH,
        "bear": Direction.BEARISH,
        "bearish": Direction.BEARISH,
        "neutral": Direction.NEUTRAL,
    }
    kinds = {
        "simple": PatternKind.SIMPLE,
        "composite": PatternKind.COMBINATION,
        "combination": PatternKind.COMBINATION,
        "trend": PatternKind.TREND,
        "gap": PatternKind.COMBINATION,
    }
    converted: list[DetectedSignal] = []
    for event in events:
        original_name = str(event.get("name", "")).strip()
        if not original_name:
            continue
        pattern_id = _canonical_name(event, original_name)
        canonical_variant = variant_for_pattern(original_name) or ""
        raw_direction = directions.get(
            str(event.get("direction", "neutral")).lower(),
            Direction.NEUTRAL,
        )
        # The candles themselves are bullish (close > open), but the
        # canonical pattern is a bearish continuation/reversal warning: its
        # defining feature is that the price centre keeps moving down in an
        # already declining background.  Keep both facts explicit so a
        # renderer does not mistake the red candle bodies for a bullish FAE
        # conclusion.
        semantic_direction = (
            Direction.BEARISH
            if pattern_id == "three_reverse_bullish"
            else raw_direction
        )
        converted.append(
            DetectedSignal(
                pattern=pattern_id,
                direction=semantic_direction,
                confidence=float(event.get("confidence", 0.5)),
                start=int(event.get("start", event.get("idx", 0))),
                end=int(event.get("end", event.get("idx", 0))),
                kind=kinds.get(
                    str(event.get("category", "simple")).lower(),
                    PatternKind.SIMPLE,
                ),
                timeframe=timeframe,
                state=SignalState.DETECTED,
                metadata={
                    "original_name": original_name,
                    "source_labels": [original_name],
                    "canonical_variant": canonical_variant,
                    "display_name_override": original_name if canonical_variant else "",
                    "source": "pattern_core_v7.detect_all",
                    "note": str(event.get("note", "")),
                    "category": str(event.get("category", "simple")),
                    "gap_kind": str(event.get("gap_kind", "")),
                    "gap_direction": str(event.get("gap_direction", "")),
                    "gap_type": str(event.get("gap_type", "")),
                    "gap_top": event.get("gap_top"),
                    "gap_bottom": event.get("gap_bottom"),
                    "gap_filled": bool(event.get("gap_filled", event.get("filled", False))),
                    "gap_fill_bars": event.get("gap_fill_bars", event.get("fill_bars")),
                    "filled": bool(event.get("filled", False)),
                    "candle_polarity": (
                        "bullish" if pattern_id == "three_reverse_bullish" else raw_direction.value
                    ),
                    "semantic_bias": (
                        "bearish" if pattern_id == "three_reverse_bullish" else semantic_direction.value
                    ),
                },
            )
        )

    # 同一检测窗口可能同时被 detect_three_red 和
    # detect_white_soldiers_variants 命中。两个名称现在共享
    # red_three_soldiers，因此只合并完全相同区间，避免重复计数；
    # 仍保留强势变体信息供前端展示。
    deduplicated: dict[tuple[str, int, int], DetectedSignal] = {}
    for signal in converted:
        key = (signal.pattern, signal.start, signal.end)
        previous = deduplicated.get(key)
        if previous is None:
            deduplicated[key] = signal
            continue
        labels = list(previous.metadata.get("source_labels", []))
        for label in signal.metadata.get("source_labels", []):
            if label not in labels:
                labels.append(label)
        previous.metadata["source_labels"] = labels
        if signal.metadata.get("canonical_variant"):
            previous.metadata["canonical_variant"] = signal.metadata[
                "canonical_variant"
            ]
        previous.confidence = max(previous.confidence, signal.confidence)

    return list(deduplicated.values())
