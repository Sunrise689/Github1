"""Timeframe-aware FAE presentation policy.

The quantitative and judgment layers should still inspect every signal that
is technically available.  This module adds the next, product-facing layer:
it describes which *kind* of signal deserves the foreground on each bar
timeframe, groups repeated detections, and preserves all de-emphasized or
conflicting signals for audit/detail views.

The current defaults follow the project decision made on 2026-08-16:

* monthly bars: simple and combination candles are the foreground; trend
  candles and chart structures remain available as supporting context;
* weekly bars: simple and combination/gap patterns are the foreground, with
  trend/chart structures available as secondary context;
* daily bars: combination/gap, trend and chart structures are the foreground;
  simple candles are supporting evidence rather than a label flood.

This is a display/role policy, not a truth filter.  A signal is never deleted
because its kind is not the preferred kind for the timeframe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .pattern_registry import canonicalize_pattern_id
from .schemas import DetectedSignal, PatternKind, SignalState, Timeframe


_TIMEFRAME_ALIASES = {
    "m": Timeframe.MONTHLY,
    "1m": Timeframe.MONTHLY,
    "month": Timeframe.MONTHLY,
    "monthly": Timeframe.MONTHLY,
    "月": Timeframe.MONTHLY,
    "月线": Timeframe.MONTHLY,
    "w": Timeframe.WEEKLY,
    "1w": Timeframe.WEEKLY,
    "week": Timeframe.WEEKLY,
    "weekly": Timeframe.WEEKLY,
    "周": Timeframe.WEEKLY,
    "周线": Timeframe.WEEKLY,
    "d": Timeframe.DAILY,
    "1d": Timeframe.DAILY,
    "day": Timeframe.DAILY,
    "daily": Timeframe.DAILY,
    "日": Timeframe.DAILY,
    "日线": Timeframe.DAILY,
    "intraday": Timeframe.INTRADAY,
    "分钟": Timeframe.INTRADAY,
    "分时": Timeframe.INTRADAY,
}


@dataclass(frozen=True)
class TimeframeProfile:
    """Foreground/supporting signal kinds for one timeframe."""

    timeframe: Timeframe
    primary_kinds: frozenset[PatternKind]
    supporting_kinds: frozenset[PatternKind]
    max_primary: int
    max_supporting: int
    rationale: str
    # ``primary_kinds`` is kept stable for older API consumers.  The
    # effective set can be widened by a product decision without breaking
    # callers that only inspect the legacy field.
    effective_primary_kinds: frozenset[PatternKind] | None = None
    # ``None`` means no mathematical hard cap; the rolling confidence gate
    # determines whether additional charts are credible enough to retain.
    max_formal_chart_patterns: int | None = None
    # Product-facing adaptive mode is intentionally small but not winner-only.
    # The value is a display budget, not a detector/decision cap: all signals
    # remain available in ``all`` and in the manual layer buckets.
    adaptive_default_limit: int = 5

    def base_role(self, kind: PatternKind) -> str:
        """Return the role before lifecycle/conflict information is applied."""
        primary_kinds = self.effective_primary_kinds or self.primary_kinds
        if kind in primary_kinds:
            return "primary_candidate"
        if kind in self.supporting_kinds:
            return "supporting"
        return "timeframe_deemphasized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe.value,
            "primary_kinds": sorted(item.value for item in self.primary_kinds),
            "supporting_kinds": sorted(item.value for item in self.supporting_kinds),
            "max_primary": self.max_primary,
            "max_supporting": self.max_supporting,
            "rationale": self.rationale,
            "effective_primary_kinds": sorted(
                item.value for item in (self.effective_primary_kinds or self.primary_kinds)
            ),
            "max_formal_chart_patterns": self.max_formal_chart_patterns,
            "adaptive_default_limit": self.adaptive_default_limit,
            "chart_selection_policy": {
                "hard_count_limit": self.max_formal_chart_patterns,
                "rolling_confidence_floors": ROLLING_CHART_CONFIDENCE_FLOORS,
                "after_third": "每增加一个候选，门槛继续上升至0.995，不以数量硬截断",
            },
        }


_PROFILES: dict[Timeframe, TimeframeProfile] = {
    Timeframe.MONTHLY: TimeframeProfile(
        timeframe=Timeframe.MONTHLY,
        primary_kinds=frozenset({PatternKind.COMBINATION}),
        supporting_kinds=frozenset({PatternKind.SIMPLE}),
        max_primary=3,
        max_supporting=5,
        rationale=(
            "月线默认突出简单K线与组合K线；趋势K线和技术图形不强行判定，"
            "保留为可审计候选。月线不设置正式技术图形标签。"
        ),
        effective_primary_kinds=frozenset({PatternKind.SIMPLE, PatternKind.COMBINATION}),
        max_formal_chart_patterns=0,
    ),
    Timeframe.WEEKLY: TimeframeProfile(
        timeframe=Timeframe.WEEKLY,
        primary_kinds=frozenset({PatternKind.SIMPLE, PatternKind.COMBINATION}),
        supporting_kinds=frozenset({PatternKind.TREND, PatternKind.CHART}),
        max_primary=4,
        max_supporting=6,
        rationale=(
            "周线默认同时看简单K线和组合K线；趋势与技术图形保留为中期结构"
            "辅助；第一个技术图形门槛相对宽松，后续候选按滚动门槛逐级收紧。"
        ),
        max_formal_chart_patterns=None,
    ),
    Timeframe.DAILY: TimeframeProfile(
        timeframe=Timeframe.DAILY,
        primary_kinds=frozenset(
            {PatternKind.COMBINATION, PatternKind.TREND, PatternKind.CHART}
        ),
        supporting_kinds=frozenset({PatternKind.SIMPLE}),
        max_primary=5,
        max_supporting=8,
        rationale=(
            "日线默认突出组合K线、趋势形态和技术图形；简单K线作为辅助证据，"
            "避免大量单根标签遮住主结构；技术图形数量由滚动置信度自然控制。"
        ),
        max_formal_chart_patterns=None,
    ),
    Timeframe.INTRADAY: TimeframeProfile(
        timeframe=Timeframe.INTRADAY,
        primary_kinds=frozenset(
            {PatternKind.SIMPLE, PatternKind.COMBINATION, PatternKind.TREND}
        ),
        supporting_kinds=frozenset({PatternKind.CHART}),
        max_primary=4,
        max_supporting=5,
        rationale="分时默认使用K线和短趋势结构，长周期技术图形仅作辅助。",
        max_formal_chart_patterns=None,
    ),
}


# Public, immutable product rule used by both the display layer and the chart
# detector.  ``None`` is intentional: weekly/daily charts do not have a hard
# count cap.  Each additional chart must pass a stricter rolling confidence
# threshold, so the quantity is self-limiting without claiming that “three”
# is a universal maximum.  A zero means monthly views do not promote a chart
# to a formal label under the current timeframe policy.
FORMAL_CHART_LIMITS: dict[Timeframe, int | None] = {
    timeframe: profile.max_formal_chart_patterns
    for timeframe, profile in _PROFILES.items()
}

ROLLING_CHART_CONFIDENCE_FLOORS = (0.65, 0.85, 0.93)


def coerce_timeframe(value: Timeframe | str | None) -> Timeframe:
    """Normalize API-friendly timeframe aliases to :class:`Timeframe`."""
    if isinstance(value, Timeframe):
        return value
    if value is None:
        return Timeframe.DAILY
    text = str(value).strip().lower()
    try:
        return Timeframe(text)
    except ValueError:
        try:
            return _TIMEFRAME_ALIASES[text]
        except KeyError as exc:
            supported = ", ".join(item.value for item in Timeframe)
            raise ValueError(
                f"unsupported timeframe {value!r}; use one of {supported}"
            ) from exc


def profile_for(timeframe: Timeframe | str | None) -> TimeframeProfile:
    """Return the immutable display profile for a timeframe."""
    return _PROFILES[coerce_timeframe(timeframe)]


def _signal_key(signal: DetectedSignal) -> tuple[str, str, int, int]:
    return (
        canonicalize_pattern_id(signal.pattern),
        signal.direction.value,
        int(signal.start),
        int(signal.end),
    )


def _state_rank(state: SignalState) -> int:
    return {
        SignalState.CONFIRMED: 5,
        SignalState.CANDIDATE: 3,
        SignalState.DETECTED: 2,
        SignalState.DEGRADED: 1,
        SignalState.INVALIDATED: -5,
        SignalState.EXPIRED: -6,
    }.get(state, 0)


def _same_group(left: DetectedSignal, right: DetectedSignal, gap_bars: int) -> bool:
    """Whether two same-pattern detections are one display group."""
    if canonicalize_pattern_id(left.pattern) != canonicalize_pattern_id(right.pattern):
        return False
    if left.direction != right.direction or left.timeframe != right.timeframe:
        return False
    return left.start <= right.end + gap_bars and right.start <= left.end + gap_bars


def _group_signals(
    signals: Sequence[DetectedSignal],
    *,
    gap_bars: int = 1,
) -> list[list[DetectedSignal]]:
    """Group overlapping/adjacent detections of the same canonical pattern."""
    ordered = sorted(
        signals,
        key=lambda item: (
            canonicalize_pattern_id(item.pattern),
            item.direction.value,
            item.timeframe.value,
            item.start,
            item.end,
        ),
    )
    groups: list[list[DetectedSignal]] = []
    for signal in ordered:
        if groups and _same_group(groups[-1][-1], signal, gap_bars):
            groups[-1].append(signal)
        else:
            groups.append([signal])
    return groups


def _serialize(signals: Iterable[DetectedSignal]) -> list[dict[str, Any]]:
    return [signal.to_dict() for signal in signals]


def build_display_view(
    signals: Sequence[DetectedSignal],
    resolution: Any | None,
    *,
    timeframe: Timeframe | str | None = None,
    merge_gap_bars: int = 1,
    display_mode: str = "adaptive",
    enabled_kinds: Sequence[str | PatternKind] | None = None,
) -> dict[str, Any]:
    """Annotate signals and produce primary/supporting/audit buckets.

    This function deliberately keeps every signal in ``all``.  The default
    frontend view can consume ``primary`` and ``supporting`` while an advanced
    view can consume ``candidates`` and ``suppressed``.  No signal is removed
    merely because a timeframe profile de-emphasizes its kind.
    """
    resolved = coerce_timeframe(timeframe)
    mode = str(display_mode or "adaptive").strip().lower()
    if mode not in {"adaptive", "manual", "audit"}:
        raise ValueError("display_mode must be adaptive, manual, or audit")
    profile = profile_for(resolved)
    items = [deepcopy(signal) for signal in signals]
    winner_key = _signal_key(resolution.winner) if resolution and resolution.winner else None
    loser_keys = {
        _signal_key(signal)
        for signal in (resolution.losers if resolution else [])
    }
    max_end = max((signal.end for signal in items), default=0)

    groups = _group_signals(items, gap_bars=merge_gap_bars)
    representative_keys: set[tuple[str, str, int, int]] = set()
    group_payload: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        # Prefer the resolved winner as the representative; otherwise prefer
        # state, confidence, and longer coverage in that order.
        representative = next(
            (item for item in group if _signal_key(item) == winner_key),
            max(
                group,
                key=lambda item: (
                    _state_rank(item.state),
                    item.confirmed,
                    item.confidence,
                    item.end - item.start,
                ),
            ),
        )
        group_id = f"{resolved.value}:{canonicalize_pattern_id(representative.pattern)}:{index}"
        rep_key = _signal_key(representative)
        representative_keys.add(rep_key)
        spans = [
            {"start": int(item.start), "end": int(item.end)} for item in group
        ]
        group_payload.append(
            {
                "group_id": group_id,
                "pattern_id": canonicalize_pattern_id(representative.pattern),
                "direction": representative.direction.value,
                "representative": rep_key,
                "member_count": len(group),
                "spans": spans,
            }
        )
        for item in group:
            item.metadata["display_group_id"] = group_id
            item.metadata["display_group_size"] = len(group)
            item.metadata["merged_spans"] = spans
            item.metadata["display_group_representative"] = item is representative

    # A key-to-group lookup is needed to mark adjacent duplicate detections.
    group_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for payload in group_payload:
        group_by_key[payload["representative"]] = payload
        for span in payload["spans"]:
            group_by_key[
                (
                    payload["pattern_id"],
                    payload["direction"],
                    int(span["start"]),
                    int(span["end"]),
                )
            ] = payload

    # Manual browsing layers are intentionally built before timeframe roles,
    # winner priority and conflict suppression are applied.  They represent
    # the complete set of quantitatively valid, lifecycle-valid detections for
    # one family.  Exact duplicates are removed, but adjacent occurrences and
    # conflict losers remain inspectable.  The frontend may limit how many are
    # drawn at once; it must not delete the remaining records from the list.
    manual_layer_items: dict[str, list[DetectedSignal]] = {
        kind.value: [] for kind in PatternKind
    }
    manual_seen: set[tuple[str, str, int, int]] = set()
    for item in items:
        key = _signal_key(item)
        if key in manual_seen or item.state in {
            SignalState.INVALIDATED,
            SignalState.EXPIRED,
        }:
            continue
        manual_seen.add(key)
        manual_item = deepcopy(item)
        manual_item.metadata["manual_layer_visible"] = True
        manual_item.metadata["manual_layer_policy"] = (
            "quantitative_and_lifecycle_valid_before_priority_or_conflict_display"
        )
        manual_layer_items.setdefault(manual_item.kind.value, []).append(manual_item)

    primary: list[DetectedSignal] = []
    supporting: list[DetectedSignal] = []
    candidates: list[DetectedSignal] = []
    suppressed: list[DetectedSignal] = []
    timeframe_deemphasized: list[DetectedSignal] = []

    for item in items:
        key = _signal_key(item)
        base_role = profile.base_role(item.kind)
        group = group_by_key.get(key)
        is_representative = key in representative_keys
        if not is_representative:
            item.metadata["display_role"] = "duplicate"
            item.metadata["display_reason"] = "同一 canonical 形态的相邻检测已合并"
            suppressed.append(item)
            continue
        if item.state in {SignalState.INVALIDATED, SignalState.EXPIRED}:
            item.metadata["display_role"] = "suppressed"
            item.metadata["display_reason"] = f"生命周期状态为 {item.state.value}"
            suppressed.append(item)
            continue
        if key in loser_keys or item.metadata.get("conflict_suppressed_by"):
            item.metadata["display_role"] = "suppressed"
            item.metadata["display_reason"] = "被冲突规则降为非主结论"
            suppressed.append(item)
            continue
        if key == winner_key:
            item.metadata["display_role"] = "primary"
            item.metadata["display_priority"] = 100
            primary.append(item)
        elif base_role == "primary_candidate":
            item.metadata["display_role"] = (
                "candidate" if item.state != SignalState.CONFIRMED else "supporting"
            )
            item.metadata["display_priority"] = 80
            if item.state in {SignalState.CANDIDATE, SignalState.DEGRADED}:
                candidates.append(item)
            else:
                supporting.append(item)
        elif base_role == "supporting":
            item.metadata["display_role"] = (
                "candidate" if item.state != SignalState.CONFIRMED else "supporting"
            )
            item.metadata["display_priority"] = 50
            if item.state in {SignalState.CANDIDATE, SignalState.DEGRADED}:
                candidates.append(item)
            else:
                supporting.append(item)
        else:
            item.metadata["display_role"] = "timeframe_deemphasized"
            item.metadata["display_priority"] = 20
            item.metadata["display_reason"] = (
                f"{resolved.value} 周期默认不将 {item.kind.value} 作为首屏主形态"
            )
            timeframe_deemphasized.append(item)

    def rank(item: DetectedSignal) -> tuple[float, ...]:
        recent = item.end / max(max_end, 1)
        age = max(0, max_end - int(item.end))
        near_window, context_window = {
            Timeframe.MONTHLY: (6, 12),
            Timeframe.WEEKLY: (8, 16),
            Timeframe.DAILY: (15, 30),
            Timeframe.INTRADAY: (20, 50),
        }[resolved]
        recent_band = 2.0 if age < near_window else 1.0 if age < context_window else 0.0
        span = item.end - item.start + 1
        return (
            float(item.metadata.get("display_priority", 0)),
            recent_band,
            float(_state_rank(item.state)),
            float(item.confirmed),
            float(item.confidence),
            float(recent),
            float(span),
        )

    primary = sorted(primary, key=rank, reverse=True)[: profile.max_primary]
    primary_keys = {_signal_key(item) for item in primary}
    # A resolved winner must remain visible even if a future profile limit is
    # tightened; it is the explicit output of ConflictResolver.
    if winner_key and winner_key not in primary_keys:
        winner_item = next((item for item in items if _signal_key(item) == winner_key), None)
        if winner_item is not None:
            primary = [winner_item, *primary[: max(0, profile.max_primary - 1)]]
            primary_keys = {_signal_key(item) for item in primary}

    supporting = sorted(
        [item for item in supporting if _signal_key(item) not in primary_keys],
        key=rank,
        reverse=True,
    )[: profile.max_supporting]
    candidates = sorted(candidates, key=rank, reverse=True)
    timeframe_deemphasized = sorted(timeframe_deemphasized, key=rank, reverse=True)
    suppressed = sorted(suppressed, key=rank, reverse=True)

    # The mobile product defaults to a small adaptive set rather than dumping
    # every detector hit.  The resolved winner is always first, followed by
    # the highest-ranked non-duplicate signals from the timeframe's primary,
    # supporting and candidate buckets.  This is a presentation budget only;
    # ``all`` and ``manual_layers`` still retain the complete evidence set.
    winner_item = next(
        (item for item in items if _signal_key(item) == winner_key),
        None,
    )
    layer_items = manual_layer_items
    if enabled_kinds is None:
        selected_kinds = set(layer_items)
    else:
        selected_kinds = {
            item.value if isinstance(item, PatternKind) else str(item)
            for item in enabled_kinds
        }
    if mode == "adaptive":
        adaptive_pool: list[DetectedSignal] = []
        if winner_item is not None:
            adaptive_pool.append(winner_item)
        adaptive_pool.extend((*primary, *supporting, *candidates))

        def span_overlap_ratio(left: DetectedSignal, right: DetectedSignal) -> float:
            """Return overlap divided by the shorter signal span."""
            intersection = max(
                0,
                min(int(left.end), int(right.end))
                - max(int(left.start), int(right.start))
                + 1,
            )
            shorter = max(
                1,
                min(
                    int(left.end) - int(left.start) + 1,
                    int(right.end) - int(right.start) + 1,
                ),
            )
            return intersection / shorter

        def is_dense_secondary(item: DetectedSignal, selected: list[DetectedSignal]) -> bool:
            """Avoid a mobile label pile-up without deleting evidence.

            The full detector/result remains in ``all`` and ``manual_layers``.
            This gate only applies to the adaptive first screen.  A larger
            chart structure owns an overlapping candle/combination label, and
            repeated same-kind events close together collapse to the strongest
            ranked representative.
            """
            item_center = (int(item.start) + int(item.end)) / 2.0
            total_span = max(max_end + 1, 1)
            recent_focus = {
                Timeframe.MONTHLY: 3,
                Timeframe.WEEKLY: 6,
                Timeframe.DAILY: 12,
                Timeframe.INTRADAY: 20,
            }[resolved]
            for existing in selected:
                overlap = span_overlap_ratio(item, existing)
                existing_center = (int(existing.start) + int(existing.end)) / 2.0
                distance = abs(item_center - existing_center)
                same_pattern = canonicalize_pattern_id(item.pattern) == canonicalize_pattern_id(existing.pattern)
                if same_pattern and (overlap > 0.0 or distance <= 3.0):
                    return True
                # A confirmed/selected chart structure is the readable parent
                # of a small candle signal drawn inside the same span.
                if (
                    existing.kind == PatternKind.CHART
                    and existing.confirmed
                    and item.kind != PatternKind.CHART
                    and max_end - int(item.end) > recent_focus
                    and int(existing.end) - int(existing.start) + 1
                    <= max(20, int(total_span * 0.55))
                ):
                    if overlap >= 0.50:
                        return True
                # Identical or nearly identical spans from different small
                # detectors are still one visual location on a phone.
                if overlap >= 0.80:
                    return True
                if (
                    item.kind != PatternKind.CHART
                    and existing.kind != PatternKind.CHART
                    and distance <= 2.0
                ):
                    return True
            return False

        default_items = []
        seen_default: set[tuple[str, str, int, int]] = set()
        for item in sorted(adaptive_pool, key=rank, reverse=True):
            key = _signal_key(item)
            if key in seen_default:
                continue
            if is_dense_secondary(item, default_items):
                continue
            seen_default.add(key)
            default_items.append(item)
        # Daily output must not become visually empty merely because no chart
        # or trend candidate survived.  The fallback order is the product
        # contract: chart/trend -> combination -> simple.  All layers remain
        # available; this only guarantees at least one readable first-screen
        # annotation when any valid signal exists.
        if not default_items:
            fallback_order = (
                (PatternKind.CHART, PatternKind.TREND),
                (PatternKind.COMBINATION,),
                (PatternKind.SIMPLE,),
            )
            eligible = [
                item
                for item in (*primary, *supporting, *candidates, *timeframe_deemphasized)
                if item.state not in {SignalState.INVALIDATED, SignalState.EXPIRED}
            ]
            for kinds in fallback_order:
                fallback = sorted(
                    [item for item in eligible if item.kind in kinds],
                    key=rank,
                    reverse=True,
                )
                if fallback:
                    default_items.append(fallback[0])
                    break
        # Keep the explicit winner at the head even though it normally ranks
        # first; this protects the product contract if a future scoring rule
        # changes the numeric priority.
        if winner_item is not None:
            winner_key_local = _signal_key(winner_item)
            default_items = [
                winner_item,
                *[
                    item
                    for item in default_items
                    if _signal_key(item) != winner_key_local
                ],
            ]
        default_items = default_items[: profile.adaptive_default_limit]
    elif mode == "manual":
        default_items = []
        for kind in sorted(selected_kinds):
            for item in layer_items.get(kind, []):
                if not any(_signal_key(item) == _signal_key(existing) for existing in default_items):
                    default_items.append(item)
    else:  # audit: callers may use the full role buckets but still get a stable default.
        default_items = [winner_item] if winner_item is not None else primary[:1]

    # Recent signals are deliberately separate from the structural winner.
    # This lets a monthly chart say “current structure” and “latest trigger”
    # without pretending that the latest single candle is the main structure.
    recent = sorted(
        [
            item
            for item in items
            if item.state not in {SignalState.INVALIDATED, SignalState.EXPIRED}
            and item.metadata.get("display_role") not in {"suppressed", "duplicate"}
        ],
        key=lambda item: (item.end, _state_rank(item.state), item.confidence),
        reverse=True,
    )

    return {
        "profile": profile.to_dict(),
        "primary": _serialize(primary),
        "supporting": _serialize(supporting),
        "candidates": _serialize(candidates),
        "suppressed": _serialize(suppressed),
        "timeframe_deemphasized": _serialize(timeframe_deemphasized),
        "recent": _serialize(recent[:5]),
        "default": _serialize(default_items),
        "manual_layers": {
            kind: _serialize(
                sorted(
                    values,
                    key=lambda item: (
                        int(item.end),
                        int(item.start),
                        float(item.confidence),
                        canonicalize_pattern_id(item.pattern),
                    ),
                    reverse=True,
                )
            )
            for kind, values in layer_items.items()
        },
        "display_preferences": {
            "mode": mode,
            "default_policy": "adaptive_top_n",
            "adaptive_default_limit": profile.adaptive_default_limit,
            "enabled_kinds": sorted(selected_kinds),
            "available_kinds": sorted(layer_items),
            "fallback_order": ["chart_or_trend", "combination", "simple"],
            "manual_layer_policy": "完整保留量化与生命周期有效候选；不受周期优先级和冲突胜负隐藏；主图仅做视觉限密度，列表保持完整。",
            "manual_layer_counts": {
                kind: len(values) for kind, values in layer_items.items()
            },
            "mobile_note": "默认仅显示自动识别重点；简单/组合/趋势/技术图形由用户独立打开，完整候选保留在列表，主图自动错峰。",
        },
        "groups": group_payload,
        "all": _serialize(items),
        "summary": {
            "primary_count": len(primary),
            "supporting_count": len(supporting),
            "candidate_count": len(candidates),
            "suppressed_count": len(suppressed),
            "timeframe_deemphasized_count": len(timeframe_deemphasized),
            "recent_count": min(len(recent), 5),
            "winner_pattern_id": (
                canonicalize_pattern_id(resolution.winner.pattern)
                if resolution and resolution.winner
                else None
            ),
            "default_count": len(default_items),
        },
    }


__all__ = [
    "TimeframeProfile",
    "FORMAL_CHART_LIMITS",
    "ROLLING_CHART_CONFIDENCE_FLOORS",
    "build_display_view",
    "coerce_timeframe",
    "profile_for",
]
