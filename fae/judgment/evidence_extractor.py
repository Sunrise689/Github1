"""Extract objective evidence and enforce first-layer quantitative sanity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .schemas import DetectedSignal, PatternKind, Timeframe


# Public contract for fields referenced by ``conflict_rules.json``.  Keeping
# this list next to the extractor makes missing rule inputs auditable without
# importing or executing the resolver.
CONFLICT_EVIDENCE_FIELDS = frozenset(
    {
        "continuation_down",
        "continuation_up",
        "bearish_bar_count",
        "bullish_bar_count",
        "consecutive_bearish_bars",
    }
)

BREAKOUT_CONFIRMATION_DISTANCE = 0.03
BREAKOUT_CONFIRMATION_BARS = 3

# Formal chart-label policy is deliberately stricter than the raw detector.
# Monthly formal chart promotion is disabled by timeframe policy.  Weekly and
# daily bars have no hard count cap: every additional chart must pass a higher
# rolling confidence floor, while rejected candidates remain in audit JSON.
FORMAL_CHART_LIMITS: dict[str, int | None] = {
    Timeframe.MONTHLY.value: 0,
    Timeframe.WEEKLY.value: None,
    Timeframe.DAILY.value: None,
    Timeframe.INTRADAY.value: None,
}

ROLLING_CHART_CONFIDENCE_THRESHOLDS = (0.65, 0.85, 0.93)


def rolling_chart_threshold(rank: int) -> float:
    """Return the confidence floor for the next chart candidate.

    The first chart is allowed to be a useful but not perfect hypothesis.
    Every subsequent chart must be substantially stronger; after the third,
    the floor continues rising toward one rather than imposing an arbitrary
    count limit.  This is the product owner's “rolling upward” rule.
    """
    index = max(0, int(rank))
    if index < len(ROLLING_CHART_CONFIDENCE_THRESHOLDS):
        return ROLLING_CHART_CONFIDENCE_THRESHOLDS[index]
    return min(0.995, ROLLING_CHART_CONFIDENCE_THRESHOLDS[-1] + 0.02 * (index - 2))


def _timeframe_value(value: Timeframe | str | None) -> str | None:
    if isinstance(value, Timeframe):
        return value.value
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "m": Timeframe.MONTHLY.value,
        "month": Timeframe.MONTHLY.value,
        "monthly": Timeframe.MONTHLY.value,
        "w": Timeframe.WEEKLY.value,
        "week": Timeframe.WEEKLY.value,
        "weekly": Timeframe.WEEKLY.value,
        "d": Timeframe.DAILY.value,
        "day": Timeframe.DAILY.value,
        "daily": Timeframe.DAILY.value,
    }
    return aliases.get(text, text)


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return finite, ordered lower-case OHLCV data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    frame = data.rename(columns={column: str(column).lower() for column in data})
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    if "volume" not in frame:
        frame = frame.assign(volume=0.0)
    frame = frame[required + ["volume"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    valid = (
        (frame["close"] > 0)
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    return frame.loc[valid].sort_index()


def _consecutive(values: Sequence[bool], wanted: bool = True) -> int:
    count = 0
    for value in reversed(values):
        if bool(value) is wanted:
            count += 1
        else:
            break
    return count


def _max_consecutive(values: Sequence[bool], wanted: bool = True) -> int:
    """Return the longest run of ``wanted`` values in a boolean sequence."""
    best = current = 0
    for value in values:
        if bool(value) is wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def breakout_confirmation(
    close: Sequence[float],
    breakout_level: float,
    direction: str,
    *,
    start: int = 0,
    timeframe: Timeframe | str | None = None,
    open_: Sequence[float] | None = None,
    high: Sequence[float] | None = None,
    low: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Apply the textbook 3%-or-three-closes breakout confirmation rule.

    ``start`` points to the first bar after the pattern's breakout point.  A
    bullish confirmation is either a close at least 3% above the level or
    three consecutive closes above it.  The bearish case is symmetric.
    """
    values = np.asarray(close, dtype=float)
    level = float(breakout_level)
    if not np.isfinite(level) or level <= 0 or start >= len(values):
        return {
            "breakout_confirmed": False,
            "breakout_confirmation_mode": "none",
            "confirmation_streak_bars": 0,
            "breakout_distance_pct": 0.0,
            "confirmation_index": None,
        }
    future = values[max(0, int(start)) :]
    bullish = str(direction).lower() in {"bull", "bullish", "up"}
    if bullish:
        outside = future > level
        distance = (future / level - 1.0) * 100.0
        strong = future >= level * (1.0 + BREAKOUT_CONFIRMATION_DISTANCE)
    else:
        outside = future < level
        distance = (1.0 - future / level) * 100.0
        strong = future <= level * (1.0 - BREAKOUT_CONFIRMATION_DISTANCE)
    streak = _max_consecutive(outside)
    period = _timeframe_value(timeframe)
    # On daily bars the user-approved rule requires the breakout to be
    # accompanied by three consecutive shrinking bodies/shadows.  Weekly and
    # monthly bars use the 3% close test directly because each bar already
    # summarizes a much longer auction.
    daily_shrink = False
    if period == Timeframe.DAILY.value and open_ is not None and high is not None and low is not None:
        opens = np.asarray(open_, dtype=float)[max(0, int(start)) :]
        highs = np.asarray(high, dtype=float)[max(0, int(start)) :]
        lows = np.asarray(low, dtype=float)[max(0, int(start)) :]
        bodies = np.abs(future - opens)
        upper_shadows = highs - np.maximum(opens, future)
        lower_shadows = np.minimum(opens, future) - lows
        shadows = np.maximum(upper_shadows + lower_shadows, 0.0)
        if len(bodies) >= 3:
            # The approved daily rule is about the candle's *body and shadows*,
            # not merely its full high-low range.  Shrinking means
            # non-increasing over the last three bars, with a small tolerance
            # for market-data rounding.
            body_shrink = bool(np.all(np.diff(bodies[-3:]) <= np.maximum(bodies[-3:-1], 1e-12) * 0.08))
            shadow_shrink = bool(np.all(np.diff(shadows[-3:]) <= np.maximum(shadows[-3:-1], 1e-12) * 0.08))
            daily_shrink = body_shrink and shadow_shrink
    confirmation_index: int | None = None
    if bool(np.any(strong)) and (period != Timeframe.DAILY.value or daily_shrink):
        mode = "distance_3pct"
        confirmed = True
        confirmation_index = int(start + np.flatnonzero(strong)[0])
    elif period != Timeframe.DAILY.value and streak >= BREAKOUT_CONFIRMATION_BARS:
        mode = "three_day_stability"
        confirmed = True
        run = 0
        for offset, value in enumerate(outside):
            run = run + 1 if bool(value) else 0
            if run >= BREAKOUT_CONFIRMATION_BARS:
                confirmation_index = int(start + offset)
                break
    else:
        mode = "none"
        confirmed = False
    return {
        "breakout_confirmed": confirmed,
        "breakout_confirmation_mode": mode,
        "confirmation_streak_bars": int(streak),
        "breakout_distance_pct": float(np.max(distance)) if len(distance) else 0.0,
        "confirmation_index": confirmation_index,
        "daily_shrink_confirmed": daily_shrink,
        "daily_body_shrink_confirmed": bool(daily_shrink and period == Timeframe.DAILY.value),
        "daily_shadow_shrink_confirmed": bool(daily_shrink and period == Timeframe.DAILY.value),
        "timeframe_rule": period or "legacy",
    }


class EvidenceExtractor:
    """Compute context, confirmation, gap, line-break, and density evidence."""

    def extract(
        self,
        data: pd.DataFrame,
        *,
        reference_levels: Mapping[str, float] | None = None,
        detected_patterns: Sequence[str] | None = None,
        signal_windows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Extract evidence consumed by expert rules and chart validation.

        ``signal_windows`` is optional metadata supplied by the judgment
        orchestrator.  When present, the extractor also emits a
        ``signal_evidence`` mapping containing candle-polarity counts for each
        signal's own span.  This keeps declarative conflict rules local to the
        candidate they are evaluating instead of accidentally applying the
        last five bars of an unrelated window.

        Continuation flags are intentionally conservative: they are true only
        when at least two bars *after* a candidate continue in one direction.
        A pattern at the end of the available data therefore remains a
        candidate until follow-up bars exist.
        """
        frame = normalize_ohlcv(data)
        if len(frame) < 5:
            raise ValueError("at least five valid bars are required")
        levels = dict(reference_levels or {})
        close = frame["close"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        open_ = frame["open"].to_numpy(float)
        volume = frame["volume"].to_numpy(float)
        previous = np.r_[close[0], close[:-1]]
        true_range = np.maximum.reduce(
            [high - low, np.abs(high - previous), np.abs(low - previous)]
        )
        atr = float(np.mean(true_range[-min(14, len(frame)) :]))
        x = np.arange(min(60, len(frame)), dtype=float)
        y = np.log(close[-len(x) :])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        total = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / total if total else 0.0
        annualized_slope = float(np.expm1(np.clip(slope * 252, -5, 5)))
        trend = (
            "up"
            if annualized_slope > 0.08 and r_squared > 0.25
            else "down"
            if annualized_slope < -0.08 and r_squared > 0.25
            else "range"
        )

        body_pct = np.abs(close - open_) / np.maximum(close, 1e-12) * 100
        returns = close / previous - 1.0
        recent_body_base = float(np.median(body_pct[-min(20, len(frame)) :]))
        big_threshold = max(2.5, recent_body_base * 2.5)
        polarity_window = min(5, len(frame))
        bullish = close > open_
        bearish = close < open_
        recent_bullish_count = int(bullish[-polarity_window:].sum())
        recent_bearish_count = int(bearish[-polarity_window:].sum())
        consecutive_bullish = _consecutive(bullish)
        consecutive_bearish = _consecutive(bearish)
        gap_up = low[1:] > high[:-1] * 1.001
        gap_down = high[1:] < low[:-1] * 0.999
        gap_events: list[tuple[int, str, float]] = []
        for index in np.flatnonzero(gap_up):
            gap_events.append((int(index + 1), "up", float(high[index])))
        for index in np.flatnonzero(gap_down):
            gap_events.append((int(index + 1), "down", float(low[index])))
        gap_events.sort()
        last_gap_filled = False
        if gap_events:
            gap_index, gap_direction, fill_level = gap_events[-1]
            if gap_index < len(frame) - 1:
                last_gap_filled = bool(
                    np.min(low[gap_index + 1 :]) <= fill_level
                    if gap_direction == "up"
                    else np.max(high[gap_index + 1 :]) >= fill_level
                )
        avg_volume = float(np.mean(volume[-min(20, len(frame)) :]))
        volume_ratio = float(volume[-1] / avg_volume) if avg_volume > 0 else 1.0

        # The same upper/lower-shadow signal is interpreted through market
        # position.  At a high position frequent shadows imply supply and
        # disagreement; at a low position they imply absorption/support.  We
        # intentionally do not assign “upper = pressure, lower = support” as
        # a hard rule: both shadows carry the same auction-information role.
        body = np.abs(close - open_)
        upper_shadow = high - np.maximum(open_, close)
        lower_shadow = np.minimum(open_, close) - low
        candle_range = np.maximum(high - low, 1e-12)
        shadow_fraction = (upper_shadow + lower_shadow) / candle_range
        shadow_threshold = float(np.nanquantile(shadow_fraction[-min(30, len(frame)) :], 0.65))
        recent_slice = slice(-min(10, len(frame)), None)
        frequent_shadow = shadow_fraction[recent_slice] >= max(0.35, shadow_threshold)
        shadow_frequency = float(np.mean(frequent_shadow)) if len(frequent_shadow) else 0.0
        upper_frequency = float(
            np.mean((upper_shadow[recent_slice] / candle_range[recent_slice]) >= 0.20)
        )
        lower_frequency = float(
            np.mean((lower_shadow[recent_slice] / candle_range[recent_slice]) >= 0.20)
        )
        recent_body = body[-min(3, len(body)) :]
        prior_body = body[-min(13, len(body)) : -min(3, len(body))]
        recent_range = candle_range[-min(3, len(candle_range)) :]
        prior_range = candle_range[-min(13, len(candle_range)) : -min(3, len(candle_range))]
        body_shrink_ratio = float(np.median(recent_body) / max(np.median(prior_body), 1e-12)) if len(prior_body) else 1.0
        range_shrink_ratio = float(np.median(recent_range) / max(np.median(prior_range), 1e-12)) if len(prior_range) else 1.0
        position_pct = float(
            (close[-1] - np.min(close)) / max(np.max(close) - np.min(close), 1e-12)
        )
        if shadow_frequency >= 0.50 and position_pct >= 0.70:
            shadow_interpretation = "高位上下影线频繁：抛压/多空分歧明显"
        elif shadow_frequency >= 0.50 and position_pct <= 0.30:
            shadow_interpretation = "低位上下影线频繁：接盘/承接明显"
        elif shadow_frequency >= 0.50:
            shadow_interpretation = "上下影线频繁：多空分歧明显，等待方向确认"
        else:
            shadow_interpretation = "影线频率未达到提醒阈值"

        evidence: dict[str, Any] = {
            "bar_count": len(frame),
            "trend": trend,
            "trend_slope": annualized_slope,
            "trend_r2": r_squared,
            "position_pct": position_pct,
            "atr": atr,
            "atr_pct": atr / close[-1] * 100.0,
            "volume_ratio": volume_ratio,
            "upper_shadow_frequency": upper_frequency,
            "lower_shadow_frequency": lower_frequency,
            "shadow_frequency": shadow_frequency,
            "body_shrink_ratio": body_shrink_ratio,
            "range_shrink_ratio": range_shrink_ratio,
            "shadow_body_shrink_confirmed": bool(
                len(recent_body) >= 3
                and body_shrink_ratio <= 0.85
                and range_shrink_ratio <= 0.90
            ),
            "shadow_interpretation": shadow_interpretation,
            "gap_count": int(gap_up.sum() + gap_down.sum()),
            "up_gap_count": int(gap_up.sum()),
            "down_gap_count": int(gap_down.sum()),
            "last_gap_direction": (
                "up"
                if len(gap_up) and gap_up[-1]
                else "down"
                if len(gap_down) and gap_down[-1]
                else "none"
            ),
            "last_gap_filled": last_gap_filled,
            "big_bullish_count_10": int(
                ((returns[-10:] > 0) & (body_pct[-10:] >= big_threshold)).sum()
            ),
            "big_bearish_count_10": int(
                ((returns[-10:] < 0) & (body_pct[-10:] >= big_threshold)).sum()
            ),
            "recent_return_pct": float(
                (close[-1] / close[-min(20, len(frame))] - 1) * 100
            ),
            # These fields are the quantitative proxies consumed by the
            # declarative conflict table.  Counts use a bounded recent window
            # so they cannot grow with a long historical input.
            "polarity_window_bars": polarity_window,
            "bullish_bar_count": recent_bullish_count,
            "bearish_bar_count": recent_bearish_count,
            "consecutive_bullish_bars": consecutive_bullish,
            "consecutive_bearish_bars": consecutive_bearish,
            "continuation_down": bool(
                trend == "down"
                and consecutive_bearish >= 2
                and len(close) >= 2
                and close[-1] < close[-2]
            ),
            "continuation_up": bool(
                trend == "up"
                and consecutive_bullish >= 2
                and len(close) >= 2
                and close[-1] > close[-2]
            ),
            "detected_patterns": list(detected_patterns or []),
        }

        # Build signal-local evidence when the caller provides positional
        # spans.  The key includes the canonical pattern and span so aliases
        # cannot overwrite one another.
        local_evidence: dict[str, dict[str, Any]] = {}
        for window in signal_windows or ():
            try:
                start = max(0, int(window["start"]))
                end = min(len(frame) - 1, int(window["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end < start:
                continue
            local = frame.iloc[start : end + 1]
            local_bullish = (local["close"] > local["open"]).to_numpy(bool)
            local_bearish = (local["close"] < local["open"]).to_numpy(bool)
            raw_confirmation_start = window.get("confirmation_start")
            try:
                parsed_confirmation_start = (
                    int(raw_confirmation_start)
                    if raw_confirmation_start is not None
                    else end + 1
                )
            except (TypeError, ValueError):
                parsed_confirmation_start = end + 1
            confirmation_start = max(0, parsed_confirmation_start)
            post = frame.iloc[confirmation_start:]
            post_close = post["close"].to_numpy(float)
            down_continuation = bool(
                len(post_close) >= 2 and np.all(np.diff(post_close) < 0)
            )
            up_continuation = bool(
                len(post_close) >= 2 and np.all(np.diff(post_close) > 0)
            )
            local_payload: dict[str, Any] = {
                "bar_count": int(len(local)),
                "bullish_bar_count": int(local_bullish.sum()),
                "bearish_bar_count": int(local_bearish.sum()),
                "consecutive_bullish_bars": _consecutive(local_bullish),
                "consecutive_bearish_bars": _consecutive(local_bearish),
                "continuation_down": down_continuation,
                "continuation_up": up_continuation,
                "post_confirmation_bars": int(len(post_close)),
            }
            pattern = str(window.get("pattern", ""))
            level = window.get("breakout_level")
            if level is None:
                if pattern in {"island_reversal_bottom", "tower_bottom"}:
                    level = float(high[end])
                elif pattern in {"island_reversal_top", "tower_top"}:
                    level = float(low[end])
            if level is not None and pattern in {
                "island_reversal_bottom",
                "tower_bottom",
                "island_reversal_top",
                "tower_top",
            }:
                direction = (
                    "bullish"
                    if pattern in {"island_reversal_bottom", "tower_bottom"}
                    else "bearish"
                )
                local_payload.update(
                    breakout_confirmation(
                        close,
                        float(level),
                        direction,
                        start=confirmation_start,
                        timeframe=window.get("timeframe"),
                        open_=open_,
                        high=high,
                        low=low,
                    )
                )
                local_payload["breakout_level"] = float(level)
                local_payload["confirmation_start"] = confirmation_start
            key = f"{pattern}:{start}:{end}"
            local_evidence[key] = local_payload
        if local_evidence:
            evidence["signal_evidence"] = local_evidence
        for name, level in levels.items():
            level = float(level)
            if not np.isfinite(level) or level <= 0:
                continue
            below = close < level
            above = close > level
            evidence[f"days_below_{name}"] = _consecutive(below)
            evidence[f"days_above_{name}"] = _consecutive(above)
            evidence[f"break_distance_{name}_pct"] = float(
                abs(close[-1] - level) / level * 100
            )
            evidence[f"break_distance_{name}_atr"] = float(
                abs(close[-1] - level) / max(atr, 1e-12)
            )
            evidence[f"reclaim_{name}"] = bool(
                len(close) >= 2 and close[-2] < level <= close[-1]
            )
        if "neckline" in levels:
            evidence["days_below_neckline"] = evidence["days_below_neckline"]
            evidence["break_distance_pct"] = evidence["break_distance_neckline_pct"]
            evidence["break_distance_atr"] = evidence["break_distance_neckline_atr"]
            evidence["reclaim_neckline"] = evidence["reclaim_neckline"]
        if "trendline" in levels:
            evidence["days_below_trendline"] = evidence["days_below_trendline"]
            evidence["reclaim_trendline"] = evidence["reclaim_trendline"]
        return evidence


class QuantitativeGate:
    """Reject impossible or over-dense detections before expert judgment."""

    def __init__(
        self,
        *,
        max_chart_patterns_per_100: int | None = None,
        min_chart_bars: int = 20,
        max_overlap_ratio: float = 0.65,
    ) -> None:
        self.max_chart_patterns_per_100 = max_chart_patterns_per_100
        self.min_chart_bars = min_chart_bars
        self.max_overlap_ratio = max_overlap_ratio

    def validate_signal(
        self,
        signal: DetectedSignal,
        evidence: Mapping[str, Any],
    ) -> tuple[bool, list[str]]:
        """Check one signal against basic statistical and logical constraints."""
        reasons: list[str] = []
        span = signal.end - signal.start + 1
        if signal.kind == PatternKind.CHART:
            family = str(signal.metadata.get("family", ""))
            family_minimum = {
                # Island reversals are defined by a local pair of gaps and
                # must not be inflated to a whole-chart box merely to pass a
                # generic duration gate.
                "island": 2,
                "v_reversal": 8,
                "flag": 8,
                "multiple_top_bottom": 10,
                "head_and_shoulders": 15,
            }.get(family, self.min_chart_bars)
            if span < family_minimum:
                reasons.append(
                    f"技术图形持续长度不足（{family or 'chart'}至少{family_minimum}根）"
                )
        if not 0.0 <= signal.confidence <= 1.0:
            reasons.append("置信度不在[0,1]")
        if (
            signal.pattern in {"large_bullish", "大阳线"}
            and int(evidence.get("big_bullish_count_10", 0)) >= 5
        ):
            reasons.append("十根K线中大阳线过密，与第一层量化常识冲突")
        if (
            signal.pattern in {"large_bearish", "大阴线"}
            and int(evidence.get("big_bearish_count_10", 0)) >= 5
        ):
            reasons.append("十根K线中大阴线过密，与第一层量化常识冲突")
        if signal.kind == PatternKind.CHART:
            amplitude_atr = float(signal.metadata.get("amplitude_atr", 0.0))
            if amplitude_atr and amplitude_atr < 2.0:
                reasons.append("图形振幅不足两个ATR，可能只是市场噪声")
            pivot_count = int(signal.metadata.get("pivot_count", 0))
            if (
                pivot_count
                and pivot_count < 3
                and signal.metadata.get("family")
                not in {"v_reversal", "rounding", "island", "base"}
            ):
                reasons.append("有效摆动点不足，无法构成技术图形")
            if signal.pattern == "symmetrical_triangle":
                if int(signal.metadata.get("wave_count", 0)) < 5:
                    reasons.append("收敛三角形至少需要多轮交替波幅，当前波浪数量不足")
                if float(signal.metadata.get("convergence_ratio", 0.0)) < 0.10:
                    reasons.append("收敛宽度不足，不能把一两次摆动命名为收敛三角形")
                if float(signal.metadata.get("start_width", 0.0)) < max(evidence.get("atr", 0.0) * 3.0, 0.0):
                    reasons.append("收敛三角形起始波幅不足")
        return not reasons, reasons

    def filter_chart_signals(
        self,
        signals: Sequence[DetectedSignal],
        total_bars: int,
        timeframe: Timeframe | str | None = None,
    ) -> list[DetectedSignal]:
        """Apply overlap, contradiction, spacing, and density suppression."""
        chart = [item for item in signals if item.kind == PatternKind.CHART]
        family_priority = {
            # Specific structures win over generic geometric approximations.
            "cup_with_handle": 5,
            "head_and_shoulders": 4,
            "island": 4,
            "base": 4,
            "multiple_top_bottom": 3,
            "rounding": 2,
            "v_reversal": 2,
            "consolidation": 1,
            "flag": 1,
        }
        chart.sort(
            key=lambda item: (
                family_priority.get(str(item.metadata.get("family", "")), 0),
                bool(item.confirmed),
                item.confidence,
            ),
            reverse=True,
        )
        kept: list[DetectedSignal] = []
        period = _timeframe_value(timeframe)
        raw_capacity = (
            len(chart)
            if self.max_chart_patterns_per_100 is None
            else max(
                1,
                int(np.ceil(total_bars / 100)) * self.max_chart_patterns_per_100,
            )
        )
        formal_limit = FORMAL_CHART_LIMITS.get(period) if period else None
        capacity = raw_capacity if formal_limit is None else min(raw_capacity, formal_limit)
        if capacity <= 0:
            return []
        mutually_exclusive = {
            "cup_with_handle",
            "head_and_shoulders",
            "island",
            "base",
            "multiple_top_bottom",
            "rounding",
            "v_reversal",
            "consolidation",
            "flag",
        }
        ranked_kept: list[DetectedSignal] = []
        for candidate in chart:
            rank = len(ranked_kept)
            threshold = rolling_chart_threshold(rank)
            if candidate.confidence < threshold:
                candidate.metadata["formal_chart_rejected"] = True
                candidate.metadata["formal_chart_reject_reason"] = (
                    f"第{rank + 1}个技术图形要求滚动置信度≥{threshold:.0%}"
                )
                continue
            conflict = False
            for existing in kept:
                overlap = max(
                    0,
                    min(candidate.end, existing.end)
                    - max(candidate.start, existing.start)
                    + 1,
                )
                union = (
                    max(candidate.end, existing.end)
                    - min(candidate.start, existing.start)
                    + 1
                )
                ratio = overlap / max(union, 1)
                opposite = candidate.direction != existing.direction
                same_family = candidate.metadata.get("family") == existing.metadata.get(
                    "family"
                )
                related_specific = {
                    frozenset({"cup_with_handle", "rounding"}),
                    frozenset({"cup_with_handle", "consolidation"}),
                    frozenset({"cup_with_handle", "flag"}),
                    frozenset({"base", "consolidation"}),
                    frozenset({"base", "rounding"}),
                }
                families = frozenset(
                    {
                        str(candidate.metadata.get("family", "")),
                        str(existing.metadata.get("family", "")),
                    }
                )
                overlap_limit = (
                    0.40 if families in related_specific else self.max_overlap_ratio
                )
                exclusive_overlap = (
                    families <= mutually_exclusive
                    and len(families) == 2
                    and ratio > 0.35
                )
                if ratio > overlap_limit and (
                    opposite or same_family or families in related_specific
                ) or exclusive_overlap:
                    conflict = True
                    break
                if (
                    candidate.pattern == existing.pattern
                    and abs(candidate.end - existing.end) < 20
                ):
                    conflict = True
                    break
            if not conflict:
                candidate.metadata["formal_chart_rank"] = len(ranked_kept) + 1
                candidate.metadata["formal_chart_threshold"] = threshold
                candidate.metadata["formal_chart_selection"] = "adaptive_rolling_confidence"
                kept.append(candidate)
                ranked_kept.append(candidate)
            if len(kept) >= capacity:
                break
        return sorted(kept, key=lambda item: item.start)
