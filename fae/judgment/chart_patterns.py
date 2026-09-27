"""Sparse, quantitative chart-pattern detection with adaptive geometry.

The detector intentionally returns few patterns. It uses swing points,
regression geometry, ATR-normalized amplitude, confirmation rules, overlap
suppression, and mutual exclusion between top and bottom interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .evidence_extractor import (
    QuantitativeGate,
    breakout_confirmation,
    normalize_ohlcv,
)
from .schemas import (
    ArcGeometry,
    DetectedSignal,
    Direction,
    LineGeometry,
    PatternGeometry,
    PatternKind,
    SignalState,
    Timeframe,
)


@dataclass
class _Swings:
    peaks: np.ndarray
    troughs: np.ndarray


def _fit_line(points: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(points) < 2:
        return 0.0, float(values[-1]) if len(values) else 0.0, 0.0
    slope, intercept = np.polyfit(points, values, 1)
    fitted = slope * points + intercept
    total = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - float(np.sum((values - fitted) ** 2)) / total if total else 1.0
    return float(slope), float(intercept), max(0.0, min(1.0, r2))


def _similar(values: np.ndarray, tolerance: float = 0.07) -> bool:
    return bool(
        len(values)
        and (float(np.max(values)) - float(np.min(values)))
        / max(abs(float(np.mean(values))), 1e-12)
        <= tolerance
    )


def _structure_arc(
    indexes: np.ndarray,
    values: np.ndarray,
    role: str,
) -> ArcGeometry:
    """Build a compact quadratic bracket through three H&S pivots.

    The arc is deliberately tied to the detected shoulder/head points rather
    than spanning the whole chart.  Consumers can therefore draw a visible
    “左肩—头—右肩” bracket without inventing extra trend lines.
    """
    points = np.asarray(indexes, dtype=float)
    heights = np.asarray(values, dtype=float)
    coefficients = np.polyfit(points, heights, 2)
    return ArcGeometry(
        role,
        int(points[0]),
        int(points[-1]),
        tuple(float(value) for value in coefficients),
    )


class ChartPatternDetector:
    """Detect reversal and consolidation structures without label flooding."""

    def __init__(
        self,
        *,
        max_patterns_per_100: int | None = None,
        min_window: int = 30,
        max_window: int = 120,
        timeframe: Timeframe = Timeframe.DAILY,
    ) -> None:
        self.min_window = min_window
        self.max_window = max_window
        self.timeframe = timeframe
        self.gate = QuantitativeGate(
            max_chart_patterns_per_100=max_patterns_per_100,
            min_chart_bars=max(20, min_window // 2),
        )

    @staticmethod
    def _atr(frame: pd.DataFrame) -> float:
        close = frame["close"].to_numpy(float)
        previous = np.r_[close[0], close[:-1]]
        tr = np.maximum.reduce(
            [
                frame["high"].to_numpy(float) - frame["low"].to_numpy(float),
                np.abs(frame["high"].to_numpy(float) - previous),
                np.abs(frame["low"].to_numpy(float) - previous),
            ]
        )
        return max(float(np.mean(tr[-min(14, len(tr)) :])), close[-1] * 0.002)

    def _swings(self, frame: pd.DataFrame, atr: float) -> _Swings:
        distance = max(3, len(frame) // 14)
        prominence = max(atr * 0.8, float(frame["close"].std()) * 0.12)
        peaks, _ = find_peaks(
            frame["high"].to_numpy(float),
            distance=distance,
            prominence=prominence,
        )
        troughs, _ = find_peaks(
            -frame["low"].to_numpy(float),
            distance=distance,
            prominence=prominence,
        )
        return _Swings(peaks=peaks, troughs=troughs)

    def _signal(
        self,
        pattern: str,
        direction: Direction,
        confidence: float,
        frame: pd.DataFrame,
        geometry: PatternGeometry,
        *,
        family: str,
        atr: float,
        confirmed: bool = False,
        measurement_height: float | None = None,
        breakout_level: float | None = None,
        false_break: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> DetectedSignal:
        offset = int(frame.attrs.get("_offset", 0))
        if offset:
            shifted_lines = [
                LineGeometry(
                    line.role,
                    line.x1 + offset,
                    line.y1,
                    line.x2 + offset,
                    line.y2,
                    line.style,
                )
                for line in geometry.lines
            ]
            shifted_arcs: list[ArcGeometry] = []
            for arc in geometry.arcs:
                a, b, c = arc.coefficients
                shifted_arcs.append(
                    ArcGeometry(
                        arc.role,
                        arc.start + offset,
                        arc.end + offset,
                        (
                            a,
                            b - 2.0 * a * offset,
                            a * offset * offset - b * offset + c,
                        ),
                    )
                )
            geometry = PatternGeometry(
                lines=shifted_lines,
                arcs=shifted_arcs,
                pivots=[
                    (index + offset, value, role)
                    for index, value, role in geometry.pivots
                ],
            )
        # ``frame`` is a detector window, not automatically the pattern span.
        # Using the whole window made an island/double-bottom cover nearly the
        # entire chart and caused every inner candle label to be suppressed.
        # Pivots/arcs are the authoritative local structure; line extensions
        # are drawing aids and therefore do not enlarge the semantic span.
        structure_indexes: list[int] = [
            int(index) for index, _, _ in geometry.pivots
        ]
        for arc in geometry.arcs:
            structure_indexes.extend((int(arc.start), int(arc.end)))
        if not structure_indexes:
            for line in geometry.lines:
                structure_indexes.extend((int(line.x1), int(line.x2)))
        if structure_indexes:
            structure_start = max(0, min(structure_indexes))
            structure_end = max(structure_start, max(structure_indexes))
        else:
            structure_start = offset
            structure_end = offset + len(frame) - 1
        height = float(
            measurement_height
            if measurement_height is not None
            else frame["high"].max() - frame["low"].min()
        )
        target = None
        if breakout_level is not None and direction != Direction.NEUTRAL:
            target = (
                breakout_level + height
                if direction == Direction.BULLISH
                else breakout_level - height
            )
        metadata: dict[str, Any] = {
            "family": family,
            "duration": structure_end - structure_start + 1,
            "detection_window_duration": len(frame),
            "amplitude_atr": height / max(atr, 1e-12),
            "pivot_count": len(geometry.pivots),
            "breakout_level": breakout_level,
            "measurement_height": height,
            "measurement_target": target,
            "measurement_is_reference_only": True,
            "false_break": false_break,
            "confirmation": (
                "收盘离开边界至少max(1%,0.5ATR)，并满足放量1.2倍或连续2日"
            ),
        }
        metadata.update(extra or {})
        return DetectedSignal(
            pattern=pattern,
            direction=direction,
            confidence=confidence,
            start=structure_start,
            end=structure_end,
            kind=PatternKind.CHART,
            timeframe=self.timeframe,
            state=SignalState.CONFIRMED if confirmed else SignalState.CANDIDATE,
            confirmed=confirmed,
            geometry=geometry,
            metadata=metadata,
        )

    def _breakout(
        self,
        frame: pd.DataFrame,
        level: float,
        direction: Direction,
        atr: float,
    ) -> tuple[bool, bool]:
        close = frame["close"].to_numpy(float)
        margin = max(level * 0.01, atr * 0.5)
        if direction == Direction.BULLISH:
            outside = close > level + margin
            reclaimed = len(close) >= 2 and close[-2] > level and close[-1] <= level
        else:
            outside = close < level - margin
            reclaimed = len(close) >= 2 and close[-2] < level and close[-1] >= level
        # Adaptive confirmation approved for FAE: weekly/monthly structures
        # use the latest close at least 3% beyond the boundary; daily
        # structures additionally require three shrinking bodies and shadows.
        # A candidate may therefore remain visible without being promoted to
        # “confirmed” when the final bars do not support the move.
        period = self.timeframe.value
        strong_close = (
            close[-1] >= level * 1.03
            if direction == Direction.BULLISH
            else close[-1] <= level * 0.97
        )
        adaptive_confirmed = strong_close
        if period == Timeframe.DAILY.value:
            opens = frame["open"].to_numpy(float)
            highs = frame["high"].to_numpy(float)
            lows = frame["low"].to_numpy(float)
            bodies = np.abs(close - opens)
            shadows = np.maximum(highs - lows - bodies, 0.0)
            if len(bodies) < 3:
                adaptive_confirmed = False
            else:
                body_shrink = bool(
                    np.all(
                        np.diff(bodies[-3:])
                        <= np.maximum(bodies[-3:-1], 1e-12) * 0.08
                    )
                )
                shadow_shrink = bool(
                    np.all(
                        np.diff(shadows[-3:])
                        <= np.maximum(shadows[-3:-1], 1e-12) * 0.08
                    )
                )
                adaptive_confirmed = strong_close and body_shrink and shadow_shrink
        # The old two-bar/volume calculation is intentionally not used as a
        # bypass: confirmation is now period-aware and follows the same
        # quantitative contract as the standalone evidence extractor.
        return bool(adaptive_confirmed), bool(reclaimed)

    def _cup_with_handle_candidate(
        self,
        frame: pd.DataFrame,
        atr: float,
    ) -> DetectedSignal | None:
        """Return a conservative bullish cup-and-handle candidate.

        The cup is fitted between two similar rims and a middle trough.  The
        handle must be a short, shallow pullback that stays well above the
        cup bottom.  This deliberately emits a candidate before breakout and
        only marks it confirmed after the rim resistance is cleared.
        """
        length = len(frame)
        if length < 60:
            return None
        close = frame["close"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        # The handle is normally a short final consolidation, not a quarter
        # of the entire cup.  Keeping it near one-eighth of the window also
        # leaves enough bars to identify the right rim.
        handle_len = max(6, min(length // 8, 16))
        initial_handle_start = length - handle_len
        if initial_handle_start < int(length * 0.62):
            return None

        left_window_end = max(12, int(length * 0.38))
        bottom_start = max(4, int(length * 0.18))
        bottom_end = min(initial_handle_start - 8, int(length * 0.70))
        if bottom_end <= bottom_start:
            return None
        left_index = int(np.argmax(high[:left_window_end]))
        bottom_index = bottom_start + int(np.argmin(low[bottom_start:bottom_end]))
        right_slice = high[bottom_index + 4 : initial_handle_start]
        if len(right_slice) < 8:
            return None
        right_index = bottom_index + 4 + int(np.argmax(right_slice))
        left_rim = float(high[left_index])
        right_rim = float(high[right_index])
        rim = float((left_rim + right_rim) / 2.0)
        rim_tolerance = max(atr * 1.5, rim * 0.12)
        depth = rim - float(low[bottom_index])
        if depth < atr * 2.0 or abs(left_rim - right_rim) > rim_tolerance:
            return None
        if float(low[bottom_index]) >= min(left_rim, right_rim) - atr:
            return None

        breakout_margin = max(rim * 0.01, atr * 0.5)
        breakout_after_rim = np.flatnonzero(
            close[right_index + 1 :] > rim + breakout_margin
        )
        breakout_index = (
            right_index + 1 + int(breakout_after_rim[0])
            if len(breakout_after_rim)
            else None
        )
        handle_end = (
            int(breakout_index)
            if breakout_index is not None and breakout_index >= right_index + 4
            else length
        )
        handle_start = max(handle_end - handle_len, right_index + 1)
        if handle_end - handle_start < 3:
            return None

        handle_close = close[handle_start:handle_end]
        handle_low = float(np.min(low[handle_start:handle_end]))
        handle_high = float(np.max(high[handle_start:handle_end]))
        handle_span = handle_high - handle_low
        handle_slope, _, _ = _fit_line(
            np.arange(len(handle_close)), handle_close
        )
        if handle_low < rim - depth * 0.5:
            return None
        if handle_high > rim + max(atr * 0.75, rim * 0.015):
            return None
        # A handle may be a slightly jagged pullback; its direction matters
        # more than a high R² straight-line fit.
        if handle_slope > atr * 0.25:
            return None

        cup_x = np.arange(left_index, right_index + 1, dtype=float)
        cup_y = close[left_index : right_index + 1]
        if len(cup_x) < 12:
            return None
        coefficients = tuple(float(value) for value in np.polyfit(cup_x, cup_y, 2))
        support_slope, support_intercept, _ = _fit_line(
            np.arange(handle_start, handle_end), low[handle_start:handle_end]
        )
        support_end_index = max(handle_start, handle_end - 1)
        support_end = support_slope * support_end_index + support_intercept
        confirmed, false_break = self._breakout(frame, rim, Direction.BULLISH, atr)
        geometry = PatternGeometry(
            lines=[
                LineGeometry(
                    "rim_resistance",
                    left_index,
                    left_rim,
                    right_index,
                    right_rim,
                ),
                LineGeometry(
                    "handle_support",
                    handle_start,
                    float(low[handle_start]),
                    support_end_index,
                    float(support_end),
                ),
            ],
            arcs=[ArcGeometry("cup", left_index, right_index, coefficients)],
            pivots=[
                (left_index, left_rim, "left_rim"),
                (bottom_index, float(low[bottom_index]), "cup_bottom"),
                (right_index, right_rim, "right_rim"),
                (handle_start, float(low[handle_start]), "handle_start"),
            ],
        )
        return self._signal(
            "cup_with_handle",
            Direction.BULLISH,
            min(0.90, 0.64 + (0.08 if confirmed else 0.0)),
            frame,
            geometry,
            family="cup_with_handle",
            atr=atr,
            confirmed=confirmed,
            measurement_height=depth,
            breakout_level=rim,
            false_break=false_break,
            extra={
                "cup_left_rim": left_index,
                "cup_bottom": bottom_index,
                "cup_right_rim": right_index,
                "handle_start": handle_start,
                "handle_span": handle_span,
            },
        )

    def _dormant_bottom_candidate(
        self,
        frame: pd.DataFrame,
        atr: float,
    ) -> DetectedSignal | None:
        """Return a low-volume, narrow-base (潜伏底) candidate.

        潜伏底 is not treated as a second rectangle name: it requires a
        meaningful preceding decline, a long quiet base and a later breakout.
        The display layer may present it as a base/box variant.
        """
        length = len(frame)
        if length < 48:
            return None
        close = frame["close"].to_numpy(float)
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        volume = frame["volume"].to_numpy(float)
        base_start = max(12, int(length * 0.45))
        # Exclude the latest breakout leg from the quiet-base statistics.
        # Otherwise a valid dormant bottom would be rejected because the
        # breakout itself widens the final window's range.
        base_end = max(base_start + 20, length - max(5, length // 10))
        base_end = min(base_end, length)
        base = close[base_start:base_end]
        if len(base) < 20:
            return None
        prior = close[:base_start]
        prior_return = float(prior[-1] / max(prior[0], 1e-12) - 1.0)
        base_high = float(np.max(high[base_start:base_end]))
        base_low = float(np.min(low[base_start:base_end]))
        base_range = base_high - base_low
        base_mean = max(float(np.mean(base)), 1e-12)
        base_volatility = float(np.std(base))
        prior_volatility = float(np.std(prior))
        prior_volume = float(np.median(volume[:base_start]))
        base_volume = float(np.median(volume[base_start:base_end]))
        quiet_range = base_range <= max(atr * 7.0, base_mean * 0.12)
        quiet_volatility = prior_volatility <= 0 or base_volatility <= prior_volatility * 0.80
        quiet_volume = prior_volume <= 0 or base_volume <= prior_volume * 0.95
        if prior_return > -0.08 or not (quiet_range and quiet_volatility and quiet_volume):
            return None

        base_x = np.arange(base_start, base_end)
        upper_slope, upper_intercept, _ = _fit_line(
            base_x, high[base_start:base_end]
        )
        lower_slope, lower_intercept, _ = _fit_line(
            base_x, low[base_start:base_end]
        )
        upper_end = upper_slope * (length - 1) + upper_intercept
        lower_end = lower_slope * (length - 1) + lower_intercept
        # Four explicit boundary points keep the pattern eligible for the
        # geometric quality gate while still drawing a simple base box.
        upper_indices = np.argsort(high[base_start:base_end])[-2:] + base_start
        lower_indices = np.argsort(low[base_start:base_end])[:2] + base_start
        pivots = [
            (int(index), float(high[index]), "base_resistance")
            for index in upper_indices
        ] + [
            (int(index), float(low[index]), "base_support")
            for index in lower_indices
        ]
        base_frame = frame.iloc[base_start:].copy()
        confirmed, false_break = self._breakout(
            base_frame, base_high, Direction.BULLISH, atr
        )
        direction = Direction.BULLISH if confirmed else Direction.NEUTRAL
        geometry = PatternGeometry(
            lines=[
                LineGeometry(
                    "base_resistance",
                    base_start,
                    float(upper_slope * base_start + upper_intercept),
                    length - 1,
                    float(upper_end),
                ),
                LineGeometry(
                    "base_support",
                    base_start,
                    float(lower_slope * base_start + lower_intercept),
                    length - 1,
                    float(lower_end),
                ),
            ],
            pivots=pivots,
        )
        return self._signal(
            "dormant_bottom",
            direction,
            min(0.84, 0.58 + (0.12 if confirmed else 0.0)),
            frame,
            geometry,
            family="base",
            atr=atr,
            confirmed=confirmed,
            measurement_height=base_range,
            breakout_level=base_high,
            false_break=false_break,
            extra={
                "base_start": base_start,
                "prior_return": prior_return,
                "volume_contraction": quiet_volume,
                "display_variant": "box_base",
            },
        )

    def _reversal_candidates(
        self,
        frame: pd.DataFrame,
        swings: _Swings,
        atr: float,
    ) -> list[DetectedSignal]:
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)
        candidates: list[DetectedSignal] = []

        def has_prior_trend(first_pivot: int, suffix: str) -> bool:
            """Require a meaningful trend before naming a reversal structure.

            A pair of similar highs/lows inside a mature box is not a double
            top/bottom by itself.  The first structural pivot must be preceded
            by a directional move of at least roughly four percent or three
            ATR, with the regression slope pointing the same way.
            """
            lookback = min(32, int(first_pivot))
            if lookback < 6:
                return False
            start = int(first_pivot) - lookback
            segment = close[start : int(first_pivot) + 1]
            x_values = np.arange(len(segment), dtype=float)
            slope, _, _ = _fit_line(x_values, segment)
            baseline = max(abs(float(np.median(segment[: min(3, len(segment))]))), 1e-12)
            terminal = float(np.median(segment[-min(3, len(segment)) :]))
            move = terminal / baseline - 1.0
            required = max(0.04, 3.0 * atr / baseline)
            return bool(
                (suffix == "top" and move >= required and slope > 0.0)
                or (suffix == "bottom" and move <= -required and slope < 0.0)
            )

        def neckline_pivot(left: int, right: int, suffix: str) -> tuple[int, float, str]:
            if suffix == "top":
                local = low[left : right + 1]
                index = left + int(np.argmin(local))
                return index, float(low[index]), "neckline_low"
            local = high[left : right + 1]
            index = left + int(np.argmax(local))
            return index, float(high[index]), "neckline_high"

        def reversal_has_followthrough(
            pivots: np.ndarray,
            heights: np.ndarray,
            neckline: float,
            suffix: str,
        ) -> bool:
            """Reject a top/bottom that price has already invalidated.

            For an unfinished candidate we still require price to have moved
            away from the second extreme.  This prevents a current high-level
            box (or a fresh all-time high) from being labelled a double top.
            """
            mean_extreme = float(np.mean(heights))
            depth = abs(mean_extreme - float(neckline))
            if depth < max(atr * 2.0, abs(mean_extreme) * 0.025):
                return False
            latest = float(close[-1])
            margin = max(atr * 0.35, abs(mean_extreme) * 0.003)
            if suffix == "top":
                if latest > float(np.max(heights)) + margin:
                    return False
                return latest <= mean_extreme - depth * 0.25
            if latest < float(np.min(heights)) - margin:
                return False
            return latest >= mean_extreme + depth * 0.25

        for indexes, values, direction, suffix in (
            (swings.peaks, high, Direction.BEARISH, "top"),
            (swings.troughs, low, Direction.BULLISH, "bottom"),
        ):
            if len(indexes) >= 2:
                pair = indexes[-2:]
                neck_index, between, neck_role = neckline_pivot(
                    int(pair[0]), int(pair[1]), suffix
                )
                if (
                    _similar(values[pair], 0.06)
                    and pair[1] - pair[0] >= 5
                    and has_prior_trend(int(pair[0]), suffix)
                    and reversal_has_followthrough(pair, values[pair], between, suffix)
                ):
                    confirmed, false_break = self._breakout(
                        frame, float(between), direction, atr
                    )
                    geometry = PatternGeometry(
                        lines=[
                            LineGeometry(
                                "neckline",
                                int(pair[0]),
                                float(between),
                                int(pair[1]),
                                float(between),
                            )
                        ],
                        pivots=[
                            (int(pair[0]), float(values[pair[0]]), suffix),
                            (neck_index, between, neck_role),
                            (int(pair[1]), float(values[pair[1]]), suffix),
                        ],
                    )
                    candidates.append(
                        self._signal(
                            f"double_{suffix}",
                            direction,
                            0.68 + 0.12 * confirmed,
                            frame,
                            geometry,
                            family="multiple_top_bottom",
                            atr=atr,
                            confirmed=confirmed,
                            measurement_height=abs(
                                float(np.mean(values[pair])) - float(between)
                            ),
                            breakout_level=float(between),
                            false_break=false_break,
                        )
                    )
            if len(indexes) >= 3:
                triple = indexes[-3:]
                heights = values[triple]
                if _similar(heights, 0.08) and has_prior_trend(int(triple[0]), suffix):
                    neckline = (
                        float(np.min(low[triple[0] : triple[-1] + 1]))
                        if suffix == "top"
                        else float(np.max(high[triple[0] : triple[-1] + 1]))
                    )
                    if reversal_has_followthrough(triple, heights, neckline, suffix):
                        confirmed, false_break = self._breakout(
                            frame, neckline, direction, atr
                        )
                        geometry = PatternGeometry(
                            lines=[
                                LineGeometry(
                                    "neckline",
                                    int(triple[0]),
                                    neckline,
                                    int(triple[-1]),
                                    neckline,
                                )
                            ],
                            pivots=[
                                (int(index), float(values[index]), suffix)
                                for index in triple
                            ],
                        )
                        candidates.append(
                            self._signal(
                                f"triple_{suffix}",
                                direction,
                                0.72 + 0.12 * confirmed,
                                frame,
                                geometry,
                                family="multiple_top_bottom",
                                atr=atr,
                                confirmed=confirmed,
                                measurement_height=abs(float(np.mean(heights)) - neckline),
                                breakout_level=neckline,
                                false_break=false_break,
                                extra={
                                    "canonical_pattern": f"double_{suffix}",
                                    "canonical_variant": "multiple",
                                },
                            )
                        )

            # Search several recent consecutive triples instead of only the
            # final three extrema.  A legitimate head-and-shoulders pattern
            # may be followed by one or more rebound peaks before the current
            # bar; those rebounds make it a compound variant, not invisible.
            recent_indexes = indexes[-min(7, len(indexes)) :]
            for start_at in range(max(0, len(recent_indexes) - 2)):
                triple = recent_indexes[start_at : start_at + 3]
                if len(triple) < 3 or not has_prior_trend(int(triple[0]), suffix):
                    continue
                heights = values[triple]
                shoulder_tolerance = abs(heights[0] - heights[2]) / max(
                    abs(float(np.mean(heights[[0, 2]]))), 1e-12
                )
                middle_extreme = (
                    heights[1] > max(heights[0], heights[2]) * 1.03
                    if suffix == "top"
                    else heights[1] < min(heights[0], heights[2]) * 0.97
                )
                if shoulder_tolerance > 0.10 or not middle_extreme:
                    continue
                opposite_indexes = swings.troughs if suffix == "top" else swings.peaks
                left_inner = [
                    int(index)
                    for index in opposite_indexes
                    if triple[0] < index < triple[1]
                ]
                right_inner = [
                    int(index)
                    for index in opposite_indexes
                    if triple[1] < index < triple[2]
                ]
                if not left_inner or not right_inner:
                    continue
                if suffix == "top":
                    inner = [
                        min(left_inner, key=lambda index: low[index]),
                        min(right_inner, key=lambda index: low[index]),
                    ]
                    inner_values = low[inner]
                else:
                    inner = [
                        max(left_inner, key=lambda index: high[index]),
                        max(right_inner, key=lambda index: high[index]),
                    ]
                    inner_values = high[inner]
                slope, intercept, _ = _fit_line(np.asarray(inner), inner_values)
                projected_neckline = float(slope * (len(frame) - 1) + intercept)
                head_neckline = float(slope * int(triple[1]) + intercept)
                height = abs(float(heights[1]) - head_neckline)
                if height < max(atr * 2.5, abs(float(heights[1])) * 0.03):
                    continue
                confirmed, false_break = self._breakout(
                    frame, projected_neckline, direction, atr
                )
                shoulder_mean = float(np.mean(heights[[0, 2]]))
                shoulder_like = [
                    int(index)
                    for index in recent_indexes
                    if index != triple[1]
                    and abs(float(values[index]) - shoulder_mean)
                    / max(abs(shoulder_mean), 1e-12)
                    <= 0.14
                ]
                compound = len(shoulder_like) >= 3 or start_at + 3 < len(recent_indexes)
                geometry_pivots = [
                    (int(triple[0]), float(heights[0]), "left_shoulder"),
                    (int(inner[0]), float(inner_values[0]), "neckline_left"),
                    (int(triple[1]), float(heights[1]), "head"),
                    (int(inner[1]), float(inner_values[1]), "neckline_right"),
                    (int(triple[2]), float(heights[2]), "right_shoulder"),
                ]
                geometry = PatternGeometry(
                    lines=[
                        LineGeometry(
                            "neckline",
                            int(inner[0]),
                            float(inner_values[0]),
                            int(inner[1]),
                            float(inner_values[1]),
                        )
                    ],
                    arcs=[
                        _structure_arc(
                            np.asarray(triple),
                            np.asarray(heights),
                            "head_shoulders_bracket",
                        )
                    ],
                    pivots=geometry_pivots,
                )
                candidates.append(
                    self._signal(
                        f"{'compound_' if compound else ''}head_and_shoulders_{suffix}",
                        direction,
                        min(0.96, 0.80 + 0.04 * compound + 0.12 * confirmed),
                        frame,
                        geometry,
                        family="head_and_shoulders",
                        atr=atr,
                        confirmed=confirmed,
                        measurement_height=height,
                        breakout_level=projected_neckline,
                        false_break=false_break,
                        extra={
                            "canonical_pattern": f"head_and_shoulders_{suffix}",
                            "canonical_variant": "compound" if compound else "standard",
                            "neckline_projection_at_latest": projected_neckline,
                            "structure_points": {
                                "left_shoulder": int(triple[0]),
                                "head": int(triple[1]),
                                "right_shoulder": int(triple[2]),
                            },
                            "structure_labels": {
                                "left_shoulder": "左肩",
                                "head": "头",
                                "right_shoulder": "右肩",
                            },
                        },
                    )
                )

        x = np.arange(len(frame), dtype=float)
        normalized = (close - close.mean()) / max(close.std(), 1e-12)
        quadratic = np.polyfit(x / max(len(frame) - 1, 1), normalized, 2)
        fitted = np.polyval(quadratic, x / max(len(frame) - 1, 1))
        total = float(np.sum((normalized - normalized.mean()) ** 2))
        r2 = 1.0 - float(np.sum((normalized - fitted) ** 2)) / total if total else 0
        vertex_norm = float(-quadratic[1] / (2.0 * quadratic[0])) if quadratic[0] else 0.5
        vertex = int(np.clip(round(vertex_norm * max(len(frame) - 1, 1)), 0, len(frame) - 1))
        flank = max(6, min(len(frame) // 4, 20))
        left_edge = max(0, vertex - flank)
        right_edge = min(len(frame) - 1, vertex + flank)
        left_slope, _, left_r2 = _fit_line(
            np.arange(left_edge, vertex + 1), close[left_edge : vertex + 1]
        )
        right_slope, _, right_r2 = _fit_line(
            np.arange(vertex, right_edge + 1), close[vertex : right_edge + 1]
        )
        move_threshold = max(atr * 2.0, abs(close[vertex]) * 0.03)
        clear_flanks = (
            abs(close[vertex] - close[left_edge]) >= move_threshold
            and abs(close[right_edge] - close[vertex]) >= move_threshold
        )
        correct_flanks = (
            left_slope < 0 < right_slope
            if quadratic[0] > 0
            else left_slope > 0 > right_slope
        )
        if (
            abs(quadratic[0]) > 0.7
            and r2 > 0.55
            and 0.20 <= vertex_norm <= 0.80
            and correct_flanks
            and clear_flanks
            and min(left_r2, right_r2) >= 0.30
        ):
            bottom = quadratic[0] > 0
            direction = Direction.BULLISH if bottom else Direction.BEARISH
            boundary = float(np.max(close) if bottom else np.min(close))
            confirmed, false_break = self._breakout(frame, boundary, direction, atr)
            geometry = PatternGeometry(
                arcs=[
                    ArcGeometry(
                        "rounding_bottom" if bottom else "rounding_top",
                        0,
                        len(frame) - 1,
                        tuple(float(value) for value in np.polyfit(x, close, 2)),
                    )
                ]
            )
            candidates.append(
                self._signal(
                    "rounding_bottom" if bottom else "rounding_top",
                    direction,
                    min(0.88, 0.55 + r2 * 0.3),
                    frame,
                    geometry,
                    family="rounding",
                    atr=atr,
                    confirmed=confirmed,
                    breakout_level=boundary,
                    false_break=false_break,
                    extra={
                        "vertex_index": vertex,
                        "pre_reversal_slope": left_slope,
                        "post_reversal_slope": right_slope,
                        "pre_reversal_move": float(close[vertex] - close[left_edge]),
                        "post_reversal_move": float(close[right_edge] - close[vertex]),
                    },
                )
            )

        extreme_bottom = int(np.argmin(low))
        extreme_top = int(np.argmax(high))
        for extreme, bottom in ((extreme_bottom, True), (extreme_top, False)):
            if len(frame) * 0.2 < extreme < len(frame) * 0.8:
                flank = max(8, min(len(frame) // 4, 24))
                left_start = max(0, extreme - flank)
                right_end = min(len(frame) - 1, extreme + flank)
                left = close[left_start : extreme + 1]
                right = close[extreme : right_end + 1]
                left_slope, _, left_r2 = _fit_line(
                    np.arange(left_start, extreme + 1), left
                )
                right_slope, _, right_r2 = _fit_line(
                    np.arange(extreme, right_end + 1), right
                )
                extreme_value = low[extreme] if bottom else high[extreme]
                tolerance = max(atr * 1.25, abs(extreme_value) * 0.015)
                nearby = (
                    np.flatnonzero(low <= extreme_value + tolerance)
                    if bottom
                    else np.flatnonzero(high >= extreme_value - tolerance)
                )
                # A V/尖顶 is a conspicuous single pivot.  Several separated
                # visits to the same price zone are a base/rectangle, not a
                # sharp reversal.
                clusters = 0
                previous_index: int | None = None
                for index in nearby:
                    if previous_index is None or int(index) - previous_index > 3:
                        clusters += 1
                    previous_index = int(index)
                unique_extreme = clusters == 1 and len(nearby) <= max(5, len(frame) // 12)
                move_threshold = max(atr * 2.0, abs(close[extreme]) * 0.03)
                clear_move = (
                    abs(close[extreme] - close[left_start]) >= move_threshold
                    and abs(close[right_end] - close[extreme]) >= move_threshold
                )
                correct = (
                    left_slope < 0 < right_slope
                    if bottom
                    else left_slope > 0 > right_slope
                )
                if correct and min(left_r2, right_r2) > 0.55 and unique_extreme and clear_move:
                    geometry = PatternGeometry(
                        lines=[
                            LineGeometry(
                                "left_leg",
                                left_start,
                                float(close[left_start]),
                                extreme,
                                float(close[extreme]),
                            ),
                            LineGeometry(
                                "right_leg",
                                extreme,
                                float(close[extreme]),
                                right_end,
                                float(close[right_end]),
                            ),
                        ],
                        pivots=[(extreme, float(close[extreme]), "vertex")],
                    )
                    candidates.append(
                        self._signal(
                            "v_bottom" if bottom else "inverted_v_top",
                            Direction.BULLISH if bottom else Direction.BEARISH,
                            0.72,
                            frame,
                            geometry,
                            family="v_reversal",
                            atr=atr,
                            confirmed=True,
                            extra={
                                "vertex_index": extreme,
                                "unique_extreme": True,
                                "pre_reversal_move": float(close[extreme] - close[left_start]),
                                "post_reversal_move": float(close[right_end] - close[extreme]),
                            },
                        )
                    )

        gap_up = low[1:] > high[:-1] * 1.001
        gap_down = high[1:] < low[:-1] * 0.999
        up_indexes = np.flatnonzero(gap_up) + 1
        down_indexes = np.flatnonzero(gap_down) + 1
        if len(up_indexes) and len(down_indexes):
            pairs: list[tuple[int, int, bool]] = []
            for first in up_indexes:
                for second in down_indexes:
                    if first < second <= first + max(12, len(frame) // 3):
                        pairs.append((int(first), int(second), True))
            for first in down_indexes:
                for second in up_indexes:
                    if first < second <= first + max(12, len(frame) // 3):
                        pairs.append((int(first), int(second), False))
            for first, second, top in sorted(pairs, key=lambda item: item[1], reverse=True):
                island_high = float(np.max(high[first:second]))
                island_low = float(np.min(low[first:second]))
                if top:
                    separated = island_low > max(float(high[first - 1]), float(high[second])) * 1.001
                    breakout_level = island_low
                    confirmation = breakout_confirmation(
                        close,
                        breakout_level,
                        "bearish",
                        start=second,
                        timeframe=self.timeframe,
                        open_=frame["open"].to_numpy(float),
                        high=high,
                        low=low,
                    )
                    followed_through = bool(confirmation["breakout_confirmed"])
                    invalidated = bool(np.any(close[second:] > island_high))
                else:
                    separated = island_high < min(float(low[first - 1]), float(low[second])) * 0.999
                    breakout_level = island_high
                    confirmation = breakout_confirmation(
                        close,
                        breakout_level,
                        "bullish",
                        start=second,
                        timeframe=self.timeframe,
                        open_=frame["open"].to_numpy(float),
                        high=high,
                        low=low,
                    )
                    followed_through = bool(confirmation["breakout_confirmed"])
                    invalidated = bool(np.any(close[second:] < island_low))
                # A valid two-gap island is still emitted as a candidate when
                # the textbook post-breakout confirmation has not happened.
                # Only structural failure (no separation or a return through
                # the island) suppresses it entirely.
                if not separated or invalidated:
                    continue
                geometry = PatternGeometry(
                    pivots=[
                        (first, float(close[first]), "island_start"),
                        (second, float(close[second]), "island_end"),
                    ]
                )
                candidates.append(
                    self._signal(
                        "island_reversal_top" if top else "island_reversal_bottom",
                        Direction.BEARISH if top else Direction.BULLISH,
                        0.84,
                        frame,
                        geometry,
                        family="island",
                        atr=atr,
                        confirmed=followed_through,
                        extra={
                            "gap_pair_confirmed": True,
                            "breakout_level": breakout_level,
                            "confirmation_start": second,
                            "breakout_confirmation_mode": confirmation[
                                "breakout_confirmation_mode"
                            ],
                            "confirmation_streak_bars": confirmation[
                                "confirmation_streak_bars"
                            ],
                        },
                    )
                )
                break
        cup = self._cup_with_handle_candidate(frame, atr)
        if cup is not None:
            candidates.append(cup)
        return candidates

    def _consolidation_candidates(
        self,
        frame: pd.DataFrame,
        swings: _Swings,
        atr: float,
    ) -> list[DetectedSignal]:
        dormant = self._dormant_bottom_candidate(frame, atr)
        if len(swings.peaks) < 2 or len(swings.troughs) < 2:
            return [dormant] if dormant is not None else []
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)
        p = swings.peaks[-min(4, len(swings.peaks)) :]
        t = swings.troughs[-min(4, len(swings.troughs)) :]
        upper_slope, upper_intercept, upper_r2 = _fit_line(p, high[p])
        lower_slope, lower_intercept, lower_r2 = _fit_line(t, low[t])
        upper_end = upper_slope * (len(frame) - 1) + upper_intercept
        lower_end = lower_slope * (len(frame) - 1) + lower_intercept
        structure_start = min(int(p[0]), int(t[0]))
        structure_end = max(int(p[-1]), int(t[-1]))
        start_width = (
            upper_slope * structure_start
            + upper_intercept
            - (lower_slope * structure_start + lower_intercept)
        )
        structure_upper_end = upper_slope * structure_end + upper_intercept
        structure_lower_end = lower_slope * structure_end + lower_intercept
        end_width = structure_upper_end - structure_lower_end
        boundary_span = max(1, structure_end - structure_start)
        upper_move = abs(upper_slope) * max(1, int(p[-1]) - int(p[0]))
        lower_move = abs(lower_slope) * max(1, int(t[-1]) - int(t[0]))
        flat_move_limit = max(atr * 0.75, float(close.mean()) * 0.015)
        directional_move_min = max(atr * 1.25, float(close.mean()) * 0.018)
        upper_fit = upper_slope * p + upper_intercept
        lower_fit = lower_slope * t + lower_intercept
        upper_residual_atr = float(np.mean(np.abs(high[p] - upper_fit))) / max(atr, 1e-12)
        lower_residual_atr = float(np.mean(np.abs(low[t] - lower_fit))) / max(atr, 1e-12)
        ordered_swings = sorted(
            [(int(index), "peak") for index in p]
            + [(int(index), "trough") for index in t]
        )
        wave_count = sum(
            1
            for left, right in zip(ordered_swings, ordered_swings[1:])
            if left[1] != right[1]
        )
        convergence_ratio = (
            (start_width - end_width) / max(abs(start_width), atr)
            if start_width
            else 0.0
        )
        geometry = PatternGeometry(
            lines=[
                LineGeometry(
                    "resistance",
                    int(p[0]),
                    float(upper_slope * int(p[0]) + upper_intercept),
                    int(p[-1]),
                    float(upper_slope * int(p[-1]) + upper_intercept),
                ),
                LineGeometry(
                    "support",
                    int(t[0]),
                    float(lower_slope * int(t[0]) + lower_intercept),
                    int(t[-1]),
                    float(lower_slope * int(t[-1]) + lower_intercept),
                ),
            ],
            pivots=[
                *[(int(index), float(high[index]), "peak") for index in p],
                *[(int(index), float(low[index]), "trough") for index in t],
            ],
        )
        candidates: list[DetectedSignal] = []
        if dormant is not None:
            candidates.append(dormant)
        pattern = ""
        direction = Direction.NEUTRAL
        triangle_ready = (
            len(p) >= 3
            and len(t) >= 3
            and wave_count >= 5
            and boundary_span >= 20
            and start_width >= atr * 3.0
            and end_width > 0.0
            and upper_residual_atr <= 1.25
            and lower_residual_atr <= 1.25
        )
        box_ready = (
            len(p) >= 2
            and len(t) >= 2
            and wave_count >= 3
            and boundary_span >= 12
            and start_width >= atr * 2.0
            and end_width > 0.0
            and upper_residual_atr <= 1.0
            and lower_residual_atr <= 1.0
        )
        upper_is_flat = upper_move <= flat_move_limit
        lower_is_flat = lower_move <= flat_move_limit
        if (
            triangle_ready
            and upper_is_flat
            and not lower_is_flat
            and lower_slope > 0.0
            and lower_move >= directional_move_min
            and lower_r2 >= 0.35
            and end_width <= start_width * 0.90
        ):
            pattern, direction = "ascending_triangle", Direction.BULLISH
        elif (
            triangle_ready
            and lower_is_flat
            and not upper_is_flat
            and upper_slope < 0.0
            and upper_move >= directional_move_min
            and upper_r2 >= 0.35
            and end_width <= start_width * 0.90
        ):
            pattern, direction = "descending_triangle", Direction.BEARISH
        elif (
            triangle_ready
            and not upper_is_flat
            and not lower_is_flat
            and upper_slope < 0.0
            and lower_slope > 0.0
            and end_width < start_width
            and convergence_ratio >= 0.10
        ):
            pattern = "symmetrical_triangle"
            direction = (
                Direction.BULLISH
                if close[-1] > upper_end
                else Direction.BEARISH
                if close[-1] < lower_end
                else Direction.NEUTRAL
            )
        elif (
            triangle_ready
            and not upper_is_flat
            and not lower_is_flat
            and upper_slope > 0.0
            and lower_slope > 0.0
            and end_width < start_width
        ):
            pattern, direction = "rising_wedge", Direction.BEARISH
        elif (
            triangle_ready
            and not upper_is_flat
            and not lower_is_flat
            and upper_slope < 0.0
            and lower_slope < 0.0
            and end_width < start_width
        ):
            pattern, direction = "falling_wedge", Direction.BULLISH
        elif (
            box_ready
            and upper_is_flat
            and lower_is_flat
        ):
            pattern, direction = "rectangle", Direction.NEUTRAL
        elif triangle_ready and end_width > start_width * 1.25:
            pattern, direction = "broadening_triangle", Direction.BEARISH
        if pattern:
            if direction == Direction.NEUTRAL:
                confirmed = close[-1] > upper_end + max(
                    atr * 0.5, upper_end * 0.01
                ) or close[-1] < lower_end - max(atr * 0.5, lower_end * 0.01)
                if confirmed:
                    direction = (
                        Direction.BULLISH
                        if close[-1] > upper_end
                        else Direction.BEARISH
                    )
                false_break = False
                breakout_level = (
                    upper_end if direction == Direction.BULLISH else lower_end
                )
            else:
                breakout_level = (
                    upper_end if direction == Direction.BULLISH else lower_end
                )
                confirmed, false_break = self._breakout(
                    frame, float(breakout_level), direction, atr
                )
            candidates.append(
                self._signal(
                    pattern,
                    direction,
                    0.56 + 0.16 * min(upper_r2, lower_r2) + 0.1 * confirmed,
                    frame,
                    geometry,
                    family="consolidation",
                    atr=atr,
                    confirmed=bool(confirmed),
                    measurement_height=max(start_width, end_width),
                    breakout_level=float(breakout_level),
                    false_break=bool(false_break),
                    extra={
                        "wave_count": wave_count,
                        "boundary_touch_count": len(p) + len(t),
                        "start_width": float(start_width),
                        "end_width": float(end_width),
                        "convergence_ratio": float(convergence_ratio),
                        "upper_boundary_move": float(upper_move),
                        "lower_boundary_move": float(lower_move),
                        "upper_boundary_residual_atr": upper_residual_atr,
                        "lower_boundary_residual_atr": lower_residual_atr,
                        # Only the canonical symmetrical-triangle detector
                        # owns this Chinese display label.  The surrounding
                        # branch also produces rising/falling wedges and
                        # rectangles; letting the override leak into those
                        # patterns would mislabel a falling wedge as a
                        # triangle in the mobile view.
                        "display_name_override": (
                            ("收敛三角形突破" if confirmed else "收敛三角形")
                            if pattern == "symmetrical_triangle"
                            else None
                        ),
                    },
                )
            )

        impulse_end = max(5, len(frame) // 4)
        impulse = close[impulse_end] - close[0]
        channel_slope, _, channel_r2 = _fit_line(
            np.arange(len(frame) - impulse_end),
            close[impulse_end:],
        )
        if abs(impulse) >= atr * 4 and channel_r2 >= 0.25:
            if (
                impulse > 0
                and channel_slope <= 0
                and abs(channel_slope) < abs(impulse) / len(frame)
            ):
                flag, direction = "ascending_flag", Direction.BULLISH
            elif (
                impulse < 0
                and channel_slope >= 0
                and abs(channel_slope) < abs(impulse) / len(frame)
            ):
                flag, direction = "descending_flag", Direction.BEARISH
            else:
                flag = ""
            if flag:
                level = float(
                    frame["high"].iloc[impulse_end:].max()
                    if direction == Direction.BULLISH
                    else frame["low"].iloc[impulse_end:].min()
                )
                confirmed, false_break = self._breakout(frame, level, direction, atr)
                candidates.append(
                    self._signal(
                        flag,
                        direction,
                        0.66 + 0.12 * confirmed,
                        frame,
                        geometry,
                        family="flag",
                        atr=atr,
                        confirmed=confirmed,
                        measurement_height=abs(float(impulse)),
                        breakout_level=level,
                        false_break=false_break,
                    )
                )
        return candidates

    def detect(self, data: pd.DataFrame) -> list[DetectedSignal]:
        """Detect sparse, mutually-consistent chart candidates.

        There is no fixed weekly/daily count cap.  The gate applies a rolling
        confidence floor (the second and third candidates must be much
        stronger than the first) and mutual exclusion, which naturally keeps
        the result sparse while preserving genuinely separate structures.
        """
        frame = normalize_ohlcv(data)
        if len(frame) < self.min_window:
            return []
        all_candidates: list[DetectedSignal] = []
        lengths = sorted(
            {
                self.min_window,
                min(45, self.max_window),
                min(60, self.max_window),
                min(80, self.max_window),
                min(100, self.max_window),
                min(len(frame), self.max_window),
            }
        )
        for length in lengths:
            if length > len(frame) or length < self.min_window:
                continue
            window = frame.iloc[-length:].copy()
            window.attrs["_offset"] = len(frame) - length
            atr = self._atr(window)
            swings = self._swings(window, atr)
            all_candidates.extend(self._reversal_candidates(window, swings, atr))
            all_candidates.extend(self._consolidation_candidates(window, swings, atr))
        return self.gate.filter_chart_signals(
            all_candidates,
            len(frame),
            timeframe=self.timeframe,
        )


def plot_pattern_geometry(
    axes: Any,
    signal: DetectedSignal,
    *,
    color: str = "#8B1E3F",
) -> None:
    """Draw adaptive straight lines, arcs, and pivots on a Matplotlib axes."""
    geometry = signal.geometry
    if geometry is None:
        return
    for line in geometry.lines:
        axes.plot(
            [line.x1, line.x2],
            [line.y1, line.y2],
            color=color,
            linestyle="--" if line.style == "dashed" else "-",
            linewidth=1.4,
            alpha=0.9,
        )
    for arc in geometry.arcs:
        x = np.linspace(arc.start, arc.end, 80)
        y = np.polyval(np.asarray(arc.coefficients, dtype=float), x)
        axes.plot(x, y, color=color, linewidth=1.6, alpha=0.9)
    if geometry.pivots:
        axes.scatter(
            [item[0] for item in geometry.pivots],
            [item[1] for item in geometry.pivots],
            s=18,
            color=color,
            zorder=6,
        )
