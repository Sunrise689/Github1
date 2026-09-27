"""Context-sensitive candlestick validation used by the FAE gate.

The legacy detector intentionally emits relaxed candidates.  The FAE layer
must still reject a label when the candle geometry contradicts the textbook
definition.  Keeping this check here leaves ``pattern_core_v7.py`` untouched
while making the judgment pipeline stricter and auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pattern_registry import canonicalize_pattern_id
from .schemas import DetectedSignal


def _column(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def _is_declining_background(close: pd.Series, start: int) -> bool:
    """Return whether the bars before a candidate form a material decline.

    ``倒三阳`` is a counter-trend-looking bullish sequence inside a decline;
    a short pullback inside a rising market must not qualify.  The local
    window is therefore checked first and, once enough history exists, a
    second 30-bar window must also point down.
    """
    if start < 5:
        return False
    lookback = min(20, start)
    segment = close.iloc[start - lookback : start].dropna()
    if len(segment) < 5 or float(segment.iloc[0]) <= 0:
        return False
    x = np.arange(len(segment), dtype=float)
    slope = float(np.polyfit(x, segment.to_numpy(float), 1)[0])
    normalized_slope = slope / float(segment.mean())
    cumulative_return = float(segment.iloc[-1] / segment.iloc[0] - 1.0)
    # A short, noisy pullback is not enough.  The threshold is deliberately
    # modest because the first layer supplies the broader statistical gate.
    if not (normalized_slope < -0.00025 and cumulative_return <= -0.005):
        return False
    if start < 30:
        return True
    broad = close.iloc[start - 30 : start].dropna()
    if len(broad) < 15 or float(broad.iloc[0]) <= 0:
        return False
    broad_x = np.arange(len(broad), dtype=float)
    broad_slope = float(np.polyfit(broad_x, broad.to_numpy(float), 1)[0])
    broad_return = float(broad.iloc[-1] / broad.iloc[0] - 1.0)
    return broad_slope / float(broad.mean()) < -0.00008 and broad_return <= -0.003


def _valid_three_reverse_bullish(
    frame: pd.DataFrame,
    start: int,
    end: int,
) -> tuple[bool, str]:
    """Validate the strict body/open/declining-background contract.

    Accepted forms are:

    * classic: three bullish bodies, each new open below the prior close and
      each close below the prior close;
    * relaxed: two such bullish bodies;
    * textbook variant: bullish, bearish, bullish, where the second bullish
      body also opens and closes below the first bullish close.
    """
    open_price = _column(frame, "open", "Open")
    close = _column(frame, "close", "Close")
    if open_price is None or close is None:
        return True, "缺少OHLC列，保留上游候选"
    start = max(0, min(len(frame) - 1, int(start)))
    end = max(start, min(len(frame) - 1, int(end)))
    if not _is_declining_background(close, start):
        return False, "缺少可量化确认的下降背景"
    opens = open_price.iloc[start : end + 1].to_numpy(float)
    closes = close.iloc[start : end + 1].to_numpy(float)
    if len(opens) < 2 or np.isnan(opens).any() or np.isnan(closes).any():
        return False, "K线数据不完整"

    bullish = closes > opens
    centers = (opens + closes) / 2.0
    if len(opens) >= 3 and bool(np.all(bullish[:3])):
        classic = bool(
            opens[1] < closes[0]
            and opens[2] < closes[1]
            and closes[1] < closes[0]
            and closes[2] < closes[1]
            and centers[1] < centers[0]
            and centers[2] < centers[1]
        )
        if classic:
            return True, "经典三阳线：连续低开且收盘递降"

    if len(opens) == 2 and bool(np.all(bullish[:2])):
        relaxed_two = bool(
            opens[1] < closes[0]
            and closes[1] < closes[0]
            and centers[1] < centers[0]
        )
        if relaxed_two:
            return True, "放宽两阳变体：低开且收盘相对前阳递降"

    if len(opens) >= 3:
        variant = bool(
            bullish[0]
            and not bullish[1]
            and bullish[2]
            and opens[2] < closes[0]
            and closes[2] < closes[0]
            and centers[2] < centers[0]
        )
        if variant:
            return True, "放宽变体：阳阴阳且后阳低开、收盘低于前阳"

    return False, "未满足低开且收盘递降的倒三阳实体条件"


def filter_contextual_candlestick_signals(
    frame: pd.DataFrame,
    signals: Sequence[DetectedSignal],
) -> tuple[list[DetectedSignal], list[dict[str, Any]]]:
    """Reject contextually impossible candlestick candidates before scoring."""
    valid: list[DetectedSignal] = []
    rejected: list[dict[str, Any]] = []
    for signal in signals:
        pattern_id = canonicalize_pattern_id(str(signal.pattern))
        if pattern_id != "three_reverse_bullish":
            valid.append(signal)
            continue
        ok, reason = _valid_three_reverse_bullish(frame, signal.start, signal.end)
        if ok:
            signal.metadata["context_validation"] = reason
            valid.append(signal)
        else:
            rejected.append(
                {
                    "pattern": pattern_id,
                    "reasons": [reason],
                    "stage": "context_validation",
                    "start": signal.start,
                    "end": signal.end,
                }
            )
    return valid, rejected
