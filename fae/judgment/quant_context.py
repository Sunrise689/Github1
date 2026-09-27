"""Backward-looking quantitative context for candlestick confirmation.

This module is the bridge between the FAE Phase-1 statistical layer and the
objective candlestick detectors.  It deliberately avoids fixed statements
such as "three bars always means confirmation": the observation window is
scaled from the available 100/120-bar context and trend strength is measured
relative to the recent return distribution.

The returned trend state is descriptive evidence only.  It does not produce a
buy/sell decision.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


TrendState = Literal["uptrend", "downtrend", "neutral"]


def adaptive_confirmation_bars(
    observations: int,
    *,
    lookback: int = 120,
    minimum: int = 2,
    maximum: int = 8,
) -> int:
    """Return a deterministic confirmation window from the quant context.

    For a 100/120-bar context this normally yields five bars.  The bounds keep
    the detector usable on shorter samples while preventing a single candle or
    an excessively long future window from deciding the pattern.
    """

    if observations < 1:
        raise ValueError("observations must be positive")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid confirmation window bounds")
    effective = max(1, min(int(observations), int(lookback)))
    scaled = int(round(np.sqrt(effective) / 2.0))
    return max(minimum, min(maximum, scaled))


def adaptive_trend_context(
    close: pd.Series,
    *,
    lookback: int = 120,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Classify each bar using rolling return quantiles and slope direction.

    A bar is an ``uptrend`` candidate when its trailing return is positive and
    lies above the prior-window 75th percentile; ``downtrend`` is the mirror
    condition at the 25th percentile.  The percentile is shifted by one bar so
    the current observation is not used to define its own threshold.  Bars
    without enough history are ``neutral``.
    """

    values = pd.to_numeric(close, errors="coerce").astype(float)
    if values.empty or values.isna().any() or (values <= 0).any():
        raise ValueError("close must contain finite positive values")
    if lookback < 20:
        raise ValueError("lookback must be at least 20")

    horizon = int(horizon or max(5, min(20, lookback // 6)))
    if horizon < 2:
        raise ValueError("horizon must be at least 2")

    trailing_return = values.pct_change(horizon)
    log_slope = np.log(values).diff(horizon) / float(horizon)
    min_periods = max(10, min(lookback // 4, len(values)))
    prior_returns = trailing_return.shift(1)
    upper = prior_returns.rolling(lookback, min_periods=min_periods).quantile(0.75)
    lower = prior_returns.rolling(lookback, min_periods=min_periods).quantile(0.25)

    up = (trailing_return > 0) & (trailing_return >= upper) & (log_slope > 0)
    down = (trailing_return < 0) & (trailing_return <= lower) & (log_slope < 0)
    state = pd.Series("neutral", index=values.index, dtype="object")
    state.loc[up.fillna(False)] = "uptrend"
    state.loc[down.fillna(False)] = "downtrend"

    return pd.DataFrame(
        {
            "state": state,
            "trailing_return": trailing_return,
            "log_slope": log_slope,
            "upper_return_quantile": upper,
            "lower_return_quantile": lower,
            "horizon": horizon,
            "lookback": lookback,
        },
        index=values.index,
    )


if __name__ == "__main__":
    prices = pd.Series(np.linspace(100.0, 140.0, 140))
    context = adaptive_trend_context(prices)
    print(adaptive_confirmation_bars(len(prices)))
    print(context.tail(1).to_dict("records")[0])
