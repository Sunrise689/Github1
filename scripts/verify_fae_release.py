"""Read-only deployment verifier for the FAE detector-to-display pipeline.

The default run uses deterministic local OHLCV data.  ``--network`` audits
the ten product-review symbols without writing previews, JSON manifests, or
any other sample artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fae.judgment import (  # noqa: E402
    JudgmentEngine,
    Timeframe,
    canonical_ids,
    from_v7_events,
)
from kline_pattern_report import (  # noqa: E402
    fetch_tencent_daily,
    resample_monthly,
    resample_weekly,
)
from pattern_core_v7 import detect_all  # noqa: E402


REVIEW_SYMBOLS = (
    "sh000001",
    "sh688981",
    "sz300750",
    "hk00700",
    "hk09988",
    "usAAPL",
    "usNVDA",
    "usTSLA",
    "usJPM",
    "usSPY",
)


def deterministic_daily(rows: int = 720) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.045 * x + 6.0 * np.sin(x / 19.0) + 2.2 * np.sin(x / 6.0)
    open_ = close - 0.45 * np.cos(x / 8.0)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=rows, freq="B"),
            "Open": open_,
            "High": np.maximum(open_, close) + 1.0,
            "Low": np.minimum(open_, close) - 1.0,
            "Close": close,
            "Volume": 1_000_000.0 + (x % 23) * 17_000.0,
        }
    )


def period_frames(daily: pd.DataFrame) -> dict[str, tuple[Timeframe, pd.DataFrame]]:
    return {
        "M": (Timeframe.MONTHLY, resample_monthly(daily).tail(72).reset_index(drop=True)),
        "W": (Timeframe.WEEKLY, resample_weekly(daily).tail(96).reset_index(drop=True)),
        "D": (Timeframe.DAILY, daily.tail(120).reset_index(drop=True)),
    }


def audit_period(
    engine: JudgmentEngine,
    frame: pd.DataFrame,
    timeframe: Timeframe,
    *,
    symbol: str,
) -> dict[str, Any]:
    if len(frame) < 3:
        raise AssertionError(f"{symbol}/{timeframe.value}: fewer than three bars")
    diagnostics: list[dict[str, Any]] = []
    events = detect_all(frame, diagnostics=diagnostics, strict=True)
    if diagnostics:
        raise AssertionError(f"{symbol}/{timeframe.value}: detector diagnostics={diagnostics}")
    signals = from_v7_events(events, timeframe=timeframe)
    result = engine.evaluate(
        frame,
        signals,
        timeframe=timeframe,
        context={"symbol": symbol},
    )
    display = result["display"]
    all_signals = display["all"]
    unknown_ids = sorted(
        {
            str(signal.get("pattern_id") or "")
            for signal in all_signals
            if str(signal.get("pattern_id") or "") not in canonical_ids()
        }
    )
    if unknown_ids:
        raise AssertionError(
            f"{symbol}/{timeframe.value}: non-canonical pattern IDs={unknown_ids!r}"
        )
    for signal in all_signals:
        start = int(signal["start"])
        end = int(signal["end"])
        if not 0 <= start <= end < len(frame):
            raise AssertionError(
                f"{symbol}/{timeframe.value}: invalid span {start}..{end}/{len(frame)}"
            )
        geometry = signal.get("geometry") or {}
        for line in geometry.get("lines", []):
            if not start <= int(line["x1"]) <= end or not start <= int(line["x2"]) <= end:
                raise AssertionError(
                    f"{symbol}/{timeframe.value}: geometry outside semantic span"
                )
    accepted = int(result["quantitative_filter"]["accepted_count"])
    if accepted and not display["default"]:
        raise AssertionError(
            f"{symbol}/{timeframe.value}: accepted signals exist but display.default is empty"
        )
    return {
        "bars": len(frame),
        "detector_events": len(events),
        "accepted": accepted,
        "default": len(display["default"]),
        "default_patterns": [item.get("pattern_id") for item in display["default"]],
        "chart_patterns": [
            item.get("pattern_id")
            for item in display["all"]
            if item.get("kind") == "chart"
        ],
        "recent_patterns": [item.get("pattern_id") for item in display["recent"]],
        "winner": display["summary"].get("winner_pattern_id"),
        "diagnostics": len(diagnostics),
    }


def audit_symbol(symbol: str, daily: pd.DataFrame) -> dict[str, Any]:
    engine = JudgmentEngine()
    return {
        period: audit_period(engine, frame, timeframe, symbol=symbol)
        for period, (timeframe, frame) in period_frames(daily).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true", help="audit the ten review symbols")
    parser.add_argument("--years", type=int, default=3, help="network history years")
    parser.add_argument("--symbols", nargs="*", default=None, help="override network symbols")
    args = parser.parse_args()

    symbols = tuple(args.symbols or REVIEW_SYMBOLS) if args.network else ("deterministic",)
    report: dict[str, Any] = {
        "mode": "network" if args.network else "offline",
        "sample_artifacts_written": False,
        "symbols": {},
        "failures": [],
    }
    for symbol in symbols:
        try:
            daily = (
                fetch_tencent_daily(symbol, years=max(1, int(args.years)))
                if args.network
                else deterministic_daily()
            )
            report["symbols"][symbol] = audit_symbol(symbol, daily)
        except Exception as exc:  # command-line audit must aggregate all symbols
            report["failures"].append(
                {
                    "symbol": symbol,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    report["ok"] = not report["failures"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
