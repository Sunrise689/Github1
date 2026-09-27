"""Generate offline FAE M/W/D preview charts for the Shanghai Composite.

This is a test/visual-audit script only.  It does not import or modify the
WeChat mini-program backend/frontend.  It fetches local-market daily bars via
the existing Tencent data helper, resamples them, runs the FAE judgment layer,
and writes three PNGs plus a compact JSON summary under ``fae/preview``.

Usage (from the project root)::

    python scripts/run_fae_sh000001_preview.py
    python scripts/run_fae_sh000001_preview.py --symbol sh688981 --name 中芯国际 --output-stem smic

The script deliberately keeps the raw FAE result in the JSON summary so the
user can inspect what was foregrounded, what remained a candidate, and what
was de-emphasized by the timeframe policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fae.judgment import JudgmentEngine, Timeframe, from_v7_events, get_pattern
from kline_pattern_report import fetch_tencent_daily, resample_monthly, resample_weekly
from pattern_core_v7 import detect_all


# PowerShell on some Windows installations still exposes a legacy code page.
# Keep the JSON summary and progress output readable without changing any
# project-wide encoding settings.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OUTPUT_DIR = PROJECT_ROOT / "fae" / "preview"
BAR_LIMIT = 120

PROFILE_KINDS = {
    Timeframe.MONTHLY: {"combination"},
    Timeframe.WEEKLY: {"simple", "combination"},
    Timeframe.DAILY: {"combination", "trend", "chart"},
}

PREVIEW_QUOTAS = {
    # The chart remains an audit view, not a raw detector dump.  Quotas keep
    # the figure readable while ensuring trend candidates such as 冉冉上升 /
    # 升势受阻 are not silently lost behind a higher-priority chart signal.
    Timeframe.MONTHLY: {"chart": 1, "combination": 8, "simple": 6, "trend": 4},
    Timeframe.WEEKLY: {"chart": 4, "combination": 8, "simple": 8, "trend": 5},
    Timeframe.DAILY: {"chart": 5, "combination": 10, "simple": 8, "trend": 6},
}

# The batch review requested by the product owner is deliberately different
# from the normal preview.  Monthly bars are sparse, so the audit view keeps
# almost every non-duplicate representative (including recent candlesticks and
# signals that were later suppressed by a larger pattern).  Weekly and daily
# views still have finite budgets because their raw detector output can be
# visually overwhelming.  The raw result remains in JSON in all cases.
AUDIT_QUOTAS = {
    Timeframe.MONTHLY: {
        "chart": 40,
        "combination": 60,
        "simple": 60,
        "trend": 40,
        "gap": 30,
        "structure": 30,
    },
    Timeframe.WEEKLY: {
        "chart": 16,
        "combination": 24,
        "simple": 18,
        "trend": 14,
        "gap": 12,
        "structure": 12,
    },
    Timeframe.DAILY: {
        "chart": 10,
        "combination": 18,
        "simple": 12,
        "trend": 10,
        "gap": 8,
        "structure": 8,
    },
}

RAW_DISPLAY_NAMES = {
    "dark_cloud": "乌云盖顶",
    "piercing": "刺透形态",
    "steady_decline": "稳步下跌",
    "steady_rising": "稳步上涨",
    "ran_ran_rising": "冉冉上升",
    "acceleration": "加速趋势",
    "看涨孕线": "看涨孕线",
    "看跌孕线": "看跌孕线",
    "长十字线": "长腿十字星",
    "十字孕线": "十字孕线",
    "搓揉线": "搓揉线",
    "螺旋桨": "螺旋桨",
    "倾盆大雨": "倾盆大雨",
    "徐缓rising": "徐缓上升",
    "徐缓decline": "徐缓下降",
    "xuhuan_rising": "徐缓上升",
    "xuhuan_decline": "徐缓下降",
    "slow_rise": "徐缓上升",
    "slow_decline": "徐缓下降",
}

DISPLAY_KIND_NAMES = {
    "simple": "简单K线",
    "combination": "组合K线",
    "trend": "趋势结构",
    "chart": "技术图形",
    "gap": "缺口结构",
    "structure": "结构/趋势线",
}


def _configure_font() -> None:
    """Prefer a Windows CJK font so generated labels remain readable."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _label_font() -> FontProperties:
    """Return an explicit CJK font for annotation labels.

    Matplotlib can use a different fallback for ``annotate`` than for the
    title.  On Windows that made the numeric prefix render while Chinese
    glyphs disappeared from the small label boxes.  Passing the installed
    YaHei font explicitly keeps the local pattern names visible in the
    audit PNGs.
    """
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        return FontProperties(fname=str(font_path), size=8.8)
    return FontProperties(family="Microsoft YaHei", size=8.8)


def _period_frame(daily: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    if timeframe == Timeframe.MONTHLY:
        frame = resample_monthly(daily)
    elif timeframe == Timeframe.WEEKLY:
        frame = resample_weekly(daily)
    else:
        frame = daily.copy()
    frame = frame.sort_values("Date").tail(BAR_LIMIT).reset_index(drop=True)
    return frame


def _name(item: dict[str, Any]) -> str:
    pattern_id = str(item.get("pattern_id") or item.get("pattern") or "形态")
    record = get_pattern(pattern_id)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    override = metadata.get("display_name_override")
    if override:
        name = str(override)
    elif record and record.get("name_zh"):
        name = str(record["name_zh"])
    else:
        raw = str(item.get("pattern") or pattern_id)
        source_labels = metadata.get("source_labels") or []
        source_name = str(source_labels[0]) if source_labels else raw
        name = RAW_DISPLAY_NAMES.get(raw) or RAW_DISPLAY_NAMES.get(source_name) or source_name
    if pattern_id == "three_reverse_bullish":
        validation = str((item.get("metadata") or {}).get("context_validation", ""))
        if "放宽" in validation:
            name = "倒三阳（放宽变体）"
    variant = item.get("variant")
    variant_names = {
        "multiple": "多重变体",
        "compound": "复合变体",
        "pennant": "三角旗变体",
        "three_white_soldiers": "强势变体",
    }
    if variant in variant_names:
        name = f"{name}（{variant_names[variant]}）"
    return name


def _pattern_name(pattern_id: str | None) -> str:
    """Resolve a canonical ID to its Chinese display name for chart titles."""
    if not pattern_id:
        return "暂无单一主结论"
    record = get_pattern(str(pattern_id))
    if record and record.get("name_zh"):
        return str(record["name_zh"])
    return RAW_DISPLAY_NAMES.get(str(pattern_id), str(pattern_id))


def _status_text(item: dict[str, Any]) -> str:
    if str(item.get("display_role", "")) == "suppressed":
        return "冲突降级"
    if str(item.get("display_role", "")) == "timeframe_deemphasized":
        return "周期弱化"
    state = str(item.get("state", "candidate"))
    return {
        "confirmed": "已确认",
        "candidate": "候选",
        "degraded": "降级",
        "detected": "待判断",
        "invalidated": "已失效",
    }.get(state, state)


def _direction_color(item: dict[str, Any]) -> str:
    direction = str(item.get("direction", "neutral"))
    if direction in {"bullish", "bull"}:
        return "#d94b4b"  # A-share convention: red is bullish/up.
    if direction in {"bearish", "bear"}:
        return "#2e9d63"
    return "#64748b"


def _line_style(item: dict[str, Any]) -> str:
    state = str(item.get("state", "candidate"))
    return {"confirmed": "-", "degraded": ":", "candidate": "--"}.get(state, "--")


def _kind_names(values: list[str] | tuple[str, ...]) -> str:
    return "、".join(DISPLAY_KIND_NAMES.get(str(value), str(value)) for value in values)


def _select_preview_items(
    result: dict[str, Any],
    timeframe: Timeframe,
    *,
    mode: str = "preview",
) -> list[dict[str, Any]]:
    """Select local, labeled representatives for an audit chart.

    The production display policy still decides which signals are primary.
    This preview additionally shows a bounded number of lower-priority trend
    and candlestick candidates so a reviewer can see what was detected and
    what was later suppressed by a conflict rule.
    """
    all_items = result.get("display", {}).get("all", [])
    display = result.get("display", {})
    if mode in {"user", "product"}:
        # The mobile surface consumes the FAE contract's adaptive bucket
        # directly.  It is intentionally smaller than the audit buckets and
        # may contain fewer than five entries when the data does not support
        # that many credible signals.
        default_items = display.get("default") or []
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for item in default_items:
            if item.get("state") in {"invalidated", "expired"}:
                continue
            if item.get("display_role") in {"duplicate", "suppressed"}:
                continue
            key = (
                str(item.get("pattern_id") or item.get("pattern") or ""),
                int(item.get("start", 0)),
                int(item.get("end", item.get("start", 0))),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
        # Chronological placement avoids label lines crossing each other.  The
        # winner remains available in the JSON order and in the subtitle.
        return sorted(selected, key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0))))
    winner_id = result.get("display", {}).get("summary", {}).get("winner_pattern_id")
    quota = AUDIT_QUOTAS[timeframe] if mode == "audit" else PREVIEW_QUOTAS[timeframe]
    items: list[dict[str, Any]] = []
    for item in all_items:
        kind = str(item.get("kind", ""))
        role = str(item.get("display_role", ""))
        if item.get("state") in {"invalidated", "expired"} or role == "duplicate":
            continue
        # In the ordinary preview, monthly/weekly only foreground the resolved
        # chart while daily keeps a few chart candidates.  The requested audit
        # mode intentionally keeps representative chart candidates in every
        # timeframe so a reviewer can correct chart recognition manually.
        if mode != "audit" and kind == "chart" and not (
            timeframe == Timeframe.DAILY or str(item.get("pattern_id")) == str(winner_id)
        ):
            continue
        if kind not in quota:
            continue
        items.append(item)

    # Always keep the resolved winner, even when its kind is de-emphasized by
    # the timeframe profile (for example, a V-bottom on a monthly chart).
    if winner_id:
        winner_candidates = [
            item
            for item in all_items
            if str(item.get("pattern_id")) == str(winner_id)
            and item.get("state") not in {"invalidated", "expired"}
            and item.get("display_role") != "duplicate"
        ]
        if winner_candidates and not any(
            str(item.get("id")) == str(winner_candidates[0].get("id")) for item in items
        ):
            items.append(winner_candidates[0])

    representatives: dict[str, dict[str, Any]] = {}
    for item in items:
        group_id = str(item.get("display_group_id") or item.get("id") or "")
        is_rep = item.get("display_group_representative", True)
        if group_id in representatives and not is_rep:
            continue
        representatives[group_id] = item
    grouped = list(representatives.values())

    def rank(item: dict[str, Any]) -> tuple[int, int, int, float]:
        role_rank = {
            "primary": 5,
            "candidate": 4,
            "supporting": 3,
            "timeframe_deemphasized": 2,
            "suppressed": 1,
        }.get(str(item.get("display_role")), 1)
        return (
            1 if str(item.get("pattern_id")) == str(winner_id) else 0,
            role_rank,
            int(item.get("end", 0)),
            float(item.get("confidence", 0.0)),
        )

    selected: list[dict[str, Any]] = []
    for kind, limit in quota.items():
        candidates = [item for item in grouped if str(item.get("kind")) == kind]
        candidates.sort(key=rank, reverse=True)
        selected.extend(candidates[:limit])

    # The winner may be a chart in a profile with a chart quota already used by
    # other candidates.  Reinsert it and trim only a non-winner item of the
    # same kind so the primary geometry is always visible.
    if winner_id:
        winner = next(
            (item for item in grouped if str(item.get("pattern_id")) == str(winner_id)),
            None,
        )
        if winner is not None and not any(
            str(item.get("id")) == str(winner.get("id")) for item in selected
        ):
            same_kind = [index for index, item in enumerate(selected) if item.get("kind") == winner.get("kind")]
            if same_kind:
                selected[same_kind[-1]] = winner
            else:
                selected.append(winner)

    return sorted(selected, key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0))))


def _draw_candles(ax: Any, frame: pd.DataFrame) -> None:
    width = 0.62
    for index, row in frame.iterrows():
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        color = "#d94b4b" if close_price >= open_price else "#2e9d63"
        ax.vlines(index, low, high, color=color, linewidth=0.9, alpha=0.9, zorder=2)
        bottom = min(open_price, close_price)
        height = max(abs(close_price - open_price), 0.5)
        ax.add_patch(
            Rectangle(
                (index - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )
        )


def _draw_geometry(
    ax: Any,
    item: dict[str, Any],
    color: str,
    style: str,
    frame: pd.DataFrame | None = None,
    compact: bool = False,
) -> None:
    geometry = item.get("geometry") or {}
    pattern_id = str(item.get("pattern_id") or item.get("pattern") or "")
    pivots = geometry.get("pivots") or []
    vertex = next((pivot for pivot in pivots if len(pivot) >= 3 and pivot[2] == "vertex"), None)
    clip_window: tuple[float, float] | None = None
    if vertex is not None and pattern_id in {"v_bottom", "inverted_v_top"}:
        # V reversals are detected on the whole window, but drawing both legs
        # across 120 bars makes the mark look like a long trendline.  Clip the
        # two legs to a local neighborhood around the vertex.
        pivot_x = float(vertex[0])
        total = max(1, int(item.get("end", pivot_x)) - int(item.get("start", pivot_x)))
        window = max(8.0, min(24.0, total * 0.24))
        clip_window = (pivot_x - window, pivot_x + window)

    clipped_lines: list[tuple[float, float, float, float]] = []
    for line in geometry.get("lines", []):
        x1 = float(line.get("x1", 0.0))
        x2 = float(line.get("x2", x1))
        y1 = float(line.get("y1", 0.0))
        y2 = float(line.get("y2", y1))
        if clip_window is not None:
            left, right = clip_window
            original_left, original_right = min(x1, x2), max(x1, x2)
            if right < original_left or left > original_right:
                continue
            left = max(left, original_left)
            right = min(right, original_right)
            if abs(x2 - x1) < 1e-12:
                y_left = y_right = y1
            else:
                slope = (y2 - y1) / (x2 - x1)
                y_left = y1 + slope * (left - x1)
                y_right = y1 + slope * (right - x1)
            if x2 < x1:
                x_left, x_right = right, left
                y_left, y_right = y_right, y_left
            else:
                x_left, x_right = left, right
            x1, x2, y1, y2 = x_left, x_right, y_left, y_right
        clipped_lines.append((x1, y1, x2, y2))
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linestyle=style,
            linewidth=2.2 if pattern_id in {"v_bottom", "inverted_v_top"} else 1.7,
            alpha=0.88,
            zorder=4,
        )

    # A wedge/triangle/rectangle should read as a bounded structure, not two
    # unexplained lines.  Shade the local envelope lightly and outline its
    # pivots; this is only a visualization aid and does not change detection.
    if len(clipped_lines) >= 2 and pattern_id not in {"v_bottom", "inverted_v_top"}:
        first, second = clipped_lines[0], clipped_lines[1]
        xs = np.linspace(max(min(first[0], first[2]), min(second[0], second[2])),
                         min(max(first[0], first[2]), max(second[0], second[2])), 80)
        if len(xs) >= 2:
            y_first = np.interp(xs, [first[0], first[2]], [first[1], first[3]])
            y_second = np.interp(xs, [second[0], second[2]], [second[1], second[3]])
            lower = np.minimum(y_first, y_second)
            upper = np.maximum(y_first, y_second)
            ax.fill_between(xs, lower, upper, color=color, alpha=0.08, zorder=1)

    for arc in geometry.get("arcs", []):
        start = int(arc.get("start", item.get("start", 0)))
        end = int(arc.get("end", item.get("end", start)))
        coefficients = arc.get("coefficients") or [0.0, 0.0, 0.0]
        xs = np.linspace(start, end, 80)
        ys = np.polyval(np.asarray(coefficients, dtype=float), xs)
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.8, alpha=0.88, zorder=4)
    valid_pivots = [pivot for pivot in pivots if len(pivot) >= 2]
    if valid_pivots:
        # Every structural chart pattern must expose the points that justify
        # its name.  This makes double bottoms/tops and island boundaries
        # auditable instead of showing only a floating text label.
        ax.scatter(
            [float(pivot[0]) for pivot in valid_pivots],
            [float(pivot[1]) for pivot in valid_pivots],
            s=20,
            color=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=6,
        )
    if vertex is not None:
        ax.scatter(
            [float(vertex[0])],
            [float(vertex[1])],
            s=30,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
    if compact and pattern_id in {
        "head_and_shoulders_top",
        "head_and_shoulders_bottom",
        "compound_head_and_shoulders_top",
        "compound_head_and_shoulders_bottom",
    }:
        pivot_labels = {
            "left_shoulder": "左肩",
            "head": "头",
            "right_shoulder": "右肩",
        }
        for pivot in pivots:
            if len(pivot) < 3 or pivot[2] not in pivot_labels:
                continue
            px, py = float(pivot[0]), float(pivot[1])
            label_offset = 0.025 * max(
                float(frame["High"].max() - frame["Low"].min()) if frame is not None else 1.0,
                1.0,
            )
            text_y = py + label_offset if pattern_id.endswith("_top") else py - label_offset
            ax.text(
                px,
                text_y,
                pivot_labels[pivot[2]],
                ha="center",
                va="bottom" if pattern_id.endswith("_top") else "top",
                fontproperties=_label_font(),
                fontsize=8.2,
                color=color,
                zorder=8,
            )


def _signal_bounds(item: dict[str, Any], frame: pd.DataFrame) -> tuple[int, int, float, float, float]:
    start = max(0, min(len(frame) - 1, int(item.get("start", 0))))
    end = max(start, min(len(frame) - 1, int(item.get("end", start))))
    segment = frame.iloc[start : end + 1]
    high = float(segment["High"].max())
    low = float(segment["Low"].min())
    pad = max((high - low) * 0.10, float(frame["Close"].iloc[-1]) * 0.002)
    return start, end, high, low, pad


def _draw_signal_box(
    ax: Any,
    item: dict[str, Any],
    frame: pd.DataFrame,
    color: str,
    style: str,
    *,
    compact: bool = False,
) -> tuple[float, float]:
    start, end, high, low, pad = _signal_bounds(item, frame)
    if item.get("kind") == "chart":
        _draw_geometry(ax, item, color, style, frame=frame, compact=compact)
        # Some candidate structures have no geometry yet; a light boundary is
        # preferable to pretending that a line was confirmed.
        if not (item.get("geometry") or {}).get("lines") and not (item.get("geometry") or {}).get("arcs"):
            ax.add_patch(
                Rectangle(
                    (start - 0.35, low),
                    max(end - start + 0.7, 0.7),
                    max(high - low, pad),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.1,
                    linestyle=style,
                    alpha=0.55,
                    zorder=3,
                )
            )
    else:
        ax.add_patch(
            Rectangle(
                (start - 0.35, low - pad * 0.20),
                max(end - start + 0.7, 0.7),
                max(high - low + pad * 0.40, pad),
                fill=False,
                edgecolor=color,
                linewidth=1.1,
                linestyle=style,
                alpha=0.62,
                zorder=4,
            )
        )
    anchor_x = (start + end) / 2
    direction = str(item.get("direction", "neutral"))
    anchor_y = low if direction in {"bullish", "bull"} else high
    geometry = item.get("geometry") or {}
    vertex = next((pivot for pivot in geometry.get("pivots", []) if len(pivot) >= 3 and pivot[2] == "vertex"), None)
    if vertex is not None and str(item.get("pattern_id") or item.get("pattern")) in {"v_bottom", "inverted_v_top"}:
        anchor_x, anchor_y = float(vertex[0]), float(vertex[1])
    return anchor_x, anchor_y


def _gap_segments(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract drawable gap zones from the normalized OHLC frame."""
    high = frame["High"].to_numpy(float)
    low = frame["Low"].to_numpy(float)
    segments: list[dict[str, Any]] = []
    for index in range(1, len(frame)):
        previous_high = float(high[index - 1])
        previous_low = float(low[index - 1])
        current_high = float(high[index])
        current_low = float(low[index])
        if current_low > previous_high * 1.001:
            direction = "up"
            bottom, top = previous_high, current_low
        elif current_high < previous_low * 0.999:
            direction = "down"
            bottom, top = current_high, previous_low
        else:
            continue
        fill_index: int | None = None
        for future in range(index + 1, len(frame)):
            if direction == "up" and low[future] <= top:
                fill_index = future
                break
            if direction == "down" and high[future] >= bottom:
                fill_index = future
                break
        segments.append(
            {
                "index": index,
                "direction": direction,
                "bottom": bottom,
                "top": top,
                "filled": fill_index is not None,
                "fill_index": fill_index,
            }
        )
    return segments


def _draw_gap_zones(
    ax: Any,
    frame: pd.DataFrame,
    items: list[dict[str, Any]],
) -> None:
    """Draw a small number of actual price-gap zones for the user view."""
    segments = _gap_segments(frame)
    if not segments:
        return
    spans = [
        (int(item.get("start", 0)), int(item.get("end", item.get("start", 0))))
        for item in items
        if item.get("kind") == "chart"
        and str(item.get("pattern_id") or "").startswith("island_")
    ]
    selected: list[dict[str, Any]] = []
    for segment in segments:
        chosen = False
        for item in items:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            gap_index = int(item.get("start", -1))
            if metadata.get("gap_top") is not None and gap_index == segment["index"]:
                chosen = True
                break
        if not chosen:
            chosen = any(start <= segment["index"] <= end for start, end in spans)
        # In a user-facing chart show recent gaps as context, but do not turn
        # every tiny historical gap into a label.
        if chosen or segment["index"] >= len(frame) - 18:
            selected.append(segment)
    selected = selected[-4:]
    for segment in selected:
        index = segment["index"]
        right = segment["fill_index"] if segment["fill_index"] is not None else min(len(frame) - 1, index + 8)
        color = "#d94b4b" if segment["direction"] == "up" else "#2e9d63"
        ax.add_patch(
            Rectangle(
                (index - 0.44, segment["bottom"]),
                max(0.9, right - index + 0.88),
                max(segment["top"] - segment["bottom"], 1e-9),
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.10,
                zorder=1.5,
            )
        )
        ax.text(
            index,
            segment["top"],
            "向上缺口" if segment["direction"] == "up" else "向下缺口",
            ha="center",
            va="bottom",
            fontproperties=_label_font(),
            fontsize=7.6,
            color=color,
            zorder=7,
        )


def _draw_labels(
    ax: Any,
    items: list[dict[str, Any]],
    frame: pd.DataFrame,
    pmin: float,
    pmax: float,
    *,
    compact: bool = False,
) -> None:
    span = max(pmax - pmin, 1.0)
    occupied: dict[str, list[tuple[float, float, int]]] = {"above": [], "below": []}
    label_font = _label_font()
    for number, item in enumerate(items, start=1):
        color = _direction_color(item)
        anchor_x, anchor_y = _draw_signal_box(
            ax,
            item,
            frame,
            color,
            _line_style(item),
            compact=compact,
        )
        direction = str(item.get("direction", "neutral"))
        # Keep the pattern name on its own line.  It remains readable on a
        # mobile-sized crop while the status is still available to reviewers.
        label = f"{number}. {_name(item)}" if compact else f"{number}. {_name(item)}\n{_status_text(item)}"
        side = "above" if direction in {"bearish", "bear"} else "below"
        start, end, high, low, pad = _signal_bounds(item, frame)
        pattern_id = str(item.get("pattern_id") or item.get("pattern") or "")

        # Put long chart structures' labels at the structure edge/pivot rather
        # than underneath the whole plot.  The geometry itself already shows
        # the span, so the leader only needs to travel a short distance.
        if item.get("kind") == "chart":
            if pattern_id in {"v_bottom", "inverted_v_top"}:
                text_x = anchor_x + (4.0 if pattern_id == "v_bottom" else -4.0)
                base_y = anchor_y + (span * 0.045 if pattern_id == "v_bottom" else -span * 0.045)
                side = "above" if pattern_id == "v_bottom" else "below"
            else:
                text_x = start + (end - start) * 0.78
                base_y = high + pad * 0.45 if side == "above" else low - pad * 0.45
        else:
            text_x = anchor_x
            base_y = high + pad * 0.45 if side == "above" else low - pad * 0.45

        label_half_width = max(2.0, min(12.0, len(label) * 0.24))
        left, right = text_x - label_half_width, text_x + label_half_width
        lane = 0
        while any(
            other_left < right + 0.5 and left < other_right + 0.5 and other_lane == lane
            for other_left, other_right, other_lane in occupied[side]
        ):
            lane += 1
        occupied[side].append((left, right, lane))
        lane_step = max(span * 0.028, pad * 0.60)
        text_y = base_y + (lane_step * lane if side == "above" else -lane_step * lane)

        # Keep labels inside the plot near the left/right edges.
        if text_x < 2:
            text_x = 2
            horizontal_alignment = "left"
        elif text_x > len(frame) - 3:
            text_x = len(frame) - 3
            horizontal_alignment = "right"
        else:
            horizontal_alignment = "center"
        status_role = str(item.get("display_role", ""))
        if status_role == "suppressed":
            color = "#94a3b8"
        ax.annotate(
            label,
            xy=(anchor_x, anchor_y),
            xytext=(text_x, text_y),
            textcoords="data",
            ha=horizontal_alignment,
            va="bottom" if side == "above" else "top",
            fontproperties=label_font,
            fontsize=8.2 if compact else 8.8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.90 if status_role != "suppressed" else 0.78,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 0.7,
                "alpha": 0.7,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=7,
        )


def _plot_period(
    frame: pd.DataFrame,
    result: dict[str, Any],
    timeframe: Timeframe,
    output_path: Path,
    as_of_date: pd.Timestamp,
    display_name: str,
    mode: str = "preview",
) -> dict[str, Any]:
    items = _select_preview_items(result, timeframe, mode=mode)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")
    _draw_candles(ax, frame)
    values = np.r_[frame["High"].to_numpy(float), frame["Low"].to_numpy(float)]
    pmin = float(values.min())
    pmax = float(values.max())
    span = max(pmax - pmin, 1.0)
    user_mode = mode in {"user", "product"}
    if user_mode:
        _draw_gap_zones(ax, frame, items)
    _draw_labels(ax, items, frame, pmin, pmax, compact=user_mode)
    ax.set_xlim(-1, len(frame))
    # Labels now sit beside their local structures, so only a modest margin is
    # needed around the price range.
    ax.set_ylim(pmin - span * 0.16, pmax + span * 0.16)
    ax.grid(axis="y", color="#e8edf3", linewidth=0.8)
    ax.grid(axis="x", color="#f1f4f8", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("指数点位")
    dates = pd.to_datetime(frame["Date"])
    tick_step = max(1, len(frame) // 10)
    ticks = list(range(0, len(frame), tick_step))
    if ticks[-1] != len(frame) - 1:
        ticks.append(len(frame) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates.iloc[index].strftime("%Y-%m-%d") for index in ticks], rotation=30, ha="right")
    period_name = {Timeframe.MONTHLY: "月K", Timeframe.WEEKLY: "周K", Timeframe.DAILY: "日K"}[timeframe]
    policy = result.get("timeframe_policy", {})
    summary = result.get("display", {}).get("summary", {})
    winner_id = summary.get("winner_pattern_id")
    winner = _pattern_name(winner_id)
    foreground = _kind_names(
        policy.get("effective_primary_kinds") or policy.get("primary_kinds", [])
    ) or "未设置"
    shadow_warning = str(result.get("evidence", {}).get("shadow_interpretation", ""))
    title_mode = "用户体验版" if user_mode else "审计" if mode == "audit" else "周期策略测试"
    ax.set_title(
        f"{display_name} {period_name} · FAE {title_mode}",
        fontsize=18,
        fontweight="bold",
        pad=22,
    )
    display_count_label = (
        f"图中标注 {len(items)} 条自适应重点"
        if user_mode
        else f"图中标注 {len(items)} 条代表性候选"
    )
    subtitle = (
        f"K线区间 {dates.iloc[0].strftime('%Y-%m-%d')} ~ {dates.iloc[-1].strftime('%Y-%m-%d')} | "
        f"数据截至 {as_of_date.strftime('%Y-%m-%d')} | "
        f"前景类型：{foreground} | "
        f"FAE总裁决：{winner} | "
        f"{display_count_label} | "
        f"主形态 {summary.get('primary_count', 0)} / 候选 {summary.get('candidate_count', 0)} / "
        f"重复或冲突过滤 {summary.get('suppressed_count', 0)}"
    )
    ax.text(
        0.01,
        1.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=9.5,
        color="#5b6470",
        va="bottom",
    )
    if user_mode:
        ax.text(
            0.99,
            0.985,
            "默认自适应：只显示重点；其他简单/组合/趋势/技术图形可由图层开关展开",
            transform=ax.transAxes,
            fontsize=8.8,
            color="#64748b",
            ha="right",
            va="top",
        )
    if shadow_warning and shadow_warning != "影线频率未达到提醒阈值":
        ax.text(
            0.01,
            0.985,
            f"影线/实体提醒：{shadow_warning}",
            transform=ax.transAxes,
            fontsize=9.2,
            color="#9a3412" if "抛压" in shadow_warning else "#166534" if "承接" in shadow_warning else "#6b7280",
            va="top",
        )
    ax.text(
        0.99,
        0.01,
        "红=偏多 绿=偏空 灰=中性；实线=已确认，虚线=候选",
        transform=ax.transAxes,
        fontsize=9,
        color="#68727d",
        ha="right",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "timeframe": timeframe.value,
        "image": str(output_path),
        "bars": len(frame),
        "start": dates.iloc[0].strftime("%Y-%m-%d"),
        "end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "data_as_of": as_of_date.strftime("%Y-%m-%d"),
        "winner_pattern_id": winner_id,
        "winner_name": winner,
        "foreground_kinds": policy.get("effective_primary_kinds") or policy.get("primary_kinds", []),
        "raw_event_count": int(result.get("quantitative_filter", {}).get("input_count", 0)),
        "accepted_signal_count": int(result.get("quantitative_filter", {}).get("accepted_count", 0)),
        "inverted_three_yang_rejections": [
            rejection
            for rejection in result.get("quantitative_filter", {}).get("rejections", [])
            if rejection.get("pattern") == "three_reverse_bullish"
        ],
        "displayed_items": [
            {
                "pattern_id": item.get("pattern_id"),
                "name": _name(item),
                "state": item.get("state"),
                "direction": item.get("direction"),
                "start": item.get("start"),
                "end": item.get("end"),
            }
            for item in items
        ],
        "display_summary": result.get("display", {}).get("summary", {}),
        "shadow_interpretation": shadow_warning,
        "shadow_frequency": result.get("evidence", {}).get("shadow_frequency"),
        "body_shrink_ratio": result.get("evidence", {}).get("body_shrink_ratio"),
        "render_mode": mode,
    }


def main(symbol: str = "sh000001", display_name: str = "上证指数", output_stem: str | None = None) -> None:
    _configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = output_stem or symbol
    daily = fetch_tencent_daily(symbol, years=12)
    as_of_date = pd.to_datetime(daily["Date"]).max()
    summaries: list[dict[str, Any]] = []
    for timeframe in (Timeframe.MONTHLY, Timeframe.WEEKLY, Timeframe.DAILY):
        frame = _period_frame(daily, timeframe)
        events = detect_all(frame, max_trend_per_direction=3)
        candlestick_signals = from_v7_events(events, timeframe=timeframe)
        result = JudgmentEngine().evaluate(
            frame,
            candlestick_signals=candlestick_signals,
            timeframe=timeframe,
            context={"symbol": symbol, "preview": True},
        )
        output_path = OUTPUT_DIR / f"{output_stem}_fae_{timeframe.value}_preview.png"
        summaries.append(_plot_period(frame, result, timeframe, output_path, as_of_date, display_name))
    summary_path = OUTPUT_DIR / f"{output_stem}_fae_preview_summary.json"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "symbol": symbol,
        "name": display_name,
        "data_source": "Tencent daily endpoint through existing local helper",
        "bar_limit_per_timeframe": BAR_LIMIT,
        "data_as_of": as_of_date.strftime("%Y-%m-%d"),
        "periods": summaries,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline FAE M/W/D preview charts.")
    parser.add_argument("--symbol", default="sh000001", help="Tencent symbol, e.g. sh688981")
    parser.add_argument("--name", default="上证指数", help="Display name")
    parser.add_argument("--output-stem", default=None, help="Filename stem under fae/preview")
    args = parser.parse_args()
    main(args.symbol, args.name, args.output_stem)
