"""Compile textbook examples into FAE decision cases.

OCR text is an internal build input only. The public artifact contains
structured pattern/context/evidence/decision records, never a raw book dump.
The scanner is resumable at page granularity and deduplicates split volumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .case_knowledge import CompiledCase
from .pattern_registry import canonicalize_pattern_id, variant_map

COMPILER_VERSION = "1.0"
CASE_MARKER = re.compile(
    r"(?P<label>"
    r"习题\s*[0-9０-９]{1,4}|示例题?|"
    r"实例\s*[一二三四五六七八九十百0-9０-９]{1,8}|"
    r"案例\s*[一二三四五六七八九十百0-9０-９]{1,8}|"
    r"例题\s*[0-9０-９]{1,4}|"
    r"Case\s+Stud(?:y|ies)\s*[0-9A-Za-z.-]*|"
    r"Example\s+[0-9A-Za-z.-]+|"
    r"Sample\s+Trade\s*[0-9A-Za-z.-]*"
    r")",
    re.IGNORECASE,
)
CASE_LANGUAGE = re.compile(
    r"参考答案|请问|回答|如何操作|操作策略|投资者|分析|"
    r"实战|往后走势|失败案例|假突破|买进|卖出|离场|持股|止损|"
    r"case study|example|sample trade|trading tactics|focus on failures",
    re.IGNORECASE,
)
DEFINITION_LANGUAGE = re.compile(
    r"技术图形一览表|K线一览表|概述|目录|定义如下|基本概念|术语表|"
    r"table of contents|list of tables|visual index|glossary",
    re.IGNORECASE,
)
FIGURE_MARKER = re.compile(
    r"(?:图|Figure|Fig\.)\s*[0-9０-９]+(?:[.-][0-9０-９]+)?",
    re.IGNORECASE,
)

PATTERN_ALIASES: dict[str, tuple[str, ...]] = {
    "abcd_bearish": ("AB=CD®, Bearish", "AB=CD, Bearish"),
    "abcd_bullish": ("AB=CD®, Bullish", "AB=CD, Bullish"),
    "bat_bearish": ("Bat®, Bearish", "Bat, Bearish"),
    "bat_bullish": ("Bat®, Bullish", "Bat, Bullish"),
    "big_m": ("Big M",),
    "big_w": ("Big W",),
    "broadening_bottom": ("Broadening Bottoms",),
    "broadening_top": ("Broadening Tops",),
    "right_angled_broadening_ascending": (
        "Broadening Formation, Right-Angled and Ascending",
        "Broadening Formation, Right Angled and Ascending",
    ),
    "right_angled_broadening_descending": (
        "Broadening Formation, Right-Angled and Descending",
        "Broadening Formation, Right Angled and Descending",
    ),
    "broadening_wedge_ascending": ("Broadening Wedge, Ascending",),
    "broadening_wedge_descending": ("Broadening Wedge, Descending",),
    "bump_and_run_bottom": (
        "Bump-and-Run Reversal, Bottom",
        "Bump and Run Reversal, Bottom",
    ),
    "bump_and_run_top": (
        "Bump-and-Run Reversal, Top",
        "Bump and Run Reversal, Top",
    ),
    "butterfly_bearish": ("Butterfly®, Bearish", "Butterfly, Bearish"),
    "butterfly_bullish": ("Butterfly®, Bullish", "Butterfly, Bullish"),
    "cloudbank": ("Cloudbanks",),
    "crab_bearish": ("Crab®, Bearish", "Crab, Bearish"),
    "crab_bullish": ("Crab®, Bullish", "Crab, Bullish"),
    "cup_with_handle": ("Cup with Handle",),
    "inverted_cup_with_handle": ("Cup with Handle, Inverted",),
    "dormant_bottom": ("潜伏底", "浅幅底", "Dormant Bottom", "Dormant Base"),
    "diving_board": ("Diving Board",),
    "flag": ("Flags", "flag pattern"),
    "high_tight_flag": ("Flags, High and Tight", "high and tight flag"),
    "gap_pattern": ("Gaps", "gap pattern"),
    "gartley_bearish": ("Gartley, Bearish",),
    "gartley_bullish": ("Gartley, Bullish",),
    "horn_bottom": ("Horn Bottoms",),
    "horn_top": ("Horn Tops",),
    "island_reversal": ("Island Reversals",),
    "measured_move_down": ("Measured Move Down",),
    "measured_move_up": ("Measured Move Up",),
    "pennant": ("Pennants", "pennant pattern", "三角旗", "三角旗形"),
    "pipe_bottom": ("Pipe Bottoms",),
    "pipe_top": ("Pipe Tops",),
    "roof": ("Roof",),
    "inverted_roof": ("Roof, Inverted",),
    "ascending_scallop": ("Scallops, Ascending",),
    "ascending_inverted_scallop": ("Scallops, Ascending and Inverted",),
    "descending_scallop": ("Scallops, Descending",),
    "descending_inverted_scallop": ("Scallops, Descending and Inverted",),
    "three_falling_peaks": ("Three Falling Peaks",),
    "three_peaks_domed_house": ("Three Peaks and Domed House",),
    "three_rising_valleys": ("Three Rising Valleys",),
    "extended_v_bottom": ("V Bottoms, Extended",),
    "v_top": ("V Tops",),
    "extended_v_top": ("V Tops, Extended",),
    "wolfe_wave_bearish": ("Wolfe Wave®, Bearish", "Wolfe Wave, Bearish"),
    "wolfe_wave_bullish": ("Wolfe Wave®, Bullish", "Wolfe Wave, Bullish"),
    "head_and_shoulders_top": (
        "头肩顶",
        "head-and-shoulders top",
        "head and shoulders top",
    ),
    "head_and_shoulders_bottom": (
        "头肩底",
        "inverse head-and-shoulders",
        "head and shoulders bottom",
    ),
    "double_top": ("双顶", "M头", "double top"),
    "double_bottom": ("双底", "W底", "double bottom"),
    "triple_top": ("三重顶", "多重顶", "triple top"),
    "triple_bottom": ("三重底", "多重底", "triple bottom"),
    "rounding_top": ("圆顶", "圆弧顶", "rounding top"),
    "rounding_bottom": ("圆底", "圆弧底", "rounding bottom"),
    "v_bottom": ("V形底", "V形反转", "v bottom"),
    "inverted_v_top": ("倒置V形", "倒V形", "inverted v"),
    "island_reversal_top": ("顶部岛形", "岛形顶", "island top"),
    "island_reversal_bottom": ("底部岛形", "岛形底", "island bottom"),
    "diamond": ("菱形", "钻石形", "diamond"),
    "symmetrical_triangle": ("收敛三角形", "对称三角形", "三角旗", "三角旗形", "symmetrical triangle"),
    "ascending_triangle": ("上升三角形", "ascending triangle"),
    "descending_triangle": ("下降三角形", "descending triangle"),
    "broadening_triangle": ("扩散三角形", "broadening triangle"),
    "ascending_flag": ("上升旗形", "bull flag", "ascending flag"),
    "descending_flag": ("下降旗形", "bear flag", "descending flag"),
    "rising_wedge": ("上升楔形", "rising wedge"),
    "falling_wedge": ("下降楔形", "falling wedge"),
    "rectangle": ("矩形", "长方矩形", "箱形整理", "箱体", "整理区间", "rectangle"),
    "trendline_break": ("趋势线", "trendline"),
    "exhaustion_gap": ("竭尽缺口", "exhaustion gap"),
    "breakaway_gap": ("突破缺口", "breakaway gap"),
    "continuation_gap": ("持续缺口", "测量缺口", "continuation gap"),
    "red_three_soldiers": ("红三兵", "three advancing white soldiers"),
    "large_bullish": ("大阳线", "long white candlestick", "long white candle"),
    "large_bearish": ("大阴线", "long black candlestick", "long black candle"),
    "small_bullish": ("小阳线",),
    "small_bearish": ("小阴线",),
    "three_black_crows": ("三只乌鸦", "黑三兵", "three black crows"),
    "engulfing": ("穿头破脚", "吞没形态", "engulfing pattern"),
    "bullish_engulfing": ("底部穿头破脚", "阳包阴", "bullish engulfing"),
    "bearish_engulfing": ("顶部穿头破脚", "阴包阳", "bearish engulfing"),
    "morning_star": ("早晨之星", "启明星", "morning star"),
    "morning_doji_star": ("早晨十字星", "morning doji star"),
    "evening_star": ("黄昏之星", "evening star"),
    "evening_doji_star": ("黄昏十字星", "evening doji star"),
    "bullish_abandoned_baby": ("看涨弃婴十字星", "bullish abandoned baby"),
    "bearish_abandoned_baby": ("看跌弃婴十字星", "bearish abandoned baby"),
    "hammer": ("锤子线", "锤头线", "hammer"),
    "hanging_man": ("吊颈线", "hanging man"),
    "shooting_star": ("射击之星", "shooting star"),
    "inverted_hammer": ("倒锤头线", "inverted hammer"),
    "doji": ("十字星", "十字线", "doji"),
    "long_legged_doji": ("长十字线", "long-legged doji"),
    "t_line": ("T字线", "dragonfly doji"),
    "inverted_t_line": ("倒T字线", "gravestone doji"),
    "spinning_top": ("螺旋桨", "spinning top"),
    "dark_cloud_cover": ("乌云盖顶", "dark-cloud cover", "dark cloud cover"),
    "piercing_pattern": ("曙光初现", "刺透形态", "piercing pattern"),
    "harami": ("身怀六甲", "孕线", "harami"),
    "tower_top": ("塔形顶", "tower top"),
    "tower_bottom": ("塔形底", "tower bottom"),
    "tweezers_top": ("平顶", "镊子顶", "tweezers top"),
    "tweezers_bottom": ("平底", "镊子底", "tweezers bottom"),
    "meeting_lines_bullish": ("好友反攻", "bullish meeting lines"),
    "meeting_lines_bearish": ("淡友反攻", "bearish meeting lines"),
    "rising_window": ("向上跳空缺口", "上涨窗口", "rising window"),
    "falling_window": ("向下跳空缺口", "下跌窗口", "falling window"),
    "three_gap_up": ("连续跳空三阳线", "three gaps up"),
    "three_gap_down": ("连续跳空三阴线", "three gaps down"),
    "low_parallel_bullish": ("低位并排阳线",),
    "low_five_bullish": ("低档五阳线",),
    "gradual_rise": ("冉冉上升形",),
    "slow_rise": ("徐缓上升形",),
    "steady_rise": ("稳步上涨形",),
    "rising_resistance": ("上升抵抗形",),
    "rising_two_stars": ("上涨二颗星", "上涨两颗星"),
    "high_parallel_bullish": ("高位并排阳线", "升势恋人肩并肩缺口"),
    "two_bullish_one_bearish": ("两红夹一黑",),
    "upside_gap_tasuki": ("跳空上扬形", "upside-gap tasuki"),
    "downside_gap_tasuki": ("跳空下跌形", "downside-gap tasuki"),
    "high_price_escape": ("高开出逃形",),
    "downside_recovery": ("下探上涨形",),
    "two_crows": ("双飞乌鸦", "upside-gap two crows"),
    "three_falling_candles": ("下跌三连阴",),
    "high_five_bearish": ("高档五阴线",),
    "low_coiling": ("低档盘旋形",),
    "continuous_decline": ("绵绵阴跌形",),
    "slow_decline": ("徐缓下跌形",),
    "decline_continuation": ("下跌不止形",),
    "falling_resistance": ("下降抵抗形",),
    "falling_three_stars": ("下跌三颗星",),
    "rise_blocked": ("升势受阻",),
    "rise_stalled": ("升势停顿",),
    "two_bearish_one_bullish": ("两黑夹一红",),
    "descending_cover": ("下降覆盖线",),
    "bullish_limp": ("阳线跛脚形",),
    "three_reverse_bullish": ("倒三阳",),
    "acceleration_line": ("加速度线",),
    "arc_line": ("弧形线",),
    "end_line": ("尽头线",),
    "kneading_line": ("搓揉线",),
    "one_price_line": ("一字线",),
    "upper_coiling": ("上档盘旋形",),
    "rainstorm": ("倾盆大雨",),
    "rising_sun": ("旭日东升",),
    "multi_party_pioneer": ("多方尖兵",),
    "short_party_pioneer": ("空方尖兵",),
    "rising_three_methods": ("上升三部曲", "rising three methods"),
    "falling_three_methods": ("下降三部曲", "falling three methods"),
    "golden_cross": ("黄金交叉", "金叉", "golden cross"),
    "death_cross": ("死亡交叉", "死叉", "death cross"),
    "silver_valley": ("银山谷",),
    "golden_valley": ("金山谷",),
    "death_valley": ("死亡谷",),
    "dragon_out_of_sea": ("蛟龙出海",),
    "guillotine": ("断头铡刀",),
    "ma_long_alignment": ("多头排列", "bullish moving-average alignment"),
    "ma_short_alignment": ("空头排列", "bearish moving-average alignment"),
}

EVIDENCE_ALIASES: dict[str, tuple[str, ...]] = {
    "neckline_break": (
        "跌破颈线",
        "突破颈线",
        "breaks the neckline",
        "neckline breakout",
    ),
    "breakout_confirmed": ("有效突破", "确认突破", "突破有效", "confirmed breakout"),
    "false_break": ("假突破", "false breakout", "throwback failure"),
    "pullback_retest": ("反抽", "回抽", "回踩", "throwback", "pullback"),
    "volume_expansion": ("成交量放大", "放量", "volume expansion", "heavy volume"),
    "volume_contraction": ("成交量萎缩", "缩量", "volume contraction", "light volume"),
    "gap": ("跳空", "缺口", "gap"),
    "multiple_gaps": ("连续缺口", "三个缺口", "multiple gaps"),
    "gap_filled": ("缺口回补", "缺口被封闭", "gap closed", "gap filled"),
    "support_break": ("跌破支撑", "breaks support", "support break"),
    "resistance_break": ("突破阻力", "breaks resistance", "resistance break"),
    "time_confirmation": ("连续3", "三个交易日", "持续超过", "confirmation days"),
    "reclaim_level": ("重新站上", "收回", "reclaim"),
    "trend_conflict": ("下降趋势中", "上涨趋势中", "趋势相反", "countertrend"),
}

CURATED_CASE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("股市操练大全++习题集", "习题66"): {
        "patterns": ["trendline_break"],
        "direction": "bullish",
        "state": "degraded",
        "evidence": ["false_break", "insufficient_time_confirmation"],
        "invalidation": ["confirmed_trendline_break"],
    },
    ("股市操练大全++习题集", "习题67"): {
        "patterns": ["double_top", "large_bullish"],
        "direction": "bearish",
        "state": "confirmed",
        "evidence": ["neckline_break", "time_confirmation"],
        "conflict": ["competing_evidence"],
        "invalidation": ["reclaim_breakout_level"],
    },
    ("股市操练大全++习题集", "习题72"): {
        "patterns": ["red_three_soldiers"],
        "direction": "neutral",
        "state": "degraded",
        "evidence": ["trend_conflict"],
        "conflict": ["competing_evidence"],
        "context_dependent": True,
        "invalidation": ["trend_reversal_confirmed"],
    },
    ("股市操练大全++习题集", "习题74"): {
        "patterns": ["island_reversal_top", "death_cross"],
        "direction": "bearish",
        "state": "confirmed",
        "evidence": ["gap", "multiple_gaps"],
        "invalidation": ["gap_refilled", "reclaim_breakout_level"],
    },
    ("股市操练大全++习题集", "习题77"): {
        "patterns": ["exhaustion_gap", "large_bullish"],
        "direction": "bearish",
        "state": "confirmed",
        "evidence": ["multiple_gaps", "gap_filled"],
        "conflict": ["competing_evidence"],
        "invalidation": ["no_follow_through"],
    },
}

CHART_PATTERNS = {
    "abcd_bearish",
    "abcd_bullish",
    "bat_bearish",
    "bat_bullish",
    "big_m",
    "big_w",
    "broadening_bottom",
    "broadening_top",
    "right_angled_broadening_ascending",
    "right_angled_broadening_descending",
    "broadening_wedge_ascending",
    "broadening_wedge_descending",
    "bump_and_run_bottom",
    "bump_and_run_top",
    "butterfly_bearish",
    "butterfly_bullish",
    "cloudbank",
    "crab_bearish",
    "crab_bullish",
    "cup_with_handle",
    "inverted_cup_with_handle",
    "diving_board",
    "flag",
    "high_tight_flag",
    "gap_pattern",
    "gartley_bearish",
    "gartley_bullish",
    "horn_bottom",
    "horn_top",
    "island_reversal",
    "measured_move_down",
    "measured_move_up",
    "pennant",
    "pipe_bottom",
    "pipe_top",
    "roof",
    "inverted_roof",
    "ascending_scallop",
    "ascending_inverted_scallop",
    "descending_scallop",
    "descending_inverted_scallop",
    "three_falling_peaks",
    "three_peaks_domed_house",
    "three_rising_valleys",
    "extended_v_bottom",
    "v_top",
    "extended_v_top",
    "wolfe_wave_bearish",
    "wolfe_wave_bullish",
    "head_and_shoulders_top",
    "head_and_shoulders_bottom",
    "double_top",
    "double_bottom",
    "triple_top",
    "triple_bottom",
    "dormant_bottom",
    "rounding_top",
    "rounding_bottom",
    "v_bottom",
    "inverted_v_top",
    "island_reversal_top",
    "island_reversal_bottom",
    "diamond",
    "symmetrical_triangle",
    "ascending_triangle",
    "descending_triangle",
    "broadening_triangle",
    "ascending_flag",
    "descending_flag",
    "rising_wedge",
    "falling_wedge",
    "rectangle",
}

# Keep the historical exported constant for callers, but make the JSON
# registry the runtime source of truth for all future compiler runs.
CANONICAL_PATTERN_VARIANTS = variant_map()


def _canonicalize_pattern_ids(patterns: list[str]) -> tuple[list[str], dict[str, str]]:
    variants: dict[str, str] = {}
    canonical: set[str] = set()
    for pattern in patterns:
        base = canonicalize_pattern_id(pattern)
        canonical.add(base)
        if base != pattern:
            variants[pattern] = base
    return sorted(canonical), variants


@dataclass
class PageRecord:
    """Internal page cache record."""

    book: str
    source_file: str
    page: int
    text: str
    confidence: float
    method: str


def _safe_cache_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{re.sub(r'[^0-9A-Za-z._-]+', '_', path.stem)[:60]}.jsonl"


def _load_page_cache(path: Path) -> dict[int, PageRecord]:
    if not path.exists():
        return {}
    records: dict[int, PageRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            record = PageRecord(**value)
            records[record.page] = record
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def _text_quality(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    readable = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact))
    replacement = compact.count("�")
    return max(0.0, min(1.0, readable / len(compact) - replacement * 0.01))


def _scan_book(
    task: tuple[str, str, str, float, int | None, int],
) -> dict[str, Any]:
    source_value, cache_value, book, zoom, max_pages, ocr_threads = task
    source = Path(source_value)
    cache_path = Path(cache_value)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_page_cache(cache_path)
    import fitz

    document = fitz.open(source)
    page_total = min(len(document), max_pages) if max_pages else len(document)
    ocr_engine: Any = None
    written = 0
    with cache_path.open("a", encoding="utf-8") as stream:
        for page_index in range(page_total):
            page_number = page_index + 1
            if page_number in cached:
                continue
            page = document[page_index]
            text = page.get_text("text", sort=True).strip()
            quality = _text_quality(text)
            method = "text"
            confidence = 0.99 * quality
            if len(re.sub(r"\s+", "", text)) < 100 or quality < 0.70:
                if ocr_engine is None:
                    from rapidocr_onnxruntime import RapidOCR

                    ocr_engine = RapidOCR(
                        intra_op_num_threads=max(1, ocr_threads),
                        inter_op_num_threads=1,
                    )
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                result, _ = ocr_engine(image)
                if result:
                    text = "\n".join(str(item[1]) for item in result)
                    scores = [float(item[2]) for item in result]
                    confidence = float(np.mean(scores)) if scores else 0.0
                else:
                    text, confidence = "", 0.0
                method = "ocr"
            record = PageRecord(
                book=book,
                source_file=str(source),
                page=page_number,
                text=text,
                confidence=confidence,
                method=method,
            )
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
            if written % 10 == 0:
                print(
                    f"[{book}] {page_number}/{page_total} pages cached",
                    flush=True,
                )
    return {
        "book": book,
        "source_file": str(source),
        "pages": page_total,
        "cache_path": str(cache_path),
    }


def _detect_named(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    lowered = re.sub(r"\s+", " ", text.lower())
    return sorted(
        key
        for key, aliases in mapping.items()
        if any(re.sub(r"\s+", " ", alias.lower()) in lowered for alias in aliases)
    )


def _last_named(text: str, mapping: dict[str, tuple[str, ...]]) -> str | None:
    """Return the most recently mentioned canonical name."""
    lowered = re.sub(r"\s+", " ", text.lower())
    best: tuple[int, int, str] | None = None
    for key, aliases in mapping.items():
        for alias in aliases:
            normalized = re.sub(r"\s+", " ", alias.lower())
            position = lowered.rfind(normalized)
            candidate = (position, len(normalized), key)
            if position >= 0 and (best is None or candidate > best):
                best = candidate
    return best[2] if best else None


def _direction(text: str, patterns: list[str]) -> str:
    bullish = len(
        re.findall(
            r"看涨|见底|买进|回升|向上突破|多方|上涨|bullish|buy|upward breakout",
            text,
            re.IGNORECASE,
        )
    )
    bearish = len(
        re.findall(
            r"看跌|见顶|卖出|离场|向下突破|空方|下跌|bearish|sell|downward breakout",
            text,
            re.IGNORECASE,
        )
    )
    if bullish == bearish:
        top_patterns = {
            "head_and_shoulders_top",
            "double_top",
            "triple_top",
            "rounding_top",
            "inverted_v_top",
            "island_reversal_top",
            "rising_wedge",
            "descending_triangle",
            "three_black_crows",
            "bearish_engulfing",
            "evening_star",
            "evening_doji_star",
            "hanging_man",
            "shooting_star",
            "dark_cloud_cover",
            "tower_top",
            "tweezers_top",
            "meeting_lines_bearish",
            "three_gap_up",
            "high_price_escape",
            "two_crows",
            "three_falling_candles",
            "death_cross",
            "death_valley",
            "ma_short_alignment",
            "large_bearish",
            "high_five_bearish",
            "low_coiling",
            "continuous_decline",
            "slow_decline",
            "decline_continuation",
            "falling_resistance",
            "falling_three_stars",
            "rise_blocked",
            "rise_stalled",
            "descending_cover",
            "short_party_pioneer",
            "falling_three_methods",
            "abcd_bearish",
            "bat_bearish",
            "big_m",
            "bump_and_run_top",
            "butterfly_bearish",
            "crab_bearish",
            "inverted_cup_with_handle",
            "gartley_bearish",
            "horn_top",
            "measured_move_down",
            "pipe_top",
            "roof",
            "three_falling_peaks",
            "three_peaks_domed_house",
            "v_top",
            "extended_v_top",
            "wolfe_wave_bearish",
        }
        bottom_patterns = {
            "head_and_shoulders_bottom",
            "double_bottom",
            "triple_bottom",
            "rounding_bottom",
            "v_bottom",
            "island_reversal_bottom",
            "falling_wedge",
            "ascending_triangle",
            "red_three_soldiers",
            "bullish_engulfing",
            "morning_star",
            "morning_doji_star",
            "hammer",
            "inverted_hammer",
            "piercing_pattern",
            "tower_bottom",
            "tweezers_bottom",
            "meeting_lines_bullish",
            "three_gap_down",
            "downside_recovery",
            "golden_cross",
            "silver_valley",
            "golden_valley",
            "dragon_out_of_sea",
            "ma_long_alignment",
            "large_bullish",
            "low_parallel_bullish",
            "low_five_bullish",
            "gradual_rise",
            "slow_rise",
            "steady_rise",
            "rising_resistance",
            "rising_two_stars",
            "high_parallel_bullish",
            "upside_gap_tasuki",
            "multi_party_pioneer",
            "rising_three_methods",
            "abcd_bullish",
            "bat_bullish",
            "big_w",
            "bump_and_run_bottom",
            "butterfly_bullish",
            "crab_bullish",
            "cup_with_handle",
            "gartley_bullish",
            "horn_bottom",
            "measured_move_up",
            "pipe_bottom",
            "inverted_roof",
            "three_rising_valleys",
            "wolfe_wave_bullish",
        }
        bullish += len(set(patterns) & bottom_patterns)
        bearish += len(set(patterns) & top_patterns)
    return (
        "bullish"
        if bullish > bearish
        else "bearish"
        if bearish > bullish
        else "neutral"
    )


def _is_contextual_direction(text: str) -> bool:
    """Recognize examples whose lesson is explicitly position-dependent."""
    has_bullish = bool(
        re.search(r"看涨|见底|买进|回升|bullish|buy", text, re.IGNORECASE)
    )
    has_bearish = bool(
        re.search(r"看跌|见顶|卖出|离场|bearish|sell", text, re.IGNORECASE)
    )
    has_context_clause = bool(
        re.search(
            r"并不是|不能只|不可只|要结合|根据.*(?:趋势|位置)|"
            r"在.*(?:低位|底部).*(?:高位|顶部)|"
            r"在.*(?:高位|顶部).*(?:低位|底部)|"
            r"depending on|context|location",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    return has_bullish and has_bearish and has_context_clause


def _context(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    trend = (
        "up"
        if re.search(r"上涨趋势|涨势中|上升途中|uptrend|rising trend", lowered)
        else "down"
        if re.search(r"下跌趋势|跌势中|下降途中|downtrend|declining trend", lowered)
        else "range"
        if re.search(r"盘整|横盘|震荡|trading range|consolidation", lowered)
        else "unknown"
    )
    position = (
        "high"
        if re.search(r"高位|顶部|上涨末端|at the top|near the high", lowered)
        else "low"
        if re.search(r"低位|底部|下跌末端|at the bottom|near the low", lowered)
        else "mid"
    )
    timeframe = (
        "monthly"
        if re.search(r"月K|月线|monthly", text, re.IGNORECASE)
        else "weekly"
        if re.search(r"周K|周线|weekly", text, re.IGNORECASE)
        else "daily"
    )
    return trend, position, timeframe


def _state(text: str) -> str:
    if re.search(r"失效|不成立|假突破|invalidat|false breakout", text, re.IGNORECASE):
        return "invalidated"
    if re.search(r"正式成立|确认有效|有效突破|confirmed|完成了", text, re.IGNORECASE):
        return "confirmed"
    if re.search(r"不能|不宜|谨慎|观察|待确认|可能|candidate", text, re.IGNORECASE):
        return "degraded"
    return "candidate"


def _invalidation_tags(text: str) -> list[str]:
    tags: list[str] = []
    lowered = text.lower()
    for tag, expressions in {
        "reclaim_breakout_level": ("重新站上", "收回颈线", "reclaim"),
        "return_inside_pattern": ("重新进入", "回到形态", "return inside"),
        "break_opposite_boundary": ("反向突破", "跌破支撑", "突破阻力"),
        "gap_refilled": ("缺口回补", "缺口封闭", "gap filled"),
        "no_follow_through": ("不能继续", "未能延续", "no follow-through"),
    }.items():
        if any(expression in lowered for expression in expressions):
            tags.append(tag)
    return tags


def _compile_segment(
    book: str,
    source_file: str,
    label: str,
    pages: list[int],
    text: str,
    average_confidence: float,
    methods: set[str],
    source_kind: str = "explicit_exercise",
    forced_patterns: list[str] | None = None,
    curated_override: dict[str, Any] | None = None,
) -> CompiledCase | None:
    if not CASE_LANGUAGE.search(text):
        return None
    if re.search(r"(?:^|\n)\s*目录\s*(?:\n|$)", text):
        return None
    if DEFINITION_LANGUAGE.search(text) and not re.search(
        r"参考答案|实例|案例|case study|sample trade", text, re.IGNORECASE
    ):
        return None
    raw_patterns = sorted(
        set(
            (curated_override or {}).get("patterns")
            or forced_patterns
            or _detect_named(text, PATTERN_ALIASES)
        )
    )
    patterns, pattern_variants = _canonicalize_pattern_ids(raw_patterns)
    evidence = _detect_named(text, EVIDENCE_ALIASES)
    evidence = sorted(set(evidence) | set((curated_override or {}).get("evidence", [])))
    max_patterns = 4 if source_kind == "figure_analysis" else 5
    if not patterns or len(patterns) > max_patterns:
        return None
    direction = str(
        (curated_override or {}).get("direction") or _direction(text, patterns)
    )
    contextual_direction = bool(
        (curated_override or {}).get(
            "context_dependent",
            _is_contextual_direction(text),
        )
    )
    if contextual_direction:
        direction = "neutral"
    if direction == "neutral" and not contextual_direction:
        return None
    trend, position, timeframe = _context(text)
    if position == "mid":
        if any(
            pattern.endswith("_top")
            or pattern
            in {
                "big_m",
                "inverted_v_top",
                "v_top",
                "extended_v_top",
                "three_falling_peaks",
                "three_peaks_domed_house",
            }
            for pattern in patterns
        ):
            position = "high"
        elif any(
            pattern.endswith("_bottom")
            or pattern
            in {
                "big_w",
                "v_bottom",
                "extended_v_bottom",
                "three_rising_valleys",
            }
            for pattern in patterns
        ):
            position = "low"
    state = str((curated_override or {}).get("state") or _state(text))
    has_answer = bool(
        re.search(
            r"参考答案|分析如下|实例|案例|case study|sample trade|result",
            text,
            re.IGNORECASE,
        )
    )
    conflict = []
    if re.search(
        r"虽然|但是|然而|相反|冲突|不能仅|despite|however", text, re.IGNORECASE
    ):
        conflict.append("competing_evidence")
    conflict = sorted(set(conflict) | set((curated_override or {}).get("conflict", [])))
    pattern_family = (
        "chart"
        if set(patterns) & CHART_PATTERNS
        else "combination"
        if any(
            pattern in patterns
            for pattern in (
                "red_three_soldiers",
                "three_black_crows",
                "bullish_engulfing",
                "bearish_engulfing",
                "morning_star",
                "evening_star",
                "engulfing",
                "harami",
                "dark_cloud_cover",
                "piercing_pattern",
                "rising_three_methods",
                "falling_three_methods",
            )
        )
        else "candlestick"
    )
    quality = (
        0.18
        + 0.22 * bool(has_answer)
        + 0.22 * min(1.0, len(patterns) / 2)
        + 0.12 * bool(evidence)
        + 0.10 * (direction != "neutral")
        + 0.10 * average_confidence
        + 0.06 * bool(label)
    )
    if quality < 0.52:
        return None
    if curated_override:
        quality = max(quality, 0.98)
    interpretation = (
        f"{state}_{direction}_{pattern_family}"
        if direction != "neutral"
        else f"{state}_context_dependent_{pattern_family}"
    )
    priority = (
        "structural"
        if pattern_family == "chart"
        else "contextual"
        if conflict
        else "local"
    )
    normalized_identity = re.sub(
        r"\W+", "", f"{book}|{label}|{'|'.join(patterns)}|{pages}"
    )
    case_id = hashlib.sha256(normalized_identity.encode("utf-8")).hexdigest()[:24]
    return CompiledCase(
        case_id=case_id,
        source_book=book,
        source_file=source_file,
        pdf_pages=pages,
        source_label=label,
        patterns=patterns,
        pattern_family=pattern_family,
        trend_context=trend,
        position_context=position,
        timeframe=timeframe,
        evidence_tags=evidence,
        conflict_tags=conflict,
        correct_direction=direction,
        correct_state=state,
        interpretation_code=interpretation,
        priority_tier=priority,
        invalidation_tags=sorted(
            set(_invalidation_tags(text))
            | set((curated_override or {}).get("invalidation", []))
        ),
        quality_score=min(1.0, quality),
        compiler_version=COMPILER_VERSION,
        metadata={
            "has_reference_analysis": has_answer,
            "source_kind": source_kind,
            "context_dependent_direction": contextual_direction,
            "curated_override": bool(curated_override),
            "internal_source_methods": sorted(methods),
            "source_ocr_confidence": round(average_confidence, 4),
            "decision_basis": {
                "patterns": patterns,
                "pattern_variants": pattern_variants,
                "evidence": evidence,
                "trend": trend,
                "position": position,
                "conflicts": conflict,
                "invalidation": sorted(
                    set(_invalidation_tags(text))
                    | set((curated_override or {}).get("invalidation", []))
                ),
            },
        },
    )


def _compile_book(cache_path: Path) -> tuple[list[CompiledCase], list[dict[str, Any]]]:
    records = sorted(_load_page_cache(cache_path).values(), key=lambda item: item.page)
    if not records:
        return [], []
    if records[0].book.startswith("Encyclopedia of Chart Patterns"):
        records = [record for record in records if record.page >= 109]
    if not records:
        return [], []
    joined_parts: list[str] = []
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for record in records:
        part = f"\n[[PAGE:{record.page}]]\n{record.text}\n"
        joined_parts.append(part)
        page_offsets.append((cursor, cursor + len(part), record.page))
        cursor += len(part)
    joined = "".join(joined_parts)
    markers = list(CASE_MARKER.finditer(joined))
    cases: list[CompiledCase] = []
    rejected: list[dict[str, Any]] = []
    covered_pages: set[int] = set()
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(joined)
        end = min(end, marker.start() + 18000)
        segment = joined[marker.start() : end]
        touched = [
            page
            for start, stop, page in page_offsets
            if start < end and stop > marker.start()
        ]
        if not touched:
            continue
        touched = list(range(min(touched), min(max(touched), min(touched) + 4) + 1))
        source_records = [record for record in records if record.page in touched]
        confidence = float(np.mean([record.confidence for record in source_records]))
        label = marker.group("label").strip()
        is_sample_trade = label.lower().startswith("sample trade")
        forced = (
            [_last_named(joined[: marker.start()], PATTERN_ALIASES)]
            if is_sample_trade
            else None
        )
        forced = [item for item in (forced or []) if item]
        case = _compile_segment(
            records[0].book,
            records[0].source_file,
            label,
            touched,
            segment,
            confidence,
            {record.method for record in source_records},
            source_kind="sample_trade" if is_sample_trade else "explicit_exercise",
            forced_patterns=forced,
            curated_override=CURATED_CASE_OVERRIDES.get((records[0].book, label)),
        )
        if case is None:
            rejected.append(
                {
                    "book": records[0].book,
                    "source_file": records[0].source_file,
                    "pdf_pages": touched,
                    "source_label": marker.group("label").strip(),
                    "reason": "未同时识别出案例分析语言和明确技术形态",
                    "ocr_confidence": round(confidence, 4),
                }
            )
        else:
            cases.append(case)
            covered_pages.update(touched)

    # Many practical examples in Nison/Bulkowski and the narrative sections of
    # 股市操练大全 are headed only by a figure number. Compile those chart-plus-
    # analysis pages too, while avoiding pages already captured as exercises.
    for index, record in enumerate(records):
        if record.page in covered_pages:
            continue
        if not FIGURE_MARKER.search(record.text) or not CASE_LANGUAGE.search(
            record.text
        ):
            continue
        if DEFINITION_LANGUAGE.search(record.text):
            continue
        sentence_count = len(re.findall(r"[。！？]|(?:\.\s)", record.text))
        if sentence_count < 2:
            continue
        window_records = records[index : min(len(records), index + 2)]
        window_text = "\n".join(item.text for item in window_records)
        figures = FIGURE_MARKER.findall(record.text)
        label = figures[0] if figures else f"page-{record.page}"
        pages = [item.page for item in window_records]
        confidence = float(np.mean([item.confidence for item in window_records]))
        case = _compile_segment(
            record.book,
            record.source_file,
            label,
            pages,
            window_text,
            confidence,
            {item.method for item in window_records},
            source_kind="figure_analysis",
        )
        if case is None:
            # Only send genuinely case-like pages to review; definitions remain
            # excluded entirely as requested.
            if _detect_named(window_text, PATTERN_ALIASES):
                rejected.append(
                    {
                        "book": record.book,
                        "source_file": record.source_file,
                        "pdf_pages": pages,
                        "source_label": label,
                        "reason": "图例页包含形态，但无法形成明确、单向的结构化判断",
                        "ocr_confidence": round(confidence, 4),
                    }
                )
        else:
            cases.append(case)
            covered_pages.update(pages)

    unique: dict[tuple[str, tuple[int, ...], tuple[str, ...]], CompiledCase] = {}
    for case in cases:
        key = (case.source_book, tuple(case.pdf_pages), tuple(case.patterns))
        previous = unique.get(key)
        if previous is None or case.quality_score > previous.quality_score:
            unique[key] = case
    return list(unique.values()), rejected


def _inventory(source_dir: Path, encyclopedia: Path | None) -> list[tuple[str, Path]]:
    files = sorted(source_dir.rglob("*.pdf"))
    selected: list[tuple[str, Path]] = []
    seen_hashes: set[str] = set()
    for path in files:
        name = path.name
        if "第7册" in name and (
            "1-300" in name or "300-524" in name or "拆分" in str(path)
        ):
            continue
        if (
            name.startswith("《股市操练大全第7册")
            and name != "《股市操练大全第7册12201723.pdf"
        ):
            continue
        stat = path.stat()
        identity = f"{stat.st_size}:{name}"
        if identity in seen_hashes:
            continue
        seen_hashes.add(identity)
        selected.append((path.stem, path))
    if encyclopedia and encyclopedia.exists():
        selected.append(("Encyclopedia of Chart Patterns 3rd", encyclopedia))
    return selected


def _write_outputs(
    output_dir: Path,
    scan_results: list[dict[str, Any]],
) -> dict[str, Any]:
    all_cases: dict[str, CompiledCase] = {}
    rejected: list[dict[str, Any]] = []
    by_book: dict[str, int] = {}
    for result in scan_results:
        cases, book_rejected = _compile_book(Path(result["cache_path"]))
        rejected.extend(book_rejected)
        for case in cases:
            all_cases.setdefault(case.case_id, case)
        by_book[result["book"]] = len(cases)
    cases = sorted(
        all_cases.values(),
        key=lambda item: (item.source_book, item.pdf_pages[0], item.case_id),
    )
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    library_path = cases_dir / "compiled_cases.jsonl"
    with library_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    review_path = cases_dir / "case_compilation_review.jsonl"
    with review_path.open("w", encoding="utf-8") as stream:
        for item in rejected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    review_markdown = [
        "# 教材案例人工复核清单",
        "",
        "这里只列出无法可靠编译为 FAE 结构化判断的案例候选，不包含教材正文。",
        "页码均为 PDF 页码，可按书名、页码回到原书人工筛选。",
        "",
        "| 教材 | PDF 页码 | 标题/图号 | 原因 | OCR 置信度 |",
        "|---|---:|---|---|---:|",
    ]
    for item in rejected:
        pages = ", ".join(str(page) for page in item["pdf_pages"])
        cells = [
            str(item["book"]),
            pages,
            str(item["source_label"]),
            str(item["reason"]),
            f"{float(item['ocr_confidence']):.2%}",
        ]
        review_markdown.append(
            "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        )
    review_markdown.append("")
    (cases_dir / "case_compilation_review.md").write_text(
        "\n".join(review_markdown),
        encoding="utf-8",
    )
    pattern_counts: dict[str, int] = {}
    for case in cases:
        for pattern in case.patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    quality_buckets = {
        "engine_eligible_gte_0_65": sum(case.quality_score >= 0.65 for case in cases),
        "stored_but_review_only_lt_0_65": sum(
            case.quality_score < 0.65 for case in cases
        ),
    }
    summary = {
        "generated": datetime.now().astimezone().isoformat(),
        "compiler_version": COMPILER_VERSION,
        "case_count": len(cases),
        "review_count": len(rejected),
        "book_count": len(scan_results),
        "scanned_page_count": sum(int(item["pages"]) for item in scan_results),
        "by_book": by_book,
        "quality_buckets": quality_buckets,
        "by_pattern": dict(
            sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "library": str(library_path),
        "manual_review_document": str(cases_dir / "case_compilation_review.md"),
        "policy": {
            "raw_ocr_text_is_internal_only": True,
            "concept_and_definition_pages_are_excluded": True,
            "existing_expert_rules_are_not_modified": True,
        },
    }
    (cases_dir / "case_library_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    """Run resumable textbook scanning and compile structured FAE cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\Users\18295\Desktop\开发资料\知识库 形态"),
    )
    parser.add_argument(
        "--encyclopedia",
        type=Path,
        default=Path(
            r"C:\Users\18295\Desktop\开发资料"
            r"\(NEW)Encyclopedia of Chart Patterns , 3rd Edition "
            r"(Thomas N. Bulkowski) 9781119739685.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--ocr-threads", type=int, default=4)
    parser.add_argument("--zoom", type=float, default=0.75)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--book-filter", action="append", default=[])
    args = parser.parse_args()
    inventory = _inventory(args.source_dir, args.encyclopedia)
    if args.book_filter:
        filters = [item.lower() for item in args.book_filter]
        inventory = [
            item
            for item in inventory
            if any(value in item[0].lower() for value in filters)
        ]
    cache_dir = args.output_dir / ".case_build_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(path),
            str(cache_dir / _safe_cache_name(path)),
            book,
            args.zoom,
            args.max_pages,
            args.ocr_threads,
        )
        for book, path in inventory
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_scan_book, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[DONE] {result['book']}: {result['pages']} pages",
                flush=True,
            )
    summary = _write_outputs(args.output_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
