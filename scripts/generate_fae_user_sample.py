"""Generate a FAE mobile-product preview sample.

This is a local visual test only.  It deliberately does not import, write, or
patch the WeChat mini-program.  The renderer consumes the same presentation
contract used by the mini-program: ``display.default`` is the adaptive
first-screen set and the full FAE result remains available for layer toggles
and detail inspection.

The default product-review set is ten representative assets, each with a
monthly, weekly, and daily chart (thirty PNGs total).  A smaller three-asset
set remains available for a quick smoke run.  This is not an audit dump; the
audit renderer remains available in ``generate_fae_audit_batch.py``.

Usage from the project root::

    python scripts/generate_fae_user_sample.py --asset-set 10

Output: ``fae/preview/user_sample_10``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fae.judgment import JudgmentEngine, Timeframe, from_v7_events
from kline_pattern_report import fetch_tencent_daily
from pattern_core_v7 import detect_all
from run_fae_sh000001_preview import (
    BAR_LIMIT,
    _configure_font,
    _period_frame,
    _plot_period,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OUTPUT_DIR_3 = PROJECT_ROOT / "fae" / "preview" / "user_sample_3"
OUTPUT_DIR_10 = PROJECT_ROOT / "fae" / "preview" / "user_sample_10"

# Keep this list stable so feedback can reference an image by number.
USER_SAMPLE_ASSETS_3: list[dict[str, str]] = [
    {"market": "A股", "symbol": "sh000001", "name": "上证指数"},
    {"market": "港股", "symbol": "hk00700", "name": "腾讯控股"},
    {"market": "美股", "symbol": "usTSLA", "name": "Tesla"},
]

# This fixed order is part of the review contract: feedback can refer to an
# image as ``008_usTSLA_daily_user.png`` without relying on generation order.
USER_SAMPLE_ASSETS_10: list[dict[str, str]] = [
    {"market": "A股", "symbol": "sh000001", "name": "上证指数"},
    {"market": "A股", "symbol": "sh688981", "name": "中芯国际"},
    {"market": "A股", "symbol": "sz300750", "name": "宁德时代"},
    {"market": "港股", "symbol": "hk00700", "name": "腾讯控股"},
    {"market": "港股", "symbol": "hk09988", "name": "阿里巴巴-SW"},
    {"market": "美股", "symbol": "usAAPL", "name": "苹果"},
    {"market": "美股", "symbol": "usNVDA", "name": "英伟达"},
    {"market": "美股", "symbol": "usTSLA", "name": "特斯拉"},
    {"market": "美股", "symbol": "usJPM", "name": "小摩"},
    {"market": "美股", "symbol": "usSPY", "name": "标普500 ETF"},
]


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def _compact_period_result(result: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Keep a small machine-readable product preview alongside each image."""
    display = result.get("display", {})
    preferences = display.get("display_preferences", {})
    return {
        "timeframe": summary.get("timeframe"),
        "image_filename": Path(str(summary.get("image", ""))).name,
        "winner_pattern_id": summary.get("winner_pattern_id"),
        "winner_name": summary.get("winner_name"),
        "default_count": int(display.get("summary", {}).get("default_count", 0)),
        "displayed_items": summary.get("displayed_items", []),
        "display_preferences": {
            "mode": preferences.get("mode"),
            "default_policy": preferences.get("default_policy"),
            "adaptive_default_limit": preferences.get("adaptive_default_limit"),
            "available_kinds": preferences.get("available_kinds", []),
            "mobile_note": preferences.get("mobile_note"),
        },
        "raw_event_count": summary.get("raw_event_count", 0),
        "accepted_signal_count": summary.get("accepted_signal_count", 0),
    }


def generate_sample(
    *,
    years: int = 12,
    output_dir: Path = OUTPUT_DIR_10,
    assets: list[dict[str, str]] | None = None,
    sample_id: str = "user_sample_10",
) -> dict[str, Any]:
    """Fetch, evaluate, and render a user-facing M/W/D sample."""
    _configure_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()
    engine = JudgmentEngine()
    selected_assets = list(assets or USER_SAMPLE_ASSETS_10)
    assets_payload: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for asset_index, asset in enumerate(selected_assets, start=1):
        symbol = asset["symbol"]
        try:
            daily = fetch_tencent_daily(symbol, years=years)
            if daily.empty:
                raise ValueError("接口返回空数据")
            as_of_date = pd.to_datetime(daily["Date"]).max()
        except Exception as exc:  # pragma: no cover - network-dependent
            failures.append(
                {
                    "asset_id": f"{asset_index:03d}",
                    "asset": asset,
                    "status": "fetch_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{asset_index:03d}] 获取失败：{symbol} - {exc}", flush=True)
            continue

        asset_record: dict[str, Any] = {
            "asset_id": f"{asset_index:03d}",
            "market": asset["market"],
            "symbol": symbol,
            "name": asset["name"],
            "data_as_of": as_of_date.strftime("%Y-%m-%d"),
            "periods": [],
        }

        for timeframe in (Timeframe.MONTHLY, Timeframe.WEEKLY, Timeframe.DAILY):
            frame = _period_frame(daily, timeframe)
            if frame.empty:
                failures.append(
                    {
                        "asset_id": f"{asset_index:03d}",
                        "asset": asset,
                        "timeframe": timeframe.value,
                        "status": "empty_period",
                    }
                )
                continue
            events = detect_all(frame, max_trend_per_direction=3)
            candlestick_signals = from_v7_events(events, timeframe=timeframe)
            result = engine.evaluate(
                frame,
                candlestick_signals=candlestick_signals,
                timeframe=timeframe,
                display_mode="adaptive",
                context={
                    "symbol": symbol,
                    "asset_id": asset_index,
                    "market": asset["market"],
                    "preview": True,
                    "product_surface": "mini_program_default",
                },
            )
            image_name = (
                f"{asset_index:03d}_{_safe_filename(symbol)}_{timeframe.value}_user.png"
            )
            image_path = output_dir / image_name
            summary = _plot_period(
                frame,
                result,
                timeframe,
                image_path,
                as_of_date,
                f"{asset_index:03d} | {asset['market']} | {asset['name']} | {symbol}",
                mode="user",
            )
            summary["asset_id"] = f"{asset_index:03d}"
            summary["market"] = asset["market"]
            summary["symbol"] = symbol
            summary["name"] = asset["name"]
            summary["image_filename"] = image_name
            asset_record["periods"].append(_compact_period_result(result, summary))
            print(
                f"[{asset_index:03d}] 已生成：{asset['market']} {symbol} {timeframe.value} 用户体验图",
                flush=True,
            )

        # The compact JSON is enough for product review and does not duplicate
        # the much larger audit JSON files.
        (output_dir / f"{asset_index:03d}_{_safe_filename(symbol)}_user.json").write_text(
            json.dumps(asset_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        assets_payload.append(asset_record)

    manifest = {
        "sample_id": sample_id,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "render_mode": "user",
        "surface_contract": {
            "default": "只呈现 FAE display.default 的自适应重点，不机械补满五条",
            "manual_layers": "完整候选保留在 FAE display.manual_layers，供前端图层开关展开",
            "frontend_source_read": "C:/Users/18295/WeChatProjects/miniprogram-1/miniprogram",
            "frontend_modified": False,
            "backend_modified": False,
        },
        "asset_count_requested": len(selected_assets),
        "asset_count_completed": len(assets_payload),
        "image_count_expected": len(selected_assets) * 3,
        "image_count_generated": sum(len(item["periods"]) for item in assets_payload),
        "bar_limit_per_timeframe": BAR_LIMIT,
        "assets": assets_payload,
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"# FAE 用户体验版样例（{len(selected_assets)} 个资产 × 月/周/日）\n\n"
        "本目录是本地测试渲染，不修改小程序前端或后端。默认展示对应 FAE 的 `display.default`：\n"
        "只保留经过周期位置、完整性、冲突消解和自适应排序后的重点，不为了凑数强行显示五条。\n"
        "简单形态、组合形态、趋势形态、技术图形、缺口、波段、趋势线、均线、支撑/阻力仍可由前端图层开关展开。\n\n"
        "## 图片编号\n\n"
        "图片编号按 manifest 中的固定资产顺序；每个编号各有 `monthly_user.png`、`weekly_user.png`、`daily_user.png`。\n"
        "`manifest.json` 和同编号的 `_user.json` 保存首屏条目、主结论、周期和可用图层信息。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the FAE mobile user sample.")
    parser.add_argument("--years", type=int, default=12, help="Years of daily history")
    parser.add_argument(
        "--asset-set",
        choices=("3", "10"),
        default="10",
        help="Fixed review set: 3 for smoke test or 10 for the product sample",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    if args.years < 2:
        raise SystemExit("--years 至少为 2")
    assets = USER_SAMPLE_ASSETS_3 if args.asset_set == "3" else USER_SAMPLE_ASSETS_10
    sample_id = f"user_sample_{args.asset_set}"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (OUTPUT_DIR_3 if args.asset_set == "3" else OUTPUT_DIR_10)
    )
    payload = generate_sample(
        years=int(args.years),
        output_dir=output_dir,
        assets=assets,
        sample_id=sample_id,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "image_count_generated": payload["image_count_generated"],
                "asset_count_completed": payload["asset_count_completed"],
                "failed_count": len(payload["failures"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
