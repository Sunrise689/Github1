"""Generate the 100-asset FAE manual-calibration audit set.

This script is intentionally separate from the mini-program.  It fetches
daily OHLCV data through the existing local/Tencent/Yahoo helpers, evaluates
the FAE judgment layer at monthly, weekly and daily frequency, and writes:

* 300 numbered PNGs (100 assets x 3 timeframes);
* one manifest that maps every image number to an asset and period;
* one detailed JSON file per asset, including candidates, winners,
  conflict-suppressed signals and quantitative-filter rejections.

The renderer uses ``mode="audit"``.  Monthly views retain a high quota of
non-duplicate signals because monthly bars are sparse.  Weekly and daily views
remain bounded so a reviewer can still read the chart; the unbounded result is
always preserved in the JSON files.

Usage from the project root::

    python scripts/generate_fae_audit_batch.py
    python scripts/generate_fae_audit_batch.py --limit 5

The output is written to ``fae/preview/audit_batch_100``.  The script never
touches the mini-program frontend or backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from kline_pattern_report import fetch_tencent_daily, resample_monthly, resample_weekly
from pattern_core_v7 import detect_all
from run_fae_sh000001_preview import (
    BAR_LIMIT,
    _configure_font,
    _period_frame,
    _plot_period,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OUTPUT_DIR = PROJECT_ROOT / "fae" / "preview" / "audit_batch_100"

# The short validation run requested after the rule revision.  It deliberately
# spans indices, A/HK large caps, US technology, finance and a broad ETF so
# the reviewer sees different volatility and market conventions without
# repeating the 100-asset batch.
GLOBAL_REPRESENTATIVE_ASSETS: list[dict[str, str]] = [
    {"market": "A股", "symbol": "sh000001", "name": "上证指数"},
    {"market": "A股", "symbol": "sh688981", "name": "中芯国际"},
    {"market": "A股", "symbol": "sz300750", "name": "宁德时代"},
    {"market": "港股", "symbol": "hk00700", "name": "腾讯控股"},
    {"market": "港股", "symbol": "hk09988", "name": "阿里巴巴-SW"},
    {"market": "美股", "symbol": "usAAPL", "name": "Apple"},
    {"market": "美股", "symbol": "usNVDA", "name": "NVIDIA"},
    {"market": "美股", "symbol": "usTSLA", "name": "Tesla"},
    {"market": "美股", "symbol": "usJPM", "name": "JPMorgan Chase"},
    {"market": "美股", "symbol": "usSPY", "name": "SPDR标普500 ETF"},
]


# A representative, deliberately fixed universe.  Keeping the list in source
# control makes the image numbers stable: a reviewer can say "A017 日K" and
# the same asset will remain A017 on a later rerun.  The selection covers broad
# indices, banks, consumer, energy, healthcare, semiconductors, new energy,
# internet/platform companies and major US/HK names.
ASSET_UNIVERSE: list[dict[str, str]] = [
    # A-share indices and large-cap/sector representatives (001-050)
    {"market": "A股", "symbol": "sh000001", "name": "上证指数"},
    {"market": "A股", "symbol": "sz399001", "name": "深证成指"},
    {"market": "A股", "symbol": "sz399006", "name": "创业板指"},
    {"market": "A股", "symbol": "sh000688", "name": "科创50"},
    {"market": "A股", "symbol": "sh000300", "name": "沪深300"},
    {"market": "A股", "symbol": "sh000905", "name": "中证500"},
    {"market": "A股", "symbol": "sh000852", "name": "中证1000"},
    {"market": "A股", "symbol": "sh601318", "name": "中国平安"},
    {"market": "A股", "symbol": "sh600519", "name": "贵州茅台"},
    {"market": "A股", "symbol": "sh600036", "name": "招商银行"},
    {"market": "A股", "symbol": "sh601398", "name": "工商银行"},
    {"market": "A股", "symbol": "sh601288", "name": "农业银行"},
    {"market": "A股", "symbol": "sh601988", "name": "中国银行"},
    {"market": "A股", "symbol": "sh601939", "name": "建设银行"},
    {"market": "A股", "symbol": "sh601857", "name": "中国石油"},
    {"market": "A股", "symbol": "sh600028", "name": "中国石化"},
    {"market": "A股", "symbol": "sh601088", "name": "中国神华"},
    {"market": "A股", "symbol": "sh600900", "name": "长江电力"},
    {"market": "A股", "symbol": "sh601012", "name": "隆基绿能"},
    {"market": "A股", "symbol": "sh688981", "name": "中芯国际"},
    {"market": "A股", "symbol": "sh688041", "name": "海光信息"},
    {"market": "A股", "symbol": "sh688012", "name": "中微公司"},
    {"market": "A股", "symbol": "sh688111", "name": "金山办公"},
    {"market": "A股", "symbol": "sh688256", "name": "寒武纪"},
    {"market": "A股", "symbol": "sz300750", "name": "宁德时代"},
    {"market": "A股", "symbol": "sz000858", "name": "五粮液"},
    {"market": "A股", "symbol": "sz000333", "name": "美的集团"},
    {"market": "A股", "symbol": "sz000651", "name": "格力电器"},
    {"market": "A股", "symbol": "sz000001", "name": "平安银行"},
    {"market": "A股", "symbol": "sz002594", "name": "比亚迪"},
    {"market": "A股", "symbol": "sz300059", "name": "东方财富"},
    {"market": "A股", "symbol": "sz000725", "name": "京东方A"},
    {"market": "A股", "symbol": "sz000063", "name": "中兴通讯"},
    {"market": "A股", "symbol": "sh600030", "name": "中信证券"},
    {"market": "A股", "symbol": "sh601166", "name": "兴业银行"},
    {"market": "A股", "symbol": "sh600276", "name": "恒瑞医药"},
    {"market": "A股", "symbol": "sh600309", "name": "万华化学"},
    {"market": "A股", "symbol": "sh600887", "name": "伊利股份"},
    {"market": "A股", "symbol": "sh600031", "name": "三一重工"},
    {"market": "A股", "symbol": "sh600089", "name": "特变电工"},
    {"market": "A股", "symbol": "sh601899", "name": "紫金矿业"},
    {"market": "A股", "symbol": "sh600111", "name": "北方稀土"},
    {"market": "A股", "symbol": "sh600438", "name": "通威股份"},
    {"market": "A股", "symbol": "sh601600", "name": "中国铝业"},
    {"market": "A股", "symbol": "sz002230", "name": "科大讯飞"},
    {"market": "A股", "symbol": "sz002415", "name": "海康威视"},
    {"market": "A股", "symbol": "sz300124", "name": "汇川技术"},
    {"market": "A股", "symbol": "sz300274", "name": "阳光电源"},
    {"market": "A股", "symbol": "sh601728", "name": "中国电信"},
    {"market": "A股", "symbol": "sh600570", "name": "恒生电子"},
    # Hong Kong representatives (051-070)
    {"market": "港股", "symbol": "hk00700", "name": "腾讯控股"},
    {"market": "港股", "symbol": "hk09988", "name": "阿里巴巴-SW"},
    {"market": "港股", "symbol": "hk03690", "name": "美团-W"},
    {"market": "港股", "symbol": "hk01810", "name": "小米集团-W"},
    {"market": "港股", "symbol": "hk00941", "name": "中国移动"},
    {"market": "港股", "symbol": "hk01299", "name": "友邦保险"},
    {"market": "港股", "symbol": "hk00005", "name": "汇丰控股"},
    {"market": "港股", "symbol": "hk00388", "name": "香港交易所"},
    {"market": "港股", "symbol": "hk02318", "name": "中国平安"},
    {"market": "港股", "symbol": "hk00939", "name": "建设银行"},
    {"market": "港股", "symbol": "hk01398", "name": "工商银行"},
    {"market": "港股", "symbol": "hk00883", "name": "中国海洋石油"},
    {"market": "港股", "symbol": "hk01088", "name": "中国神华"},
    {"market": "港股", "symbol": "hk02020", "name": "安踏体育"},
    {"market": "港股", "symbol": "hk02269", "name": "药明生物"},
    {"market": "港股", "symbol": "hk09618", "name": "京东集团-SW"},
    {"market": "港股", "symbol": "hk09633", "name": "农夫山泉"},
    {"market": "港股", "symbol": "hk01024", "name": "快手-W"},
    {"market": "港股", "symbol": "hk09961", "name": "携程集团-S"},
    {"market": "港股", "symbol": "hk06690", "name": "海尔智家"},
    # US representatives (071-100); ``us`` is the explicit Yahoo prefix.
    {"market": "美股", "symbol": "usAAPL", "name": "Apple"},
    {"market": "美股", "symbol": "usMSFT", "name": "Microsoft"},
    {"market": "美股", "symbol": "usNVDA", "name": "NVIDIA"},
    {"market": "美股", "symbol": "usAMZN", "name": "Amazon"},
    {"market": "美股", "symbol": "usGOOGL", "name": "Alphabet"},
    {"market": "美股", "symbol": "usMETA", "name": "Meta Platforms"},
    {"market": "美股", "symbol": "usTSLA", "name": "Tesla"},
    {"market": "美股", "symbol": "usAVGO", "name": "Broadcom"},
    {"market": "美股", "symbol": "usAMD", "name": "AMD"},
    {"market": "美股", "symbol": "usNFLX", "name": "Netflix"},
    {"market": "美股", "symbol": "usJPM", "name": "JPMorgan Chase"},
    {"market": "美股", "symbol": "usV", "name": "Visa"},
    {"market": "美股", "symbol": "usMA", "name": "Mastercard"},
    {"market": "美股", "symbol": "usKO", "name": "Coca-Cola"},
    {"market": "美股", "symbol": "usWMT", "name": "Walmart"},
    {"market": "美股", "symbol": "usXOM", "name": "Exxon Mobil"},
    {"market": "美股", "symbol": "usCVX", "name": "Chevron"},
    {"market": "美股", "symbol": "usORCL", "name": "Oracle"},
    {"market": "美股", "symbol": "usINTC", "name": "Intel"},
    {"market": "美股", "symbol": "usQCOM", "name": "Qualcomm"},
    {"market": "美股", "symbol": "usCOST", "name": "Costco"},
    {"market": "美股", "symbol": "usBA", "name": "Boeing"},
    {"market": "美股", "symbol": "usDIS", "name": "Walt Disney"},
    {"market": "美股", "symbol": "usUBER", "name": "Uber"},
    {"market": "美股", "symbol": "usCRM", "name": "Salesforce"},
    {"market": "美股", "symbol": "usADBE", "name": "Adobe"},
    {"market": "美股", "symbol": "usPYPL", "name": "PayPal"},
    {"market": "美股", "symbol": "usNKE", "name": "Nike"},
    {"market": "美股", "symbol": "usSPY", "name": "SPDR标普500 ETF"},
    {"market": "美股", "symbol": "usQQQ", "name": "Invesco纳斯达克100 ETF"},
]


def _numbered_assets(assets: list[dict[str, str]]) -> list[dict[str, Any]]:
    numbered: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        numbered.append({"id": index, **asset})
    return numbered


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _fetch_one(asset: dict[str, Any], years: int) -> tuple[int, pd.DataFrame | None, str | None]:
    try:
        frame = fetch_tencent_daily(str(asset["symbol"]), years=years)
        if frame.empty:
            raise ValueError("接口返回空数据")
        return int(asset["id"]), frame, None
    except Exception as exc:  # pragma: no cover - network-dependent branch
        return int(asset["id"]), None, f"{type(exc).__name__}: {exc}"


def _evaluate_asset(
    asset: dict[str, Any],
    daily: pd.DataFrame,
    output_dir: Path,
    *,
    batch_id: str = "audit_batch_100",
) -> dict[str, Any]:
    asset_id = int(asset["id"])
    symbol = str(asset["symbol"])
    title_name = f"{asset_id:03d} | {asset['market']} | {asset['name']} | {symbol}"
    as_of_date = pd.to_datetime(daily["Date"]).max()
    periods: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {}
    engine = JudgmentEngine()

    for timeframe in (Timeframe.MONTHLY, Timeframe.WEEKLY, Timeframe.DAILY):
        frame = _period_frame(daily, timeframe)
        if frame.empty:
            raise ValueError(f"{timeframe.value} 重采样后没有有效 K 线")
        events = detect_all(frame, max_trend_per_direction=3)
        candlestick_signals = from_v7_events(events, timeframe=timeframe)
        result = engine.evaluate(
            frame,
            candlestick_signals=candlestick_signals,
            timeframe=timeframe,
            context={
                "symbol": symbol,
                "asset_id": asset_id,
                "market": asset["market"],
                "preview": True,
                "audit_batch": batch_id,
            },
        )
        image_name = f"{asset_id:03d}_{_safe_filename(symbol)}_{timeframe.value}_audit.png"
        image_path = output_dir / image_name
        period_summary = _plot_period(
            frame,
            result,
            timeframe,
            image_path,
            as_of_date,
            title_name,
            mode="audit",
        )
        period_summary["asset_id"] = asset_id
        period_summary["market"] = asset["market"]
        period_summary["symbol"] = symbol
        period_summary["name"] = asset["name"]
        period_summary["image_filename"] = image_name
        periods.append(period_summary)
        detailed_results[timeframe.value] = result

    payload = {
        "asset": asset,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_source": "Tencent daily endpoint for A/HK; Yahoo public chart endpoint for US",
        "data_as_of": as_of_date.strftime("%Y-%m-%d"),
        "bar_limit_per_timeframe": BAR_LIMIT,
        "render_mode": "audit",
        "periods": periods,
        "fae_results": detailed_results,
    }
    json_path = output_dir / f"{asset_id:03d}_{_safe_filename(symbol)}_audit.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "asset": asset,
        "status": "ok",
        "data_as_of": as_of_date.strftime("%Y-%m-%d"),
        "json_filename": json_path.name,
        "periods": periods,
    }


def generate_batch(
    *,
    assets: list[dict[str, Any]],
    years: int = 12,
    fetch_workers: int = 3,
    output_dir: Path = OUTPUT_DIR,
    batch_id: str = "audit_batch_100",
) -> dict[str, Any]:
    _configure_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    numbered = _numbered_assets(assets)

    # Persist the exact ordered universe before networking starts.  Even if a
    # data source is temporarily unavailable, the image IDs remain stable and
    # the failure is visible in the manifest rather than silently skipped.
    catalog_path = output_dir / "asset_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "generated_at": started_at.isoformat(timespec="seconds"),
                "asset_count_requested": len(numbered),
                "assets": numbered,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fetched: dict[int, pd.DataFrame] = {}
    failures: list[dict[str, Any]] = []
    print(f"开始获取 {len(numbered)} 个资产的日线数据（并发 {fetch_workers}）...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(fetch_workers))) as pool:
        futures = {pool.submit(_fetch_one, asset, years): asset for asset in numbered}
        for future in as_completed(futures):
            asset = futures[future]
            asset_id, frame, error = future.result()
            if frame is None:
                failures.append({"asset": asset, "status": "fetch_failed", "error": error})
                print(f"[{asset_id:03d}] 获取失败：{asset['symbol']} - {error}", flush=True)
            else:
                fetched[asset_id] = frame
                print(f"[{asset_id:03d}] 获取成功：{asset['market']} {asset['symbol']} {len(frame)} 根", flush=True)

    results: list[dict[str, Any]] = []
    for asset in numbered:
        asset_id = int(asset["id"])
        daily = fetched.get(asset_id)
        if daily is None:
            continue
        try:
            result = _evaluate_asset(asset, daily, output_dir, batch_id=batch_id)
            results.append(result)
            print(
                f"[{asset_id:03d}] 已生成：{asset['market']} {asset['symbol']} "
                f"月/周/日三张审计图",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - data-dependent branch
            failures.append(
                {
                    "asset": asset,
                    "status": "evaluate_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{asset_id:03d}] 分析失败：{asset['symbol']} - {exc}", flush=True)

    results.sort(key=lambda item: int(item["asset"]["id"]))
    failures.sort(key=lambda item: int(item["asset"]["id"]))
    image_count = sum(len(item.get("periods", [])) for item in results)
    manifest = {
        "batch_id": batch_id,
        "generated_at": started_at.isoformat(timespec="seconds"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "render_mode": "audit",
        "rules": {
            "monthly": "保留高配额的非重复信号，尤其保留最近 K 线信号；被大形态覆盖的从属信号仍写入 JSON",
            "weekly": "技术图形不设固定数量上限；第一个候选门槛约65%，第二个约85%，后续继续滚动提高",
            "daily": "技术图形不设固定数量上限；第一个候选门槛约65%，第二个约85%，后续继续滚动提高；原始候选与冲突降级结果写入 JSON",
            "precedence": "已确认的大形态优先于从属小形态；被压制项不从 FAE 结果删除",
            "monthly_chart_policy": "月K不强行推广正式技术图形，若检测层有原始候选也不进入正式图形结果",
            "mobile_default": "前端默认只读取 display.default 的自适应胜出项；其余层通过手动开关展开",
        },
        "asset_count_requested": len(numbered),
        "asset_count_fetched": len(fetched),
        "asset_count_completed": len(results),
        "image_count_expected": len(numbered) * 3,
        "image_count_generated": image_count,
        "failed_assets": failures,
        "assets": results,
        "manual_review": {
            "how_to_reference": "使用三位编号 + 周期，例如 017 日K、083 月K",
            "correction_format": "请指出编号、周期、应保留/删除/改名的形态，以及正确的主从关系",
            "detailed_json_per_asset": "每个资产的同名 _audit.json 保存全部 FAE 候选、winner、suppressed 和冲突理由",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    index_path = output_dir / "review_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "asset_id",
                "market",
                "symbol",
                "name",
                "timeframe",
                "image_filename",
                "winner_pattern_id",
                "winner_name",
                "displayed_item_count",
            ],
        )
        writer.writeheader()
        for asset_result in results:
            asset = asset_result["asset"]
            for period in asset_result.get("periods", []):
                writer.writerow(
                    {
                        "asset_id": f"{int(asset['id']):03d}",
                        "market": asset["market"],
                        "symbol": asset["symbol"],
                        "name": asset["name"],
                        "timeframe": period.get("timeframe"),
                        "image_filename": period.get("image_filename"),
                        "winner_pattern_id": period.get("winner_pattern_id"),
                        "winner_name": period.get("winner_name"),
                        "displayed_item_count": len(period.get("displayed_items", [])),
                    }
                )
    readme = output_dir / "README.md"
    readme.write_text(
        f"# FAE {len(numbered)} 资产人工校准审计集\n\n"
        f"本目录由 `scripts/generate_fae_audit_batch.py` 生成（{batch_id}）。每个资产固定一个三位编号，\n"
        "每个编号对应月 K、周 K、日 K 三张图和一个详细 JSON。\n\n"
        "## 文件命名\n\n"
        "`001_sh000001_monthly_audit.png` = 001 号资产的月 K 审计图。\n\n"
        f"`review_index.csv` 提供 {len(numbered) * 3} 张图片的编号、市场、资产、周期和 FAE 主结论索引。\n\n"
        "## 人工反馈格式\n\n"
        "请使用“编号 + 周期”，例如：`017 日K：下降楔形应保留，倒三阳应删除；\n"
        "052 周K：吞没从属于下降覆盖线`。所有修正将作为 FAE 人工判断层的校准样本。\n\n"
        "## 模式说明\n\n"
        "这是审计版，不是最终用户首屏版。月 K 不强行推广正式技术图形；\n"
        "周 K 和日 K 的正式技术图形不设硬数量上限，而是按候选排名使用逐级提高的滚动置信度门槛。\n"
        "图片为了人工复核保留代表性候选；完整候选、冲突降级、重复合并信息都在同名 JSON 中，\n"
        "移动端默认只读取 JSON 中的 display.default，其他层由用户手动开关展开。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "review_index": str(index_path),
                "asset_count_completed": len(results),
                "image_count_generated": image_count,
                "failed_asset_count": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 100-asset FAE audit image set.")
    parser.add_argument("--limit", type=int, default=100, help="Only process the first N fixed assets")
    parser.add_argument("--years", type=int, default=12, help="Years of daily history to request")
    parser.add_argument("--fetch-workers", type=int, default=3, help="Concurrent data fetch workers")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit 必须大于 0")
    if args.years < 2:
        raise SystemExit("--years 至少为 2，月 K/周 K 才有足够历史")
    selected = ASSET_UNIVERSE[: min(int(args.limit), len(ASSET_UNIVERSE))]
    generate_batch(
        assets=selected,
        years=int(args.years),
        fetch_workers=max(1, int(args.fetch_workers)),
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
