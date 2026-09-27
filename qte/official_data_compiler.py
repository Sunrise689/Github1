"""Download and compile official factor/macro data for QTE.

The compiler runs offline from the application runtime: it snapshots public
official files, records hashes and versions, normalizes low-frequency series,
and emits descriptive historical statistics. It never makes a trading decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from .schemas import QuantEvidenceRecord

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
NORMALIZED_ROOT = DATA_ROOT / "normalized"
EVIDENCE_PATH = ROOT / "evidence" / "compiled_official_data_evidence.jsonl"
DATASET_REGISTRY = DATA_ROOT / "official_dataset_registry.json"
SOURCE_INDEX = DATA_ROOT / "official_data_source_index.json"
COMPILER_VERSION = "1.0"
USER_AGENT = "QTE-Offline-Compiler/1.0 (descriptive research; no trading decisions)"


FRENCH_DATASETS: list[dict[str, Any]] = [
    {
        "dataset_id": "ken_french_ff3_monthly",
        "title": "Fama/French 3 Research Factors (Monthly)",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
        "archive": "F-F_Research_Data_Factors_CSV.zip",
        "inner": "F-F_Research_Data_Factors.csv",
        "market": "US",
        "frequency": "monthly",
        "factor_map": {"Mkt-RF": "market_risk_premium", "SMB": "size", "HML": "value_pb", "RF": "risk_free_rate"},
        "description": "Kenneth French 官方美国市场、规模、价值和无风险利率月度序列。",
        "plain_description": "用作者本人维护的数据，看大盘、公司大小和相对便宜程度在历史月份里的表现。",
    },
    {
        "dataset_id": "ken_french_ff5_monthly",
        "title": "Fama/French 5 Research Factors (2x3, Monthly)",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "archive": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "inner": "F-F_Research_Data_5_Factors_2x3.csv",
        "market": "US",
        "frequency": "monthly",
        "factor_map": {"Mkt-RF": "market_risk_premium", "SMB": "size", "HML": "value_pb", "RMW": "profitability_rmw", "CMA": "investment_cma", "RF": "risk_free_rate"},
        "description": "Kenneth French 官方五因子月度序列，在市场、规模和价值基础上加入盈利能力与投资。",
        "plain_description": "除了看大盘、大小和便宜程度，还把公司赚钱能力和扩张方式放进同一张历史成绩单。",
    },
    {
        "dataset_id": "ken_french_momentum_monthly",
        "title": "Fama/French Momentum Factor (Monthly)",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
        "archive": "F-F_Momentum_Factor_CSV.zip",
        "inner": "F-F_Momentum_Factor.csv",
        "market": "US",
        "frequency": "monthly",
        "factor_map": {"Mom": "momentum"},
        "description": "Kenneth French 官方动量因子月度序列，基于规模与过去收益的独立排序组合。",
        "plain_description": "用官方组合数据检查过去相对强势与弱势股票组的历史收益差。",
    },
]


FRED_SERIES: list[dict[str, Any]] = [
    {"series_id":"FEDFUNDS","title":"Effective Federal Funds Rate","title_zh":"有效联邦基金利率","unit":"percent","native_frequency":"monthly","aggregation":"last_available","category":"rates","plain_description":"美国短期政策利率环境的核心背景。"},
    {"series_id":"DGS2","title":"Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity","title_zh":"美国2年期国债收益率","unit":"percent","native_frequency":"daily","aggregation":"monthly_average","category":"rates","plain_description":"市场对较短期限利率的历史定价背景。"},
    {"series_id":"DGS10","title":"Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity","title_zh":"美国10年期国债收益率","unit":"percent","native_frequency":"daily","aggregation":"monthly_average","category":"rates","plain_description":"长期利率和贴现环境的常用背景指标。"},
    {"series_id":"T10Y2Y","title":"10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity","title_zh":"美国10年减2年期限利差","unit":"percentage_points","native_frequency":"daily","aggregation":"monthly_average","category":"yield_curve","plain_description":"比较长期与短期利率，描述收益率曲线是更陡还是更平。"},
    {"series_id":"CPIAUCSL","title":"Consumer Price Index for All Urban Consumers: All Items in U.S. City Average","title_zh":"美国消费者价格指数","unit":"index_1982_1984_100","native_frequency":"monthly","aggregation":"last_available","category":"inflation","plain_description":"观察美国总体价格水平和通胀背景。"},
    {"series_id":"UNRATE","title":"Unemployment Rate","title_zh":"美国失业率","unit":"percent","native_frequency":"monthly","aggregation":"last_available","category":"labor","plain_description":"观察美国劳动力市场冷热的宏观背景。"},
    {"series_id":"GDPC1","title":"Real Gross Domestic Product","title_zh":"美国实际国内生产总值","unit":"billions_chained_2017_dollars","native_frequency":"quarterly","aggregation":"quarterly","category":"growth","plain_description":"观察美国经济实际产出的季度变化。"},
    {"series_id":"INDPRO","title":"Industrial Production: Total Index","title_zh":"美国工业生产指数","unit":"index_2017_100","native_frequency":"monthly","aggregation":"last_available","category":"growth","plain_description":"观察美国工业活动的月度变化。"},
    {"series_id":"BAMLH0A0HYM2","title":"ICE BofA US High Yield Index Option-Adjusted Spread","title_zh":"美国高收益债期权调整利差","unit":"percent","native_frequency":"daily","aggregation":"monthly_average","category":"credit","plain_description":"描述高收益债相对国债需要多少额外补偿，是信用压力背景。"},
    {"series_id":"VIXCLS","title":"CBOE Volatility Index: VIX","title_zh":"VIX市场波动指数","unit":"index","native_frequency":"daily","aggregation":"monthly_average","category":"market_stress","plain_description":"描述期权市场隐含的短期波动与市场压力背景。"},
]


# These broad official sources are useful to QTE, but downloading an arbitrary
# slice would create a misleading "complete macro database".  Register them now
# with an explicit deferred status; add concrete series only when a product
# question and a stable API field have been selected.  Chinese data remains
# backend-eligible and is controlled only at the presentation layer.
REGISTERED_EXTENDED_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "world_bank_open_data",
        "provider": "World Bank Open Data",
        "source_url": "https://data.worldbank.org/",
        "market": "global",
        "kind": "official_macro_catalog",
        "status": "registered_not_compiled",
        "priority": "P2",
        "reason": "Broad catalog; select named indicators before compiling to avoid unbounded and irrelevant data.",
        "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
    },
    {
        "source_id": "imf_data",
        "provider": "International Monetary Fund Data",
        "source_url": "https://data.imf.org/",
        "market": "global",
        "kind": "official_macro_catalog",
        "status": "registered_not_compiled",
        "priority": "P2",
        "reason": "Broad catalog; compile only named macro series needed by a QTE explanation.",
        "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
    },
    {
        "source_id": "china_nbs_national_data",
        "provider": "National Bureau of Statistics of China - National Data",
        "source_url": "https://data.stats.gov.cn/",
        "market": "CN",
        "kind": "official_macro_catalog",
        "status": "registered_not_compiled",
        "priority": "P2",
        "reason": "Backend source is allowed. Compile named series when directly relevant; frontend visibility is a separate product policy.",
        "display_policy": {
            "backend_eligible": True,
            "frontend_default": "hidden",
            "frontend_condition": "explicit_cn_scope_or_internal_review",
        },
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, output: Path, *, retries: int = 3) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(15, 90))
            response.raise_for_status()
            if not response.content:
                raise ValueError(f"empty response from {url}")
            output.write_bytes(response.content)
            return
        except Exception as exc:  # network/compiler boundary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _parse_french_csv(raw: str, dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines = raw.replace("\r\n", "\n").splitlines()
    version_match = re.search(r"created using the\s+(\d{6})\s+CRSP database", raw, re.I)
    version = version_match.group(1) if version_match else "unknown"
    header_index = next((i for i, line in enumerate(lines) if line.startswith(",")), None)
    if header_index is None:
        raise ValueError(f"monthly header not found for {dataset['dataset_id']}")
    header = [cell.strip() or "date" for cell in next(csv.reader([lines[header_index]]))]
    records: list[dict[str, Any]] = []
    for line in lines[header_index + 1 :]:
        if not re.match(r"^\s*\d{6}\s*,", line):
            if records:
                break
            continue
        cells = next(csv.reader([line]))
        if len(cells) != len(header):
            continue
        stamp = cells[0].strip()
        item: dict[str, Any] = {"date": f"{stamp[:4]}-{stamp[4:6]}-01"}
        valid = True
        for name, value in zip(header[1:], cells[1:]):
            try:
                number = float(value.strip())
            except ValueError:
                valid = False
                break
            item[name] = None if number in {-99.99, -999.0} else number / 100.0
        if valid:
            records.append(item)
    if not records:
        raise ValueError(f"no monthly observations parsed for {dataset['dataset_id']}")
    return records, {"data_version": f"CRSP_{version}", "header": header[1:], "start": records[0]["date"], "end": records[-1]["date"]}


def _aggregate_monthly(rows: list[tuple[str, float]], method: str) -> list[dict[str, Any]]:
    if method == "quarterly":
        return [{"date": stamp, "value": value} for stamp, value in rows]
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for stamp, value in rows:
        buckets[stamp[:7]].append((stamp, value))
    result: list[dict[str, Any]] = []
    for month in sorted(buckets):
        values = buckets[month]
        if method == "monthly_average":
            value = float(np.mean([item[1] for item in values]))
        else:
            value = values[-1][1]
        result.append({"date": month + "-01", "value": value})
    return result


def _parse_fred_csv(path: Path, series: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        value_key = series["series_id"]
        for row in reader:
            raw = str(row.get(value_key, "")).strip()
            if raw in {"", ".", "NA", "NaN"}:
                continue
            try:
                rows.append((str(row["observation_date"]), float(raw)))
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no FRED observations parsed for {series['series_id']}")
    normalized = _aggregate_monthly(rows, series["aggregation"])
    return normalized, {"start": normalized[0]["date"], "end": normalized[-1]["date"], "raw_observations": len(rows), "normalized_observations": len(normalized)}


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def _factor_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    arithmetic_mean = float(np.mean(array) * 12.0)
    volatility = float(np.std(array, ddof=1) * math.sqrt(12.0)) if len(array) > 1 else 0.0
    return {
        "annualized_arithmetic_mean": arithmetic_mean,
        "annualized_volatility": volatility,
        "mean_to_volatility": arithmetic_mean / volatility if volatility else 0.0,
        "maximum_drawdown": _max_drawdown(array),
        "positive_month_ratio": float(np.mean(array > 0)),
        "worst_month": float(np.min(array)),
        "best_month": float(np.max(array)),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def compile_french(*, refresh: bool) -> tuple[list[dict[str, Any]], list[QuantEvidenceRecord], list[dict[str, Any]]]:
    registry: list[dict[str, Any]] = []
    evidence: list[QuantEvidenceRecord] = []
    source_rows: list[dict[str, Any]] = []
    canonical_claim_dataset = {"market_risk_premium": "ken_french_ff3_monthly", "size": "ken_french_ff3_monthly", "value_pb": "ken_french_ff3_monthly", "profitability_rmw": "ken_french_ff5_monthly", "investment_cma": "ken_french_ff5_monthly", "momentum": "ken_french_momentum_monthly"}
    for dataset in FRENCH_DATASETS:
        archive = RAW_ROOT / "ken_french" / dataset["archive"]
        if refresh or not archive.exists():
            _download(dataset["url"], archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"not a valid ZIP: {archive}")
        with zipfile.ZipFile(archive) as zipped:
            raw = zipped.read(dataset["inner"]).decode("utf-8-sig", errors="replace")
        records, meta = _parse_french_csv(raw, dataset)
        normalized = NORMALIZED_ROOT / "ken_french" / f"{dataset['dataset_id']}.jsonl"
        _write_jsonl(normalized, ({"dataset_id": dataset["dataset_id"], **row} for row in records))
        archive_hash = _sha256(archive)
        registry_item = {
            "dataset_id": dataset["dataset_id"],
            "provider": "Kenneth R. French Data Library",
            "title": dataset["title"],
            "description": dataset["description"],
            "plain_description": dataset["plain_description"],
            "source_url": dataset["url"],
            "market": dataset["market"],
            "frequency": dataset["frequency"],
            "unit": "decimal_return",
            "data_version": meta["data_version"],
            "download_date": date.today().isoformat(),
            "source_sha256": archive_hash,
            "sample_start": meta["start"],
            "sample_end": meta["end"],
            "observation_count": len(records),
            "fields": meta["header"],
            "normalized_file": str(normalized.relative_to(ROOT)).replace("\\", "/"),
            "license_note": "Public research data; retain provider attribution and source version.",
        }
        registry.append(registry_item)
        source_rows.append({"source_id": dataset["dataset_id"], "kind": "official_factor_dataset", "status": "compiled", **registry_item})
        for field, factor_id in dataset["factor_map"].items():
            if canonical_claim_dataset.get(factor_id) != dataset["dataset_id"]:
                continue
            values = [float(row[field]) for row in records if row.get(field) is not None]
            stats = _factor_statistics(values)
            source_title = dataset["title"]
            evidence.append(
                QuantEvidenceRecord(
                    evidence_id=f"qe_{dataset['dataset_id']}_{factor_id}",
                    evidence_type="official_data_recomputed",
                    factor_id=factor_id,
                    claim=(f"根据 {source_title} 官方月度序列，本地按算术月均值×12计算的样本期历史年化均值为 {_pct(stats['annualized_arithmetic_mean'])}，年化波动率为 {_pct(stats['annualized_volatility'])}，复利路径最大回撤为 {_pct(stats['maximum_drawdown'])}。"),
                    claim_plain=(f"这份官方历史成绩单里，该因子的年化平均差异约为 {_pct(stats['annualized_arithmetic_mean'])}；但过程中曾出现约 {_pct(stats['maximum_drawdown'])} 的最大回撤，所以平均值不能代表持有过程，也不预示未来。"),
                    caveat="这是基于当前下载版本的历史复算，不预示未来；未计实际产品费用、融资、借券、税费和投资者可得交易成本。Kenneth French 历史数据可能随 CRSP 修订而变化。",
                    source_id=dataset["dataset_id"],
                    source_title=source_title,
                    source_kind="official_factor_dataset",
                    source_pages="not_applicable",
                    source_url=dataset["url"],
                    source_sha256=archive_hash,
                    market=dataset["market"],
                    period=f"{meta['start'][:7]} to {meta['end'][:7]}",
                    frequency="monthly",
                    calculation="arithmetic annualized mean = mean(monthly return)*12; annualized volatility = sample_std(monthly return)*sqrt(12); maximum drawdown = min(compounded wealth/running peak-1)",
                    data_version=f"{meta['data_version']};downloaded_{date.today().isoformat()}",
                    quality=0.98,
                    display_policy={"backend_eligible": True, "frontend_default": "visible"},
                    policy_flags={"no_raw_text": True, "historical_qualifier_required": True, "no_trade_advice": True, "official_data": True, "locally_recomputed": True, "engine_eligible": True},
                    metadata={"field": field, "observation_count": len(values), **{key: round(value, 8) for key, value in stats.items()}},
                )
            )
    return registry, evidence, source_rows


def compile_fred(*, refresh: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for series in FRED_SERIES:
        raw_path = RAW_ROOT / "fred" / f"{series['series_id']}.csv"
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series['series_id']}"
        if refresh or not raw_path.exists():
            _download(url, raw_path)
        records, meta = _parse_fred_csv(raw_path, series)
        normalized = NORMALIZED_ROOT / "fred" / f"{series['series_id']}.jsonl"
        _write_jsonl(normalized, ({"dataset_id": f"fred_{series['series_id'].lower()}", "series_id": series["series_id"], **row} for row in records))
        digest = _sha256(raw_path)
        item = {
            "dataset_id": f"fred_{series['series_id'].lower()}",
            "provider": "Federal Reserve Bank of St. Louis (FRED)",
            "series_id": series["series_id"],
            "title": series["title"],
            "title_zh": series["title_zh"],
            "description": f"FRED official series {series['series_id']} used as descriptive macro context.",
            "plain_description": series["plain_description"],
            "source_url": f"https://fred.stlouisfed.org/series/{series['series_id']}",
            "download_url": url,
            "market": "US",
            "frequency": "quarterly" if series["aggregation"] == "quarterly" else "monthly",
            "native_frequency": series["native_frequency"],
            "aggregation": series["aggregation"],
            "unit": series["unit"],
            "category": series["category"],
            "data_version": f"downloaded_{date.today().isoformat()}",
            "download_date": date.today().isoformat(),
            "source_sha256": digest,
            "sample_start": meta["start"],
            "sample_end": meta["end"],
            "raw_observation_count": meta["raw_observations"],
            "observation_count": meta["normalized_observations"],
            "normalized_file": str(normalized.relative_to(ROOT)).replace("\\", "/"),
            "usage_policy": "context_only_no_causal_claim",
            "coverage_note": (
                "The public FRED CSV response returned observations from "
                f"{meta['start']} through {meta['end']} on the download date; "
                "QTE does not infer unavailable earlier observations."
            ),
        }
        registry.append(item)
        source_rows.append({"source_id": item["dataset_id"], "kind": "official_macro_dataset", "status": "compiled", **item})
    return registry, source_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile official QTE factor and macro datasets.")
    parser.add_argument("--refresh", action="store_true", help="Download current official files even when a local snapshot exists.")
    args = parser.parse_args()
    french_registry, evidence, french_sources = compile_french(refresh=args.refresh)
    fred_registry, fred_sources = compile_fred(refresh=args.refresh)
    registry = {
        "schema_version": "1.0",
        "generated": datetime.now().astimezone().isoformat(),
        "compiler_version": COMPILER_VERSION,
        "datasets": french_registry + fred_registry,
        "registered_not_compiled_sources": REGISTERED_EXTENDED_SOURCES,
        "policy": {
            "factor_data_is_historical_not_predictive": True,
            "macro_data_is_context_only": True,
            "macro_correlation_does_not_imply_causation": True,
            "raw_files_are_versioned_official_snapshots": True,
        },
    }
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    source_index = {
        "schema_version": "1.0",
        "generated": datetime.now().astimezone().isoformat(),
        "sources": french_sources + fred_sources + REGISTERED_EXTENDED_SOURCES,
    }
    SOURCE_INDEX.write_text(json.dumps(source_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _write_jsonl(EVIDENCE_PATH, (item.to_dict() for item in evidence))
    summary = {
        "dataset_count": len(registry["datasets"]),
        "factor_dataset_count": len(french_registry),
        "macro_dataset_count": len(fred_registry),
        "official_data_evidence_count": len(evidence),
        "registered_not_compiled_source_count": len(REGISTERED_EXTENDED_SOURCES),
        "registry": str(DATASET_REGISTRY),
        "source_index": str(SOURCE_INDEX),
        "evidence": str(EVIDENCE_PATH),
    }
    (DATA_ROOT / "official_data_compilation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
