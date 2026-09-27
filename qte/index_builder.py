"""Build QTE's offline sparse index from every engine-eligible artifact.

The index is a package artifact, not a database service.  Raw PDF/OCR text is
never indexed; only structured summaries, definitions, caveats and provenance
are searchable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "knowledge" / "compiled_quant_knowledge.jsonl"
PAPER_EVIDENCE = ROOT / "evidence" / "compiled_quant_evidence.jsonl"
OFFICIAL_EVIDENCE = ROOT / "evidence" / "compiled_official_data_evidence.jsonl"
FACTORS = ROOT / "factor_registry.json"
LEXICON = ROOT / "translation_lexicon.json"
DATASETS = ROOT / "data" / "official_dataset_registry.json"
FACTOR_LINKS = ROOT / "knowledge" / "factor_knowledge_links.json"
FRAMEWORKS = (
    ROOT / "attribution_framework.json",
    ROOT / "validation_pipeline.json",
    ROOT / "view_translation_schema.json",
    ROOT / "ai_integration_protocol.json",
)
INDEX = ROOT / "index"
SCHEMA_VERSION = "3.0"
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+./-]*|[0-9]+(?:\.[0-9]+)?|[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character bigrams without jieba."""
    output: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", raw):
            if len(raw) == 1:
                output.append(raw)
            else:
                output.extend(raw[index : index + 2] for index in range(len(raw) - 1))
        else:
            output.append(raw)
    return output


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _engine_eligible(record: dict[str, Any], quality_key: str) -> bool:
    flags = record.get("policy_flags") or {}
    return (
        float(record.get(quality_key, 0.0)) >= 0.65
        and bool(record.get("caveat"))
        and flags.get("engine_eligible", True) is not False
        and flags.get("review_only", False) is not True
    )


def _knowledge_records() -> Iterator[dict[str, Any]]:
    for item in _jsonl(KNOWLEDGE):
        if not _engine_eligible(item, "quality_score"):
            continue
        yield {**item, "doc_id": item["knowledge_id"], "record_type": "knowledge"}


def _evidence_records(path: Path, record_type: str) -> Iterator[dict[str, Any]]:
    for item in _jsonl(path):
        if not _engine_eligible(item, "quality"):
            continue
        resolved_type = "evidence_pending" if item.get("evidence_type") == "evidence_pending" else record_type
        yield {**item, "doc_id": item["evidence_id"], "record_type": resolved_type}


def _factor_records() -> Iterator[dict[str, Any]]:
    if not FACTORS.exists():
        return
    payload = json.loads(FACTORS.read_text(encoding="utf-8"))
    canonical = [(factor, False) for factor in payload.get("factors", [])]
    compatibility = [(factor, True) for factor in payload.get("legacy_compatibility_factors", [])]
    for factor, is_legacy in [*canonical, *compatibility]:
        caveat = "；".join(factor.get("failure_scenarios", []))
        availability = factor.get("data_availability") or {}
        calculation_implementation = factor.get("calculation_implementation", "not_implemented")
        yield {
            **factor,
            "doc_id": f"factor_definition_{factor['factor_id']}",
            "record_type": "legacy_factor_definition" if is_legacy else "factor_definition",
            "factor_ids": [factor["factor_id"]],
            "plain_summary": factor.get("plain_definition", ""),
            "academic_summary": factor.get("academic_definition", ""),
            "caveat": caveat,
            "quality_score": 1.0,
            "market": "global",
            "is_legacy_compatibility": is_legacy,
            "definition_only": calculation_implementation != "implemented" or not bool(availability.get("current")),
            "numeric_result_available": calculation_implementation == "implemented" and bool(availability.get("current")),
            "result_capability": "definition_and_historical_evidence_only" if calculation_implementation != "implemented" or not bool(availability.get("current")) else "calculation_result_allowed",
            "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
            "policy_flags": {
                "engine_eligible": True,
                "no_trade_advice": True,
                "no_raw_text": True,
                "llm_must_not_calculate": calculation_implementation != "implemented",
            },
        }


def _factor_link_records() -> Iterator[dict[str, Any]]:
    if not FACTOR_LINKS.exists():
        return
    payload = json.loads(FACTOR_LINKS.read_text(encoding="utf-8"))
    for item in payload.get("factor_links", []):
        factor_id = str(item["factor_id"])
        pending = item.get("link_status") == "evidence_pending"
        yield {
            **item,
            "doc_id": f"factor_evidence_link_{factor_id}",
            "record_type": "factor_evidence_link",
            "factor_ids": [factor_id],
            "plain_summary": "该因子已关联到现有结构化知识和证据。" if not pending else "该因子的专门实证证据尚未补齐，只能介绍定义和资料缺口。",
            "academic_summary": json.dumps(item.get("evidence_assessment", {}), ensure_ascii=False),
            "caveat": "关联表示来源与页码可追溯，不代表因果或未来有效；evidence_pending 不得改写为已验证。",
            "quality_score": 1.0,
            "market": "global",
            "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
            "policy_flags": {"engine_eligible": True, "no_trade_advice": True, "no_raw_text": True},
        }


def _framework_records() -> Iterator[dict[str, Any]]:
    for path in FRAMEWORKS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        framework_id = path.stem
        yield {
            "doc_id": f"framework_{framework_id}",
            "record_type": "framework_specification",
            "framework_id": framework_id,
            "title": framework_id,
            "plain_summary": str(payload.get("purpose", "")),
            "academic_summary": json.dumps(payload, ensure_ascii=False),
            "caveat": "这是知识与集成规范，不是已执行的数值计算，也不构成投资建议。",
            "quality_score": 1.0,
            "market": "global",
            "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
            "policy_flags": {"engine_eligible": True, "no_trade_advice": True, "no_raw_text": True},
        }


def _dataset_records() -> Iterator[dict[str, Any]]:
    if not DATASETS.exists():
        return
    payload = json.loads(DATASETS.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []):
        yield {
            **dataset,
            "doc_id": f"dataset_{dataset['dataset_id']}",
            "record_type": "official_dataset",
            "source_id": dataset["dataset_id"],
            "source_title": dataset.get("title", dataset["dataset_id"]),
            "plain_summary": dataset.get("plain_description", ""),
            "academic_summary": dataset.get("description", ""),
            "caveat": dataset.get("coverage_note", "历史数据仅用于描述，不预示未来；宏观数据不得直接解释为因果关系。"),
            "quality_score": 0.95,
            "period": f"{dataset.get('sample_start', '')} to {dataset.get('sample_end', '')}",
            "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
            "policy_flags": {"engine_eligible": True, "no_trade_advice": True, "official_data": True},
        }


def _translation_records() -> Iterator[dict[str, Any]]:
    if not LEXICON.exists():
        return
    payload = json.loads(LEXICON.read_text(encoding="utf-8"))
    for index, item in enumerate(payload.get("terms", []), start=1):
        yield {
            **item,
            "doc_id": f"translation_term_{index:03d}",
            "record_type": "translation_term",
            "plain_definition": item.get("plain", ""),
            "caveat": item.get("usage_note", ""),
            "quality_score": 1.0,
            "market": "global",
            "display_policy": {"backend_eligible": True, "frontend_default": "visible"},
            "policy_flags": {"engine_eligible": True, "no_trade_advice": True, "no_raw_text": True},
        }


def load_records() -> list[dict[str, Any]]:
    """Load all structured, engine-eligible QTE records."""
    return [
        *_knowledge_records(),
        *_evidence_records(PAPER_EVIDENCE, "paper_evidence"),
        *_evidence_records(OFFICIAL_EVIDENCE, "official_data_evidence"),
        *_factor_records(),
        *_dataset_records(),
        *_translation_records(),
        *_factor_link_records(),
        *_framework_records(),
    ]


def _factor_lookup() -> dict[str, dict[str, Any]]:
    if not FACTORS.exists():
        return {}
    payload = json.loads(FACTORS.read_text(encoding="utf-8"))
    items = [*payload.get("factors", []), *payload.get("legacy_compatibility_factors", [])]
    return {item["factor_id"]: item for item in items}


def _text(record: dict[str, Any], factor_map: dict[str, dict[str, Any]]) -> str:
    values: list[Any] = []
    for key in (
        "doc_id", "record_type", "term", "factor_id", "factor_ids", "topic_ids", "market", "period",
        "source_id", "source_title", "title", "title_zh", "series_id", "category", "knowledge_type",
        "claim", "claim_plain", "academic_summary", "plain_summary", "academic_definition",
        "plain_definition", "caveat", "calculation", "computation", "method_tags", "evidence_tags",
        "risk_tags", "validation_tags", "implementation_tags", "aliases", "name_zh", "name_en",
        "translation_key", "computation", "required_data", "failure_scenarios", "evidence_assessment",
        "data_availability", "calculation_implementation", "framework_id", "link_status", "registered_source_refs",
        "knowledge_ids", "evidence_ids", "pending_evidence_ids", "result_capability",
    ):
        values.append(record.get(key, ""))
    factor_ids = record.get("factor_ids") or ([record["factor_id"]] if record.get("factor_id") else [])
    for factor_id in factor_ids:
        factor = factor_map.get(str(factor_id), {})
        values.extend(
            factor.get(key, "")
            for key in ("name_zh", "name_en", "aliases", "category", "academic_definition", "plain_definition")
        )
    return " ".join(
        json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        for value in values
        if value not in (None, "", [], {})
    )


def build(records: Iterable[dict[str, Any]] | None = None, output_dir: Path = INDEX) -> dict[str, Any]:
    records = list(load_records() if records is None else records)
    factor_map = _factor_lookup()
    docs: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for record in records:
        factor_ids = record.get("factor_ids") or ([record["factor_id"]] if record.get("factor_id") else [])
        if factor_ids and not record.get("factor_categories"):
            record = {
                **record,
                "factor_categories": sorted(
                    {factor_map[str(factor_id)]["category"] for factor_id in factor_ids if str(factor_id) in factor_map}
                ),
            }
        text = _text(record, factor_map)
        tokens = tokenize(text)
        docs.append({"doc_id": record["doc_id"], "text": text, "tokens": tokens, "record": record})
        type_counts[str(record["record_type"])] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone().isoformat()
    docs_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated": generated,
        "doc_count": len(docs),
        "by_record_type": dict(sorted(type_counts.items())),
        "docs": docs,
    }
    (output_dir / "docs.json").write_text(
        json.dumps(docs_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []
    for doc in docs:
        lengths.append(len(doc["tokens"]))
        document_frequency.update(set(doc["tokens"]))
    np.savez_compressed(
        output_dir / "bm25_index.npz",
        schema_version=np.array([SCHEMA_VERSION]),
        generated=np.array([generated]),
        doc_count=np.array([len(docs)], dtype=np.int64),
        document_lengths=np.asarray(lengths, dtype=np.int32),
        document_frequency=np.array([json.dumps(document_frequency, ensure_ascii=False)]),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated": generated,
        "doc_count": len(docs),
        "by_record_type": dict(sorted(type_counts.items())),
        "docs_file": str(output_dir / "docs.json"),
        "index_file": str(output_dir / "bm25_index.npz"),
    }
    (output_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the complete QTE BM25 index.")
    parser.add_argument("--output", type=Path, default=INDEX)
    args = parser.parse_args()
    print(json.dumps(build(output_dir=args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
