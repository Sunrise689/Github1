"""Link canonical QTE factors to already compiled knowledge and evidence.

This is a build-time provenance utility, not a numerical factor engine.  It
never reads or copies raw PDF text.  Links are made only from registered
source IDs, registered PDF-page references, existing factor IDs, and existing
evidence IDs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "factor_registry.json"
KNOWLEDGE = ROOT / "knowledge" / "compiled_quant_knowledge.jsonl"
PAPER_EVIDENCE = ROOT / "evidence" / "compiled_quant_evidence.jsonl"
OFFICIAL_EVIDENCE = ROOT / "evidence" / "compiled_official_data_evidence.jsonl"
OUTPUT = ROOT / "knowledge" / "factor_knowledge_links.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pages(value: Any) -> set[int]:
    """Expand explicit page expressions such as ``1-3, 8``.

    Non-page sentinels (``not_available`` and ``not_applicable``) intentionally
    return an empty set and can never create a false source-page match.
    """
    output: set[int] = set()
    text = str(value or "")
    for start_text, end_text in re.findall(r"(?<![A-Za-z0-9_])(\d+)(?:\s*-\s*(\d+))?", text):
        start = int(start_text)
        end = int(end_text or start_text)
        if start <= 0 or end < start or end - start > 2000:
            continue
        output.update(range(start, end + 1))
    return output


def _direct_knowledge_ids(
    factor_id: str,
    source_refs: Iterable[dict[str, Any]],
    knowledge: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    matches: list[dict[str, Any]] = []

    for record in knowledge:
        if factor_id in record.get("factor_ids", []):
            ids.add(str(record["knowledge_id"]))
            matches.append(
                {
                    "knowledge_id": record["knowledge_id"],
                    "source_id": record["source_id"],
                    "pdf_pages": record.get("pdf_pages", []),
                    "match_basis": "existing_factor_id",
                }
            )

    for source in source_refs:
        source_id = str(source.get("source_id", ""))
        registered_pages = _pages(source.get("pages"))
        if not source_id or not registered_pages:
            continue
        for record in knowledge:
            if record.get("source_id") != source_id:
                continue
            overlap = sorted(registered_pages.intersection(record.get("pdf_pages", [])))
            if not overlap:
                continue
            knowledge_id = str(record["knowledge_id"])
            ids.add(knowledge_id)
            matches.append(
                {
                    "knowledge_id": knowledge_id,
                    "source_id": source_id,
                    "pdf_pages": record.get("pdf_pages", []),
                    "matched_pages": overlap,
                    "match_basis": "registered_source_page_overlap",
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for match in matches:
        key = str(match["knowledge_id"])
        previous = deduped.get(key)
        if previous is None or match["match_basis"] == "registered_source_page_overlap":
            deduped[key] = match
    return sorted(ids), [deduped[key] for key in sorted(deduped)]


def build(output: Path = OUTPUT) -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    knowledge = _jsonl(KNOWLEDGE)
    evidence = [*_jsonl(PAPER_EVIDENCE), *_jsonl(OFFICIAL_EVIDENCE)]
    links: list[dict[str, Any]] = []

    for factor in registry.get("factors", []):
        factor_id = str(factor["factor_id"])
        source_refs = list(factor.get("sources", []))
        knowledge_ids, knowledge_matches = _direct_knowledge_ids(factor_id, source_refs, knowledge)
        registered_source_ids = {str(item.get("source_id", "")) for item in source_refs if item.get("source_id")}
        factor_evidence: list[dict[str, Any]] = []
        evidence_matches: list[dict[str, Any]] = []
        for item in evidence:
            if item.get("factor_id") == factor_id:
                factor_evidence.append(item)
                evidence_matches.append(
                    {
                        "evidence_id": item["evidence_id"],
                        "source_id": item.get("source_id", ""),
                        "match_basis": "existing_factor_id",
                    }
                )
            elif item.get("source_id") in registered_source_ids:
                factor_evidence.append(item)
                evidence_matches.append(
                    {
                        "evidence_id": item["evidence_id"],
                        "source_id": item.get("source_id", ""),
                        "match_basis": "registered_source_id",
                    }
                )
        factor_evidence = list({str(item["evidence_id"]): item for item in factor_evidence}.values())
        evidence_matches = list({str(item["evidence_id"]): item for item in evidence_matches}.values())
        pending = factor.get("evidence_assessment", {}).get("status") == "evidence_pending"
        if pending:
            link_status = "evidence_pending"
        elif factor_evidence or knowledge_ids:
            link_status = "linked_existing_artifacts"
        elif source_refs:
            link_status = "registered_source_pages_only"
        else:
            link_status = "unlinked"
        links.append(
            {
                "factor_id": factor_id,
                "name_zh": factor.get("name_zh", ""),
                "name_en": factor.get("name_en", ""),
                "link_status": link_status,
                "evidence_assessment": factor.get("evidence_assessment", {}),
                "registered_source_refs": source_refs,
                "knowledge_ids": knowledge_ids,
                "knowledge_matches": knowledge_matches,
                "evidence_ids": sorted(str(item["evidence_id"]) for item in factor_evidence),
                "evidence_matches": sorted(evidence_matches, key=lambda item: str(item["evidence_id"])),
                "pending_evidence_ids": sorted(
                    str(item["evidence_id"])
                    for item in factor_evidence
                    if item.get("evidence_type") == "evidence_pending"
                ),
                "link_rule": "Only exact existing factor IDs or overlap with explicitly registered source pages; no semantic guess and no invented page.",
            }
        )

    payload = {
        "schema_version": "1.0",
        "generated": datetime.now().astimezone().isoformat(),
        "purpose": "把 38 个正式因子关联到既有结构化知识和证据；空关联与证据缺口必须显式保留。",
        "canonical_factor_count": len(links),
        "compiled_knowledge_record_count": len(knowledge),
        "evidence_record_count": len(evidence),
        "link_method": [
            "existing factor_id exact match",
            "registered source_id and PDF-page overlap",
            "existing evidence factor_id exact match"
        ],
        "policy": {
            "raw_source_text_copied": False,
            "semantic_guessing_allowed": False,
            "missing_links_are_visible": True,
            "evidence_pending_is_not_empirical_evidence": True
        },
        "factor_links": links,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "output": str(output),
        "factor_count": len(links),
        "with_knowledge": sum(bool(item["knowledge_ids"]) for item in links),
        "with_evidence": sum(bool(item["evidence_ids"]) for item in links),
        "pending": sum(item["link_status"] == "evidence_pending" for item in links),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build QTE factor-to-knowledge provenance links.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
