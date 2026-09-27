"""Typed QTE knowledge records.

QTE stores descriptive, source-traceable evidence. It never emits a trading
decision. Raw book text and OCR output are build inputs only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class QuantKnowledgeRecord:
    """One machine-usable concept compiled from a source page range."""

    knowledge_id: str
    source_id: str
    source_title: str
    source_kind: str
    source_file: str
    source_sha256: str
    pdf_pages: list[int]
    topic_ids: list[str]
    factor_ids: list[str]
    knowledge_type: str
    academic_summary: str
    plain_summary: str
    caveat: str
    market: str = "global"
    period: str = "not_applicable"
    method_tags: list[str] = field(default_factory=list)
    evidence_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    validation_tags: list[str] = field(default_factory=list)
    implementation_tags: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    compiler_version: str = "2.0"
    display_policy: dict[str, Any] = field(default_factory=dict)
    policy_flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantKnowledgeRecord":
        return cls(**dict(value))


@dataclass(frozen=True)
class QuantEvidenceRecord:
    """One auditable empirical or official-data statement."""

    evidence_id: str
    evidence_type: str
    factor_id: str
    claim: str
    claim_plain: str
    caveat: str
    source_id: str
    source_title: str
    source_kind: str
    source_pages: str
    source_url: str
    source_sha256: str
    market: str
    period: str
    frequency: str
    calculation: str
    data_version: str
    quality: float
    display_policy: dict[str, Any] = field(default_factory=dict)
    policy_flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantEvidenceRecord":
        return cls(**dict(value))
