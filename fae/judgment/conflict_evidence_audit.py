"""Audit the evidence fields consumed by declarative conflict rules.

Run from the project root with::

    python -m fae.judgment.conflict_evidence_audit

The audit is intentionally small and deterministic.  It checks both the
static contract (every condition field is declared by the extractor) and the
runtime contract (the extractor emits the fields for a valid OHLCV frame).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evidence_extractor import CONFLICT_EVIDENCE_FIELDS, EvidenceExtractor


ROOT = Path(__file__).resolve().parents[2]
JUDGMENT = ROOT / "fae" / "judgment"
CONFLICT_RULES_PATH = JUDGMENT / "conflict_rules.json"
REPORT_PATH = JUDGMENT / "conflict_evidence_audit.md"


def _load_rules() -> list[dict[str, Any]]:
    payload = json.loads(CONFLICT_RULES_PATH.read_text(encoding="utf-8"))
    rules = payload.get("rules", []) if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        raise TypeError("conflict_rules.json must contain a rules list")
    return [rule for rule in rules if isinstance(rule, dict)]


def _sample_data(rows: int = 12) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.2 * x + np.sin(x / 2.0)
    open_ = close - 0.1
    return pd.DataFrame(
        {
            "open": open_,
            "high": close + 0.5,
            "low": open_ - 0.5,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )


def audit() -> dict[str, Any]:
    rules = _load_rules()
    required_fields = sorted(
        {
            str(field)
            for rule in rules
            for field in (
                rule.get("condition", {}).get("evidence", {})
                if isinstance(rule.get("condition", {}).get("evidence", {}), dict)
                else {}
            )
        }
    )
    missing_static = sorted(set(required_fields) - set(CONFLICT_EVIDENCE_FIELDS))
    evidence = EvidenceExtractor().extract(
        _sample_data(),
        signal_windows=[{"pattern": "tower_bottom", "start": 2, "end": 4}],
    )
    local = evidence.get("signal_evidence", {}).get("tower_bottom:2:4", {})
    missing_runtime = sorted(
        field
        for field in required_fields
        if field not in evidence and field not in local
    )
    return {
        "rule_count": len(rules),
        "required_fields": required_fields,
        "available_fields": sorted(CONFLICT_EVIDENCE_FIELDS),
        "missing_static": missing_static,
        "missing_runtime": missing_runtime,
        "runtime_top_level": sorted(
            field for field in required_fields if field in evidence
        ),
        "runtime_local": sorted(field for field in required_fields if field in local),
        "rule_ids": [str(rule.get("rule_id", "")) for rule in rules],
    }


def render_report(result: dict[str, Any]) -> str:
    status = "PASS" if not result["missing_static"] and not result["missing_runtime"] else "FAIL"
    lines = [
        "# FAE Conflict Evidence Audit",
        "",
        f"- Status: **{status}**",
        f"- Conflict rules checked: **{result['rule_count']}**",
        "- Scope: declarative evidence conditions in `conflict_rules.json`",
        "",
        "## Evidence fields used by rules",
        "",
        *[f"- `{field}`" for field in result["required_fields"]],
        "",
        "## Availability",
        "",
        f"- Static extractor contract: `{', '.join(result['available_fields'])}`",
        f"- Runtime top-level fields: `{', '.join(result['runtime_top_level']) or 'none'}`",
        f"- Runtime signal-local fields: `{', '.join(result['runtime_local']) or 'none'}`",
        f"- Missing statically: `{', '.join(result['missing_static']) or 'none'}`",
        f"- Missing at runtime: `{', '.join(result['missing_runtime']) or 'none'}`",
        "",
        "## Semantics",
        "",
        "`bullish_bar_count`, `bearish_bar_count`, and consecutive counts are "
        "calculated on the candidate span when signal-local evidence is available. "
        "`continuation_up/down` requires at least two post-candidate closes moving "
        "strictly in that direction; a candidate at the end of the data is not "
        "invalidated merely because the global trend points the other way.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    result = audit()
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "status": "PASS"
                if not result["missing_static"] and not result["missing_runtime"]
                else "FAIL",
                "rule_count": result["rule_count"],
                "required_fields": result["required_fields"],
                "missing_static": result["missing_static"],
                "missing_runtime": result["missing_runtime"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not result["missing_static"] and not result["missing_runtime"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
