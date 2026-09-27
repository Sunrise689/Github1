"""Validate the FAE Pattern Registry and its cross-layer references."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .pattern_registry import REGISTRY_PATH


ROOT = Path(__file__).resolve().parents[2]
CONFLICT_PATH = ROOT / "fae" / "judgment" / "conflict_rules.json"
VALID_CATEGORIES = {
    "simple_candlestick",
    "combination_candlestick",
    "trend_candlestick",
    "chart_pattern",
    "structure",
    "gap",
}
VALID_STATUS = {"implemented", "partial", "missing", "needs_review"}
VALID_PRIORITY = {"high", "medium", "low"}
REQUIRED_FIELDS = {
    "pattern_id",
    "name_zh",
    "name_en",
    "aliases",
    "category",
    "detector",
    "timeframes",
    "definition",
    "preconditions",
    "confirmation",
    "forbidden_contexts",
    "conflicts_with",
    "can_overlap_with",
    "display_priority",
    "visualization",
    "status",
}


def _source_has_symbol(module_name: str, symbol: str) -> bool:
    relative = Path(*module_name.split("."))
    path = ROOT / f"{relative}.py"
    if not path.exists():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_name, _, method_name = symbol.partition(".")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return True
            if isinstance(node, ast.ClassDef):
                if method_name and node.name == class_name and any(
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                    for child in node.body
                ):
                    return True
    return False


def _validate() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"registry JSON cannot be loaded: {exc}"]

    patterns = registry.get("patterns")
    if not isinstance(patterns, list):
        return registry, ["patterns must be a list"]

    ids = [item.get("pattern_id") for item in patterns if isinstance(item, dict)]
    duplicates = [item for item, count in Counter(ids).items() if item and count > 1]
    errors.extend(f"duplicate pattern_id: {item}" for item in duplicates)
    canonical = {item for item in ids if isinstance(item, str)}

    alias_owner: dict[str, str] = {}
    for item in patterns:
        if not isinstance(item, dict):
            errors.append("pattern entry is not an object")
            continue
        pattern_id = item.get("pattern_id", "<missing>")
        missing = REQUIRED_FIELDS - set(item)
        errors.extend(f"{pattern_id}: missing field {field}" for field in sorted(missing))
        category = item.get("category")
        if category not in VALID_CATEGORIES:
            errors.append(f"{pattern_id}: invalid category {category!r}")
        status = item.get("status")
        if status not in VALID_STATUS:
            errors.append(f"{pattern_id}: invalid status {status!r}")
        if item.get("display_priority") not in VALID_PRIORITY:
            errors.append(f"{pattern_id}: invalid display_priority")
        detector = item.get("detector")
        if detector == "missing":
            if status != "missing":
                errors.append(f"{pattern_id}: detector=missing requires status=missing")
        elif isinstance(detector, str) and ":" in detector:
            module_name, symbol = detector.split(":", 1)
            if not _source_has_symbol(module_name, symbol):
                errors.append(f"{pattern_id}: detector reference not found: {detector}")
        else:
            errors.append(f"{pattern_id}: detector must be module:symbol or missing")
        visualization = item.get("visualization")
        if not isinstance(visualization, dict) or not visualization.get("type"):
            errors.append(f"{pattern_id}: visualization must contain type")
        for relation in ("conflicts_with", "can_overlap_with"):
            values = item.get(relation, [])
            if not isinstance(values, list):
                errors.append(f"{pattern_id}: {relation} must be a list")
                continue
            for target in values:
                if target not in canonical:
                    errors.append(f"{pattern_id}: {relation} references unknown ID {target}")
                if target == pattern_id:
                    errors.append(f"{pattern_id}: self-reference in {relation}")
        for alias in [*item.get("aliases", []), item.get("name_zh"), item.get("name_en")]:
            if not isinstance(alias, str) or not alias:
                continue
            owner = alias_owner.get(alias)
            if owner and owner != pattern_id:
                errors.append(f"alias collision: {alias!r} belongs to {owner} and {pattern_id}")
            alias_owner[alias] = str(pattern_id)
            if alias in canonical and alias != pattern_id:
                errors.append(f"alias collides with canonical ID: {alias}")

    for old, target in registry.get("variant_map", {}).items():
        if target not in canonical:
            errors.append(f"variant_map target missing: {old} -> {target}")
        if old in canonical:
            errors.append(f"variant_map source is also canonical: {old}")
    for old, target in registry.get("legacy_aliases", {}).items():
        if target not in canonical:
            errors.append(f"legacy_alias target missing: {old} -> {target}")

    try:
        conflict_payload = json.loads(CONFLICT_PATH.read_text(encoding="utf-8"))
        conflict_rules = conflict_payload.get("rules", [])
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"conflict_rules.json cannot be loaded: {exc}")
        conflict_rules = []
    conflict_seen: set[str] = set()
    for rule in conflict_rules:
        if not isinstance(rule, dict):
            errors.append("conflict rule entry is not an object")
            continue
        rule_id = str(rule.get("rule_id", "<missing>"))
        required = {
            "rule_id",
            "patterns",
            "condition",
            "winner_rule",
            "reject_reason",
            "requires_confirmation",
        }
        errors.extend(
            f"{rule_id}: conflict rule missing field {field}"
            for field in sorted(required - set(rule))
        )
        if rule_id in conflict_seen:
            errors.append(f"duplicate conflict rule_id: {rule_id}")
        conflict_seen.add(rule_id)
        for pattern_id in rule.get("patterns", []):
            if pattern_id and pattern_id not in canonical:
                errors.append(f"{rule_id}: unknown conflict pattern {pattern_id}")
        for field in ("winner_pattern", "contextual_alternative"):
            value = rule.get(field)
            if value and value not in canonical:
                errors.append(f"{rule_id}: unknown {field} {value}")

    return registry, errors


def main() -> int:
    registry, errors = _validate()
    if errors:
        print("Registry validation FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    patterns = registry["patterns"]
    print("Registry validation OK")
    print(f"patterns={len(patterns)}")
    print("categories=" + json.dumps(Counter(item["category"] for item in patterns), ensure_ascii=False))
    print("statuses=" + json.dumps(Counter(item["status"] for item in patterns), ensure_ascii=False))
    print(f"variant_map={len(registry.get('variant_map', {}))}")
    print(f"legacy_aliases={len(registry.get('legacy_aliases', {}))}")
    conflict_payload = json.loads(CONFLICT_PATH.read_text(encoding="utf-8"))
    print(f"conflict_rules={len(conflict_payload.get('rules', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
