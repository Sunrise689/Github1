"""Audit the canonical pattern vocabulary across the FAE judgment layer.

The audit is deliberately read-only with respect to the detector and rule
implementations.  It compares the detector's emitted IDs, expert-rule IDs,
compiled-case IDs, adapter aliases, compiler variant mappings, and the
frontend contract.  It produces a human-readable Markdown report so naming
drift is visible before a Pattern Registry is introduced.

Run from the project root::

    python -m fae.judgment.pattern_consistency_audit

The report is written next to this module as ``pattern_consistency_audit.md``.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
JUDGMENT = ROOT / "fae" / "judgment"
CHART_SOURCE = JUDGMENT / "chart_patterns.py"
RULES_PATH = JUDGMENT / "expert_rules.json"
ADAPTERS_SOURCE = JUDGMENT / "adapters.py"
COMPILER_SOURCE = JUDGMENT / "textbook_case_compiler.py"
FRONTEND_PROMPT = JUDGMENT / "frontend_pattern_knowledge_prompt.md"
CASE_INDEX = JUDGMENT / "cases" / "case_library_index.json"
REGISTRY_PATH = JUDGMENT / "pattern_registry.json"
CONFLICT_PATH = JUDGMENT / "conflict_rules.json"
REPORT_PATH = JUDGMENT / "pattern_consistency_audit.md"

VALID_CATEGORIES = {
    "simple_candlestick",
    "combination_candlestick",
    "trend_candlestick",
    "chart_pattern",
    "structure",
    "gap",
}

# These are the literal/dynamic outputs of ChartPatternDetector.  Dynamic
# branches are listed explicitly because the detector constructs them from
# suffix/flag variables rather than literal arguments at every call site.
DETECTOR_IDS = {
    "cup_with_handle",
    "dormant_bottom",
    "double_top",
    "double_bottom",
    "triple_top",
    "triple_bottom",
    "head_and_shoulders_top",
    "head_and_shoulders_bottom",
    "compound_head_and_shoulders_top",
    "compound_head_and_shoulders_bottom",
    "rounding_top",
    "rounding_bottom",
    "v_bottom",
    "inverted_v_top",
    "island_reversal_top",
    "island_reversal_bottom",
    "ascending_triangle",
    "descending_triangle",
    "symmetrical_triangle",
    "rising_wedge",
    "falling_wedge",
    "rectangle",
    "broadening_triangle",
    "ascending_flag",
    "descending_flag",
}

CHART_CANONICAL_IDS = {
    "cup_with_handle",
    "dormant_bottom",
    "double_top",
    "double_bottom",
    "head_and_shoulders_top",
    "head_and_shoulders_bottom",
    "rounding_top",
    "rounding_bottom",
    "v_bottom",
    "inverted_v_top",
    "island_reversal_top",
    "island_reversal_bottom",
    "ascending_triangle",
    "descending_triangle",
    "symmetrical_triangle",
    "rising_wedge",
    "falling_wedge",
    "rectangle",
    "broadening_triangle",
    "ascending_flag",
    "descending_flag",
}

LEGACY_VARIANTS = {
    "triple_top": "double_top",
    "triple_bottom": "double_bottom",
    "compound_head_and_shoulders_top": "head_and_shoulders_top",
    "compound_head_and_shoulders_bottom": "head_and_shoulders_bottom",
    "pennant": "symmetrical_triangle",
    "diamond": "rectangle",
    "v_top": "inverted_v_top",
    "cup_handle": "cup_with_handle",
    "latent_bottom": "dormant_bottom",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _literal_assignment(path: Path, name: str) -> Any:
    """Read a simple literal assignment without importing project code."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if value is None:
                    return None
                try:
                    return ast.literal_eval(value)
                except (ValueError, TypeError):
                    # Runtime aliases may now be loaded from the Registry;
                    # non-literal assignments are intentionally ignored here.
                    return None
    return None


def _frontend_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for match in re.finditer(r"`([a-z][a-z0-9_]+)`", text):
        value = match.group(1)
        if value in DETECTOR_IDS or value in CHART_CANONICAL_IDS or value in LEGACY_VARIANTS:
            ids.add(value)
    return ids


def _chart_rule_ids(rules: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for rule in rules:
        if str(rule.get("rule_id", "")).startswith("chart_"):
            ids.update(str(item) for item in rule.get("applies_to", []))
    return ids


def _normalise(value: str, aliases: dict[str, str], variants: dict[str, str]) -> str:
    """Resolve IDs through explicit compiler variants and adapter values."""

    return variants.get(value, value) if value not in aliases else aliases[value]


def _audit() -> dict[str, Any]:
    rules = _load_json(RULES_PATH)
    if isinstance(rules, dict):
        rules = rules.get("rules", [])
    case_index = _load_json(CASE_INDEX)
    registry = _load_json(REGISTRY_PATH)
    conflict_payload = _load_json(CONFLICT_PATH)
    conflict_rules = conflict_payload.get("rules", [])
    registry_records = registry.get("patterns", [])
    registry_ids = {
        str(item.get("pattern_id"))
        for item in registry_records
        if isinstance(item, dict) and item.get("pattern_id")
    }
    registry_variants = {
        str(old): str(target)
        for old, target in registry.get("variant_map", {}).items()
    }
    case_ids = set(case_index.get("by_pattern", {}))
    aliases = _literal_assignment(ADAPTERS_SOURCE, "_ALIASES") or {}
    variants = _literal_assignment(COMPILER_SOURCE, "CANONICAL_PATTERN_VARIANTS") or {}
    if not aliases:
        aliases = {
            str(alias): str(item.get("pattern_id"))
            for item in registry_records
            if isinstance(item, dict)
            for alias in [
                item.get("name_zh"),
                item.get("name_en"),
                *item.get("aliases", []),
            ]
            if alias
        }
    aliases_values = set(str(item) for item in aliases.values())
    chart_rule_ids = _chart_rule_ids(rules)
    frontend_ids = _frontend_ids(FRONTEND_PROMPT.read_text(encoding="utf-8"))

    # A source-level inventory is useful in the report, while the explicit
    # set above captures dynamic f-string branches that AST cannot evaluate.
    source_families = sorted(
        set(re.findall(r'family\s*=\s*["\']([^"\']+)["\']', CHART_SOURCE.read_text(encoding="utf-8")))
    )
    source_signal_literals = sorted(
        set(re.findall(r'_signal\(\s*["\']([a-z][a-z0-9_]*)["\']', CHART_SOURCE.read_text(encoding="utf-8")))
    )

    chart_case_ids = {
        item
        for item in case_ids
        if item in CHART_CANONICAL_IDS
        or item in DETECTOR_IDS
        or item in LEGACY_VARIANTS
        or item in {"pennant", "diamond", "inverted_cup_with_handle"}
    }
    unresolved_case_ids = sorted(
        item
        for item in chart_case_ids
        if item not in DETECTOR_IDS
        and item not in CHART_CANONICAL_IDS
        and item not in LEGACY_VARIANTS
    )
    rule_unresolved = sorted(
        item
        for item in chart_rule_ids
        if item not in DETECTOR_IDS
        and item not in CHART_CANONICAL_IDS
        and item not in LEGACY_VARIANTS
    )
    chart_rules_missing_detector = sorted(
        item
        for item in chart_rule_ids
        if item in CHART_CANONICAL_IDS and item not in DETECTOR_IDS
    )
    detector_without_rule = sorted(
        item
        for item in CHART_CANONICAL_IDS
        if item in DETECTOR_IDS and item not in chart_rule_ids
    )
    frontend_without_detector = sorted(
        item
        for item in frontend_ids
        if item in CHART_CANONICAL_IDS and item not in DETECTOR_IDS
    )
    aliases_shadowing_canonical = sorted(
        name for name in aliases if name in CHART_CANONICAL_IDS or name in DETECTOR_IDS
    )
    legacy_in_case_index = sorted(item for item in case_ids if item in LEGACY_VARIANTS)
    compiler_chart_ids = set(
        _literal_assignment(COMPILER_SOURCE, "CHART_PATTERNS") or []
    )
    compiler_legacy = sorted(item for item in compiler_chart_ids if item in LEGACY_VARIANTS)
    detector_canonical = {
        registry_variants.get(item, item) for item in DETECTOR_IDS
    }
    registry_missing_detector = sorted(
        str(item.get("pattern_id"))
        for item in registry_records
        if isinstance(item, dict)
        and item.get("status") == "missing"
        and item.get("pattern_id")
    )
    registry_id_set = set(registry_ids)
    conflict_unresolved: set[str] = set()
    for rule in conflict_rules:
        for pattern_id in rule.get("patterns", []):
            if pattern_id and pattern_id not in registry_id_set:
                conflict_unresolved.add(str(pattern_id))
        for field in ("winner_pattern", "contextual_alternative"):
            pattern_id = rule.get(field)
            if pattern_id and pattern_id not in registry_id_set:
                conflict_unresolved.add(str(pattern_id))

    return {
        "detector_ids": sorted(DETECTOR_IDS),
        "detector_source_literals": source_signal_literals,
        "detector_families": source_families,
        "chart_rule_ids": sorted(chart_rule_ids),
        "case_ids": sorted(case_ids),
        "chart_case_ids": sorted(chart_case_ids),
        "frontend_ids": sorted(frontend_ids),
        "aliases_count": len(aliases),
        "aliases_values": sorted(aliases_values),
        "variants": variants,
        "legacy_variants": LEGACY_VARIANTS,
        "legacy_in_case_index": legacy_in_case_index,
        "compiler_chart_ids": sorted(compiler_chart_ids),
        "compiler_legacy": compiler_legacy,
        "registry_ids": sorted(registry_ids),
        "registry_category_counts": dict(
            Counter(str(item.get("category")) for item in registry_records)
        ),
        "registry_status_counts": dict(
            Counter(str(item.get("status")) for item in registry_records)
        ),
        "detector_not_registered": sorted(detector_canonical - registry_ids),
        "registry_missing_detector": registry_missing_detector,
        "conflict_rule_count": len(conflict_rules),
        "conflict_unresolved": sorted(conflict_unresolved),
        "unresolved_case_ids": unresolved_case_ids,
        "rule_unresolved": rule_unresolved,
        "chart_rules_missing_detector": chart_rules_missing_detector,
        "detector_without_rule": detector_without_rule,
        "frontend_without_detector": frontend_without_detector,
        "aliases_shadowing_canonical": aliases_shadowing_canonical,
        "rule_count": len(rules),
        "case_count": int(case_index.get("case_count", 0)),
    }


def _bullets(items: list[str], empty: str = "无") -> str:
    return "\n".join(f"- `{item}`" for item in items) if items else f"- {empty}"


def render_report(result: dict[str, Any]) -> str:
    detector_ids = result["detector_ids"]
    chart_rules = result["chart_rule_ids"]
    canonical = sorted(CHART_CANONICAL_IDS)
    return f"""# FAE 形态一致性审计报告

审计时间：自动生成（本地文件扫描）  
审计范围：`chart_patterns.py`、`expert_rules.json`、`adapters.py`、`textbook_case_compiler.py`、`case_library_index.json`、`frontend_pattern_knowledge_prompt.md`。  
本报告只读比较现有层，不修改检测器、规则或案例。

## 结论摘要

- 图形检测器候选 ID：**{len(detector_ids)}** 个；源码显式 `family`：**{len(result['detector_families'])}** 个。
- 图形规则覆盖 ID：**{len(chart_rules)}** 个；规则文件总数：**{result['rule_count']}** 条。
- 图形 canonical 候选集合：**{len(canonical)}** 个。
- 案例索引总量：**{result['case_count']}** 条；其中图形/变体相关 ID：**{len(result['chart_case_ids'])}** 个。
- 当前最重要的结构性问题是：案例编译和图形检测都保留了若干历史变体 ID，而 canonical 化主要发生在适配/编译路径；Registry 应把这些变体正式声明为“主 ID + variant”，避免规则、案例和前端各自解释。

## 1. 检测器清单

### ChartPatternDetector 输出

{_bullets(detector_ids)}

### Detector family（源码）

{_bullets(result['detector_families'])}

### 源码中可直接看到的 `_signal` 字面量

{_bullets(result['detector_source_literals'])}

动态 f-string（如 `double_{{suffix}}`、`{{compound_}}head_and_shoulders_{{suffix}}`）已通过显式清单补入审计。

## 1.1 Registry 覆盖

- Registry 条目：**{len(result['registry_ids'])}** 个。
- 冲突规则：**{result['conflict_rule_count']}** 条。
- 按类别：`{json.dumps(result['registry_category_counts'], ensure_ascii=False)}`
- 按状态：`{json.dumps(result['registry_status_counts'], ensure_ascii=False)}`

### 检测器 canonical ID 未登记

{_bullets(result['detector_not_registered'])}

### Registry 中明确缺失检测器的条目

{_bullets(result['registry_missing_detector'])}

### 冲突规则引用但 Registry 中不存在的 ID

{_bullets(result['conflict_unresolved'])}

## 2. 规则覆盖

### `chart_*` 规则涉及的 ID

{_bullets(chart_rules)}

### 规则引用但当前没有对应检测器/兼容映射

{_bullets(result['rule_unresolved'])}

### 已有 canonical 检测器但没有 `chart_*` 规则

{_bullets(result['detector_without_rule'])}

## 3. 案例库与编译器

### 案例索引中的图形/变体 ID

{_bullets(result['chart_case_ids'])}

### 案例索引中的历史变体（应由 Registry 的 `variant_of`/兼容别名承接）

{_bullets(result['legacy_in_case_index'])}

### 编译器 `CHART_PATTERNS` 中的历史变体

{_bullets(result['compiler_legacy'])}

### 无法归入当前图形检测器或显式变体映射的案例 ID

{_bullets(result['unresolved_case_ids'])}

案例库存在大量 K 线、趋势线和指标类 ID；它们不应被误报为 chart pattern 缺失。后续 Registry 应按 category 分开登记。

## 4. 前端契约

### 前端提示词中出现的相关 ID

{_bullets(result['frontend_ids'])}

### 前端声称 canonical、但检测器当前没有输出的 ID

{_bullets(result['frontend_without_detector'])}

### 别名键与 canonical ID 同名（需要 Registry 明确优先级）

{_bullets(result['aliases_shadowing_canonical'])}

前端应始终以序列化的 `pattern_id` 为统计/去重键，以 `variant` 保存多重、复合、三角旗等历史称呼；旧 `pattern` 字段只作兼容展示。

## 5. 已确认的归一化关系

| 历史/变体 ID | canonical ID | 处理方式 |
|---|---|---|
{''.join(f'| `{old}` | `{new}` | 作为 variant/兼容输入，不单独建立主形态 |\n' for old, new in sorted(LEGACY_VARIANTS.items()))}|

## 6. 风险与建议

1. `triple_top`/`triple_bottom`、复合头肩形、`pennant`、`diamond`、`v_top` 等名称仍可能出现在旧案例或规则边界；Registry 应集中声明 canonical ID、alias、variant，而不是继续扩展检测器输出数量。
2. `chart_triple_top_bottom` 规则目前仍把三重顶/底作为 `applies_to`，但用户确认它们属于双顶/底的多重变体；Registry 建立后应把规则引用改成 canonical `double_top`/`double_bottom`，保留 `variant=multiple`。
3. `inverted_cup_with_handle` 在案例编译器/案例索引中存在，但当前图形检测器没有对应实现；本阶段标记为 `case_only/needs_review`，不自动伪造检测器。
4. 结构层（趋势线、支撑/阻力、Pivot）和缺口层不应混入 chart pattern 统计；Registry 需用 category 进行硬隔离。
5. 建议下一步按本报告冻结 canonical 集合，再建立 `pattern_registry.json` 和 `conflict_rules.json`，最后让规则、案例检索与前端都只消费 Registry。

## 7. 机器可复核字段

- 变体映射：`textbook_case_compiler.py::CANONICAL_PATTERN_VARIANTS`
- 输入别名：`adapters.py::_ALIASES`
- Registry：`pattern_registry.json`
- Registry 校验：`validate_registry.py`
- 图形规则：`expert_rules.json` 中 `rule_id` 以 `chart_` 开头的条目
- 报告生成脚本：`pattern_consistency_audit.py`
"""


def main() -> int:
    result = _audit()
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "detector_count": len(result["detector_ids"]),
                "chart_rule_count": len(result["chart_rule_ids"]),
                "case_count": result["case_count"],
                "rule_unresolved": result["rule_unresolved"],
                "detector_without_rule": result["detector_without_rule"],
                "legacy_case_ids": result["legacy_in_case_index"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
