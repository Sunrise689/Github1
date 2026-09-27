"""Runtime access to the single FAE Pattern Registry.

This module keeps name normalisation in one place.  The detector still emits
its own raw candidate names, but every downstream consumer can resolve those
names through :func:`canonicalize_pattern_id` without maintaining a second
copy of the alias/variant table.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("pattern_registry.json")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Load and cache the local registry; no network or mutable global state."""

    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("patterns"), list):
        raise ValueError("pattern_registry.json must contain a patterns list")
    return value


@lru_cache(maxsize=1)
def pattern_records() -> dict[str, dict[str, Any]]:
    """Return canonical pattern records indexed by ``pattern_id``."""

    return {
        str(record["pattern_id"]): record
        for record in load_registry()["patterns"]
        if isinstance(record, dict) and record.get("pattern_id")
    }


@lru_cache(maxsize=1)
def canonical_ids() -> frozenset[str]:
    """Return all registered canonical IDs, including explicit missing entries."""

    return frozenset(pattern_records())


@lru_cache(maxsize=1)
def alias_map() -> dict[str, str]:
    """Build aliases from the registry records and compatibility maps."""

    registry = load_registry()
    result: dict[str, str] = {}
    for record in registry["patterns"]:
        pattern_id = str(record["pattern_id"])
        result[pattern_id] = pattern_id
        for field in ("name_zh", "name_en"):
            value = record.get(field)
            if value:
                result[str(value)] = pattern_id
        for value in record.get("aliases", []):
            result[str(value)] = pattern_id
    for old, target in registry.get("variant_map", {}).items():
        result[str(old)] = str(target)
    for old, target in registry.get("legacy_aliases", {}).items():
        result[str(old)] = str(target)
    return result


@lru_cache(maxsize=1)
def variant_map() -> dict[str, str]:
    """Return raw historical IDs and their canonical parent IDs."""

    return {
        str(old): str(target)
        for old, target in load_registry().get("variant_map", {}).items()
    }


@lru_cache(maxsize=1)
def legacy_alias_map() -> dict[str, str]:
    """Return legacy raw IDs that were renamed without becoming variants."""

    return {
        str(old): str(target)
        for old, target in load_registry().get("legacy_aliases", {}).items()
    }


@lru_cache(maxsize=1)
def variant_labels() -> dict[str, str]:
    """Return stable display variants for historical IDs."""

    return {
        "三个白色武士": "three_white_soldiers",
        "三白兵": "three_white_soldiers",
        "三重顶": "multiple_top",
        "多重顶": "multiple_top",
        "三重底": "multiple_bottom",
        "多重底": "multiple_bottom",
        "复合头肩顶": "compound_head_and_shoulders",
        "复合头肩底": "compound_head_and_shoulders",
        "三角旗": "pennant",
        "三角旗形": "pennant",
        "看涨孕线": "bullish_harami",
        "看跌孕线": "bearish_harami",
        "十字孕线": "doji_harami",
        "平顶": "flat_top",
        "平底": "flat_bottom",
        "黑三兵": "black_three_soldiers",
        "菱形": "box_consolidation",
        "钻石形": "box_consolidation",
        "菱形顶": "box_consolidation",
        "箱形整理": "box_consolidation",
        "箱体": "box_consolidation",
        "整理区间": "box_consolidation",
        "triple_top": "multiple",
        "triple_bottom": "multiple",
        "compound_head_and_shoulders_top": "compound",
        "compound_head_and_shoulders_bottom": "compound",
        "pennant": "pennant",
        "diamond": "historical_diamond",
        "v_top": "legacy_v_top",
        "cup_handle": "legacy_cup_handle",
        "latent_bottom": "legacy_latent_bottom",
    }


def canonicalize_pattern_id(value: str) -> str:
    """Resolve a raw ID, English name, or Chinese alias to a canonical ID."""

    key = str(value).strip()
    return alias_map().get(key, key)


def variant_for_pattern(value: str) -> str | None:
    """Return the stable variant label for a legacy raw ID, if one exists."""

    return variant_labels().get(str(value).strip())


def registry_alias_map() -> dict[str, str]:
    """Return a copy suitable for compatibility with the old adapter API."""

    return dict(alias_map())


def get_pattern(pattern_id: str) -> dict[str, Any] | None:
    """Look up a canonical record after applying ID/alias normalisation."""

    return pattern_records().get(canonicalize_pattern_id(pattern_id))
