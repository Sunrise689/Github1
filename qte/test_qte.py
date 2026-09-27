"""Offline regression and acceptance tests for the QTE knowledge layer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .index_builder import FACTORS, FRAMEWORKS, ROOT, build, load_records
from .retrieval import reload_index, search
from .translate import explain_terms, translate_text


CANONICAL_FACTOR_IDS = {
    "value_pe", "value_pb", "value_ps", "value_ev_ebitda", "quality_roe", "quality_gross_margin",
    "profitability_rmw", "quality_accruals", "growth_revenue", "growth_earnings", "investment_cma",
    "leverage", "dividend_yield", "earnings_surprise", "momentum_12_1", "momentum_6_1",
    "short_term_reversal", "residual_momentum", "volume_price_divergence", "volatility_realized",
    "idiosyncratic_volatility", "maximum_drawdown", "skewness", "kurtosis", "turnover",
    "liquidity_amihud", "trend_strength", "money_flow", "market_beta", "industry_beta", "low_beta",
    "industry_momentum", "industry_relative_strength", "size", "industry_rank", "correlation_structure",
    "crowding", "multi_factor",
}
LEGACY_FACTOR_IDS = {"market_risk_premium", "momentum", "quality_gross_profitability", "low_volatility", "dividend"}
CATEGORY_VALUES = {
    "fundamental_valuation", "fundamental_quality", "fundamental_growth", "fundamental_other",
    "technical_momentum", "technical_volatility", "technical_liquidity", "market_beta", "market_structure",
}
REQUIRED_FACTOR_FIELDS = {
    "factor_id", "name_zh", "name_en", "aliases", "category", "academic_definition", "plain_definition",
    "computation", "required_data", "evidence_assessment", "failure_scenarios", "translation_key", "sources",
    "status", "calculation_implementation", "data_availability",
}
PENDING_FACTORS = {
    "momentum_6_1", "volume_price_divergence", "money_flow", "industry_relative_strength", "industry_rank"
}
REQUIRED_NEW_TERMS = {
    "行业Beta", "残差动量", "量价背离", "特质波动率", "收益偏度", "收益峰度", "Amihud流动性", "资金流向",
    "行业动量", "行业相对强弱", "杠杆率", "盈利超预期", "分析师情绪", "拥挤度", "应计项", "市销率",
    "企业价值倍数", "归因残差",
}
REQUIRED_NEW_RULES = {
    "cross_factor_validation", "multi_level_beta", "factor_data_availability", "industry_context_required",
    "scenario_analysis", "no_single_factor_conclusion",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = _jsonl(ROOT / "knowledge" / "compiled_quant_knowledge.jsonl")
        cls.paper_evidence = _jsonl(ROOT / "evidence" / "compiled_quant_evidence.jsonl")
        cls.official_evidence = _jsonl(ROOT / "evidence" / "compiled_official_data_evidence.jsonl")
        cls.source_index = json.loads((ROOT / "qte_source_index.json").read_text(encoding="utf-8"))
        cls.registry = json.loads(FACTORS.read_text(encoding="utf-8"))
        cls.datasets = json.loads((ROOT / "data" / "official_dataset_registry.json").read_text(encoding="utf-8"))
        cls.lexicon = json.loads((ROOT / "translation_lexicon.json").read_text(encoding="utf-8"))
        cls.rules = json.loads((ROOT / "quant_rules.json").read_text(encoding="utf-8"))
        cls.links = json.loads((ROOT / "knowledge" / "factor_knowledge_links.json").read_text(encoding="utf-8"))

    def test_compilation_counts(self) -> None:
        summary = json.loads((ROOT / "knowledge" / "knowledge_library_index.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["compiled_source_count"], 18)
        self.assertEqual(summary["scanned_page_count"], 3922)
        self.assertEqual(len(self.knowledge), summary["knowledge_record_count"])
        paper = [item for item in self.paper_evidence if item["evidence_type"] == "paper_reported"]
        pending = [item for item in self.paper_evidence if item["evidence_type"] == "evidence_pending"]
        self.assertEqual(len(paper), summary["paper_evidence_count"])
        self.assertEqual(len(pending), summary["pending_evidence_count"])
        self.assertEqual(len(self.paper_evidence), summary["evidence_record_count"])
        self.assertEqual(len(paper), 9)
        self.assertEqual(len(pending), 5)
        self.assertEqual(len(self.official_evidence), 6)

    def test_all_engine_knowledge_is_traceable(self) -> None:
        compiled_sources = {item["source_id"]: item for item in self.source_index["sources"] if item["status"] == "compiled"}
        for record in self.knowledge:
            self.assertIn(record["source_id"], compiled_sources)
            self.assertTrue(record["source_sha256"])
            self.assertEqual(record["source_sha256"], compiled_sources[record["source_id"]]["sha256"])
            self.assertTrue(record["pdf_pages"])
            self.assertTrue(all(isinstance(page, int) and page > 0 for page in record["pdf_pages"]))
            self.assertTrue(record["plain_summary"])
            self.assertTrue(record["caveat"])
            self.assertTrue(record["policy_flags"]["no_raw_text"])
            self.assertGreaterEqual(record["quality_score"], 0.65)

    def test_factor_registry_contract(self) -> None:
        factors = self.registry["factors"]
        legacy = self.registry["legacy_compatibility_factors"]
        runtime_style = self.registry["runtime_style_factors"]
        self.assertEqual({item["factor_id"] for item in factors}, CANONICAL_FACTOR_IDS)
        self.assertEqual({item["factor_id"] for item in legacy}, LEGACY_FACTOR_IDS)
        self.assertEqual(self.registry["canonical_factor_count"], 38)
        self.assertEqual(self.registry["compatibility_factor_count"], 5)
        self.assertEqual(self.registry["runtime_style_factor_count"], 6)
        self.assertEqual(len(runtime_style), 6)
        self.assertFalse(
            {item["factor_id"] for item in runtime_style} & CANONICAL_FACTOR_IDS
        )
        self.assertEqual({item["category"] for item in factors}, CATEGORY_VALUES)
        self.assertEqual(len({item["translation_key"] for item in factors}), 38)
        priorities = [item["data_availability"]["priority"] for item in factors]
        self.assertEqual({level: priorities.count(level) for level in ("P0", "P1", "P2")}, {"P0": 16, "P1": 8, "P2": 14})
        for factor in factors:
            self.assertEqual(REQUIRED_FACTOR_FIELDS - set(factor), set(), factor["factor_id"])
            self.assertTrue(factor["aliases"], factor["factor_id"])
            self.assertTrue(factor["required_data"], factor["factor_id"])
            self.assertTrue(factor["failure_scenarios"], factor["factor_id"])
            self.assertTrue(factor["sources"], factor["factor_id"])
            self.assertEqual(factor["status"], "implemented")
            self.assertEqual(factor["calculation_implementation"], "not_implemented")
            self.assertIsInstance(factor["data_availability"]["current"], bool)
            self.assertTrue(factor["data_availability"]["requires"])

    def test_factor_references_are_registered(self) -> None:
        registered = CANONICAL_FACTOR_IDS | LEGACY_FACTOR_IDS
        referenced = {factor for record in self.knowledge for factor in record.get("factor_ids", [])}
        referenced.update(record["factor_id"] for record in self.paper_evidence + self.official_evidence)
        self.assertEqual(referenced - registered, set())

    def test_evidence_pending_is_honest(self) -> None:
        pending = [item for item in self.paper_evidence if item["evidence_type"] == "evidence_pending"]
        self.assertEqual({item["factor_id"] for item in pending}, PENDING_FACTORS)
        self.assertEqual(len({item["evidence_id"] for item in pending}), 5)
        for record in pending:
            self.assertEqual(record["source_id"], "qte_evidence_gap_registry")
            self.assertEqual(record["source_pages"], "not_available")
            self.assertEqual(record["source_sha256"], "")
            self.assertEqual(record["calculation"], "not_computed")
            self.assertTrue(record["policy_flags"]["evidence_pending"])
            self.assertTrue(record["policy_flags"]["no_empirical_claim"])
            self.assertTrue(record["policy_flags"]["no_numeric_result"])
            self.assertTrue(record["metadata"]["needed_source"])

    def test_paper_and_official_evidence_provenance(self) -> None:
        compiled_sources = {item["source_id"]: item for item in self.source_index["sources"] if item["status"] == "compiled"}
        datasets = {item["dataset_id"]: item for item in self.datasets["datasets"]}
        for record in self.paper_evidence:
            if record["evidence_type"] == "evidence_pending":
                continue
            self.assertEqual(record["source_sha256"], compiled_sources[record["source_id"]]["sha256"])
            self.assertNotIn("待", record["source_pages"])
            self.assertTrue(record["claim_plain"] and record["caveat"])
        for record in self.official_evidence:
            dataset = datasets[record["source_id"]]
            self.assertEqual(record["source_sha256"], dataset["source_sha256"])
            self.assertIn(dataset["data_version"], record["data_version"])
            self.assertTrue(record["calculation"])
            self.assertTrue(record["policy_flags"]["locally_recomputed"])

    def test_official_raw_hashes(self) -> None:
        for dataset in self.datasets["datasets"]:
            if dataset["provider"].startswith("Kenneth"):
                raw = ROOT / "data" / "raw" / "ken_french" / Path(dataset["source_url"]).name
            else:
                raw = ROOT / "data" / "raw" / "fred" / f"{dataset['series_id']}.csv"
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), dataset["source_sha256"])

    def test_frameworks_parse_and_preserve_boundary(self) -> None:
        self.assertEqual(len(FRAMEWORKS), 4)
        for path in FRAMEWORKS:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            if path.name == "ai_integration_protocol.json":
                self.assertTrue(payload["scope"]["numerical_engine_included"])
                self.assertEqual(
                    payload["scope"]["factor_engine_status"],
                    "implemented_deterministic_runtime",
                )
            else:
                self.assertFalse(payload["scope"]["numerical_engine_included"])
        attribution = json.loads((ROOT / "attribution_framework.json").read_text(encoding="utf-8"))
        probability = attribution["scenario_output"]["historical_probability_contract"]
        self.assertIn("unavailable", probability["allowed_status"])
        self.assertIn("不得", probability["rule"])
        integration = json.loads((ROOT / "ai_integration_protocol.json").read_text(encoding="utf-8"))
        self.assertFalse(integration["scope"]["llm_may_calculate"])
        self.assertTrue((ROOT / "factor_engine.py").exists())

    def test_lexicon_and_rules_contract(self) -> None:
        terms = [item["term"] for item in self.lexicon["terms"]]
        rule_ids = [item["rule_id"] for item in self.rules["rules"]]
        self.assertGreaterEqual(len(terms), 55)
        self.assertEqual(len(terms), len(set(terms)))
        self.assertEqual(REQUIRED_NEW_TERMS - set(terms), set())
        self.assertGreaterEqual(len(rule_ids), 19)
        self.assertEqual(len(rule_ids), len(set(rule_ids)))
        self.assertEqual(REQUIRED_NEW_RULES - set(rule_ids), set())

    def test_factor_knowledge_links(self) -> None:
        links = self.links["factor_links"]
        self.assertEqual(self.links["canonical_factor_count"], 38)
        self.assertEqual({item["factor_id"] for item in links}, CANONICAL_FACTOR_IDS)
        self.assertEqual(self.links["compiled_knowledge_record_count"], 1330)
        for item in links:
            self.assertNotEqual(item["link_status"], "unlinked", item["factor_id"])
            if item["link_status"] == "evidence_pending":
                self.assertTrue(item["pending_evidence_ids"], item["factor_id"])
        official_links = {item["factor_id"]: item for item in links}
        for factor_id in ("market_beta", "size", "value_pb", "momentum_12_1", "profitability_rmw", "investment_cma"):
            self.assertTrue(
                any(match["match_basis"] == "registered_source_id" for match in official_links[factor_id]["evidence_matches"]),
                factor_id,
            )

    def test_china_policy_is_backend_not_compiler_exclusion(self) -> None:
        cn = [record for record in self.knowledge if record["market"] == "CN"]
        self.assertTrue(cn)
        self.assertTrue(all(record["display_policy"]["backend_eligible"] for record in cn))
        self.assertTrue(all(record["display_policy"]["frontend_default"] == "hidden" for record in cn))
        nbs = next(item for item in self.datasets["registered_not_compiled_sources"] if item["source_id"] == "china_nbs_national_data")
        self.assertTrue(nbs["display_policy"]["backend_eligible"])

    def test_duplicate_and_optional_missing_are_explicit(self) -> None:
        self.assertEqual(len(self.source_index["duplicates"]), 1)
        self.assertEqual(len(self.source_index["missing"]), 1)
        self.assertTrue(self.source_index["missing"][0]["optional_remote"])

    def test_index_build_in_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = build(output_dir=Path(directory))
            self.assertEqual(summary["doc_count"], len(load_records()))
            self.assertGreaterEqual(summary["doc_count"], 1450)
            self.assertEqual(summary["by_record_type"]["knowledge"], 1330)
            self.assertEqual(summary["by_record_type"]["factor_definition"], 38)
            self.assertEqual(summary["by_record_type"]["legacy_factor_definition"], 5)
            self.assertEqual(summary["by_record_type"]["factor_evidence_link"], 38)
            self.assertEqual(summary["by_record_type"]["translation_term"], 55)
            self.assertEqual(summary["by_record_type"]["framework_specification"], 4)
            self.assertEqual(summary["by_record_type"]["evidence_pending"], 5)
            with np.load(Path(directory) / "bm25_index.npz", allow_pickle=False) as index:
                self.assertEqual(int(index["doc_count"][0]), summary["doc_count"])


class TestRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build()
        reload_index()
        cls.registry = json.loads(FACTORS.read_text(encoding="utf-8"))

    def test_multilingual_queries(self) -> None:
        for query in ("价值因子为什么会回撤", "momentum crash transaction cost", "Beta 大盘敏感度", "联邦基金利率"):
            self.assertTrue(search(query, top_k=3), query)

    def test_every_factor_chinese_english_alias_is_searchable(self) -> None:
        for factor in self.registry["factors"]:
            factor_id = factor["factor_id"]
            for label, query in (
                ("zh", factor["name_zh"]),
                ("en", factor["name_en"]),
                ("alias", factor["aliases"][0]),
            ):
                results = search(query, top_k=10, filters={"factor_id": factor_id})
                self.assertTrue(results, f"{factor_id} {label}: {query}")
                self.assertTrue(
                    any(item.get("record_type") == "factor_definition" and item.get("factor_id") == factor_id for item in results),
                    f"{factor_id} {label}: {query}",
                )

    def test_legacy_factor_filter(self) -> None:
        results = search("历史风险和失效", top_k=10, filters={"factor_id": "momentum"})
        self.assertTrue(results)
        self.assertTrue(all("momentum" in (item.get("factor_ids") or [item.get("factor_id")]) for item in results))

    def test_unavailable_factor_returns_definition_not_result(self) -> None:
        for factor in self.registry["factors"]:
            if factor["data_availability"]["current"]:
                continue
            results = search(factor["name_zh"], top_k=10, filters={"factor_id": factor["factor_id"]})
            definition = next(item for item in results if item.get("record_type") == "factor_definition")
            self.assertTrue(definition["definition_only"], factor["factor_id"])
            self.assertFalse(definition["numeric_result_available"], factor["factor_id"])
            self.assertEqual(definition["result_capability"], "definition_and_historical_evidence_only")

    def test_no_factor_definition_claims_numeric_result(self) -> None:
        for factor in self.registry["factors"]:
            result = next(
                item for item in search(factor["name_zh"], top_k=10, filters={"factor_id": factor["factor_id"]})
                if item.get("record_type") == "factor_definition"
            )
            self.assertFalse(result["numeric_result_available"], factor["factor_id"])

    def test_pending_search_surfaces_gap(self) -> None:
        for factor_id in PENDING_FACTORS:
            results = search("证据是否可靠 还缺什么", top_k=20, filters={"factor_id": factor_id})
            self.assertTrue(any(item.get("record_type") == "evidence_pending" for item in results), factor_id)

    def test_frameworks_are_searchable(self) -> None:
        self.assertTrue(search("为什么涨跌 因子贡献 残差 R平方", top_k=10, filters={"record_type": "framework_specification"}))
        self.assertTrue(search("验证不是运气 样本外 多重检验", top_k=10, filters={"record_type": "framework_specification"}))

    def test_china_backend_and_frontend_filter(self) -> None:
        backend = search("中国因子投资", top_k=5)
        self.assertTrue(any(record.get("market") == "CN" for record in backend))
        frontend = search("中国因子投资", top_k=10, filters={"frontend_visible": True})
        self.assertFalse(any(record.get("market") == "CN" for record in frontend))

    def test_translation_terms_are_searchable(self) -> None:
        results = search("Delta Gamma 是什么", top_k=5)
        terms = {record.get("term") for record in results if record.get("record_type") == "translation_term"}
        self.assertIn("Delta", terms)
        self.assertIn("Gamma", terms)

    def test_translation_preserves_original(self) -> None:
        original = "Beta=1.30，最大回撤=-42.50%，Delta=0.60。"
        translated = translate_text(original)
        self.assertTrue(translated.startswith(original))
        self.assertIn("大白话翻译：", translated)
        self.assertIn("Beta", {item["term"] for item in explain_terms(original)})
        self.assertIn("Delta", {item["term"] for item in explain_terms(original)})
        for token in ("1.30", "-42.50%", "0.60"):
            self.assertIn(token, translated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
