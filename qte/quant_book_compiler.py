"""Compile books and papers into source-traceable QTE knowledge.

Every PDF page is scanned. Raw extracted/OCR text is an internal, resumable
build cache beside the source library; the QTE artifact contains only compact
structured summaries, page references, risk boundaries, and quality metadata.
The output is descriptive evidence and never a trading decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import QuantEvidenceRecord, QuantKnowledgeRecord

COMPILER_VERSION = "3.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path("C:/Users/18295/Desktop/QTE\u7406\u8bba")
DEFAULT_CATALOG = ROOT / "source_catalog.json"
DEFAULT_CACHE_DIR = DEFAULT_SOURCE_DIR / ".qte_build_cache"
DEFAULT_KNOWLEDGE_DIR = ROOT / "knowledge"
DEFAULT_EVIDENCE = ROOT / "evidence" / "compiled_quant_evidence.jsonl"


@dataclass
class PageRecord:
    source_id: str
    source_file: str
    page: int
    text: str
    confidence: float
    method: str
    text_sha1: str


TOPICS: dict[str, dict[str, Any]] = {
    "factor_foundations": {
        "aliases": ["factor investing", "factor premium", "factor premiums", "因子投资", "因子收益", "因子溢价", "risk factor", "return factor"],
        "factor_ids": [],
        "type": "definition",
        "academic": "因子是可测量、可重复定义的资产特征或共同收益来源；研究重点是定义、跨样本证据、经济解释和可实施性。",
        "plain": "因子就是给资产贴上一个可以用数据算出来的标签，再比较不同标签组在历史上有没有稳定差别。",
        "caveat": "因子是观察和归因工具，不是保证未来有效的赚钱配方。",
        "tags": ["factor_definition", "historical_evidence"],
    },
    "market_factor": {
        "aliases": ["market factor", "market risk premium", "mkt-rf", "rm-rf", "equity premium", "股票市场因子", "市场风险溢价", "市场因子"],
        "factor_ids": ["market_risk_premium"],
        "type": "factor_method",
        "academic": "市场因子描述广泛市场组合相对无风险利率的超额收益，是多因子模型中用于分离系统性市场暴露的基准项。",
        "plain": "先把整个市场一起涨跌带来的部分单独算出来，剩下的差别才可能来自价值、规模等其他因素。",
        "caveat": "市场溢价是历史样本统计，基准、无风险利率和样本区间都会影响结果。",
        "tags": ["market_premium", "benchmark"],
    },
    "beta": {
        "aliases": ["market beta", "beta exposure", "security market line", "capital asset pricing model", " capm ", "市场 beta", "市场贝塔", "贝塔", "证券市场线"],
        "factor_ids": ["market_beta", "low_volatility"],
        "type": "statistical_method",
        "academic": "Beta 是资产收益对市场收益的历史敏感度估计，用于描述系统性风险暴露，而非独立的方向预测。",
        "plain": "Beta 说的是它平时跟大盘一起动得有多猛，不是在预测下一次一定涨还是跌。",
        "caveat": "Beta 会随估计窗口、市场状态和数据频率变化，短样本尤其不稳定。",
        "tags": ["beta", "systematic_risk"],
    },
    "size": {
        "aliases": ["size factor", "size premium", "small minus big", " smb ", "small-cap", "small cap", "market capitalization", "规模因子", "规模效应", "小市值", "市值因子"],
        "factor_ids": ["size"],
        "type": "factor_method",
        "academic": "规模因子比较小市值与大市值资产组合的历史收益差异，并需同时评估流动性、微盘股和交易成本影响。",
        "plain": "把公司按大小分组，看看小公司和大公司的历史表现是否有系统差别。",
        "caveat": "规模效应对微盘股、流动性、退市处理和交易成本非常敏感。",
        "tags": ["size", "liquidity"],
    },
    "value": {
        "aliases": ["value factor", "value premium", "book-to-market", "book to market", " hml ", "earnings-to-price", "earnings yield", "valuation factor", "价值因子", "价值溢价", "账面市值比", "市净率", "市盈率", "估值因子"],
        "factor_ids": ["value_pe", "value_pb"],
        "type": "factor_method",
        "academic": "价值因子按价格相对基本面指标进行截面排序，比较相对便宜与相对昂贵资产的历史收益差异。",
        "plain": "先用同一把尺子比较谁相对便宜、谁相对贵，再观察两组过去有没有长期差别。",
        "caveat": "便宜可能反映真实经营风险；口径、行业结构和价值陷阱都会改变结果。",
        "tags": ["value", "valuation"],
    },
    "momentum": {
        "aliases": ["momentum factor", "momentum premium", "price momentum", "time series momentum", "cross-sectional momentum", "winners and losers", "umd", "动量因子", "动量效应", "价格动量", "赢家组合", "输家组合"],
        "factor_ids": ["momentum"],
        "type": "factor_method",
        "academic": "动量因子依据过去相对表现排序，检验赢家与输家组合在随后区间的收益延续差异。",
        "plain": "它测的是价格有没有惯性：过去相对强的，之后是否平均还能强一阵。",
        "caveat": "快速市场反转可能造成动量崩溃，高换手和交易成本也会削弱可实施结果。",
        "tags": ["momentum", "trend_persistence"],
    },
    "profitability_quality": {
        "aliases": ["profitability factor", "gross profitability", "gross profits-to-assets", "robust minus weak", " rmw ", "quality factor", "return on equity", " roe ", "盈利能力因子", "毛利率因子", "质量因子", "净资产收益率", "盈利能力"],
        "factor_ids": ["profitability_rmw", "quality_roe", "quality_gross_profitability"],
        "type": "factor_method",
        "academic": "盈利能力与质量因子使用利润、资本回报、稳健性等基本面指标，对企业经营质量进行可重复排序。",
        "plain": "它不是只看公司便不便宜，还看公司赚钱的质量和持续性怎么样。",
        "caveat": "不同会计口径、杠杆和无形资产处理会显著影响质量指标。",
        "tags": ["profitability", "quality", "accounting"],
    },
    "investment_factor": {
        "aliases": ["investment factor", "conservative minus aggressive", " cma ", "asset growth", "investment-to-assets", "投资因子", "资产增长", "保守投资", "激进投资"],
        "factor_ids": ["investment_cma"],
        "type": "factor_method",
        "academic": "投资因子比较资产扩张较保守与较激进企业的历史收益差异，是五因子模型中的企业投资维度。",
        "plain": "它比较公司扩张得慢还是快，并观察两类公司过去的回报是否存在系统差别。",
        "caveat": "资产增长可能反映机会也可能反映过度投资，结论依赖会计定义和市场环境。",
        "tags": ["investment", "asset_growth"],
    },
    "low_risk": {
        "aliases": ["low-risk investing", "low risk factor", "low volatility", "minimum variance", "betting against beta", " bab ", "idiosyncratic volatility", "低风险因子", "低波动因子", "低波动率", "最小方差", "低贝塔"],
        "factor_ids": ["low_volatility", "market_beta"],
        "type": "factor_method",
        "academic": "低风险研究检验低 Beta、低总波动或低特质波动资产的风险调整后表现，并比较统计风险与基本面风险定义。",
        "plain": "风险更大的资产不一定给出更好的历史回报；低波动研究就是检查这件事。",
        "caveat": "低风险组合仍会亏损，且可能有行业集中、估值变贵、杠杆和拥挤风险。",
        "tags": ["low_risk", "volatility"],
    },
    "dividend": {
        "aliases": ["dividend yield", "dividend factor", "dividend growth", "payout yield", "股息率", "股息因子", "分红率", "股利收益率"],
        "factor_ids": ["dividend"],
        "type": "factor_method",
        "academic": "股息维度使用现金分红相对价格或分红增长进行排序，并检验收入特征、估值和质量之间的关系。",
        "plain": "看公司给出的现金分红相对股价有多少，以及这种分红能不能持续。",
        "caveat": "高股息可能只是股价下跌的结果，也可能因分红削减而不可持续。",
        "tags": ["dividend", "income"],
    },
    "factor_model": {
        "aliases": ["factor model", "multifactor model", "multi-factor model", "three-factor model", "five-factor model", "four-factor model", "arbitrage pricing theory", "因子模型", "多因子模型", "三因子模型", "五因子模型", "四因子模型", "套利定价理论"],
        "factor_ids": [],
        "type": "statistical_method",
        "academic": "多因子模型把资产收益分解为共同因子暴露、因子收益和无法由模型解释的残差，用于归因与检验。",
        "plain": "它像拆账单：总表现里有多少来自大盘、价值、动量等共同因素，还有多少暂时解释不了。",
        "caveat": "模型只能解释其定义范围内的关系；漏掉变量、因子重叠和估计误差都会影响归因。",
        "tags": ["factor_model", "attribution"],
    },
    "regression_significance": {
        "aliases": ["linear regression", "time-series regression", "cross-sectional regression", "fama-macbeth", "t-statistic", "t statistic", "statistical significance", "r-squared", "r²", "回归分析", "时间序列回归", "截面回归", "t 值", "t值", "统计显著", "决定系数"],
        "factor_ids": [],
        "type": "validation_method",
        "academic": "回归与显著性统计用于估计变量关系及其不确定性，但必须结合模型设定、样本结构和多重检验判断。",
        "plain": "回归是在估计两件事通常怎么一起变化；t 值等指标是在问这个结果会不会只是噪声。",
        "caveat": "统计显著不等于经济上重要，更不自动代表因果关系或未来稳定。",
        "tags": ["regression", "significance"],
    },
    "correlation_diversification": {
        "aliases": ["correlation coefficient", "low correlation", "diversification", "covariance", "相关系数", "低相关", "分散化", "多元化", "协方差"],
        "factor_ids": [],
        "type": "portfolio_method",
        "academic": "相关性和协方差描述资产或因子共同变动的历史结构，是组合分散化与风险聚合的基础输入。",
        "plain": "关键不是每样东西单独好不好，而是它们会不会在同一时间一起出问题。",
        "caveat": "相关性会随市场状态变化，危机期可能突然升高；相关也不等于因果。",
        "tags": ["correlation", "diversification"],
    },
    "backtest_oos": {
        "aliases": ["backtest", "backtesting", "out-of-sample", "out of sample", "walk-forward", "holdout sample", "回测", "样本外", "滚动验证", "留出样本"],
        "factor_ids": [],
        "type": "validation_method",
        "academic": "回测通过历史数据复现规则；样本外检验用于评估规则在未参与设计的数据中能否保持方向和量级。",
        "plain": "先用一段历史想办法，再用没参与调参的另一段历史考试，避免把答案背下来。",
        "caveat": "一次样本外测试也可能被反复选择；时间泄漏和数据修订会制造虚假稳定性。",
        "tags": ["backtest", "out_of_sample"],
    },
    "overfitting_bias": {
        "aliases": ["overfitting", "data mining", "multiple testing", "p-hacking", "selection bias", "survivorship bias", "look-ahead bias", "data snooping", "过拟合", "数据挖掘", "多重检验", "幸存者偏差", "前视偏差", "选择偏差"],
        "factor_ids": [],
        "type": "validation_method",
        "academic": "过拟合和研究偏差会把样本噪声、事后信息或被遗漏的失败对象误认成稳定规律。",
        "plain": "如果试了很多方法只挑最好看的一个，或者只看活下来的公司，结果很容易显得虚假优秀。",
        "caveat": "需要预先定义规则、完整样本、时间点数据、样本外验证和多重检验修正共同控制。",
        "tags": ["overfitting", "bias", "multiple_testing"],
    },
    "data_integrity": {
        "aliases": ["point-in-time data", "data cleaning", "missing data", "restatement", "delisting", "database bias", "as-reported", "时间点数据", "数据清洗", "缺失值", "财报更正", "退市收益", "数据库偏差", "可得时点"],
        "factor_ids": [],
        "type": "data_method",
        "academic": "量化研究需要按当时真实可得的信息构造数据，并明确缺失值、退市、财报更正和数据库修订规则。",
        "plain": "不能拿今天才知道的数据假装当年已经知道；否则回测会偷偷看到未来。",
        "caveat": "数据供应商更新和历史回填会改变结果，必须记录版本与下载日期。",
        "tags": ["point_in_time", "data_quality"],
    },
    "transaction_costs_turnover": {
        "aliases": ["transaction cost", "transactions costs", "trading cost", "market impact", "bid-ask spread", "turnover", "portfolio turnover", "交易成本", "冲击成本", "买卖价差", "换手率", "组合换手"],
        "factor_ids": [],
        "type": "implementation_method",
        "academic": "可实施收益必须扣除佣金、价差、冲击成本、融资与借券成本，并与组合换手和容量共同评估。",
        "plain": "纸面上赚到的差异，真正交易时会被手续费、价差和大额成交的冲击吃掉一部分。",
        "caveat": "成本取决于时期、市场、规模和执行方式，固定成本假设通常过于乐观。",
        "tags": ["costs", "turnover", "implementation"],
    },
    "portfolio_construction": {
        "aliases": ["portfolio construction", "portfolio optimization", "mean-variance", "portfolio weights", "risk budget", "tracking error", "组合构建", "组合优化", "均值方差", "组合权重", "风险预算", "跟踪误差"],
        "factor_ids": [],
        "type": "portfolio_method",
        "academic": "组合构建把预期收益、风险、相关性和实施约束转换为权重，并控制集中度、基准偏离和交易成本。",
        "plain": "知道哪些特征可能有用还不够，还要决定每样放多少，避免某一项把整个组合拖垮。",
        "caveat": "优化结果对输入误差很敏感，过度精确的权重可能只是估计噪声。",
        "tags": ["portfolio_construction", "optimization"],
    },
    "rebalancing": {
        "aliases": ["rebalancing", "rebalance", "reconstitution", "portfolio maintenance", "再平衡", "调仓", "组合维护", "定期重构"],
        "factor_ids": [],
        "type": "implementation_method",
        "academic": "再平衡规定信号更新、权重回归目标和交易触发规则，是因子暴露稳定性与成本之间的折中。",
        "plain": "组合不能每天追着最新排名乱动，也不能永远不管；需要规定什么时候值得调整。",
        "caveat": "频率越高不一定越好，高换手可能超过信号更新带来的收益。",
        "tags": ["rebalancing", "turnover"],
    },
    "performance_attribution": {
        "aliases": ["performance attribution", "performance measurement", "information ratio", "active return", "active risk", "tracking error", "绩效归因", "业绩归因", "信息比率", "主动收益", "主动风险"],
        "factor_ids": [],
        "type": "validation_method",
        "academic": "绩效归因区分市场、因子、选股、时点选择和实施成本贡献；信息比率衡量主动收益相对主动风险的历史效率。",
        "plain": "不是只看最后赚没赚，而是拆清楚到底靠大盘、因子、个别选择还是运气。",
        "caveat": "归因结果依赖基准和因子模型；换一个模型可能得到不同解释。",
        "tags": ["performance_attribution", "information_ratio"],
    },
    "risk_drawdown": {
        "aliases": ["drawdown", "maximum drawdown", "tail risk", "crash risk", "downside risk", "回撤", "最大回撤", "尾部风险", "崩溃风险", "下行风险"],
        "factor_ids": [],
        "type": "risk_method",
        "academic": "回撤与尾部风险刻画均值和波动率之外的路径损失，帮助识别因子在压力期的非对称风险。",
        "plain": "平均表现好不代表过程好受；回撤是在看从高点掉下来最痛的一段。",
        "caveat": "历史最大回撤不是未来损失上限，短样本尤其容易漏掉极端情形。",
        "tags": ["drawdown", "tail_risk"],
    },
    "liquidity_capacity": {
        "aliases": ["liquidity", "illiquidity", "capacity", "short-sale constraint", "short selling", "borrow cost", "流动性", "非流动性", "策略容量", "卖空约束", "借券成本"],
        "factor_ids": [],
        "type": "implementation_method",
        "academic": "流动性、卖空条件和策略容量决定研究收益能否以目标规模实现，也是小盘和多空因子的重要约束。",
        "plain": "理论上有机会，不等于真能按那个价格买到、卖掉或借到足够多的资产。",
        "caveat": "压力期流动性可能突然消失，历史平均成本会低估极端执行风险。",
        "tags": ["liquidity", "capacity"],
    },
    "behavioral_explanation": {
        "aliases": ["behavioral finance", "behavioral bias", "overreaction", "underreaction", "limits to arbitrage", "investor sentiment", "行为金融", "行为偏差", "过度反应", "反应不足", "套利限制", "投资者情绪"],
        "factor_ids": [],
        "type": "economic_explanation",
        "academic": "行为解释把部分因子溢价归因于投资者反应不足、过度反应、偏好和套利限制，而非单一风险补偿。",
        "plain": "价格偏差可能来自人会犯相似的错，而且专业资金也不一定能立刻把错误纠正。",
        "caveat": "风险解释与行为解释可能同时成立，经验相关性通常不能单独区分因果机制。",
        "tags": ["behavioral_finance", "limits_to_arbitrage"],
    },
    "expected_returns": {
        "aliases": ["expected return", "expected returns", "risk premium", "risk premia", "historical return", "forward-looking", "预期收益", "风险溢价", "历史收益", "前瞻收益"],
        "factor_ids": [],
        "type": "definition",
        "academic": "预期收益是条件性的未来平均概念，历史平均值只是估计输入，还需要估值、风险、制度变化和不确定性约束。",
        "plain": "过去平均赚多少不是未来的承诺，它最多是估计未来时的一块参考材料。",
        "caveat": "样本均值估计误差很大，结构变化会让长期历史平均失去代表性。",
        "tags": ["expected_return", "risk_premium"],
    },
    "macro_regimes": {
        "aliases": ["interest rate", "inflation", "business cycle", "economic growth", "term premium", "credit premium", "currency carry", "macro factor", "利率", "通胀", "经济周期", "经济增长", "期限溢价", "信用溢价", "货币套利", "宏观因子"],
        "factor_ids": [],
        "type": "context_method",
        "academic": "宏观变量可用于描述因子表现所处的利率、通胀、增长、信用和波动环境，但需要避免把同时发生误写成因果。",
        "plain": "同一个因子在不同经济天气里可能表现不同；宏观数据是背景，不是单独的涨跌按钮。",
        "caveat": "宏观数据有发布时间滞后和修订，历史同期分析必须使用当时可得版本。",
        "tags": ["macro_context", "regime"],
    },
    "conflicts_price_support": {
        "aliases": ["price support", "bank-affiliated mutual funds", "conflict of interest", "fund flows", "价格支持", "关联基金", "利益冲突", "基金流量"],
        "factor_ids": [],
        "type": "supplementary_research",
        "academic": "机构关联关系和利益冲突可能影响基金交易与价格形成，是评估市场微观结构和样本外可迁移性时的补充背景。",
        "plain": "有些交易不是因为资产本身更好，而可能因为机构之间存在关系或其他目标。",
        "caveat": "该主题不是 QTE 核心因子证据，特定国家和机构样本不能直接推广到广泛市场。",
        "tags": ["market_microstructure", "conflicts_of_interest"],
    },
}


SOURCE_OVERVIEWS: dict[str, tuple[str, str, str, list[str]]] = {
    "book_active_portfolio_management": ("该书建立从收益预测、风险模型到组合构建、交易成本和绩效归因的主动量化管理框架。", "它负责告诉 QTE：一个看起来有效的信号，怎样经过风险、成本和组合约束后才算可用。", "机构主动管理框架不等于普通投资者的交易建议。", ["factor_model", "portfolio_construction", "performance_attribution"]),
    "book_expected_returns": ("该书综合讨论跨资产风险溢价、估值、宏观环境和历史收益估计，是 QTE 解释预期收益与风险边界的综合来源。", "它帮助 QTE 说明各种历史收益从哪里可能来，以及为什么过去平均不能当成未来保证。", "跨资产长期样本存在口径和时代差异。", ["expected_returns", "macro_regimes", "value"]),
    "book_quant_equity_pm": ("该书覆盖因子选择、股票排序、风险模型、组合权重、再平衡、成本、回测与归因。", "它负责把“找到因子”接到“能不能真正做成组合”这一步。", "示例参数和历史结果需按数据版本复核。", ["portfolio_construction", "transaction_costs_turnover", "backtest_oos"]),
    "book_quant_momentum": ("该书从行为解释、信号构造、组合设计和历史验证讨论量化动量。", "它把动量从一句“强者恒强”拆成可计算、可检验、会失效的规则。", "实践型回测需与独立论文和官方数据交叉验证。", ["momentum", "backtest_oos", "risk_drawdown"]),
    "book_quant_value": ("该书讨论量化价值筛选、企业质量、估值指标和系统化实施。", "它帮助 QTE 解释“便宜”怎样被量化，以及为什么便宜不等于一定值得买。", "实践型策略结果不能替代跨市场学术证据。", ["value", "profitability_quality", "overfitting_bias"]),
    "book_magic_formula_en": ("该书以盈利质量与估值相结合的排序方法讲解系统化价值投资。", "核心是用两个简单问题筛选：生意质量怎么样，价格相对贵不贵。", "该书属于实践方法，不能单独作为普遍有效性的证明。", ["value", "profitability_quality"]),
    "book_factor_guide": ("该书以持久性、广泛性、直觉性、可实施性等标准筛选因子并讨论多因子配置。", "它帮助 QTE 判断一个因子是长期证据，还是只在一小段数据里看起来漂亮。", "评级仍依赖所引用研究的样本和实施假设。", ["factor_foundations", "overfitting_bias", "transaction_costs_turnover"]),
    "book_factor_investing_cn": ("该书系统覆盖中国市场因子数据、时间点处理、因子检验、组合构建与风险模型。", "它让 QTE 能理解中文市场里的因子研究，也强调不能偷看未来数据。", "中国案例可进入后端研究；前端是否展示由产品展示策略单独控制。", ["factor_foundations", "data_integrity", "backtest_oos"]),
    "book_magic_formula_zh": ("该中文译本以通俗案例解释盈利质量与估值结合的系统化筛选思路。", "它适合提供中文人话表达，但证据权重仍以论文和官方数据为主。", "与英文原著属于同一作品，不应被重复计为两份独立证据。", ["value", "profitability_quality"]),
    "paper_aqr_factor_2023": ("该综述区分因子投资中的事实与常见误解，强调跨市场证据、经济解释、实施和组合方式。", "它帮助 QTE 回答：什么算因子，哪些争议是真的，哪些只是误解。", "综述结论需回到其引用的原始研究和数据核验。", ["factor_foundations", "overfitting_bias", "portfolio_construction"]),
    "paper_ff1993": ("该论文构建股票与债券收益的共同风险因子框架，奠定市场、规模、价值等因子模型基础。", "它解释为什么要把股票表现拆成大盘、公司大小和相对便宜程度等共同部分。", "结论基于特定美国历史样本和组合构造。", ["market_factor", "size", "value", "factor_model"]),
    "paper_novy_marx_2013": ("该论文研究总盈利能力相对资产的预测信息及其与价值维度的互补关系。", "它说明只看便宜不够，公司赚钱能力可能是价值的另一面。", "盈利指标受会计定义和样本构造影响。", ["profitability_quality", "value"]),
    "paper_price_support_2014": ("该论文研究银行关联基金的价格支持与利益冲突，是市场微观结构补充材料。", "它提醒 QTE：机构交易有时受关联关系影响，不一定只反映资产价值。", "该研究不是核心因子论文，默认进入补充/复核层。", ["conflicts_price_support", "liquidity_capacity"]),
    "paper_aqr_momentum_2014": ("该论文检验动量投资的常见质疑，包括规模、空头端、成本和跨市场稳定性。", "它用证据回答动量是不是只在少数小股票或纸面回测里有效。", "作者汇总的历史证据不预示未来，动量仍有严重反转风险。", ["momentum", "transaction_costs_turnover", "risk_drawdown"]),
    "paper_aqr_value_2015": ("该论文讨论价值投资的定义、风险与行为解释、组合集中和其他因子冗余等争议。", "它帮助 QTE 区分“便宜公司”与“坏公司”，也说明价值为什么可能长期难熬。", "价值定义和样本时期会影响结论。", ["value", "behavioral_explanation", "factor_model"]),
    "paper_carhart_1997": ("该论文使用无幸存者偏差的基金样本研究业绩持续性，并引入动量因子的四因子归因框架。", "它提醒 QTE：基金过去表现持续，可能来自共同因子、费用或偶然持仓，而不一定是经理能力。", "基金样本结论不能直接等同于个股动量策略收益。", ["momentum", "performance_attribution", "overfitting_bias"]),
    "paper_ff1992": ("该论文研究美国股票平均收益横截面与规模、账面市值比及 Beta 的关系。", "它是“公司大小和相对便宜程度为什么进入因子模型”的经典来源。", "结果来自 1963 至 1990 年美国样本，后续市场和模型需独立验证。", ["size", "value", "beta"]),
    "paper_ff2015_nber": ("该论文在市场、规模和价值因子基础上加入盈利能力与投资因子，评估五因子模型的解释力。", "它把公司赚不赚钱、扩张快不快也纳入收益拆分。", "模型仍可能遗漏动量等维度，且部分因子存在重叠。", ["factor_model", "profitability_quality", "investment_factor"]),
    "paper_aqr_low_risk_2020": ("该论文梳理低风险投资的跨市场、样本外、成本与经济解释证据。", "它解释为什么风险更大不一定历史回报更好，也强调低风险组合仍会亏损。", "低风险定义多样，结果受行业、估值、杠杆和拥挤影响。", ["low_risk", "transaction_costs_turnover", "risk_drawdown"]),
}


CURATED_EVIDENCE: dict[str, list[dict[str, Any]]] = {
    "paper_ff1992": [
        {"pages":"1","factor_id":"value_pb","market":"US","period":"1963-07 to 1990-12","claim":"在该论文的美国股票样本中，规模与账面市值比共同刻画了平均收益横截面的主要差异；在控制规模后，账面市值比仍与平均收益相关。","plain":"在这段美国历史样本里，公司大小和账面价值相对股价的高低，比只看 Beta 更能区分不同股票组的平均表现。","caveat":"这是特定历史样本和排序方法的结论，不预示未来；后续数据、交易成本和模型设定可能改变结果。","quality":0.96,"calculation":"paper-reported cross-sectional portfolio sorts and regressions"},
        {"pages":"1, 10-21","factor_id":"size","market":"US","period":"1963-07 to 1990-12","claim":"该论文报告，在其样本与控制变量设定下，规模与平均股票收益存在负向横截面关系。","plain":"在这段历史样本中，小公司组和大公司组的平均表现有系统差别，但这不是说小公司以后一定更好。","caveat":"微盘股、流动性、退市处理和后续时期稳健性必须另行检验。","quality":0.94,"calculation":"paper-reported size sorts and Fama-MacBeth regressions"},
    ],
    "paper_ff1993": [
        {"pages":"1-4","factor_id":"market_risk_premium","market":"US","period":"1963 to 1991","claim":"该论文提出用市场、规模与账面市值比相关组合解释股票收益的共同变动，并以期限与违约相关因子描述债券收益共同变动。","plain":"作者把股票和债券的共同涨跌拆成几类大因素，用来判断一项收益到底来自哪里。","caveat":"因子构造、样本市场和估计期间具有特定性，模型解释不等于因果。","quality":0.96,"calculation":"paper-reported factor-mimicking portfolio regressions"},
    ],
    "paper_carhart_1997": [
        {"pages":"1-2","factor_id":"momentum","market":"US","period":"1962 to 1993","claim":"该论文在无幸存者偏差的基金样本中发现，基金业绩持续性大部分可由共同因子、费用和一年期收益持续解释，并使用动量因子扩展归因模型。","plain":"基金过去表现延续，不一定都是经理更聪明；可能是持有了当时有动量的资产，也可能只是费用更低或运气。","caveat":"研究对象是共同基金业绩归因，不应直接解释为动量策略的未来收益保证。","quality":0.96,"calculation":"paper-reported survivor-bias-free mutual-fund regressions"},
    ],
    "paper_novy_marx_2013": [
        {"pages":"1-2","factor_id":"quality_gross_profitability","market":"US","period":"1963-07 to 2010-12","claim":"该论文报告，总利润相对资产的盈利能力指标对平均收益具有与账面市值比相近的预测信息，并与传统价值信号互补。","plain":"在该美国历史样本里，既便宜又能稳定赚取较高毛利润的公司组，提供了不同于只看便宜的观察角度。","caveat":"盈利能力依赖会计定义；历史预测关系不预示未来，也不等于单只公司质量判断。","quality":0.96,"calculation":"paper-reported portfolio sorts and Fama-MacBeth regressions"},
    ],
    "paper_aqr_momentum_2014": [
        {"pages":"1-3, 20-24","factor_id":"momentum","market":"global","period":"multiple historical samples","claim":"该文汇总并检验动量在多市场、多资产和样本外时期的证据，同时讨论空头端、规模与交易成本等常见质疑。","plain":"动量不是只在一小段美国小盘股历史里出现，但它仍会经历很痛的反转，不能把长期证据当成每期都有效。","caveat":"该文属于证据综述与再检验；具体数字应回到引用数据版本复算。","quality":0.88,"calculation":"paper synthesis and robustness tests"},
    ],
    "paper_aqr_value_2015": [
        {"pages":"1-3, 24-28","factor_id":"value_pb","market":"global","period":"multiple historical samples","claim":"该文检验价值投资关于集中度、坏公司、因子冗余、长多限制及风险解释的常见说法，并强调价值定义与组合实施的重要性。","plain":"价值不是随便买便宜公司，也不只适用于少数集中持仓；但为什么会有价值溢价，风险和行为解释都不能被简单排除。","caveat":"该文为综合检验，结论依赖价值度量和所引用样本；历史结果不预示未来。","quality":0.88,"calculation":"paper synthesis and robustness tests"},
    ],
    "paper_aqr_factor_2023": [
        {"pages":"3-4, 19-22","factor_id":"multi_factor","market":"global","period":"multiple historical samples","claim":"该综述强调评价因子需要同时考虑长期与跨市场证据、经济解释、样本外表现、成本和组合实施，不能仅凭单次显著回测。","plain":"一个因子值得相信，不是因为一张回测图好看，而是要在不同地方、不同时间经得住验证，并且扣完成本还能落地。","caveat":"综述汇总多个研究，具体因子结论应与原始论文及官方数据交叉核对。","quality":0.88,"calculation":"review of empirical and implementation literature"},
    ],
    "paper_ff2015_nber": [
        {"pages":"1-4","factor_id":"profitability_rmw","market":"US","period":"1963 to 2013","claim":"该五因子研究在市场、规模和价值之外加入盈利能力与投资维度，报告模型对多类平均股票收益的解释有所改善。","plain":"除了公司大小和便不便宜，作者还加入了赚不赚钱、扩张快不快来拆解历史收益。","caveat":"五因子模型仍未包含动量，且 HML 等因子可能与新增因子重叠；模型解释不预示未来。","quality":0.95,"calculation":"paper-reported time-series asset-pricing tests"},
    ],
    "paper_aqr_low_risk_2020": [
        {"pages":"3-4, 8-10, 17-18","factor_id":"low_volatility","market":"global","period":"multiple samples through 2019","claim":"该文汇总低风险策略在样本内、样本外、多国家和多资产中的历史风险调整后证据，并讨论交易成本与小盘股依赖。","plain":"低风险资产在很多历史样本里并没有因为更稳就必然少赚，但低风险组合在市场下跌时照样可能亏损。","caveat":"低风险度量和组合构造多样，且估值、行业集中、杠杆与拥挤会改变实际结果。","quality":0.88,"calculation":"paper synthesis and cross-market robustness tests"},
    ],
}


PENDING_FACTOR_EVIDENCE: list[dict[str, Any]] = [
    {
        "factor_id": "momentum_6_1",
        "needed": "专门采用过去 6 个月、跳过最近 1 个月口径的原始论文或可复算数据，并与 12-1 动量和短期反转做并列稳健性检验。",
        "existing_context": [
            {"source_id": "book_quant_momentum", "pages": "27-28", "scope": "一般动量窗口讨论，不足以证明 6-1 专门口径"}
        ],
    },
    {
        "factor_id": "volume_price_divergence",
        "needed": "明确定义价与量方向、观察窗口、信号阈值和交易时点的专门研究或可复算数据。",
        "existing_context": [
            {"source_id": "book_quant_equity_pm", "pages": "226", "scope": "量价信息的一般性提及，不是量价背离专门实证"}
        ],
    },
    {
        "factor_id": "money_flow",
        "needed": "可审计的逐笔成交方向或统一大单口径、供应商方法文档，以及样本外和成本调整后的专门研究。",
        "existing_context": [],
    },
    {
        "factor_id": "industry_relative_strength",
        "needed": "明确行业分类、行业指数、市场基准和观察窗口的行业相对强弱原始研究或可复算数据。",
        "existing_context": [
            {"source_id": "book_active_portfolio_management", "pages": "321-323", "scope": "行业和动量方法背景，不是该专门口径的完整实证"}
        ],
    },
    {
        "factor_id": "industry_rank",
        "needed": "预先固定行业内综合因子、权重和分位算法，并提供独立样本外验证；当前定义属于产品构造，不是已验证溢价。",
        "existing_context": [
            {"source_id": "book_active_portfolio_management", "pages": "77-103", "scope": "行业风险与分类背景，不是行业内综合排名的专门实证"}
        ],
    },
]


def _pending_evidence() -> list[QuantEvidenceRecord]:
    """Create honest gap records; these records contain no empirical claim."""
    output: list[QuantEvidenceRecord] = []
    for index, item in enumerate(PENDING_FACTOR_EVIDENCE, start=1):
        factor_id = str(item["factor_id"])
        output.append(
            QuantEvidenceRecord(
                evidence_id=f"qe_pending_{factor_id}",
                evidence_type="evidence_pending",
                factor_id=factor_id,
                claim="当前 QTE 没有足够的专门、可核验实证证据支持该因子口径；本记录只登记证据缺口，不构成有效性结论。",
                claim_plain="这个概念可以先讲清楚，但现在还没有足够资料证明这套具体算法靠不靠谱，所以只能介绍，不能拿它下方向性结论。",
                caveat="不得把因子定义、一般性教材背景或相邻因子的证据改写成该因子已经验证；补齐来源后仍须通过 validation_pipeline。",
                source_id="qte_evidence_gap_registry",
                source_title="QTE Evidence Gap Registry",
                source_kind="internal_gap_registry",
                source_pages="not_available",
                source_url="",
                source_sha256="",
                market="global",
                period="not_available",
                frequency="not_applicable",
                calculation="not_computed",
                data_version="qte_gap_registry_2026-08-13",
                quality=1.0,
                display_policy={"backend_eligible": True, "frontend_default": "visible"},
                policy_flags={
                    "no_raw_text": True,
                    "historical_qualifier_required": True,
                    "no_trade_advice": True,
                    "engine_eligible": True,
                    "evidence_pending": True,
                    "no_empirical_claim": True,
                    "no_numeric_result": True,
                },
                metadata={"needed_source": item["needed"], "existing_context": item["existing_context"]},
            )
        )
    return output


def _safe_cache_name(source_id: str, path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{source_id}_{digest}.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_quality(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    readable = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact))
    replacement = compact.count("�")
    return max(0.0, min(1.0, readable / len(compact) - replacement * 0.01))


def _load_cache(path: Path) -> dict[int, PageRecord]:
    if not path.exists():
        return {}
    records: dict[int, PageRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = PageRecord(**json.loads(line))
            records[record.page] = record
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def _ocr_page(page: Any, zoom: float, threads: int) -> tuple[str, float]:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=max(1, threads), inter_op_num_threads=1)
    # Render inside the open PyMuPDF document.  Keeping this helper valid is
    # important even though the bulk scanner currently reuses one OCR engine.
    import fitz

    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    result, _ = engine(image)
    if not result:
        return "", 0.0
    text = "\n".join(str(item[1]) for item in result)
    scores = [float(item[2]) for item in result]
    return text, float(np.mean(scores)) if scores else 0.0


def scan_source(source: dict[str, Any], path: Path, cache_path: Path, *, zoom: float = 1.8, ocr_threads: int = 2) -> dict[str, Any]:
    import fitz

    cached = _load_cache(cache_path)
    document = fitz.open(path)
    ocr_engine: Any = None
    written = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        for index in range(len(document)):
            page_number = index + 1
            if page_number in cached:
                continue
            page = document[index]
            text = page.get_text("text", sort=True).strip()
            quality = _text_quality(text)
            method = "text"
            confidence = 0.99 * quality
            force_ocr = bool(source.get("force_ocr"))
            content_len = len(re.sub(r"\s+", "", text))
            should_ocr = force_ocr or (content_len < 80 and 3 < page_number < len(document) - 2)
            if should_ocr:
                if ocr_engine is None:
                    from rapidocr_onnxruntime import RapidOCR

                    ocr_engine = RapidOCR(intra_op_num_threads=max(1, ocr_threads), inter_op_num_threads=1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
                result, _ = ocr_engine(image)
                if result:
                    text = "\n".join(str(item[1]) for item in result)
                    scores = [float(item[2]) for item in result]
                    confidence = float(np.mean(scores)) if scores else 0.0
                else:
                    text, confidence = "", 0.0
                method = "ocr"
            record = PageRecord(
                source_id=source["source_id"],
                source_file=str(path),
                page=page_number,
                text=text,
                confidence=confidence,
                method=method,
                text_sha1=hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest(),
            )
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
            if written % 50 == 0:
                print(f"[{source['source_id']}] {page_number}/{len(document)} pages cached", flush=True)
    records = _load_cache(cache_path)
    methods: dict[str, int] = defaultdict(int)
    for record in records.values():
        methods[record.method] += 1
    return {
        "source_id": source["source_id"],
        "source_file": str(path),
        "sha256": _sha256(path),
        "pages": len(document),
        "cache_path": str(cache_path),
        "methods": dict(methods),
        "mean_confidence": round(float(np.mean([r.confidence for r in records.values()])), 4) if records else 0.0,
    }


def _normalize(text: str) -> str:
    return " " + re.sub(r"\s+", " ", text.lower()) + " "


def _topic_scores(text: str) -> dict[str, int]:
    normalized = _normalize(text)
    scores: dict[str, int] = {}
    for topic_id, topic in TOPICS.items():
        score = 0
        for alias in topic["aliases"]:
            phrase = re.sub(r"\s+", " ", alias.lower())
            hits = normalized.count(phrase)
            if hits:
                specificity = 3 if len(phrase.strip()) >= 16 else 2 if len(phrase.strip()) >= 8 else 1
                score += min(hits, 4) * specificity
        if score >= 3:
            scores[topic_id] = score
    return scores


def _page_is_noise(text: str) -> bool:
    compact = _normalize(text)
    if len(compact) < 120:
        return True
    if re.search(r"\b(index|bibliography|references)\b", compact) and len(compact) < 2500:
        return True
    return False


def _runs(pages: list[int], *, max_span: int = 4) -> list[list[int]]:
    if not pages:
        return []
    result: list[list[int]] = []
    current = [pages[0]]
    for page in pages[1:]:
        if page == current[-1] + 1 and len(current) < max_span:
            current.append(page)
        else:
            result.append(current)
            current = [page]
    result.append(current)
    return result


def _quality(source: dict[str, Any], confidence: float, topic_score: int) -> float:
    authority = {
        "primary_research": 0.94,
        "primary_synthesis": 0.87,
        "secondary_methodology": 0.82,
        "secondary_synthesis": 0.80,
        "secondary_practitioner": 0.74,
        "supplementary_practitioner": 0.68,
        "supplementary_research": 0.62,
    }.get(str(source.get("authority_level")), 0.70)
    return round(max(0.0, min(0.98, authority * 0.78 + confidence * 0.14 + min(topic_score, 12) / 12 * 0.08)), 4)


def _display_policy(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("market_scope") == "CN":
        return {
            "backend_eligible": True,
            "frontend_default": "hidden",
            "frontend_condition": "explicit_market_CN_or_internal_research",
            "reason": "China cases are retained for backend research and filtered at the presentation layer.",
        }
    return {"backend_eligible": True, "frontend_default": "visible"}


def compile_source(source: dict[str, Any], scan: dict[str, Any]) -> tuple[list[QuantKnowledgeRecord], list[dict[str, Any]]]:
    path = Path(scan["source_file"])
    records = sorted(_load_cache(Path(scan["cache_path"])).values(), key=lambda item: item.page)
    source_hash = scan["sha256"]
    topic_pages: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    review: list[dict[str, Any]] = []
    for page in records:
        if _page_is_noise(page.text):
            continue
        scores = _topic_scores(page.text)
        if not scores:
            continue
        for topic_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]:
            topic_pages[topic_id].append((page.page, score, page.confidence))

    output: list[QuantKnowledgeRecord] = []
    overview = SOURCE_OVERVIEWS.get(source["source_id"])
    if overview:
        academic, plain, caveat, focus_topics = overview
        quality = _quality(source, float(scan.get("mean_confidence", 0.8)), 8)
        output.append(
            QuantKnowledgeRecord(
                knowledge_id=f"qk_{source['source_id']}_overview",
                source_id=source["source_id"],
                source_title=source["title"],
                source_kind=source["kind"],
                source_file=path.name,
                source_sha256=source_hash,
                pdf_pages=[1],
                topic_ids=focus_topics,
                factor_ids=sorted({factor for topic in focus_topics for factor in TOPICS.get(topic, {}).get("factor_ids", [])}),
                knowledge_type="source_overview",
                academic_summary=academic,
                plain_summary=plain,
                caveat=caveat,
                market=str(source.get("market_scope", "global")),
                method_tags=["document_level_synthesis"],
                quality_score=quality,
                compiler_version=COMPILER_VERSION,
                display_policy=_display_policy(source),
                policy_flags={"no_raw_text": True, "historical_qualifier_required": True, "no_trade_advice": True, "engine_eligible": quality >= 0.65, "review_only": quality < 0.65},
                metadata={"authors": source.get("authors", []), "work_id": source.get("work_id"), "page_count": scan["pages"], "authority_level": source.get("authority_level")},
            )
        )

    for topic_id, values in topic_pages.items():
        topic = TOPICS[topic_id]
        value_by_page = {page: (score, confidence) for page, score, confidence in values}
        candidates = sorted(value_by_page)
        spans = _runs(candidates)
        spans.sort(key=lambda span: max(value_by_page[p][0] for p in span), reverse=True)
        for ordinal, span in enumerate(spans[:10], start=1):
            score = max(value_by_page[p][0] for p in span)
            confidence = float(np.mean([value_by_page[p][1] for p in span]))
            quality = _quality(source, confidence, score)
            record = QuantKnowledgeRecord(
                knowledge_id=f"qk_{source['source_id']}_{topic_id}_{span[0]:04d}_{ordinal:02d}",
                source_id=source["source_id"],
                source_title=source["title"],
                source_kind=source["kind"],
                source_file=path.name,
                source_sha256=source_hash,
                pdf_pages=span,
                topic_ids=[topic_id],
                factor_ids=list(topic.get("factor_ids", [])),
                knowledge_type=str(topic["type"]),
                academic_summary=str(topic["academic"]),
                plain_summary=str(topic["plain"]),
                caveat=str(topic["caveat"]),
                market=str(source.get("market_scope", "global")),
                method_tags=[str(item) for item in topic.get("tags", [])],
                evidence_tags=["source_page_topic_match"],
                risk_tags=[tag for tag in topic.get("tags", []) if tag in {"tail_risk", "liquidity", "costs", "overfitting", "bias"}],
                validation_tags=[tag for tag in topic.get("tags", []) if tag in {"out_of_sample", "multiple_testing", "significance", "backtest"}],
                implementation_tags=[tag for tag in topic.get("tags", []) if tag in {"implementation", "turnover", "capacity", "portfolio_construction", "rebalancing"}],
                quality_score=quality,
                compiler_version=COMPILER_VERSION,
                display_policy=_display_policy(source),
                policy_flags={"no_raw_text": True, "historical_qualifier_required": True, "no_trade_advice": True, "engine_eligible": quality >= 0.65, "review_only": quality < 0.65},
                metadata={"authors": source.get("authors", []), "work_id": source.get("work_id"), "authority_level": source.get("authority_level"), "topic_match_score": score, "source_text_confidence": round(confidence, 4)},
            )
            if quality >= 0.65:
                output.append(record)
            else:
                review.append({"source_id": source["source_id"], "source_title": source["title"], "source_file": path.name, "pdf_pages": span, "topic_id": topic_id, "reason": "quality_below_0.65", "quality_score": quality, "source_text_confidence": round(confidence, 4)})
    return output, review


def _format_pages(pages: Iterable[int]) -> str:
    values = sorted(set(int(page) for page in pages))
    return ", ".join(str(page) for page in values)


def _paper_evidence(source: dict[str, Any], scan: dict[str, Any]) -> list[QuantEvidenceRecord]:
    items = CURATED_EVIDENCE.get(source["source_id"], [])
    result: list[QuantEvidenceRecord] = []
    for index, item in enumerate(items, start=1):
        result.append(
            QuantEvidenceRecord(
                evidence_id=f"qe_{source['source_id']}_{index:03d}",
                evidence_type="paper_reported",
                factor_id=item["factor_id"],
                claim=item["claim"],
                claim_plain=item["plain"],
                caveat=item["caveat"],
                source_id=source["source_id"],
                source_title=source["title"],
                source_kind=source["kind"],
                source_pages=item["pages"],
                source_url=str(source.get("source_url", "")),
                source_sha256=scan["sha256"],
                market=item["market"],
                period=item["period"],
                frequency="paper_defined",
                calculation=item["calculation"],
                data_version="local_pdf_sha256:" + scan["sha256"][:16],
                quality=float(item["quality"]),
                display_policy=_display_policy(source),
                policy_flags={"no_raw_text": True, "historical_qualifier_required": True, "no_trade_advice": True, "paper_reported_not_recomputed": True, "engine_eligible": True},
                metadata={"authors": source.get("authors", []), "work_id": source.get("work_id")},
            )
        )
    return result


def inventory(source_dir: Path, catalog_path: Path) -> tuple[list[tuple[dict[str, Any], Path]], list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))["sources"]
    files_by_name = {path.name: path for path in source_dir.rglob("*.pdf")}
    selected: list[tuple[dict[str, Any], Path]] = []
    missing: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    for source in catalog:
        path = files_by_name.get(source["local_file"])
        if path is None:
            missing.append({"source_id": source["source_id"], "local_file": source["local_file"], "optional_remote": bool(source.get("optional_remote", False))})
            continue
        digest = _sha256(path)
        if digest in seen_hashes:
            duplicates.append({"source_id": source["source_id"], "duplicate_of": seen_hashes[digest], "sha256": digest, "file": path.name})
            continue
        seen_hashes[digest] = source["source_id"]
        selected.append((source, path))
    catalog_names = {item["local_file"] for item in catalog}
    for path in source_dir.rglob("*.pdf"):
        if path.name not in catalog_names:
            digest = _sha256(path)
            if digest in seen_hashes:
                duplicates.append({"source_id": None, "duplicate_of": seen_hashes[digest], "sha256": digest, "file": path.name})
            else:
                missing.append({"source_id": None, "local_file": path.name, "reason": "unregistered_pdf"})
    return selected, missing, duplicates


def write_outputs(
    sources: list[dict[str, Any]],
    scans: list[dict[str, Any]],
    knowledge: list[QuantKnowledgeRecord],
    evidence: list[QuantEvidenceRecord],
    review: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    knowledge_dir: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge = sorted({item.knowledge_id: item for item in knowledge}.values(), key=lambda item: (item.source_id, item.pdf_pages[0], item.knowledge_id))
    with (knowledge_dir / "compiled_quant_knowledge.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in knowledge:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    pending_evidence = _pending_evidence()
    all_evidence = sorted([*evidence, *pending_evidence], key=lambda item: item.evidence_id)
    with evidence_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in all_evidence:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    with (knowledge_dir / "knowledge_review.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in review:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    review_md = [
        "# QTE 人工复核清单",
        "",
        "这里只记录低置信度候选与缺失来源，不包含书籍或论文原文。",
        "",
        "| 来源 | PDF 页码 | 主题 | 原因 | 质量分 |",
        "|---|---:|---|---|---:|",
    ]
    for item in review:
        review_md.append(f"| {item.get('source_title', item.get('source_id', ''))} | {_format_pages(item.get('pdf_pages', []))} | {item.get('topic_id', '')} | {item.get('reason', '')} | {float(item.get('quality_score', 0)):.2f} |")
    if missing:
        review_md.extend(["", "## 缺失或未登记来源", ""])
        for item in missing:
            review_md.append(f"- `{item.get('local_file')}`：{item.get('reason', 'catalog source not found')}（optional={item.get('optional_remote', False)}）")
    (knowledge_dir / "knowledge_review.md").write_text("\n".join(review_md) + "\n", encoding="utf-8", newline="\n")

    scan_by_source = {item["source_id"]: item for item in scans}
    source_index = {
        "schema_version": "2.0",
        "generated": datetime.now().astimezone().isoformat(),
        "compiler_version": COMPILER_VERSION,
        "sources": [
            {
                **source,
                "status": "compiled" if source["source_id"] in scan_by_source else "missing_optional" if source.get("optional_remote") else "missing",
                "local_file": source.get("local_file"),
                "sha256": scan_by_source.get(source["source_id"], {}).get("sha256", ""),
                "pdf_pages": scan_by_source.get(source["source_id"], {}).get("pages", 0),
                "text_methods": scan_by_source.get(source["source_id"], {}).get("methods", {}),
                "mean_text_confidence": scan_by_source.get(source["source_id"], {}).get("mean_confidence", 0),
                "knowledge_records": sum(item.source_id == source["source_id"] for item in knowledge),
                "evidence_records": sum(item.source_id == source["source_id"] for item in evidence),
            }
            for source in sources
        ],
        "duplicates": duplicates,
        "missing": missing,
        "policy": {
            "raw_text_is_internal_build_input_only": True,
            "all_records_require_source_pages": True,
            "quality_below_0_65_is_review_only": True,
            "china_material_backend_allowed": True,
            "china_frontend_visibility_is_display_policy": True,
            "evidence_pending_is_not_empirical_evidence": True,
            "pending_records_must_not_contain_invented_pages_or_results": True,
        },
    }
    (ROOT / "qte_source_index.json").write_text(json.dumps(source_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    by_source: dict[str, int] = defaultdict(int)
    by_topic: dict[str, int] = defaultdict(int)
    for item in knowledge:
        by_source[item.source_id] += 1
        for topic in item.topic_ids:
            by_topic[topic] += 1
    summary = {
        "schema_version": "2.0",
        "generated": datetime.now().astimezone().isoformat(),
        "compiler_version": COMPILER_VERSION,
        "compiled_source_count": len(scans),
        "unique_pdf_count": len(scans),
        "scanned_page_count": sum(int(item["pages"]) for item in scans),
        "knowledge_record_count": len(knowledge),
        "paper_evidence_count": len(evidence),
        "pending_evidence_count": len(pending_evidence),
        "evidence_record_count": len(all_evidence),
        "review_count": len(review),
        "missing_count": len(missing),
        "duplicate_count": len(duplicates),
        "by_source": dict(sorted(by_source.items())),
        "by_topic": dict(sorted(by_topic.items(), key=lambda pair: pair[1], reverse=True)),
        "quality_buckets": {
            "engine_eligible_gte_0_65": sum(item.quality_score >= 0.65 for item in knowledge),
            "review_only_lt_0_65": len(review),
        },
        "artifacts": {
            "knowledge": str(knowledge_dir / "compiled_quant_knowledge.jsonl"),
            "evidence": str(evidence_path),
            "review": str(knowledge_dir / "knowledge_review.md"),
            "source_index": str(ROOT / "qte_source_index.json"),
        },
    }
    (knowledge_dir / "knowledge_library_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile local QTE books and papers into structured knowledge.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--zoom", type=float, default=1.8)
    parser.add_argument("--ocr-threads", type=int, default=2)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    selected, missing, duplicates = inventory(args.source_dir, args.catalog)
    catalog_sources = json.loads(args.catalog.read_text(encoding="utf-8"))["sources"]
    scans: list[dict[str, Any]] = []
    for source, path in selected:
        cache_path = args.cache_dir / _safe_cache_name(source["source_id"], path)
        scan = scan_source(source, path, cache_path, zoom=args.zoom, ocr_threads=args.ocr_threads)
        scans.append(scan)
        print(f"[DONE] {source['source_id']}: {scan['pages']} pages", flush=True)
    if args.scan_only:
        print(json.dumps({"scans": scans, "missing": missing, "duplicates": duplicates}, ensure_ascii=False, indent=2))
        return 0

    source_by_id = {source["source_id"]: source for source in catalog_sources}
    knowledge: list[QuantKnowledgeRecord] = []
    evidence: list[QuantEvidenceRecord] = []
    review: list[dict[str, Any]] = []
    for scan in scans:
        source = source_by_id[scan["source_id"]]
        compiled, rejected = compile_source(source, scan)
        knowledge.extend(compiled)
        review.extend(rejected)
        evidence.extend(_paper_evidence(source, scan))
    summary = write_outputs(catalog_sources, scans, knowledge, evidence, review, missing, duplicates, args.knowledge_dir, args.evidence)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
