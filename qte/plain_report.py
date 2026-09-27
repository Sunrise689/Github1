"""QTE 大白话报告生成器（模板化翻译层，不依赖 LLM）。

定位（对应 ai_integration_protocol.json 第三层·出口翻译）：
- 把 factor_engine 输出的结构化归因结果，翻译成普通用户能直接读懂的句子。
- 术语措辞优先取 translation_lexicon.json 的 plain 字段，保持一致口径。
- 不改变任何数字、正负号、单位，只追加解释。
- LLM 不可用时本模块即兜底出口；LLM 可用时，本模块输出作为 LLM 润色底稿。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QTE_DIR = Path(__file__).resolve().parent
LEXICON_PATH = QTE_DIR / "translation_lexicon.json"

_lexicon_cache: dict[str, dict[str, str]] | None = None


def _lexicon() -> dict[str, dict[str, str]]:
    """按 term 建索引：term -> {plain, usage_note}。"""
    global _lexicon_cache
    if _lexicon_cache is None:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _lexicon_cache = {item["term"]: item for item in data.get("terms", [])}
    return _lexicon_cache


def _term_plain(term: str) -> str | None:
    item = _lexicon().get(term)
    return item.get("plain") if item else None


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "数据不足"
    return f"{x * 100:.{digits}f}%"


def _up_down(x: float | None) -> str:
    if x is None:
        return "数据不足"
    return "涨" if x >= 0 else "跌"


# ---------------- 各因子的区间话术 ----------------

def _beta_text(beta: float | None) -> str:
    if beta is None:
        return "数据不足，算不出跟大盘的联动"
    if beta < 0:
        return "跟大盘反着走：大盘涨它平均跌，大盘跌它平均涨"
    if beta < 0.3:
        return "跟大盘联动很弱，基本走自己的路"
    if beta < 0.7:
        return "跟大盘有一定联动，但不算紧"
    if beta <= 1.3:
        return "跟大盘基本同步：大盘动 1%，它平均也动 1% 左右"
    if beta <= 1.8:
        return "比大盘动得更猛：大盘动 1%，它平均动得更多"
    return "比大盘动得猛得多，大盘的小波动会被它放大"


def _vol_text(vol: float | None) -> str:
    if vol is None:
        return "数据不足"
    if vol < 0.15:
        return "上下起伏比较小"
    if vol < 0.30:
        return "上下起伏中等偏高"
    if vol < 0.50:
        return "上下起伏偏大，颠簸明显"
    return "上下起伏非常大"


def _skew_text(skew: float | None) -> str:
    if skew is None:
        return None
    if skew < -0.5:
        return "极端行情偏向大跌：历史上猛跌的日子比猛涨的日子更突出"
    if skew > 0.5:
        return "极端行情偏向大涨：历史上猛涨的日子比猛跌的日子更突出"
    return "大涨和大跌的极端日子大体对称"


def _kurt_text(kurt: float | None) -> str:
    if kurt is None:
        return None
    if kurt > 3:
        return "出现极端涨跌的概率明显比正常情况频繁"
    if kurt > 1:
        return "偶尔会出现比平常更极端的涨跌"
    return "极端涨跌的出现频率接近正常水平"


def _trend_text(strength: float | None) -> str:
    if strength is None:
        return None
    above = strength * 100
    if strength >= 0.6:
        return f"过去一年约 {above:.0f}% 的时间站在长期平均线上方，多数时候重心在抬高"
    if strength <= 0.4:
        return f"过去一年约 {100 - above:.0f}% 的时间在长期平均线下方，多数时候重心在下移"
    return f"过去一年在长期平均线上下反复，方向感不强（上方时间约 {above:.0f}%）"


def _beta_text_ind(beta: float | None) -> str:
    if beta is None:
        return ""
    if beta < 0.5:
        return "它跟行业联动不算紧，更多走自己的节奏。"
    if beta <= 1.3:
        return "它跟行业基本同步：行业动 1%，它平均也动 1% 左右。"
    return "它比行业动得更猛，行业的波动会被它放大。"


def _momentum_text(mom: float | None, label: str = "过去一年") -> str | None:
    if mom is None:
        return None
    direction = "涨" if mom >= 0 else "跌"
    strength = "惯性不错" if mom >= 0.2 else "惯性温和" if mom >= 0 else "惯性偏弱" if mom >= -0.2 else "惯性很弱"
    return f"{label}{_up_down(mom)}了 {_pct(abs(mom))}，{strength}"


# ---------------- 主要因子简单总结（只解释贡献最大的几个） ----------------

def _major_factors(contributions: list, max_items: int = 3, cum_share: float = 0.5) -> list:
    """选主要因子：按 |贡献| 从大到小，累计解释份额达到 50% 或取满 max_items 即停。"""
    picked: list = []
    accum = 0.0
    for item in contributions:
        picked.append(item)
        accum += item.get("share") or 0.0
        if accum >= cum_share or len(picked) >= max_items:
            break
    return picked


def _factor_mechanism(fid: str, item: dict, attribution: dict) -> str | None:
    """单个因子的机制解释：说清它是什么、为什么把价格推向这一边。

    只用结构化结果里已有的数值（暴露值、贡献、行业信息），不改变数字。
    """
    contrib = item.get("contribution") or 0.0
    direction = "往上抬" if contrib >= 0 else "往下拽"
    amount = f"约 {_pct(abs(contrib))}"
    exposures = attribution.get("exposures", {})
    industry = attribution.get("industry", {}) or {}
    style_sources = attribution.get("style_sources", {}) or {}

    benchmark = str(attribution.get("benchmark") or "").lower()
    bench_names = {
        "sh000001": "上证指数", "sh000300": "上证指数", "sh000905": "上证指数",
        "us^gspc": "标普500", "usspy": "标普500", "us^ixic": "纳斯达克指数", "us^dji": "道琼斯指数",
        "hkhsi": "恒生指数",
    }
    market_name = bench_names.get(benchmark, "大盘")

    if fid == "market_beta":
        beta = exposures.get("market_beta")
        if beta is not None and abs(beta) > 0.05:
            bench_win = contrib / beta
            return (f"大盘带动：归因区间内{market_name}约{_up_down(bench_win)}了 {_pct(abs(bench_win))}，"
                    f"这只股票与{market_name}的联动系数约 {beta:.2f}（{market_name}每动 1%，它平均动 {abs(beta):.2f}%），"
                    f"折算下来{market_name}把它{direction}了 {amount}。")
        return None
    if fid == "industry_beta":
        ind_name = industry.get("name", "所在行业")
        ind_beta = industry.get("industry_beta")
        if ind_beta is not None and abs(ind_beta) > 0.05:
            ind_win = contrib / ind_beta
            return (f"行业带动：归因区间内「{ind_name}」行业基准约{_up_down(ind_win)}了 {_pct(abs(ind_win))}，"
                    f"这只股票对行业的联动系数约 {ind_beta:.2f}，"
                    f"行业整体走势把它{direction}了 {amount}。")
        return f"行业带动：跟着「{ind_name}」行业一起动的部分，{direction}了 {amount}。"
    if fid == "momentum_12_1":
        mom = exposures.get("momentum_12_1")
        if mom is None:
            return None
        same = (mom >= 0) == (contrib >= 0)
        why = (f"这份惯性在归因区间内仍在同向延续，因此{direction}了 {amount}"
               if same else
               f"但归因区间内这股惯性已明显减弱，对价格的影响转为 {amount}的{('拖累' if contrib < 0 else '抬升')}")
        return (f"动量惯性：它过去一年{_up_down(mom)}了 {_pct(abs(mom))}。"
                f"动量因子的含义是“过去的趋势在统计上倾向于延续”，{why}。")
    if fid == "residual_momentum":
        rm = exposures.get("residual_momentum")
        if rm is None:
            return None
        return (f"自身动量：剔除大盘和行业影响后，它自己过去一年{_up_down(rm)}了 {_pct(abs(rm))}，"
                f"属于自身走势惯性的那部分，{direction}了 {amount}。")
    if fid == "short_term_reversal":
        rev = exposures.get("short_term_reversal")
        if rev is None:
            return None
        return (f"短期反转：最近一个月它{_up_down(rev)}了 {_pct(abs(rev))}，"
                f"短期“涨多了回调、跌多了反弹”的效应计入 {amount}。")
    if fid == "volatility_realized":
        vol = exposures.get("volatility_realized")
        if vol is None:
            return None
        return (f"波动影响：年化波动率约 {_pct(vol)}，{_vol_text(vol)}。"
                f"波动大意味着每日涨跌摆幅大，统计上高波动股票的长期平均回报往往偏低（波动损耗）；"
                f"本项是对这一风险特征在区间收益中的统计记账，计入 {amount}（{_up_down(contrib)}向）。")
    if fid == "trend_strength":
        ts = exposures.get("trend_strength")
        t = _trend_text(ts)
        if not t:
            return None
        return (f"趋势强度：{t}；趋势特征会影响资金行为的延续性，"
                f"本项为统计记账，计入 {amount}（{_up_down(contrib)}向）。")
    if fid.startswith("style_"):
        skey = fid[6:]
        style_name = style_sources.get(skey)
        style_desc = {
            "value": "低估值股票", "growth": "高成长股票", "dividend": "高分红股票",
            "momentum": "近期强势股票", "quality": "高盈利质量股票", "lowvol": "低波动股票",
        }.get(skey, "该类股票")
        base = f"风格带动：衡量的是归因区间内市场上{style_desc}整体的强弱"
        if style_name:
            base += f"（基准：{style_name}）"
        return base + f"，不是公司自身利好利空；回归结果计入 {amount}。"
    return None


def _factor_summary(attribution: dict) -> list[str]:
    """主要因子简单总结：只解释贡献最大的几个（前三大或累计解释 50%），

    每个因子说清它是什么、为什么把价格推向这一边。其余因子不在这里逐条展开，
    全部因子的数值展示由因子画像热力图承担。
    """
    contributions = attribution.get("contributions", [])
    labels = {
        "market_beta": "大盘带动", "industry_beta": "行业带动",
        "momentum_12_1": "动量惯性", "short_term_reversal": "短期反转",
        "residual_momentum": "自身动量", "volatility_realized": "波动影响",
        "trend_strength": "趋势强度",
        "style_value": "价值风格", "style_growth": "成长风格",
        "style_dividend": "红利风格", "style_momentum": "动量风格",
        "style_quality": "质量风格", "style_lowvol": "低波风格",
    }
    lines = []
    for item in _major_factors(contributions):
        fid = item.get("factor_id")
        text = _factor_mechanism(fid, item, attribution)
        if text:
            lines.append(text)
        else:
            contrib = item.get("contribution") or 0.0
            direction = "往上抬" if contrib >= 0 else "往下拽"
            lines.append(f"{labels.get(fid, fid)}：把价格{direction}了约 {_pct(abs(contrib))}。")
    return lines


# ---------------- 主入口 ----------------

def build_plain_report(attribution: dict[str, Any]) -> dict[str, Any]:
    """把 factor_engine.run_attribution 的结构化结果翻译成分节大白话。"""
    period_return = attribution.get("period_return")
    contributions = attribution.get("contributions", [])
    residual = attribution.get("residual", {})
    exposures = attribution.get("exposures", {})
    industry = attribution.get("industry", {})
    fundamentals = attribution.get("fundamentals", {})
    window = attribution.get("window", {})
    days = window.get("attribution_days", 0)
    months = max(1, round(days / 21))

    lines_summary: list[str] = []
    lines_split: list[str] = []
    lines_industry: list[str] = []
    lines_fundamental: list[str] = []
    lines_lookback: list[str] = []
    lines_risk: list[str] = []
    lines_gap: list[str] = []

    # 一句话总结
    direction = _up_down(period_return)
    name = (fundamentals or {}).get("name") or attribution.get("symbol")
    lines_summary.append(
        f"{name} 最近约 {months} 个月{direction}了 {_pct(abs(period_return))}。下面把它拆开说清楚。"
    )

    # 拆账：只逐条列出主要因子（前三大或累计解释 50%），其余不逐条展开
    benchmark = str(attribution.get("benchmark") or "").lower()
    _bench_names = {
        "sh000001": "上证指数", "sh000300": "上证指数", "sh000905": "上证指数",
        "us^gspc": "标普500", "usspy": "标普500", "us^ixic": "纳斯达克指数", "us^dji": "道琼斯指数",
        "hkhsi": "恒生指数",
    }
    market_name = _bench_names.get(benchmark, "大盘")
    factor_labels = {
        "market_beta": market_name,
        "industry_beta": "行业",
        "momentum_12_1": "动量惯性",
        "short_term_reversal": "短期反转",
        "residual_momentum": "自身动量",
        "volatility_realized": "波动影响",
        "trend_strength": "趋势强度",
        "style_value": "价值风格", "style_growth": "成长风格",
        "style_dividend": "红利风格", "style_momentum": "动量风格",
        "style_quality": "质量风格", "style_lowvol": "低波风格",
    }
    for item in _major_factors(contributions):
        fid = item.get("factor_id")
        label = factor_labels.get(fid, fid)
        contrib = item.get("contribution")
        share = item.get("share")
        if contrib is None:
            continue
        c_dir = _up_down(contrib)
        if fid == "industry_beta":
            ind_name = industry.get("name", "所在行业")
            lines_split.append(f"主要因子：{ind_name}行业带动，约 {_pct(abs(contrib))}（{c_dir}向）。")
        elif fid.startswith("style_"):
            style_desc = {
                "style_value": "低估值股票整体的强弱",
                "style_growth": "高成长股票整体的强弱",
                "style_dividend": "高分红股票整体的强弱",
                "style_momentum": "市场'强者恒强'效应",
                "style_quality": "高质量（会赚钱）股票整体的强弱",
                "style_lowvol": "市场对'稳的股票'的偏好",
            }.get(fid, "风格因素")
            lines_split.append(f"主要因子：{label}带动（{style_desc}），约 {_pct(abs(contrib))}（{c_dir}向）。")
        else:
            lines_split.append(f"主要因子：{label}带动，约 {_pct(abs(contrib))}（{c_dir}向）。")

    # 残差如实说
    residual_share = residual.get("share")
    if residual_share is not None:
        if residual_share >= 0.4:
            lines_split.append(
                f"其余约 {_pct(residual_share, 0)} 用已知因子解释不了，更可能是公司自身的原因"
                "（消息、基本面变化等），引擎如实标为“解释不了”，不硬编理由。"
            )
        else:
            lines_split.append(f"其余约 {_pct(residual_share, 0)} 是已知因子暂时解释不了的部分。")

    # 行业背景
    if industry.get("available"):
        ind_beta = industry.get("industry_beta")
        ind_mom = industry.get("industry_momentum_12")
        ind_rs = industry.get("industry_relative_strength")
        ind_name = industry.get("name", "行业")
        bench_src = industry.get("benchmark_source")
        if bench_src and str(bench_src).startswith("恒生指数("):
            src_note = f"（「{ind_name}」暂无专属行业指数，如实以「恒生指数」作基准）"
        elif bench_src:
            src_note = f"（以「{bench_src}」为基准，官方口径）"
        else:
            src_note = ""
        lines_industry.append(f"它属于「{ind_name}」。{_beta_text_ind(ind_beta)}{src_note}")
        if ind_mom is not None:
            lines_industry.append(
                f"这个行业过去一年{_up_down(ind_mom)}了 {_pct(abs(ind_mom))}。"
                + (f"同期比大盘少{_pct(abs(ind_rs))}，行业整体偏弱。" if ind_rs is not None and ind_rs < 0
                   else (f"同期比大盘多{_pct(abs(ind_rs))}，行业整体偏强。" if ind_rs is not None and ind_rs > 0 else ""))
            )
    else:
        note = industry.get("note")
        lines_gap.append("行业层面的拆解暂时算不了" + (f"（{note}）。" if note else "。"))

    # 基本面快照
    if fundamentals.get("available"):
        pe = fundamentals.get("pe_dynamic")
        size_bucket = fundamentals.get("size_bucket")
        cap = fundamentals.get("total_market_cap")
        prefix = str(attribution.get("symbol") or "")[:2].lower()
        cap_unit = {"hk": "亿港元", "us": "亿美元"}.get(prefix, "亿")
        if size_bucket:
            cap_note = f"约 {cap / 1e8:.0f} {cap_unit}" if cap else ""
            lines_fundamental.append(f"体量：{size_bucket}（{cap_note}）。")
        if pe is not None and pe > 0:
            pe_note = "估值偏低" if pe < 15 else "估值中等" if pe < 30 else "估值偏高"
            lines_fundamental.append(
                f"当前市盈率（动态）约 {pe:.1f}，{pe_note}。注：这是当前快照，历史高低位置需要历史估值序列才能判断，暂不编造。"
            )
        fin = fundamentals.get("financials") or {}
        if fin:
            period = fin.get("stat_date", "")
            pub = fin.get("pub_date", "")
            lines_fundamental.append(f"财报面（截至 {period}，{pub} 披露）：")
            roe = fin.get("quality_roe")
            if roe is not None:
                roe_note = "赚钱效率很高" if roe > 0.08 else "赚钱效率不错" if roe > 0.03 else "赚钱效率一般" if roe > 0 else "这个季度在亏钱"
                lines_fundamental.append(f"· 赚钱效率：单季度净资产收益率约 {_pct(roe)}，{roe_note}。")
            gm = fin.get("quality_gross_margin")
            if gm is not None:
                gm_note = "产品很赚钱" if gm > 0.6 else "产品毛利不错" if gm > 0.3 else "毛利偏薄" if gm > 0.1 else "毛利很薄"
                lines_fundamental.append(f"· 生意质地：毛利率约 {_pct(gm)}，{gm_note}。")
            ge = fin.get("growth_earnings")
            if ge is not None:
                ge_dir = "增长" if ge >= 0 else "下滑"
                lines_fundamental.append(f"· 成长速度：净利润同比{ge_dir} {_pct(abs(ge))}。")
            lev = fin.get("leverage")
            if lev is not None:
                lev_note = "几乎不靠借钱经营" if lev < 0.3 else "负债适中" if lev < 0.6 else "负债偏高"
                lines_fundamental.append(f"· 家底风险：资产负债率约 {_pct(lev)}，{lev_note}。")
            lines_fundamental.append("注：以上为最近已披露季报数据，只描述过去，不预示未来。")

    # 回头看：动量类
    mom12 = exposures.get("momentum_12_1")
    mom_text = _momentum_text(mom12)
    if mom_text:
        lines_lookback.append(mom_text + "。")
    res_mom = exposures.get("residual_momentum")
    if res_mom is not None and mom12 is not None:
        if res_mom < mom12 - 0.05:
            lines_lookback.append(
                f"而且扣掉大盘影响后它更{_up_down(res_mom)}（约 {_pct(abs(res_mom))}）：过去一年大盘其实是"
                f"{_up_down(-res_mom + mom12) if mom12 - res_mom >= 0 else '跌'}的，它这部分{direction}不完全是被大盘拖累。"
            )
        elif res_mom > mom12 + 0.05:
            lines_lookback.append(
                f"扣掉大盘影响后它的真实惯性比表面更好（约 {_pct(abs(res_mom))}{_up_down(res_mom)}），有一部分是大盘贡献的。"
            )
    reversal = exposures.get("short_term_reversal")
    if reversal is not None:
        lines_lookback.append(f"最近一个月它{_up_down(reversal)}了 {_pct(abs(reversal))}。")

    # 风险侧
    beta = exposures.get("market_beta")
    vol = exposures.get("volatility_realized")
    if vol is not None:
        lines_risk.append(f"颠簸程度：年化波动约 {_pct(vol)}，{_vol_text(vol)}。")
    dd = exposures.get("maximum_drawdown")
    if dd is not None:
        lines_risk.append(f"最难受的时候：从高点到低点{_up_down(dd)}过 {_pct(abs(dd))}。")
    skew_line = _skew_text(exposures.get("skewness"))
    if skew_line:
        lines_risk.append(skew_line + "。")
    kurt_line = _kurt_text(exposures.get("kurtosis"))
    if kurt_line:
        lines_risk.append(kurt_line + "。")
    trend_line = _trend_text(exposures.get("trend_strength"))
    if trend_line:
        lines_risk.append(trend_line + "。")

    # 能力缺口如实说（行业缺口已在行业分节处理，这里只说其余）
    for key in attribution.get("pending", {}):
        if key == "quality_roe" or key == "growth_revenue":
            lines_gap.append("财报类因子（盈利能力、成长速度）暂时算不了，等财报数据源接入后补上。")
            break

    return {
        "summary": lines_summary,
        "factor_summary": _factor_summary(attribution),
        "attribution": lines_split,
        "industry": lines_industry,
        "fundamental": lines_fundamental,
        "lookback": lines_lookback,
        "risk": lines_risk,
        "gap": lines_gap,
        "disclaimer": attribution.get("qualifier", "以上为历史统计描述，不预示未来表现，不构成投资建议。"),
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(QTE_DIR))
    from factor_engine import run_attribution

    symbol = sys.argv[1] if len(sys.argv) > 1 else "sh518880"
    raw = run_attribution(symbol)
    report = build_plain_report(raw)
    for section, lines in report.items():
        if isinstance(lines, list) and lines:
            print(f"【{section}】")
            for line in lines:
                print("  " + line)
        elif isinstance(lines, str):
            print(lines)
