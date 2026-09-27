"""QTE 因子热力图报告（无 LLM 依赖，确定性输出）。

产品形态（产品方定稿）：
- 一张统一热力图：因子按 基本面/技术面/市场面 分组排成网格，颜色=该维度强弱。
- 每个因子格只放 名称+数值；点击解释由前端查 QTE 知识库（factor_registry +
  translation_lexicon + BM25 证据），不在图片里生成长文本。
- 图底部为"深入"统计区（R²、偏度、峰度、特质波动等统计语言），小程序里默认折叠，
  图片中以小号字呈现示意。
- 颜色语义：红=该维度上偏强/偏好，绿=偏弱/偏差（仅描述历史状态，不构成好坏建议）。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

QTE_DIR = Path(__file__).resolve().parent
REPORT_DIR = QTE_DIR / "reports"

# 字体：抖音美好体优先（需把 DouyinSansBold.ttf 放到 qte/fonts/ 目录），无则回退 Noto Sans CJK
from matplotlib import font_manager as _fm
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_DOUYIN_FONT = None
if _FONT_DIR.exists():
    for _fn in _FONT_DIR.iterdir():
        if _fn.suffix.lower() in (".ttf", ".otf", ".ttc"):
            try:
                _fm.fontManager.addfont(str(_fn))
                _DOUYIN_FONT = _fm.FontProperties(fname=str(_fn)).get_name()
                break
            except Exception:
                pass
if _DOUYIN_FONT:
    plt.rcParams["font.sans-serif"] = [_DOUYIN_FONT, "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 色阶：浅色柔和风格（低饱和度，避免刺眼）
def _heat_color(score: float | None) -> str:
    """score ∈ [0,1] → 浅色；None 给浅灰（数据缺失）。"""
    if score is None:
        return "#f5f5f7"
    s = max(0.0, min(1.0, score))
    # 浅色渐变：从浅蓝到浅橙红，饱和度低
    if s < 0.5:
        t = s / 0.5
        r = int(230 + t * (240 - 230))
        g = int(240 + t * (245 - 240))
        b = int(250 + t * (235 - 250))
    else:
        t = (s - 0.5) / 0.5
        r = int(240 + t * (255 - 240))
        g = int(245 + t * (225 - 245))
        b = int(235 + t * (210 - 235))
    return f"#{r:02x}{g:02x}{b:02x}"


def _clamp(x: float | None, lo: float, hi: float) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(lo, min(hi, x))


def build_heatmap_cells(attribution: dict) -> list[dict]:
    """把归因结果整理为热力图格子（前端渲染同一份数据）。

    每格：factor_id / group / label / display / score(0-1) / note
    score 用于着色；因子口径与 factor_registry 对齐。
    """
    exposures = attribution.get("exposures", {})
    fundamentals = attribution.get("fundamentals") or {}
    financials = fundamentals.get("financials") or {}
    industry = attribution.get("industry", {})
    cells: list[dict] = []

    def inv(x):  # 越低越好的维度反转
        return None if x is None else 1.0 - x

    # ---- 市场面 ----
    beta = exposures.get("market_beta")
    if beta is not None:
        if beta < 0:
            beta_note = "跟大盘反着走"
        elif beta < 0.3:
            beta_note = "基本不跟大盘走，独立性很强"
        elif beta < 0.7:
            beta_note = "跟大盘有一定联动，但不紧"
        elif beta <= 1.3:
            beta_note = "跟大盘基本同步，同涨同跌"
        else:
            beta_note = "比大盘动得更凶，波动被放大"
        cells.append({"factor_id": "market_beta", "group": "市场面", "label": "市场联动",
                      "display": f"{beta:.2f}", "score": _clamp(beta, 0, 2) / 2,
                      "note": beta_note})
    if industry.get("available"):
        ib = industry.get("industry_beta")
        if ib is not None:
            if abs(ib) > 2:
                ib_note = "与行业的联动关系不稳定，仅供参考"
            elif ib < 0.5:
                ib_note = "更多走自己的节奏，不太跟行业"
            elif ib <= 1.3:
                ib_note = "跟行业基本同步，行业强它则强"
            else:
                ib_note = "比行业动得更凶，行业波动被放大"
            cells.append({"factor_id": "industry_beta", "group": "市场面", "label": "行业联动",
                          "display": f"{ib:.2f}", "score": _clamp(abs(ib), 0, 2) / 2,
                          "note": ib_note})
        im = industry.get("industry_momentum_12")
        if im is not None:
            im_note = "所在行业近一年整体偏强" if im > 0.05 else "所在行业近一年整体偏弱" if im < -0.05 else "所在行业近一年方向不明显"
            cells.append({"factor_id": "industry_momentum", "group": "市场面", "label": "行业动量",
                          "display": f"{im * 100:+.0f}%", "score": _clamp((im + 0.5) / 1.0, 0, 1),
                          "note": im_note})
    # ---- 技术面 ----
    mom = exposures.get("momentum_12_1")
    if mom is not None:
        mom_note = "过去一年惯性很强，趋势延续" if mom > 0.2 else "过去一年惯性温和" if mom > 0 else "过去一年惯性偏弱，重心下移" if mom > -0.2 else "过去一年惯性很弱，跌得多"
        cells.append({"factor_id": "momentum_12_1", "group": "技术面", "label": "动量惯性",
                      "display": f"{mom * 100:+.0f}%", "score": _clamp((mom + 0.5) / 1.0, 0, 1),
                      "note": mom_note})
    rev = exposures.get("short_term_reversal")
    if rev is not None:
        rev_note = "最近一个月在反弹" if rev > 0.02 else "最近一个月在小跌" if rev < -0.02 else "最近一个月基本横盘"
        cells.append({"factor_id": "short_term_reversal", "group": "技术面", "label": "最近1月",
                      "display": f"{rev * 100:+.1f}%", "score": _clamp((rev + 0.2) / 0.4, 0, 1),
                      "note": rev_note})
    vol = exposures.get("volatility_realized")
    if vol is not None:
        vol_note = "上下起伏很小，相对稳" if vol < 0.15 else "上下起伏中等" if vol < 0.3 else "上下起伏偏大，颠簸明显"
        cells.append({"factor_id": "volatility_realized", "group": "技术面", "label": "颠簸程度",
                      "display": f"{vol * 100:.0f}%", "score": inv(_clamp(vol / 0.6, 0, 1)),
                      "note": vol_note})
    dd = exposures.get("maximum_drawdown")
    if dd is not None:
        dd_note = "历史上最痛的时候，从高点跌到这个幅度"
        cells.append({"factor_id": "maximum_drawdown", "group": "技术面", "label": "最痛回撤",
                      "display": f"{dd * 100:.0f}%", "score": inv(_clamp(abs(dd), 0, 1)),
                      "note": dd_note})
    trend = exposures.get("trend_strength")
    if trend is not None:
        tr_note = "多数时间重心在抬高" if trend > 0.6 else "多数时间重心在下移" if trend < 0.4 else "方向感不强，上下反复"
        cells.append({"factor_id": "trend_strength", "group": "技术面", "label": "趋势强度",
                      "display": f"{trend * 100:.0f}%", "score": _clamp(trend, 0, 1),
                      "note": tr_note})
    # ---- 基本面 ----
    pe = fundamentals.get("pe_dynamic")
    if pe is not None:
        pe_note = "当前估值相对便宜" if pe < 15 else "当前估值中等" if pe < 30 else "当前估值偏贵"
        cells.append({"factor_id": "value_pe", "group": "基本面", "label": "估值水平",
                      "display": f"{pe:.1f}", "score": inv(_clamp(pe / 60, 0, 1)),
                      "note": pe_note})
    cap = fundamentals.get("total_market_cap")
    if cap:
        cap_yi = cap / 1e8
        cap_note = "体量很大，属于超大盘" if cap_yi >= 1000 else "体量较大" if cap_yi >= 300 else "体量中等" if cap_yi >= 100 else "体量偏小"
        cells.append({"factor_id": "size", "group": "基本面", "label": "市值体量",
                      "display": f"{cap_yi:.0f}亿", "score": _clamp(math.log10(max(cap, 1)) - 8, 0, 4) / 4,
                      "note": cap_note})
    roe = financials.get("quality_roe")
    if roe is not None:
        roe_note = "单季度赚钱效率很高" if roe > 0.08 else "单季度赚钱效率不错" if roe > 0.03 else "单季度赚钱效率一般" if roe > 0 else "这个季度在亏钱"
        cells.append({"factor_id": "quality_roe", "group": "基本面", "label": "赚钱效率",
                      "display": f"{roe * 100:.1f}%", "score": _clamp(roe / 0.10, 0, 1),
                      "note": roe_note})
    gm = financials.get("quality_gross_margin")
    if gm is not None:
        gm_note = "产品很赚钱，生意质地好" if gm > 0.6 else "毛利不错" if gm > 0.3 else "毛利偏薄" if gm > 0.1 else "毛利很薄"
        cells.append({"factor_id": "quality_gross_margin", "group": "基本面", "label": "生意质地",
                      "display": f"{gm * 100:.0f}%", "score": _clamp(gm, 0, 1),
                      "note": gm_note})
    ge = financials.get("growth_earnings")
    if ge is not None:
        ge_note = "赚钱速度在加快" if ge > 0.1 else "赚钱速度在放慢" if ge < -0.1 else "赚钱速度变化不大"
        cells.append({"factor_id": "growth_earnings", "group": "基本面", "label": "成长速度",
                      "display": f"{ge * 100:+.0f}%", "score": _clamp((ge + 0.5) / 1.0, 0, 1),
                      "note": ge_note})
    lev = financials.get("leverage")
    if lev is not None:
        lev_note = "几乎不靠借钱经营" if lev < 0.3 else "负债适中" if lev < 0.6 else "负债偏高，靠借钱经营的比例大"
        cells.append({"factor_id": "leverage", "group": "基本面", "label": "家底风险",
                      "display": f"{lev * 100:.0f}%", "score": inv(_clamp(lev, 0, 1)),
                      "note": lev_note})
    # 新增基本面因子
    pb = fundamentals.get("pb")
    if pb is not None:
        pb_note = "市净率相对便宜" if pb < 1.5 else "市净率中等" if pb < 3 else "市净率偏高"
        cells.append({"factor_id": "value_pb", "group": "基本面", "label": "市净率",
                      "display": f"{pb:.2f}", "score": inv(_clamp(pb / 5, 0, 1)),
                      "note": pb_note})
    rev_g = fundamentals.get("revenue_growth")
    if rev_g is not None:
        rg_note = "营收在快速增长" if rev_g > 15 else "营收增长放缓" if rev_g < 0 else "营收稳步增长"
        cells.append({"factor_id": "growth_revenue", "group": "基本面", "label": "营收增速",
                      "display": f"{rev_g:+.1f}%", "score": _clamp((rev_g + 30) / 60, 0, 1),
                      "note": rg_note})
    div_y = fundamentals.get("dividend_yield")
    if div_y is not None:
        dy_note = "分红慷慨" if div_y > 4 else "分红适中" if div_y > 2 else "分红较少"
        cells.append({"factor_id": "dividend_yield", "group": "基本面", "label": "股息率",
                      "display": f"{div_y:.2f}%", "score": _clamp(div_y / 5, 0, 1),
                      "note": dy_note})
    # 新增技术面因子
    m61 = exposures.get("momentum_6_1")
    if m61 is not None:
        m61_note = "近半年动量偏强" if m61 > 0.05 else "近半年动量偏弱" if m61 < -0.05 else "近半年方向不明"
        cells.append({"factor_id": "momentum_6_1", "group": "技术面", "label": "动量(6-1月)",
                      "display": f"{m61 * 100:+.0f}%", "score": _clamp((m61 + 0.5) / 1.0, 0, 1),
                      "note": m61_note})
    vpd = exposures.get("volume_price_divergence")
    if vpd is not None:
        vpd_note = "量价配合良好" if vpd > 0.3 else "量价背离，需警惕" if vpd < -0.3 else "量价关系一般"
        cells.append({"factor_id": "volume_price_divergence", "group": "技术面", "label": "量价背离",
                      "display": f"{vpd:.2f}", "score": _clamp((vpd + 1) / 2, 0, 1),
                      "note": vpd_note})
    va = exposures.get("volume_activity")
    if va is not None:
        va_note = "近期交投活跃" if va > 1.5 else "交投清淡" if va < 0.7 else "交投平稳"
        cells.append({"factor_id": "volume_activity", "group": "技术面", "label": "量能活跃度",
                      "display": f"{va:.2f}", "score": _clamp(va / 2, 0, 1),
                      "note": va_note})
    # 特质波动率
    iv = exposures.get("idiosyncratic_volatility")
    if iv is not None:
        iv_note = "个股独立波动较小" if iv < 0.2 else "个股独立波动较大"
        cells.append({"factor_id": "idiosyncratic_volatility", "group": "技术面", "label": "特质波动",
                      "display": f"{iv * 100:.1f}%", "score": inv(_clamp(iv / 0.6, 0, 1)),
                      "note": iv_note})
    return cells


def render_heatmap_report(attribution: dict, out_path: str | Path | None = None) -> Path:
    """渲染统一热力图报告图片。"""
    fundamentals = attribution.get("fundamentals") or {}
    name = str(fundamentals.get("name") or attribution.get("symbol", "")).replace(" ", "")
    window = attribution.get("window", {})
    period_return = attribution.get("period_return") or 0.0
    cells = build_heatmap_cells(attribution)
    groups = ["市场面", "技术面", "基本面"]
    by_group = {g: [c for c in cells if c["group"] == g] for g in groups}

    n_cols = 4
    cell_h = 1.42
    group_gap = 0.42
    total_rows = sum(math.ceil(len(by_group[g]) / n_cols) for g in groups if by_group[g])
    fig_h = 2.0 + total_rows * (cell_h + 0.12) + len([g for g in groups if by_group[g]]) * group_gap + 1.2
    fig, ax = plt.subplots(figsize=(9.6, fig_h), dpi=110)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f8fb")

    # 标题
    direction = "涨" if period_return >= 0 else "跌"
    dir_color = "#d64545" if period_return >= 0 else "#0a8a52"
    ax.text(0.02, 9.88, f"{name} · 因子画像", fontsize=18, fontweight="bold", va="top", color="#1f2330")
    ax.text(n_cols - 0.02, 9.88,
            f"近{window.get('attribution_days', 0)}日 {direction} {abs(period_return) * 100:.1f}%",
            fontsize=12, ha="right", va="top", color=dir_color, fontweight="bold")

    y = 9.42
    cell_w = (n_cols - 0.36) / n_cols
    for g in groups:
        items = by_group[g]
        if not items:
            continue
        ax.text(0.03, y, g, fontsize=13, fontweight="bold", color="#3d4457", va="top")
        y -= group_gap
        for idx, cell in enumerate(items):
            col = idx % n_cols
            if idx > 0 and col == 0:
                y -= cell_h + 0.12
            x = 0.18 + col * cell_w
            box = FancyBboxPatch((x, y - cell_h + 0.05), cell_w - 0.10, cell_h - 0.08,
                                 boxstyle="round,pad=0.02", linewidth=0.8,
                                 edgecolor="#e3e7f0", facecolor=_heat_color(cell["score"]))
            ax.add_patch(box)
            cx = x + (cell_w - 0.10) / 2
            ax.text(cx, y - 0.22, cell["label"], ha="center", va="center",
                    fontsize=10.5, color="#3d4457", fontweight="bold")
            ax.text(cx, y - 0.62, cell["display"], ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#1f2330")
            note = cell.get("note", "")
            if note:
                if len(note) > 13:
                    note = note[:12] + "…"
                ax.text(cx, y - cell_h + 0.28, note, ha="center", va="center",
                        fontsize=7.8, color="#6e717b")
        y -= cell_h + 0.12 + 0.08

    # 图例（浅色渐变条）
    lg_y = y + 0.05
    ax.text(0.18, lg_y, "颜色深浅 = 该维度的强弱程度（仅描述历史状态，不构成好坏判断）",
            fontsize=8.5, color="#8a90a3", va="top")
    for i in range(16):
        ax.add_patch(plt.Rectangle((3.6 + i * 0.085, lg_y - 0.13), 0.085, 0.13,
                                   facecolor=_heat_color(i / 15), edgecolor="none"))

    # 页脚
    ax.text(0.18, 0.34, "统计归因 · 用所有人都懂的语言，解释涨跌的原因", fontsize=10,
            fontweight="bold", color="#3d4457", va="bottom")
    ax.text(0.18, 0.14, "点击任一因子可查看通俗解释。以上为历史统计描述，不预示未来表现，不构成投资建议。",
            fontsize=8.5, color="#b02a25", va="bottom")

    if out_path is None:
        out_path = REPORT_DIR / f"{attribution.get('symbol', 'asset')}_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(out_path)


if __name__ == "__main__":
    sys.path.insert(0, str(QTE_DIR))
    from factor_engine import run_attribution

    symbol = sys.argv[1] if len(sys.argv) > 1 else "sz000858"
    raw = run_attribution(symbol)
    print(json.dumps(build_heatmap_cells(raw), ensure_ascii=False, indent=1)[:200])
    img = render_heatmap_report(raw)
    print("热力图报告:", img)
