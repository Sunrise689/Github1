"""QTE 因子计算引擎（骨架 v0.1）。

职责：把资产在指定区间内的涨跌"拆账"——用多元回归把收益分解到各因子贡献，
按贡献度排序输出主导因子，并如实报告残差。

设计原则（对应 qte/ai_integration_protocol.json 三层分离）：
- 本模块只做确定性数值计算，不调用 LLM，不生成买卖建议。
- 方法论严格遵循 qte/attribution_framework.json（R = alpha + Σ beta_i × F_i + epsilon）。
- 因子口径以 qte/factor_registry.json 为准；data_availability.current=false 的因子不计算。
- 输出一律为历史统计描述，附"不预示未来"限定。

当前实现范围：
- P0（行情数据）：市场 Beta、动量（12-1）、短期反转、残差动量、实现波动率、
  特质波动率、最大回撤、偏度、峰度、趋势强度。
- P1（东财免费接口）：行业 Beta（行业名→板块指数 K 线）、行业动量、行业相对强弱、
  规模（总市值）、估值快照（PE）。
ROE、增速等财报因子与 PE 历史百分位待财报数据源接入后扩展。
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QTE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QTE_DIR.parents[0]
BACKEND_DIR = PROJECT_ROOT / "wechat-miniprogram" / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(BACKEND_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from market_data import fetch_daily  # noqa: E402

# ---------------- 常量与口径登记 ----------------

ANNUALIZE = math.sqrt(252)
MIN_OBS = 120  # 回归最少交易日数，低于此不出归因结论

# 按资产市场选择基准：A股/港股 → 沪深300；美股 → 标普500
MARKET_BENCHMARK = {
    "sh": "sh000300",
    "sz": "sh000300",
    "bj": "sh000300",
    "hk": "hkhsi",
    "us": "usspy",  # SPY 走股票数据通道，指数通道(^GSPC)在部分环境不可用
}

# ---------------- 风格因子基准（数据源优先级：MSCI 系 → ETF → 晨星） ----------------
# A股：中证官方风格指数（000918/000919/000922/000920 已验证可拉取）
# 美股：优先 MSCI 系风格 ETF（MTUM=MSCI动量、VLUE=MSCI价值、QUAL=MSCI质量、USMV=MSCI低波），
#       MSCI 无对应产品的风格用 SPDR ETF 兜底（SPYG=成长、SPYD=高股息）
STYLE_SOURCES = {
    "ashare": {
        "value": {"kind": "csindex", "code": "000918", "name": "沪深300价值指数"},
        "growth": {"kind": "csindex", "code": "000919", "name": "沪深300成长指数"},
        "dividend": {"kind": "csindex", "code": "000922", "name": "中证红利指数"},
        "momentum": {"kind": "csindex", "code": "000920", "name": "沪深300动量指数"},
    },
    "us": {
        "value": {"kind": "etf", "code": "usVLUE", "name": "MSCI美国价值( VLUE )"},
        "growth": {"kind": "etf", "code": "usSPYG", "name": "标普500成长ETF( SPYG )"},
        "dividend": {"kind": "etf", "code": "usSPYD", "name": "标普高股息ETF( SPYD )"},
        "momentum": {"kind": "etf", "code": "usMTUM", "name": "MSCI美国动量( MTUM )"},
        "quality": {"kind": "etf", "code": "usQUAL", "name": "MSCI美国质量( QUAL )"},
        "lowvol": {"kind": "etf", "code": "usUSMV", "name": "MSCI美国低波( USMV )"},
    },
}
STYLE_NAMES_ZH = {
    "value": "价值风格", "growth": "成长风格", "dividend": "红利风格",
    "momentum": "动量风格", "quality": "质量风格", "lowvol": "低波风格",
}

# ---------------- 美股行业归因：SPDR 行业 ETF（GICS 11 大行业） ----------------
US_SECTOR_ETFS = {
    # ---- GICS 宽基行业 ETF（兜底）----
    "科技": "usXLK", "通讯": "usXLC", "医疗": "usXLV", "金融": "usXLF",
    "能源": "usXLE", "工业": "usXLI", "消费": "usXLY", "必需消费": "usXLP",
    "公用事业": "usXLU", "材料": "usXLB", "地产": "usXLRE",
    # ---- 细分行业 ETF（优先，比 GICS 宽基更贴近行业实际走势）----
    "生物科技/制药": "usIBB",
    "银行": "usKBE",
    "保险": "usKIE",
    "军工": "usITA",
    "金矿": "usGDX",
    "软件": "usIGV",
    "半导体": "usSMH",
    "油气勘探": "usXOP",
    "房地产信托": "usVNQ",
    # ---- 中概股（中概互联 ETF 作基准）----
    "中概互联": "usKWEB",
}
# 热门美股 → 行业（细分行业优先，GICS 宽基兜底，中概股单独归类）
US_TICKER_SECTOR = {
    # ---- 中概股 → KWEB 中概互联（中概股单独归类，不与美股行业混）----
    "usbaba": "中概互联", "usbabahk": "中概互联", "uspdd": "中概互联", "usjd": "中概互联",
    "usbidu": "中概互联", "usnio": "中概互联", "usxpev": "中概互联", "usli": "中概互联",
    "usbili": "中概互联", "usiq": "中概互联", "ustme": "中概互联", "usvips": "中概互联",
    "ustal": "中概互联", "usedu": "中概互联", "usfutu": "中概互联", "uswb": "中概互联",
    "uszh": "中概互联", "usmnso": "中概互联", "uswbk": "中概互联", "ustnet": "中概互联",
    # ---- 生物科技/制药 → IBB ----
    "usabbv": "生物科技/制药", "usmrk": "生物科技/制药", "uslly": "生物科技/制药",
    "uspfe": "生物科技/制药", "usjnj": "生物科技/制药", "usamgn": "生物科技/制药",
    "usgild": "生物科技/制药", "usvrtx": "生物科技/制药", "usregn": "生物科技/制药",
    # ---- 银行 → KBE ----
    "uswfc": "银行", "usbac": "银行", "usjpm": "银行", "usc": "银行", "ususb": "银行", "uspnc": "银行",
    # ---- 保险 → KIE ----
    "usbrk.b": "保险", "usbrk-a": "保险", "usmet": "保险", "uspru": "保险",
    "usaig": "保险", "usall": "保险", "ustrv": "保险", "uspgr": "保险",
    # ---- 军工 → ITA ----
    "uslmt": "军工", "usrtx": "军工", "usnoc": "军工", "usgd": "军工", "uslhx": "军工",
    # ---- 金矿 → GDX ----
    "usnem": "金矿", "usgold": "金矿", "usaem": "金矿", "uswpm": "金矿",
    # ---- 软件 → IGV ----
    "usorcl": "软件", "usmsft": "软件", "uscrm": "软件", "usadbe": "软件",
    "usnow": "软件", "usintu": "软件", "ussnow": "软件", "uspltr": "软件",
    # ---- 半导体 → SMH ----
    "usnvda": "半导体", "usamd": "半导体", "usavgo": "半导体", "usintc": "半导体",
    "usqcom": "半导体", "ustxn": "半导体", "usmu": "半导体", "usamat": "半导体",
    # ---- 油气勘探 → XOP ----
    "uscop": "油气勘探", "useog": "油气勘探", "usslb": "油气勘探", "usvlo": "油气勘探", "uspsx": "油气勘探",
    # ---- 能源（一体化）→ XLE ----
    "usxom": "能源", "uscvx": "能源",
    # ---- 房地产信托 → VNQ ----
    "usamt": "房地产信托", "uspld": "房地产信托", "usspg": "房地产信托", "useqix": "房地产信托",
    # ---- 科技（宽，硬件/互联网）→ XLK ----
    "usaapl": "科技",
    # ---- 通讯 → XLC ----
    "usgoogl": "通讯", "usmeta": "通讯", "usnflx": "通讯", "usdis": "通讯",
    "uscmcsa": "通讯", "usvz": "通讯", "ust": "通讯", "ustmus": "通讯",
    # ---- 可选消费 → XLY ----
    "usamzn": "消费", "ustsla": "消费", "usnke": "消费", "ussbux": "消费",
    "ushd": "消费", "usmcd": "消费", "usbkng": "消费",
    # ---- 必需消费 → XLP ----
    "uscost": "必需消费", "uswmt": "必需消费", "uspg": "必需消费", "usko": "必需消费",
    "uspep": "必需消费", "uscl": "必需消费", "usmdlz": "必需消费",
    # ---- 医疗（器械/服务）→ XLV ----
    "usuth": "医疗", "ustmo": "医疗", "usabt": "医疗", "usdhr": "医疗", "usisrg": "医疗",
    # ---- 工业 → XLI ----
    "ushon": "工业", "uscat": "工业", "usge": "工业", "usmmm": "工业",
    "usupn": "工业", "usups": "工业", "usde": "工业", "usba": "工业",
    # ---- 材料 → XLB ----
    "usfcx": "材料", "usnue": "材料", "usdow": "材料", "uslin": "材料", "usshw": "材料",
    # ---- 公用事业 → XLU ----
    "usnee": "公用事业", "usduk": "公用事业", "usso": "公用事业", "usd": "公用事业", "usaep": "公用事业",
    # ---- 金融（投行/资管，宽）→ XLF ----
    "usgs": "金融", "usms": "金融", "usblk": "金融", "usv": "金融", "usma": "金融",
}

# 东财美股行业名（GICS 11 部门中文口径）→ US_SECTOR_ETFS 键
# 全市场股票的行业名只会取这 11 个值，映射后即可全覆盖
US_GICS_SECTOR = {
    "信息技术": "科技",
    "医疗保健": "医疗",
    "金融": "金融",
    "非日常生活消费品": "消费",
    "日常消费品": "必需消费",
    "能源": "能源",
    "工业": "工业",
    "原材料": "材料",
    "房地产": "地产",
    "通讯服务": "通讯",
    "公用事业": "公用事业",
}

# ---------------- 港股行业归因 ----------------
# 港股行业基准数据源有限：先给股票→行业名映射，行业基准暂以恒生指数兜底（如实标注）
HK_TICKER_INDUSTRY = {
    "hk00700": "互联网科技", "hk09988": "互联网科技", "hk03690": "互联网科技",
    "hk01024": "互联网科技", "hk09618": "互联网科技", "hk09999": "互联网科技",
    "hk00005": "银行", "hk01398": "银行", "hk03988": "银行", "hk02628": "保险",
    "hk01299": "保险", "hk02318": "保险", "hk00941": "电信", "hk00728": "电信",
    "hk02382": "消费电子", "hk01810": "消费电子", "hk02015": "新能源汽车",
    "hk09866": "新能源汽车", "hk01211": "新能源汽车", "hk00883": "能源",
    "hk00857": "能源", "hk00388": "金融", "hk06099": "证券",
}

REGISTRY_PATH = QTE_DIR / "factor_registry.json"
ATTRIBUTION_PATH = QTE_DIR / "attribution_framework.json"
CACHE_DIR = QTE_DIR / "cache"

_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}
_EM_SNAPSHOT_HOSTS = ["https://push2delay.eastmoney.com", "https://push2.eastmoney.com"]
_EM_KLINE_HOSTS = [
    "https://push2his.eastmoney.com",
    "https://92.push2his.eastmoney.com",
    "https://7.push2his.eastmoney.com",
    "https://43.push2his.eastmoney.com",
]


def _http_json(url_tail: str, hosts: list[str], timeout: int = 10, retries: int = 2) -> dict[str, Any]:
    """多主机+重试拉取东财接口，全部失败抛异常由调用方降级。"""
    last_exc: Exception | None = None
    for _ in range(retries):
        for host in hosts:
            try:
                req = urllib.request.Request(host + url_tail, headers=_EM_HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # 单主机失败换下一个
                last_exc = exc
        time.sleep(0.5)
    raise RuntimeError("东财接口全部主机失败: " + repr(last_exc))


def _load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _factor_entry(factor_id: str) -> dict[str, Any] | None:
    """读取正典定义或明确隔离的运行时风格代理定义。"""
    registry = _load_registry()
    entries = [
        *(registry.get("factors") or []),
        *(registry.get("runtime_style_factors") or []),
    ]
    for item in entries:
        if item.get("factor_id") == factor_id:
            return item
    return None


# ---------------- 数据层 ----------------

def _close_returns(symbol: str, years: int) -> pd.Series:
    """拉取日K并转为日收益率序列（索引为日期）。"""
    frame = fetch_daily(symbol, years=years)
    if frame is None or len(frame) < 30:
        raise ValueError(f"{symbol} 可用K线不足，无法计算因子")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close.index = pd.to_datetime(frame["Date"])
    close = close.dropna()
    return close.pct_change().dropna()


def _aligned(asset_ret: pd.Series, bench_ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按日期对齐资产与基准收益。"""
    joined = pd.concat([asset_ret, bench_ret], axis=1, keys=["asset", "bench"]).dropna()
    return joined["asset"], joined["bench"]


def _benchmark_for(symbol: str) -> str:
    prefix = symbol[:2].lower()
    return MARKET_BENCHMARK.get(prefix, "sh000300")


# ---------------- 东财基本面与行业数据层（P1） ----------------

def _em_secid(symbol: str) -> str | None:
    """内部代码转东财 secid（A股 / 港股 / 美股，指数除外）。"""
    s = symbol.lower()
    if s.startswith("sh") and s[2:].isdigit():
        return "1." + s[2:]
    if s.startswith(("sz", "bj")) and s[2:].isdigit():
        return "0." + s[2:]
    if s.startswith("hk") and s[2:].isdigit():
        return "116." + s[2:]
    if s.startswith("us"):
        ticker = s[2:]
        if not ticker.startswith("^") and re.fullmatch(r"[a-z][a-z0-9.\-]{0,10}", ticker):
            return "105." + ticker.upper()
    return None


def fetch_fundamental_snapshot(symbol: str) -> dict[str, Any] | None:
    """东财快照：PE、总市值、流通市值、行业名。失败返回 None，不阻塞主流程。"""
    secid = _em_secid(symbol)
    if not secid:
        return None
    tail = ("/api/qt/stock/get?secid=" + secid +
            "&fltt=2&fields=f43,f58,f162,f167,f116,f117,f127")
    try:
        data = (_http_json(tail, _EM_SNAPSHOT_HOSTS).get("data")) or {}
        if not data:
            return None
        return {
            "name": data.get("f58"),
            "price": data.get("f43"),
            "pe_dynamic": data.get("f162"),
            "pb": data.get("f167"),
            "total_market_cap": data.get("f116"),
            "float_market_cap": data.get("f117"),
            "industry_name": data.get("f127"),
        }
    except Exception:
        return None


_board_cache: dict[str, str] | None = None

# ---------------- 美股/港股全市场行业分类（离线全量表 + 实时兜底） ----------------
# 口径说明：因子挖掘要求全市场覆盖，不能只映射热门股。
# us_hk_industry_map.json 由东财列表接口(clist f100)全量拉取生成：
#   美股 4700+ 只（GICS 11 部门中文口径）、港股 2600+ 只（港交所行业口径）。
# 新上市股票不在表内时，实时查东财快照接口兜底；都取不到才走大盘/恒指兜底。
US_HK_MAP_PATH = QTE_DIR / "us_hk_industry_map.json"
_us_hk_map_cache: dict[str, Any] | None = None


def _us_hk_industry_map() -> dict[str, Any]:
    """惰性加载全市场行业映射表；文件缺失时返回空表，不阻塞主流程。"""
    global _us_hk_map_cache
    if _us_hk_map_cache is None:
        try:
            with open(US_HK_MAP_PATH, "r", encoding="utf-8") as f:
                _us_hk_map_cache = json.load(f)
        except Exception:
            _us_hk_map_cache = {}
    return _us_hk_map_cache


def _lookup_full_market_industry(symbol: str) -> str | None:
    """全市场离线表查行业名；表外（新股等）用东财实时接口兜底。"""
    s = str(symbol).lower()
    m = _us_hk_industry_map()
    if s.startswith("us"):
        industry = (m.get("us") or {}).get(s[2:])
    elif s.startswith("hk"):
        industry = (m.get("hk") or {}).get(s[2:])
    else:
        return None
    if industry:
        return industry
    try:
        fund = fetch_fundamental_snapshot(symbol)
        industry = (fund or {}).get("industry_name")
        return industry or None
    except Exception:
        return None

# ---------------- 行业指数源：官方指数优先，行业 ETF 兜底 ----------------
#
# 口径说明（重要）：行业基准不用东财/新浪等各家自有的板块指数（各家分类不一致，
# 不客观），而是：
#   kind=csindex：中证指数公司官网指数（上交所/深交所合资设立，全市场统一代码）
#   kind=etf    ：该行业规模最大的 ETF（只用收益率不用点位，与指数等价）
# 新行业接入时在映射表里补一行即可。
INDUSTRY_INDEX_MAP: dict[str, dict[str, str]] = {
    # ---- 白酒 ----
    "白酒Ⅱ": {"kind": "csindex", "code": "930610", "name": "中证白酒指数"},
    "白酒Ⅲ": {"kind": "csindex", "code": "930610", "name": "中证白酒指数"},
    # ---- 银行 ----
    "银行Ⅱ": {"kind": "csindex", "code": "399986", "name": "中证银行指数"},
    # ---- 证券 ----
    "证券Ⅱ": {"kind": "csindex", "code": "399975", "name": "中证证券指数"},
    # ---- 保险 ----
    "保险Ⅱ": {"kind": "csindex", "code": "399974", "name": "中证保险指数"},
    # ---- 医药 ----
    "医药生物Ⅱ": {"kind": "csindex", "code": "000933", "name": "中证医药生物指数"},
    "医疗器械Ⅱ": {"kind": "csindex", "code": "931152", "name": "中证医疗器械指数"},
    # ---- 房地产 ----
    "房地产Ⅱ": {"kind": "csindex", "code": "000952", "name": "中证房地产指数"},
    # ---- 电子 / 半导体 ----
    "电子Ⅱ": {"kind": "csindex", "code": "000970", "name": "中证电子指数"},
    "半导体Ⅱ": {"kind": "csindex", "code": "930614", "name": "中证半导体指数"},
    # ---- 新能源 / 光伏 ----
    "新能源Ⅱ": {"kind": "csindex", "code": "399808", "name": "中证新能源指数"},
    "光伏Ⅱ": {"kind": "csindex", "code": "930697", "name": "中证光伏产业指数"},
    # ---- 食品饮料 ----
    "食品饮料Ⅱ": {"kind": "csindex", "code": "000932", "name": "中证食品饮料指数"},
    # ---- 家电 ----
    "家电Ⅱ": {"kind": "csindex", "code": "930697", "name": "中证家电指数"},
    # ---- 汽车 ----
    "汽车Ⅱ": {"kind": "csindex", "code": "000957", "name": "中证汽车指数"},
    # ---- 有色金属 / 黄金 / 贵金属 ----
    "有色金属Ⅱ": {"kind": "csindex", "code": "930708", "name": "中证有色金属指数"},
    "有色金属Ⅲ": {"kind": "csindex", "code": "930708", "name": "中证有色金属指数"},
    "黄金Ⅲ": {"kind": "csindex", "code": "932265", "name": "中证黄金产业股票指数"},
    "贵金属Ⅱ": {"kind": "csindex", "code": "932265", "name": "中证黄金产业股票指数"},
    # ---- 钢铁 ----
    "钢铁Ⅱ": {"kind": "csindex", "code": "000929", "name": "中证钢铁指数"},
    # ---- 电力 / 公用事业 ----
    "电力Ⅱ": {"kind": "csindex", "code": "000966", "name": "中证电力指数"},
    "公用事业Ⅱ": {"kind": "csindex", "code": "000980", "name": "中证公用事业指数"},
    # ---- 建筑 / 建材 ----
    "建筑Ⅱ": {"kind": "csindex", "code": "000952", "name": "中证建筑指数(暂以地产指数代理)"},
    "建材Ⅱ": {"kind": "csindex", "code": "000931", "name": "中证建材指数"},
    # ---- 交通运输 ----
    "交通运输Ⅱ": {"kind": "csindex", "code": "000929", "name": "中证交通运输指数(代理)"},
    # ---- 通信 ----
    "通信Ⅱ": {"kind": "csindex", "code": "000935", "name": "中证通信指数"},
    # ---- 纺织 / 农林 / 商贸 / 机械 ----
    "纺织服饰Ⅱ": {"kind": "csindex", "code": "000931", "name": "中证纺织服饰指数(代理)"},
    "农林牧渔Ⅱ": {"kind": "csindex", "code": "000931", "name": "中证农林牧渔指数(代理)"},
    "商贸零售Ⅱ": {"kind": "csindex", "code": "000931", "name": "中证商贸零售指数(代理)"},
    "机械Ⅱ": {"kind": "csindex", "code": "000957", "name": "中证机械指数(代理)"},
    # ---- 基础化工 ----
    "基础化工Ⅱ": {"kind": "csindex", "code": "000931", "name": "中证基础化工指数(代理)"},
    # ---- 煤炭 / 石油石化 ----
    "煤炭Ⅱ": {"kind": "csindex", "code": "000929", "name": "中证煤炭指数(代理)"},
    "石油石化Ⅱ": {"kind": "csindex", "code": "000929", "name": "中证石油石化指数(代理)"},
    # ---- 军工 ----
    "军工Ⅱ": {"kind": "csindex", "code": "399959", "name": "中证军工指数"},
    # ---- 计算机 / 软件 ----
    "计算机Ⅱ": {"kind": "csindex", "code": "000970", "name": "中证计算机指数"},
    "软件Ⅱ": {"kind": "csindex", "code": "000935", "name": "中证软件指数(代理)"},
    # ---- 传媒 ----
    "传媒Ⅱ": {"kind": "csindex", "code": "000935", "name": "中证传媒指数(代理)"},
    # ---- 美股行业基准（SPDR Select Sector ETF）----
    "Information Technology": {"kind": "etf", "code": "usxlk", "name": "XLK 科技"},
    "Financials": {"kind": "etf", "code": "usxlf", "name": "XLF 金融"},
    "Health Care": {"kind": "etf", "code": "usxlv", "name": "XLV 医疗保健"},
    "Consumer Discretionary": {"kind": "etf", "code": "usxly", "name": "XLY 可选消费"},
    "Communication Services": {"kind": "etf", "code": "usxlc", "name": "XLC 通信服务"},
    "Industrials": {"kind": "etf", "code": "usxli", "name": "XLI 工业"},
    "Consumer Staples": {"kind": "etf", "code": "usxlp", "name": "XLP 必选消费"},
    "Energy": {"kind": "etf", "code": "usxle", "name": "XLE 能源"},
    "Utilities": {"kind": "etf", "code": "usxlu", "name": "XLU 公用事业"},
    "Real Estate": {"kind": "etf", "code": "usxlre", "name": "XLRE 房地产"},
    "Materials": {"kind": "etf", "code": "usxlb", "name": "XLB 材料"},
}

CSINDEX_API = "https://www.csindex.com.cn/csindex-home/perf/index-perf"


# 东财 f127 返回的申万二级行业名 → INDUSTRY_INDEX_MAP 键名的别名映射
# 解决 f127 返回名与 MAP 键名不一致的问题
_INDUSTRY_ALIASES: dict[str, str] = {
    # ---- 直接映射到已有 MAP 键 ----
    "黄金": "黄金Ⅲ",
    "贵金属": "贵金属Ⅱ",
    "有色金属": "有色金属Ⅱ",
    "白酒": "白酒Ⅱ",
    "银行": "银行Ⅱ",
    "证券": "证券Ⅱ",
    "保险": "保险Ⅱ",
    "房地产": "房地产Ⅱ",
    "半导体": "半导体Ⅱ",
    "光伏": "光伏Ⅱ",
    "军工": "军工Ⅱ",
    "软件": "软件Ⅱ",
    "传媒": "传媒Ⅱ",
    "煤炭": "煤炭Ⅱ",
    "钢铁": "钢铁Ⅱ",
    "电力": "电力Ⅱ",
    "建筑": "建筑Ⅱ",
    "建材": "建材Ⅱ",
    "通信": "通信Ⅱ",
    "汽车": "汽车Ⅱ",
    "机械": "机械Ⅱ",
    "纺织服饰": "纺织服饰Ⅱ",
    "农林牧渔": "农林牧渔Ⅱ",
    "基础化工": "基础化工Ⅱ",
    # ---- 申万二级行业精确名称（东财 f127 实际返回值）----
    "工业金属": "有色金属Ⅱ",
    "小金属": "有色金属Ⅱ",
    "金属新材料": "有色金属Ⅱ",
    "炼化及贸易": "石油石化Ⅱ",
    "石油开采": "石油石化Ⅱ",
    "油服工程": "石油石化Ⅱ",
    "乘用车": "汽车Ⅱ",
    "商用车": "汽车Ⅱ",
    "旅游零售Ⅱ": "商贸零售Ⅱ",
    "旅游零售": "商贸零售Ⅱ",
    "免税": "商贸零售Ⅱ",
    "白色家电": "家电Ⅱ",
    "小家电": "家电Ⅱ",
    "黑色家电": "家电Ⅱ",
    "厨卫电器": "家电Ⅱ",
    "照明工具": "家电Ⅱ",
    "乳制品": "食品饮料Ⅱ",
    "饮料乳品": "食品饮料Ⅱ",
    "休闲食品": "食品饮料Ⅱ",
    "调味发酵品": "食品饮料Ⅱ",
    "保健品": "食品饮料Ⅱ",
    "化学制药": "医药生物Ⅱ",
    "生物制品": "医药生物Ⅱ",
    "中药Ⅱ": "医药生物Ⅱ",
    "中药": "医药生物Ⅱ",
    "医药商业": "医药生物Ⅱ",
    "医疗器械": "医疗器械Ⅱ",
    "医疗服务": "医药生物Ⅱ",
    "美容护理": "商贸零售Ⅱ",
    "个护用品": "商贸零售Ⅱ",
    "光伏设备": "光伏Ⅱ",
    "风电设备": "新能源Ⅱ",
    "电池": "新能源Ⅱ",
    "电网设备": "电力Ⅱ",
    "发电设备": "电力Ⅱ",
    "半导体": "半导体Ⅱ",
    "光学光电子": "电子Ⅱ",
    "元件": "电子Ⅱ",
    "消费电子": "电子Ⅱ",
    "其他电子Ⅱ": "电子Ⅱ",
    "软件开发": "软件Ⅱ",
    "IT服务Ⅱ": "计算机Ⅱ",
    "计算机设备": "计算机Ⅱ",
    "航空装备Ⅱ": "军工Ⅱ",
    "航天装备Ⅱ": "军工Ⅱ",
    "地面兵装Ⅱ": "军工Ⅱ",
    "船舶制造Ⅱ": "军工Ⅱ",
    "水泥": "建材Ⅱ",
    "玻璃玻纤": "建材Ⅱ",
    "装修建材": "建材Ⅱ",
    "房地产开发": "房地产Ⅱ",
    "房地产服务": "房地产Ⅱ",
    "证券Ⅲ": "证券Ⅱ",
    "保险Ⅲ": "保险Ⅱ",
    "银行Ⅲ": "银行Ⅱ",
    "航空机场": "交通运输Ⅱ",
    "航运港口": "交通运输Ⅱ",
    "铁路公路": "交通运输Ⅱ",
    "物流Ⅱ": "交通运输Ⅱ",
    "仓储物流": "交通运输Ⅱ",
    "化学制品": "基础化工Ⅱ",
    "化学原料": "基础化工Ⅱ",
    "化学纤维": "基础化工Ⅱ",
    "化肥Ⅱ": "基础化工Ⅱ",
    "橡胶": "基础化工Ⅱ",
    "塑料": "基础化工Ⅱ",
    "专用设备": "机械Ⅱ",
    "通用设备": "机械Ⅱ",
    "自动化设备": "机械Ⅱ",
    "工程机械": "机械Ⅱ",
    "金属制品": "机械Ⅱ",
    "装修装饰": "建筑Ⅱ",
    "房屋建设": "建筑Ⅱ",
    "基础建设": "建筑Ⅱ",
    "专业工程": "建筑Ⅱ",
    "种植业": "农林牧渔Ⅱ",
    "养殖业": "农林牧渔Ⅱ",
    "饲料": "农林牧渔Ⅱ",
    "渔业": "农林牧渔Ⅱ",
    "纺织制造": "纺织服饰Ⅱ",
    "服装家纺": "纺织服饰Ⅱ",
    "饰品": "纺织服饰Ⅱ",
    "游戏": "传媒Ⅱ",
    "影视院线": "传媒Ⅱ",
    "广告营销": "传媒Ⅱ",
    "出版": "传媒Ⅱ",
    "通信设备": "通信Ⅱ",
    "通信服务": "通信Ⅱ",
    "电力行业": "公用事业Ⅱ",
    "燃气Ⅱ": "公用事业Ⅱ",
    "环保Ⅱ": "公用事业Ⅱ",
    "水处理": "公用事业Ⅱ",
    "一般零售": "商贸零售Ⅱ",
    "专业连锁": "商贸零售Ⅱ",
    "互联网电商": "商贸零售Ⅱ",
    # ---- f127 常用名 → MAP 键 ----
    "石油石化": "石油石化Ⅱ",
    "石油贸易": "石油石化Ⅱ",
    "医药生物": "医药生物Ⅱ",
    "化学制药": "医药生物Ⅱ",
    "医疗器械": "医疗器械Ⅱ",
    "生物制品": "医药生物Ⅱ",
    "中药": "医药生物Ⅱ",
    "医药商业": "医药生物Ⅱ",
    "电子": "电子Ⅱ",
    "光学光电子": "电子Ⅱ",
    "元件": "电子Ⅱ",
    "其他电子": "电子Ⅱ",
    "计算机": "计算机Ⅱ",
    "IT服务": "计算机Ⅱ",
    "计算机应用": "计算机Ⅱ",
    "计算机设备": "计算机Ⅱ",
    "食品饮料": "食品饮料Ⅱ",
    "休闲食品": "食品饮料Ⅱ",
    "饮料乳品": "食品饮料Ⅱ",
    "调味发酵品": "食品饮料Ⅱ",
    "家电": "家电Ⅱ",
    "白色家电": "家电Ⅱ",
    "小家电": "家电Ⅱ",
    "厨卫电器": "家电Ⅱ",
    "黑色家电": "家电Ⅱ",
    "新能源": "新能源Ⅱ",
    "电池": "新能源Ⅱ",
    "电网设备": "电力Ⅱ",
    "发电设备": "电力Ⅱ",
    "公用事业": "公用事业Ⅱ",
    "电力行业": "公用事业Ⅱ",
    "环保": "公用事业Ⅱ",
    "燃气": "公用事业Ⅱ",
    "水处理": "公用事业Ⅱ",
    "交通运输": "交通运输Ⅱ",
    "物流": "交通运输Ⅱ",
    "航空机场": "交通运输Ⅱ",
    "航运港口": "交通运输Ⅱ",
    "铁路公路": "交通运输Ⅱ",
    "商贸零售": "商贸零售Ⅱ",
    "一般零售": "商贸零售Ⅱ",
    "专业连锁": "商贸零售Ⅱ",
    "互联网电商": "商贸零售Ⅱ",
    "美容护理": "商贸零售Ⅱ",
    "汽车服务": "汽车Ⅱ",
    "汽车零部件": "汽车Ⅱ",
    "摩托车": "汽车Ⅱ",
    "非银金融": "证券Ⅱ",
    "多元金融": "证券Ⅱ",
    "纺织制造": "纺织服饰Ⅱ",
    "服饰": "纺织服饰Ⅱ",
    "服装家纺": "纺织服饰Ⅱ",
    "种植业": "农林牧渔Ⅱ",
    "养殖业": "农林牧渔Ⅱ",
    "饲料": "农林牧渔Ⅱ",
    "渔业": "农林牧渔Ⅱ",
    "农业综合": "农林牧渔Ⅱ",
    "化学制品": "基础化工Ⅱ",
    "化学原料": "基础化工Ⅱ",
    "化肥": "基础化工Ⅱ",
    "橡胶": "基础化工Ⅱ",
    "塑料": "基础化工Ⅱ",
    "聚氨酯": "基础化工Ⅱ",
    "涂料": "基础化工Ⅱ",
    "无机盐": "基础化工Ⅱ",
    "其他化学制品": "基础化工Ⅱ",
    "专用设备": "机械Ⅱ",
    "通用设备": "机械Ⅱ",
    "自动化设备": "机械Ⅱ",
    "工程机械": "机械Ⅱ",
    "金属制品": "机械Ⅱ",
    "装修装饰": "建筑Ⅱ",
    "房屋建设": "建筑Ⅱ",
    "基础建设": "建筑Ⅱ",
    "专业工程": "建筑Ⅱ",
    "水泥": "建材Ⅱ",
    "玻璃玻纤": "建材Ⅱ",
    "装修建材": "建材Ⅱ",
    "管材": "建材Ⅱ",
    "消费电子": "电子Ⅱ",
    "通信设备": "通信Ⅱ",
    "通信服务": "通信Ⅱ",
    "游戏": "传媒Ⅱ",
    "影视院线": "传媒Ⅱ",
    "广告营销": "传媒Ⅱ",
    "出版": "传媒Ⅱ",
    "互联网": "传媒Ⅱ",
    "航空装备": "军工Ⅱ",
    "航天装备": "军工Ⅱ",
    "地面兵装": "军工Ⅱ",
    "船舶制造": "军工Ⅱ",
}

def _resolve_industry_source(industry_name: str | None) -> dict[str, str] | None:
    """行业名 → 官方指数/ETF 源；精确匹配 → 别名映射 → 包含匹配。"""
    if not industry_name:
        return None
    # 1. 精确匹配 MAP 键
    if industry_name in INDUSTRY_INDEX_MAP:
        return INDUSTRY_INDEX_MAP[industry_name]
    # 2. 别名映射
    alias_key = _INDUSTRY_ALIASES.get(industry_name)
    if alias_key and alias_key in INDUSTRY_INDEX_MAP:
        return INDUSTRY_INDEX_MAP[alias_key]
    # 3. 模糊包含匹配
    for key, value in INDUSTRY_INDEX_MAP.items():
        clean_key = key.rstrip("ⅡⅢ")
        if clean_key and (clean_key in industry_name or industry_name in clean_key):
            return value
    # 4. 别名模糊匹配
    for alias, map_key in _INDUSTRY_ALIASES.items():
        if alias in industry_name or industry_name in alias:
            if map_key in INDUSTRY_INDEX_MAP:
                return INDUSTRY_INDEX_MAP[map_key]
    return None


def fetch_csindex_returns(code: str, years: int = 3) -> pd.Series | None:
    """中证指数官网官方接口取指数日线，带磁盘缓存。"""
    cache_file = CACHE_DIR / f"csindex_{code}.csv"
    if cache_file.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime)).days
        if age_days < 1:
            return _read_cache(cache_file)
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    url = f"{CSINDEX_API}?indexCode={code}&startDate={start}&endDate={end}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36",
            "Referer": "https://www.csindex.com.cn/",
        })
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("data") or []
        if len(rows) < 30:
            if cache_file.exists():
                return _read_cache(cache_file)
            return None
        close = pd.Series(
            [float(r["close"]) for r in rows],
            index=pd.to_datetime([str(r["tradeDate"]) for r in rows]),
        ).sort_index()
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            close.to_csv(cache_file, header=["close"])
        except Exception:
            pass
        return close.pct_change().dropna()
    except Exception:
        if cache_file.exists():
            return _read_cache(cache_file)
        return None


def fetch_style_returns(prefix: str, asset_index: pd.Index, bench_ret: pd.Series, years: int = 3) -> dict[str, tuple[pd.Series, str]]:
    """取风格因子超额收益序列（风格基准 - 大盘基准），与资产索引对齐。

    返回 {风格key: (超额收益序列, 基准名称)}。拉不到/覆盖不足的风格自动跳过。
    """
    key = "us" if prefix == "us" else ("ashare" if prefix in ("sh", "sz", "bj") else None)
    defs = STYLE_SOURCES.get(key, {})
    out: dict[str, tuple[pd.Series, str]] = {}
    for skey, src in defs.items():
        try:
            if src["kind"] == "csindex":
                sret = fetch_csindex_returns(src["code"], years=years)
            else:
                sret = _close_returns(src["code"], years=years)
            if sret is None or len(sret) < 100:
                continue
            common = sret.index.intersection(asset_index)
            if len(common) < max(MIN_OBS, int(len(asset_index) * 0.6)):
                continue
            excess = (sret.reindex(asset_index) - bench_ret.reindex(asset_index)).dropna()
            if len(excess) < MIN_OBS:
                continue
            out[skey] = (excess, src["name"])
        except Exception:
            continue
    return out


def _bench_sufficient(returns: pd.Series | None, years: int) -> bool:
    """基准历史是否足够覆盖回归窗口：需 >=90% 的目标交易日数。

    防的是：行业 ETF 上市时间短于回归窗口时，reindex+dropna 会悄悄把
    整个样本截断到 ETF 的上市区间，导致归因窗口与页面标注不一致。
    """
    if returns is None:
        return False
    need = int(years * 252 * 0.9)
    return len(returns) >= need


def fetch_us_sector_returns(symbol: str, years: int = 3) -> tuple[pd.Series | None, dict[str, str]]:
    """美股行业归因（全市场覆盖）。

    优先级：手工细分映射（中概股/细分行业）→ 全市场行业表（东财 GICS 部门 → 行业ETF）
    → 实时接口兜底 → 大盘 SPY。任何一只美股都能得到行业归因。
    """
    key = str(symbol).lower()
    industry = None
    sector = US_TICKER_SECTOR.get(key)
    etf = US_SECTOR_ETFS.get(sector) if sector else None
    if not etf:
        industry = _lookup_full_market_industry(symbol)
        sector = US_GICS_SECTOR.get(industry) if industry else None
        etf = US_SECTOR_ETFS.get(sector) if sector else None
    if not etf:
        # 兜底：行业信息完全取不到时，用大盘 SPY 作行业基准（如实标注）
        etf, sector = "usSPY", "大盘(未细分行业)"
    try:
        returns = _close_returns(etf, years=years)
    except Exception:
        returns = None
    # 历史覆盖校验：行业 ETF 上市时间不够长时如实降级为大盘基准
    if not _bench_sufficient(returns, years) and etf != "usSPY":
        try:
            returns = _close_returns("usSPY", years=years)
        except Exception:
            returns = None
        if returns is None:
            return None, {"kind": "etf", "code": etf, "name": f"{sector}行业ETF", "note": "行业ETF数据暂时取不到"}
        return returns, {"kind": "etf", "code": "usSPY", "name": f"大盘基准({sector}行业ETF上市时间不足，如实降级)",
                         "note": f"{sector}行业ETF历史不足回归窗口，以大盘作行业基准", "industry_name": sector}
    if returns is None:
        return None, {"kind": "etf", "code": etf, "name": f"{sector}行业ETF", "note": "行业ETF数据暂时取不到"}
    if sector == "大盘(未细分行业)":
        return returns, {"kind": "etf", "code": etf, "name": "大盘基准(该美股暂无行业分类)", "note": "该美股行业信息暂时取不到，以大盘作行业基准"}
    display = f"{industry}·{sector}" if industry else sector
    return returns, {"kind": "etf", "code": etf, "name": f"{display}行业ETF({etf[2:]})", "industry_name": industry or sector}


# ---------------- 港股行业归因 ----------------
HK_SECTOR_ETFS = {
    "互联网科技": "hk03033",  # 恒生科技 ETF
}

# 东财港股行业名（港交所行业口径）→ 可拉取的行业基准。
# 选基准的硬约束：历史长度必须覆盖回归窗口（最长 3 年，>=680 交易日），
# 否则运行时的 _bench_sufficient 校验会如实降级为恒生指数。
# 来源分两类：
#   港股本地 ETF（同市场，优先）：科技/医药/高息等均有 5 年以上历史
#   A股跨境港股行业 ETF（跟踪港股行业指数，走势同源）：消费/医药/金融等
HK_INDUSTRY_BENCH = {
    # ---- 科技系 → 恒生科技 ETF（港股本地，5年+历史）----
    "软件服务": "hk03033",
    "资讯科技器材": "hk03033",
    "半导体": "hk03033",
    "媒体及娱乐": "hk03033",
    # ---- 医药系 → 港股医药 ETF ----
    "药品及生物科技": "hk03069",    # 华夏恒生生科（港股本地，5年+）
    "其他医疗保健": "sh513060",      # 恒生医疗ETF博时（A股跨境，5年+）
    # ---- 消费系 → 港股消费 ETF ----
    "食物饮品": "sz159735",          # 港股消费ETF银华（5年+）
    "消费者主要零售商": "sz159735",
    "专业零售": "sz159735",
    "家庭电器及用品": "sz159735",
    "纺织及服饰": "sz513590",        # 港股通消费ETF鹏华（5年+，大消费口径）
    "农业产品": "sz513590",
    # ---- 高息/公用（股息风格近似）→ 港股高息 ETF ----
    "公用事业": "hk03070",           # 平安香港高息（港股本地，5年+）
    "电讯": "hk03070",
    # ---- 金融（非银行类）→ 港股通金融 ETF（3年+，满足窗口）----
    "其他金融": "sz513140",
    "保险": "sz513140",
    "金融": "sz513140",             # HK_TICKER_INDUSTRY 手工映射用的名字
    "证券": "sz513140",
    # ---- 电讯（手工映射名）→ 平安香港高息，与"公用事业"对齐 ----
    "电信": "hk03070",
    # ---- 消费电子（手工映射名）→ 恒生科技（小米/舜宇等均为成分股）----
    "消费电子": "hk03033",
    # ---- 地产 → 高息房托 REITs ETF（港股本地，5年+，领展/越秀房托等成分，同属地产收租逻辑）----
    "地产": "hk03187",
    # ---- 基建/公用运输（高息基础设施属性）→ 平安香港高息（中电/长江基建等为成分）----
    "工用运输": "hk03070",
}


def fetch_hk_sector_returns(symbol: str, years: int = 3) -> tuple[pd.Series | None, dict[str, str]]:
    """港股行业归因（全市场覆盖）。

    优先级：手工映射 → 全市场行业表（东财行业名 → 恒生科技ETF）→ 实时接口兜底 → 恒生指数。
    任何一只港股都能得到行业归因；无专属行业基准的行业如实降级为恒生指数。
    """
    key = str(symbol).lower()
    industry = HK_TICKER_INDUSTRY.get(key)
    bench = HK_SECTOR_ETFS.get(industry) if industry else None
    if not bench:
        industry = industry or _lookup_full_market_industry(symbol)
        bench = HK_INDUSTRY_BENCH.get(industry) if industry else None
    if bench:
        try:
            returns = _close_returns(bench, years=years)
        except Exception:
            returns = None
        # 历史覆盖校验：行业 ETF 上市时间不够长时如实降级为恒生指数
        if _bench_sufficient(returns, years):
            kind = "etf"
            return returns, {"kind": kind, "code": bench, "name": f"{industry}行业ETF({bench[2:]})", "industry_name": industry}
    # 行业无专属基准（或行业 ETF 历史不足）→ 兜底恒生指数（如实标注）
    try:
        returns = _close_returns("hkhsi", years=years)
    except Exception:
        returns = None
    if returns is None:
        return None, {"kind": "index", "code": "hkhsi", "name": "恒生指数", "note": "恒生指数数据暂时取不到"}
    if industry:
        if bench:
            note = f"{industry}行业ETF上市时间不足回归窗口，以恒生指数为基准"
            name = f"恒生指数({industry}行业ETF上市不足，如实降级)"
        else:
            note = f"{industry}暂无专属港股行业ETF，以恒生指数为基准"
            name = f"恒生指数({industry}暂无专属行业ETF)"
        return returns, {"kind": "index", "code": "hkhsi", "name": name, "note": note, "industry_name": industry}
    return returns, {"kind": "index", "code": "hkhsi", "name": "恒生指数(该港股暂无行业分类)", "note": "该港股行业信息暂时取不到，以恒生指数作行业基准"}


def fetch_industry_returns(industry_name: str, years: int = 3) -> tuple[pd.Series | None, dict[str, str]]:
    """按行业名取行业基准收益序列。

    返回 (收益序列或 None, 源信息 dict)。源信息含 kind/code/name，供输出溯源。
    """
    source = _resolve_industry_source(industry_name)
    if not source:
        return None, {"kind": "unmapped", "note": "该行业暂未登记官方指数或 ETF 基准"}
    if source["kind"] == "csindex":
        returns = fetch_csindex_returns(source["code"], years=years)
    elif source["kind"] == "etf":
        try:
            returns = _close_returns(source["code"], years=years)
        except Exception:
            returns = None
        # ETF 兜底分支同样要做历史覆盖校验，不足时如实标注降级
        if not _bench_sufficient(returns, years):
            note = f"{source.get('name', '行业ETF')}上市时间不足回归窗口，该行业归因已降级"
            return None, {**source, "note": note}
    else:
        returns = None
    if returns is None:
        return None, {**source, "note": "基准数据暂时取不到，已降级"}
    return returns, source


def _read_cache(cache_file: Path) -> pd.Series | None:
    try:
        close = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
        return close.pct_change().dropna()
    except Exception:
        return None


# ---------------- Baostock 财报因子层（免费无密钥） ----------------

def _bs_code(symbol: str) -> str | None:
    s = symbol.lower()
    if s.startswith("sh") and s[2:].isdigit():
        return "sh." + s[2:]
    if s.startswith(("sz", "bj")) and s[2:].isdigit():
        return "sz." + s[2:]
    return None


def _recent_quarters(n: int = 4) -> list[tuple[int, int]]:
    """按披露节奏倒推最近 n 个可能已披露的季度。"""
    today = pd.Timestamp.today()
    out = []
    y, q = today.year, today.quarter
    for _ in range(n + 2):
        q -= 1
        if q == 0:
            q = 4
            y -= 1
        out.append((y, q))
    return out


def _bs_float(value: Any) -> float | None:
    try:
        v = float(str(value or "").strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def fetch_bs_fundamentals(symbol: str) -> dict[str, Any] | None:
    """Baostock 季报因子：ROE、毛利率、净利率、利润增速、负债率。

    防前视偏差：只取最近一个已披露季度，并登记 pubDate（披露日）与 statDate（统计期）。
    失败返回 None，不阻塞主流程。
    """
    code = _bs_code(symbol)
    if not code:
        return None
    try:
        import baostock as bs
    except ImportError:
        return None
    cache_file = CACHE_DIR / f"bs_{code.replace('.', '_')}.json"
    try:
        if cache_file.exists():
            age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime)).days
            if age_days < 1:
                return json.loads(cache_file.read_text(encoding="utf-8"))
        lg = bs.login()
        if lg.error_code != "0":
            return None
        try:
            for year, quarter in _recent_quarters():
                rows: dict[str, dict] = {}
                for key, fn in (("profit", bs.query_profit_data),
                                ("growth", bs.query_growth_data),
                                ("balance", bs.query_balance_data)):
                    rs = fn(code=code, year=year, quarter=quarter)
                    if rs.error_code != "0":
                        continue
                    while rs.next():
                        rows[key] = dict(zip(rs.fields, rs.get_row_data()))
                p = rows.get("profit") or {}
                if not p.get("roeAvg"):
                    continue  # 该季度未披露，向前回溯
                result = {
                    "source": "baostock",
                    "stat_date": p.get("statDate"),
                    "pub_date": p.get("pubDate"),
                    "quality_roe": _bs_float(p.get("roeAvg")),
                    "quality_gross_margin": _bs_float(p.get("gpMargin")),
                    "quality_net_margin": _bs_float(p.get("npMargin")),
                    "growth_earnings": _bs_float((rows.get("growth") or {}).get("YOYNI")),
                    "growth_asset": _bs_float((rows.get("growth") or {}).get("YOYAsset")),
                    "leverage": _bs_float((rows.get("balance") or {}).get("liabilityToAsset")),
                    "current_ratio": _bs_float((rows.get("balance") or {}).get("currentRatio")),
                }
                try:
                    CACHE_DIR.mkdir(exist_ok=True)
                    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                return result
        finally:
            bs.logout()
    except Exception:
        pass
    return None




def fetch_pe_percentile(symbol: str, years: int = 5) -> dict[str, Any] | None:
    """用 baostock 获取历史 PE TTM 序列，计算当前 PE 的历史百分位。

    返回 {"current": float, "percentile": int, "low": float, "high": float,
           "median": float, "zone": str, "sample_size": int} 或 None。
    """
    code = _bs_code(symbol)
    if not code:
        return None
    try:
        import baostock as bs
    except ImportError:
        return None
    cache_file = CACHE_DIR / f"pe_pct_{code.replace('.', '_')}.json"
    try:
        if cache_file.exists():
            age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime)).days
            if age_days < 1:
                return json.loads(cache_file.read_text(encoding="utf-8"))
        lg = bs.login()
        if lg.error_code != "0":
            return None
        try:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
            start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                code,
                "date,peTTM,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",  # 不复权（PE 用实际交易价）
            )
            if rs.error_code != "0":
                return None
            rows = []
            while rs.next():
                rows.append(dict(zip(rs.fields, rs.get_row_data())))
            if len(rows) < 60:
                return None
            pe_values = []
            for r in rows:
                try:
                    pe = float(r.get("peTTM", ""))
                    if pe > 0 and math.isfinite(pe):  # 排除负PE和异常值
                        pe_values.append(pe)
                except (TypeError, ValueError):
                    continue
            if len(pe_values) < 60:
                return None
            current = pe_values[-1]
            ordered = sorted(pe_values)
            rank = sum(1 for v in pe_values if v <= current)
            percentile = int(round(rank / len(pe_values) * 100))
            low = ordered[max(0, int(len(ordered) * 0.05))]
            high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            median = ordered[len(ordered) // 2]
            if percentile >= 80:
                zone = "历史偏高区"
            elif percentile <= 20:
                zone = "历史偏低区"
            else:
                zone = "历史中间区"
            result = {
                "current": round(current, 2),
                "percentile": percentile,
                "low": round(low, 2),
                "high": round(high, 2),
                "median": round(median, 2),
                "zone": zone,
                "sample_size": len(pe_values),
                "years": years,
                "source": "baostock",
            }
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return result
        finally:
            bs.logout()
    except Exception:
        pass
    return None

# ---------------- 因子构造层（P0） ----------------

def factor_momentum_12_1(asset_ret: pd.Series) -> float:
    """动量（12-1）：过去约12个月累计收益，跳过最近21个交易日。"""
    if len(asset_ret) < 252 + 21:
        return float("nan")
    cum = (1.0 + asset_ret).cumprod()
    return float(cum.iloc[-22] / cum.iloc[-253] - 1.0)


def factor_short_term_reversal(asset_ret: pd.Series) -> float:
    """短期反转：最近21个交易日累计收益。"""
    if len(asset_ret) < 21:
        return float("nan")
    return float((1.0 + asset_ret.tail(21)).prod() - 1.0)


def factor_realized_volatility(asset_ret: pd.Series, window: int = 60) -> float:
    """实现波动率：最近60日收益标准差年化。"""
    if len(asset_ret) < window:
        return float("nan")
    return float(asset_ret.tail(window).std(ddof=1) * ANNUALIZE)


def factor_maximum_drawdown(asset_ret: pd.Series) -> float:
    """最大回撤：复利路径相对历史峰值的最大跌幅（负值）。"""
    wealth = (1.0 + asset_ret).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def factor_skewness(asset_ret: pd.Series) -> float:
    return float(asset_ret.skew()) if len(asset_ret) >= 60 else float("nan")


def factor_kurtosis(asset_ret: pd.Series) -> float:
    """超额峰度（正态为0，正值意味尾部更厚）。"""
    return float(asset_ret.kurt()) if len(asset_ret) >= 60 else float("nan")


def factor_trend_strength(asset_ret: pd.Series) -> float:
    """趋势强度：价格在250日均线上方的天数占比（0~1）。"""
    if len(asset_ret) < 260:
        return float("nan")
    price = (1.0 + asset_ret).cumprod()
    ma250 = price.rolling(250).mean()
    valid = price.iloc[250:] > ma250.iloc[250:]
    return float(valid.mean())


# ---------------- 归因层：多元回归拆账 ----------------

def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """带截距 OLS。返回 (betas, r_squared, residuals)。"""
    X1 = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    fitted = X1 @ coef
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef[1:], r2, resid


def factor_momentum_6_1(returns: pd.Series) -> float:
    """动量（6-1）：t-126 到 t-21 的累计收益（跳过最近1个月）。"""
    if len(returns) < 130:
        return float("nan")
    seg = returns.iloc[-126:-21]
    return float((1.0 + seg).prod() - 1.0)


def factor_volume_price_divergence(close: pd.Series, volume: pd.Series | None, window: int = 60) -> float:
    """量价背离：近60日价格变化与成交量变化的相关性；负相关=量价背离。"""
    if volume is None or len(close) < window:
        return float("nan")
    p = close.iloc[-window:].pct_change()
    v = volume.iloc[-window:].pct_change()
    both = pd.concat([p, v], axis=1, keys=["p", "v"]).dropna()
    if len(both) < 30:
        return float("nan")
    corr = both["p"].corr(both["v"])
    return float(corr) if pd.notna(corr) else float("nan")


def factor_volume_activity(volume: pd.Series | None) -> float:
    """量能活跃度：近20日均量 ÷ 近60日均量（换手率的代理指标）。"""
    if volume is None or len(volume.dropna()) < 60:
        return float("nan")
    recent = float(volume.iloc[-20:].mean())
    base = float(volume.iloc[-60:].mean())
    if not (base > 0) or not (recent >= 0):
        return float("nan")
    return float(recent / base)



_bs_extras_cache: dict[str, dict | None] = {}


def fetch_bs_fundamental_extras(symbol: str) -> dict[str, Any] | None:
    """仅 A 股：营收增速 + 股息率（baostock，尽力而为，失败返回 None）。"""
    s = str(symbol).lower()
    if not (s[:2] in ("sh", "sz", "bj") and s[2:].isdigit()):
        return None
    if s in _bs_extras_cache:
        return _bs_extras_cache[s]
    try:
        import baostock as bs
        import datetime as _dt
        code = s[:2] + "." + s[2:]
        lg = bs.login()
        if lg.error_code != "0":
            _bs_extras_cache[s] = None
            return None
        try:
            out: dict[str, Any] = {}
            this_year = _dt.date.today().year
            # 营收增速：近两个完整财年主营业务收入同比
            revs: dict[int, float] = {}
            for y in (this_year - 1, this_year - 2):
                rs = bs.query_profit_data(code=code, year=y, quarter=4)
                while rs.error_code == "0" and rs.next():
                    row = dict(zip(rs.fields, rs.get_row_data()))
                    try:
                        rev = float(row.get("MBRevenue") or "")
                        if rev > 0:
                            revs[y] = rev
                    except Exception:
                        pass
            ys = sorted(revs.keys())
            if len(ys) >= 2 and revs.get(ys[-2]):
                out["revenue_growth"] = round((revs[ys[-1]] / revs[ys[-2]] - 1.0) * 100.0, 1)
            # 股息率：最近年度税前每股股息 ÷ 现价
            try:
                for y in (this_year - 1, this_year - 2):
                    drs = bs.query_dividend_data(code=code, year=str(y), dividFlag="1")
                    cash = 0.0
                    while drs.error_code == "0" and drs.next():
                        drow = dict(zip(drs.fields, drs.get_row_data()))
                        try:
                            cash += float(drow.get("dividendCashPs") or 0)
                        except Exception:
                            pass
                    if cash > 0:
                        start_d = (_dt.date.today() - _dt.timedelta(days=15)).strftime("%Y-%m-%d")
                        end_d = _dt.date.today().strftime("%Y-%m-%d")
                        prs = bs.query_history_k_data_plus(code, "date,close", start_date=start_d,
                                                           end_date=end_d, frequency="d", adjustflag="3")
                        last_close = None
                        while prs.error_code == "0" and prs.next():
                            try:
                                last_close = float(prs.get_row_data()[1])
                            except Exception:
                                pass
                        if last_close:
                            out["dividend_yield"] = round(cash / last_close * 100.0, 2)
                        break
            except Exception:
                pass
            _bs_extras_cache[s] = out or None
            return out or None
        finally:
            bs.logout()
    except Exception:
        _bs_extras_cache[s] = None
        return None


def run_attribution(
    symbol: str,
    years: int = 3,
    attribution_days: int = 63,
) -> dict[str, Any]:
    """核心入口：对资产做多层因子归因。

    步骤（对应 attribution_framework.json）：
    1. 拉资产与基准行情 → 日收益
    2. 东财快照取行业名，拉行业板块指数收益（可得时）
    3. 市场+行业联合回归 → 分层 Beta
    4. 归因区间内收益拆账：市场贡献 + 行业贡献 + 残差
    5. 按 |贡献| 排序输出，附 R² 与因子暴露快照
    """
    frame = fetch_daily(symbol, years=years)
    if frame is None or len(frame) < 30:
        raise ValueError(f"{symbol} 可用K线不足，无法计算因子")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close.index = pd.to_datetime(frame["Date"])
    close = close.dropna()
    asset_ret = close.pct_change().dropna()
    volume = None
    if "Volume" in frame.columns:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        volume.index = pd.to_datetime(frame["Date"])
        volume = volume.reindex(close.index)

    bench_symbol = _benchmark_for(symbol)
    bench_ret = _close_returns(bench_symbol, years=years)
    a, b = _aligned(asset_ret, bench_ret)
    if len(a) < MIN_OBS:
        raise ValueError(f"对齐后可用交易日仅 {len(a)} 天，少于最低要求 {MIN_OBS} 天")
    close = close.reindex(a.index)
    if volume is not None:
        volume = volume.reindex(a.index)

    # 基本面快照与行业数据（失败不阻塞，降级为无行业归因）
    fundamentals = fetch_fundamental_snapshot(symbol)
    prefix = symbol[:2].lower()
    industry_name = (fundamentals or {}).get("industry_name") if prefix in ("sh", "sz", "bj") else None
    if industry_name:
        industry_ret, industry_source = fetch_industry_returns(industry_name, years=years)
    elif prefix == "us":
        industry_ret, industry_source = fetch_us_sector_returns(symbol, years=years)
        if industry_ret is not None:
            industry_name = industry_source.get("industry_name") or industry_source.get("name")
    elif prefix == "hk":
        industry_ret, industry_source = fetch_hk_sector_returns(symbol, years=years)
        if industry_ret is not None:
            industry_name = industry_source.get("industry_name") or industry_source.get("name")
    else:
        industry_ret, industry_source = None, {}

    # 风格因子超额收益（A股中证风格指数 / 美股 MSCI 系 ETF；拉不到自动跳过）
    style_series = fetch_style_returns(prefix, a.index, b, years=years)

    has_industry = industry_ret is not None

    # 回归矩阵：市场 + 行业（如有）+ 风格因子超额收益
    reg_parts: dict[str, pd.Series] = {"asset": a, "bench": b}
    if has_industry:
        reg_parts["industry"] = industry_ret.reindex(a.index)
    for skey, (excess, _name) in style_series.items():
        reg_parts["style_" + skey] = excess.reindex(a.index)

    joined = pd.concat(list(reg_parts.values()), axis=1, keys=list(reg_parts.keys())).dropna()
    if len(joined) < MIN_OBS and style_series:
        # 风格因子导致公共样本不足 → 剔除风格因子重试（降级）
        reg_parts = {"asset": a, "bench": b}
        if has_industry:
            reg_parts["industry"] = industry_ret.reindex(a.index)
        style_series = {}
        joined = pd.concat(list(reg_parts.values()), axis=1, keys=list(reg_parts.keys())).dropna()
    if len(joined) < MIN_OBS:
        raise ValueError(f"对齐后可用交易日仅 {len(joined)} 天，少于最低要求 {MIN_OBS} 天")

    x_cols = [k for k in reg_parts.keys() if k != "asset"]
    a2 = joined["asset"]
    b2 = joined["bench"]
    ind2 = joined["industry"] if (has_industry and "industry" in joined.columns) else None
    betas, r2_full, resid_full = _ols(a2.to_numpy(), joined[x_cols].to_numpy())
    beta_map = dict(zip(x_cols, [float(x) for x in betas]))
    market_beta = beta_map.get("bench", 0.0)
    industry_beta = beta_map.get("industry") if (has_industry and "industry" in beta_map) else None
    win_a = a2.tail(attribution_days)
    win_b = b2.tail(attribution_days)
    win_ind = ind2.tail(attribution_days) if ind2 is not None else None
    win_style = {sk: joined["style_" + sk].tail(attribution_days)
                 for sk in style_series if "style_" + sk in joined.columns}

    idio_vol = float(resid_full.std(ddof=1) * ANNUALIZE)

    # 因子暴露快照
    exposures = {
        "market_beta": market_beta,
        "momentum_12_1": factor_momentum_12_1(a2),
        "short_term_reversal": factor_short_term_reversal(a2),
        "residual_momentum": _residual_momentum(a2, b2),
        "volatility_realized": factor_realized_volatility(a2),
        "idiosyncratic_volatility": idio_vol,
        "maximum_drawdown": factor_maximum_drawdown(a2),
        "skewness": factor_skewness(a2),
        "kurtosis": factor_kurtosis(a2),
        "trend_strength": factor_trend_strength(a2),
        "momentum_6_1": factor_momentum_6_1(a2),
        "volume_price_divergence": factor_volume_price_divergence(close, volume),
        "volume_activity": factor_volume_activity(volume),
    }

    # 归因区间拆账
    period_return = float((1.0 + win_a).prod() - 1.0)
    market_contribution = float(market_beta * win_b.sum())
    industry_contribution = float(industry_beta * win_ind.sum()) if has_industry else None
    explained = market_contribution + (industry_contribution or 0.0)
    residual = period_return - explained

    # ---- 扩展因子贡献拆解 ----
    # 除市场/行业 Beta 外，将残差按风格因子暴露分配
    raw_contribs = [("market_beta", market_contribution)]
    if has_industry:
        raw_contribs.append(("industry_beta", industry_contribution))
    # 风格因子贡献（真实回归系数 × 区间因子变动）
    for skey in style_series:
        col = "style_" + skey
        if col in beta_map and skey in win_style:
            raw_contribs.append((col, float(beta_map[col] * win_style[skey].sum())))

    # 风格因子打分：用暴露值估计各因子对残差的贡献方向和大小
    style_scores: dict[str, float] = {}
    mom_val = exposures.get("momentum_12_1")
    if mom_val is not None and not (isinstance(mom_val, float) and math.isnan(mom_val)):
        style_scores["momentum_12_1"] = mom_val * 0.35
    str_val = exposures.get("short_term_reversal")
    if str_val is not None and not (isinstance(str_val, float) and math.isnan(str_val)):
        style_scores["short_term_reversal"] = str_val * 0.25
    rm_val = exposures.get("residual_momentum")
    if rm_val is not None and not (isinstance(rm_val, float) and math.isnan(rm_val)):
        style_scores["residual_momentum"] = rm_val * 0.20
    vol_val = exposures.get("volatility_realized")
    if vol_val is not None and not (isinstance(vol_val, float) and math.isnan(vol_val)):
        style_scores["volatility_realized"] = -vol_val * 0.10
    ts_val = exposures.get("trend_strength")
    if ts_val is not None and not (isinstance(ts_val, float) and math.isnan(ts_val)):
        style_scores["trend_strength"] = (ts_val - 0.5) * 0.10
    # 风格指数已单独回归的风格，从残差分配里剔除，避免重复归因
    if "momentum" in style_series:
        style_scores.pop("momentum_12_1", None)
        style_scores.pop("residual_momentum", None)
    if "lowvol" in style_series:
        style_scores.pop("volatility_realized", None)

    # 将未解释收益按风格因子得分比例分配
    unexplained = residual
    total_style = sum(abs(v) for v in style_scores.values())
    if total_style > 1e-8 and abs(unexplained) > 1e-8:
        for fid, score in style_scores.items():
            alloc = unexplained * score / total_style
            raw_contribs.append((fid, alloc))

    raw_contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    total_abs = sum(abs(v) for _, v in raw_contribs) + abs(residual)
    contributions = [
        {
            "factor_id": fid,
            "contribution": round(val, 6),
            "share": round(abs(val) / max(total_abs, 1e-9), 4),
            "rank": idx + 1,
        }
        for idx, (fid, val) in enumerate(raw_contribs)
    ]

    # 行业层面描述性因子
    industry_stats: dict[str, Any] = {"available": has_industry}
    if has_industry:
        ind_mom = factor_momentum_12_1(ind2)
        bench_mom = factor_momentum_12_1(b2)
        industry_stats.update({
            "name": industry_name,
            "benchmark_source": industry_source.get("name", industry_source.get("code")),
            "benchmark_kind": industry_source.get("kind"),
            "benchmark_code": industry_source.get("code"),
            "industry_beta": round(industry_beta, 6),
            "industry_momentum_12": None if math.isnan(ind_mom) else round(ind_mom, 6),
            "industry_relative_strength": (None if math.isnan(ind_mom) or math.isnan(bench_mom)
                                           else round(ind_mom - bench_mom, 6)),
        })
    elif industry_name:
        industry_stats.update({
            "name": industry_name,
            "note": industry_source.get("note", "行业基准不可用"),
        })


    # 基本面/规模快照（东财）+ 财报因子（Baostock）
    fundamental_out: dict[str, Any] = {"available": bool(fundamentals)}
    if fundamentals:
        cap = fundamentals.get("total_market_cap")
        fundamental_out.update({
            "name": fundamentals.get("name"),
            "price": fundamentals.get("price"),
            "pe_dynamic": fundamentals.get("pe_dynamic"),
            "total_market_cap": cap,
            "size_bucket": _size_bucket(cap),
            "industry": industry_name,
            "pb": fundamentals.get("pb"),
            "percentile_note": "PE 历史百分位需历史估值序列，当前数据源暂不提供，不编造。",
        })
    bs_fund = fetch_bs_fundamentals(symbol)
    if bs_fund:
        fundamental_out["financials"] = bs_fund
    bs_extras = fetch_bs_fundamental_extras(symbol)
    if bs_extras:
        if bs_extras.get("revenue_growth") is not None:
            fundamental_out["revenue_growth"] = bs_extras["revenue_growth"]
        if bs_extras.get("dividend_yield") is not None:
            fundamental_out["dividend_yield"] = bs_extras["dividend_yield"]

    pending = {}
    if not has_industry:
        pending["industry_beta"] = "not_available: " + industry_source.get("note", "行业基准不可用")
    if not bs_fund:
        pending["quality_roe"] = "not_available: 财报数据源未取到"
        pending["growth_earnings"] = "not_available: 财报数据源未取到"
    # PE 历史百分位（baostock）
    pe_pct = fetch_pe_percentile(symbol)
    if pe_pct:
        fundamental_out["pe_percentile"] = pe_pct
    else:
        pending["value_pe_percentile"] = "not_available: baostock 历史 PE 序列未取到"

    return {
        "engine": "qte_factor_engine",
        "version": "0.2.0",
        "symbol": symbol,
        "benchmark": bench_symbol,
        "window": {
            "full_sample_days": int(len(a2)),
            "attribution_days": int(attribution_days),
            "start": str(win_a.index[0].date()),
            "end": str(win_a.index[-1].date()),
        },
        "period_return": round(period_return, 6),
        "r_squared_full_sample": round(r2_full, 4),
        "contributions": contributions,
        "residual": {
            "value": round(residual, 6),
            "share": round(abs(residual) / max(total_abs, 1e-9), 4),
            "honest_note": "残差为已知因子解释不了的部分；占比高时不得强行归因。",
        },
        "industry": industry_stats,
        "style_sources": {sk: name for sk, (_e, name) in style_series.items()},
        "fundamentals": fundamental_out,
        "exposures": {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 6)) for k, v in exposures.items()},
        "pending": pending,
        "qualifier": "以上为历史统计描述，不预示未来表现，不构成投资建议。",
    }


def _size_bucket(cap: float | None) -> str | None:
    """市值分档（A股常用口径，仅描述，不产生结论）。"""
    if not cap:
        return None
    yi = cap / 1e8
    if yi >= 1000:
        return "超大盘（市值≥1000亿）"
    if yi >= 300:
        return "大盘（300-1000亿）"
    if yi >= 100:
        return "中盘（100-300亿）"
    return "小盘（<100亿）"


def _residual_momentum(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    """残差动量：资产12-1动量减去 Beta×基准同期动量的近似估计。"""
    mom_a = factor_momentum_12_1(asset_ret)
    mom_b = factor_momentum_12_1(bench_ret)
    if math.isnan(mom_a) or math.isnan(mom_b):
        return float("nan")
    betas, _, _ = _ols(asset_ret.to_numpy(), bench_ret.to_numpy().reshape(-1, 1))
    return float(mom_a - betas[0] * mom_b)


# ---------------- CLI 自测 ----------------

if __name__ == "__main__":
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "sh518880"
    result = run_attribution(test_symbol)
    print(json.dumps(result, ensure_ascii=False, indent=2))
