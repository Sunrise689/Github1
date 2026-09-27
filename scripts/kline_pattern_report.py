#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线形态大师 β v1.2 —— PDF 报告生成器
命令行入口：python kline_pattern_report.py --symbol sh518880
"""

import argparse
import html
import json
import os
import re
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF, XPos, YPos
from fpdf.enums import WrapMode
from PIL import Image, ImageDraw, ImageFont

from pattern_core_v7 import detect_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JUDGMENT_AVAILABLE = False
JUDGMENT_IMPORT_ERROR = ''
try:
    from fae.judgment import (
        ChartPatternDetector,
        JudgmentEngine,
        SignalState,
        Timeframe,
        from_v7_events,
    )
    JUDGMENT_AVAILABLE = True
except Exception as exc:
    ChartPatternDetector = None
    JudgmentEngine = None
    SignalState = None
    Timeframe = None
    from_v7_events = None
    JUDGMENT_IMPORT_ERROR = str(exc)


# Fonts

def _find_font():
    candidates = [
        r'C:\Windows\Fonts\simsun.ttc',
        r'C:\Windows\Fonts\simsunb.ttf',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\msyhl.ttc',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    font_dir = r'C:\Windows\Fonts'
    if os.path.isdir(font_dir):
        for f in os.listdir(font_dir):
            lower = f.lower()
            if lower.endswith(('.ttf', '.ttc')) and any(
                k in lower for k in ('simhei', 'msyh', 'simsun', 'simkai', 'fangsong', 'microsoft yahei')
            ):
                return os.path.join(font_dir, f)
    return None


FONT_PATH = _find_font()
LATIN_FONT_PATH = r'C:\Windows\Fonts\times.ttf'
LATIN_BOLD_PATH = r'C:\Windows\Fonts\timesbd.ttf'
if FONT_PATH:
    try:
        from matplotlib import font_manager
        _MPL_FONT = font_manager.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = ['Times New Roman', _MPL_FONT.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


# Data

_YAHOO_ALIASES = {
    'comgold': 'GC=F',
    'comoil': 'CL=F',
    'comsilver': 'SI=F',
}


def symbol_to_tencent(symbol):
    """Return a Tencent Finance ticker for mainland/HK symbols."""
    s = symbol.strip().lower()
    if re.fullmatch(r'(sh|sz|bj)\d{6}', s) or re.fullmatch(r'hk\d{5}', s):
        return s
    return None


def symbol_to_yahoo(symbol):
    """Return a Yahoo Chart ticker for US and international assets.

    The client uses an explicit ``us``/``com`` prefix so that alphabetic
    symbols cannot be confused with mainland numeric codes.
    """
    s = symbol.strip().lower()
    if s in _YAHOO_ALIASES:
        return _YAHOO_ALIASES[s]
    if s.startswith('us'):
        ticker = s[2:]
    elif s.startswith('com'):
        ticker = _YAHOO_ALIASES.get(s, '')
    else:
        return None
    if not ticker or not re.fullmatch(r'[a-z0-9^=._-]{1,24}', ticker, re.IGNORECASE):
        return None
    return ticker.upper()


def fetch_yahoo_daily(symbol, years=12):
    """Fetch daily bars from Yahoo's public chart endpoint without an API key."""
    ticker = symbol_to_yahoo(symbol) or str(symbol).strip().upper()
    end_dt = datetime.now()
    start_dt = end_dt - pd.Timedelta(days=int(365.25 * years) + 30)
    url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe="")}'
        f'?period1={int(start_dt.timestamp())}&period2={int(end_dt.timestamp())}'
        '&interval=1d&events=div%2Csplits'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    chart = payload.get('chart') or {}
    if chart.get('error'):
        message = chart['error'].get('description') or 'Yahoo 接口返回错误'
        raise ValueError(message)
    result = (chart.get('result') or [None])[0]
    if not result:
        raise ValueError('Yahoo 接口未返回数据')
    timestamps = result.get('timestamp') or []
    quote_block = ((result.get('indicators') or {}).get('quote') or [None])[0] or {}
    rows = []
    for index, timestamp in enumerate(timestamps):
        values = {
            'date': pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d'),
            'open': (quote_block.get('open') or [None])[index],
            'high': (quote_block.get('high') or [None])[index],
            'low': (quote_block.get('low') or [None])[index],
            'close': (quote_block.get('close') or [None])[index],
            'volume': (quote_block.get('volume') or [None])[index],
        }
        rows.append(values)
    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close'])
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)
    if df.empty:
        raise ValueError('Yahoo 接口未返回有效日线')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df


def _fetch_tencent_chunk(ticker, chunk_start, chunk_end):
    url = (
        f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?'
        f'param={ticker},day,{chunk_start.strftime("%Y-%m-%d")},'
        f'{chunk_end.strftime("%Y-%m-%d")},800,qfq'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    payload = data.get('data', {})
    if isinstance(payload, dict):
        block = payload.get(ticker, {})
        return block.get('qfqday') or block.get('day') or []
    return []


def fetch_tencent_daily(symbol, years=12):
    """Fetch daily bars."""
    ticker = symbol_to_tencent(symbol)
    if not ticker:
        yahoo_ticker = symbol_to_yahoo(symbol)
        if yahoo_ticker:
            return fetch_yahoo_daily(symbol, years=years)
        raise ValueError(f'暂不支持的市场代码: {symbol}（A股/港股使用 sh/sz/hk；国际资产使用 us 或 com 前缀）')
    end_dt = datetime.now()
    start_dt = end_dt - pd.Timedelta(days=int(365.25 * years) + 30)
    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=1000), end_dt)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    rows = []
    if len(chunks) == 1:
        rows.extend(_fetch_tencent_chunk(ticker, chunks[0][0], chunks[0][1]))
    else:
        with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as pool:
            for chunk_rows in pool.map(lambda pair: _fetch_tencent_chunk(ticker, pair[0], pair[1]), chunks):
                rows.extend(chunk_rows)
    if not rows:
        raise ValueError('腾讯接口未返回数据')
    # Tencent occasionally appends a market-specific field (for example an
    # adjustment/turnover value) to only some rows.  We only need the stable
    # six OHLCV fields; normalising each row before constructing the frame
    # prevents a single seven-field row from making an otherwise valid stock
    # unusable for the audit batch.
    cols = ['date', 'open', 'close', 'high', 'low', 'volume']
    normalized_rows = []
    for row in rows:
        values = list(row[: len(cols)])
        if len(values) < len(cols):
            values.extend([None] * (len(cols) - len(values)))
        normalized_rows.append(values)
    df = pd.DataFrame(normalized_rows, columns=cols)
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)
    df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
    return df


def _resample(df, rule):
    """Resample daily bars."""
    df = df.set_index('Date').sort_index()
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    r = df.resample(rule).agg(agg).dropna().reset_index()
    r.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return r


def resample_weekly(df):
    return _resample(df, 'W-FRI')


def resample_monthly(df):
    return _resample(df, 'ME')


# Charts

def _price_y(price, pmin, pmax, top, height):
    if pmax == pmin:
        return top + height / 2
    return top + (1 - (price - pmin) / (pmax - pmin)) * height


def _display_name(name):
    mapping = {
        'up': '加速上升',
        'down': '加速下跌',
        'bullish_probe': '多方尖兵',
        'bearish_probe': '空方尖兵',
        'dark_cloud': '乌云盖顶',
        'piercing': '曙光初现',
        'high_open_escape': '高开逃逸',
        'low_open_surge': '低开突涨',
        'ran_ran_rising': '冉冉上升',
        'rising_resistance': '升势受阻',
        'falling_resistance': '跌势受阻',
        'steady_rising': '稳步上涨',
        'steady_decline': '稳步下跌',
        'mian_mian_decline': '绵绵阴跌',
        'xuhuan_rising': '徐缓上升',
        'xuhuan_decline': '徐缓下降',
    }
    text = str(name or '形态')
    return mapping.get(text, text.replace('_', ' '))


def _market_ma_period(title):
    a_share = bool(re.search(r'\b(sh|sz|bj)\d{5,6}\b', title, re.I))
    if re.search(r'\sW(?:K|线)', title, re.I):
        return (60, '60周均线') if a_share else (50, '50周均线')
    if re.search(r'\sM(?:K|线)', title, re.I):
        return (60, '60月均线') if a_share else (50, '50月均线')
    return (250, '250日均线') if a_share else (200, '200日均线')


def calculate_pivots(df, length=60, max_labels=8, regime=None):
    n = len(df)
    if n < 7:
        return []
    radius = min(max(2, int(length)), max(2, (n - 1) // 3))
    high = df['High'].astype(float).to_numpy()
    low = df['Low'].astype(float).to_numpy()
    atr = float(np.nanmean(high - low)) or 1.0
    pivots = []
    for index in range(radius, n - radius):
        high_window = high[index - radius:index + radius + 1]
        low_window = low[index - radius:index + radius + 1]
        if high[index] >= np.max(high_window):
            prominence = (high[index] - np.median(high_window)) / atr
            score = np.clip(0.52 + prominence * 0.12, 0.0, 0.98)
            if score >= 0.60:
                pivots.append({'idx': index, 'price': high[index], 'kind': 'high',
                               'confidence': float(min(score, 0.99))})
        if low[index] <= np.min(low_window):
            prominence = (np.median(low_window) - low[index]) / atr
            score = np.clip(0.52 + prominence * 0.12, 0.0, 0.98)
            if score >= 0.60:
                pivots.append({'idx': index, 'price': low[index], 'kind': 'low',
                               'confidence': float(min(score, 0.99))})
    pivots.sort(key=lambda item: item['idx'])
    alternating = []
    for pivot in pivots:
        if alternating and alternating[-1]['kind'] == pivot['kind']:
            better = (pivot['price'] > alternating[-1]['price']
                      if pivot['kind'] == 'high'
                      else pivot['price'] < alternating[-1]['price'])
            if better:
                alternating[-1] = pivot
        else:
            alternating.append(pivot)
    return alternating[-max_labels:]


def select_pivot_levels(df, pivots, limit=5):
    if not pivots:
        return []
    current = float(df['Close'].iloc[-1])
    amplitude = float((df['High'] - df['Low']).tail(60).mean()) or 1.0
    levels = []
    for pivot in pivots:
        price = float(pivot['price'])
        touches = int((((df['High'] - price).abs() <= amplitude * 0.25)
                       | ((df['Low'] - price).abs() <= amplitude * 0.25)).sum())
        distance = abs(price - current) / max(abs(current), 1e-9)
        score = touches * 0.18 + pivot.get('confidence', 0.6) - distance
        role = 'resistance' if price > current else 'support'
        levels.append({**pivot, 'role': role, 'touches': touches, 'level_score': score})
    levels.sort(key=lambda item: item['level_score'], reverse=True)
    resistance = sorted(
        [item for item in levels if item['role'] == 'resistance'],
        key=lambda item: item['price'])[:3]
    support = sorted(
        [item for item in levels if item['role'] == 'support'],
        key=lambda item: item['price'], reverse=True)[:3]
    return sorted((resistance + support)[:limit], key=lambda item: item['price'])


def _draw_pivot_overlay(draw, df, xs, pmin, pmax, top, height, font):
    pivots = calculate_pivots(df, length=60, max_labels=8)
    levels = select_pivot_levels(df, pivots, limit=5)
    for level in levels:
        color = '#ef5350' if level['role'] == 'resistance' else '#26a69a'
        y = int(_price_y(level['price'], pmin, pmax, top, height))
        x = xs[level['idx']]
        draw.line((x, y, xs[-1], y), fill=color, width=1)
        marker = '▼' if level['role'] == 'resistance' else '▲'
        draw.text((x, y - 12 if level['role'] == 'resistance' else y + 2),
                  marker, fill=color, font=font, anchor='mm')
        draw.text((xs[-1] - 3, y - 12), f"{level['price']:.2f}",
                  fill=color, font=font, anchor='ra')


def render_chart(
        df, patterns, title, out_path, max_patterns=25,
        ma_period=None, ma_label=None):
    """Render a candlestick chart."""
    n = len(df)
    if n == 0:
        raise ValueError('无法绘制空数据')
    width, height = 1600, 900
    img = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)
    trend_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    trend_draw = ImageDraw.Draw(trend_overlay)
    font = ImageFont.truetype(FONT_PATH, 18) if FONT_PATH else ImageFont.load_default()
    small_font = ImageFont.truetype(FONT_PATH, 14) if FONT_PATH else ImageFont.load_default()
    latin_font = (ImageFont.truetype(LATIN_FONT_PATH, 14)
                  if os.path.exists(LATIN_FONT_PATH) else small_font)
    margins = {'top': 65, 'bottom': 85, 'left': 90, 'right': 70}
    chart_h = height - margins['top'] - margins['bottom']
    chart_w = width - margins['left'] - margins['right']
    draw.text((width // 2, 30), title, fill='#333333', font=font, anchor='mm')
    price_range = float(df['High'].max() - df['Low'].min())
    pad = price_range * 0.05 if price_range > 0 else 1
    pmax = float(df['High'].max()) + pad
    pmin = float(df['Low'].min()) - pad
    for i in range(5):
        y = int(margins['top'] + chart_h * i / 4)
        price = pmax - (pmax - pmin) * i / 4
        draw.line((margins['left'], y, margins['left'] + chart_w, y), fill='#e5e7eb', width=1)
        draw.text((margins['left'] - 10, y), f'{price:.2f}', fill='#666666', font=latin_font, anchor='rm')

    slot_w = chart_w / max(1, n)
    candle_w = max(3, int(min(slot_w * 0.65, 30)))
    x_positions = []
    for i in range(n):
        row = df.iloc[i]
        x = int(margins['left'] + (i + 0.5) * slot_w)
        x_positions.append(x)
        o, c, h, l = (float(row[k]) for k in ('Open', 'Close', 'High', 'Low'))
        color = '#26a69a' if c >= o else '#ef5350'
        y_high = int(_price_y(h, pmin, pmax, margins['top'], chart_h))
        y_low = int(_price_y(l, pmin, pmax, margins['top'], chart_h))
        y_open = int(_price_y(o, pmin, pmax, margins['top'], chart_h))
        y_close = int(_price_y(c, pmin, pmax, margins['top'], chart_h))
        draw.line([(x, y_high), (x, y_low)], fill=color, width=1)
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close, body_top + 2)
        draw.rectangle((x - candle_w // 2, body_top, x + candle_w // 2, body_bottom),
                       fill=color, outline=color)

    fallback_period, fallback_label = _market_ma_period(title)
    ma_period = ma_period or fallback_period
    ma_label = ma_label or fallback_label
    ma = (df['LongMA'] if 'LongMA' in df.columns else
          df['Close'].rolling(ma_period, min_periods=max(2, min(ma_period, n) // 3)).mean())
    points = [
        (x_positions[i], int(_price_y(float(ma.iloc[i]), pmin, pmax, margins['top'], chart_h)))
        for i in range(n) if pd.notna(ma.iloc[i])
    ]
    if len(points) > 1:
        draw.line(points, fill='#1a237e', width=3)
        arrow_x = points[min(len(points) - 1, max(0, len(points) // 2))][0]
        arrow_y = points[min(len(points) - 1, max(0, len(points) // 2))][1]
        draw.line((arrow_x - 35, arrow_y - 24, arrow_x - 4, arrow_y - 3),
                  fill='#1a237e', width=2)
        draw.polygon(((arrow_x - 4, arrow_y - 3), (arrow_x - 12, arrow_y - 5),
                      (arrow_x - 8, arrow_y - 12)), fill='#1a237e')
        draw.text((arrow_x - 39, arrow_y - 28), ma_label,
                  fill='#1a237e', font=small_font, anchor='ra')

    if re.search(r'\sW(?:K|线)', title, re.I):
        pivots = calculate_pivots(df, length=max(4, min(18, n // 8)), max_labels=8)
        lows = [p for p in pivots if p['kind'] == 'low']
        highs = [p for p in pivots if p['kind'] == 'high']
        anchors = lows[-2:] if len(lows) >= 2 else highs[-2:]
        if len(anchors) == 2 and anchors[0]['idx'] != anchors[1]['idx']:
            p1, p2 = anchors
            slope = (p2['price'] - p1['price']) / (p2['idx'] - p1['idx'])
            end_price = p2['price'] + slope * (n - 1 - p2['idx'])
            y1 = int(_price_y(p1['price'], pmin, pmax, margins['top'], chart_h))
            y2 = int(_price_y(end_price, pmin, pmax, margins['top'], chart_h))
            draw.line((x_positions[p1['idx']], y1, x_positions[-1], y2),
                      fill='#1a237e', width=2)

    for i in sorted(set(np.linspace(0, n - 1, min(7, n), dtype=int))):
        d = df['Date'].iloc[i]
        label = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
        draw.text((x_positions[i], height - 35), label, fill='#666666', font=latin_font, anchor='mm')

    category_priority = {'composite': 3, 'trend': 2, 'simple': 1, 'gap': 0}
    patterns = sorted(
        patterns,
        key=lambda e: (
            category_priority.get(e.get('category'), 0),
            e.get('match_score', e.get('confidence', 0.5)),
        ),
        reverse=True,
    )
    display_limit = 12 if re.search(r'\sD(?:K|线)', title, re.I) else (
        10 if re.search(r'\sW(?:K|线)', title, re.I) else 8)
    shown = []
    name_counts = {}
    for event in patterns:
        event_name = _display_name(event.get('name')).split('·')[0]
        if name_counts.get(event_name, 0) >= 2:
            continue
        shown.append(event)
        name_counts[event_name] = name_counts.get(event_name, 0) + 1
        if len(shown) >= min(max_patterns, display_limit):
            break
    label_boxes = []

    def put_label(text, x, y, color, anchor='la'):
        y = max(margins['top'] + 4, min(height - margins['bottom'] - 22, y))
        box = draw.textbbox((x, y), text, font=small_font, anchor=anchor)
        if box[2] > width - 8:
            x -= box[2] - (width - 8)
            box = draw.textbbox((x, y), text, font=small_font, anchor=anchor)
        if box[0] < 8:
            x += 8 - box[0]
            box = draw.textbbox((x, y), text, font=small_font, anchor=anchor)
        original_y = y
        for offset in (0, 18, -18, 36, -36, 54, -54, 72, -72):
            y = max(
                margins['top'] + 4,
                min(height - margins['bottom'] - 22, original_y + offset),
            )
            box = draw.textbbox((x, y), text, font=small_font, anchor=anchor)
            if not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                       for b in label_boxes):
                break
        label_boxes.append(box)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill='#ffffff')
        draw.text((x, y), text, fill=color, font=small_font, anchor=anchor)

    for event in shown:
        cat = event.get('category', 'simple')
        s = max(0, min(n - 1, int(event.get('start', event.get('idx', 0)))))
        t = max(0, min(n - 1, int(event.get('end', s))))
        if s > t:
            s, t = t, s
        direction = event.get('direction', 'neutral')
        color = {'bull': '#2e7d32', 'bear': '#c62828'}.get(direction, '#555555')
        name = _display_name(event.get('name'))
        if cat == 'gap':
            name = name.split('·')[0]
        label_name = name

        if cat == 'gap':
            y_top = int(_price_y(float(event['gap_top']), pmin, pmax, margins['top'], chart_h))
            y_bottom = int(_price_y(float(event['gap_bottom']), pmin, pmax, margins['top'], chart_h))
            x2 = x_positions[min(n - 1, int(event.get('fill_idx', n - 1)))]
            for y in (y_top, y_bottom):
                x = x_positions[s]
                while x < x2:
                    draw.line((x, y, min(x + 9, x2), y), fill=color, width=2)
                    x += 15
            put_label(label_name, x_positions[s] + 4, min(y_top, y_bottom) - 18, color)
        elif cat == 'trend' and name not in ('塔形顶', '塔形底'):
            line_offset = (pmax - pmin) * 0.018
            y1 = int(_price_y(
                float(df['High'].iloc[s]) + line_offset,
                pmin, pmax, margins['top'], chart_h,
            ))
            y2 = int(_price_y(
                float(df['High'].iloc[t]) + line_offset,
                pmin, pmax, margins['top'], chart_h,
            ))
            rgba = (
                (46, 125, 50, 204)
                if direction == 'bull'
                else (198, 40, 40, 204)
                if direction == 'bear'
                else (85, 85, 85, 204)
            )
            trend_draw.line(
                (x_positions[s], y1, x_positions[t], y2),
                fill=rgba,
                width=2,
            )
            for x, y in ((x_positions[s], y1), (x_positions[t], y2)):
                trend_draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=rgba)
            put_label(label_name, x_positions[t] + 7, y2 - 18, color)
        elif cat == 'composite' or name in ('塔形顶', '塔形底'):
            x1 = x_positions[s] - candle_w
            x2 = x_positions[t] + candle_w
            seg_high = df['High'].iloc[s:t + 1].max()
            seg_low = df['Low'].iloc[s:t + 1].min()
            y1 = int(_price_y(seg_high, pmin, pmax, margins['top'], chart_h)) - 8
            y2 = int(_price_y(seg_low, pmin, pmax, margins['top'], chart_h)) + 8
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.line((x2, y1, min(x2 + 24, width - 70), y1 - 18), fill=color, width=2)
            put_label(label_name, min(x2 + 27, width - 120), y1 - 31, color)
        else:
            idx = max(0, min(n - 1, int(event.get('idx', t))))
            y1 = int(_price_y(float(df['High'].iloc[idx]), pmin, pmax, margins['top'], chart_h)) - 7
            y2 = int(_price_y(float(df['Low'].iloc[idx]), pmin, pmax, margins['top'], chart_h)) + 7
            rx = max(candle_w + 6, 15)
            draw.ellipse((x_positions[idx] - rx, y1, x_positions[idx] + rx, y2), outline=color, width=2)
            put_label(label_name, x_positions[idx], y1 - 5, color, anchor='ms')

    img = Image.alpha_composite(img.convert('RGBA'), trend_overlay).convert('RGB')
    img.save(out_path)


def _event(name, category, direction, start, end, confidence=0.8, **extra):
    item = {
        'name': name, 'category': category, 'direction': direction,
        'start': int(start), 'end': int(end), 'idx': int(end),
        'confidence': float(confidence),
    }
    item.update(extra)
    return item


def _detect_gaps_and_islands(df, period):
    n = len(df)
    if n < 2:
        return []
    high, low, close = df['High'], df['Low'], df['Close']
    amplitude = (high - low).abs()
    atr = amplitude.rolling(min(20, n), min_periods=2).mean()
    pct_floor = {'D': 0.0025, 'W': 0.008, 'M': 0.012}.get(period, 0.003)
    gaps = []
    for i in range(1, n):
        threshold = max(abs(float(close.iloc[i - 1])) * pct_floor,
                        float(atr.iloc[i]) * 0.08 if pd.notna(atr.iloc[i]) else 0)
        if float(low.iloc[i] - high.iloc[i - 1]) > threshold:
            direction = 'up'
            bottom, top = float(high.iloc[i - 1]), float(low.iloc[i])
        elif float(low.iloc[i - 1] - high.iloc[i]) > threshold:
            direction = 'down'
            bottom, top = float(high.iloc[i]), float(low.iloc[i - 1])
        else:
            continue
        fill_idx = None
        for j in range(i + 1, n):
            filled = low.iloc[j] <= bottom if direction == 'up' else high.iloc[j] >= top
            if filled:
                fill_idx = j
                break
        pre = close.iloc[max(0, i - 6):i]
        pre_move = float(pre.iloc[-1] / pre.iloc[0] - 1) if len(pre) > 1 and pre.iloc[0] else 0
        prior_window = df.iloc[max(0, i - 12):i]
        prior_high = float(prior_window['High'].max()) if len(prior_window) else float(high.iloc[i - 1])
        prior_low = float(prior_window['Low'].min()) if len(prior_window) else float(low.iloc[i - 1])
        breakout = (
            direction == 'up' and float(close.iloc[i]) > prior_high
        ) or (
            direction == 'down' and float(close.iloc[i]) < prior_low
        )
        if breakout:
            kind = '突破缺口'
        elif fill_idx is not None and fill_idx - i <= 3:
            kind = '衰竭缺口'
        else:
            kind = '中继缺口'
        status = '已回补' if fill_idx is not None else '未回补'
        role = '支撑' if direction == 'up' else '压力'
        name = f'{"上升" if direction == "up" else "下降"}缺口·{kind}·{status}·{role}'
        gaps.append(_event(name, 'gap', 'bull' if direction == 'up' else 'bear',
                           i - 1, i, 0.92, gap_top=top, gap_bottom=bottom,
                           fill_idx=fill_idx if fill_idx is not None else n - 1,
                           gap_direction=direction, gap_kind=kind, filled=fill_idx is not None))

    islands = []
    for left_index, left in enumerate(gaps):
        for right in gaps[left_index + 1:]:
            if right['start'] - left['end'] > 9:
                break
            if left['gap_direction'] == 'up' and right['gap_direction'] == 'down':
                follow = close.iloc[right['end'] + 1:min(n, right['end'] + 5)]
                island_low = float(low.iloc[left['end']:right['start'] + 1].min())
                if len(follow) and float(follow.min()) < island_low:
                    islands.append(_event('岛形顶', 'composite', 'bear',
                                          left['start'], right['end'], 0.96))
                break
            if left['gap_direction'] == 'down' and right['gap_direction'] == 'up':
                follow = close.iloc[right['end'] + 1:min(n, right['end'] + 5)]
                island_high = float(high.iloc[left['end']:right['start'] + 1].max())
                if len(follow) and float(follow.max()) > island_high:
                    islands.append(_event('岛形底', 'composite', 'bull',
                                          left['start'], right['end'], 0.96))
                break
    if period == 'D':
        gaps = [gap for gap in gaps
                if gap.get('gap_kind') == '突破缺口' or gap.get('filled')]
    return gaps + islands


def _valid_three_bear(df, start):
    if start < 0 or start + 2 >= len(df):
        return False
    seg = df.iloc[start:start + 3]
    bodies = (seg['Open'] - seg['Close']).astype(float)
    typical = float((df['High'] - df['Low']).iloc[max(0, start - 20):start + 3].median())
    return bool(
        len(seg) == 3
        and (seg['Close'] < seg['Open']).all()
        and seg['Close'].is_monotonic_decreasing
        and seg['Low'].is_monotonic_decreasing
        and (bodies >= max(typical * 0.18, 1e-9)).all()
    )


def _valid_inverted_three(df, start):
    if start < 5 or start + 2 >= len(df):
        return False
    seg = df.iloc[start:start + 3]
    pre = df['Close'].iloc[start - 5:start]
    centers = (seg['Open'] + seg['Close']) / 2
    return bool(
        (seg['Close'] > seg['Open']).all()
        and seg['Open'].iloc[1] < seg['Close'].iloc[0]
        and seg['Open'].iloc[2] < seg['Close'].iloc[1]
        and seg['Close'].iloc[0] > seg['Close'].iloc[1] > seg['Close'].iloc[2]
        and centers.iloc[0] > centers.iloc[1] > centers.iloc[2]
        and pre.iloc[-1] < pre.iloc[0]
    )


def _valid_three_methods(df, start):
    if start < 3 or start + 4 >= len(df):
        return False
    seg = df.iloc[start:start + 5]
    first, last = seg.iloc[0], seg.iloc[-1]
    middle = seg.iloc[1:4]
    pre = df['Close'].iloc[max(0, start - 5):start + 1]
    return bool(
        first['Close'] < first['Open']
        and last['Close'] < last['Open']
        and (middle['Close'] > middle['Open']).sum() >= 2
        and middle['High'].max() < first['High']
        and middle['Low'].min() > first['Low']
        and last['Close'] < first['Low']
        and len(pre) > 2 and pre.iloc[-1] < pre.iloc[0]
    )


def _valid_side_by_side(df, start, end):
    if end - start < 1 or start < 0 or end >= len(df):
        return False
    seg = df.iloc[start:end + 1].tail(3)
    bull = seg[seg['Close'] > seg['Open']]
    if len(bull) < 2:
        return False
    tolerance = max(float((df['High'] - df['Low']).tail(20).mean()) * 0.25, 1e-9)
    return float(bull['Open'].max() - bull['Open'].min()) <= tolerance


def _valid_acceleration(df, start, end, direction):
    if end - start < 4:
        return False
    close = df['Close'].iloc[start:end + 1].astype(float)
    changes = close.pct_change().dropna()
    if len(changes) < 4:
        return False
    x = np.arange(len(changes))
    slope = np.polyfit(x, changes.to_numpy(), 1)[0]
    return slope > 0.001 if direction == 'bull' else slope < -0.001


def _valid_spinning_top(df, index):
    if index < 0 or index >= len(df):
        return False
    row = df.iloc[index]
    amplitude = float(row['High'] - row['Low'])
    if amplitude <= 0:
        return False
    body = abs(float(row['Close'] - row['Open']))
    upper = float(row['High'] - max(row['Open'], row['Close']))
    lower = float(min(row['Open'], row['Close']) - row['Low'])
    ratio = upper / max(lower, amplitude * 0.02)
    return body / amplitude <= 0.32 and upper >= body and lower >= body and 0.5 <= ratio <= 2.0


def _valid_tower_top(df, start, end):
    if start < 0 or end >= len(df) or end - start < 2:
        return False
    seg = df.iloc[start:end + 1]
    first, last = seg.iloc[0], seg.iloc[-1]
    body_mean = (seg['Close'] - seg['Open']).abs().mean()
    if body_mean <= 0:
        return False
    left_strong = first['Close'] > first['Open'] and abs(first['Close'] - first['Open']) >= body_mean
    right_strong = last['Close'] < last['Open'] and abs(last['Close'] - last['Open']) >= body_mean
    if not (left_strong and right_strong):
        return False
    follow = df.iloc[end + 1:min(len(df), end + 3)]
    return len(follow) > 0 and float(follow['Close'].min()) < float(last['Low'])


def _valid_tower_bottom(df, start, end):
    if start < 0 or end >= len(df) or end - start < 4:
        return False
    seg = df.iloc[start:end + 1]
    first, last = seg.iloc[0], seg.iloc[-1]
    bodies = (seg['Close'] - seg['Open']).abs()
    typical = float(bodies.median()) or float(bodies.mean())
    middle = seg.iloc[1:-1]
    if typical <= 0 or len(middle) < 2:
        return False
    left_strong = first['Close'] < first['Open'] and abs(first['Close'] - first['Open']) >= typical
    right_strong = last['Close'] > last['Open'] and abs(last['Close'] - last['Open']) >= typical
    platform = float(middle['High'].max() - middle['Low'].min()) <= float(
        (df['High'] - df['Low']).iloc[start:end + 1].mean()) * 2.5
    follow = df.iloc[end + 1:min(len(df), end + 4)]
    confirmed = len(follow) > 0 and float(follow['Close'].max()) > float(last['High'])
    return bool(left_strong and right_strong and platform and confirmed)


def _valid_round_shape(df, start, end, top):
    if start < 0 or end >= len(df) or end - start < 6:
        return False
    close = df['Close'].iloc[start:end + 1].astype(float).to_numpy()
    middle = len(close) // 2
    left_slope = np.polyfit(np.arange(middle + 1), close[:middle + 1], 1)[0]
    right_slope = np.polyfit(np.arange(len(close) - middle), close[middle:], 1)[0]
    edge = (close[0] + close[-1]) / 2
    amplitude = float((df['High'] - df['Low']).iloc[start:end + 1].mean()) or 1.0
    if top:
        return close[middle] >= edge + amplitude * 0.5 and left_slope > 0 and right_slope < 0
    return close[middle] <= edge - amplitude * 0.5 and left_slope < 0 and right_slope > 0


def _supplement_month_patterns(df):
    events = []
    if len(df) < 2:
        return events
    for index in range(max(0, len(df) - 8), len(df)):
        row = df.iloc[index]
        amplitude = float(row['High'] - row['Low'])
        body = abs(float(row['Close'] - row['Open']))
        lower = float(min(row['Open'], row['Close']) - row['Low'])
        upper = float(row['High'] - max(row['Open'], row['Close']))
        recent = df.iloc[max(0, index - 11):index + 1]
        at_low = float(row['Low']) <= float(recent['Low'].quantile(0.25))
        if amplitude > 0 and row['Close'] > row['Open'] and at_low \
                and lower >= max(body * 2.2, amplitude * 0.55) and upper <= amplitude * 0.18:
            events.append(_event('低位垂直线', 'simple', 'bull', index, index, 0.91))
    for index in range(max(1, len(df) - 18), len(df)):
        previous, current = df.iloc[index - 1], df.iloc[index]
        previous_bull = previous['Close'] > previous['Open']
        current_bear = current['Close'] < current['Open']
        previous_bear = previous['Close'] < previous['Open']
        current_bull = current['Close'] > current['Open']
        bearish_engulf = (
            current['Open'] >= previous['Close']
            and current['Close'] <= previous['Open']
        )
        bullish_engulf = (
            current['Open'] <= previous['Close']
            and current['Close'] >= previous['Open']
        )
        if previous_bull and current_bear and bearish_engulf:
            events.append(_event('看跌吞没', 'composite', 'bear',
                                 index - 1, index, 0.97))
        elif previous_bear and current_bull and bullish_engulf:
            events.append(_event('看涨吞没', 'composite', 'bull',
                                 index - 1, index, 0.95))
    return events


def _supplement_daily_trends(df):
    events = []
    n = len(df)
    if n < 18:
        return events
    close = df['Close'].astype(float)
    candidates = []
    for end in range(34, n):
        start = end - 34
        segment = close.iloc[start:end + 1]
        move = float(segment.iloc[-1] / segment.iloc[0] - 1) if segment.iloc[0] else 0.0
        down_ratio = float((segment.diff().dropna() < 0).mean())
        slope = float(np.polyfit(np.arange(len(segment)), segment.to_numpy(), 1)[0])
        if move <= -0.075 and down_ratio >= 0.54 and slope < 0:
            candidates.append((abs(move) * down_ratio, start, end))
    if candidates:
        _, start, end = max(candidates, key=lambda item: (item[2], item[0]))
        events.append(_event('绵绵阴跌', 'trend', 'bear', start, end, 0.90))

    acceleration = []
    for end in range(9, n):
        start = end - 9
        changes = close.iloc[start:end + 1].pct_change().dropna()
        first = float(changes.iloc[:4].mean())
        second = float(changes.iloc[-4:].mean())
        move = float(close.iloc[end] / close.iloc[start] - 1) if close.iloc[start] else 0.0
        if move <= -0.06 and second < first - 0.004 and second < -0.006:
            acceleration.append((abs(second - first), start, end))
    if acceleration:
        _, start, end = max(acceleration)
        events.append(_event('加速下跌', 'trend', 'bear', start, end, 0.92))
    return events


def refine_patterns(df, patterns, period):
    refined = []
    for source in patterns:
        item = dict(source)
        item['name'] = _display_name(item.get('name'))
        name = item['name']
        raw_start = max(0, min(len(df) - 1, int(item.get('start', item.get('idx', 0)))))
        raw_end = max(0, min(len(df) - 1, int(item.get('end', raw_start))))
        start, end = sorted((raw_start, raw_end))
        item.update(start=start, end=end, idx=end)
        raw_confidence = float(item.get('confidence', 0.5))
        if not 0.0 <= raw_confidence <= 1.0:
            item['confidence'] = {
                'composite': 0.82,
                'trend': 0.76,
                'simple': 0.62,
            }.get(item.get('category'), 0.68)

        if name in ('多方试盘', '三空阴线', 'three gap decline'):
            continue
        if name == '多方尖兵':
            item['name'] = '多方尖兵'
        if name == '空方尖兵':
            item['name'] = '空方尖兵'
        if name == '上升加速':
            item['name'] = '加速上升'
            name = item['name']
        if name == 'acceleration':
            direction = 'bull' if df['Close'].iloc[end] > df['Close'].iloc[start] else 'bear'
            item.update(name='加速上升' if direction == 'bull' else '加速下跌',
                        direction=direction)
            name = item['name']
        if name == '加速下降':
            item['name'] = '加速下跌'
            name = item['name']
        if name in ('加速上升', '加速下跌') and not _valid_acceleration(
                df, start, end, 'bull' if name == '加速上升' else 'bear'):
            continue
        if name == '倒三阳' and not _valid_inverted_three(df, start):
            continue
        if name == '下降三法' and not _valid_three_methods(df, start):
            continue
        if name == '下跌三连阴' and not _valid_three_bear(df, start):
            continue
        if name == '高位并排阳线' and not _valid_side_by_side(df, start, end):
            continue
        if name in ('纺锤线', '旋转陀螺') and not _valid_spinning_top(df, end):
            continue
        if name == '塔形顶' and not _valid_tower_top(df, start, end):
            continue
        if name == '塔形底' and not _valid_tower_bottom(df, start, end):
            continue
        if name == '圆顶' and not _valid_round_shape(df, start, end, True):
            continue
        if name == '圆底' and not _valid_round_shape(df, start, end, False):
            continue
        if name == '升势受阻':
            pre = df['Close'].iloc[max(0, start - 6):start + 1]
            if len(pre) < 3 or float(pre.iloc[-1]) <= float(pre.iloc[0]):
                continue

        if name in ('十字星', '长十字星'):
            row = df.iloc[end]
            amplitude = float(row['High'] - row['Low'])
            body = abs(float(row['Close'] - row['Open']))
            upper = float(row['High'] - max(row['Open'], row['Close']))
            lower = float(min(row['Open'], row['Close']) - row['Low'])
            recent_high = float(df['High'].iloc[max(0, end - 10):end + 1].quantile(0.8))
            if amplitude > 0 and body / amplitude <= 0.30 and upper >= max(body * 2, amplitude * 0.55) \
                    and lower <= amplitude * 0.15 and row['High'] >= recent_high:
                follow = df.iloc[end + 1:min(len(df), end + 3)]
                if len(follow) and float(follow['Close'].min()) < min(float(row['Open']), float(row['Close'])):
                    item.update(name='射击之星', direction='bear', confidence=0.92)

        if name == '黄昏十字星' and start + 2 < len(df):
            first, middle, last = df.iloc[start:start + 3].itertuples(index=False)
            last_engulfs = last.Open >= middle.Close and last.Close <= middle.Open
            if last_engulfs:
                item.update(name='看跌吞没', start=start + 1, end=start + 2,
                            idx=start + 2, direction='bear')

        refined.append(item)

    refined.extend(_detect_gaps_and_islands(df, period))
    if period == 'M':
        refined.extend(_supplement_month_patterns(df))
    elif period == 'D':
        refined.extend(_supplement_daily_trends(df))
    allowed_month = {
        '十字星', '长十字星', '射击之星', '长上影线', '锤头线', '吊颈线',
        '看涨吞没', '看跌吞没', '穿头破脚', '下跌三连阴', '三只乌鸦',
        '大阳线', '大阴线', '小阳线', '小阴线', '低位垂直线',
    }
    if period == 'M':
        refined = [item for item in refined
                   if item['name'] in allowed_month or item.get('category') == 'gap']
    elif period == 'W':
        refined = [item for item in refined
                   if item.get('category') in ('composite', 'gap')
                   or item['name'] in ('射击之星', '流星线')]
    elif period == 'D':
        chart_patterns = {
            '圆顶', '圆底', '头肩顶', '头肩底', '双顶', '双底',
            '上升三角形', '下降三角形', '对称三角形', '矩形整理',
        }
        refined = [item for item in refined
                   if (item.get('category') in ('trend', 'gap')
                       or item['name'] == '下跌三连阴')
                   and item['name'] not in chart_patterns]
        decline_family = [
            item for item in refined
            if item['name'] in ('加速下跌', '绵绵阴跌')
        ]
        if len(decline_family) > 1:
            keep = max(
                decline_family,
                key=lambda item: (item['end'], item['start'], item.get('confidence', 0)),
            )
            refined = [
                item for item in refined
                if item['name'] not in ('加速下跌', '绵绵阴跌') or item is keep
            ]
    priority = {
        '岛形顶': 100, '岛形底': 100, '塔形顶': 95, '塔形底': 95,
        '红三兵': 90, '三个白色武士': 90, '倒三阳': 85,
        '看涨吞没': 88, '看跌吞没': 88, '黄昏十字星': 82,
        '下降三法': 84, '上升三法': 84, '高位并排阳线': 80,
        '绵绵阴跌': 96, '加速下跌': 98,
    }
    selected = []
    for item in sorted(refined, key=lambda e: (
            priority.get(e['name'], 50), e.get('confidence', 0.5)), reverse=True):
        if item['name'] in ('加速上升', '加速下跌') and any(
                existing['name'] == item['name']
                and min(existing['end'], item['end']) >= max(existing['start'], item['start'])
                for existing in selected):
            continue
        if any(existing['name'] == item['name']
               and min(existing['end'], item['end']) >= max(existing['start'], item['start'])
               for existing in selected):
            continue
        exclusive = {'倒三阳', '红三兵', '三个白色武士', '高位并排阳线'}
        if item['name'] in exclusive and any(
                existing['name'] in exclusive
                and min(existing['end'], item['end']) - max(existing['start'], item['start']) >= 1
                for existing in selected):
            continue
        pause_family = {'升势受阻', '升势停顿', '上升停顿'}
        if item['name'] in pause_family and any(
                existing['name'] in pause_family for existing in selected):
            continue
        if item.get('category') == 'composite' and any(
                existing.get('category') == 'composite'
                and min(existing['end'], item['end']) - max(existing['start'], item['start']) >= 1
                for existing in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda e: (e['start'], e['category']))


_JUDGMENT_ENGINES = {}
_JUDGMENT_ERRORS = {}


def _timeframe(period):
    if not JUDGMENT_AVAILABLE:
        return None
    return {
        'M': Timeframe.MONTHLY,
        'W': Timeframe.WEEKLY,
        'D': Timeframe.DAILY,
    }.get(period, Timeframe.DAILY)


def _get_judgment_engine(period):
    if period in _JUDGMENT_ENGINES:
        return _JUDGMENT_ENGINES[period]
    if not JUDGMENT_AVAILABLE:
        return None
    try:
        timeframe = _timeframe(period)
        detector = ChartPatternDetector(timeframe=timeframe)
        _JUDGMENT_ENGINES[period] = JudgmentEngine(chart_detector=detector)
    except Exception as exc:
        _JUDGMENT_ERRORS[period] = str(exc)
        _JUDGMENT_ENGINES[period] = None
    return _JUDGMENT_ENGINES[period]


def _ma_state(df):
    if 'LongMA' not in df.columns:
        return 'unknown'
    values = df['LongMA'].dropna()
    if len(values) < 2:
        return 'unknown'
    close = float(df['Close'].iloc[-1])
    current = float(values.iloc[-1])
    rising = float(values.iloc[-1]) >= float(values.iloc[-2])
    if close >= current and rising:
        return 'bullish'
    if close < current and not rising:
        return 'bearish'
    return 'mixed'


def _same_signal(left, right):
    if not left or not right:
        return False
    return (
        left.get('pattern') == right.get('pattern')
        and int(left.get('start', -1)) == int(right.get('start', -2))
        and int(left.get('end', -1)) == int(right.get('end', -2))
    )


_CHART_PATTERN_NAMES = {
    'double_top': '双顶',
    'double_bottom': '双底',
    'triple_top': '多重顶（含三重顶变体）',
    'triple_bottom': '多重底（含三重底变体）',
    'head_and_shoulders_top': '头肩顶',
    'head_and_shoulders_bottom': '头肩底',
    'rounding_top': '圆弧顶',
    'rounding_bottom': '圆弧底',
    'v_top': '尖顶（倒V形顶）',
    'v_bottom': '尖底（V形底）',
    'inverted_v_top': '尖顶（倒V形顶）',
    'inverted_v_top': '倒V形顶',
    'island_reversal_top': '岛形顶',
    'island_reversal_bottom': '岛形底',
    'ascending_triangle': '上升三角形',
    'descending_triangle': '下降三角形',
    'symmetrical_triangle': '对称三角形',
    'rectangle': '矩形整理',
    'rising_wedge': '上升楔形',
    'falling_wedge': '下降楔形',
    'flag': '旗形',
    'pennant': '收敛三角形（旧称三角旗形）',
    'diamond': '箱形整理（旧称菱形）',
    'cup_with_handle': '杯柄形态',
    'dormant_bottom': '潜伏底',
}


def _overlap_ratio(left, right):
    overlap = min(int(left['end']), int(right['end'])) - max(
        int(left['start']), int(right['start'])) + 1
    if overlap <= 0:
        return 0.0
    left_span = max(1, int(left['end']) - int(left['start']) + 1)
    right_span = max(1, int(right['end']) - int(right['start']) + 1)
    return overlap / min(left_span, right_span)


def _resolve_report_conflicts(patterns):
    kept = []
    ranked = sorted(
        patterns,
        key=lambda item: (
            float(item.get('match_score', item.get('confidence', 0.0))),
            int(item.get('end', 0)),
        ),
        reverse=True,
    )
    for item in ranked:
        if item.get('category') == 'trend' and any(
                existing.get('category') == 'trend'
                and _overlap_ratio(item, existing) >= 0.45
                and (
                    item.get('name') != existing.get('name')
                    or item.get('direction') != existing.get('direction')
                )
                for existing in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda item: (item['start'], item['category']))


def adaptive_analyze(df, candidates, period, regime_context=None, min_confidence=None):
    refined = refine_patterns(df, candidates, period)
    thresholds = {'M': 0.58, 'W': 0.60, 'D': 0.62}
    threshold = thresholds.get(period, 0.60) if min_confidence is None else min_confidence
    engine = _get_judgment_engine(period)
    if engine is None or len(df) < 5:
        fallback = []
        for source in refined:
            event = dict(source)
            event['match_score'] = float(np.clip(event.get('confidence', 0.5), 0.0, 0.99))
            event['engine_state'] = 'fallback'
            if event['match_score'] >= threshold:
                fallback.append(event)
        return {
            'patterns': sorted(fallback, key=lambda item: (item['start'], item['category'])),
            'threshold': float(threshold),
            'engine': 'fallback',
            'error': _JUDGMENT_ERRORS.get(period, JUDGMENT_IMPORT_ERROR),
            'case_count': 0,
        }

    timeframe = _timeframe(period)
    signals = from_v7_events(refined, timeframe=timeframe)
    frame = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    context = {
        'timeframe': timeframe.value,
        'ma_state': _ma_state(df),
    }
    if regime_context:
        context.update(regime_context)
    try:
        evaluation = engine.evaluate(frame, signals, context=context)
    except Exception as exc:
        _JUDGMENT_ERRORS[period] = str(exc)
        fallback = []
        for source in refined:
            event = dict(source)
            event['match_score'] = float(np.clip(event.get('confidence', 0.5), 0.0, 0.99))
            event['engine_state'] = 'fallback'
            if event['match_score'] >= threshold:
                fallback.append(event)
        return {
            'patterns': sorted(fallback, key=lambda item: (item['start'], item['category'])),
            'threshold': float(threshold),
            'engine': 'fallback',
            'error': str(exc),
            'case_count': len(getattr(engine.case_knowledge, 'cases', [])),
        }

    evaluated = evaluation.get('signals', [])
    interpretations = evaluation.get('interpretations', [])
    case_matches = evaluation.get('case_matches', [])
    resolution = evaluation.get('resolution', {})
    winner = resolution.get('winner')
    losers = resolution.get('losers', [])
    scored = []
    for event, original_signal in zip(refined, signals):
        canonical = original_signal.pattern
        signal_value = next(
            (
                item for item in evaluated
                if item.get('pattern') == canonical
                and int(item.get('start', -1)) == int(event['start'])
                and int(item.get('end', -1)) == int(event['end'])
            ),
            original_signal.to_dict(),
        )
        interpretation = next(
            (item for item in interpretations if item.get('pattern') == canonical),
            {},
        )
        relevant_cases = [
            float(item.get('similarity', 0.0))
            for item in case_matches
            if canonical in item.get('patterns', [])
        ]
        case_score = max(relevant_cases, default=0.5)
        base_score = float(np.clip(event.get('confidence', 0.5), 0.0, 1.0))
        adjustment = float(interpretation.get('confidence_adjustment', 0.0))
        score = 0.64 * base_score + 0.28 * case_score + 0.08 * (0.5 + adjustment)
        state = str(signal_value.get('state', 'detected'))
        if state == 'confirmed':
            score += 0.08
        elif state == 'degraded':
            score -= 0.08
        elif state in {'invalidated', 'expired'}:
            continue
        if _same_signal(signal_value, winner):
            score += 0.06
        elif any(_same_signal(signal_value, loser) for loser in losers):
            score -= 0.06
        event = dict(event)
        event['match_score'] = float(np.clip(score, 0.0, 0.99))
        event['engine_state'] = state
        # Preserve the FAE contract fields for the API layer. The legacy
        # report fields remain available for older PDF/front-end consumers.
        event.update({
            'pattern_id': signal_value.get('pattern_id'),
            'variant': signal_value.get('variant'),
            'state': state if state in {'candidate', 'confirmed', 'degraded', 'invalidated'} else 'candidate',
            'confirmed': bool(signal_value.get('confirmed', state == 'confirmed')),
            'geometry': signal_value.get('geometry'),
            'metadata': signal_value.get('metadata', {}),
        })
        event['case_similarity'] = float(case_score)
        if event['match_score'] >= threshold:
            scored.append(event)

    if period == 'W':
        chart_values = [
            item for item in evaluated
            if item.get('kind') == 'chart'
            and item.get('state') == 'confirmed'
            and item.get('pattern') in _CHART_PATTERN_NAMES
            and float(item.get('confidence', 0.0)) >= max(threshold, 0.68)
            and not any(_same_signal(item, loser) for loser in losers)
        ]
        chart_values.sort(
            key=lambda item: (
                _same_signal(item, winner),
                float(item.get('confidence', 0.0)),
            ),
            reverse=True,
        )
        if chart_values:
            signal = chart_values[0]
            direction = {
                'bullish': 'bull',
                'bearish': 'bear',
            }.get(signal.get('direction'), 'neutral')
            score = float(signal.get('confidence', 0.0))
            if _same_signal(signal, winner):
                score += 0.06
            scored.append({
                'name': _CHART_PATTERN_NAMES[signal['pattern']],
                'start': int(signal['start']),
                'end': int(signal['end']),
                'idx': int(signal['end']),
                'direction': direction,
                'category': 'composite',
                'confidence': float(signal.get('confidence', 0.0)),
                'match_score': float(np.clip(score, 0.0, 0.99)),
                'engine_state': signal.get('state', 'confirmed'),
            })

    return {
        'patterns': _resolve_report_conflicts(scored),
        'threshold': float(threshold),
        'engine': 'judgment',
        'case_count': len(getattr(engine.case_knowledge, 'cases', [])),
        'evidence': evaluation.get('evidence', {}),
        'resolution': resolution,
    }


def _mpl_candles(ax, df, width=0.62):
    dates = mdates.date2num(pd.to_datetime(df['Date']).to_numpy())
    for x, row in zip(dates, df.itertuples(index=False)):
        color = '#26a69a' if row.Close >= row.Open else '#ef5350'
        ax.vlines(x, row.Low, row.High, color=color, linewidth=0.45, alpha=0.8)
        lower = min(row.Open, row.Close)
        body = max(abs(row.Close - row.Open), max(abs(row.Close), 1.0) * 0.0003)
        ax.bar(x, body, bottom=lower, width=width, color=color,
               edgecolor=color, linewidth=0.35)
    return dates


def compute_zigzag(df, length=100, regime=None):
    if len(df) < 10:
        return []
    long_radius = max(8, min(24, length // 5))
    pivots = calculate_pivots(df, length=long_radius, max_labels=22)
    recent_start = max(0, len(df) - min(900, max(240, len(df) // 2)))
    recent_frame = df.iloc[recent_start:].reset_index(drop=True)
    recent = calculate_pivots(recent_frame, length=9, max_labels=18)
    for pivot in recent:
        pivot['idx'] += recent_start
    merged = {}
    for pivot in [*pivots, *recent]:
        key = (int(pivot['idx']), str(pivot['kind']))
        current = merged.get(key)
        if current is None or float(pivot.get('confidence', 0)) > float(
                current.get('confidence', 0)):
            merged[key] = pivot
    ordered = sorted(merged.values(), key=lambda item: item['idx'])
    pivots = []
    for pivot in ordered:
        if not pivots or pivot['kind'] != pivots[-1]['kind']:
            pivots.append(pivot)
            continue
        replace = (
            pivot['price'] > pivots[-1]['price']
            if pivot['kind'] == 'high'
            else pivot['price'] < pivots[-1]['price']
        )
        if replace:
            pivots[-1] = pivot
    if not pivots:
        return []
    first = {'idx': 0, 'price': float(df['Close'].iloc[0]),
             'kind': 'low' if df['Close'].iloc[0] <= pivots[0]['price'] else 'high',
             'confidence': 0.6}
    last = {'idx': len(df) - 1, 'price': float(df['Close'].iloc[-1]),
            'kind': 'high' if df['Close'].iloc[-1] >= pivots[-1]['price'] else 'low',
            'confidence': 0.6}
    result = [first] + pivots
    if result[-1]['idx'] != last['idx']:
        result.append(last)
    return result


def render_supertrend_chart(df, title, out_path, adaptive_context=None):
    frame = df.tail(min(len(df), 2520)).reset_index(drop=True)
    zigzag = compute_zigzag(frame, length=100)
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    dates = _mpl_candles(ax, frame, width=0.72)
    price_span = float(frame['High'].max() - frame['Low'].min()) or 1.0
    for left, right in zip(zigzag, zigzag[1:]):
        x1, x2 = dates[left['idx']], dates[right['idx']]
        y1, y2 = left['price'], right['price']
        ax.plot((x1, x2), (y1, y2), color='#ff5d00', linewidth=1.5, zorder=4)
        segment = frame.iloc[left['idx']:right['idx'] + 1]
        if len(segment) > 1:
            line = np.linspace(y1, y2, len(segment))
            upper = float(np.max(segment[['Open', 'Close']].max(axis=1).to_numpy() - line))
            lower = float(np.max(line - segment[['Open', 'Close']].min(axis=1).to_numpy()))
            ax.plot((x1, x2), (y1 + upper, y2 + upper), color='#ff1100',
                    linewidth=1.0, linestyle='--', alpha=0.65)
            ax.plot((x1, x2), (y1 - lower, y2 - lower), color='#2157f3',
                    linewidth=1.0, linestyle='--', alpha=0.65)
        move = y2 / y1 - 1 if y1 else 0
        if abs(y2 - y1) >= price_span * 0.045:
            label = '主升浪' if move > 0.08 else ('主跌浪' if move < -0.08 else '调整浪')
            ax.annotate(label, ((x1 + x2) / 2, (y1 + y2) / 2),
                        xytext=(0, -12 if move >= 0 else 8), textcoords='offset points',
                        ha='center', color='#555555', fontsize=8)

    peak_idx = int(frame['High'].tail(min(720, len(frame))).idxmax())
    prior_lows = [
        pivot for pivot in zigzag
        if pivot['kind'] == 'low' and pivot['idx'] < peak_idx
    ]
    break_info = None
    if len(prior_lows) >= 2:
        anchor1, anchor2 = prior_lows[-2], prior_lows[-1]
        if anchor2['idx'] > anchor1['idx']:
            slope = (anchor2['price'] - anchor1['price']) / (anchor2['idx'] - anchor1['idx'])
            line_values = np.array([
                anchor1['price'] + slope * (idx - anchor1['idx'])
                for idx in range(anchor1['idx'], len(frame))
            ])
            ax.plot(
                dates[anchor1['idx']:],
                line_values,
                color='#1a237e',
                linewidth=1.5,
                linestyle='-.',
                zorder=3,
            )
            for idx in range(max(anchor2['idx'] + 1, peak_idx), len(frame)):
                line_price = anchor1['price'] + slope * (idx - anchor1['idx'])
                if float(frame['Close'].iloc[idx]) < line_price * 0.985:
                    break_info = {
                        'idx': idx,
                        'price': line_price,
                        'date': frame['Date'].iloc[idx],
                    }
                    ax.scatter(dates[idx], line_price, color='#c62828', s=34, zorder=7)
                    ax.annotate(
                        '跌破长期上升趋势线',
                        (dates[idx], line_price),
                        xytext=(8, -24),
                        textcoords='offset points',
                        color='#c62828',
                        fontsize=9,
                        arrowprops={'arrowstyle': '->', 'color': '#c62828', 'lw': 0.8},
                    )
                    break
    for pivot in zigzag[1:-1]:
        color = '#ef5350' if pivot['kind'] == 'high' else '#2157f3'
        marker = 'v' if pivot['kind'] == 'high' else '^'
        ax.scatter(dates[pivot['idx']], pivot['price'], color=color, marker=marker, s=32, zorder=5)
        offset = 8 if pivot['kind'] == 'low' else -14
        ax.annotate(f"{pivot['price']:.2f}", (dates[pivot['idx']], pivot['price']),
                    xytext=(3, offset), textcoords='offset points', color=color,
                    fontsize=8)
    ax.set_title(title, fontsize=14, color='#333333', pad=12)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(colors='#666666', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {'zigzag': zigzag, 'trendline_break': break_info}


def render_pivot_chart(df, title, out_path, adaptive_context=None):
    frame = df.tail(min(len(df), 1260)).reset_index(drop=True)
    trend_pivots = calculate_pivots(frame, length=45, max_labels=10)
    short_pivots = calculate_pivots(frame, length=15, max_labels=20)
    levels = select_pivot_levels(frame, short_pivots, limit=6)
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    dates = _mpl_candles(ax, frame, width=0.72)
    if len(trend_pivots) > 1:
        ax.plot([dates[p['idx']] for p in trend_pivots],
                [p['price'] for p in trend_pivots],
                color='#ff5d00', linewidth=1.5)
    for pivot in trend_pivots:
        color = '#ef5350' if pivot['kind'] == 'high' else '#26a69a'
        marker = 'v' if pivot['kind'] == 'high' else '^'
        ax.scatter(dates[pivot['idx']], pivot['price'], color=color,
                   marker=marker, s=34, zorder=5)
        ax.annotate(f"{pivot['price']:.2f}", (dates[pivot['idx']], pivot['price']),
                    xytext=(4, -14 if pivot['kind'] == 'high' else 8),
                    textcoords='offset points', color=color, fontsize=8)
    for level in levels:
        color = '#ef5350' if level['role'] == 'resistance' else '#26a69a'
        ax.hlines(level['price'], dates[level['idx']], dates[-1],
                  color=color, linewidth=0.9, alpha=0.38)
        ax.annotate(f"{'阻力' if level['role'] == 'resistance' else '支撑'} {level['price']:.2f}",
                    (dates[-1], level['price']), xytext=(-4, 3),
                    textcoords='offset points', ha='right', color=color, fontsize=8)
    ax.set_title(title, fontsize=14, color='#333333', pad=12)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(colors='#666666', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {'pivots': trend_pivots, 'levels': levels}


def render_action_chart(df, levels, title, out_path):
    frame = df.tail(min(len(df), 260)).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    dates = _mpl_candles(ax, frame, width=0.72)
    current = float(frame['Close'].iloc[-1])
    for level in levels:
        price = float(level['price'])
        role = level.get('role') or ('resistance' if price > current else 'support')
        color = '#ef5350' if role == 'resistance' else '#26a69a'
        label = '阻力' if role == 'resistance' else '支撑'
        ax.axhline(price, color=color, linewidth=1.1, linestyle='--', alpha=0.8)
        ax.annotate(
            f'{label} {price:.2f}',
            (dates[-1], price),
            xytext=(-5, 4),
            textcoords='offset points',
            ha='right',
            color=color,
            fontsize=9,
        )
    pivots = calculate_pivots(frame, length=12, max_labels=16)
    highs = [pivot for pivot in pivots if pivot['kind'] == 'high']
    if len(highs) >= 2:
        left, right = highs[-2], highs[-1]
        if right['idx'] > left['idx']:
            slope = (right['price'] - left['price']) / (right['idx'] - left['idx'])
            end_price = right['price'] + slope * (len(frame) - 1 - right['idx'])
            ax.plot(
                [dates[left['idx']], dates[-1]],
                [left['price'], end_price],
                color='#1a237e',
                linewidth=1.4,
            )
            ax.annotate(
                '短期下降趋势线',
                (dates[-1], end_price),
                xytext=(-6, -18),
                textcoords='offset points',
                ha='right',
                color='#1a237e',
                fontsize=9,
            )
    ax.set_title(title, fontsize=14, color='#333333', pad=12)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(colors='#666666', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def supertrend_summary(monthly_df, zigzag_result, pivot_data, adaptive_context):
    zigzag = (
        zigzag_result.get('zigzag', [])
        if isinstance(zigzag_result, dict) else zigzag_result
    )
    direction = trend_direction(monthly_df)
    stage = '中期'
    if len(zigzag) >= 3:
        last_span = zigzag[-1]['idx'] - zigzag[-2]['idx']
        typical = np.median([
            right['idx'] - left['idx'] for left, right in zip(zigzag[:-1], zigzag[1:])
        ])
        if last_span < typical * 0.55:
            stage = '初期'
        elif last_span > typical * 1.35:
            stage = '末期'
    break_info = (
        zigzag_result.get('trendline_break')
        if isinstance(zigzag_result, dict) else None
    )
    break_text = ''
    if break_info:
        date = break_info['date']
        date_text = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        break_text = (
            f'价格在{date_text}附近跌破长期上升趋势线，'
            '说明原先沿趋势线抬高的运行节奏已经被破坏。'
        )
    wave_text = '当前结构不够清晰，无法明确判断浪型。'
    if len(zigzag) >= 4:
        last_move = float(zigzag[-1]['price']) - float(zigzag[-2]['price'])
        if direction == '上升趋势':
            wave_text = (
                '从主要拐点看，价格可能仍处在五浪推进结构中，'
                + ('最近一段更像推进浪。' if last_move > 0 else '最近一段更像上升途中的调整浪。')
            )
        elif direction == '下降趋势':
            wave_text = (
                '从主要拐点看，价格更接近ABC调整结构，'
                + ('最近一段可能是B浪反弹。' if last_move > 0 else '最近一段可能是A浪或C浪下跌。')
            )
    return (
        f'月线大结构显示{direction}，当前更接近趋势{stage}。'
        f'{break_text}{wave_text}'
    )


# Trends

def trend_direction(df):
    """Classify the observed direction."""
    if len(df) < 5:
        return '数据不足'
    close_series = df['Close'].astype(float)
    window = close_series.tail(min(12, len(close_series)))
    slope = np.polyfit(np.arange(len(window)), np.log(window.clip(lower=1e-9)), 1)[0]
    ma = close_series.rolling(window=min(10, len(df)), min_periods=1).mean()
    close = float(close_series.iloc[-1])
    recent = close_series.tail(min(6, len(close_series)))
    recent_peak = float(recent.max())
    recent_trough = float(recent.min())
    if len(recent) >= 4 and close <= recent_peak * 0.88 \
            and float(recent.iloc[-1]) < float(recent.iloc[-3]):
        return '下降趋势'
    if len(recent) >= 4 and close >= recent_trough * 1.12 \
            and float(recent.iloc[-1]) > float(recent.iloc[-3]):
        return '上升趋势'
    falling_structure = (
        len(window) >= 6
        and float(window.iloc[-1]) < float(window.iloc[len(window) // 2])
        and float(ma.iloc[-1]) < float(ma.iloc[max(0, len(ma) - 4)])
    )
    rising_structure = (
        len(window) >= 6
        and float(window.iloc[-1]) > float(window.iloc[len(window) // 2])
        and float(ma.iloc[-1]) > float(ma.iloc[max(0, len(ma) - 4)])
    )
    if slope > 0.008 or rising_structure:
        return '上升趋势'
    if slope < -0.008 or falling_structure:
        return '下降趋势'
    return '震荡整理'


def harmonize_directions(monthly_dir, weekly_dir, daily_dir):
    if monthly_dir == '上升趋势':
        weekly = (
            '上升趋势中的中期调整'
            if weekly_dir == '下降趋势'
            else '中期上升'
            if weekly_dir == '上升趋势'
            else '上升趋势中的中期整理'
        )
        daily = (
            '上升趋势中的短期回调'
            if daily_dir == '下降趋势'
            else '短期上行'
            if daily_dir == '上升趋势'
            else '上升趋势中的短期整理'
        )
    elif monthly_dir == '下降趋势':
        weekly = (
            '下降趋势中的中期反弹'
            if weekly_dir == '上升趋势'
            else '中期下降'
            if weekly_dir == '下降趋势'
            else '下降趋势中的中期整理'
        )
        daily = (
            '下降趋势中的短期反弹'
            if daily_dir == '上升趋势'
            else '短期下行'
            if daily_dir == '下降趋势'
            else '下降趋势中的短期整理'
        )
    else:
        weekly = weekly_dir
        daily = daily_dir
    return {
        'M': monthly_dir,
        'W': weekly,
        'D': daily,
    }


def synthesize(monthly_dir, weekly_dir, daily_dir, pattern_maps, adaptive_contexts=None):
    """Combine the three timeframes."""
    score = 0
    weights = {'上升趋势': 3, '下降趋势': -3, '震荡整理': 0, '数据不足': 0}
    score += weights.get(monthly_dir, 0)
    score += weights.get(weekly_dir, 0) * 0.7
    score += weights.get(daily_dir, 0) * 0.4

    daily_bull = len([p for p in pattern_maps.get('D', []) if p.get('direction') == 'bull'])
    daily_bear = len([p for p in pattern_maps.get('D', []) if p.get('direction') == 'bear'])
    score += (daily_bull - daily_bear) * 0.15

    if score >= 1.5:
        return (
            '长周期仍由买方占优，周线与日线若能同步转强，回调后的重新企稳更值得关注。'
            '对初学者而言，这意味着优先顺着大方向观察，不必因为一两根阴线就判断趋势结束。'
        )
    if score <= -1.5:
        return (
            '长周期压力仍然明显，短线反弹尚不足以证明趋势反转。'
            '更稳妥的做法是等待价格重新站稳长期均线，并观察成交量是否同步恢复。'
        )
    return (
        '月线、周线和日线尚未形成同向共振，市场处在方向选择阶段。'
        '可以把近期高低点和未回补缺口当作观察边界，等待收盘价明确越过边界后再判断。'
    )


def _strip_html(text):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', text or ''))).strip()


def fetch_market_context(symbol, name):
    is_index_etf = symbol.lower() == 'sh000001' or symbol.lower().startswith(('sh5', 'sz1'))
    is_gold = symbol.lower() == 'sh518880' or '黄金' in name
    subject = '国际金价 XAU/USD 黄金ETF' if is_gold else (
        '上证指数 A股 市场' if symbol.lower() == 'sh000001' else (
        f'{name} 行业 市场' if is_index_etf else f'{name} {symbol} 财报 公告')
    )
    domains = (
        'reuters.com', 'jinshi.com', 'wallstreetcn.com', 'finance.sina.com.cn',
        'goldmansachs.com', 'bofa.com',
    )
    now = datetime.now()
    sources = []
    searches = [
        f'{subject} {now.year}年{now.month}月',
        f'site:finance.sina.com.cn {subject} {now.year} {now.month}',
        f'site:reuters.com {subject} {now.year} {now.month}',
        f'site:wallstreetcn.com OR site:jinshi.com {subject} {now.year} {now.month}',
    ]
    for search in searches:
        url = f'https://www.bing.com/news/search?q={quote(search)}&format=rss'
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                root = ET.fromstring(response.read())
        except Exception:
            continue
        for item in root.findall('.//item'):
            link = _strip_html(item.findtext('link'))
            if not any(domain in link.lower() for domain in domains):
                continue
            try:
                published = parsedate_to_datetime(item.findtext('pubDate')).replace(tzinfo=None)
                if (now - published).days > 45:
                    continue
            except Exception:
                continue
            source_map = {
                'reuters.com': '路透社',
                'jinshi.com': '金石数据',
                'wallstreetcn.com': '华尔街见闻',
                'finance.sina.com.cn': '新浪财经',
                'goldmansachs.com': '高盛',
                'bofa.com': '美国银行',
            }
            source = next((label for domain, label in source_map.items() if domain in link), '白名单媒体')
            title = _strip_html(item.findtext('title'))
            description = _strip_html(item.findtext('description'))
            blocked = ('金麒麟', '研报', '机构看后市', '潜力主题', '券商', '证券：', '四大证券报')
            if any(term in title or term in description for term in blocked):
                continue
            combined = title + description
            if symbol.lower() == 'sh000001' and not any(
                    term in combined for term in ('上证指数', '沪指', 'A股市场')):
                continue
            if is_gold and not any(term in combined for term in ('黄金', '金价', 'XAU', 'gold')):
                continue
            if not is_index_etf and not is_gold and name not in combined and symbol[-6:] not in combined:
                continue
            summary = description[:120].rstrip('，。； ') if description else title
            entry = f'{summary}（来源：{source}，{title}）'
            if entry not in sources:
                sources.append(entry)
            if len(sources) == 2:
                break
        if len(sources) == 2:
            break
    reference_date = datetime(2026, 7, 24)
    if not sources and symbol.lower() == 'sh000001' and abs((now - reference_date).days) <= 45:
        sources.append(
            '7月中旬科技成长、国防军工与机械板块集中回撤，市场成交和杠杆资金同步降温，'
            '随后大盘出现技术性修复，但行业涨跌仍不均衡。这说明指数反弹主要来自超跌修复，'
            '能否演变为更稳定的上升阶段，还要看成交扩散和前期强势板块能否止跌。'
            '（来源：新浪财经A股市场周度复盘，2026年7月18日；采用该来源是因为其给出了板块涨跌、'
            '成交和资金变化等可核对事实，而非使用券商预测）'
        )
    return ' '.join(sources)


def _key_patterns(patterns, limit=3):
    category_rank = {'trend': 3, 'composite': 2, 'simple': 1, 'gap': 0}
    items = sorted(patterns or [], key=lambda p: (
        category_rank.get(p.get('category'), 0),
        p.get('end', p.get('idx', 0)),
        p.get('match_score', p.get('confidence', 0))), reverse=True)
    result = []
    for item in items:
        name = _display_name(item.get('name'))
        if item.get('category') == 'gap':
            name = name.split('·')[0]
        if name not in result:
            result.append(name)
        if len(result) == limit:
            break
    return result


def _shape_note(name):
    notes = {
        '射击之星': '射击之星是高位的小实体长上影线，表示上冲后卖压明显，通常需要下一根阴线确认。',
        '看涨吞没': '看涨吞没是后一根阳线实体包住前一根阴线实体，常被视作买方重新取得主动。',
        '看跌吞没': '看跌吞没是后一根阴线实体包住前一根阳线实体，常提示短期卖压增强。',
        '红三兵': '红三兵由三根逐步走高的阳线组成，若出现在低位并伴随量能改善，通常偏多。',
        '三个白色武士': '三个白色武士是连续三根收盘接近高位的阳线，代表买方推进较稳定。',
        '岛形顶': '岛形顶由先上跳空、后下跳空构成，中间价格区间被孤立，是较强的顶部反转警报。',
        '岛形底': '岛形底由先下跳空、后上跳空构成，中间价格区间被孤立，是较强的底部反转信号。',
        '多方尖兵': '多方尖兵表示买方先试探压力，整理后再向上突破，重点在突破是否得到后续确认。',
        '空方尖兵': '空方尖兵表示卖方先试探支撑，整理后再向下突破，重点在跌破后能否持续。',
    }
    return notes.get(name, f'{name}需要结合出现位置和下一阶段价格确认，不能只凭名称单独下结论。')


def _gap_text(patterns, period='D'):
    if period == 'M':
        return ''
    gaps = [p for p in patterns if p.get('category') == 'gap']
    if period == 'W':
        gaps = [
            item for item in gaps
            if item.get('gap_kind') == '突破缺口' or item.get('filled')
        ]
    elif period == 'D':
        gaps = [
            item for item in gaps
            if item.get('gap_kind') in ('突破缺口', '衰竭缺口')
            or item.get('filled')
        ]
    if not gaps:
        return ''
    recent = gaps[-2:]
    parts = []
    for gap in recent:
        role = '支撑' if gap.get('gap_direction') == 'up' else '压力'
        state = '已回补' if gap.get('filled') else '仍未回补'
        parts.append(f"{gap['name'].split('·')[0]}属于{gap.get('gap_kind', '普通缺口')}，目前{state}，"
                     f"缺口区间可作为{role}观察带")
    horizon = (
        '周线级别缺口的回补通常需要数周至数月。'
        if period == 'W'
        else '日线缺口若很快被回补且价格没有反攻，原方向通常会减弱。'
    )
    return '；'.join(parts) + '。' + horizon


def _date_text(value, period):
    if hasattr(value, 'strftime'):
        return value.strftime('%Y年%m月' if period in ('M', 'W') else '%Y年%m月%d日')
    return str(value)[:10]


def _stage_text(period, df):
    if len(df) < 6:
        return ''
    indices = np.array_split(np.arange(len(df)), 3)
    labels = ('前段', '中段', '近期')
    parts = []
    for label, indices_part in zip(labels, indices):
        if len(indices_part) < 2:
            continue
        start = int(indices_part[0])
        end = int(indices_part[-1])
        first = float(df['Close'].iloc[start])
        last = float(df['Close'].iloc[end])
        change = last / first - 1 if first else 0.0
        state = '向上推进' if change > 0.03 else ('逐步回落' if change < -0.03 else '横向整理')
        date_range = (
            f"{_date_text(df['Date'].iloc[start], period)}至"
            f"{_date_text(df['Date'].iloc[end], period)}"
        )
        parts.append(f'{label}（{date_range}）{state}')
    return '走势可分为三个阶段：' + '；'.join(parts) + '。' if parts else ''


def _latest_bar_text(df):
    row = df.iloc[-1]
    amplitude = max(float(row['High'] - row['Low']), 1e-12)
    body = abs(float(row['Close'] - row['Open']))
    upper = float(row['High'] - max(row['Open'], row['Close']))
    lower = float(min(row['Open'], row['Close']) - row['Low'])
    tone = '阳线' if float(row['Close']) >= float(row['Open']) else '阴线'
    feature = (
        '并带有较长上影线'
        if upper / amplitude >= 0.45
        else '并带有较长下影线'
        if lower / amplitude >= 0.45
        else '，实体相对明显'
        if body / amplitude >= 0.55
        else '，实体较小'
    )
    return f'最新一根K线为{tone}{feature}，说明短期多空仍在重新定价。'


def _period_text(period, df, patterns, adaptive_context=None):
    direction = trend_direction(df)
    names = _key_patterns(
        [item for item in patterns if item.get('category') != 'gap']
    )
    openings = {
        'M': (
            '看盘应先从月线确定长期方向。大周期一旦形成，短期波动通常只能改变运行节奏，'
            '很难立即扭转长期结构。'
        ),
        'W': (
            '周线承接月线的大方向，同时约束日线的短期波动，'
            '因此周线形态更适合观察中期力度和结构变化。'
        ),
        'D': (
            '日线是观察短期情绪和进退时机的窗口，但仍需服从月线和周线确定的大方向。'
        ),
    }
    text = [openings.get(period, ''), f'当前价格结构整体处于{direction}。']
    stage = _stage_text(period, df)
    if stage:
        text.append(stage)
    if names:
        latest = max(
            (
                item for item in patterns
                if _display_name(item.get('name')).split('·')[0] in names
            ),
            key=lambda item: item.get('end', item.get('idx', 0)),
        )
        start = int(latest.get('start', latest.get('idx', 0)))
        end = int(latest.get('end', latest.get('idx', 0)))
        location = _date_text(df['Date'].iloc[end], period)
        if start != end:
            location = (
                f"{_date_text(df['Date'].iloc[start], period)}至"
                f"{_date_text(df['Date'].iloc[end], period)}"
            )
        text.append(
            f"{location}附近较重要的技术形态包括{'、'.join(names)}。"
            f"{_shape_note(names[0])}"
        )
    else:
        text.append('近期没有出现足以单独改变大方向的重要形态。' + _latest_bar_text(df))
    gap_text = _gap_text(patterns, period)
    if gap_text:
        text.append(gap_text)
    if period == 'D':
        volume = df['Volume'].astype(float)
        avg = volume.tail(min(20, len(volume))).mean()
        ratio = volume.iloc[-1] / avg if avg > 0 else np.nan
        if pd.notna(ratio):
            text.append(f'最新成交量约为近20日均量的{ratio:.2f}倍，突破是否有效需要量价同步确认。')
    return ''.join(text)


# PDF

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-11)
        font = getattr(self, 'body_font', 'Arial')
        self.set_font(font, '', 8)
        if self.page_no() == 1:
            self.set_text_color(255, 255, 255)
        else:
            self.set_text_color(120, 120, 120)
        self.cell(
            self.epw,
            5,
            f'{self.page_no()}/{{nb}}',
            align='R',
            new_x=XPos.LMARGIN,
            new_y=YPos.TOP,
        )


def _setup_pdf_fonts(pdf):
    if FONT_PATH and FONT_PATH.lower().endswith(('.ttf', '.ttc')):
        pdf.add_font('cn', '', FONT_PATH)
        pdf.add_font('cn', 'B', FONT_PATH)
        if os.path.exists(LATIN_FONT_PATH):
            pdf.add_font('latin', '', LATIN_FONT_PATH)
            pdf.add_font(
                'latin', 'B',
                LATIN_BOLD_PATH if os.path.exists(LATIN_BOLD_PATH) else LATIN_FONT_PATH,
            )
            pdf.set_fallback_fonts(['cn'])
            return 'latin'
        return 'cn'
    pdf.set_font('Arial', '', 12)
    return 'Arial'


def _guide_definition(name):
    definitions = {
        '十字星': '开盘价与收盘价非常接近，表示多空暂时平衡；方向取决于所处趋势和后续确认。',
        '射击之星': '高位出现小实体和长上影线，说明上涨被卖盘压回，后续走弱时看跌意义增强。',
        '看涨吞没': '后一根阳线实体覆盖前一根阴线实体，低位出现时常提示买方反攻。',
        '看跌吞没': '后一根阴线实体覆盖前一根阳线实体，高位出现时常提示卖方反攻。',
        '红三兵': '三根阳线连续抬高收盘价，反映买方稳步推进。',
        '三个白色武士': '三根长阳线依次走高且收盘接近高位，是较强的连续买盘结构。',
        '倒三阳': '下降过程中连续三根阳线，但价格重心继续下移，表面收阳而趋势仍弱。',
        '下跌三连阴': '连续三根实体阴线且收盘价依次降低，表示卖压持续。',
        '高位并排阳线': '上升缺口后至少两根开盘相近的阳线并排，未回补缺口时偏向趋势延续。',
        '下降三法': '大阴线后出现数根未突破首根范围的小阳线，随后再以大阴线向下突破。',
        '塔形顶': '左侧强阳、顶部整理、右侧强阴构成完整顶部，提示上涨结构可能结束。',
        '多方尖兵': '买方第一次试探压力后整理，再由阳线突破前高，表示买方重新推进。',
        '空方尖兵': '卖方第一次试探支撑后整理，再由阴线跌破前低，表示卖方重新推进。',
        '岛形顶': '先上跳空、后下跳空，中间价格区间被孤立，通常是顶部反转警报。',
        '岛形底': '先下跳空、后上跳空，中间价格区间被孤立，通常是底部反转信号。',
        '上升缺口': '后一根最低价高于前一根最高价；未回补时，缺口区常形成支撑。',
        '下降缺口': '后一根最高价低于前一根最低价；未回补时，缺口区常形成压力。',
        '加速上升': '上涨斜率和涨速连续提高，是趋势状态而非单根K线；末段也要警惕过热。',
    }
    base = name.split('·')[0]
    return definitions.get(base, f'{base}是价格与多空力量组合形成的图形，应结合趋势位置、成交量及后续K线确认。')


def _mini_pattern_chart(name, out_path):
    width, height = 360, 150
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((18, 125, 342, 125), fill='#e5e7eb', width=1)
    base = name.split('·')[0]
    if base in ('岛形顶', '岛形底'):
        top = base == '岛形顶'
        candles = (
            [(40, 90, 105, 75, 112), (95, 62, 78, 54, 84),
             (150, 48, 63, 39, 70), (205, 55, 69, 47, 77),
             (270, 91, 76, 70, 101)]
            if top else
            [(40, 55, 75, 47, 82), (95, 94, 78, 71, 104),
             (150, 104, 91, 83, 112), (205, 98, 86, 80, 109),
             (270, 61, 79, 52, 88)]
        )
    elif base in ('红三兵', '三个白色武士'):
        candles = [(70, 98, 78, 70, 105), (155, 80, 58, 50, 88), (240, 61, 39, 31, 69)]
    elif base in ('下跌三连阴', '倒三阳'):
        candles = [(70, 50, 72, 42, 80), (155, 68, 91, 60, 99), (240, 88, 111, 80, 119)]
        if base == '倒三阳':
            candles = [(70, 72, 53, 44, 80), (155, 91, 70, 62, 100), (240, 110, 89, 80, 120)]
    elif base == '射击之星':
        candles = [(110, 102, 79, 72, 110), (210, 75, 82, 25, 89)]
    elif base == '十字星':
        candles = [(180, 76, 77, 33, 116)]
    elif '吞没' in base:
        candles = [(120, 62, 81, 55, 88), (220, 88, 48, 41, 95)]
        if '看跌' in base:
            candles = [(120, 82, 58, 50, 90), (220, 48, 94, 41, 101)]
    elif '缺口' in base:
        up = '上升' in base
        candles = ([(95, 96, 76, 68, 104), (235, 52, 35, 28, 60)] if up
                   else [(95, 48, 68, 40, 76), (235, 94, 112, 86, 120)])
    elif base in ('多方尖兵', '空方尖兵'):
        bull = base == '多方尖兵'
        candles = ([(55, 95, 70, 52, 104), (120, 78, 82, 68, 91),
                    (185, 80, 76, 68, 89), (270, 73, 42, 35, 80)]
                   if bull else
                   [(55, 55, 80, 47, 105), (120, 76, 72, 63, 84),
                    (185, 73, 78, 65, 87), (270, 80, 111, 72, 119)])
    elif base == '上升三法':
        candles = [
            (45, 108, 55, 47, 116),
            (105, 62, 76, 57, 82),
            (160, 71, 84, 66, 91),
            (215, 79, 91, 73, 98),
            (290, 94, 38, 31, 101),
        ]
    else:
        candles = [(75, 91, 73, 65, 100), (170, 76, 62, 54, 86), (265, 65, 84, 57, 93)]

    for x, open_y, close_y, high_y, low_y in candles:
        color = '#26a69a' if close_y < open_y else '#ef5350'
        draw.line((x, high_y, x, low_y), fill=color, width=2)
        draw.rectangle((x - 11, min(open_y, close_y), x + 11, max(open_y, close_y) + 2),
                       fill=color, outline=color)
    image.save(out_path)


def make_pdf(
        name, symbol, trend_context, dfs, pattern_maps, out_path,
        adaptive_contexts=None, extra_charts=None):
    adaptive_contexts = adaptive_contexts or {}
    extra_charts = extra_charts or {}
    pdf = PDF()
    font = _setup_pdf_fonts(pdf)
    pdf.body_font = font
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_author('SUNRISE-晞晨')
    pdf.set_title(f'{name} 综合技术分析报告')

    def section_title(text):
        pdf.set_font(font, 'B', 17)
        pdf.set_text_color(35, 35, 35)
        pdf.cell(0, 12, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(26, 35, 126)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    def subsection_title(text):
        pdf.set_font(font, 'B', 13)
        pdf.set_text_color(26, 35, 126)
        pdf.cell(0, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def paragraph(text, size=11, line_height=7):
        if not text:
            return
        pdf.set_font(font, '', size)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            pdf.epw,
            line_height,
            text,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align='L',
            wrapmode=WrapMode.CHAR,
        )
        pdf.ln(2)

    def nearest_bounds(frame, lookback):
        current = float(frame['Close'].iloc[-1])
        pivots = calculate_pivots(
            frame,
            length=max(4, min(18, len(frame) // 8)),
            max_labels=14,
        )
        levels = select_pivot_levels(frame, pivots, limit=8)
        supports = sorted(
            (float(item['price']) for item in levels if float(item['price']) < current),
            reverse=True,
        )
        resistances = sorted(
            float(item['price']) for item in levels if float(item['price']) > current
        )
        support = supports[0] if supports else float(frame['Low'].tail(lookback).min())
        resistance = (
            resistances[0]
            if resistances
            else float(frame['High'].tail(lookback).max())
        )
        return support, resistance

    monthly_dir = trend_direction(dfs['M'])
    weekly_dir = trend_direction(dfs['W'])
    daily_dir = trend_direction(dfs['D'])
    direction_labels = harmonize_directions(monthly_dir, weekly_dir, daily_dir)

    pdf.add_page()
    pdf.set_fill_color(18, 38, 86)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_fill_color(34, 87, 122)
    pdf.rect(0, 0, 210, 68, 'F')
    pdf.set_fill_color(38, 166, 154)
    pdf.rect(0, 255, 210, 42, 'F')
    pdf.set_fill_color(255, 255, 255)
    pdf.ellipse(84, 27, 42, 42, 'F')
    pdf.set_xy(84, 38)
    pdf.set_font('latin' if font == 'latin' else font, 'B', 17)
    pdf.set_text_color(18, 38, 86)
    pdf.cell(42, 12, 'K', align='C')
    pdf.set_y(88)
    pdf.set_font(font, 'B', 25)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(
        0, 16, f'{name}\n综合技术分析报告',
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C',
    )
    pdf.ln(12)
    pdf.set_font(font, '', 13)
    pdf.cell(0, 10, f'标的代码：{symbol}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.cell(
        0, 10, f'生成日期：{datetime.now().strftime("%Y-%m-%d")}',
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C',
    )
    pdf.set_y(229)
    pdf.set_font(font, '', 10)
    pdf.cell(0, 7, '作者：SUNRISE-晞晨', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.cell(0, 7, 'yuanxing@coze.email', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.add_page()
    section_title('阅读说明')
    paragraph(
        '这份报告先看多年价格结构，再依次缩小到月线、周线和日线。这样可以先确定市场所处的大阶段，'
        '再观察中期位置和短期节奏，避免被某一根K线带偏。'
    )
    paragraph(
        '形态不是必然结果，而是买卖双方力量变化留下的线索。阅读时应把形态出现的位置、成交量、'
        '关键支撑和后续收盘表现放在一起理解。重要形态只用一两句话说明，完整定义请参见附录形态速查。'
    )
    paragraph('本报告用于研究和风险观察，不构成投资建议。')
    paragraph(
        '本报告由 SUNRISE-晞晨 基于 AI 辅助生成，作者享有全部著作权。'
        '如希望转载或进一步研究，请联系作者。'
    )

    pdf.add_page()
    section_title('长期趋势判断')
    subsection_title('大尺度趋势结构（趋势线 + 波浪）')
    paragraph(extra_charts.get('super_summary', '多年结构数据不足，暂以月线方向作为长期参考。'), 12, 8)
    super_path = extra_charts.get('supertrend')
    if super_path and os.path.exists(super_path):
        image_y = pdf.get_y()
        pdf.image(super_path, x=10, y=image_y, w=190, h=78)
        pdf.set_y(image_y + 81)
    paragraph(
        '橙色折线连接主要拐点，主升浪、调整浪和主跌浪展示价格运行节奏。'
        '艾略特波浪理论认为，上升行情往往以1至5浪推进，随后进入ABC三浪调整；'
        '道氏理论则把趋势分为主要趋势、次级趋势和小趋势，判断时应先服从主要趋势。',
        9, 6,
    )

    pdf.add_page()
    subsection_title('水平支撑位与阻力位')
    pivot_path = extra_charts.get('pivot')
    if pivot_path and os.path.exists(pivot_path):
        image_y = pdf.get_y()
        pdf.image(pivot_path, x=10, y=image_y, w=190, h=79)
        pdf.set_y(image_y + 82)
    levels = extra_charts.get('pivot_data', {}).get('levels', [])
    current = float(dfs['D']['Close'].iloc[-1])
    supports = sorted(
        (float(item['price']) for item in levels if float(item['price']) < current),
        reverse=True,
    )
    resistances = sorted(
        float(item['price']) for item in levels if float(item['price']) > current
    )
    nearest_support = supports[0] if supports else float(dfs['D']['Low'].tail(30).min())
    nearest_resistance = (
        resistances[0]
        if resistances
        else float(dfs['D']['High'].tail(30).max())
    )
    paragraph(
        f'主要支撑位约在{nearest_support:.2f}元。该位置接近最近的结构低点，'
        '若收盘价持续落在其下方，原有趋势结构可能进一步转弱。'
        f'主要阻力位约在{nearest_resistance:.2f}元。该位置接近最近的结构高点，'
        '若价格有效站稳其上方，趋势才可能出现实质改善。',
        11, 7,
    )
    subsection_title('综合来看')
    paragraph(
        f'月线处于{direction_labels["M"]}，决定当前大方向；'
        f'周线表现为{direction_labels["W"]}；日线表现为{direction_labels["D"]}。'
        f'{synthesize(monthly_dir, weekly_dir, daily_dir, pattern_maps, adaptive_contexts)}',
        12, 8
    )
    key = _key_patterns(pattern_maps.get('W', []) + pattern_maps.get('D', []), 2)
    if key:
        paragraph('当前最值得跟踪的是' + '、'.join(key) + '。' + _shape_note(key[0]))
    if trend_context:
        subsection_title('市场情况概述')
        paragraph(trend_context, 11, 7)

    period_names = {
        'M': '月线分析',
        'W': '周线分析',
        'D': '日线分析',
    }
    for period in ['M', 'W', 'D']:
        print(f'  正在排版：{period_names[period]}')
        pdf.add_page()
        section_title(period_names[period])
        paragraph(
            _period_text(
                period, dfs[period], pattern_maps.get(period, []),
                adaptive_contexts.get(period),
            ),
            10, 6,
        )
        img_path = extra_charts.get(period, f'{symbol}_{period.lower()}.png')
        if os.path.exists(img_path):
            image_y = pdf.get_y()
            if image_y + 104 > 278:
                pdf.add_page()
                image_y = 18
            pdf.image(img_path, x=10, y=image_y, w=190, h=104)

    pdf.add_page()
    section_title('结论与操作建议')
    paragraph(
        synthesize(monthly_dir, weekly_dir, daily_dir, pattern_maps, adaptive_contexts),
        12, 8,
    )
    action_path = extra_charts.get('action')
    if action_path and os.path.exists(action_path):
        pdf.image(action_path, x=10, y=58, w=190, h=76)
        pdf.set_y(138)
    observed = []
    for period in ('M', 'W', 'D'):
        for event in sorted(
                pattern_maps.get(period, []),
                key=lambda item: item.get('end', 0),
                reverse=True):
            if event.get('category') == 'gap':
                continue
            label = _display_name(event.get('name'))
            if label not in observed:
                observed.append(label)
            if len(observed) >= 3:
                break
        if len(observed) >= 3:
            break
    if observed:
        paragraph(
            '本次实际检测并完成结构筛选的近期信号包括：' + '、'.join(observed) + '。'
            + ' '.join(_shape_note(item) for item in observed[:2]),
            10, 6,
        )
    else:
        paragraph('近期没有形成需要单独强调的形态，结论主要依据趋势、均线和价格结构。', 10, 6)
    monthly_support, monthly_resistance = nearest_bounds(dfs['M'], 8)
    weekly_support, weekly_resistance = nearest_bounds(dfs['W'], 16)
    daily_support, daily_resistance = nearest_bounds(dfs['D'], 30)
    paragraph(
        f'长线：月线处于{direction_labels["M"]}。最近支撑约为{monthly_support:.2f}元，'
        f'最近阻力约为{monthly_resistance:.2f}元。若月线持续守住支撑，长期结构仍可观察；'
        '若有效跌破且下月不能收复，则需重新评估趋势。',
        11, 7,
    )
    paragraph(
        f'中线：周线表现为{direction_labels["W"]}。最近支撑约为{weekly_support:.2f}元，'
        f'最近阻力约为{weekly_resistance:.2f}元。若周线有效站稳阻力上方，中期力度可能改善；'
        '若支撑失守，则应提高风险控制级别。',
        11, 7,
    )
    paragraph(
        f'短线：日线表现为{direction_labels["D"]}。最近支撑约为{daily_support:.2f}元，'
        f'最近阻力约为{daily_resistance:.2f}元。若日线有效站稳阻力上方，'
        '可继续观察反弹或上行的延续性；若跌破支撑后不能快速收复，短线风险将明显增加。',
        11, 7,
    )
    paragraph(_gap_text(pattern_maps.get('D', []), 'D'), 10, 6)

    all_names = sorted({
        (_display_name(p.get('name')).split('·')[0]
         if p.get('category') == 'gap' else _display_name(p.get('name')))
        for ps in pattern_maps.values() for p in ps
    })
    all_names = [item for item in all_names if item not in ('三空阴线', '多方试盘', 'three gap decline')]
    with tempfile.TemporaryDirectory(prefix='kline-pattern-guides-') as guide_dir:
        for index, pattern_name in enumerate(all_names):
            if index % 6 == 0:
                pdf.add_page()
                section_title('附录：形态速查')
                paragraph('示意图用于理解结构；实际判断还要结合趋势、位置、成交量和后续确认。', 8, 5)
            row = (index % 6) // 2
            col = (index % 6) % 2
            y = 42 + row * 78
            x = 10 + col * 98
            guide_path = os.path.join(guide_dir, f'guide_{index}.png')
            _mini_pattern_chart(pattern_name, guide_path)
            pdf.image(guide_path, x=x, y=y, w=42, h=18)
            pdf.set_xy(x + 44, y)
            pdf.set_font(font, 'B', 10)
            pdf.cell(52, 6, pattern_name)
            pdf.set_xy(x, y + 21)
            pdf.set_font(font, '', 8)
            pdf.multi_cell(92, 4.2, _guide_definition(pattern_name), align='L')

        pdf.output(out_path)


# CLI

def main():
    parser = argparse.ArgumentParser(description='K线形态大师 PDF 报告生成器')
    parser.add_argument('--symbol', default='sh518880', help='标的代码，如 sh518880 / hk00700')
    parser.add_argument('--name', default=None, help='标的显示名称')
    parser.add_argument('--output', default=None, help='输出 PDF 路径')
    parser.add_argument('--periods', default='M,W,D', help='输出周期')
    parser.add_argument('--monthly-bars', type=int, default=60, help='月 K 展示根数')
    parser.add_argument('--weekly-bars', type=int, default=120, help='周 K 展示根数')
    parser.add_argument('--daily-bars', type=int, default=180, help='日 K 展示根数')
    parser.add_argument('--trend-context', default='', help='市场面/基本面信息（1-2 句话）')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    code = symbol.upper()
    name = args.name.strip() if args.name else code
    if args.output:
        out_pdf_path = Path(args.output.strip()).expanduser()
        if not out_pdf_path.is_absolute():
            out_pdf_path = PROJECT_ROOT / out_pdf_path
    else:
        out_pdf_path = PROJECT_ROOT / f'{code}_K线形态报告.pdf'
    output_dir = out_pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trend_context = args.trend_context.strip()
    if not trend_context:
        print('正在检索白名单市场信息...')
        trend_context = fetch_market_context(symbol, name)

    print(f'[{code}] 正在获取数据...')
    daily_full = fetch_tencent_daily(symbol, years=12)
    print(f'日线数据：{len(daily_full)} 根')

    weekly_full = resample_weekly(daily_full)
    monthly_full = resample_monthly(daily_full)
    a_share = symbol.lower().startswith(('sh', 'sz', 'bj'))
    daily_ma = 250 if a_share else 200
    daily_full['LongMA'] = daily_full['Close'].rolling(daily_ma, min_periods=daily_ma).mean()
    daily_ma_series = daily_full.set_index('Date')['LongMA'].sort_index()
    weekly_full['LongMA'] = [
        daily_ma_series.asof(date) for date in weekly_full['Date']
    ]
    monthly_full['LongMA'] = [
        daily_ma_series.asof(date) for date in monthly_full['Date']
    ]

    dfs = {
        'D': daily_full.tail(args.daily_bars).reset_index(drop=True),
        'W': weekly_full.tail(args.weekly_bars).reset_index(drop=True),
        'M': monthly_full.tail(args.monthly_bars).reset_index(drop=True),
    }

    print('正在检测形态...')
    pattern_maps = {}
    adaptive_contexts = {}
    engine_status_reported = False
    for period, df in dfs.items():
        if len(df) < 3:
            pattern_maps[period] = []
            adaptive_contexts[period] = {'patterns': []}
            continue
        raw_patterns = detect_all(df, max_trend_per_direction=3)
        result = adaptive_analyze(df, raw_patterns, period)
        adaptive_contexts[period] = result
        pattern_maps[period] = result['patterns']
        if not engine_status_reported:
            if result.get('engine') == 'judgment':
                print(
                    'FAE状态：FAE判定引擎'
                    f"（案例库 {result.get('case_count', 0)} 条）"
                )
            else:
                detail = result.get('error') or '引擎不可用'
                print(f'FAE状态：规则回退（{detail}）')
            engine_status_reported = True
        print(f"  {period}：{len(pattern_maps[period])} 个有效事件")

    print('正在绘制图表...')
    chart_paths = {}
    for period, df in dfs.items():
        period_lower = period.lower()
        out_png = output_dir / f'{code}_{period_lower}.png'
        period_label = {'M': '月', 'W': '周', 'D': '日'}[period]
        render_chart(
            df,
            pattern_maps[period],
            f'{name} {period_label}K线技术形态',
            str(out_png),
            ma_period=daily_ma,
            ma_label=f'{daily_ma}日均线',
        )
        chart_paths[period] = str(out_png)
        print(f'  已保存：{out_png}')

    super_path = output_dir / f'{code}_supertrend.png'
    pivot_path = output_dir / f'{code}_pivot.png'
    zigzag = render_supertrend_chart(
        daily_full,
        f'{name} 长期趋势结构与主要波段',
        str(super_path),
        adaptive_contexts.get('D'),
    )
    pivot_data = render_pivot_chart(
        daily_full,
        f'{name} 主要支撑位与阻力位',
        str(pivot_path),
        adaptive_contexts.get('D'),
    )
    super_summary = supertrend_summary(
        dfs['M'], zigzag, pivot_data, adaptive_contexts.get('D', {})
    )
    print(f'  已保存：{super_path}')
    print(f'  已保存：{pivot_path}')
    action_path = output_dir / f'{code}_action.png'
    render_action_chart(
        daily_full,
        pivot_data.get('levels', []),
        f'{name} 支撑、阻力与趋势观察',
        str(action_path),
    )
    print(f'  已保存：{action_path}')

    print('正在生成 PDF...')
    extra_charts = {
        **chart_paths,
        'supertrend': str(super_path),
        'pivot': str(pivot_path),
        'pivot_data': pivot_data,
        'super_summary': super_summary,
        'action': str(action_path),
    }
    make_pdf(
        name, code, trend_context, dfs, pattern_maps, str(out_pdf_path),
        adaptive_contexts=adaptive_contexts,
        extra_charts=extra_charts,
    )
    print(f'完成！报告：{out_pdf_path}')


if __name__ == '__main__':
    main()
