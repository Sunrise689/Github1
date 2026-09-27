import pandas as pd
import numpy as np

from fae.judgment.quant_context import (
    adaptive_confirmation_bars,
    adaptive_trend_context,
)


# ==================== 提示词1：大阳线/大阴线判定重构 ====================

def detect_big_candles(df, body_multiplier=2.0, shadow_ratio=0.15, body_ratio=0.85):
    """
    检测大阳线/大阴线K线形态。

    返回: pd.Series, 值域为 {'大阳线', '大阴线', ''}
    """
    n = len(df)
    result = pd.Series('', index=df.index)
    if n < 2:
        return result

    open_price = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_price).abs()
    amplitude = high - low
    upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low

    avg_body_20 = body.rolling(window=min(20, n), min_periods=2).mean()
    total_shadow = upper_shadow + lower_shadow

    is_bullish = close > open_price
    is_bearish = close < open_price

    big_body = body > avg_body_20 * body_multiplier
    tight_shadow = total_shadow / amplitude.replace(0, np.nan) < shadow_ratio
    high_body_ratio = body / amplitude.replace(0, np.nan) > body_ratio

    bullish_candidate = is_bullish & big_body & tight_shadow & high_body_ratio
    bearish_candidate = is_bearish & big_body & tight_shadow & high_body_ratio

    if n <= 50:
        target_ratio = 0.10
    elif n <= 120:
        target_ratio = 0.10 - (n - 50) * (0.02 / 70)
    else:
        target_ratio = max(0.06, 0.08 - (n - 120) * (0.02 / 80))

    target_count = max(1, int(round(n * target_ratio)))

    candidate_score = pd.Series(0.0, index=df.index)
    candidate_score[bullish_candidate | bearish_candidate] = body

    candidate_indices = candidate_score[candidate_score > 0].sort_values(ascending=False).index
    selected_indices = candidate_indices[:target_count]

    selected_bullish = bullish_candidate.loc[selected_indices]
    selected_bearish = bearish_candidate.loc[selected_indices]

    result.loc[selected_bullish[selected_bullish].index] = '大阳线'
    result.loc[selected_bearish[selected_bearish].index] = '大阴线'

    return result


# ==================== 提示词2：形态优先级排他规则 ====================

PRIORITY_ORDER = ['看涨吞没', '看跌吞没', '刺透形态', '乌云盖顶',
                  '大阳线', '大阴线', '旭日东升', '倾盆大雨',
                  '射击之星', '吊颈线', '锤子线', '倒锤头', '上吊线',
                  '长十字星', '十字星', '纺锤线', 'T字线', '倒T字线', '一字线']


def dedup_patterns(patterns: dict) -> dict:
    """
    对同一日期命中的多个形态进行优先级排他，只保留力度最强的一个。

    输入: patterns = {date: ['大阳线', '看涨吞没', '刺透形态']}
    输出: {date: ['看涨吞没']}  # 只保留最强的
    """
    result = {}
    for date, pattern_list in patterns.items():
        if not pattern_list:
            result[date] = []
            continue

        best_pattern = None
        best_priority = float('inf')
        for pattern in pattern_list:
            if pattern in PRIORITY_ORDER:
                priority = PRIORITY_ORDER.index(pattern)
                if priority < best_priority:
                    best_priority = priority
                    best_pattern = pattern

        result[date] = [best_pattern] if best_pattern is not None else pattern_list

    return result


# ==================== 提示词3：刺透形态/乌云盖顶判定校正 ====================

def detect_piercing(df, gap_threshold=0.005):
    """
    检测刺透形态（看涨Piercing Line）。

    返回: pd.Series, True=刺透形态
    """
    n = len(df)
    result = pd.Series(False, index=df.index)
    if n < 2:
        return result

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    prev_body = body.shift(1)

    prev_big_bearish = (prev_body > avg_body_20.shift(1)) & (prev_close < prev_open)
    curr_big_bullish = (body > avg_body_20) & (close > open_p)
    gap_down = (open_p - prev_close) / prev_close.abs().replace(0, np.nan) < -gap_threshold
    prev_body_mid = (prev_open + prev_close) / 2
    close_above_mid = close > prev_body_mid
    # 刺透只进入前一根阴线实体，不能完整覆盖前实体；完整覆盖应由
    # ``detect_engulfing`` 唯一命名为看涨吞没。
    close_below_prev_open = close < (prev_open - prev_open.abs() * 1e-4)
    no_long_lower_shadow = (lower_shadow / amplitude.replace(0, np.nan)) < 0.30

    result = (
        prev_big_bearish
        & curr_big_bullish
        & gap_down
        & close_above_mid
        & close_below_prev_open
        & no_long_lower_shadow
    )
    return result


def detect_dark_cloud(df, gap_threshold=0.005):
    """
    检测乌云盖顶（看跌Dark Cloud Cover）。

    返回: pd.Series, True=乌云盖顶
    """
    n = len(df)
    result = pd.Series(False, index=df.index)
    if n < 2:
        return result

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)

    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    prev_body = body.shift(1)

    prev_big_bullish = (prev_body > avg_body_20.shift(1)) & (prev_close > prev_open)
    curr_big_bearish = (body > avg_body_20) & (close < open_p)
    gap_up = (open_p - prev_close) / prev_close.abs().replace(0, np.nan) > gap_threshold
    prev_body_mid = (prev_open + prev_close) / 2
    close_below_mid = close < prev_body_mid
    # 乌云盖顶只深入前一根阳线实体；若收盘已经越过前一实体开盘价，
    # 就是看跌吞没而不是乌云盖顶。
    close_above_prev_open = close > (prev_open + prev_open.abs() * 1e-4)
    no_long_upper_shadow = (upper_shadow / amplitude.replace(0, np.nan)) < 0.30

    result = (
        prev_big_bullish
        & curr_big_bearish
        & gap_up
        & close_below_mid
        & close_above_prev_open
        & no_long_upper_shadow
    )
    return result


# ==================== 提示词4：十字星/纺锤线判定修复 ====================

def detect_doji(df, body_amp_ratio=0.08):
    """
    检测十字星、长十字星和纺锤线。

    返回: dict {'十字星': Series, '长十字星': Series, '纺锤线': Series}
    """
    n = len(df)

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    amp_safe = amplitude.replace(0, np.nan)
    body_amp = body / amp_safe

    is_doji = body_amp < body_amp_ratio

    long_shadow_threshold = avg_amp_20 * 0.4
    is_long_legged = (
        is_doji
        & (upper_shadow > long_shadow_threshold)
        & (lower_shadow > long_shadow_threshold)
    )

    is_spinning_top = (
        (body_amp >= body_amp_ratio)
        & (body_amp < 0.25)
        & (upper_shadow > avg_amp_20 * 0.15)
        & (lower_shadow > avg_amp_20 * 0.15)
    )

    return {
        '十字星': is_doji,
        '长十字星': is_long_legged,
        '纺锤线': is_spinning_top,
    }


# ==================== 提示词5：射击之星/倒锤头/吊颈线判定修复 ====================

def detect_shooting_star_hammer(df, lookback=10, confirmation_bars=None):
    """
    检测射击之星、锤子线、倒锤头和吊颈线。

    返回: dict {'射击之星': Series, '锤子线': Series,
                '倒锤头': Series, '吊颈线': Series}
    """
    n = len(df)

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    amp_safe = amplitude.replace(0, np.nan)

    shape_match = (
        (upper_shadow / amp_safe > 0.55)
        & (lower_shadow / amp_safe < 0.10)
        & (body / amp_safe < 0.30)
    )

    rolling_high = high.rolling(window=lookback, min_periods=3).quantile(0.80)
    rolling_low = low.rolling(window=lookback, min_periods=3).quantile(0.20)

    # 顶/底背景用影线极值判断，而不是收盘价，避免长影线形态被排除。
    is_high = high > rolling_high
    is_low = low < rolling_low

    # 射击之星与倒锤头共享影线外观，但方向背景相反。
    shooting_star = shape_match & is_high
    inverted_hammer = shape_match & is_low

    # 锤子线/吊颈线必须是长下影线，不能复用射击之星的长上影线条件。
    hammer_shape = (
        (lower_shadow / amp_safe > 0.55)
        & (upper_shadow / amp_safe < 0.10)
        & (body / amp_safe < 0.30)
    )
    if confirmation_bars is None:
        confirmation_bars = adaptive_confirmation_bars(len(df), lookback=120)
    confirmation_bars = max(1, int(confirmation_bars))
    future_closes = pd.concat(
        [close.shift(-offset) for offset in range(1, confirmation_bars + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [low.shift(-offset) for offset in range(1, confirmation_bars + 1)],
        axis=1,
    )
    enough_future = future_closes.notna().all(axis=1)

    # FAE Phase-1量化背景：使用100/120根K线内的滚动收益分位数和斜率，
    # 不把“3根上涨”这种固定规则当成较大趋势。
    quant_lookback = max(20, min(120, len(df)))
    quant_context = adaptive_trend_context(close, lookback=quant_lookback)
    uptrend_context = quant_context["state"].eq("uptrend")
    downtrend_context = quant_context["state"].eq("downtrend")

    # 锤子线：下跌背景，且确认窗口内所有收盘价都没有跌破下影线低点。
    hammer_hold = future_closes.ge(low, axis=0).all(axis=1)
    hammer = hammer_shape & is_low & downtrend_context & enough_future & hammer_hold

    # 吊颈线：上涨背景；后续确认窗口内必须有K线跌破下影线低点。
    breakout_confirm = future_lows.lt(low, axis=0).any(axis=1) & enough_future
    hanging_man = hammer_shape & is_high & uptrend_context & breakout_confirm

    return {
        '射击之星': shooting_star,
        '锤子线': hammer,
        '倒锤头': inverted_hammer,
        '吊颈线': hanging_man,
    }


# ==================== 提示词6：缺口判定 ====================

def detect_gaps(df):
    """
    检测跳空缺口及回补情况。

    返回: DataFrame, 每行一个缺口
        - gap_date: 缺口出现的日期（后一根K线的日期）
        - gap_type: 'up' 向上跳空 / 'down' 向下跳空
        - gap_top: 缺口上沿
        - gap_bottom: 缺口下沿
        - filled: 是否被回补
        - fill_date: 回补日期（如已回补）
        - fill_bars: 几根K线后回补的
    """
    n = len(df)
    if n < 2:
        return pd.DataFrame()

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_gap = prev_high < low
    down_gap = prev_low > high

    gaps = []
    for i in range(1, n):
        if up_gap.iloc[i]:
            gap_type = 'up'
            gap_bottom = low.iloc[i]
            gap_top = prev_high.iloc[i]
        elif down_gap.iloc[i]:
            gap_type = 'down'
            gap_bottom = prev_low.iloc[i]
            gap_top = high.iloc[i]
        else:
            continue

        filled = False
        fill_date = None
        fill_bars = None

        for j in range(i + 1, n):
            if gap_type == 'up':
                if low.iloc[j] <= gap_top and high.iloc[j] >= gap_bottom:
                    filled = True
                    fill_date = df.index[j]
                    fill_bars = j - i
                    break
            else:
                if high.iloc[j] >= gap_bottom and low.iloc[j] <= gap_top:
                    filled = True
                    fill_date = df.index[j]
                    fill_bars = j - i
                    break

        gaps.append({
            'gap_date': df.index[i],
            'gap_type': gap_type,
            'gap_top': gap_top,
            'gap_bottom': gap_bottom,
            'filled': filled,
            'fill_date': fill_date,
            'fill_bars': fill_bars,
        })

    return pd.DataFrame(gaps)


# ==================== 提示词7：组合形态的框线+箭头标注方式 ====================

def draw_pattern_box(ax, start_idx, end_idx, name, color, df, y_range):
    """
    在ax上画一个细线框围住start_idx到end_idx的K线，从框右上角拉箭头到空白位置写name。

    参数:
        ax: matplotlib Axes对象
        start_idx, end_idx: 形态起始/结束索引
        name: 形态名称
        color: 线条/文字颜色
        df: K线DataFrame
        y_range: 当前y轴显示范围 (ymin, ymax)，用于计算箭头空白位置
    """
    import matplotlib.patches as patches

    seg = df.iloc[start_idx:end_idx + 1]
    x_left = start_idx - 0.5
    x_right = end_idx + 0.5
    y_bottom = seg['low'].min()
    y_top = seg['high'].max()

    # 画细线框
    rect = patches.FancyBboxPatch(
        (x_left, y_bottom),
        x_right - x_left,
        y_top - y_bottom,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor='none',
        alpha=0.8
    )
    ax.add_patch(rect)

    # 箭头起点：框右上角
    arrow_x = x_right
    arrow_y = y_top

    # 箭头终点：右侧空白区域
    text_x = end_idx + 2
    text_y = y_top + (y_top - y_bottom) * 0.15
    if y_range is not None:
        text_y = min(text_y, y_range[1] - (y_range[1] - y_range[0]) * 0.05)

    ax.annotate(
        name,
        xy=(arrow_x, arrow_y),
        xytext=(text_x, text_y),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        fontsize=9,
        color=color,
        va='bottom'
    )


# ==================== 提示词8：平顶/平底判定 ====================

def detect_flat_bottom_top(df, price_tolerance=0.003, lookback=10):
    """
    检测平顶和平底形态。

    返回: {'平底': [(idx_start, idx_end, level), ...],
            '平顶': [(idx_start, idx_end, level), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    amplitude = high - low
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()
    tolerance = avg_amp_20 * price_tolerance

    flat_bottoms = []
    flat_tops = []

    for i in range(n - 2):
        window = df.iloc[i:min(i + lookback, n)]
        lows = window['low'] if 'low' in window.columns else window['Low']
        highs = window['high'] if 'high' in window.columns else window['High']

        # 平底：多根K线最低价接近
        min_low = lows.min()
        near_low = (lows - min_low).abs() <= tolerance.iloc[i:min(i + lookback, n)]
        if near_low.sum() >= 3:
            idxs = window.index[near_low.values]
            flat_bottoms.append((
                df.index.get_loc(idxs[0]),
                df.index.get_loc(idxs[-1]),
                min_low
            ))

        # 平顶：多根K线最高价接近
        max_high = highs.max()
        near_high = (highs - max_high).abs() <= tolerance.iloc[i:min(i + lookback, n)]
        if near_high.sum() >= 3:
            idxs = window.index[near_high.values]
            flat_tops.append((
                df.index.get_loc(idxs[0]),
                df.index.get_loc(idxs[-1]),
                max_high
            ))

    return {'平底': flat_bottoms, '平顶': flat_tops}


# ==================== 提示词9：上升三法/下降三法判定 ====================

def detect_three_methods(df):
    """
    检测上升三法和下降三法。

    返回: {'上升三法': [(idx0, idx_mid_start, idx_mid_end, idx_last), ...],
            '下降三法': [(idx0, idx_mid_start, idx_mid_end, idx_last), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    rising_three = []
    falling_three = []

    for i in range(n - 4):
        idx0 = i
        idx_mid_start = i + 1
        idx_mid_end = i + 3
        idx_last = i + 4

        # 上升三法
        first_bullish = close.iloc[idx0] > open_p.iloc[idx0]
        first_big = body.iloc[idx0] > avg_body_20.iloc[idx0]
        last_bullish = close.iloc[idx_last] > open_p.iloc[idx_last]
        last_big = body.iloc[idx_last] > avg_body_20.iloc[idx_last]

        mid_bodies = body.iloc[idx_mid_start:idx_mid_end + 1]
        mid_closes = close.iloc[idx_mid_start:idx_mid_end + 1]
        mid_opens = open_p.iloc[idx_mid_start:idx_mid_end + 1]
        mid_bearish_count = (mid_closes < mid_opens).sum()

        mid_low_not_break = low.iloc[idx_mid_start:idx_mid_end + 1].min() >= open_p.iloc[idx0]
        last_high_new = close.iloc[idx_last] > high.iloc[idx_mid_start:idx_mid_end + 1].max()

        if (first_bullish and first_big and last_bullish and last_big
                and mid_bearish_count >= 2
                and mid_low_not_break
                and last_high_new):
            rising_three.append((idx0, idx_mid_start, idx_mid_end, idx_last))

        # 下降三法
        first_bearish = close.iloc[idx0] < open_p.iloc[idx0]
        first_big_fall = body.iloc[idx0] > avg_body_20.iloc[idx0]
        last_bearish = close.iloc[idx_last] < open_p.iloc[idx_last]
        last_big_fall = body.iloc[idx_last] > avg_body_20.iloc[idx_last]

        mid_bullish_count = (mid_closes > mid_opens).sum()
        mid_high_not_break = high.iloc[idx_mid_start:idx_mid_end + 1].max() <= open_p.iloc[idx0]
        last_low_new = close.iloc[idx_last] < low.iloc[idx_mid_start:idx_mid_end + 1].min()

        if (first_bearish and first_big_fall and last_bearish and last_big_fall
                and mid_bullish_count >= 2
                and mid_high_not_break
                and last_low_new):
            falling_three.append((idx0, idx_mid_start, idx_mid_end, idx_last))

    return {'上升三法': rising_three, '下降三法': falling_three}


# ==================== 提示词11：黑三兵/三只乌鸦判定 ====================

def detect_black_three_soldiers(df):
    """
    检测黑三兵、三只乌鸦和下跌三连阴。

    返回: {'黑三兵': [(idx0, idx1, idx2), ...],
            '三只乌鸦': [(idx0, idx1, idx2), ...],
            '下跌三连阴': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    black_three = []
    three_crows = []
    three_down = []

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        closes = close.iloc[idx0:idx2 + 1]
        opens = open_p.iloc[idx0:idx2 + 1]
        lows_seg = low.iloc[idx0:idx2 + 1]
        highs_seg = high.iloc[idx0:idx2 + 1]
        bodies = body.iloc[idx0:idx2 + 1]

        # 三根都是阴线，收盘价依次降低
        all_bearish = (closes < opens).all()
        descending = (closes.iloc[0] > closes.iloc[1] > closes.iloc[2])

        if not (all_bearish and descending):
            continue

        three_down.append((idx0, idx1, idx2))

        # 黑三兵：小阴线，开盘在前一根实体范围内
        small_bodies = (bodies < avg_body_20.iloc[idx0:idx2 + 1]).all()
        open_in_prev_body = (
            (opens.iloc[1] <= closes.iloc[0]) and (opens.iloc[1] >= opens.iloc[0])
            and (opens.iloc[2] <= closes.iloc[1]) and (opens.iloc[2] >= opens.iloc[1])
        )
        if small_bodies and open_in_prev_body:
            black_three.append((idx0, idx1, idx2))

        # 三只乌鸦：实体饱满、下影线极短（收盘接近最低价）
        lower_shadows = pd.concat([opens, closes], axis=1).min(axis=1) - lows_seg
        short_lower_shadows = (lower_shadows / bodies.replace(0, np.nan) < 0.10).all()
        big_bodies = (bodies >= avg_body_20.iloc[idx0:idx2 + 1]).all()
        if big_bodies and short_lower_shadows:
            three_crows.append((idx0, idx1, idx2))

    return {
        '黑三兵': black_three,
        '三只乌鸦': three_crows,
        '下跌三连阴': three_down,
    }


# ==================== 提示词12：组合形态全量检测 ====================

def detect_all_composite_patterns(df) -> dict:
    """
    检测所有组合形态，返回 {形态名: [(起始idx, 结束idx), ...]}。

    当前覆盖：覆盖线组合、两红加一黑、两黑加一红、上升三法、下降三法、
             并排阳线/阴线、平顶/平底、早晨十字星/黄昏十字星、黑三兵、下跌三连阴。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    patterns = {
        '下降覆盖线': [], '上升覆盖线': [],
        '两红加一黑': [], '两黑加一红': [],
        '上升三法': [], '下降三法': [],
        '并排阳线': [], '并排阴线': [],
        '平顶': [], '平底': [],
        '早晨十字星': [], '黄昏十字星': [],
        '黑三兵': [], '下跌三连阴': []
    }

    for i in range(n - 3):
        # 下降覆盖线：阴 -> 阳回抽 -> 阴吞阳
        if (close.iloc[i] < open_p.iloc[i] and
            close.iloc[i+1] > open_p.iloc[i+1] and
            close.iloc[i+2] < open_p.iloc[i+2] and
            close.iloc[i+2] < open_p.iloc[i+1]):
            patterns['下降覆盖线'].append((i, i+2))

        # 上升覆盖线：阳 -> 阴回抽 -> 阳吞阴
        if (close.iloc[i] > open_p.iloc[i] and
            close.iloc[i+1] < open_p.iloc[i+1] and
            close.iloc[i+2] > open_p.iloc[i+2] and
            close.iloc[i+2] > open_p.iloc[i+1]):
            patterns['上升覆盖线'].append((i, i+2))

    for i in range(n - 2):
        # 两红加一黑
        if (close.iloc[i] > open_p.iloc[i] and
            close.iloc[i+1] > open_p.iloc[i+1] and
            close.iloc[i+2] < open_p.iloc[i+2] and
            close.iloc[i+2] > close.iloc[i]):
            patterns['两红加一黑'].append((i, i+2))

        # 两黑加一红
        if (close.iloc[i] < open_p.iloc[i] and
            close.iloc[i+1] < open_p.iloc[i+1] and
            close.iloc[i+2] > open_p.iloc[i+2] and
            close.iloc[i+2] < close.iloc[i]):
            patterns['两黑加一红'].append((i, i+2))

        # 早晨十字星：大阴 -> 十字星 -> 大阳
        if (body.iloc[i] > avg_body_20.iloc[i] and close.iloc[i] < open_p.iloc[i] and
            body.iloc[i+1] / max(high.iloc[i+1] - low.iloc[i+1], 1e-9) < 0.08 and
            body.iloc[i+2] > avg_body_20.iloc[i+2] and close.iloc[i+2] > open_p.iloc[i+2] and
            close.iloc[i+2] > open_p.iloc[i]):
            patterns['早晨十字星'].append((i, i+2))

        # 黄昏十字星：大阳 -> 十字星 -> 大阴
        if (body.iloc[i] > avg_body_20.iloc[i] and close.iloc[i] > open_p.iloc[i] and
            body.iloc[i+1] / max(high.iloc[i+1] - low.iloc[i+1], 1e-9) < 0.08 and
            body.iloc[i+2] > avg_body_20.iloc[i+2] and close.iloc[i+2] < open_p.iloc[i+2] and
            close.iloc[i+2] < open_p.iloc[i]):
            patterns['黄昏十字星'].append((i, i+2))

    # 上升三法 / 下降三法
    three_methods = detect_three_methods(df)
    patterns['上升三法'] = three_methods.get('上升三法', [])
    patterns['下降三法'] = three_methods.get('下降三法', [])

    # 并排阳线/阴线
    for i in range(n - 1):
        body0, body1 = body.iloc[i], body.iloc[i+1]
        body0_safe = max(abs(float(body0)), 1e-9)
        if (close.iloc[i] > open_p.iloc[i] and close.iloc[i+1] > open_p.iloc[i+1] and
            abs(float(body0) - float(body1)) / body0_safe < 0.25):
            patterns['并排阳线'].append((i, i+1))
        if (close.iloc[i] < open_p.iloc[i] and close.iloc[i+1] < open_p.iloc[i+1] and
            abs(float(body0) - float(body1)) / body0_safe < 0.25):
            patterns['并排阴线'].append((i, i+1))

    # 平顶/平底
    flat = detect_flat_bottom_top(df)
    patterns['平底'] = flat.get('平底', [])
    patterns['平顶'] = flat.get('平顶', [])

    # 黑三兵 / 下跌三连阴
    black = detect_black_three_soldiers(df)
    patterns['黑三兵'] = black.get('黑三兵', [])
    patterns['下跌三连阴'] = black.get('下跌三连阴', [])

    return patterns


def draw_composite_patterns(df, patterns, ax):
    """
    在ax上画出所有组合形态的框+箭头标注。
    """
    import matplotlib.patches as patches

    colors = {
        '下降覆盖线': 'blue', '上升覆盖线': 'red',
        '两红加一黑': 'blue', '两黑加一红': 'red',
        '上升三法': 'red', '下降三法': 'blue',
        '并排阳线': 'red', '并排阴线': 'blue',
        '平顶': 'green', '平底': 'green',
        '早晨十字星': 'red', '黄昏十字星': 'blue',
        '黑三兵': 'blue', '下跌三连阴': 'blue'
    }

    for name, ranges in patterns.items():
        color = colors.get(name, 'gray')
        for r in ranges:
            if name in ['平顶', '平底']:
                start_idx, end_idx, level = r
                ax.hlines(level, start_idx - 0.5, end_idx + 0.5,
                          colors=color, linewidth=1.2, linestyles='--')
                ax.text(end_idx + 1, level, name, color=color, fontsize=9, va='center')
            else:
                start_idx, end_idx = r[0], r[-1]
                seg = df.iloc[start_idx:end_idx + 1]
                y_bottom = seg['low'].min()
                y_top = seg['high'].max()
                rect = patches.Rectangle(
                    (start_idx - 0.5, y_bottom),
                    end_idx - start_idx + 1,
                    y_top - y_bottom,
                    linewidth=1.0,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.annotate(
                    name,
                    xy=(end_idx + 0.5, y_top),
                    xytext=(end_idx + 2, y_top + (y_top - y_bottom) * 0.1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
                    fontsize=9,
                    color=color
                )


# ==================== 提示词13：旭日东升 / 倾盆大雨 ====================

def detect_sun_and_rain(df, gap_threshold=None):
    """
    检测旭日东升和倾盆大雨。

    返回: {'旭日东升': Series, '倾盆大雨': Series}
    """
    n = len(df)
    result = pd.Series(False, index=df.index)
    if n < 2:
        return {'旭日东升': result.copy(), '倾盆大雨': result.copy()}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    if gap_threshold is None:
        gap_threshold = avg_amp_20 * 0.3

    prev_open = open_p.shift(1)
    prev_close = close.shift(1)

    # 旭日东升：前大阴，后大阳跳空低开，后阳收盘 > 前阴开盘
    prev_bear = (prev_close < prev_open) & (body.shift(1) > avg_body_20.shift(1))
    curr_bull = (close > open_p) & (body > avg_body_20)
    sun_gap = (open_p - prev_close) < -gap_threshold
    sun_cover = close > prev_open
    sun = prev_bear & curr_bull & sun_gap & sun_cover

    # 倾盆大雨：前大阳，后大阴跳空高开，后阴收盘 < 前阳开盘
    prev_bull = (prev_close > prev_open) & (body.shift(1) > avg_body_20.shift(1))
    curr_bear = (close < open_p) & (body > avg_body_20)
    rain_gap = (open_p - prev_close) > gap_threshold
    rain_cover = close < prev_open
    rain = prev_bull & curr_bear & rain_gap & rain_cover

    return {'旭日东升': sun, '倾盆大雨': rain}


# ==================== 提示词14：早晨之星 / 黄昏之星 ====================

def detect_morning_evening_star(df):
    """
    检测早晨之星、黄昏之星及其十字星变体。

    返回: {'早晨之星': [(idx1, idx2, idx3), ...],
            '黄昏之星': [(idx1, idx2, idx3), ...],
            '早晨十字星': [(idx1, idx2, idx3), ...],
            '黄昏十字星': [(idx1, idx2, idx3), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = (high - low).clip(lower=0)
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_range_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    morning_stars = []
    evening_stars = []
    morning_dojis = []
    evening_dojis = []

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        amp0 = amplitude.iloc[idx0]
        amp1 = amplitude.iloc[idx1]
        body0 = body.iloc[idx0]
        body1 = body.iloc[idx1]
        body2 = body.iloc[idx2]

        if amp0 == 0 or amp1 == 0:
            continue

        body_amp_ratio1 = body1 / amp1

        # 早晨之星/早晨十字星：第一大阴，中间星线，第三大阳
        first_bear = body0 > avg_body_20.iloc[idx0] and close.iloc[idx0] < open_p.iloc[idx0]
        third_bull = body2 > avg_body_20.iloc[idx2] and close.iloc[idx2] > open_p.iloc[idx2]
        third_penetrate = close.iloc[idx2] > (open_p.iloc[idx0] + close.iloc[idx0]) / 2
        middle_gap = low.iloc[idx1] > high.iloc[idx0] or high.iloc[idx1] < low.iloc[idx0]

        if first_bear and third_bull and third_penetrate:
            if body_amp_ratio1 < 0.08:
                morning_dojis.append((idx0, idx1, idx2))
            elif body_amp_ratio1 < 0.25:
                morning_stars.append((idx0, idx1, idx2))

        # 黄昏之星/黄昏十字星：第一大阳，中间星线，第三大阴
        first_bull = body0 > avg_body_20.iloc[idx0] and close.iloc[idx0] > open_p.iloc[idx0]
        third_bear = body2 > avg_body_20.iloc[idx2] and close.iloc[idx2] < open_p.iloc[idx2]
        third_penetrate_down = close.iloc[idx2] < (open_p.iloc[idx0] + close.iloc[idx0]) / 2

        if first_bull and third_bear and third_penetrate_down:
            if body_amp_ratio1 < 0.08:
                evening_dojis.append((idx0, idx1, idx2))
            elif body_amp_ratio1 < 0.25:
                evening_stars.append((idx0, idx1, idx2))

    return {
        '早晨之星': morning_stars,
        '黄昏之星': evening_stars,
        '早晨十字星': morning_dojis,
        '黄昏十字星': evening_dojis,
    }


# ==================== 弃婴十字星（双缺口严格定义） ====================

def detect_abandoned_baby(df, allow_edge_doji=False):
    """
    检测看涨/看跌弃婴十字星。

    严格条件：三根K线中间为十字星，且中间K线的最高/最低价
    均与两侧K线完全隔离，形成十字星两侧的两个真实缺口。
    首尾实体不强制“大阳/大阴”，允许小实体和纺锤线；默认不允许
    首尾本身也是十字星，以免与“三星形”混淆。若业务确认要采用
    更宽松口径，可显式传入 ``allow_edge_doji=True``。
    单边缺口不计为弃婴十字星，而保留为早晨/黄昏十字星。

    返回:
        {'看涨弃婴十字星': [(idx0, idx1, idx2), ...],
         '看跌弃婴十字星': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    if n < 3:
        return {'看涨弃婴十字星': [], '看跌弃婴十字星': []}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = (high - low).replace(0, np.nan)
    body_ratio = body / amplitude
    quant_lookback = max(20, min(120, n))
    trend_context = adaptive_trend_context(close, lookback=quant_lookback)
    bullish = []
    bearish = []

    for i in range(n - 2):
        j, k = i + 1, i + 2
        doji = (body.iloc[j] / amplitude.iloc[j]) <= 0.08
        if not doji:
            continue

        edge_doji_ok = allow_edge_doji
        first_edge_ok = edge_doji_ok or body_ratio.iloc[i] > 0.08
        third_edge_ok = edge_doji_ok or body_ratio.iloc[k] > 0.08
        if not (first_edge_ok and third_edge_ok):
            continue

        # 看涨弃婴：下跌实体 -> 向下隔离的十字星 -> 向上隔离的阳线。
        bullish_case = (
            close.iloc[i] < open_p.iloc[i]
            and close.iloc[k] > open_p.iloc[k]
            and trend_context["state"].iloc[i] == "downtrend"
            and high.iloc[j] < low.iloc[i]
            and low.iloc[k] > high.iloc[j]
        )
        if bullish_case:
            bullish.append((i, j, k))

        # 看跌弃婴：上涨实体 -> 向上隔离的十字星 -> 向下隔离的阴线。
        bearish_case = (
            close.iloc[i] > open_p.iloc[i]
            and close.iloc[k] < open_p.iloc[k]
            and trend_context["state"].iloc[i] == "uptrend"
            and low.iloc[j] > high.iloc[i]
            and high.iloc[k] < low.iloc[j]
        )
        if bearish_case:
            bearish.append((i, j, k))

    return {
        '看涨弃婴十字星': bullish,
        '看跌弃婴十字星': bearish,
    }


# ==================== 提示词15：身怀六甲（孕线） ====================

def detect_harami(df):
    """
    检测看涨孕线、看跌孕线和十字孕线。

    返回: {'看涨孕线': Series, '看跌孕线': Series, '十字孕线': Series}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_range_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
    prev_range_high = high.shift(1)
    prev_range_low = low.shift(1)

    body_amp_ratio = body / amplitude.replace(0, np.nan)
    is_doji = body_amp_ratio < 0.08

    # 教材的严格定义是“前一根实体包住后一根K线（包括影线）”。
    # 实盘数据存在四舍五入和极小跳价，因此只允许一个相对前一根
    # 波幅很小的容差；明显越界仍然不能叫孕线。若影线超出前一根实体，
    # 但仍落在前一根完整 high/low 范围内，则作为放宽变体保留。
    tolerance = np.maximum(
        avg_range_20.fillna(amplitude.shift(1)).fillna(0.0) * 0.05,
        close.abs() * 0.001,
    )
    current_range_high = high
    current_range_low = low
    strict_contained = (
        (current_range_high <= prev_body_high + tolerance)
        & (current_range_low >= prev_body_low - tolerance)
    )
    relaxed_contained = (
        (current_range_high <= prev_range_high + tolerance)
        & (current_range_low >= prev_range_low - tolerance)
    )
    current_body_contained = (
        (open_p <= prev_body_high + tolerance)
        & (open_p >= prev_body_low - tolerance)
        & (close <= prev_body_high + tolerance)
        & (close >= prev_body_low - tolerance)
    )
    contained = current_body_contained & (strict_contained | relaxed_contained)

    # 看涨孕线：前阴后小阳/小实体，后实体被前实体包容
    prev_bear = prev_close < prev_open
    curr_small = body < avg_body_20 * 0.55
    bullish_harami = (
        prev_bear
        & curr_small
        & contained
    )

    # 看跌孕线：前阳后小阴/小实体
    prev_bull = prev_close > prev_open
    bearish_harami = (
        prev_bull
        & curr_small
        & contained
    )

    # 十字孕线：第二根是十字星
    cross_harami = (bullish_harami | bearish_harami) & is_doji

    return {
        '看涨孕线': bullish_harami,
        '看跌孕线': bearish_harami,
        '十字孕线': cross_harami,
    }


# ==================== 提示词16：T字线 / 倒T字线 / 一字线 ====================

def detect_t_shapes(df):
    """
    检测T字线、倒T字线和一字线。

    返回: {'T字线': Series, '倒T字线': Series, '一字线': Series}
    """
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    # 一字线：开收高低都相等
    is_one_line = (open_p == close) & (close == high) & (high == low)

    # T字线：开盘=收盘=最高，有下影线
    is_t = (open_p == close) & (close == high) & (low < close)

    # 倒T字线：开盘=收盘=最低，有上影线
    is_inverted_t = (open_p == close) & (close == low) & (high > close)

    return {
        'T字线': is_t & ~is_one_line,
        '倒T字线': is_inverted_t & ~is_one_line,
        '一字线': is_one_line,
    }


# ==================== 提示词17：长十字线 / 螺旋桨 ====================

def detect_long_doji_propeller(df):
    """
    检测长十字线和螺旋桨。

    返回: {'长十字线': Series, '螺旋桨': Series}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    amp_safe = amplitude.replace(0, np.nan)
    body_amp_ratio = body / amp_safe
    upper_ratio = upper_shadow / amp_safe
    lower_ratio = lower_shadow / amp_safe

    # 长十字线：十字星 + 上下影线均长（>25%振幅）
    is_long_doji = (body_amp_ratio < 0.08) & (upper_ratio > 0.25) & (lower_ratio > 0.25)

    # 螺旋桨：小实体（>=0.08且<0.25）+ 上下影线均长且接近对称
    is_propeller = (
        (body_amp_ratio >= 0.08)
        & (body_amp_ratio < 0.25)
        & (upper_ratio > 0.25)
        & (lower_ratio > 0.25)
        & ((upper_shadow - lower_shadow).abs() / amplitude.replace(0, np.nan) < 0.10)
    )

    return {
        '长十字线': is_long_doji,
        '螺旋桨': is_propeller,
    }


# ==================== 提示词18：塔形底 / 塔形顶 ====================

def detect_pagoda(df, min_mid_bars=3):
    """
    检测塔形底和塔形顶。

    返回: {'塔形底': [(start_idx, mid_start, mid_end, end_idx), ...],
            '塔形顶': [(start_idx, mid_start, mid_end, end_idx), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    bottoms = []
    tops = []

    for start_idx in range(n - min_mid_bars - 2):
        first_body = body.iloc[start_idx]
        first_bear = close.iloc[start_idx] < open_p.iloc[start_idx]
        first_bull = close.iloc[start_idx] > open_p.iloc[start_idx]

        if first_body < avg_body_20.iloc[start_idx]:
            continue

        for mid_end in range(start_idx + min_mid_bars + 1, min(start_idx + 15, n - 1)):
            mid_start = start_idx + 1
            mid_seg = df.iloc[mid_start:mid_end]
            mid_high = mid_seg['high'].max() if 'high' in mid_seg.columns else mid_seg['High'].max()
            mid_low = mid_seg['low'].min() if 'low' in mid_seg.columns else mid_seg['Low'].min()

            # 塔形底：大阴 -> 横盘 -> 大阳突破
            if first_bear:
                last_bull = close.iloc[mid_end] > open_p.iloc[mid_end]
                last_body = body.iloc[mid_end]
                if (last_bull and last_body > avg_body_20.iloc[mid_end] and
                    close.iloc[mid_end] > mid_high and
                    mid_high - mid_low < first_body * 2):
                    bottoms.append((start_idx, mid_start, mid_end - 1, mid_end))
                    break

            # 塔形顶：大阳 -> 横盘 -> 大阴跌破
            if first_bull:
                last_bear = close.iloc[mid_end] < open_p.iloc[mid_end]
                last_body = body.iloc[mid_end]
                if (last_bear and last_body > avg_body_20.iloc[mid_end] and
                    close.iloc[mid_end] < mid_low and
                    mid_high - mid_low < first_body * 2):
                    tops.append((start_idx, mid_start, mid_end - 1, mid_end))
                    break

    return {'塔形底': bottoms, '塔形顶': tops}


# ==================== 提示词19：圆底 / 圆顶 ====================

def detect_rounding(df, min_bars=7):
    """
    检测圆底和圆顶形态。

    返回: {'圆底': [(start_idx, bottom_idx, end_idx), ...],
            '圆顶': [(start_idx, top_idx, end_idx), ...]}
    """
    n = len(df)
    close = df['close'] if 'close' in df.columns else df['Close']

    bottoms = []
    tops = []

    for start_idx in range(n - min_bars):
        for end_idx in range(start_idx + min_bars, min(start_idx + min_bars + 20, n)):
            seg = close.iloc[start_idx:end_idx + 1]
            x = np.arange(len(seg))

            # 线性拟合
            slope, intercept = np.polyfit(x, seg.values, 1)
            fitted = slope * x + intercept
            residuals = seg.values - fitted

            # 圆底：中间低两边高，残差呈下凹
            bottom_idx = seg.idxmin()
            bottom_pos = seg.index.get_loc(bottom_idx)
            relative_pos = bottom_pos / len(seg)

            if (slope > -0.001 and slope < 0.001 and
                0.3 < relative_pos < 0.7 and
                residuals[0] > 0 and residuals[-1] > 0 and residuals.min() < -np.std(residuals)):
                bottoms.append((start_idx, bottom_pos, end_idx))
                break

            # 圆顶：中间高两边低，残差呈上凸
            top_idx = seg.idxmax()
            top_pos = seg.index.get_loc(top_idx)
            relative_pos = top_pos / len(seg)

            if (slope > -0.001 and slope < 0.001 and
                0.3 < relative_pos < 0.7 and
                residuals[0] < 0 and residuals[-1] < 0 and residuals.max() > np.std(residuals)):
                tops.append((start_idx, top_pos, end_idx))
                break

    return {'圆底': bottoms, '圆顶': tops}


# ==================== 提示词20：红三兵 ====================

def detect_three_red(df):
    """
    检测红三兵。

    返回: {'红三兵': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    three_red = []

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        closes = close.iloc[idx0:idx2 + 1]
        opens = open_p.iloc[idx0:idx2 + 1]
        highs_seg = high.iloc[idx0:idx2 + 1]
        bodies = body.iloc[idx0:idx2 + 1]

        all_bullish = (closes > opens).all()
        ascending = (closes.iloc[0] < closes.iloc[1] < closes.iloc[2])
        upper_shadows = highs_seg - pd.concat([opens, closes], axis=1).max(axis=1)
        short_upper = (upper_shadows / bodies.replace(0, np.nan) < 0.10).all()
        open_in_prev = (
            (opens.iloc[1] >= opens.iloc[0]) and (opens.iloc[1] <= closes.iloc[0])
            and (opens.iloc[2] >= opens.iloc[1]) and (opens.iloc[2] <= closes.iloc[1])
        )

        if all_bullish and ascending and short_upper and open_in_prev:
            three_red.append((idx0, idx1, idx2))

    return {'红三兵': three_red}


# ==================== 提示词21：三个白色武士 / 升势受阻 / 升势停顿 ====================

def detect_white_soldiers_variants(df):
    """
    检测三个白色武士、升势受阻和升势停顿。

    返回: {'三个白色武士': [(idx0, idx1, idx2), ...],
            '升势受阻': [(idx0, idx1, idx2), ...],
            '升势停顿': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    soldiers = []
    advance_block = []
    deliberation = []

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        closes = close.iloc[idx0:idx2 + 1]
        opens = open_p.iloc[idx0:idx2 + 1]
        highs_seg = high.iloc[idx0:idx2 + 1]
        lows_seg = low.iloc[idx0:idx2 + 1]
        bodies = body.iloc[idx0:idx2 + 1]

        all_bullish = (closes > opens).all()
        ascending = (closes.iloc[0] < closes.iloc[1] < closes.iloc[2])

        if not (all_bullish and ascending):
            continue

        upper_shadows = highs_seg - pd.concat([opens, closes], axis=1).max(axis=1)
        lower_shadows = pd.concat([opens, closes], axis=1).min(axis=1) - lows_seg

        # 三个白色武士：实体饱满大阳线，下影线短，开盘在实体中上部
        big_bodies = (bodies > avg_body_20.iloc[idx0:idx2 + 1]).all()
        short_lower = (lower_shadows / bodies.replace(0, np.nan) < 0.10).all()
        open_in_upper = (
            (opens.iloc[1] >= opens.iloc[0] + (closes.iloc[0] - opens.iloc[0]) * 0.5)
            and (opens.iloc[2] >= opens.iloc[1] + (closes.iloc[1] - opens.iloc[1]) * 0.5)
        )
        if big_bodies and short_lower and open_in_upper:
            soldiers.append((idx0, idx1, idx2))
            continue

        # 升势受阻：实体递缩，上影线递长
        shrinking = bodies.iloc[0] > bodies.iloc[1] > bodies.iloc[2]
        upper_growing = upper_shadows.iloc[0] < upper_shadows.iloc[1] < upper_shadows.iloc[2]
        if shrinking and upper_growing:
            advance_block.append((idx0, idx1, idx2))
            continue

        # 升势停顿：前2根大/中阳，第3根小实体；三根均为阳线，
        # 且至少带有可见上下影线，体现声势受阻而非三根光头阳线。
        has_shadow = (
            ((upper_shadows / (highs_seg - lows_seg).replace(0, np.nan)) > 0.05)
            | ((lower_shadows / (highs_seg - lows_seg).replace(0, np.nan)) > 0.05)
        ).all()
        if (bodies.iloc[0] > avg_body_20.iloc[idx0] and
            bodies.iloc[1] > avg_body_20.iloc[idx1] and
            bodies.iloc[2] < avg_body_20.iloc[idx2] * 0.5 and
            has_shadow):
            deliberation.append((idx0, idx1, idx2))

    return {
        '三个白色武士': soldiers,
        '升势受阻': advance_block,
        '升势停顿': deliberation,
    }


# ==================== 趋势形态通用辅助函数 ====================

def _linear_trend_stats(values):
    """计算序列的线性回归斜率和R²。"""
    x = np.arange(len(values))
    if len(values) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    ss_res = ((values - fitted) ** 2).sum()
    ss_tot = ((values - values.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return slope, r_squared


def _classify_body(body, avg_body_20):
    """将实体分类为大/中/小/十字星。"""
    ratio = body / avg_body_20.replace(0, np.nan)
    return pd.cut(ratio, bins=[-np.inf, 0.08, 0.8, 2.0, np.inf],
                  labels=['十字星', '小实体', '中实体', '大实体'])


def _trend_body_ok(df, start, end, avg_body_20):
    """
    趋势形态实体分类约束：区间内大实体与十字星占比均不超过 30%，
    保证趋势由连续中小实体主导。
    """
    if end <= start or avg_body_20 is None or avg_body_20.empty:
        return True
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']
    seg_body = (close.iloc[start:end + 1] - open_p.iloc[start:end + 1]).abs()
    seg_avg = avg_body_20.iloc[start:end + 1]
    if len(seg_body) == 0 or seg_avg.isna().all():
        return True
    classes = _classify_body(seg_body, seg_avg)
    big_ratio = (classes == '大实体').sum() / len(classes)
    doji_ratio = (classes == '十字星').sum() / len(classes)
    return big_ratio <= 0.3 and doji_ratio <= 0.3


# ==================== 提示词22：冉冉上升形 ====================

def detect_ran_ran_rising(df, min_bars=8, small_ratio=0.7, bull_ratio=0.7):
    """
    检测冉冉上升形，返回 [(start_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - min_bars):
        for end in range(start + min_bars, min(start + 30, n)):
            seg = df.iloc[start:end + 1]
            seg_body = body.iloc[start:end + 1]
            seg_avg = avg_body_20.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]

            small_count = (seg_body < seg_avg * 0.8).sum()
            big_count = (seg_body > seg_avg * 2.0).sum()
            bull_count = (seg_close > open_p.iloc[start:end + 1]).sum()
            total = len(seg)

            if small_count / total < small_ratio or big_count > 0:
                continue
            if bull_count / total < bull_ratio:
                continue

            slope, r2 = _linear_trend_stats(seg_close.values)
            if slope <= 0 or r2 < 0.3:
                continue

            results.append((start, end, float(r2)))
            break

    # 相邻合并并限制频率
    results = _merge_ranges(results)
    return results[:3]


# ==================== 提示词23：绵绵阴跌形 ====================

def detect_mian_mian_decline(df, min_bars=8, small_ratio=0.7, bear_ratio=0.7):
    """
    检测绵绵阴跌形，返回 [(start_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - min_bars):
        for end in range(start + min_bars, min(start + 30, n)):
            seg_body = body.iloc[start:end + 1]
            seg_avg = avg_body_20.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]

            small_count = (seg_body < seg_avg * 0.8).sum()
            big_count = (seg_body > seg_avg * 2.0).sum()
            bear_count = (seg_close < open_p.iloc[start:end + 1]).sum()
            total = len(seg_body)

            if small_count / total < small_ratio or big_count > 0:
                continue
            if bear_count / total < bear_ratio:
                continue

            slope, r2 = _linear_trend_stats(seg_close.values)
            if slope >= 0 or r2 < 0.3:
                continue

            results.append((start, end, float(r2)))
            break

    results = _merge_ranges(results)
    return results[:3]


# ==================== 提示词24：徐缓上升形 ====================

def detect_xuhuan_rising(df):
    """
    检测徐缓上升形，返回 [(start_idx, phase_change_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - 8):
        # 寻找后期加速突破点
        for phase_change in range(start + 5, min(start + 15, n - 1)):
            end = min(phase_change + 2, n - 1)

            # 初期小阳线阶段
            early_body = body.iloc[start:phase_change]
            early_avg = avg_body_20.iloc[start:phase_change]
            early_bull = close.iloc[start:phase_change] > open_p.iloc[start:phase_change]

            if len(early_body) < 5:
                continue
            if (early_body > early_avg * 0.8).sum() / len(early_body) > 0.3:
                continue
            if early_bull.sum() / len(early_body) < 0.7:
                continue

            # 后期中/大阳线突破
            late_body = body.iloc[phase_change:end + 1]
            late_avg = avg_body_20.iloc[phase_change:end + 1]
            late_bull = close.iloc[phase_change:end + 1] > open_p.iloc[phase_change:end + 1]

            if (late_body > late_avg * 1.5).sum() < len(late_body):
                continue
            if not late_bull.all():
                continue

            # 整体趋势向上
            slope, r2 = _linear_trend_stats(close.iloc[start:end + 1].values)
            if slope <= 0 or r2 < 0.3:
                continue

            results.append((start, phase_change, end, float(r2)))
            break

    return results[:3]


# ==================== 提示词25：徐缓下降形 ====================

def detect_xuhuan_decline(df):
    """
    检测徐缓下降形，返回 [(start_idx, phase_change_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - 8):
        for phase_change in range(start + 5, min(start + 15, n - 1)):
            end = min(phase_change + 2, n - 1)

            early_body = body.iloc[start:phase_change]
            early_avg = avg_body_20.iloc[start:phase_change]
            early_bear = close.iloc[start:phase_change] < open_p.iloc[start:phase_change]

            if len(early_body) < 5:
                continue
            if (early_body > early_avg * 0.8).sum() / len(early_body) > 0.3:
                continue
            if early_bear.sum() / len(early_body) < 0.7:
                continue

            late_body = body.iloc[phase_change:end + 1]
            late_avg = avg_body_20.iloc[phase_change:end + 1]
            late_bear = close.iloc[phase_change:end + 1] < open_p.iloc[phase_change:end + 1]

            if (late_body > late_avg * 1.5).sum() < len(late_body):
                continue
            if not late_bear.all():
                continue

            slope, r2 = _linear_trend_stats(close.iloc[start:end + 1].values)
            if slope >= 0 or r2 < 0.3:
                continue

            results.append((start, phase_change, end, float(r2)))
            break

    return results[:3]


# ==================== 提示词27：上升抵抗形 ====================

def detect_rising_resistance(df, min_bars=5):
    """
    检测上升受阻形，返回 [(start_idx, end_idx, confidence), ...]。

    上升受阻不是连续跳高、收盘一路抬升。它要求先有上涨背景，随后
    出现阴阳交错、净推进衰减、上影线/回撤增加的阻滞区间。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    body = (close - open_p).abs()
    amplitude = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)

    results = []
    for start in range(max(6, min_bars), n - min_bars + 1):
        prior = close.iloc[max(0, start - 12):start]
        if len(prior) < 5 or float(prior.iloc[0]) == 0:
            continue
        prior_return = float(prior.iloc[-1] / prior.iloc[0] - 1.0)
        if prior_return < 0.025:
            continue
        for end in range(start + min_bars - 1, min(start + 12, n)):
            seg_open = open_p.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]
            seg_body = body.iloc[start:end + 1]
            seg_amp = amplitude.iloc[start:end + 1]
            seg_upper = upper_shadow.iloc[start:end + 1]
            total = len(seg_close)
            bull_count = int((seg_close > seg_open).sum())
            bear_count = int((seg_close < seg_open).sum())
            if bull_count == 0 or bear_count == 0:
                continue
            net_progress = float((seg_close.iloc[-1] - seg_close.iloc[0]) / max(abs(seg_close.iloc[0]), 1e-9))
            body_shrink = float(seg_body.iloc[-2:].mean()) <= float(seg_body.iloc[:2].mean()) * 0.85
            upper_pressure = float((seg_upper / seg_amp).fillna(0.0).mean()) >= 0.22
            stalled = -0.025 <= net_progress <= 0.035
            if not (stalled and (body_shrink or upper_pressure)):
                continue
            confidence = min(
                0.92,
                0.60
                + min(prior_return, 0.12)
                + 0.08 * (min(bull_count, bear_count) / max(total, 1))
                + (0.08 if upper_pressure else 0.0),
            )
            results.append((start, end, float(confidence)))
            break

    return _merge_ranges(results)[:3]


# ==================== 提示词28：下降抵抗形 ====================

def detect_falling_resistance(df, min_bars=5):
    """
    检测下降抵抗形，返回 [(start_idx, end_idx, confidence), ...]。

    必须先有下降背景，再出现阴阳交错、净下跌衰减及下影线/承接；
    单根小阳线夹在连续下跌中不构成下降抵抗。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    body = (close - open_p).abs()
    amplitude = (high - low).replace(0, np.nan)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    results = []
    for start in range(max(6, min_bars), n - min_bars + 1):
        prior = close.iloc[max(0, start - 12):start]
        if len(prior) < 5 or float(prior.iloc[0]) == 0:
            continue
        prior_return = float(prior.iloc[-1] / prior.iloc[0] - 1.0)
        if prior_return > -0.025:
            continue
        for end in range(start + min_bars - 1, min(start + 12, n)):
            seg_open = open_p.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]
            seg_body = body.iloc[start:end + 1]
            seg_amp = amplitude.iloc[start:end + 1]
            seg_lower = lower_shadow.iloc[start:end + 1]
            total = len(seg_close)
            bull_count = int((seg_close > seg_open).sum())
            bear_count = int((seg_close < seg_open).sum())
            if bull_count == 0 or bear_count == 0:
                continue
            net_progress = float((seg_close.iloc[-1] - seg_close.iloc[0]) / max(abs(seg_close.iloc[0]), 1e-9))
            body_shrink = float(seg_body.iloc[-2:].mean()) <= float(seg_body.iloc[:2].mean()) * 0.85
            lower_support = float((seg_lower / seg_amp).fillna(0.0).mean()) >= 0.22
            stalled = -0.035 <= net_progress <= 0.025
            if not (stalled and (body_shrink or lower_support)):
                continue
            confidence = min(
                0.92,
                0.60
                + min(abs(prior_return), 0.12)
                + 0.08 * (min(bull_count, bear_count) / max(total, 1))
                + (0.08 if lower_support else 0.0),
            )
            results.append((start, end, float(confidence)))
            break

    return _merge_ranges(results)[:3]


# ==================== 提示词30：稳步上涨形 ====================

def detect_steady_rising(df, min_bars=6, bull_ratio=0.6):
    """
    检测稳步上涨形，返回 [(start_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - min_bars):
        for end in range(start + min_bars, min(start + 30, n)):
            seg_body = body.iloc[start:end + 1]
            seg_avg = avg_body_20.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]
            seg_open = open_p.iloc[start:end + 1]

            mid_count = ((seg_body >= seg_avg * 0.8) & (seg_body < seg_avg * 2.0)).sum()
            bull_count = (seg_close > seg_open).sum()
            total = len(seg_body)

            if mid_count / total < 0.6 or bull_count / total < bull_ratio:
                continue

            # 阴线不连续超过2根
            bear_runs = (seg_close < seg_open).astype(int)
            max_bear_run = 0
            current = 0
            for v in bear_runs:
                if v:
                    current += 1
                    max_bear_run = max(max_bear_run, current)
                else:
                    current = 0
            if max_bear_run > 2:
                continue

            slope, r2 = _linear_trend_stats(seg_close.values)
            if slope <= 0 or r2 < 0.3:
                continue

            results.append((start, end, float(r2)))
            break

    return _merge_ranges(results)[:3]


# ==================== 提示词31：下跌不止形 ====================

def detect_steady_decline(df, min_bars=6, bear_ratio=0.6):
    """
    检测下跌不止形，返回 [(start_idx, end_idx, confidence), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for start in range(n - min_bars):
        for end in range(start + min_bars, min(start + 30, n)):
            seg_body = body.iloc[start:end + 1]
            seg_avg = avg_body_20.iloc[start:end + 1]
            seg_close = close.iloc[start:end + 1]
            seg_open = open_p.iloc[start:end + 1]

            mid_count = ((seg_body >= seg_avg * 0.8) & (seg_body < seg_avg * 2.0)).sum()
            bear_count = (seg_close < seg_open).sum()
            total = len(seg_body)

            if mid_count / total < 0.6 or bear_count / total < bear_ratio:
                continue

            bull_runs = (seg_close > seg_open).astype(int)
            max_bull_run = 0
            current = 0
            for v in bull_runs:
                if v:
                    current += 1
                    max_bull_run = max(max_bull_run, current)
                else:
                    current = 0
            if max_bull_run > 2:
                continue

            slope, r2 = _linear_trend_stats(seg_close.values)
            if slope >= 0 or r2 < 0.3:
                continue

            results.append((start, end, float(r2)))
            break

    return _merge_ranges(results)[:3]


# ==================== 提示词32：高开出逃形 ====================

def detect_high_open_escape(df):
    """
    检测高开出逃形，返回 Series[bool]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_close_20 = close.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)

    rolling_high_5 = high.rolling(window=5, min_periods=3).max()

    is_big_bear = body > avg_body_20 * 1.8
    high_open = open_p > rolling_high_5 * (1 + avg_amp_20 / avg_close_20 * 0.5)
    no_upper_shadow = upper_shadow / amplitude.replace(0, np.nan) < 0.08
    close_near_low = (close - low) / amplitude.replace(0, np.nan) < 0.15

    return is_big_bear & high_open & no_upper_shadow & close_near_low


# ==================== 提示词33：下探上涨形 ====================

def detect_low_open_surge(df):
    """
    检测下探上涨形，返回 Series[bool]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_close_20 = close.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    rolling_low_5 = low.rolling(window=5, min_periods=3).min()

    is_big_bull = body > avg_body_20 * 1.8
    low_open = open_p < rolling_low_5 * (1 - avg_amp_20 / avg_close_20 * 0.5)
    no_lower_shadow = lower_shadow / amplitude.replace(0, np.nan) < 0.08
    close_near_high = (high - close) / amplitude.replace(0, np.nan) < 0.15

    return is_big_bull & low_open & no_lower_shadow & close_near_high


# ==================== 提示词36：下跌三颗星 ====================

def detect_falling_three_stars(df):
    """
    检测下跌三颗星，返回 [(idx0, idx1, idx2, idx3, idx4), ...]。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for i in range(n - 4):
        idx0, idx1, idx2, idx3, idx4 = i, i + 1, i + 2, i + 3, i + 4

        first_bear = close.iloc[idx0] < open_p.iloc[idx0]
        first_big = body.iloc[idx0] > avg_body_20.iloc[idx0] * 1.5
        mid_high = high.iloc[idx1:idx4].max()
        mid_low = low.iloc[idx1:idx4].min()
        mid_small = (body.iloc[idx1:idx4] < avg_body_20.iloc[idx1:idx4] * 0.8).all()
        last_bear = close.iloc[idx4] < open_p.iloc[idx4]
        last_big = body.iloc[idx4] > avg_body_20.iloc[idx4] * 1.5

        if (first_bear and first_big and
            mid_small and
            mid_high < low.iloc[idx0] and
            last_bear and last_big and
            close.iloc[idx4] < mid_low):
            results.append((idx0, idx1, idx2, idx3, idx4))

    return results


# ==================== 提示词A：多方尖兵 ====================

def detect_bullish_probe(df):
    """
    检测多方尖兵，分草稿和确认两步。
    返回: [(first_idx, low_idx, break_idx, confidence), ...]
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_close_20 = close.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    results = []

    for first_idx in range(n - 8):
        first_bull = close.iloc[first_idx] > open_p.iloc[first_idx]
        first_body = body.iloc[first_idx]
        upper_shadow = high.iloc[first_idx] - max(open_p.iloc[first_idx], close.iloc[first_idx])

        if not first_bull or first_body < avg_body_20.iloc[first_idx]:
            continue

        # 长上影线判定
        if amplitude.iloc[first_idx] == 0:
            continue
        upper_ratio = upper_shadow / amplitude.iloc[first_idx]
        if not (upper_ratio > 0.5 or upper_ratio > 0.75):
            continue

        for break_idx in range(first_idx + 3, min(first_idx + 10, n)):
            mid_seg = df.iloc[first_idx + 1:break_idx]
            if len(mid_seg) < 2:
                continue
            if low.iloc[first_idx + 1:break_idx].min() < open_p.iloc[first_idx]:
                continue

            last_bull = close.iloc[break_idx] > open_p.iloc[break_idx]
            last_upper_shadow = high.iloc[break_idx] - max(open_p.iloc[break_idx], close.iloc[break_idx])
            last_upper_ratio = last_upper_shadow / amplitude.iloc[break_idx] if amplitude.iloc[break_idx] != 0 else 1

            threshold = high.iloc[first_idx] * (1 + avg_amp_20.iloc[break_idx] / avg_close_20.iloc[break_idx] * 0.3)

            if (last_bull and last_upper_ratio < 0.10 and
                close.iloc[break_idx] > threshold):
                # 后续5根确认
                if break_idx + 5 < n:
                    future_close = close.iloc[break_idx + 1:break_idx + 6]
                    if future_close.is_monotonic_increasing or future_close.iloc[-1] >= close.iloc[break_idx]:
                        results.append((first_idx, break_idx - 1, break_idx, 1.0))
                        break

    return results


# ==================== 提示词B：空方尖兵 ====================

def detect_bearish_probe(df):
    """
    检测空方尖兵，分草稿和确认两步。
    返回: [(first_idx, high_idx, break_idx, confidence), ...]
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_close_20 = close.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = (high - low).rolling(window=min(20, n), min_periods=5).mean()

    results = []

    for first_idx in range(n - 8):
        first_bear = close.iloc[first_idx] < open_p.iloc[first_idx]
        first_body = body.iloc[first_idx]
        lower_shadow = min(open_p.iloc[first_idx], close.iloc[first_idx]) - low.iloc[first_idx]

        if not first_bear or first_body < avg_body_20.iloc[first_idx]:
            continue

        amp = high.iloc[first_idx] - low.iloc[first_idx]
        if amp == 0:
            continue
        lower_ratio = lower_shadow / amp
        if not (lower_ratio > 0.5 or lower_ratio > 0.75):
            continue

        for break_idx in range(first_idx + 3, min(first_idx + 10, n)):
            mid_seg = df.iloc[first_idx + 1:break_idx]
            if len(mid_seg) < 2:
                continue
            if high.iloc[first_idx + 1:break_idx].max() > open_p.iloc[first_idx]:
                continue

            last_bear = close.iloc[break_idx] < open_p.iloc[break_idx]
            last_lower_shadow = min(open_p.iloc[break_idx], close.iloc[break_idx]) - low.iloc[break_idx]
            last_amp = high.iloc[break_idx] - low.iloc[break_idx]
            last_lower_ratio = last_lower_shadow / last_amp if last_amp != 0 else 1

            threshold = low.iloc[first_idx] * (1 - avg_amp_20.iloc[break_idx] / avg_close_20.iloc[break_idx] * 0.3)

            if (last_bear and last_lower_ratio < 0.10 and
                close.iloc[break_idx] < threshold):
                if break_idx + 5 < n:
                    future_close = close.iloc[break_idx + 1:break_idx + 6]
                    if future_close.is_monotonic_decreasing or future_close.iloc[-1] <= close.iloc[break_idx]:
                        results.append((first_idx, break_idx - 1, break_idx, 1.0))
                        break

    return results


# ==================== 提示词C/D：塔形底/塔形顶（严格版） ====================

def detect_tower_bottom(df, min_mid=3, max_mid=8):
    """
    检测塔形底（严格版，含经典+变体）。
    返回: [(left_idx, mid_start, mid_end, right_idx, type, confidence), ...]
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = (high - low).rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for left_idx in range(n - min_mid - 2):
        left_bear = close.iloc[left_idx] < open_p.iloc[left_idx]
        left_big = body.iloc[left_idx] > avg_body_20.iloc[left_idx] * 1.5
        if not (left_bear and left_big):
            continue

        for mid_end in range(left_idx + min_mid + 1, min(left_idx + max_mid + 2, n - 1)):
            mid_start = left_idx + 1
            mid_seg = df.iloc[mid_start:mid_end]
            if len(mid_seg) < min_mid:
                continue

            mid_bodies = body.iloc[mid_start:mid_end]
            mid_avg = avg_body_20.iloc[mid_start:mid_end]
            if (mid_bodies > mid_avg * 0.8).any():
                continue

            mid_high = high.iloc[mid_start:mid_end].max()
            mid_low = low.iloc[mid_start:mid_end].min()
            if mid_high > high.iloc[left_idx] or mid_low < low.iloc[left_idx]:
                continue

            right_idx = mid_end
            right_bull = close.iloc[right_idx] > open_p.iloc[right_idx]
            right_big = body.iloc[right_idx] > avg_body_20.iloc[right_idx] * 1.5
            if not (right_bull and right_big):
                continue
            if close.iloc[right_idx] <= mid_high:
                continue
            # 塔形底的右侧阳线应从整理区间内启动。若它以明显跳空、
            # 远高于整理区间重心的方式出现，更接近突破缺口/单根大阳线，
            # 不能把整个片段追认成塔形底。
            reference_range = avg_amp_20.iloc[right_idx]
            if not np.isfinite(reference_range) or reference_range <= 0:
                reference_range = max(float(mid_high - mid_low), 1e-12)
            if (
                open_p.iloc[right_idx] > mid_high + reference_range * 0.60
                or close.iloc[right_idx] - mid_high
                > max(reference_range * 1.80, abs(close.iloc[right_idx]) * 0.04)
            ):
                continue

            amp = mid_high - mid_low
            ptype = 'classic' if amp < avg_amp_20.iloc[mid_start] * 0.5 else 'variant'
            results.append((left_idx, mid_start, mid_end - 1, right_idx, ptype, 1.0))
            break

    return results


def detect_tower_top(df, min_mid=3, max_mid=8):
    """
    检测塔形顶（严格版，含经典+变体）。
    返回: [(left_idx, mid_start, mid_end, right_idx, type, confidence), ...]
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = (high - low).rolling(window=min(20, n), min_periods=5).mean()

    results = []
    for left_idx in range(n - min_mid - 2):
        left_bull = close.iloc[left_idx] > open_p.iloc[left_idx]
        left_big = body.iloc[left_idx] > avg_body_20.iloc[left_idx] * 1.5
        if not (left_bull and left_big):
            continue

        for mid_end in range(left_idx + min_mid + 1, min(left_idx + max_mid + 2, n - 1)):
            mid_start = left_idx + 1
            mid_seg = df.iloc[mid_start:mid_end]
            if len(mid_seg) < min_mid:
                continue

            mid_bodies = body.iloc[mid_start:mid_end]
            mid_avg = avg_body_20.iloc[mid_start:mid_end]
            if (mid_bodies > mid_avg * 0.8).any():
                continue

            mid_high = high.iloc[mid_start:mid_end].max()
            mid_low = low.iloc[mid_start:mid_end].min()
            if mid_high > high.iloc[left_idx] or mid_low < low.iloc[left_idx]:
                continue

            right_idx = mid_end
            right_bear = close.iloc[right_idx] < open_p.iloc[right_idx]
            right_big = body.iloc[right_idx] > avg_body_20.iloc[right_idx] * 1.5
            if not (right_bear and right_big):
                continue
            if close.iloc[right_idx] >= mid_low:
                continue

            amp = mid_high - mid_low
            ptype = 'classic' if amp < avg_amp_20.iloc[mid_start] * 0.5 else 'variant'
            results.append((left_idx, mid_start, mid_end - 1, right_idx, ptype, 1.0))
            break

    return results


# ==================== 提示词E：盘旋段判定（供复用） ====================

def is_coiling(df, start_idx, end_idx):
    """
    判断一段是否是小阴小阳盘旋。
    返回: bool
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    seg = df.iloc[start_idx:end_idx + 1]
    seg_body = body.iloc[start_idx:end_idx + 1]
    seg_avg_body = avg_body_20.iloc[start_idx:end_idx + 1]
    seg_amp = amplitude.iloc[start_idx:end_idx + 1]
    seg_avg_amp = avg_amp_20.iloc[start_idx:end_idx + 1]
    seg_close = close.iloc[start_idx:end_idx + 1]

    small_bodies = (seg_body < seg_avg_body * 0.8).all()
    narrow_amp = (seg_amp.max() - seg_amp.min()) < seg_avg_amp.mean() * 0.6

    x = np.arange(len(seg_close))
    slope, r2 = _linear_trend_stats(seg_close.values)

    return small_bodies and narrow_amp and r2 < 0.2


# ==================== 连续跳空三阳线 / 三阴线 ====================

def detect_three_gap_rise(df):
    """检测连续跳空三阳线，允许影线重叠但要求实体逐根跳空。"""
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']
    results = []
    for i in range(n - 2):
        o0, o1, o2 = open_p.iloc[i:i + 3]
        c0, c1, c2 = close.iloc[i:i + 3]
        if c0 > o0 and c1 > o1 and c2 > o2 and o1 > c0 and o2 > c1:
            results.append((i, i + 1, i + 2))
    return {'连续跳空三阳线': results}

def detect_three_gap_decline(df):
    """
    检测连续跳空三阴线，要求后续开盘低于前一根收盘，允许影线重叠。
    """
    n = len(df)
    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    results = []
    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        all_bear = (
            close.iloc[idx0] < open_p.iloc[idx0]
            and close.iloc[idx1] < open_p.iloc[idx1]
            and close.iloc[idx2] < open_p.iloc[idx2]
        )
        gap1 = open_p.iloc[idx1] < close.iloc[idx0]
        gap2 = open_p.iloc[idx2] < close.iloc[idx1]

        if all_bear and gap1 and gap2:
            results.append((idx0, idx1, idx2))

    return {'连续跳空三阴线': results}


# ==================== 提示词G：加速度线 ====================

def detect_acceleration(df, mode='backtest'):
    """
    检测加速度线。
    mode='backtest': 确认触顶/触底后回溯标记
    mode='realtime': 标记加速嫌疑

    返回: [{'start': idx, 'end': idx, 'type': 'up'/'down', 'confidence': float}, ...]
    """
    n = len(df)
    close = df['close'] if 'close' in df.columns else df['Close']
    open_p = df['open'] if 'open' in df.columns else df['Open']
    body = (close - open_p).abs()

    body_ma5 = body.rolling(window=5, min_periods=3).mean()

    accelerations = []
    for i in range(5, n - 5):
        # 斜率加速：close 5日斜率递增
        x = np.arange(5)
        slope_now, _ = _linear_trend_stats(close.iloc[i - 4:i + 1].values)
        slope_prev, _ = _linear_trend_stats(close.iloc[i - 9:i - 4].values)

        body_accel = body_ma5.iloc[i] > body_ma5.iloc[i - 2]
        slope_accel = abs(slope_now) > abs(slope_prev) * 1.2

        if not (body_accel and slope_accel):
            continue

        direction = 'up' if slope_now > 0 else 'down'

        if mode == 'realtime':
            accelerations.append({'start': i - 4, 'end': i, 'type': direction, 'confidence': 0.5})
        else:
            # backtest模式：后续需出现反转
            future = close.iloc[i + 1:i + 6]
            if direction == 'up' and (future < close.iloc[i]).any():
                accelerations.append({'start': i - 4, 'end': i, 'type': direction, 'confidence': 0.8})
            elif direction == 'down' and (future > close.iloc[i]).any():
                accelerations.append({'start': i - 4, 'end': i, 'type': direction, 'confidence': 0.8})

    return accelerations[:3]


# ==================== 提示词H：缺口回补趋势判定 ====================

def detect_window_support_resistance(df, lookforward=5):
    """
    检测跳空缺口，并判定回补后的支撑/阻力效果。

    返回: DataFrame，每行一个缺口。
    """
    gaps = detect_gaps(df)
    if gaps.empty:
        return gaps

    n = len(df)
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    support_held = []
    for _, row in gaps.iterrows():
        fill_idx = df.index.get_loc(row['fill_date']) if pd.notna(row['fill_date']) else None
        if fill_idx is None or fill_idx + lookforward >= n:
            support_held.append(None)
            continue

        future_close = close.iloc[fill_idx + 1:fill_idx + 1 + lookforward]
        future_low = low.iloc[fill_idx + 1:fill_idx + 1 + lookforward]
        future_high = high.iloc[fill_idx + 1:fill_idx + 1 + lookforward]

        if row['gap_type'] == 'up':
            # 上升缺口：支撑有效 = 后续不再跌破回补那根最低价
            held = (future_low >= low.iloc[fill_idx]).all()
        else:
            # 下降缺口：阻力有效 = 后续不再突破回补那根最高价
            held = (future_high <= high.iloc[fill_idx]).all()

        support_held.append(bool(held))

    gaps['support_held'] = support_held
    return gaps


# ==================== 提示词26：高位并排阳线 / 低位并排阳线 ====================

def detect_side_by_side_gap(df, body_similarity=0.30):
    """
    检测高位并排阳线和低位并排阳线。

    与提示词12的"并排阳线/阴线"区别：本形态必须包含跳空缺口。
    高位并排阳线 = 上涨途中向上跳空后两根阳线并排，缺口未回补；
    低位并排阳线 = 下跌途中向下跳空后两根阳线并排，缺口未回补。

    返回: {'高位并排阳线': [(gap_idx, bar1_idx, bar2_idx), ...],
            '低位并排阳线': [(gap_idx, bar1_idx, bar2_idx), ...]}
    """
    n = len(df)
    high_side = []
    low_side = []

    if n < 4:
        return {'高位并排阳线': high_side, '低位并排阳线': low_side}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    # 20日均线作为趋势方向自适应基准
    ma20 = close.rolling(window=min(20, n), min_periods=5).mean()

    for i in range(1, n - 2):
        # i 为跳空K线，与 i-1 之间形成缺口
        up_gap = low.iloc[i] > high.iloc[i - 1]
        down_gap = high.iloc[i] < low.iloc[i - 1]

        if not (up_gap or down_gap):
            continue

        bar1, bar2 = i, i + 1

        # 并排的两根必须是阳线
        if not (close.iloc[bar1] > open_p.iloc[bar1] and close.iloc[bar2] > open_p.iloc[bar2]):
            continue

        # 实体大小相近：差异小于 body_similarity（默认30%）
        body1 = abs(close.iloc[bar1] - open_p.iloc[bar1])
        body2 = abs(close.iloc[bar2] - open_p.iloc[bar2])
        if body1 == 0 or abs(body1 - body2) / body1 > body_similarity:
            continue

        if up_gap:
            # 上升缺口未被回补：后续两根K线的低点都未跌破缺口下沿（前高）
            gap_bottom = high.iloc[i - 1]
            if low.iloc[bar1:bar2 + 1].min() <= gap_bottom:
                continue
            # 位于上涨途中：收盘价在均线之上
            if close.iloc[i] > ma20.iloc[i]:
                high_side.append((i, bar1, bar2))
        else:
            # 下降缺口未被回补：后续两根K线的高点都未升破缺口上沿（前低）
            gap_top = low.iloc[i - 1]
            if high.iloc[bar1:bar2 + 1].max() >= gap_top:
                continue
            # 位于下跌途中：收盘价在均线之下
            if close.iloc[i] < ma20.iloc[i]:
                low_side.append((i, bar1, bar2))

    return {'高位并排阳线': high_side, '低位并排阳线': low_side}


# ==================== 提示词29：好友反攻 / 淡友反攻 ====================

def detect_meeting_lines(df, close_tolerance=None):
    """
    检测好友反攻（看涨）和淡友反攻（看跌）。

    判定逻辑：
    - 好友反攻：前大阴 + 后大阳，后阳收盘 ≈ 前阴收盘
    - 淡友反攻：前大阳 + 后大阴，后阴收盘 ≈ 前阳收盘
    - close_tolerance 自适应：默认取 avg_amp_20 * 5%

    返回: {'好友反攻': Series, '淡友反攻': Series}
    """
    n = len(df)
    result_bull = pd.Series(False, index=df.index)
    result_bear = pd.Series(False, index=df.index)

    if n < 2:
        return {'好友反攻': result_bull, '淡友反攻': result_bear}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    if close_tolerance is None:
        # 默认容忍度为近期平均振幅的5%
        close_tolerance = avg_amp_20 * 0.05

    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    prev_body = body.shift(1)

    # 好友反攻：前大阴 + 后大阳，收盘接近前收盘
    prev_big_bear = (prev_body > avg_body_20.shift(1)) & (prev_close < prev_open)
    curr_big_bull = (body > avg_body_20) & (close > open_p)
    close_match_bull = (close - prev_close).abs() <= close_tolerance
    result_bull = prev_big_bear & curr_big_bull & close_match_bull

    # 淡友反攻：前大阳 + 后大阴，收盘接近前收盘
    prev_big_bull = (prev_body > avg_body_20.shift(1)) & (prev_close > prev_open)
    curr_big_bear = (body > avg_body_20) & (close < open_p)
    close_match_bear = (close - prev_close).abs() <= close_tolerance
    result_bear = prev_big_bull & curr_big_bear & close_match_bear

    return {'好友反攻': result_bull, '淡友反攻': result_bear}


# ==================== 提示词34：双飞乌鸦 ====================

def detect_two_crows(df, lookback=20):
    """
    检测双飞乌鸦（看跌反转）。

    判定逻辑：
    1. 出现在上涨末端（收盘价处于近期高位）
    2. 第一根为阳线（或小阴）
    3. 第二根跳空高开阴线，收盘仍在第一根实体范围内或上方
    4. 第三根阴线开盘更高、收盘更低，形成两只乌鸦叠罗汉

    返回: {'双飞乌鸦': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    result = []

    if n < 3:
        return {'双飞乌鸦': result}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    # 近期高位判定：收盘价高于 lookback 窗口的80%分位
    rolling_high = close.rolling(window=lookback, min_periods=5).quantile(0.80)

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2

        # 第一根至少收阳（允许小阴，但不允许大阴）
        first_bullish = close.iloc[idx0] >= open_p.iloc[idx0]

        # 第二根：跳空高开阴线
        second_bear = close.iloc[idx1] < open_p.iloc[idx1]
        second_gap_up = open_p.iloc[idx1] > close.iloc[idx0]
        # 第二根收盘未跌破第一根实体下沿，保持在前一根实体内或上方
        second_stay_above = close.iloc[idx1] >= open_p.iloc[idx0]

        # 第三根：阴线，开盘更高，收盘更低
        third_bear = close.iloc[idx2] < open_p.iloc[idx2]
        third_open_higher = open_p.iloc[idx2] > open_p.iloc[idx1]
        third_close_lower = close.iloc[idx2] < close.iloc[idx1]

        # 处于上涨末端
        in_high = close.iloc[idx0] >= rolling_high.iloc[idx0]

        if (first_bullish and second_bear and second_gap_up and second_stay_above and
                third_bear and third_open_higher and third_close_lower and in_high):
            result.append((idx0, idx1, idx2))

    return {'双飞乌鸦': result}


# ==================== 提示词35：阳线跛脚形 / 倒三阳 ====================

def detect_fake_yang(df, lookback=20):
    """
    检测"假阳线"陷阱：阳线跛脚形和倒三阳。

    判定逻辑：
    - 阳线跛脚形：连续3根阳线，但收盘价逐根降低。
    - 倒三阳：下降过程中出现的低开阳线序列；支持经典三阳线、
      两根低开阳线，以及“两根低开阳线夹一根阴线”的放宽版本。

    返回: {'阳线跛脚形': [(idx0, idx1, idx2), ...],
            '倒三阳': [(idx0, idx1, idx2), ...]}
    """
    n = len(df)
    limping = []
    inverted = []

    if n < 3:
        return {'阳线跛脚形': limping, '倒三阳': inverted}

    open_p = df['open'] if 'open' in df.columns else df['Open']
    close = df['close'] if 'close' in df.columns else df['Close']

    # 倒三阳的硬背景由FAE Phase-1滚动收益分位数和斜率判定；
    # 阳线跛脚形仍保留原有的高位条件，二者不要混用背景。
    quant_lookback = max(20, min(120, n))
    quant_context = adaptive_trend_context(close, lookback=quant_lookback)
    rolling_high = close.rolling(window=min(lookback, n), min_periods=5).quantile(0.80)

    for i in range(n - 2):
        idx0, idx1, idx2 = i, i + 1, i + 2
        c0, c1, c2 = close.iloc[idx0], close.iloc[idx1], close.iloc[idx2]
        o0, o1, o2 = open_p.iloc[idx0], open_p.iloc[idx1], open_p.iloc[idx2]

        if idx0 < 1:
            continue

        # 阳线跛脚形仍要求连续三根阳线、收盘逐级走低。
        in_high = c0 >= rolling_high.iloc[idx0]
        if (in_high and c0 > o0 and c1 > o1 and c2 > o2
                and c0 > c1 > c2):
            limping.append((idx0, idx1, idx2))

        in_decline = quant_context["state"].iloc[idx0] == "downtrend"
        if not in_decline:
            continue

        # 倒三阳经典版：三根连续低开阳线。
        three_bullish = c0 > o0 and c1 > o1 and c2 > o2
        if three_bullish and o1 < c0 and o2 < c1:
            inverted.append((idx0, idx2))
        else:
            # 倒三阳放宽版一：两根连续低开阳线即可。
            two_bullish = c0 > o0 and c1 > o1 and o1 < c0
            if two_bullish:
                inverted.append((idx0, idx1))

        # 倒三阳放宽版二：两根低开阳线中间夹一根阴线。
        bullish_bearish_bullish = (
            c0 > o0 and c1 < o1 and c2 > o2 and o2 < c0
        )
        if bullish_bearish_bullish:
            inverted.append((idx0, idx2))

    return {'阳线跛脚形': limping, '倒三阳': inverted}


# ==================== 提示词37：镊子线 / 尽头线 / 搓揉线 ====================

def detect_tweezers_exhaustion_knead(df, price_tolerance=None):
    """
    检测镊子线、尽头线和搓揉线。

    判定逻辑：
    - 顶部/底部镊子线：2根K线最高/最低价齐平，位于趋势末端
    - 三根镊子线：大→小→大，最高/最低价齐平
    - 顶部/底部尽头线：大K线后紧跟小K线，且小K线位于大K线影线顶端/底端
    - 搓揉线：倒T字线（长上影）与T字线（长下影）交替出现

    返回: {
        '顶部镊子线': Series,
        '底部镊子线': Series,
        '三根镊子线': Series,
        '顶部尽头线': Series,
        '底部尽头线': Series,
        '搓揉线': Series
    }
    """
    n = len(df)

    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    body = (close - open_p).abs()
    amplitude = high - low
    avg_body_20 = body.rolling(window=min(20, n), min_periods=5).mean()
    avg_amp_20 = amplitude.rolling(window=min(20, n), min_periods=5).mean()

    if price_tolerance is None:
        price_tolerance = avg_amp_20 * 0.003

    amp_safe = amplitude.replace(0, np.nan)
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low

    # 趋势位置：80/20分位判定高位/低位
    rolling_high = close.rolling(window=min(20, n), min_periods=5).quantile(0.80)
    rolling_low = close.rolling(window=min(20, n), min_periods=5).quantile(0.20)

    top_tweezers = pd.Series(False, index=df.index)
    bottom_tweezers = pd.Series(False, index=df.index)
    three_tweezers = pd.Series(False, index=df.index)
    top_exhaustion = pd.Series(False, index=df.index)
    bottom_exhaustion = pd.Series(False, index=df.index)
    knead = pd.Series(False, index=df.index)

    for i in range(1, n):
        # 顶部镊子线：2根最高价相同，前阳/中性后阴，位于高位
        high_match = abs(high.iloc[i] - high.iloc[i - 1]) <= price_tolerance.iloc[i]
        prev_not_bear = close.iloc[i - 1] >= open_p.iloc[i - 1]
        curr_bear = close.iloc[i] < open_p.iloc[i]
        in_high = close.iloc[i] >= rolling_high.iloc[i]
        if high_match and prev_not_bear and curr_bear and in_high:
            top_tweezers.iloc[i] = True

        # 底部镊子线：2根最低价相同，前阴/中性后阳，位于低位
        low_match = abs(low.iloc[i] - low.iloc[i - 1]) <= price_tolerance.iloc[i]
        prev_not_bull = close.iloc[i - 1] <= open_p.iloc[i - 1]
        curr_bull = close.iloc[i] > open_p.iloc[i]
        in_low = close.iloc[i] <= rolling_low.iloc[i]
        if low_match and prev_not_bull and curr_bull and in_low:
            bottom_tweezers.iloc[i] = True

        # 顶部尽头线：前大阳 + 当前小K线位于上影线顶端
        prev_big_bull = (body.iloc[i - 1] > avg_body_20.iloc[i - 1]) and (close.iloc[i - 1] > open_p.iloc[i - 1])
        curr_small = body.iloc[i] < avg_body_20.iloc[i] * 0.5
        prev_body_top = max(open_p.iloc[i - 1], close.iloc[i - 1])
        in_upper_shadow = (low.iloc[i] >= prev_body_top) and (high.iloc[i] <= high.iloc[i - 1])
        if prev_big_bull and curr_small and in_upper_shadow:
            top_exhaustion.iloc[i] = True

        # 底部尽头线：前大阴 + 当前小K线位于下影线底端
        prev_big_bear = (body.iloc[i - 1] > avg_body_20.iloc[i - 1]) and (close.iloc[i - 1] < open_p.iloc[i - 1])
        prev_body_bottom = min(open_p.iloc[i - 1], close.iloc[i - 1])
        in_lower_shadow = (high.iloc[i] <= prev_body_bottom) and (low.iloc[i] >= low.iloc[i - 1])
        if prev_big_bear and curr_small and in_lower_shadow:
            bottom_exhaustion.iloc[i] = True

        # 搓揉线：长上影线小实体 + 长下影线小实体，或反之
        small_prev = body.iloc[i - 1] / amp_safe.iloc[i - 1] < 0.25
        small_curr = body.iloc[i] / amp_safe.iloc[i] < 0.25
        long_upper_prev = upper_shadow.iloc[i - 1] / amp_safe.iloc[i - 1] > 0.40
        long_lower_prev = lower_shadow.iloc[i - 1] / amp_safe.iloc[i - 1] > 0.40
        long_upper_curr = upper_shadow.iloc[i] / amp_safe.iloc[i] > 0.40
        long_lower_curr = lower_shadow.iloc[i] / amp_safe.iloc[i] > 0.40

        knead_pattern1 = small_prev and long_upper_prev and small_curr and long_lower_curr
        knead_pattern2 = small_prev and long_lower_prev and small_curr and long_upper_curr
        if knead_pattern1 or knead_pattern2:
            knead.iloc[i] = True

    # 三根镊子线：大→小→大，最高/最低价齐平
    for i in range(2, n):
        idx0, idx1, idx2 = i - 2, i - 1, i
        big0 = body.iloc[idx0] > avg_body_20.iloc[idx0]
        big2 = body.iloc[idx2] > avg_body_20.iloc[idx2]
        small1 = body.iloc[idx1] < avg_body_20.iloc[idx1] * 0.5

        # 顶部：大阳 + 小K线 + 大阴，最高价齐平
        top_three = (big0 and close.iloc[idx0] > open_p.iloc[idx0] and
                     small1 and big2 and close.iloc[idx2] < open_p.iloc[idx2])
        # 底部：大阴 + 小K线 + 大阳，最低价齐平
        bottom_three = (big0 and close.iloc[idx0] < open_p.iloc[idx0] and
                        small1 and big2 and close.iloc[idx2] > open_p.iloc[idx2])

        high_match = (abs(high.iloc[idx0] - high.iloc[idx1]) <= price_tolerance.iloc[i] and
                      abs(high.iloc[idx1] - high.iloc[idx2]) <= price_tolerance.iloc[i])
        low_match = (abs(low.iloc[idx0] - low.iloc[idx1]) <= price_tolerance.iloc[i] and
                     abs(low.iloc[idx1] - low.iloc[idx2]) <= price_tolerance.iloc[i])

        if (top_three and high_match) or (bottom_three and low_match):
            three_tweezers.iloc[i] = True

    return {
        '顶部镊子线': top_tweezers,
        '底部镊子线': bottom_tweezers,
        '三根镊子线': three_tweezers,
        '顶部尽头线': top_exhaustion,
        '底部尽头线': bottom_exhaustion,
        '搓揉线': knead,
    }


def _ohlcv_arrays(df):
    """Return numeric OHLC series using either lower- or upper-case columns."""
    open_p = df['open'] if 'open' in df.columns else df['Open']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    return tuple(pd.to_numeric(x, errors='coerce').astype(float)
                 for x in (open_p, high, low, close))


def _candle_components(df):
    """Compute reusable body, shadow, range and rolling body statistics."""
    open_p, high, low, close = _ohlcv_arrays(df)
    body = (close - open_p).abs()
    amplitude = (high - low).clip(lower=0.0)
    upper = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower = pd.concat([open_p, close], axis=1).min(axis=1) - low
    avg_body = body.rolling(window=min(20, max(len(df), 1)), min_periods=3).median()
    avg_body = avg_body.replace(0, np.nan).fillna(body.median()).fillna(0.0)
    return open_p, high, low, close, body, amplitude, upper, lower, avg_body


def detect_engulfing(df, min_body_ratio=0.55):
    """Detect standard two-candle bullish/bearish engulfing (吞没/穿头破脚)."""
    if len(df) < 2:
        false = pd.Series(False, index=df.index)
        return {'看涨吞没': false.copy(), '看跌吞没': false.copy()}
    open_p, _, _, close, body, amplitude, _, _, avg_body = _candle_components(df)
    prev_open, prev_close = open_p.shift(1), close.shift(1)
    prev_body = body.shift(1)
    prev_bear = prev_close < prev_open
    prev_bull = prev_close > prev_open
    curr_bull = close > open_p
    curr_bear = close < open_p
    prev_valid = (
        (prev_body >= avg_body.shift(1).fillna(avg_body) * min_body_ratio)
        & (prev_body / amplitude.shift(1).replace(0, np.nan) >= 0.20)
    )
    curr_valid = (
        (body >= avg_body * min_body_ratio)
        & (body / amplitude.replace(0, np.nan) >= 0.20)
    )
    tol = close.abs() * 1e-4
    bullish = (
        prev_bear & curr_bull & prev_valid & curr_valid
        & (open_p <= prev_close + tol) & (close >= prev_open - tol)
    )
    bearish = (
        prev_bull & curr_bear & prev_valid & curr_valid
        & (open_p >= prev_close - tol) & (close <= prev_open + tol)
    )
    return {'看涨吞没': bullish, '看跌吞没': bearish}


def _merge_ranges(ranges):
    """合并重叠或相邻的区间。"""
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(ranges[0])]
    for r in ranges[1:]:
        if r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged[-1][2] = max(merged[-1][2], r[2]) if len(r) > 2 else merged[-1][2]
        else:
            merged.append(list(r))
    return [tuple(m) for m in merged]



# ==================== v7 统一入口 detect_all ====================

def detect_all(
    df,
    max_trend_per_direction=3,
    volume_col=None,
    diagnostics=None,
    strict=False,
):
    """
    统一 K 线形态检测入口。

    返回事件列表，每个事件为字典：
        {
            'name': str,        # 形态中文名
            'category': str,    # 'trend' | 'composite' | 'simple'
            'direction': str,   # 'bull' | 'bear' | 'neutral'
            'start': int,       # 起始索引（含）
            'end': int,         # 结束索引（含）
            'idx': int,         # 主要 K 线索引
            'confidence': float,
            'note': str,
        }

    参数:
        max_trend_per_direction: 同方向趋势形态最多保留数量（按 confidence 排序）
        volume_col: 成交量列名，None 时自动识别 Volume/volume/vol
    """
    import pandas as pd
    import numpy as np
    import inspect

    n = len(df)
    if n < 3:
        return []

    events = []

    def add_event(name, category, direction, start, end, confidence=0.5, note='', **extra):
        event = {
            'name': name,
            'category': category,
            'direction': direction,
            'start': max(0, int(start)),
            'end': min(n - 1, int(end)),
            'idx': min(n - 1, int(end)),
            'confidence': float(confidence),
            'note': note,
        }
        # Keep detector-specific evidence on the event instead of throwing it
        # away at the common-event boundary.  The FAE renderer uses these
        # fields to draw an actual gap zone (not merely a text label).
        event.update(extra)
        events.append(event)

    # 方向判断辅助
    _BULL_KEYWORDS = ['阳', '涨', '好友', '旭日', '刺透', '锤', '启明', '晨', '塔底', '岛底',
                      '白', '三红', '飞', '反攻', '上升', '上涨', '底部', '多头', '看涨']
    _BEAR_KEYWORDS = ['阴', '跌', '淡友', '乌云', '黄昏', '星', '塔顶', '岛顶', '黑', '三黑',
                      '乌鸦', '下降', '下跌', '顶部', '空头', '看跌']

    _EXPLICIT_DIRECTIONS = {
        'piercing': 'bull',
        'dark_cloud': 'bear',
        'tower_bottom': 'bull',
        'tower_top': 'bear',
        'bullish_probe': 'bull',
        'bearish_probe': 'bear',
        'ran_ran_rising': 'bull',
        'mian_mian_decline': 'bear',
        'xuhuan_rising': 'bull',
        'xuhuan_decline': 'bear',
        'steady_rising': 'bull',
        'steady_decline': 'bear',
        # 受阻形是反向预警，不等于已经反转。
        'rising_resistance': 'bear',
        'falling_resistance': 'bull',
        '并排阳线': 'bull',
        '并排阴线': 'bear',
        '两红加一黑': 'bull',
        '两黑加一红': 'bear',
        '阳线跛脚形': 'bear',
        '看涨孕线': 'bull',
        '看跌孕线': 'bear',
        '十字孕线': 'neutral',
        'T字线': 'neutral',
        '倒T字线': 'neutral',
        '上升覆盖线': 'bull',
        '下降覆盖线': 'bear',
        '低位并排阳线': 'bull',
        '倾盆大雨': 'bear',
        '双飞乌鸦': 'bear',
        '好友反攻': 'bull',
        '淡友反攻': 'bear',
        '平底': 'bull',
        '平顶': 'bear',
        '旭日东升': 'bull',
        '黑三兵': 'bear',
        'high_open_escape': 'bear',
        'low_open_surge': 'bull',
        'falling_three_stars': 'bear',
    }

    _SERIES_PATTERN_SPANS = {
        '看涨吞没': 2,
        '看跌吞没': 2,
        'piercing': 2,
        'dark_cloud': 2,
        '看涨孕线': 2,
        '看跌孕线': 2,
        '十字孕线': 2,
        '旭日东升': 2,
        '倾盆大雨': 2,
        '好友反攻': 2,
        '淡友反攻': 2,
        '顶部镊子线': 2,
        '底部镊子线': 2,
        '三根镊子线': 3,
        '顶部尽头线': 2,
        '底部尽头线': 2,
        '搓揉线': 2,
    }

    def dir_of(name):
        name = str(name)
        if name in _EXPLICIT_DIRECTIONS:
            return _EXPLICIT_DIRECTIONS[name]
        if any(k in name for k in _BULL_KEYWORDS):
            return 'bull'
        if any(k in name for k in _BEAR_KEYWORDS):
            return 'bear'
        return 'neutral'

    def add_series_bool(s, name, category, confidence=0.6):
        if not isinstance(s, pd.Series):
            return
        for idx_pos, val in s.items():
            if not bool(val):
                continue
            pos = df.index.get_loc(idx_pos)
            if isinstance(pos, slice):
                pos = pos.start if pos.start is not None else 0
            span = max(1, int(_SERIES_PATTERN_SPANS.get(str(name), 1)))
            add_event(name, category, dir_of(name), pos - span + 1, pos, confidence)

    def add_series_label(s, category, confidence=0.6):
        if not isinstance(s, pd.Series):
            return
        for idx_pos, val in s.items():
            if not isinstance(val, str) or not val.strip():
                continue
            pos = df.index.get_loc(idx_pos)
            if isinstance(pos, slice):
                pos = pos.start if pos.start is not None else 0
            add_event(val, category, dir_of(val), pos, pos, confidence)

    def _range_event(name, category, item, confidence=0.6):
        """Normalize the historical detector tuple formats.

        Detectors predate the common event contract and return several shapes:
        ``(start, end)``, ``(start, end, confidence)``, three/five candle
        index tuples, acceleration ``(start, phase, end, r2)`` and flat
        support ``(start, end, price)``.  This adapter preserves the actual
        candle span instead of mistaking an index or a price for confidence.
        """
        if isinstance(item, dict):
            start = item.get('start', item.get('idx', 0))
            end = item.get('end', start)
            conf = item.get('confidence', confidence)
            extra = {
                key: value
                for key, value in item.items()
                if key not in {'name', 'start', 'end', 'idx', 'confidence'}
            }
            add_event(item.get('name', name), category,
                      dir_of(item.get('name', name)), start, end, conf, **extra)
            return
        if not isinstance(item, (list, tuple)):
            add_event(name, category, dir_of(name), item, item, confidence)
            return
        if not item:
            return
        values = list(item)
        start = int(values[0])
        end = start
        conf = float(confidence)
        extra = {}
        if str(name) in {'平顶', '平底'} and len(values) >= 3:
            end = int(values[1])
            extra['structure_level'] = float(values[2])
        else:
            index_values = []
            for value in values:
                if isinstance(value, (int, np.integer)) and 0 <= int(value) < n:
                    index_values.append(int(value))
                else:
                    break
            if index_values:
                end = max(index_values)
            elif len(values) >= 2:
                end = int(values[1])
            # A trailing float in [0, 1] is confidence, not a bar index.
            if len(values) >= 3 and isinstance(values[-1], (float, np.floating)):
                trailing = float(values[-1])
                if 0.0 <= trailing <= 1.0:
                    conf = trailing
            if len(index_values) >= 3:
                extra['structure_points'] = index_values
        add_event(name, category, dir_of(name), start, end, conf, **extra)

    def add_dict_of_series(d, category, confidence=0.6):
        if not isinstance(d, dict):
            return
        for key, val in d.items():
            if isinstance(val, pd.Series):
                add_series_bool(val, key, category, confidence)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _range_event(key, category, item, confidence)

    def add_list_ranges(lst, name, category, confidence=0.6):
        if not isinstance(lst, (list, tuple)):
            return
        for item in lst:
            _range_event(name, category, item, confidence)

    # 逐个调用检测器。部署版不得再静默吞掉探测器异常：默认保持
    # 旧的 list 返回合同，同时把失败写入调用方提供的 diagnostics；
    # 测试/审计可使用 strict=True 立即失败。
    detectors = [
        # 大阳线/大阴线仅作内部参考，不作为最终标注
        # ('detect_big_candles', 'simple'),
        ('detect_engulfing', 'composite'),
        ('detect_doji', 'simple'),
        ('detect_piercing', 'composite'),
        ('detect_dark_cloud', 'composite'),
        ('detect_shooting_star_hammer', 'simple'),
        ('detect_t_shapes', 'simple'),
        ('detect_long_doji_propeller', 'simple'),
        ('detect_harami', 'composite'),
        ('detect_sun_and_rain', 'composite'),
        ('detect_meeting_lines', 'composite'),
        ('detect_high_open_escape', 'composite'),
        ('detect_low_open_surge', 'composite'),
        ('detect_tweezers_exhaustion_knead', 'composite'),
        ('detect_gaps', 'gap'),
        ('detect_window_support_resistance', 'simple'),
        ('detect_flat_bottom_top', 'composite'),
        ('detect_three_methods', 'composite'),
        ('detect_black_three_soldiers', 'composite'),
        ('detect_morning_evening_star', 'composite'),
        ('detect_abandoned_baby', 'composite'),
        ('detect_all_composite_patterns', 'composite'),
        ('detect_fake_yang', 'composite'),
        ('detect_two_crows', 'composite'),
        ('detect_side_by_side_gap', 'composite'),
        ('detect_three_gap_rise', 'composite'),
        ('detect_three_red', 'composite'),
        ('detect_white_soldiers_variants', 'composite'),
        ('detect_falling_three_stars', 'composite'),
        ('detect_three_gap_decline', 'composite'),
        ('detect_tower_bottom', 'trend'),
        ('detect_tower_top', 'trend'),
        ('detect_pagoda', 'trend'),
        ('detect_rounding', 'trend'),
        ('detect_bullish_probe', 'trend'),
        ('detect_bearish_probe', 'trend'),
        ('detect_acceleration', 'trend'),
        ('detect_ran_ran_rising', 'trend'),
        ('detect_mian_mian_decline', 'trend'),
        ('detect_xuhuan_rising', 'trend'),
        ('detect_xuhuan_decline', 'trend'),
        ('detect_rising_resistance', 'trend'),
        ('detect_falling_resistance', 'trend'),
        ('detect_steady_rising', 'trend'),
        ('detect_steady_decline', 'trend'),
    ]

    for func_name, category in detectors:
        func = globals().get(func_name)
        if func is None:
            if diagnostics is not None:
                diagnostics.append({
                    'stage': 'detector_lookup',
                    'detector': func_name,
                    'error_type': 'MissingDetector',
                    'message': '探测器未注册',
                })
            if strict:
                raise RuntimeError(f'pattern detector is not registered: {func_name}')
            continue
        try:
            result = func(df)
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.append({
                    'stage': 'detector_execution',
                    'detector': func_name,
                    'error_type': type(exc).__name__,
                    'message': f'{func_name} 执行失败',
                })
            if strict:
                raise RuntimeError(f'pattern detector failed: {func_name}') from exc
            continue

        if isinstance(result, pd.Series):
            if result.dtype == object:
                add_series_label(result, category)
            else:
                add_series_bool(result, func_name.replace('detect_', ''), category)
        elif isinstance(result, dict):
            add_dict_of_series(result, category)
        elif isinstance(result, pd.DataFrame):
            # 缺口类 DataFrame，逐行添加
            for _, row in result.iterrows():
                try:
                    idx_pos = row.get('gap_date', _)
                    pos = df.index.get_loc(idx_pos)
                    if isinstance(pos, slice):
                        pos = pos.start if pos.start is not None else 0
                except Exception:
                    continue
                gap_type = str(row.get('gap_type', '')).lower()
                name = (
                    '向上缺口' if gap_type == 'up'
                    else '向下缺口' if gap_type == 'down'
                    else str(row.get('gap_type', func_name))
                )
                gap_direction = 'up' if gap_type == 'up' else 'down' if gap_type == 'down' else ''
                add_event(
                    name,
                    category,
                    dir_of(name),
                    pos,
                    pos,
                    0.6,
                    gap_kind='rising_gap' if gap_type == 'up' else 'falling_gap',
                    gap_direction=gap_direction,
                    gap_type=gap_type,
                    gap_top=float(row.get('gap_top')) if pd.notna(row.get('gap_top')) else None,
                    gap_bottom=float(row.get('gap_bottom')) if pd.notna(row.get('gap_bottom')) else None,
                    gap_filled=bool(row.get('filled', False)),
                    gap_fill_bars=(
                        int(row.get('fill_bars'))
                        if pd.notna(row.get('fill_bars')) else None
                    ),
                )
        elif isinstance(result, (list, tuple)):
            add_list_ranges(result, func_name.replace('detect_', ''), category)

    # 趋势形态：加实体分类约束，同方向按 confidence 排序并限制数量，
    # 相邻合并，且在任意 120 根 K 线窗口内不超过 3 个
    trend_events = [e for e in events if e['category'] == 'trend']
    other_events = [e for e in events if e['category'] != 'trend']

    # 计算自适应 avg_body_20 用于实体分类约束
    close = df['close'] if 'close' in df.columns else df['Close']
    open_p = df['open'] if 'open' in df.columns else df['Open']
    body_all = (close - open_p).abs()
    avg_body_20_all = body_all.rolling(window=min(20, n), min_periods=min(5, n)).mean()

    # 过滤不满足实体分类约束的趋势形态
    trend_events = [
        e for e in trend_events
        if _trend_body_ok(df, e['start'], e['end'], avg_body_20_all)
    ]

    trend_events.sort(key=lambda x: x['confidence'], reverse=True)
    kept = []
    for e in trend_events:
        direction = e['direction']
        same_dir = [k for k in kept if k['direction'] == direction]
        if len(same_dir) >= max_trend_per_direction:
            continue
        # 与已保留的趋势区间重叠或相邻则合并，避免同一趋势重复标注
        merged = False
        for k in kept:
            if k['direction'] != direction:
                continue
            inter = min(e['end'], k['end']) - max(e['start'], k['start'])
            adj = max(e['start'], k['start']) - min(e['end'], k['end'])
            if inter >= 0 or adj <= 2:
                k['start'] = min(k['start'], e['start'])
                k['end'] = max(k['end'], e['end'])
                k['confidence'] = max(k['confidence'], e['confidence'])
                k['idx'] = k['end']
                merged = True
                break
        if not merged:
            kept.append(e)

    # 120 根 K 线窗口内趋势形态总数不超过 3 个
    window_limit = 3
    kept_sorted = sorted(kept, key=lambda x: x['start'])
    final_kept = []
    for e in kept_sorted:
        window_start = e['start']
        window_end = min(n - 1, window_start + 119)
        same_window = [k for k in final_kept
                       if not (k['end'] < window_start or k['start'] > window_end)]
        if len(same_window) < window_limit:
            final_kept.append(e)
    kept = final_kept

    # 大形态已标注区域过滤掉其内部的基本/组合形态，避免信号冲突
    trend_ranges = [(e['start'], e['end']) for e in kept]
    filtered_other = []
    for e in other_events:
        inside = any(s <= e['idx'] <= t for s, t in trend_ranges)
        if not inside:
            filtered_other.append(e)

    final_events = kept + filtered_other
    final_events.sort(key=lambda x: (x['start'], x['category']))
    return final_events
