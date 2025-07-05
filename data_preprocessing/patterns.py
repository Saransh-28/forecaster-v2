
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import pandas as pd

from numba import njit
from scipy.signal import find_peaks



@njit(cache=True)
def _hs_core(highs, lows,
             shoulder_tol: float,
             min_head_diff: float):
    n = len(highs)
    is_peak   = np.zeros(n, dtype=np.bool_)
    is_trough = np.zeros(n, dtype=np.bool_)
    pattern   = np.zeros(n, dtype=np.int8)

    for t in range(2, n):

        if highs[t-1] > highs[t-2] and highs[t-1] > highs[t]:
            is_peak[t-1] = True
        if lows[t-1]  < lows[t-2]  and lows[t-1]  < lows[t]:
            is_trough[t-1] = True

        if t >= 3:
            ls, hd, rs = t-3, t-2, t-1

            if is_peak[ls] and is_peak[hd] and is_peak[rs]:
                lhs, head, rhs = highs[ls], highs[hd], highs[rs]
                if (
                    head > lhs * (1 + min_head_diff)
                    and head > rhs * (1 + min_head_diff)
                    and abs(lhs - rhs) / max(lhs, rhs) < shoulder_tol
                ):
                    pattern[t] = 1

            if is_trough[ls] and is_trough[hd] and is_trough[rs]:
                lhs, head, rhs = lows[ls], lows[hd], lows[rs]
                if (
                    head < lhs * (1 - min_head_diff)
                    and head < rhs * (1 - min_head_diff)
                    and abs(lhs - rhs) / max(lhs, rhs) < shoulder_tol
                ):
                    pattern[t] = -1
    return pattern


def detect_head_shoulder(
    df: pd.DataFrame,
    shoulder_tolerance: float = 0.03,
    min_head_to_shoulder_diff: float = 0.01,
) -> pd.DataFrame:
    highs = df["High"].to_numpy(np.float64)
    lows  = df["Low"].to_numpy(np.float64)

    pattern = _hs_core(highs, lows,
                       shoulder_tolerance,
                       min_head_to_shoulder_diff)

    out = df.copy()
    out["head_shoulder_pattern"] = pd.Series(pattern)
    out['head_shoulder_pattern'] = out['head_shoulder_pattern'].replace(0, np.nan)
    out['head_shoulder_pattern'] = out['head_shoulder_pattern'].ffill(limit=5).fillna(0).astype(int)

    return out




def detect_multiple_tops_bottoms(df, window=5):
    high_roll_max = df["High"].rolling(window=window).max()
    low_roll_min = df["Low"].rolling(window=window).min()
    close_roll_max = df["Close"].rolling(window=window).max()
    close_roll_min = df["Close"].rolling(window=window).min()

    mask_top = (high_roll_max >= df["High"].shift(1)) & (
        close_roll_max < df["Close"].shift(1)
    )
    mask_bottom = (low_roll_min <= df["Low"].shift(1)) & (
        close_roll_min > df["Close"].shift(1)
    )
    df["multiple_top_bottom_pattern"] = 0
    df.loc[mask_top, "multiple_top_bottom_pattern"] = 1
    df.loc[mask_bottom, "multiple_top_bottom_pattern"] = -1
    df['multiple_top_bottom_pattern'] = df['multiple_top_bottom_pattern'].replace(0, np.nan)
    df['multiple_top_bottom_pattern'] = df['multiple_top_bottom_pattern'].ffill(limit=5).fillna(0).astype(int)

    # Clean up temporary variables
    del (
        high_roll_max,
        low_roll_min,
        close_roll_max,
        close_roll_min,
        mask_top,
        mask_bottom,
    )
    return df





def detect_triangle_pattern(df, window=7):
    high_roll_max = df["High"].rolling(window=window).max()
    low_roll_min = df["Low"].rolling(window=window).min()

    mask_ascending_triangle = (
        (high_roll_max >= df["High"].shift(1))
        & (low_roll_min <= df["Low"].shift(1))
        & (df["Close"] > df["Close"].shift(1))
    )
    mask_descending_triangle = (
        (high_roll_max <= df["High"].shift(1))
        & (low_roll_min >= df["Low"].shift(1))
        & (df["Close"] < df["Close"].shift(1))
    )

    df["triangle_pattern"] = 0
    df.loc[mask_ascending_triangle, "triangle_pattern"] = 1
    df.loc[mask_descending_triangle, "triangle_pattern"] = -1
    df['triangle_pattern'] = df['triangle_pattern'].replace(0, np.nan)
    df['triangle_pattern'] = df['triangle_pattern'].ffill(limit=5).fillna(0).astype(int)

    del high_roll_max, low_roll_min, mask_ascending_triangle, mask_descending_triangle
    return df



@njit(cache=True)
def _wedge_core(highs, lows, window):
    n = len(highs)
    pattern = np.zeros(n, dtype=np.int8)
    w = window

    for i in range(n):
        if i >= w - 1 and i >= 1:
            start = i - w + 1
            hi_max = highs[start : i + 1].max()
            lo_min = lows[start : i + 1].min()

            diff_hi = highs[i] - highs[start]
            trend_hi = 1 if diff_hi > 0 else (-1 if diff_hi < 0 else 0)

            diff_lo = lows[i] - lows[start]
            trend_lo = 1 if diff_lo > 0 else (-1 if diff_lo < 0 else 0)

            prev_hi = highs[i - 1]
            prev_lo = lows[i - 1]

            if (
                hi_max >= prev_hi
                and lo_min <= prev_lo
                and trend_hi == 1
                and trend_lo == 1
            ):
                pattern[i] = 1

            elif (
                hi_max <= prev_hi
                and lo_min >= prev_lo
                and trend_hi == -1
                and trend_lo == -1
            ):
                pattern[i] = -1
    return pattern


def detect_wedge(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    highs = df["High"].to_numpy(np.float64)
    lows  = df["Low"].to_numpy(np.float64)

    pattern = _wedge_core(highs, lows, int(window))

    out = df.copy()
    out["wedge_pattern"] = pd.Series(pattern)
    out['wedge_pattern'] = out['wedge_pattern'].replace(0, np.nan)
    out['wedge_pattern'] = out['wedge_pattern'].ffill(limit=5).fillna(0).astype(int)

    return out



def detect_channel(df, window=10, channel_range_threshold=0.03):
    high_roll_max = df["High"].rolling(window=window).max()
    low_roll_min = df["Low"].rolling(window=window).min()

    channel_trend_high = (
        df["High"]
        .rolling(window=window)
        .apply(
            lambda x: 1
            if (x.iloc[-1] - x.iloc[0]) > 0
            else -1
            if (x.iloc[-1] - x.iloc[0]) < 0
            else 0,
            raw=False,
        )
    )
    channel_trend_low = (
        df["Low"]
        .rolling(window=window)
        .apply(
            lambda x: 1
            if (x.iloc[-1] - x.iloc[0]) > 0
            else -1
            if (x.iloc[-1] - x.iloc[0]) < 0
            else 0,
            raw=False,
        )
    )
    average_channel = (high_roll_max + low_roll_min) / 2
    channel_width = high_roll_max - low_roll_min
    relative_width = channel_width / average_channel
    mask_channel_up = (
        (high_roll_max >= df["High"].shift(1))
        & (low_roll_min <= df["Low"].shift(1))
        & (relative_width <= channel_range_threshold)
        & (channel_trend_high == 1)
        & (channel_trend_low == 1)
    )
    mask_channel_down = (
        (high_roll_max <= df["High"].shift(1))
        & (low_roll_min >= df["Low"].shift(1))
        & (relative_width <= channel_range_threshold)
        & (channel_trend_high == -1)
        & (channel_trend_low == -1)
    )
    df["channel_pattern"] = 0
    df.loc[mask_channel_up, "channel_pattern"] = 1
    df.loc[mask_channel_down, "channel_pattern"] = -1
    df['channel_pattern'] = df['channel_pattern'].replace(0, np.nan)
    df['channel_pattern'] = df['channel_pattern'].ffill(limit=5).fillna(0).astype(int)
    del high_roll_max, low_roll_min, channel_trend_high, channel_trend_low
    del average_channel, channel_width, relative_width
    del mask_channel_up, mask_channel_down
    return df


@njit(cache=True)
def _double_core(highs, lows,
                 min_peak_distance,
                 threshold,
                 min_valley_diff,
                 stamp_mode):
    n = len(highs)
    pattern = np.zeros(n, dtype=np.int8)
    last_peak_idx   = -1
    last_trough_idx = -1
    for t in range(2, n):
        if highs[t-1] > highs[t-2] and highs[t-1] > highs[t]:
            pk_idx = t - 1
            if last_peak_idx != -1 and pk_idx - last_peak_idx >= min_peak_distance:
                pk1, pk2 = highs[last_peak_idx], highs[pk_idx]

                if abs(pk1 - pk2) / max(pk1, pk2) <= threshold:
                    valley = lows[last_peak_idx : pk_idx + 1].min()

                    if ((pk1 - valley) / pk1 >= min_valley_diff and
                        (pk2 - valley) / pk2 >= min_valley_diff):

                        mark = t if stamp_mode == 0 else pk_idx
                        pattern[mark] = 1
            last_peak_idx = pk_idx
        if lows[t-1] < lows[t-2] and lows[t-1] < lows[t]:
            tr_idx = t - 1
            if last_trough_idx != -1 and tr_idx - last_trough_idx >= min_peak_distance:
                tr1, tr2 = lows[last_trough_idx], lows[tr_idx]
                if abs(tr1 - tr2) / max(tr1, tr2) <= threshold:
                    peak = highs[last_trough_idx : tr_idx + 1].max()
                    if ((peak - tr1) / tr1 >= min_valley_diff and
                        (peak - tr2) / tr2 >= min_valley_diff):
                        mark = t if stamp_mode == 0 else tr_idx
                        pattern[mark] = -1
            last_trough_idx = tr_idx
    return pattern


def detect_double_top_bottom(
    df: pd.DataFrame,
    min_peak_distance: int = 4,
    threshold: float = 0.02,
    min_valley_diff: float = 0.01,
    stamp: str = "confirmation",
) -> pd.DataFrame:
    highs = df["High"].to_numpy(np.float64)
    lows  = df["Low"].to_numpy(np.float64)

    stamp_mode = 0 if stamp == "confirmation" else 1

    pattern = _double_core(highs, lows,
                           int(min_peak_distance),
                           float(threshold),
                           float(min_valley_diff),
                           stamp_mode)

    out = df.copy()
    out["double_pattern"] = pd.Series(pattern)
    out['double_pattern'] = out['double_pattern'].replace(0, np.nan)
    out['double_pattern'] = out['double_pattern'].ffill(limit=5).fillna(0).astype(int)

    return out


@njit(cache=True)
def _pivot_core(highs, lows, window):
    n = highs.size
    sig = np.zeros(n, dtype=np.int8)
    w = window
    for i in range(w, n - 1):
        cur_hi = highs[i]
        nxt_hi = highs[i + 1]
        cur_lo = lows[i]
        nxt_lo = lows[i + 1]
        if cur_hi >= highs[i - w : i].max() and cur_hi > nxt_hi:
            sig[i + 1] = 4
        if cur_lo <= lows[i - w : i].min() and cur_lo < nxt_lo:
            sig[i + 1] = 1
    return sig


def find_pivots(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    highs = df["High"].to_numpy(np.float64)
    lows  = df["Low"].to_numpy(np.float64)
    signal = _pivot_core(highs, lows, int(window))
    out = df.copy()
    out["pivot_signal"] = pd.Series(signal)
    out['pivot_signal'] = out['pivot_signal'].replace(0, np.nan)
    out['pivot_signal'] = out['pivot_signal'].ffill(limit=5).fillna(0).astype(int)
    return out


@njit(cache=True)
def compute_corridor_values(t, prices, window, sigma_multiplier):
    n = t.shape[0]
    baseline = np.empty(n)
    lower = np.empty(n)
    upper = np.empty(n)
    slopes = np.empty(n)
    intercepts = np.empty(n)
    sigmas = np.empty(n)
    for i in range(n):
        if window <= 0 or i < window - 1:
            start = 0
        else:
            start = i - window + 1
        m = i - start + 1
        if m < 2:
            baseline[i] = prices[i]
            lower[i] = prices[i]
            upper[i] = prices[i]
            slopes[i] = 0.0
            intercepts[i] = np.log(prices[i]) if prices[i] > 0 else 0.0
            sigmas[i] = 0.0
            continue
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        last_x = 0.0
        for j in range(start, i + 1):
            elapsed = t[j] - t[start] + 1.0
            x_val = np.log(elapsed)
            y_val = np.log(prices[j])
            sum_x += x_val
            sum_y += y_val
            sum_xx += x_val * x_val
            sum_xy += x_val * y_val
            if j == i:
                last_x = x_val
        m_float = float(m)
        mean_x = sum_x / m_float
        mean_y = sum_y / m_float
        var_x = sum_xx / m_float - mean_x * mean_x
        if var_x < 1e-12:
            slope = 0.0
        else:
            cov_xy = sum_xy / m_float - mean_x * mean_y
            slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x
        sum_res2 = 0.0
        for j in range(start, i + 1):
            elapsed = t[j] - t[start] + 1.0
            x_val = np.log(elapsed)
            y_val = np.log(prices[j])
            pred = intercept + slope * x_val
            res = y_val - pred
            sum_res2 += res * res
        sigma = np.sqrt(sum_res2 / m_float)
        baseline_log = intercept + slope * last_x
        base_val = np.exp(baseline_log)
        baseline[i] = base_val
        lower[i] = np.exp(baseline_log - sigma_multiplier * sigma)
        upper[i] = np.exp(baseline_log + sigma_multiplier * sigma)
        slopes[i] = slope
        intercepts[i] = intercept
        sigmas[i] = sigma
    return baseline, lower, upper, slopes, intercepts, sigmas



def power_law_corridor(
    df, price_column="Close", window=10, sigma_multiplier=1.0, timestamp_column=None
):
    df = df.copy()
    if timestamp_column is not None:
        times = pd.to_datetime(df[timestamp_column]).values.astype("datetime64[ns]")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be a DatetimeIndex or provide a timestamp_column."
            )
        times = df.index.values.astype("datetime64[ns]")
    t = (times.astype(np.int64) / 1e9) / 86400.0
    prices = df[price_column].values.astype(np.float64)
    window_val = 0 if window is None else int(window)
    _, _, _, slopes, _, sigmas = compute_corridor_values(
        t, prices, window_val, sigma_multiplier
    )
    df["slope"] = slopes
    df["sigma"] = sigmas

    return df


@njit(cache=True)
def _liq_core(high, low, op, cl,
              lookback: int,
              min_wick_ratio: float,
              confirm_next: bool):
    n = high.size
    sig = np.zeros(n, dtype=np.int8)
    end = n - 1 if not confirm_next else n - 2
    for i in range(lookback, end + 1):
        cr = high[i] - low[i]
        if cr <= 0.0:
            continue
        prev_hi = high[i - lookback : i].max()
        prev_lo = low[i - lookback : i].min()
        upper_wick = high[i] - max(op[i], cl[i])
        lower_wick = min(op[i], cl[i]) - low[i]
        if high[i] > prev_hi:
            if (upper_wick >= min_wick_ratio * cr and
                cl[i]     <  high[i] - 0.3 * cr):
                if confirm_next:
                    if cl[i + 1] < cl[i]:
                        sig[i] = -1
                else:
                    sig[i] = -1
        if low[i] < prev_lo:
            if (lower_wick >= min_wick_ratio * cr and
                cl[i]     >  low[i]  + 0.3 * cr):
                if confirm_next:
                    if cl[i + 1] > cl[i]:
                        sig[i] = 1
                else:
                    sig[i] = 1
    return sig


def detect_liquidity_grabs(df: pd.DataFrame,
                                 lookback: int = 10,
                                 min_wick_ratio: float = 0.6,
                                 confirm_next_bar: bool = False) -> pd.DataFrame:
    high = df["High"].to_numpy(np.float64)
    low  = df["Low"].to_numpy(np.float64)
    op   = df["Open"].to_numpy(np.float64)
    cl   = df["Close"].to_numpy(np.float64)
    sig = _liq_core(high, low, op, cl,
                    int(lookback),
                    float(min_wick_ratio),
                    bool(confirm_next_bar))
    out = df.copy()
    out["Liquidity_Grab"] = sig
    out['Liquidity_Grab'] = out['Liquidity_Grab'].replace(0, np.nan)
    out['Liquidity_Grab'] = out['Liquidity_Grab'].ffill(limit=5).fillna(0).astype(int)

    return out



@njit(cache=True)
def _sweep(high, low, close, idx, lookback, direction):
    if idx < lookback:
        return False
    start = idx - lookback
    if direction == 0:
        prev_high = high[start:idx].max()
        return high[idx] > prev_high and close[idx] < prev_high
    else:
        prev_low = low[start:idx].min()
        return low[idx] < prev_low and close[idx] > prev_low


@njit(cache=True)
def _ifvg_core(high, low, close,
               lookback_sweep: int,
               require_sweep: bool):

    n = high.size
    signal = np.zeros(n, np.int64)     
    fvg_tp = np.zeros(n, np.float64)   

    ub_out = np.empty(n, np.float64); ub_out[:] = np.nan
    lb_out = np.empty(n, np.float64); lb_out[:] = np.nan

    tp_arr   = np.zeros(n,  np.int8)
    ub_arr   = np.zeros(n,  np.float64)
    lb_arr   = np.zeros(n,  np.float64)
    used_arr = np.zeros(n,  np.bool_)
    active   = 0

    for i in range(n):

        if i >= 2:
            if high[i-2] < low[i]:                    
                tp_arr[active] = 1
                ub_arr[active] = low[i]
                lb_arr[active] = high[i-2]
                used_arr[active] = False
                active += 1
            if low[i-2] > high[i]:                  
                tp_arr[active] = -1
                ub_arr[active] = low[i-2]
                lb_arr[active] = high[i]
                used_arr[active] = False
                active += 1

        for j in range(active):
            if used_arr[j]:
                continue

            tp   = tp_arr[j]
            ub   = ub_arr[j]
            lb   = lb_arr[j]

            if tp == 1:                              
                if close[i] < lb or low[i] < lb:
                    ok = True
                    if require_sweep:
                        ok = _sweep(high, low, close, i, lookback_sweep, 0)
                    if ok:
                        signal[i]   = -1
                        fvg_tp[i]   =  1.0            
                        ub_out[i]   = ub
                        lb_out[i]   = lb
                        used_arr[j] = True

            else:                                     
                if close[i] > ub or high[i] > ub:
                    ok = True
                    if require_sweep:
                        ok = _sweep(high, low, close, i, lookback_sweep, 1)
                    if ok:
                        signal[i]   =  1
                        fvg_tp[i]   = -1.0
                        ub_out[i]   = ub
                        lb_out[i]   = lb
                        used_arr[j] = True

    return signal, fvg_tp, ub_out, lb_out


def generate_ifvg_signals(df: pd.DataFrame,
                                lookback_sweep: int = 20,
                                require_sweep: bool = False) -> pd.DataFrame:
    df = df.sort_index().reset_index(drop=True)
    high  = df["High"].to_numpy(np.float64)
    low   = df["Low"].to_numpy(np.float64)
    close = df["Close"].to_numpy(np.float64)
    sig, tp, ub, lb = _ifvg_core(high, low, close,
                                 int(lookback_sweep),
                                 bool(require_sweep))
    out = df.copy()
    upper_band = pd.Series(ub)
    lower_band = pd.Series(lb)
    upper_band = upper_band.fillna(out["Close"])
    lower_band = lower_band.fillna(out["Close"])
    out["Ivfg_Signal"]         = sig.astype(np.int64)
    out["FVG_UpperBound_high"]  = (upper_band > out["High"]).astype(np.int64)
    out["FVG_LowerBound_low"]   = (lower_band < out["Low"]).astype(np.int64)
    out["FVG_UpperBound_close"] = (upper_band > out["Close"]).astype(np.int64)
    out["FVG_LowerBound_close"] = (lower_band < out["Close"]).astype(np.int64)
    out['Ivfg_Signal'] = out['Ivfg_Signal'].replace(0, np.nan)
    out['Ivfg_Signal'] = out['Ivfg_Signal'].ffill(limit=5).fillna(0).astype(int)
    return out




def get_pattern_recognition_features(df, context):
    df = detect_head_shoulder(df, **context["detect_head_shoulder"])
    df = detect_multiple_tops_bottoms(df, **context["detect_multiple_tops_bottoms"])
    df = detect_triangle_pattern(df, **context["detect_triangle_pattern"])
    df = detect_wedge(df, **context["detect_wedge"])
    df = detect_channel(df, **context["detect_channel"])
    df = detect_double_top_bottom(df, **context["detect_double_top_bottom"])
    df = find_pivots(df, **context["find_pivots"])
    df = power_law_corridor(df, **context["power_law_corridor"])
    df = detect_liquidity_grabs(df, **context["detect_liquidity_grabs"])
    df = generate_ifvg_signals(df, **context["generate_ifvg_signals"])
    return df




@njit(cache=True)
def _hns_top_jit(high: np.ndarray,
                 low: np.ndarray,
                 bars_left: int,
                 tolerance: float):
    n = high.shape[0]
    pivots    = np.empty(n, np.int64)
    pc        = 0
    flags     = np.zeros(n, np.bool_)
    ls_idxs   = np.full(n, -1, np.int64)
    head_idxs = np.full(n, -1, np.int64)
    for i in range(bars_left, n):
        hi = high[i]
        is_peak = True
        for j in range(i - bars_left, i):
            if high[j] > hi:
                is_peak = False
                break
        if not is_peak:
            continue
        pivots[pc] = i
        pc += 1
        if pc >= 3:
            ls   = pivots[pc - 3]
            head = pivots[pc - 2]
            rs   = pivots[pc - 1]
            h_ls = high[ls]; h_hd = high[head]; h_rs = hi
            if not (h_hd > h_ls and h_hd > h_rs):
                continue
            if abs(h_rs - h_ls) / h_ls > tolerance:
                continue
            v1 = low[ls]
            for m in range(ls + 1, head + 1):
                if low[m] < v1:
                    v1 = low[m]
            v2 = low[head]
            for m in range(head + 1, rs + 1):
                if low[m] < v2:
                    v2 = low[m]
            if v2 > v1:
                flags[rs]     = True
                ls_idxs[rs]   = ls
                head_idxs[rs] = head
    return flags, ls_idxs, head_idxs


def head_and_shoulders_top_signal(df: pd.DataFrame,
                                  bars_left: int = 3,
                                  tolerance: float = 0.02,
                                  warmup: int = None) -> pd.Series:
    if 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("DataFrame must contain 'High' and 'Low' columns")
    if warmup is None:
        warmup = bars_left * 3
    high = df['High'].to_numpy()
    low  = df['Low'].to_numpy()
    flags, ls_idxs, head_idxs = _hns_top_jit(high, low, bars_left, tolerance)
    sig = np.zeros(len(df), dtype=int)
    lsf = np.zeros(len(df), dtype=int)
    hidx = np.zeros(len(df), dtype=int)
    for rs, is_pat in enumerate(flags):
        if is_pat:
            sig[rs] = -1
            lsf[rs] = ls_idxs[rs] - flags[rs] 
            hidx[rs] = head_idxs[rs] - flags[rs] 
    sig[:warmup] = 0
    lsf[:warmup] = 0
    hidx[:warmup] = 0
    df["hs_top_signal"] = pd.Series(sig, index=df.index)
    df['hs_dis'] = (lsf)
    df['rs_dis'] = (hidx)
    df['hs_top_signal'] = df['hs_top_signal'].replace(0, np.nan)
    df['hs_top_signal'] = df['hs_top_signal'].ffill(limit=5).fillna(0).astype(int)
    return df



@njit(cache=True)
def _pennant_jit(high: np.ndarray,
                 low: np.ndarray,
                 bars_left: int,
                 slope_tol: float,
                 flat_tol: float):
    n = high.shape[0]
    pivots = np.empty(n, np.int64)
    ptypes = np.empty(n, np.int8)
    pc = 0
    flags = np.zeros((n, 5), np.bool_)
    for i in range(bars_left, n):
        hi = high[i]
        lo = low[i]
        is_peak = True
        is_trough = True
        for j in range(i - bars_left, i):
            if high[j] > hi:
                is_peak = False
            if low[j] < lo:
                is_trough = False
        if is_peak:
            pivots[pc] = i
            ptypes[pc] = 1
            pc += 1
        elif is_trough:
            pivots[pc] = i
            ptypes[pc] = -1
            pc += 1
        else:
            continue
        if pc >= 5 and ptypes[pc - 1] == 1:
            p1 = pivots[pc - 5]
            t1 = pivots[pc - 4]
            p2 = pivots[pc - 3]
            t2 = pivots[pc - 2]
            p3 = pivots[pc - 1] 
            dp = p3 - p1
            dt = t2 - t1
            if dp == 0 or dt == 0:
                continue
            slope_peaks   = (high[p3] - high[p1]) / dp
            slope_troughs = ( low[t2] -  low[t1]) / dt
            if slope_peaks < 0 and slope_troughs > 0:
                if abs(abs(slope_peaks) - slope_troughs) <= slope_tol:
                    flags[p3, 0] = True
            if abs(slope_peaks) <= flat_tol and slope_troughs > 0:
                flags[p3, 1] = True
            if abs(slope_troughs) <= flat_tol and slope_peaks < 0:
                flags[p3, 2] = True
            if slope_peaks > 0 and slope_troughs > 0:
                if slope_troughs > slope_peaks and abs(slope_troughs - slope_peaks) <= slope_tol:
                    flags[p3, 3] = True
            if slope_peaks < 0 and slope_troughs < 0:
                if abs(slope_peaks) > abs(slope_troughs) and abs(abs(slope_peaks) - abs(slope_troughs)) <= slope_tol:
                    flags[p3, 4] = True

    return flags


def detect_pennants(df: pd.DataFrame,
                    bars_left: int = 3,
                    slope_tol: float = 0.02,
                    flat_tol: float = 0.02,
                    warmup: int = None) -> pd.DataFrame:
    if {'High','Low'} - set(df.columns):
        raise ValueError("DataFrame must contain 'High' and 'Low' columns")
    if warmup is None:
        warmup = bars_left * 5
    high = df['High'].to_numpy()
    low  = df['Low'].to_numpy()
    flags = _pennant_jit(high, low, bars_left, slope_tol, flat_tol)
    patterns = [
        'sym_triangle',
        'asc_triangle',
        'desc_triangle',
        'rising_wedge',
        'falling_wedge'
    ]
    out = df.copy()
    for idx, name in enumerate(patterns):
        col = flags[:, idx].astype(int)
        col[:warmup] = 0
        out[name] = col
    return out



def get_pattern_recognition_features_new(df, context):
    df = head_and_shoulders_top_signal(df, **context["head_and_shoulders_top_signal"])
    df = detect_pennants(df, **context["detect_pennants"])
    return df








