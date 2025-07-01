from numba import njit
import numpy as np
import pandas as pd
import talib


@njit
def compute_atr_numba(high, low, close, length):
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl > hc and hl > lc else (hc if hc > lc else lc)
    if n >= length:
        s = 0.0
        for k in range(length):
            s += tr[k]
        atr[length - 1] = s / length
        for k in range(length, n):
            atr[k] = (atr[k - 1] * (length - 1) + tr[k]) / length
    return atr


@njit
def bwab_signals(
    high, low, close, ma_len=39, range_len=10, atr_len=18, atr_mult=1.2, entry_buildup=6
):
    n = len(close)
    long_sig = np.zeros(n, np.bool_)
    short_sig = np.zeros(n, np.bool_)
    ma = np.full(n, np.nan)
    for i in range(ma_len - 1, n):
        s = 0.0
        for j in range(i - ma_len + 1, i + 1):
            s += close[j]
        ma[i] = s / ma_len
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)
    for i in range(range_len, n):
        hmax = high[i - range_len]
        lmin = low[i - range_len]
        for j in range(i - range_len + 1, i):
            if high[j] > hmax:
                hmax = high[j]
            if low[j] < lmin:
                lmin = low[j]
        range_high[i] = hmax
        range_low[i] = lmin
    atr = compute_atr_numba(high, low, close, atr_len)
    for i in range(range_len, n):
        if np.isnan(ma[i]) or np.isnan(atr[i]):
            continue
        lo_hi = high[i - entry_buildup]
        hi_hi = high[i - entry_buildup]
        lo_lo = low[i - entry_buildup]
        hi_lo = low[i - entry_buildup]
        for j in range(i - entry_buildup + 1, i):
            if high[j] < lo_hi:
                lo_hi = high[j]
            if high[j] > hi_hi:
                hi_hi = high[j]
            if low[j] < lo_lo:
                lo_lo = low[j]
            if low[j] > hi_lo:
                hi_lo = low[j]
        cond_hi = lo_hi <= ma[i] <= hi_hi
        cond_lo = lo_lo <= ma[i] <= hi_lo
        bh = range_high[i] + atr_mult * atr[i]
        bl = range_low[i] - atr_mult * atr[i]
        if high[i] > bh and cond_hi:
            long_sig[i] = True
        elif low[i] < bl and cond_lo:
            short_sig[i] = True
    return long_sig, short_sig


def mark_bwab_signals(
    df, ma_len=39, range_len=10, atr_len=18, atr_mult=1.2, entry_buildup=6
):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    long_sig, short_sig = bwab_signals(
        high=high,
        low=low,
        close=close,
        ma_len=ma_len,
        range_len=range_len,
        atr_len=atr_len,
        atr_mult=atr_mult,
        entry_buildup=entry_buildup,
    )
    df_result = df.copy()
    long = long_sig.astype(int)
    short = short_sig.astype(int)
    df_result["bwab_signal"] = long - short
    df_result["bwab_signal"] = df_result["bwab_signal"].replace(0, np.nan)
    df_result["bwab_signal"] = (
        df_result["bwab_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return df_result


@njit
def triple_ma_signals(close, s1=5, s2=15, s3=30):
    n = len(close)
    ma1 = np.full(n, np.nan)
    ma2 = np.full(n, np.nan)
    ma3 = np.full(n, np.nan)
    sum1 = sum2 = sum3 = 0.0
    for i in range(n):
        sum1 += close[i]
        if i >= s1:
            sum1 -= close[i - s1]
        if i >= s1 - 1:
            ma1[i] = sum1 / s1
        sum2 += close[i]
        if i >= s2:
            sum2 -= close[i - s2]
        if i >= s2 - 1:
            ma2[i] = sum2 / s2
        sum3 += close[i]
        if i >= s3:
            sum3 -= close[i - s3]
        if i >= s3 - 1:
            ma3[i] = sum3 / s3
    long_sig = np.zeros(n, np.bool_)
    short_sig = np.zeros(n, np.bool_)
    stage = np.zeros(n, np.int8)
    for i in range(1, n):
        stage[i] = stage[i - 1]
        a1, a2, a3 = ma1[i], ma2[i], ma3[i]
        pa1, pa2, pa3 = ma1[i - 1], ma2[i - 1], ma3[i - 1]
        if np.isnan(a1) or np.isnan(a2) or np.isnan(a3):
            continue
        c, pc = close[i], close[i - 1]
        up = a1 > pa1 and a2 > pa2 and a3 > pa3
        down = a1 < pa1 and a2 < pa2 and a3 < pa3
        # long entry stages
        if stage[i] == 0 and up and pc <= a1 and c > a1:
            long_sig[i], stage[i] = True, 1
            continue
        if stage[i] == 1 and pc <= a2 and c > a2:
            long_sig[i], stage[i] = True, 2
            continue
        if stage[i] == 2 and pc <= a3 and c > a3:
            long_sig[i], stage[i] = True, 3
            continue
        # long exit: cross below latest MA
        if stage[i] > 0:
            last_ma = a3 if stage[i] == 3 else (a2 if stage[i] == 2 else a1)
            if c < last_ma:
                short_sig[i], stage[i] = True, 0
                continue
        # short entry stages
        if stage[i] == 0 and down and pc >= a1 and c < a1:
            short_sig[i], stage[i] = True, -1
            continue
        if stage[i] == -1 and pc >= a2 and c < a2:
            short_sig[i], stage[i] = True, -2
            continue
        if stage[i] == -2 and pc >= a3 and c < a3:
            short_sig[i], stage[i] = True, -3
            continue
        # short exit: cross above latest MA
        if stage[i] < 0:
            last_ma = a3 if stage[i] == -3 else (a2 if stage[i] == -2 else a1)
            if c > last_ma:
                long_sig[i], stage[i] = True, 0
                continue
    return short_sig, long_sig


def apply_triple_ma(df, s1=5, s2=15, s3=30):
    cl = df["Close"].values
    longs, shorts = triple_ma_signals(cl, s1=s1, s2=s2, s3=s3)
    out = df.copy()
    long = longs.astype(int)
    short = shorts.astype(int)
    out["triple_ma_signal"] = long - short
    out["triple_ma_signal"] = out["triple_ma_signal"].replace(0, np.nan)
    out["triple_ma_signal"] = (
        out["triple_ma_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return out


def williams_vix_fix(df: pd.DataFrame, length: int = 22) -> pd.Series:
    highest_high = df["High"].rolling(window=length, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=length, min_periods=1).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    wvf = 100 * (highest_high - df["Close"]) / denom
    return wvf


def compute_wvf_bands(wvf: pd.Series, bbl: int = 20, mult: float = 2.0):
    wvf_ma = wvf.rolling(window=bbl, min_periods=1).mean()
    wvf_std = wvf.rolling(window=bbl, min_periods=1).std()
    wvf_upper = wvf_ma + mult * wvf_std
    return wvf_ma, wvf_std, wvf_upper


def generate_market_bottom_signals(
    original_df: pd.DataFrame, length: int = 2, bbl: int = 15, mult: float = 2.0
) -> pd.DataFrame:
    df = original_df.copy()
    wvf = williams_vix_fix(df, length)
    _, _, upper = compute_wvf_bands(wvf, bbl, mult)
    cross_signal = (wvf.shift(1) > upper.shift(1)) & (wvf <= upper)
    rolling_low = df["Low"].rolling(window=length, min_periods=1).min()
    bottom_condition = df["Low"] == rolling_low
    original_df["bottom_signal"] = (cross_signal & bottom_condition).astype(int)
    original_df["bottom_signal"] = original_df["bottom_signal"].replace(0, np.nan)
    original_df["bottom_signal"] = (
        original_df["bottom_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return original_df


@njit
def _compute_williams_r_signals(high, low, close, period):
    n = high.shape[0]
    sig = np.zeros(n, dtype=np.int8)
    wr = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        hh = high[i - period + 1]
        ll = low[i - period + 1]
        for j in range(i - period + 1, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]

        rng = hh - ll
        if rng == 0.0:
            wr[i] = 0.0
        else:
            wr[i] = ((close[i] - hh) / rng) * 100.0

    for i in range(1, n):
        prev = wr[i - 1]
        curr = wr[i]
        if np.isnan(prev) or np.isnan(curr):
            continue
        if curr > -80.0 and prev <= -80.0:
            sig[i] = 1
        elif curr < -20.0 and prev >= -20.0:
            sig[i] = -1

    return sig, wr


def williams_r_signals(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    data = df.copy()
    data.sort_index(inplace=True)
    high = data["High"].to_numpy(dtype=np.float64)
    low = data["Low"].to_numpy(dtype=np.float64)
    close = data["Close"].to_numpy(dtype=np.float64)
    sig, val = _compute_williams_r_signals(high, low, close, period)
    data["williams_r_signal"] = sig
    data["williams_r"] = val
    data["williams_r_signal"] = data["williams_r_signal"].replace(0, np.nan)
    data["williams_r_signal"] = (
        data["williams_r_signal"].ffill(limit=5).fillna(0).astype(int)
    )

    return data


@njit
def rolling_max_index(arr, window):
    n = len(arr)
    out = np.full(n, -1, dtype=np.int32)
    dq = np.empty(n, dtype=np.int32)
    head = 0
    tail = -1
    for i in range(n):
        while tail >= head and dq[head] < i - window + 1:
            head += 1
        while tail >= head and arr[i] >= arr[dq[tail]]:
            tail -= 1
        tail += 1
        dq[tail] = i
        if i >= window - 1:
            out[i] = dq[head]
    return out


@njit
def rolling_min_index(arr, window):
    n = len(arr)
    out = np.full(n, -1, dtype=np.int32)
    dq = np.empty(n, dtype=np.int32)
    head = 0
    tail = -1
    for i in range(n):
        while tail >= head and dq[head] < i - window + 1:
            head += 1
        while tail >= head and arr[i] <= arr[dq[tail]]:
            tail -= 1
        tail += 1
        dq[tail] = i
        if i >= window - 1:
            out[i] = dq[head]
    return out


@njit
def rolling_sum(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= window:
            s -= arr[i - window]
        if i >= window - 1:
            out[i] = s
    return out


@njit
def _enhanced_lh_tick_jit(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    lookback: int,
    ma_period: int,
    atr_period: int,
    atr_mult: float,
):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = tr1 if (tr1 >= tr2 and tr1 >= tr3) else (tr2 if tr2 >= tr3 else tr3)
    atr_sum = rolling_sum(tr, atr_period)
    atr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr[i] = atr_sum[i] / atr_period
    close_sum = rolling_sum(closes, ma_period)
    sma = np.full(n, np.nan)
    for i in range(ma_period - 1, n):
        sma[i] = close_sum[i] / ma_period
    avg_vol = None
    if volumes.size > 0:
        vol_sum = rolling_sum(volumes, lookback)
        avg_vol = np.full(n, np.nan)
        for i in range(lookback - 1, n):
            avg_vol[i] = vol_sum[i] / lookback
    min_idx = rolling_min_index(lows, lookback)
    max_idx = rolling_max_index(highs, lookback)
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        mi = min_idx[i]
        ma = max_idx[i]
        if mi != -1:
            support[i] = lows[mi]
        if ma != -1:
            resistance[i] = highs[ma]
    long_sig = np.zeros(n, dtype=np.int8)
    short_sig = np.zeros(n, dtype=np.int8)
    for i in range(max(lookback, ma_period, atr_period), n):
        if (
            np.isnan(sma[i])
            or np.isnan(atr[i])
            or np.isnan(support[i])
            or np.isnan(resistance[i])
        ):
            continue
        is_up = closes[i] > sma[i]
        is_down = closes[i] < sma[i]
        vol_ok = True
        if avg_vol is not None and not np.isnan(avg_vol[i]):
            vol_ok = volumes[i] > avg_vol[i]
        if (
            lows[i] <= support[i]
            and closes[i] > support[i] + atr_mult * atr[i]
            and is_up
            and vol_ok
        ):
            long_sig[i] = 1
        if (
            highs[i] >= resistance[i]
            and closes[i] < resistance[i] - atr_mult * atr[i]
            and is_down
            and vol_ok
        ):
            short_sig[i] = 1
    return long_sig, short_sig, support, resistance, sma, atr


def lowest_highest_tick_strategy(
    df: pd.DataFrame,
    lookback: int = 30,
    ma_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 0.5,
) -> pd.DataFrame:
    req = {"High", "Low", "Close"}
    if not req.issubset(df.columns):
        raise ValueError(f"DataFrame must have columns: {req}")
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    vols = df["Volume"].to_numpy() if "Volume" in df.columns else np.empty(0)
    out = df.copy()
    ls, ss, sup, res, sma, atr = _enhanced_lh_tick_jit(
        highs, lows, closes, vols, lookback, ma_period, atr_period, atr_mult
    )
    long = ss
    short = ls
    out["lht_signal"] = long - short
    out["lht_signal"] = out["lht_signal"].replace(0, np.nan)
    out["lht_signal"] = out["lht_signal"].ffill(limit=5).fillna(0).astype(int)

    return out


@njit
def two_pole_filter_numba(src, length):
    alpha = 2.0 / (length + 1)
    n = len(src)
    out = np.full(n, np.nan)
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        x = src[i]
        if np.isnan(x):
            continue
        if i == 0 or np.isnan(out[i - 1]):
            s1 = x
            s2 = x
        else:
            s1 = (1 - alpha) * s1 + alpha * x
            s2 = (1 - alpha) * s2 + alpha * s1
        out[i] = s2
    return out


@njit
def compute_sma_n1_numba(close, sma_len):
    n = len(close)
    sma_n1 = np.full(n, np.nan)
    for i in range(sma_len - 1, n):
        s = 0.0
        for j in range(i - sma_len + 1, i + 1):
            s += close[j]
        mean = s / sma_len
        diff_sum = 0.0
        for j in range(i - sma_len + 1, i + 1):
            diff_sum += close[j] - mean
        dev = diff_sum / sma_len
        diff_sq_sum = 0.0
        for j in range(i - sma_len + 1, i + 1):
            val = close[j] - mean
            diff_sq_sum += val * val
        std = np.sqrt(diff_sq_sum / sma_len)
        if std > 1e-6:
            sma_n1[i] = (close[i] - mean - dev) / std
        else:
            sma_n1[i] = 0.0
    return sma_n1


def compute_two_pole_oscillator(df, filt_len=20, sma_len=25, threshold=0.1):
    close = df["Close"].values
    sma_n1 = compute_sma_n1_numba(close, sma_len)
    sma_n1 = np.clip(sma_n1, -3, 3)
    two_p = two_pole_filter_numba(sma_n1, filt_len)

    def lag_array(arr, lag):
        out = np.empty_like(arr)
        out[:lag] = np.nan
        out[lag:] = arr[:-lag]
        return out

    two_pp = lag_array(two_p, 4)
    two_p_lag1 = lag_array(two_p, 1)
    two_pp_lag1 = lag_array(two_pp, 1)
    sell = (
        (two_p_lag1 < two_pp_lag1) & (two_p > two_pp) & (two_p < -threshold)
    ).astype(int)
    buy = ((two_p_lag1 > two_pp_lag1) & (two_p < two_pp) & (two_p > threshold)).astype(
        int
    )
    result = df.copy()
    delta = np.append([np.nan], np.diff(two_p))
    result["two_pole_momentum"] = delta
    result["two_pole_signal"] = buy - sell
    result["two_pole_signal"] = result["two_pole_signal"].replace(0, np.nan)
    result["two_pole_signal"] = (
        result["two_pole_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return result


@njit
def ema_numba(src, length):
    alpha = 2 / (length + 1)
    n = len(src)
    out = np.empty(n)
    out[0] = src[0]
    for i in range(1, n):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


@njit
def adaptive_t3(src, rsi_len=14, min_len=5, max_len=50, v=0.7, volat=40):
    n = len(src)
    rsi = np.empty(n)
    gain = np.zeros(n)
    loss = np.zeros(n)

    for i in range(1, n):
        delta = src[i] - src[i - 1]
        gain[i] = max(delta, 0)
        loss[i] = max(-delta, 0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    avg_gain[rsi_len] = np.mean(gain[1 : rsi_len + 1])
    avg_loss[rsi_len] = np.mean(loss[1 : rsi_len + 1])
    rsi[:rsi_len] = np.nan

    for i in range(rsi_len, n):
        avg_gain[i] = (avg_gain[i - 1] * (rsi_len - 1) + gain[i]) / rsi_len
        avg_loss[i] = (avg_loss[i - 1] * (rsi_len - 1) + loss[i]) / rsi_len
        rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 100
        rsi[i] = 100 - (100 / (1 + rs))

    rsi_scale = 1 - rsi / 100
    length = np.round(min_len + (max_len - min_len) * rsi_scale).astype(np.int32)
    t3 = np.full(n, np.nan)

    for i in range(max_len * 6, n):
        l = length[i]
        e1 = ema_numba(src[i - l * 6 + 1 : i + 1], l)
        e2 = ema_numba(e1, l)
        e3 = ema_numba(e2, l)
        e4 = ema_numba(e3, l)
        e5 = ema_numba(e4, l)
        e6 = ema_numba(e5, l)

        c1 = -(v**3)
        c2 = 3 * v**2 + 3 * v**3
        c3 = -6 * v**2 - 3 * v - 3 * v**3
        c4 = 1 + 3 * v + 3 * v**2 + v**3
        t3[i] = c1 * e6[-1] + c2 * e5[-1] + c3 * e4[-1] + c4 * e3[-1]

    return t3


def apply_adaptive_t3_signals(df, rsi_len=25, min_len=5, max_len=40, v=0.5, volat=40):
    src = df["Close"].values
    t3_vals = adaptive_t3(
        src, rsi_len=rsi_len, min_len=min_len, max_len=max_len, v=v, volat=volat
    )

    result = df.copy()
    t3 = pd.Series(t3_vals)
    result["t3_signal"] = np.where(
        t3 > t3.shift(2), 1, np.where(t3 < t3.shift(2), -1, 0)
    )
    result["t3_signal"] = result["t3_signal"].replace(0, np.nan)
    result["t3_signal"] = result["t3_signal"].ffill(limit=5).fillna(0).astype(int)
    return result


@njit
def next_pivot_signal_inner(close, hist_len=10, fore_len=20, lookback=100, method=0):
    n = len(close)
    long_sig = np.zeros(n, np.bool_)
    short_sig = np.zeros(n, np.bool_)
    for i in range(hist_len + fore_len + lookback, n):
        rec = close[i - hist_len : i]
        best_sim = -1.0
        best_pos = -1
        for j in range(i - fore_len - lookback, i - fore_len):
            if j - hist_len < 0:
                continue
            cand = close[j - hist_len : j]
            if method == 0:
                cr = np.corrcoef(rec, cand)[0, 1]
                sim = cr if not np.isnan(cr) else -1
            elif method == 1:
                num = np.dot(rec, cand)
                sim = num / (np.linalg.norm(rec) * np.linalg.norm(cand) + 1e-8)
            else:
                diff = rec - cand
                sim = 1.0 / (1.0 + np.dot(diff, diff))
            if sim > best_sim:
                best_sim = sim
                best_pos = j
        if best_pos == -1 or best_pos + fore_len >= n:
            continue
        pivot = close[best_pos + fore_len - 1]
        curr = close[i - 1]
        long_sig[i] = pivot > curr
        short_sig[i] = pivot < curr
    return long_sig, short_sig


def next_pivot_signals(df, hist_len=10, fore_len=20, lookback=100, method=0):
    close = df["Close"].values
    long_sig, short_sig = next_pivot_signal_inner(
        close, hist_len=hist_len, fore_len=fore_len, lookback=lookback, method=method
    )
    df_result = df.copy()
    long = long_sig.astype(int)
    short = short_sig.astype(int)
    df_result["next_pivot_signal"] = long - short
    df_result["next_pivot_signal"] = df_result["next_pivot_signal"].replace(0, np.nan)
    df_result["next_pivot_signal"] = (
        df_result["next_pivot_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return df_result


@njit
def _compute_market_sentiment_signals(
    close,
    high,
    low,
    rsi_len=14,
    stoch_k_len=14,
    stoch_k_smooth=3,
    cci_len=20,
    bbp_len=13,
    ma_len=20,
    st_len=10,
    st_mult=3,
    reg_len=25,
    ms_len=5,
):
    n = len(close)
    long_sig = np.zeros(n, dtype=np.bool_)
    short_sig = np.zeros(n, dtype=np.bool_)

    def calc_rsi(prices, length):
        rsis = np.full(prices.shape, np.nan)
        for i in range(length, len(prices)):
            gain = 0.0
            loss = 0.0
            for j in range(i - length + 1, i + 1):
                diff = prices[j] - prices[j - 1]
                if diff > 0:
                    gain += diff
                else:
                    loss -= diff
            if loss == 0:
                rsis[i] = 100
            else:
                rs = gain / loss
                rsis[i] = 100 - (100 / (1 + rs))
        return rsis

    def normalize_series(series):
        min_val = np.nanmin(series)
        max_val = np.nanmax(series)
        if max_val - min_val == 0:
            return np.zeros_like(series)
        return 100 * (series - min_val) / (max_val - min_val)

    rsi_val = normalize_series(calc_rsi(close, rsi_len))

    stoch_val = np.zeros(n)
    for i in range(stoch_k_len + stoch_k_smooth - 1, n):
        ll = np.min(low[i - stoch_k_len + 1 : i + 1])
        hh = np.max(high[i - stoch_k_len + 1 : i + 1])
        if hh - ll != 0:
            raw_k = 100 * (close[i] - ll) / (hh - ll)
        else:
            raw_k = 0
        smoothed_k = 0.0
        for _ in range(stoch_k_smooth):
            smoothed_k += raw_k
        stoch_val[i] = smoothed_k / stoch_k_smooth

    cci_val = np.zeros(n)
    for i in range(cci_len, n):
        tp = (high[i] + low[i] + close[i]) / 3
        sum_tp = 0.0
        for j in range(i - cci_len + 1, i + 1):
            sum_tp += (high[j] + low[j] + close[j]) / 3
        ma = sum_tp / cci_len
        sum_dev = 0.0
        for j in range(i - cci_len + 1, i + 1):
            sum_dev += abs((high[j] + low[j] + close[j]) / 3 - ma)
        md = sum_dev / cci_len
        cci_val[i] = (tp - ma) / (0.015 * md) if md != 0 else 0

    bbp_val = np.zeros(n)
    for i in range(bbp_len, n):
        ema = 0.0
        for j in range(i - bbp_len + 1, i + 1):
            ema += close[j]
        ema /= bbp_len
        bbp_val[i] = (high[i] + low[i]) - 2 * ema

    ma_val = np.zeros(n)
    for i in range(ma_len, n):
        ma_sum = 0.0
        for j in range(i - ma_len + 1, i + 1):
            ma_sum += close[j]
        ma_val[i] = close[i] - (ma_sum / ma_len)

    st_val = np.zeros(n)
    for i in range(st_len, n):
        atr = 0.0
        for j in range(i - st_len + 1, i + 1):
            atr += high[j] - low[j]
        atr /= st_len
        up = (high[i] + low[i]) / 2 + st_mult * atr
        down = (high[i] + low[i]) / 2 - st_mult * atr
        st_val[i] = 1 if close[i] > up else -1 if close[i] < down else 0

    reg_val = np.zeros(n)
    for i in range(reg_len, n):
        x = np.arange(reg_len)
        y = close[i - reg_len + 1 : i + 1]
        sx = np.sum(x)
        sy = np.sum(y)
        sxy = np.sum(x * y)
        sx2 = np.sum(x * x)
        denom = reg_len * sx2 - sx * sx
        if denom != 0:
            m = (reg_len * sxy - sx * sy) / denom
            reg_val[i] = m
        else:
            reg_val[i] = 0

    ms_val = np.zeros(n)
    for i in range(ms_len, n):
        ph = np.max(high[i - ms_len + 1 : i + 1])
        pl = np.min(low[i - ms_len + 1 : i + 1])
        ms_val[i] = 1 if close[i] > ph else -1 if close[i] < pl else 0

    for i in range(n):
        count = 0
        summation = 0.0
        for val in [
            rsi_val[i],
            stoch_val[i],
            cci_val[i],
            bbp_val[i],
            ma_val[i],
            st_val[i],
            reg_val[i],
            ms_val[i],
        ]:
            if not np.isnan(val):
                summation += val
                count += 1
        if count > 0:
            sentiment = summation / count
            if sentiment > 60:
                long_sig[i] = True
            elif sentiment < 40:
                short_sig[i] = True

    return long_sig, short_sig


def market_sentiment_signals(
    df,
    rsi_len=14,
    stoch_k_len=14,
    stoch_k_smooth=3,
    cci_len=20,
    bbp_len=13,
    ma_len=20,
    st_len=10,
    st_mult=3,
    reg_len=25,
    ms_len=5,
):
    long_sig, short_sig = _compute_market_sentiment_signals(
        close=df["Close"].values,
        high=df["High"].values,
        low=df["Low"].values,
        rsi_len=rsi_len,
        stoch_k_len=stoch_k_len,
        stoch_k_smooth=stoch_k_smooth,
        cci_len=cci_len,
        bbp_len=bbp_len,
        ma_len=ma_len,
        st_len=st_len,
        st_mult=st_mult,
        reg_len=reg_len,
        ms_len=ms_len,
    )
    df_result = df.copy()
    long = long_sig.astype(int)
    short = short_sig.astype(int)
    df_result["market_sentiment_signal"] = long - short
    df_result["market_sentiment_signal"] = df_result["market_sentiment_signal"].replace(
        0, np.nan
    )
    df_result["market_sentiment_signal"] = (
        df_result["market_sentiment_signal"].ffill(limit=5).fillna(0).astype(int)
    )
    return df_result


@njit
def compute_pivots_lookback(low, high, window):
    n = len(low)
    pivots = np.zeros(n, dtype=np.int8)
    for row in range(window, n):
        is_low = True
        is_high = True
        for i in range(row - window, row):
            if low[row] > low[i]:
                is_low = False
            if high[row] < high[i]:
                is_high = False
        if is_low and is_high:
            pivots[row] = 3
        elif is_high:
            pivots[row] = 1
        elif is_low:
            pivots[row] = 2
    return pivots


@njit
def compute_ema_signal_numba(open_, close, ema, backcandles):
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)
    for row in range(backcandles, n):
        upt = True
        dnt = True
        for i in range(row - backcandles, row):
            if max(open_[i], close[i]) >= ema[i]:
                dnt = False
            if min(open_[i], close[i]) <= ema[i]:
                upt = False
        if upt and dnt:
            signal[row] = 3
        elif upt:
            signal[row] = 2
        elif dnt:
            signal[row] = 1
    return signal


def ema_pivot_signals(df, ema_length=45, backcandles=15, pivot_window=10):
    ema = df["Close"].ewm(span=ema_length, adjust=False).mean()
    df["EMASignal"] = compute_ema_signal_numba(
        df["Open"].values, df["Close"].values, ema.values, backcandles
    )
    df["isPivot"] = compute_pivots_lookback(
        df["Low"].values, df["High"].values, pivot_window
    )
    df["isPivot"] = df["isPivot"].replace(0, np.nan)
    df["isPivot"] = df["isPivot"].ffill(limit=5).fillna(0).astype(int)
    df["pointpos"] = np.where(
        df["isPivot"] == 2,
        df["Low"] - 1e-3,
        np.where(df["isPivot"] == 1, df["High"] + 1e-3, 0),
    )
    df["pointpos"] = df["pointpos"].replace(0, np.nan)
    df["pointpos"] = df["pointpos"].ffill(limit=5).fillna(0).astype(int)
    return df


@njit
def compute_harsi_signals(
    close,
    high,
    low,
    rsi,
    stoch_k,
    stoch_d,
    ema,
    vol_osc,
    adx_p,
    adx_m,
    vi_p,
    vi_m,
    ost=80,
    use_ema=True,
    use_vol=True,
    use_adx=True,
    use_vortex=True,
):
    n = len(close)
    buy = np.zeros(n, dtype=np.int8)
    sell = np.zeros(n, dtype=np.int8)
    prev_k, prev_d = stoch_k[0], stoch_d[0]

    for i in range(1, n):
        hasi = rsi[i] - rsi[i - 1]
        cross_up = (
            (prev_k < prev_d) and (stoch_k[i] > stoch_d[i]) and (stoch_k[i] < ost)
        )
        cross_down = (
            (prev_k > prev_d) and (stoch_k[i] < stoch_d[i]) and (stoch_k[i] > ost)
        )

        ok = (hasi > 0) and cross_up
        if use_ema:
            ok &= close[i] > ema[i]
        if use_vol:
            ok &= vol_osc[i] > 0
        if use_adx:
            ok &= (adx_p[i] > adx_m[i]) and (adx_p[i] > 25)
        if use_vortex:
            ok &= vi_p[i] > vi_m[i]

        buy[i] = 1 if ok else 0
        sell[i] = 1 if hasi < 0 and cross_down else 0

        prev_k, prev_d = stoch_k[i], stoch_d[i]

    return np.where(buy == 1, 1, np.where(sell == 1, -1, 0))


def harsi_strategy(
    df,
    use_ema=True,
    use_vol=True,
    use_adx=True,
    use_vortex=True,
    ost=80,
    rsi_period=14,
    fastk_period=14,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0,
    ema_period=30,
    volume_sma_slow=20,
    volume_sma_fast=5,
    adx_p_period=14,
    adx_m_period=14,
):
    close, high, low, volume = (
        df["Close"].values,
        df["High"].values,
        df["Low"].values,
        df["Volume"].values,
    )
    rsi = talib.RSI(close, timeperiod=rsi_period)
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype,
    )
    ema = talib.EMA(close, timeperiod=ema_period)
    vol_osc = talib.SMA(volume, timeperiod=volume_sma_fast) - talib.SMA(
        volume, timeperiod=volume_sma_slow
    )
    adx_p = talib.PLUS_DI(high, low, close, timeperiod=adx_p_period)
    adx_m = talib.MINUS_DI(high, low, close, timeperiod=adx_m_period)
    vi_p = adx_p
    vi_m = adx_m
    signal = compute_harsi_signals(
        close=close,
        high=high,
        low=low,
        rsi=rsi,
        stoch_k=slowk,
        stoch_d=slowd,
        ema=ema,
        vol_osc=vol_osc,
        adx_p=adx_p,
        adx_m=adx_m,
        vi_p=vi_p,
        vi_m=vi_m,
        ost=ost,
        use_ema=use_ema,
        use_vol=use_vol,
        use_adx=use_adx,
        use_vortex=use_vortex,
    )
    df["harsi_signal"] = signal
    df["harsi_signal"] = df["harsi_signal"].replace(0, np.nan)
    df["harsi_signal"] = df["harsi_signal"].ffill(limit=5).fillna(0).astype(int)
    return df


def apply_strategies(df, context):
    df = mark_bwab_signals(df, **context["mark_bwab_signals"])
    df = apply_triple_ma(df, **context["apply_triple_ma"])
    df = generate_market_bottom_signals(df, **context["generate_market_bottom_signals"])
    df = williams_r_signals(df, **context["williams_r_signals"])
    df = lowest_highest_tick_strategy(df, **context["lowest_highest_tick_strategy"])
    df = compute_two_pole_oscillator(df, **context["compute_two_pole_oscillator"])
    df = apply_adaptive_t3_signals(df, **context["apply_adaptive_t3_signals"])
    df = next_pivot_signals(df, **context["next_pivot_signals"])
    df = market_sentiment_signals(
        df, **context["market_sentiment_signals"]
    )
    df = ema_pivot_signals(df, **context["ema_pivot_signals"])
    df = harsi_strategy(df, **context["harsi_strategy"])
    return df
