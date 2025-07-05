from datetime import timedelta
import pandas as pd
import numpy as np
import requests
import math
from numba import njit
from data_preprocessing.technical_features import get_technical_indicator_features
from tenacity import retry


FEATURE_NAMES = [
    "high_low_diff", "high_distance", "low_distance", "high_occur_first",
    "H_High", "H_Low", "H_Open", "H_Close", "H_Volume",
    "C_High", "C_Low", "C_Open", "C_Close", "C_Volume"
]
_KIND = {"random_walk": 0, "price": 1, "change": 2}



@retry
def fetch_coinmetrics_asset_metrics(
    assets: str = "btc",
    metrics: str = "AdrActCnt,NVTAdj,TxCnt",
    frequency: str = "1d",
    start_time: str = "2019-01-01",
    end_time: str = None
) -> pd.DataFrame:
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": assets,
        "metrics": metrics,
        "frequency": frequency,
    }
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time

    resp = requests.get(url, params=params, headers={"Accept-Encoding": "gzip"})
    resp.raise_for_status()
    data = resp.json().get("data", [])

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    metric_list = [m.strip() for m in metrics.split(",")]
    for m in metric_list:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    df = df.set_index(["time", "asset"])
    df_wide = df[metric_list].unstack("asset")
    df_wide.columns = [f"{metric}" for metric, asset in df_wide.columns]
    df_wide = df_wide.sort_index(ascending=True)
    df_wide.index.name="datetime"
    df_wide.index = df_wide.index + timedelta(days=1)
    return df_wide


@retry
def get_fear_greed_index():
    resp = requests.get("https://api.alternative.me/fng/?limit=0")
    data = resp.json()["data"]
    df_fng = (
        pd.DataFrame(data)
        .assign(date=lambda df: pd.to_datetime(df.timestamp, unit="s", utc=True))
        .set_index("date")["value"]
        .astype(int)
        .rename("FearGreedIndex")
    )
    return df_fng


@njit(cache=True, fastmath=True, parallel=True, inline="always")
def _std_sample(x):
    n = x.size
    if n <= 1:
        return 0.0
    mu = x.sum() / n
    return math.sqrt(((x - mu) ** 2).sum() / (n - 1))

@njit(cache=True, inline="always")
def _rs_full(buf, kind_id):
    if kind_id == 0:
        incs = buf[1:] - buf[:-1]
        mean_inc = (buf[-1] - buf[0]) / incs.size
    elif kind_id == 1:                
        incs = buf[1:] / buf[:-1] - 1.0
        mean_inc = incs.sum() / incs.size
    else:
        incs = buf
        mean_inc = incs.sum() / incs.size

    dev = incs - mean_inc
    Z   = np.cumsum(dev)
    R   = Z.max() - Z.min()
    if R == 0.0:
        return 0.0
    S   = _std_sample(incs)
    if S == 0.0:
        return 0.0
    return R / S


@njit(cache=True)
def _hurst_core(series, kind_id, windows):
    RS = np.empty(windows.size, np.float64)

    for i in range(windows.size):
        w, acc, cnt = windows[i], 0.0, 0
        for start in range(0, series.size, w):
            if start + w > series.size:
                break
            val = _rs_full(series[start:start+w], kind_id)
            if val != 0.0:
                acc += val;  cnt += 1
        RS[i] = acc / cnt if cnt else np.nan

    mask = np.isfinite(RS) & (RS > 0.0)
    if mask.sum() < 2:
        return np.nan, np.nan

    log_w  = np.log10(windows[mask].astype(np.float64))
    log_rs = np.log10(RS[mask])
    mx, my = log_w.mean(), log_rs.mean()
    cov    = ((log_w - mx) * (log_rs - my)).sum()
    var    = ((log_w - mx) ** 2).sum()
    H      = cov / var
    c      = math.pow(10.0, my - H * mx)
    return H, c



def compute_Hc(series, *, kind="price",
               min_window=10, max_window=None, step=0.25):
    if len(series) < min_window:
        raise ValueError("Series length must be >= 100")

    if isinstance(series, (pd.Series, np.ndarray)):
        buf = np.asarray(series, dtype=np.float64)
    else:
        buf = np.array(series, dtype=np.float64)
    if not np.isfinite(buf).all():
        raise ValueError("Series contains NaNs or infs")

    kind_id = _KIND[kind]
    max_window = max_window or (len(buf) - 1)

    ws = np.power(
        10.0,
        np.arange(math.log10(min_window), math.log10(max_window), step)
    ).astype(np.int64)
    ws = np.unique(ws)
    if ws[-1] != len(buf):
        ws = np.append(ws, len(buf))

    H, c = _hurst_core(buf, kind_id, ws)
    return H, c, [ws.tolist(), []]



def compute_features(subdf):
    if len(subdf) < 101:
        return pd.Series(
            {
                "high_low_diff": 0,
                "high_distance": 0,
                "low_distance": 0,
                "high_occur_first": 0,
                "H_High": 0,
                "H_Low": 0,
                "H_Open": 0,
                "H_Close": 0,
                "H_Volume": 0,
                "C_High": 0,
                "C_Low": 0,
                "C_Open": 0,
                "C_Close": 0,
                "C_Volume": 0,
            }
        )
    try:
        highs = subdf["High"].values
        lows = subdf["Low"].values
    except Exception as e:
        raise e
    pos_high = highs.argmax()
    pos_low = lows.argmin()
    n_bars = len(subdf)
    high_low_diff = abs(pos_high - pos_low) / n_bars if n_bars != 0 else 0
    high_distance = pos_high / n_bars if n_bars != 0 else 0
    low_distance = pos_low / n_bars if n_bars != 0 else 0
    high_occur_first = int(pos_high < pos_low)

    try:
        H_Open, c_Open, _ = compute_Hc(subdf["Open"], kind="random_walk")
    except:
        H_Open, c_Open = 0,0
    try:
        H_Close, c_Close, _ = compute_Hc(subdf["Close"], kind="random_walk")
    except:
        H_Close, c_Close = 0,0
    return pd.Series(
        {
            "high_low_diff": high_low_diff,
            "high_distance": high_distance,
            "low_distance": low_distance,
            "high_occur_first": high_occur_first,
            "H_Open": H_Open,
            "H_Close": H_Close,
            "C_Open": c_Open,
            "C_Close": c_Close,
        }
    )




def get_date_time_feature(df):
    df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)
    df["weekday"] = df.index.weekday
    df["hour"] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    return df

def safe_divide(numerator, denominator, fill_value=1):
    return np.where(denominator == 0, fill_value, numerator / denominator)

def rolling_rank_percentile(x, w):
    return x.rolling(window=w).rank(pct=True)


def candle_features(df):
    for col in ['Open', 'Close', 'High', 'Low']:
        df[col] = df[col].astype(float)
    df["rolling_quantile"] = rolling_rank_percentile(df['Close'].pct_change(), 25)

    df["low_pct_open"] = safe_divide(df['Open'] - df['Low'], df['Open'])
    df["low_pct_open_rolling_quantile"] = rolling_rank_percentile(df['low_pct_open'], 15)

    df["low_pct_close"] = safe_divide(df['Close'] - df['Low'], df['Close'])
    df["low_pct_close_rolling_quantile"] = rolling_rank_percentile(df['low_pct_close'], 15)

    df["high_pct_open"] = safe_divide(df['High'] - df['Open'], df['Open'])
    df["high_pct_open_rolling_quantile"] = rolling_rank_percentile(df['high_pct_open'], 15)

    df["high_pct_close"] = safe_divide(df['High'] - df['Close'], df['Close'])
    df["high_pct_close_rolling_quantile"] = rolling_rank_percentile(df['high_pct_close'], 15)

    df["open_close_pct"] = safe_divide(df["Close"] - df["Open"], df["Open"])
    df["open_close_pct_rolling_quantile"] = rolling_rank_percentile(df['open_close_pct'], 15)

    df["high_low_comp_open"] = safe_divide(df["low_pct_close"], df["low_pct_open"])
    df["high_low_comp_open_rolling_quantile"] = rolling_rank_percentile(df['high_low_comp_open'], 15)

    df["high_low_comp_close"] = safe_divide(df["high_pct_close"], df["high_pct_open"])
    df["high_low_comp_close_rolling_quantile"] = rolling_rank_percentile(df['high_low_comp_close'], 15)

    df["body_size_relative"] = safe_divide(abs(df["Close"] - df["Open"]), df["High"] - df["Low"])
    df["body_size_relative_rolling_quantile"] = rolling_rank_percentile(df['body_size_relative'], 15)

    candle_range = df["High"] - df["Low"]
    df["upper_shadow_pct"] = safe_divide(df["High"] - df[["Close", "Open"]].max(axis=1), candle_range)
    df["upper_shadow_pct_rolling_quantile"] = rolling_rank_percentile(df['upper_shadow_pct'], 15)

    df["lower_shadow_pct"] = safe_divide(df[["Close", "Open"]].min(axis=1) - df["Low"], candle_range)
    df["lower_shadow_pct_rolling_quantile"] = rolling_rank_percentile(df['lower_shadow_pct'], 15)

    df["is_inside_bar"] = ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).astype(int)

    df.fillna(0, inplace=True)
    return df


@njit(cache=True)
def _vwap_numba(high, low, close, volume):
    cum_pv = 0.0
    cum_vol = 0.0
    for i in range(high.shape[0]):
        tp = (high[i] + low[i] + close[i]) / 3.0
        v  = volume[i]
        cum_pv += tp * v
        cum_vol += v
    return cum_pv / cum_vol if cum_vol > 0.0 else np.nan

def vwap_group(grp: pd.DataFrame) -> float:
    h = grp['High'].values
    l = grp['Low'].values
    c = grp['Close'].values
    v = grp['Volume'].values
    return _vwap_numba(h, l, c, v)

@njit(cache=True)
def kurtosis_numba(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 4:
        return np.nan
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n
    m2 = 0.0
    m4 = 0.0
    for i in range(n):
        d = x[i] - mean
        d2 = d * d
        m2 += d2
        m4 += d2 * d2
    m2 /= n
    m4 /= n
    if m2 <= 0.0:
        return np.nan
    return m4 / (m2 * m2) - 3.0


def kurtosis_pct_change(series: pd.Series) -> float:
    arr = series.values
    if arr.size < 2:
        return np.nan
    ret = (arr[1:] - arr[:-1]) / arr[:-1]
    return kurtosis_numba(ret)



def create_ohlcv_features(df_ohlcv, resample_timeframe):
    resampler = df_ohlcv.resample(resample_timeframe, label="right", closed="right")
    df_ohlcv_resampled = resampler.agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
        KurtosisPctClose=('Close', kurtosis_pct_change)
    )

    df_ta = get_technical_indicator_features(
        df_ohlcv_resampled.copy(),
        open_col="Open",
        high_col="High",
        low_col="Low",
        close_col="Close",
        volume_col="Volume",
    )

    vwap = resampler.apply(vwap_group)
    df_ohlcv_resampled["vwap"] = vwap
    df_ohlcv_resampled["open_vwap"] = (df_ohlcv_resampled['Open'] > df_ohlcv_resampled['vwap'].rolling(15).mean().fillna(0)).astype(int)

    df_ohlcv_resampled['vwap_ratio'] = 1 - vwap/df_ohlcv_resampled['Close']
    df_ohlcv_resampled['vwap_ratio_rolling'] = safe_divide(df_ohlcv_resampled['vwap_ratio'], df_ohlcv_resampled['vwap_ratio'].rolling(20).mean().fillna(0))
    df_ohlcv_resampled['liquidity_ratio'] = safe_divide(df_ohlcv_resampled['Volume'], df_ohlcv_resampled['Volume'].rolling(24).std().fillna(0))

    df_ohlcv_resampled['kurtosis_30'] = df_ohlcv_resampled["Close"].rolling(30).apply(lambda x: kurtosis_pct_change(x))

    df_combined = pd.merge(df_ohlcv_resampled, df_ta, left_index=True, right_index=True)
    df_combined["hurst_30"] = df_combined["Close"].rolling(30).apply(lambda x: compute_Hc(x)[0])
    df_final = get_date_time_feature(df_combined)

    return df_final

