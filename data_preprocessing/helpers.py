from data_preprocessing.base_features import create_ohlcv_features
from data_preprocessing.context import load_context
from data_preprocessing.patterns import get_pattern_recognition_features, get_pattern_recognition_features_new
from data_preprocessing.strategies import apply_strategies
from data_preprocessing.technical_features import drop_ohlcv
from data_preprocessing.volatility import get_volatility_features
import numpy as np
import pandas as pd


def create_feature_data(df, timeframe, token):
    context = load_context(token, timeframe)
    ohlcv = create_ohlcv_features(df, timeframe)
    data = get_volatility_features(ohlcv, context)
    data = get_pattern_recognition_features(data, context)
    data = get_pattern_recognition_features_new(data, context)
    data = apply_strategies(data, context)
    data.index = ohlcv.index
    data = drop_ohlcv(data)
    return data



def fill_nan_inf(df):
    df_filled = df.replace([np.inf, -np.inf], [0, 0])
    df_filled = df_filled.fillna(df_filled.mean())
    return df_filled


def merge_timeframes(
    tf_dfs,
    order,
    suffix_fmt="_{tf}"
):
    if not order:
        raise ValueError("`order` must contain at least one timeframe key.")

    missing = [tf for tf in order if tf not in tf_dfs]
    if missing:
        raise KeyError(f"The following timeframes are missing from tf_dfs: {missing}")

    base_key = order[0]
    base = tf_dfs[base_key].copy()
    base.index = pd.to_datetime(base.index, utc=True)
    merged = base.copy()

    base_suffix = suffix_fmt.format(tf=base_key)
    if base_suffix:
        merged.columns = [col + base_suffix for col in merged.columns]

    for tf in order[1:]:
        df = tf_dfs[tf].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        suffix = suffix_fmt.format(tf=tf)
        df_suff = df.add_suffix(suffix)
        merged = merged.join(df_suff, how='left')
        new_cols = df_suff.columns
        merged[new_cols] = merged[new_cols].ffill()
    return merged


