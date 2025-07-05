from data_preprocessing.backtest_helpers import convert_to_long_short, extrapolate_signals, get_predictions, print_token_report, run_positional_backtest, vectorbt_backtest_token
from data_preprocessing.helpers import (
    create_feature_data,
    fill_nan_inf,
    merge_timeframes,
)
import numpy as np
import pandas as pd
from data_preprocessing.strategies import compute_atr_numba
from numba import njit


@njit(cache=True)
def _calculate_normal_targets(
    close,
    high,
    low,
    tp_percent,
    sl_percent,
    max_bars,
    long_targets,
    short_targets,
    bars_to_exit,
    n,
):
    tp_percent /= 100
    sl_percent /= 100
    for i in range(n - 1):
        entry_price = close[i]
        long_tp = entry_price * (1 + tp_percent)
        long_sl = entry_price * (1 - sl_percent)
        short_tp = entry_price * (1 - tp_percent)
        short_sl = entry_price * (1 + sl_percent)
        long_outcome = 0
        short_outcome = 0
        exit_bar = -1
        for j in range(i + 1, min(i + max_bars + 1, n)):
            curr_high = high[j]
            curr_low = low[j]
            if long_outcome == 0:
                if curr_high >= long_tp and curr_low <= long_sl:
                    if j > i + 1:
                        prev_close = close[j - 1]
                        if prev_close > entry_price:
                            long_outcome = 1
                        else:
                            long_outcome = -1
                    else:
                        long_outcome = (
                            1 if curr_high - long_tp <= long_sl - curr_low else -1
                        )
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_high >= long_tp:
                    long_outcome = 1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_low <= long_sl:
                    long_outcome = -1
                    if exit_bar == -1:
                        exit_bar = j - i
            if short_outcome == 0:
                if curr_low <= short_tp and curr_high >= short_sl:
                    if j > i + 1:
                        prev_close = close[j - 1]
                        if prev_close < entry_price:
                            short_outcome = 1
                        else:
                            short_outcome = -1
                    else:
                        short_outcome = (
                            1 if short_tp - curr_low <= curr_high - short_sl else -1
                        )
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_low <= short_tp:
                    short_outcome = 1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_high >= short_sl:
                    short_outcome = -1
                    if exit_bar == -1:
                        exit_bar = j - i
            if long_outcome != 0 and short_outcome != 0:
                break
        if i + 1 < n:
            if long_outcome == 1:
                long_targets[i + 1] = 1
            if short_outcome == 1:
                short_targets[i + 1] = 1
            if exit_bar > 0:
                bars_to_exit[i + 1] = exit_bar - 1



@njit(cache=True)
def _calculate_targets_atr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_bars: int,
    long_targets: np.ndarray,
    short_targets: np.ndarray,
    bars_to_exit: np.ndarray,
    n: int,
):
    min_tp_pct = 0.005
    max_tp_pct = 0.04

    for i in range(n - 1):
        if np.isnan(atr[i]):
            continue

        entry_price = close[i]
        atr_val      = atr[i]

        tp_dist = tp_mult * atr_val
        sl_dist = sl_mult * atr_val

        tp_pct = tp_dist / entry_price
        if tp_pct < min_tp_pct or tp_pct > max_tp_pct:
            continue

        long_tp  = entry_price + tp_dist
        long_sl  = entry_price - sl_dist
        short_tp = entry_price - tp_dist
        short_sl = entry_price + sl_dist

        long_outcome  = 0
        short_outcome = 0
        exit_bar      = -1

        stop = i + max_bars + 1
        if stop > n:
            stop = n

        for j in range(i + 1, stop):
            curr_high = high[j]
            curr_low  = low[j]

            if long_outcome == 0:
                if curr_high >= long_tp and curr_low <= long_sl:
                    if j > i + 1:
                        prev_close = close[j - 1]
                        long_outcome = 1 if prev_close > entry_price else -1
                    else:
                        long_outcome = 1 if (curr_high - long_tp) <= (long_sl - curr_low) else -1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_high >= long_tp:
                    long_outcome = 1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_low <= long_sl:
                    long_outcome = -1
                    if exit_bar == -1:
                        exit_bar = j - i

            if short_outcome == 0:
                if curr_low <= short_tp and curr_high >= short_sl:
                    if j > i + 1:
                        prev_close = close[j - 1]
                        short_outcome = 1 if prev_close < entry_price else -1
                    else:
                        short_outcome = 1 if (short_tp - curr_low) <= (curr_high - short_sl) else -1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_low <= short_tp:
                    short_outcome = 1
                    if exit_bar == -1:
                        exit_bar = j - i
                elif curr_high >= short_sl:
                    short_outcome = -1
                    if exit_bar == -1:
                        exit_bar = j - i

            if long_outcome != 0 and short_outcome != 0:
                break

        if i + 1 < n:
            if long_outcome == 1:
                long_targets[i + 1] = 1
            elif long_outcome == -1:
                long_targets[i + 1] = -1

            if short_outcome == 1:
                short_targets[i + 1] = 1
            elif short_outcome == -1:
                short_targets[i + 1] = -1

            if exit_bar > 0:
                bars_to_exit[i + 1] = exit_bar - 1


def create_trade_targets(
    df, target_type="atr", tp=2.5, sl=1, max_bars=60 * 10, atr_period=60 * 5
):
    result = df.copy()
    if max_bars is None:
        max_bars = len(df) - 1
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    long_targets = np.zeros(n)
    short_targets = np.zeros(n)
    bars_to_exit = np.full(n, np.nan)

    if target_type == "atr":
        atr = compute_atr_numba(high, low, close, atr_period)
        _calculate_targets_atr(
            close=close,
            high=high,
            low=low,
            atr=atr,
            tp_mult=tp,
            sl_mult=sl,
            max_bars=max_bars,
            long_targets=long_targets,
            short_targets=short_targets,
            bars_to_exit=bars_to_exit,
            n=n
        )
        result["Long_Target"] = long_targets
        result["Short_Target"] = short_targets
        result["Bars_To_Exit"] = bars_to_exit
        mask_long  = result["Long_Target"]  == 1
        mask_short = result["Short_Target"] == 1
        result["Target"] = 0
        result.loc[mask_long,  "Target"] = 1
        result.loc[mask_short, "Target"] = 2
        # result = result.drop(["Long_Target", "Short_Target"], axis=1)
        return result['Target']
    else:
        _calculate_normal_targets(
            close,
            high,
            low,
            tp,
            sl,
            max_bars,
            long_targets,
            short_targets,
            bars_to_exit,
            n,
        )
        result["Long_Target"] = long_targets
        result["Short_Target"] = short_targets
        result["Bars_To_Exit"] = bars_to_exit
        result["Target"] = 0
        result.loc[
            (result["Long_Target"] == 1) & (result["Short_Target"] == 0), "Target"
        ] = 1
        result.loc[
            (result["Long_Target"] == 0) & (result["Short_Target"] == 1), "Target"
        ] = 2
        result = result.drop(["Long_Target", "Short_Target"], axis=1)
        return result["Target"]


def create_all_features(
    df, token, target_type="atr", tp=0.01, sl=0.01, max_bars=60 * 24, atr_period=60 * 5
):
    data_1h = (
        fill_nan_inf(create_feature_data(df, "1h", token))
        .round(5)
        .sort_index(ascending=True)
    )
    data_2h = (
        fill_nan_inf(create_feature_data(df, "2h", token))
        .round(5)
        .sort_index(ascending=True)
    )
    data_4h = (
        fill_nan_inf(create_feature_data(df, "4h", token))
        .round(5)
        .sort_index(ascending=True)
    )

    target = (
        create_trade_targets(
            df,
            target_type=target_type,
            tp=tp,
            sl=sl,
            max_bars=max_bars,
            atr_period=atr_period,
        )
        .shift(-1)
        .fillna(0)
    )

    # try:
    if data_1h.index.max() in target.index:
        y_1h = target[data_1h.index]
    else:
        y_1h = target[data_1h.index[:-1]]
        data_1h = data_1h.iloc[:-1, :]

    if data_2h.index.max() in target.index:
        y_2h = target[data_2h.index]
    else:
        y_2h = target[data_2h.index[:-1]]
        data_2h = data_2h.iloc[:-1, :]

    if data_4h.index.max() in target.index:
        y_4h = target[data_4h.index]
    else:
        y_4h = target[data_4h.index[:-1]]
        data_4h = data_4h.iloc[:-1, :]
    # except:
    #     print(target)

    data_1h_2h_4h = merge_timeframes(
        tf_dfs={"1h": data_1h, "2h": data_2h, "4h": data_4h},
        order=["1h", "2h", "4h"],
    ).dropna()
    y_1h_2h_4h = target[data_1h_2h_4h.index]

    data_1h_2h = merge_timeframes(
        tf_dfs={"1h": data_1h, "2h": data_2h},
        order=["1h", "2h"],
    ).dropna()
    y_1h_2h = target[data_1h_2h.index]

    data_1h_4h = merge_timeframes(
        tf_dfs={"1h": data_1h, "4h": data_4h},
        order=["1h", "4h"],
    ).dropna()
    y_1h_4h = target[data_1h_4h.index]

    data_2h_4h = merge_timeframes(
        tf_dfs={"2h": data_2h, "4h": data_4h},
        order=["2h", "4h"],
    ).dropna()
    y_2h_4h = target[data_2h_4h.index]

    return {
        f"X_1h": data_1h,
        f"y_1h": y_1h,
        f"X_2h": data_2h,
        f"y_2h": y_2h,
        f"X_4h": data_4h,
        f"y_4h": y_4h,
        f"X_combined_1h_2h_4h": data_1h_2h_4h,
        f"y_combined_1h_2h_4h": y_1h_2h_4h,
        f"X_combined_1h_4h": data_1h_4h,
        f"y_combined_1h_4h": y_1h_4h,
        f"X_combined_1h_2h": data_1h_2h,
        f"y_combined_1h_2h": y_1h_2h,
        f"X_combined_2h_4h": data_2h_4h,
        f"y_combined_2h_4h": y_2h_4h,
    }


def merge_token_helper(main_df, comp_df, complementry_token_name):
    btc_renamed = comp_df.add_suffix(f"_{complementry_token_name}")
    token_renamed = main_df.add_suffix("_main")
    merged = pd.merge(
        btc_renamed, token_renamed, left_index=True, right_index=True, how="inner"
    ).dropna()
    return merged


def merge_token_data(data_token_main, data_token_comp, complementry_token_name):
    main_1h = data_token_main[f"X_1h"]
    comp_1h = data_token_comp[f"X_1h"]
    merged_1h = merge_token_helper(main_df=main_1h, comp_df=comp_1h, complementry_token_name=complementry_token_name)
    y_merged_1h = data_token_main["y_1h"][merged_1h.index]

    main_2h = data_token_main[f"X_2h"]
    comp_2h = data_token_comp[f"X_2h"]
    merged_2h = merge_token_helper(main_df=main_2h, comp_df=comp_2h, complementry_token_name=complementry_token_name)
    y_merged_2h = data_token_main["y_2h"][merged_2h.index]

    main_4h = data_token_main[f"X_4h"]
    comp_4h = data_token_comp[f"X_4h"]
    merged_4h = merge_token_helper(main_df=main_4h, comp_df=comp_4h, complementry_token_name=complementry_token_name)
    y_merged_4h = data_token_main["y_4h"][merged_4h.index]

    main_combined_1h_2h_4h = data_token_main[f"X_combined_1h_2h_4h"]
    comp_combined_1h_2h_4h = data_token_comp[f"X_combined_1h_2h_4h"]
    merged_combined_1h_2h_4h = merge_token_helper(
        main_df=main_combined_1h_2h_4h, comp_df=comp_combined_1h_2h_4h, complementry_token_name=complementry_token_name
    )
    y_merged_combined_1h_2h_4h = data_token_main["y_combined_1h_2h_4h"][
        merged_combined_1h_2h_4h.index
    ]

    main_combined_1h_2h = data_token_main[f"X_combined_1h_2h"]
    comp_combined_1h_2h = data_token_comp[f"X_combined_1h_2h"]
    merged_combined_1h_2h = merge_token_helper(
        main_df=main_combined_1h_2h, comp_df=comp_combined_1h_2h, complementry_token_name=complementry_token_name
    )
    y_merged_combined_1h_2h = data_token_main["y_combined_1h_2h"][
        merged_combined_1h_2h.index
    ]

    main_combined_1h_4h = data_token_main[f"X_combined_1h_4h"]
    comp_combined_1h_4h = data_token_comp[f"X_combined_1h_4h"]
    merged_combined_1h_4h = merge_token_helper(
        main_df=main_combined_1h_4h, comp_df=comp_combined_1h_4h, complementry_token_name=complementry_token_name
    )
    y_merged_combined_1h_4h = data_token_main["y_combined_1h_4h"][
        merged_combined_1h_4h.index
    ]

    main_combined_2h_4h = data_token_main[f"X_combined_2h_4h"]
    comp_combined_2h_4h = data_token_comp[f"X_combined_2h_4h"]
    merged_combined_2h_4h = merge_token_helper(
        main_df=main_combined_2h_4h, comp_df=comp_combined_2h_4h, complementry_token_name=complementry_token_name
    )
    y_merged_combined_2h_4h = data_token_main["y_combined_2h_4h"][
        merged_combined_2h_4h.index
    ]

    return {
        f"X_1h": merged_1h,
        f"y_1h": y_merged_1h,
        f"X_2h": merged_2h,
        f"y_2h": y_merged_2h,
        f"X_4h": merged_4h,
        f"y_4h": y_merged_4h,
        f"X_combined_1h_2h_4h": merged_combined_1h_2h_4h,
        f"y_combined_1h_2h_4h": y_merged_combined_1h_2h_4h,
        f"X_combined_1h_4h": merged_combined_1h_4h,
        f"y_combined_1h_4h": y_merged_combined_1h_4h,
        f"X_combined_1h_2h": merged_combined_1h_2h,
        f"y_combined_1h_2h": y_merged_combined_1h_2h,
        f"X_combined_2h_4h": merged_combined_2h_4h,
        f"y_combined_2h_4h": y_merged_combined_2h_4h,
    }



def create_merged_feature(
    main_feature_df,
    complementry_data_df,
    complementry_token_name,
    main_token_name,
    target_type="pct",
    tp=0.01,
    sl=0.01,
    max_bars=60*24,
    atr_period=60*5,
):
    feature = create_all_features(
        df=main_feature_df,
        token=main_token_name,
        target_type=target_type,
        tp=tp, sl=sl,
        max_bars=max_bars,
        atr_period=atr_period,
    )
    comp_feature = create_all_features(
        df=complementry_data_df,
        token=complementry_token_name,
        target_type=target_type,
        tp=tp, sl=sl,
        max_bars=max_bars,
        atr_period=atr_period,
    )

    merged_feature = merge_token_data(
        data_token_main=feature,
        data_token_comp=comp_feature,
        complementry_token_name=complementry_token_name,
    )
    return merged_feature




def evaluate_token_strategy(
    *,
    model,
    data: pd.DataFrame,
    data_1m: pd.DataFrame,
    target_x: str,
    target_y: str,
    val_end_date: pd.Timestamp,
    token_name: str,
    classes: tuple = (0, 1, 2),
    long_proba: float,
    short_proba: float,
    trade_size: float = 1,
    use_proba: bool = True,
    atr_target: bool = True,

    tp: float = 6,
    sl: float = 6,
    atr_period: int = 60 * 5,

    tp_pct: float = 0.006,
    sl_pct: float = 0.006,
    nn = True,
):
    print_token_report(
        model,
        { token_name: (
            data[target_x].loc[val_end_date:],
            data[target_y].loc[val_end_date:]
        )},
        classes=classes,
        pos_conf=long_proba,
        neg_conf=short_proba,
        nn = nn
    )

    test_x = data[target_x].loc[val_end_date:]
    test_1m = data_1m.loc[val_end_date:]

    signals = get_predictions(
        model, test_x,
        long_proba=long_proba,
        short_proba=short_proba,
        use_proba=use_proba,
        nn = nn
    )
    long_sig, short_sig = convert_to_long_short(signals)
    long_sig, short_sig = extrapolate_signals(test_1m, long_sig, short_sig)

    size = pd.Series(trade_size, index=test_1m.index)
    positional_out = run_positional_backtest(
        df=test_1m,
        long_sig=long_sig,
        short_sig=short_sig,
        tp=tp,
        sl=sl,
        long_size=size,
        short_size=size,
        position_exipration=None,
        atr_target=atr_target,
        atr_period=atr_period
    )

    vectorbt_pf = vectorbt_backtest_token(
        test_1m,
        long_sig,
        short_sig,
        tp=tp_pct,
        sl=sl_pct
    )

    return positional_out, vectorbt_pf
