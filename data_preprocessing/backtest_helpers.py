from sklearn.metrics import classification_report
from tabulate import tabulate
import vectorbt as vbt
from vectorbt.portfolio.enums import StopExitPrice, StopEntryPrice, SizeType
from backtester import run_backtest
from numba import njit
import numpy as np
import pandas as pd
from data_preprocessing.strategies import compute_atr_numba


def get_predictions(
    model,
    X,
    use_proba: bool = False,
    long_proba: float = 0.4,
    short_proba: float = 0.4,
    nn: bool = False
) -> pd.Series:
    if nn:
        proba_mat = model.predict(X)
        raw_preds = np.argmax(proba_mat, axis=1).astype(int)
    else:
        raw_preds = model.predict(X).flatten().astype(int)
        proba_mat = None
        if use_proba:
            proba_mat = model.predict_proba(X)

    if use_proba:
        max_proba = proba_mat.max(axis=1)
    else:
        max_proba = None

    final = []
    for i, p in enumerate(raw_preds):
        if use_proba:
            if p == 1 and max_proba[i] >= long_proba:
                final.append(1)
            elif p == 2 and max_proba[i] >= short_proba:
                final.append(2)
            else:
                final.append(0)
        else:
            final.append(p)

    arr = np.array(final).flatten()
    return pd.Series(arr, index=X.index)


def convert_to_long_short(predictions, long_sign=1, short_sign=2):
    long = predictions == long_sign
    short = predictions == short_sign

    return long, short


def extrapolate_signals(X, long_signal, short_signal):
    long = long_signal.reindex(X.index, fill_value=False)
    short = short_signal.reindex(X.index, fill_value=False)

    return long, short


def print_token_report(
    model,
    data,
    classes=(1, 2),
    pos_conf=0.5,
    neg_conf=0.5,
    nn: bool = False
):
    reports = {}
    preds_counts = {}
    all_preds = None

    for name, (X_test, y_test) in data.items():
        if not nn:
            proba_vals = model.predict_proba(X_test)
            raw_preds = model.predict(X_test).flatten().astype(int)
        else:
            proba_vals = model.predict(X_test)
            raw_preds = np.argmax(proba_vals, axis=1).astype(int)

        max_proba = proba_vals.max(axis=1)

        final_preds = []
        for i, p in enumerate(raw_preds):
            if p == classes[0] and max_proba[i] >= pos_conf:
                final_preds.append(classes[0])
            elif p == classes[1] and max_proba[i] >= neg_conf:
                final_preds.append(classes[1])
            else:
                final_preds.append(0)
        preds = np.array(final_preds)

        rep = classification_report(
            y_test.astype(int),
            preds.astype(int),
            output_dict=True,
            zero_division=0
        )

        if all_preds is None:
            all_preds = preds.copy()
        else:
            all_preds += preds
        reports[name] = rep
        preds_counts[name] = pd.Series(preds).value_counts().to_dict()
        preds_counts[f"{name}_org"] = pd.Series(y_test).value_counts().to_dict()

    metrics = ["precision"]
    headers = ["class"] + [f"{nm}_{m}".center(20) for nm in reports for m in metrics]
    table = []
    total_samples = list(data.values())[0][0].shape[0]
    for cls in classes:
        row = [str(cls)]
        for name in reports:
            prec = reports[name].get(str(cls), {}).get("precision", 0.0)
            cnt = preds_counts[name].get(cls, 0)
            cnt2 = preds_counts[f"{name}_org"].get(cls, 0)
            row.append(f"{prec:.2f} ({cnt} | {cnt2} | {total_samples})")
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="github"))
    print(f"Prediction time - {np.mean(all_preds != 0):.2f}")

    last_X, _ = list(data.values())[-1]
    print(f"Start Date => {last_X.index.min()}, End Date => {last_X.index.max()}")


def vectorbt_backtest(
    _open,
    _high,
    _low,
    _close,
    long,
    short,
    tp=0.02,
    sl=0.01,
    size=1,
    max_size=10,
    fee=0.005,
    slippage=0.005,
):
    pf = vbt.Portfolio.from_signals(
        close=_close.to_numpy(),
        entries=long,
        exits=None,
        short_entries=short,
        short_exits=None,
        open=_open.to_numpy(),
        high=_high.to_numpy(),
        low=_low.to_numpy(),
        sl_stop=sl,
        tp_stop=tp,
        stop_entry_price=StopEntryPrice.Price,
        stop_exit_price=StopExitPrice.Price,
        init_cash=100,
        size_type=SizeType.Value,
        size=size,
        max_size=max_size,
        direction="both",
        accumulate=False,
        update_value=False,
        use_stops=True,
        fees=fee,
        slippage=slippage,
        freq="1m",
        cash_sharing=True,
    )
    return pf


def vectorbt_backtest_token(
    df,
    long_signal,
    short_signal,
    tp=0.02,
    sl=0.01,
    size=1,
    max_size=10,
    fee=0.0005,
    slippage=0.0002,
):
    max_index = long_signal.index.max()
    _open = df["Open"].loc[:max_index]
    _high = df["High"].loc[:max_index]
    _low = df["Low"].loc[:max_index]
    _close = df["Close"].loc[:max_index]

    pf = vectorbt_backtest(
        _open,
        _high,
        _low,
        _close,
        long_signal,
        short_signal,
        tp,
        sl,
        size,
        max_size,
        fee,
        slippage,
    )
    return pf


@njit
def get_tp_sl_numba(
    close, high, low, long_arr, short_arr, tp, sl, atr_period, atr_target
):
    n = len(close)
    tp_long = np.full(n, np.nan)
    sl_long = np.full(n, np.nan)
    tp_short = np.full(n, np.nan)
    sl_short = np.full(n, np.nan)

    if atr_target:
        atr = compute_atr_numba(high, low, close, atr_period)
        for i in range(n):
            if long_arr[i]:
                if not np.isnan(atr[i]):
                    tp_long[i] = close[i] + tp * atr[i]
                    sl_long[i] = close[i] - sl * atr[i]
            if short_arr[i]:
                if not np.isnan(atr[i]):
                    tp_short[i] = close[i] - tp * atr[i]
                    sl_short[i] = close[i] + sl * atr[i]
    else:
        tp /= 100
        sl /= 100
        for i in range(n):
            if long_arr[i]:
                tp_long[i] = close[i] * (1 + tp)
                sl_long[i] = close[i] * (1 - sl)
            if short_arr[i]:
                tp_short[i] = close[i] * (1 - tp)
                sl_short[i] = close[i] * (1 + sl)

    return tp_long, sl_long, tp_short, sl_short


def get_tp_sl(df, long, short, tp=2, sl=1, atr_period=60 * 5, atr_target=True):
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    long_arr = long.astype(np.bool_)
    short_arr = short.astype(np.bool_)

    return get_tp_sl_numba(
        close, high, low, long_arr, short_arr, tp, sl, atr_period, atr_target
    )


def ensure_array(x, dtype):
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr, dtype=dtype)


def run_positional_backtest(
    df,
    long_sig,
    short_sig,
    tp,
    sl,
    long_size,
    short_size,
    position_exipration=None,
    entry_fee_rate=0.0005,
    exit_fee_rate=0.0005,
    slippage_rate=0.0002,
    initial_equity=10000,
    atr_period=60 * 5,
    atr_target=True,
):
    ts = ensure_array(df.index.view('int64')/1e9, dtype=np.float64)

    _open = ensure_array(df["Open"],  dtype=np.float64)
    high  = ensure_array(df["High"],  dtype=np.float64)
    low   = ensure_array(df["Low"],   dtype=np.float64)
    close = ensure_array(df["Close"], dtype=np.float64)

    long_sig  = ensure_array(long_sig,  dtype=np.bool_)
    short_sig = ensure_array(short_sig, dtype=np.bool_)

    long_size  = ensure_array(long_size,  dtype=np.float64)
    short_size = ensure_array(short_size, dtype=np.float64)
    if position_exipration is None:
        exp_arr = np.full(len(ts), np.inf, dtype=np.float64)
    else:
        exp_arr = ensure_array(position_exipration, dtype=np.float64)

    long_tp, long_sl, short_tp, short_sl = get_tp_sl(
        df=df, long=long_sig, short=short_sig,
        tp=tp, sl=sl, atr_period=atr_period, atr_target=atr_target
    )
    long_tp   = ensure_array(long_tp,   dtype=np.float64)
    long_sl   = ensure_array(long_sl,   dtype=np.float64)
    short_tp  = ensure_array(short_tp,  dtype=np.float64)
    short_sl  = ensure_array(short_sl,  dtype=np.float64)

    out = run_backtest(
        timestamp        = ts,
        open             = _open,
        high             = high,
        low              = low,
        close            = close,
        long_signals     = long_sig,
        short_signals    = short_sig,
        long_tp          = long_tp,
        long_sl          = long_sl,
        short_tp         = short_tp,
        short_sl         = short_sl,
        long_size        = long_size,
        short_size       = short_size,
        expiration_times = exp_arr,
        entry_fee_rate   = entry_fee_rate,
        exit_fee_rate    = exit_fee_rate,
        slippage_rate    = slippage_rate,
        initial_equity   = initial_equity,
    )
    return out
