import numpy as np

def garman_klass_volatility(df):
    log_high_low = np.log(df["High"] / df["Low"])
    log_close_open = np.log(df["Close"] / df["Open"])
    gk_variance = 0.5 * log_high_low**2 - (2 * np.log(2) - 1) * log_close_open**2
    gk_volatility = np.sqrt(gk_variance)
    df["garman_klass_volatility"] = gk_volatility
    df["log_high_low"] = log_high_low
    df["log_close_open"] = log_close_open
    del log_high_low, log_close_open, gk_variance, gk_volatility
    return df


def parkinson_volatility(df):
    log_high_low = np.log(df["High"] / df["Low"])
    parkinson_variance = (1 / (4 * np.log(2))) * log_high_low**2
    parkinson_volatility = np.sqrt(parkinson_variance)
    df["parkinson_volatility"] = parkinson_volatility
    del log_high_low, parkinson_variance, parkinson_volatility
    return df


def rogers_satchell_volatility(df):
    log_high_open = np.log(df["High"] / df["Open"])
    log_low_open = np.log(df["Low"] / df["Open"])
    log_close_open = np.log(df["Close"] / df["Open"])
    rs_variance = log_high_open * (log_high_open - log_close_open) + log_low_open * (
        log_low_open - log_close_open
    )
    rs_volatility = np.sqrt(rs_variance)
    df["rogers_satchell_volatility"] = rs_volatility
    df["log_low_open"] = log_low_open
    del log_high_open, log_low_open, log_close_open, rs_variance, rs_volatility
    return df


def yang_zhang_volatility(df, k):
    log_open_prev_close = np.log(df["Open"] / df["Close"].shift(1))
    log_close_open = np.log(df["Close"] / df["Open"])
    log_high_low = np.log(df["High"] / df["Low"])
    log_close_prev_close = np.log(df["Close"] / df["Close"].shift(1))
    close_volatility = log_close_prev_close**2
    open_volatility = log_open_prev_close**2
    rs_volatility = log_high_low * (log_high_low - log_close_open)
    yz_variance = open_volatility + k * close_volatility + (1 - k) * rs_volatility
    yz_volatility = np.sqrt(yz_variance)
    df["yang_zhang_volatility"] = yz_volatility
    df["rs_volatility"] = rs_volatility
    del log_open_prev_close, log_close_open, log_high_low, log_close_prev_close
    del close_volatility, open_volatility, rs_volatility, yz_variance, yz_volatility
    return df


def close_to_close_volatility(df, window=5):
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    volatility = log_returns.rolling(window=window).std()
    df["close_to_close_volatility"] = volatility
    del log_returns, volatility
    return df


def get_volatility_features(df, context):
    df = garman_klass_volatility(df)
    df = parkinson_volatility(df)
    df = rogers_satchell_volatility(df)
    df = yang_zhang_volatility(df, **context['yang_zhang_volatility'])
    df = close_to_close_volatility(df, **context['close_to_close_volatility'])
    return df
