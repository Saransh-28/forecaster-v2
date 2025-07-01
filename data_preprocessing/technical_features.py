import numpy as np
from talib import *

def get_cycle_indicator(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    # df['HT_DCPERIOD'] = HT_DCPERIOD(df[close_col])
    # df['HT_DCPHASE'] = HT_DCPHASE(df[close_col])
    # df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = HT_PHASOR(df[close_col])
    df['HT_PHASOR_sine'], df['HT_PHASOR_leadsine'] = HT_SINE(df[close_col])
    df['HT_TRENDMODE'] = HT_TRENDMODE(df[close_col])
    return df


def get_momentum_indicator(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    # df['ADX'] = ADX(_high, _low, _close, timeperiod=14)
    df['ADXR'] = ADXR(_high, _low, _close, timeperiod=14)
    df['APO'] = APO(_close, fastperiod=12, slowperiod=26, matype=0)
    df['AROONOSC'] = AROONOSC(_high, _low, timeperiod=14)
    # df['BOP'] = BOP(_open, _high, _low, _close)
    df['CCI'] = CCI(_high, _low, _close, timeperiod=14)
    df['CMO'] = CMO(_close, timeperiod=14)
    df['DX'] = DX(_high, _low, _close, timeperiod=14)
    df['MFI'] = MFI(_high, _low, _close, _volume, timeperiod=14)
    # df['MINUS_DI'] = MINUS_DI(_high, _low, _close, timeperiod=14)
    # df['MINUS_DM'] = MINUS_DM(_high, _low, timeperiod=14)
    df['MOM'] = MOM(_close, timeperiod=20)
    df['PLUS_DI'] = PLUS_DI(_high, _low, _close, timeperiod=14)
    df['PLUS_DM'] = PLUS_DM(_high, _low, timeperiod=14)
    # df['PPO'] = PPO(_close, fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = ROC(_close, timeperiod=20)
    df['ROCP'] = ROCP(_close, timeperiod=20)
    df['ROCR'] = ROCR(_close, timeperiod=20)
    df['ROCR100'] = ROCR100(_close, timeperiod=20)
    df['RSI'] = RSI(_close, timeperiod=14)
    df['TRIX'] = TRIX(_close, timeperiod=20)
    df['ULTOSC'] = ULTOSC(_high, _low, _close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = WILLR(_high, _low, _close, timeperiod=14)
    # df['AROON_down'], df['AROON_up'] = AROON(_high, _low, timeperiod=14)
    _, df['MACDEXT_macdsignal'], _ = MACDEXT(_close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    _, df['MACDFIX_macdsignal'], _ = MACDFIX(_close, signalperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = STOCH(_high, _low, _close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    _, df['STOCHF_fastd'] = STOCHF(_high, _low, _close, fastk_period=5, fastd_period=3, fastd_matype=0)
    # df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = STOCHRSI(_close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    return df

def get_pattern_recognition(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    # df['CDL3INSIDE'] = CDL3INSIDE(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLDOJI'] = CDLDOJI(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLDOJISTAR'] = CDLDOJISTAR(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLDRAGONFLYDOJI'] = CDLDRAGONFLYDOJI(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLENGULFING'] = CDLENGULFING(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLEVENINGSTAR'] = CDLEVENINGSTAR(_open, _high, _low, _close, penetration=0).rolling(window=20, min_periods=1).max()
    # df['CDLGAPSIDESIDEWHITE'] = CDLGAPSIDESIDEWHITE(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLGRAVESTONEDOJI'] = CDLGRAVESTONEDOJI(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLHAMMER'] = CDLHAMMER(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLHANGINGMAN'] = CDLHANGINGMAN(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLINVERTEDHAMMER'] = CDLINVERTEDHAMMER(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLLONGLEGGEDDOJI'] = CDLLONGLEGGEDDOJI(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLMORNINGSTAR'] = CDLMORNINGSTAR(_open, _high, _low, _close, penetration=0).rolling(window=20, min_periods=1).max()
    # df['CDLRICKSHAWMAN'] = CDLRICKSHAWMAN(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDLSEPARATINGLINES'] = CDLSEPARATINGLINES(_open, _high, _low, _close).rolling(window=20, min_periods=1).max()
    # df['CDL3LINESTRIKE'] = CDL3LINESTRIKE(_open, _high, _low, _close)
    # df['CDL3OUTSIDE'] = CDL3OUTSIDE(_open, _high, _low, _close)
    # df['CDLADVANCEBLOCK'] = CDLADVANCEBLOCK(_open, _high, _low, _close)
    # df['CDLBELTHOLD'] = CDLBELTHOLD(_open, _high, _low, _close)
    # df['CDLCLOSINGMARUBOZU'] = CDLCLOSINGMARUBOZU(_open, _high, _low, _close)
    # df['CDLHARAMI'] = CDLHARAMI(_open, _high, _low, _close)
    # df['CDLHARAMICROSS'] = CDLHARAMICROSS(_open, _high, _low, _close)
    # df['CDLHIGHWAVE'] = CDLHIGHWAVE(_open, _high, _low, _close)
    # df['CDLHIKKAKE'] = CDLHIKKAKE(_open, _high, _low, _close)
    # df['CDLHIKKAKEMOD'] = CDLHIKKAKEMOD(_open, _high, _low, _close)
    # df['CDLLONGLINE'] = CDLLONGLINE(_open, _high, _low, _close)
    # df['CDLMARUBOZU'] = CDLMARUBOZU(_open, _high, _low, _close)
    # df['CDLMATCHINGLOW'] = CDLMATCHINGLOW(_open, _high, _low, _close)
    # df['CDLSHOOTINGSTAR'] = CDLSHOOTINGSTAR(_open, _high, _low, _close)
    # df['CDLSHORTLINE'] = CDLSHORTLINE(_open, _high, _low, _close)
    # df['CDLSPINNINGTOP'] = CDLSPINNINGTOP(_open, _high, _low, _close)
    # df['CDLTAKURI'] = CDLTAKURI(_open, _high, _low, _close)
    # df['CDLXSIDEGAP3METHODS'] = CDLXSIDEGAP3METHODS(_open, _high, _low, _close)
    # df['CDL2CROWS'] = CDL2CROWS(_open, _high, _low, _close)
    # df['CDL3BLACKCROWS'] = CDL3BLACKCROWS(_open, _high, _low, _close)
    # df['CDL3STARSINSOUTH'] = CDL3STARSINSOUTH(_open, _high, _low, _close)
    # df['CDL3WHITESOLDIERS'] = CDL3WHITESOLDIERS(_open, _high, _low, _close)
    # df['CDLABANDONEDBABY'] = CDLABANDONEDBABY(_open, _high, _low, _close, penetration=0)
    # df['CDLBREAKAWAY'] = CDLBREAKAWAY(_open, _high, _low, _close)
    # df['CDLCONCEALBABYSWALL'] = CDLCONCEALBABYSWALL(_open, _high, _low, _close)
    # df['CDLCOUNTERATTACK'] = CDLCOUNTERATTACK(_open, _high, _low, _close)
    # df['CDLDARKCLOUDCOVER'] = CDLDARKCLOUDCOVER(_open, _high, _low, _close, penetration=0)
    # df['CDLEVENINGDOJISTAR'] = CDLEVENINGDOJISTAR(_open, _high, _low, _close, penetration=0)
    # df['CDLHOMINGPIGEON'] = CDLHOMINGPIGEON(_open, _high, _low, _close)
    # df['CDLIDENTICAL3CROWS'] = CDLIDENTICAL3CROWS(_open, _high, _low, _close)
    # df['CDLINNECK'] = CDLINNECK(_open, _high, _low, _close)
    # df['CDLKICKING'] = CDLKICKING(_open, _high, _low, _close)
    # df['CDLKICKINGBYLENGTH'] = CDLKICKINGBYLENGTH(_open, _high, _low, _close)
    # df['CDLLADDERBOTTOM'] = CDLLADDERBOTTOM(_open, _high, _low, _close)
    # df['CDLMATHOLD'] = CDLMATHOLD(_open, _high, _low, _close, penetration=0)
    # df['CDLMORNINGDOJISTAR'] = CDLMORNINGDOJISTAR(_open, _high, _low, _close, penetration=0)
    # df['CDLONNECK'] = CDLONNECK(_open, _high, _low, _close)
    # df['CDLPIERCING'] = CDLPIERCING(_open, _high, _low, _close)
    # df['CDLRISEFALL3METHODS'] = CDLRISEFALL3METHODS(_open, _high, _low, _close)
    # df['CDLSTALLEDPATTERN'] = CDLSTALLEDPATTERN(_open, _high, _low, _close)
    # df['CDLSTICKSANDWICH'] = CDLSTICKSANDWICH(_open, _high, _low, _close)
    # df['CDLTASUKIGAP'] = CDLTASUKIGAP(_open, _high, _low, _close)
    # df['CDLTHRUSTING'] = CDLTHRUSTING(_open, _high, _low, _close)
    # df['CDLTRISTAR'] = CDLTRISTAR(_open, _high, _low, _close)
    # df['CDLUNIQUE3RIVER'] = CDLUNIQUE3RIVER(_open, _high, _low, _close)
    # df['CDLUPSIDEGAP2CROWS'] = CDLUPSIDEGAP2CROWS(_open, _high, _low, _close)
    return df


def get_price_transform(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]
    
    df['AVGPRICE'] = DIV(_close - AVGPRICE(_open, _high, _low, _close), _close)
    df['MEDPRICE'] = DIV(_close - MEDPRICE(_high, _low), _close)
    df['TYPPRICE'] = DIV(_close - TYPPRICE(_high, _low, _close), _close)
    df['WCLPRICE'] = DIV(_close - WCLPRICE(_high, _low, _close), _close)
    return df

def get_statistic(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    # df['BETA'] = BETA(_close, _high, timeperiod=5)
    # df['CORREL'] = CORREL(_close, _high, timeperiod=20)
    # df['LINEARREG'] = LINEARREG(_close, timeperiod=14)
    df['LINEARREG_ANGLE'] = LINEARREG_ANGLE(_close, timeperiod=14)
    # df['LINEARREG_INTERCEPT'] = LINEARREG_INTERCEPT(_close, timeperiod=14)
    df['LINEARREG_SLOPE'] = LINEARREG_SLOPE(_close, timeperiod=14)
    df['STDDEV'] = STDDEV(_close, timeperiod=5, nbdev=1)
    df['TSF'] = TSF(_close, timeperiod=14)

    # df['BETA'] = df['BETA'].pct_change()
    # df['CORREL'] = df['CORREL'].pct_change()
    # df['LINEARREG'] = df['LINEARREG'].pct_change()
    df['LINEARREG_ANGLE'] = df['LINEARREG_ANGLE'].diff().fillna(0)
    # df['LINEARREG_INTERCEPT'] = df['LINEARREG_INTERCEPT'].pct_change()
    # df['LINEARREG_SLOPE'] = df['LINEARREG_SLOPE'].diff().fillna(0)
    df['STDDEV'] = df['STDDEV'].pct_change()
    df['TSF'] = df['TSF'].pct_change()

    # df['VAR'] = VAR(_close, timeperiod=5, nbdev=1)
    return df

def get_volatility_indicator(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    df['ATR'] = ATR(_high, _low, _close, timeperiod=24)
    # high_vol_atr = (df['ATR'] > df['ATR'].rolling(60).quantile(0.8))
    # low_vol_atr = (df['ATR'] < df['ATR'].rolling(60).quantile(0.2))
    # df['VOL_REGIME'] = np.select([high_vol_atr, low_vol_atr],[2, 0], default=1)
    df['NATR'] = NATR(_high, _low, _close, timeperiod=24)
    # df['TRANGE'] = TRANGE(_high, _low, _close)
    return df


def get_volumeindicator(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    # df['AD'] = AD(_high, _low, _close, _volume)
    df['ADOSC'] = ADOSC(_high, _low, _close, _volume, fastperiod=3, slowperiod=10)
    # df['OBV'] = OBV(_close, _volume)
    return df

def get_overlap_studies(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    _open = df[open_col]
    _close = df[close_col]
    _low = df[low_col]
    _high = df[high_col]
    _volume = df[volume_col]

    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = BBANDS(_close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    # df['MAMA_mama'], df['MAMA_fama'] = MAMA(_close, fastlimit=0, slowlimit=0)
    df['DEMA'] = DEMA(_close, timeperiod=20)
    df['EMA'] = EMA(_close, timeperiod=20)
    df['HT_TRENDLINE'] = HT_TRENDLINE(_close)
    df['MA'] = MA(_close, timeperiod=20, matype=0)
    df['KAMA'] = KAMA(_close, timeperiod=20)
    # df['MAVP'] = MAVP(_close, 20, minperiod=2, maxperiod=20)
    df['MIDPOINT'] = MIDPOINT(_close, timeperiod=14)
    df['MIDPRICE'] = MIDPRICE(_high, _low, timeperiod=14)
    df['SAR'] = SAR(_high, _low, acceleration=0, maximum=0)
    df['SAREXT'] = SAREXT(_high, _low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    df['SMA'] = SMA(_close, timeperiod=20)
    df['T3'] = T3(_close, timeperiod=5, vfactor=0)
    df['TEMA'] = TEMA(_close, timeperiod=20)
    df['TRIMA'] = TRIMA(_close, timeperiod=20)
    df['WMA'] = WMA(_close, timeperiod=20)
    return df

def drop_ohlcv(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    return df.drop([close_col, open_col, low_col, high_col, volume_col], axis=1)


def get_technical_indicator_features(df, close_col = "Close", open_col="Open", high_col="High", low_col="Low", volume_col="Volume"):
    df = get_pattern_recognition(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    df = get_statistic(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    df = get_volumeindicator(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    df = get_momentum_indicator(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    df = get_volatility_indicator(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    # df = get_price_transform(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    # df = get_cycle_indicator(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    # df = get_overlap_studies(df, close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col, volume_col=volume_col)
    df = drop_ohlcv(df)
    return df