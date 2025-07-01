from data_preprocessing.base_features import create_ohlcv_features
from data_preprocessing.context import load_context
from data_preprocessing.patterns import get_pattern_recognition_features, get_pattern_recognition_features_new
from data_preprocessing.strategies import apply_strategies
from data_preprocessing.technical_features import drop_ohlcv
from data_preprocessing.volatility import get_volatility_features


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






