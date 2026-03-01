import pandas as pd
import numpy as np
import pytest
from src.ml_features import build_features, FEATURE_COLS


def make_df(n=100):
    """테스트용 최소 DataFrame 생성"""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.uniform(1000, 5000, n),
    })
    return df


def test_build_features_returns_series():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    assert isinstance(features, pd.Series)


def test_build_features_has_all_cols():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    for col in FEATURE_COLS:
        assert col in features.index, f"피처 누락: {col}"


def test_build_features_no_nan():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    assert not features.isna().any(), f"NaN 존재: {features[features.isna()]}"


def test_side_encoding():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    long_feat  = build_features(df_ind, signal="LONG")
    short_feat = build_features(df_ind, signal="SHORT")
    assert long_feat["side"] == 1
    assert short_feat["side"] == 0
