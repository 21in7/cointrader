import pandas as pd
import numpy as np
import pytest
from src.ml_features import build_features, FEATURE_COLS


def _make_df(n=10, base_price=1.0):
    """테스트용 더미 캔들 DataFrame 생성."""
    closes = [base_price * (1 + i * 0.001) for i in range(n)]
    return pd.DataFrame({
        "close": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "volume": [1000.0] * n,
        "rsi": [50.0] * n, "macd": [0.0] * n, "macd_signal": [0.0] * n,
        "macd_hist": [0.0] * n, "bb_upper": [c * 1.02 for c in closes],
        "bb_lower": [c * 0.98 for c in closes], "ema9": closes,
        "ema21": closes, "ema50": closes, "atr": [0.01] * n,
        "stoch_k": [50.0] * n, "stoch_d": [50.0] * n,
        "vol_ma20": [1000.0] * n,
        "adx": [20.0] * n,
    })


def test_build_features_with_btc_eth_has_26_features():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    assert len(features) == 26

def test_build_features_without_btc_eth_has_18_features():
    xrp_df = _make_df(10, base_price=1.0)
    features = build_features(xrp_df, "LONG")
    assert len(features) == 18

def test_build_features_btc_ret_1_correct():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    btc_closes = btc_df["close"]
    expected_btc_ret_1 = (btc_closes.iloc[-1] - btc_closes.iloc[-2]) / btc_closes.iloc[-2]
    assert abs(features["btc_ret_1"] - expected_btc_ret_1) < 1e-6

def test_build_features_rs_zero_when_btc_ret_zero():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    btc_df["close"] = 50000.0  # 모든 캔들 동일
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    assert features["primary_btc_rs"] == 0.0

def test_feature_cols_has_24_items():
    """Legacy test — updated to 26 after OI derived features added."""
    from src.ml_features import FEATURE_COLS
    assert len(FEATURE_COLS) == 26


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


BASE_FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
]

def test_build_features_has_all_cols():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    for col in BASE_FEATURE_COLS:
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


@pytest.fixture
def sample_df_with_indicators():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    return ind.calculate_all()


def test_build_features_uses_provided_oi_funding(sample_df_with_indicators):
    """oi_change, funding_rate 파라미터가 제공되면 실제 값이 피처에 반영된다."""
    feat = build_features(
        sample_df_with_indicators,
        signal="LONG",
        oi_change=0.05,
        funding_rate=0.0002,
    )
    assert feat["oi_change"] == pytest.approx(0.05)
    assert feat["funding_rate"] == pytest.approx(0.0002)


def test_build_features_defaults_to_zero_when_not_provided(sample_df_with_indicators):
    """oi_change, funding_rate 파라미터 미제공 시 0.0으로 채워진다."""
    feat = build_features(sample_df_with_indicators, signal="LONG")
    assert feat["oi_change"] == pytest.approx(0.0)
    assert feat["funding_rate"] == pytest.approx(0.0)


def test_feature_cols_has_26_items():
    from src.ml_features import FEATURE_COLS
    assert len(FEATURE_COLS) == 26


def test_build_features_with_oi_derived_params():
    """oi_change_ma5, oi_price_spread 파라미터가 피처에 반영된다."""
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(
        xrp_df, "LONG",
        btc_df=btc_df, eth_df=eth_df,
        oi_change=0.05, funding_rate=0.0002,
        oi_change_ma5=0.03, oi_price_spread=0.12,
    )
    assert features["oi_change_ma5"] == pytest.approx(0.03)
    assert features["oi_price_spread"] == pytest.approx(0.12)


def test_build_features_oi_derived_defaults_to_zero():
    """oi_change_ma5, oi_price_spread 미제공 시 0.0으로 채워진다."""
    xrp_df = _make_df(10, base_price=1.0)
    features = build_features(xrp_df, "LONG")
    assert features["oi_change_ma5"] == pytest.approx(0.0)
    assert features["oi_price_spread"] == pytest.approx(0.0)
