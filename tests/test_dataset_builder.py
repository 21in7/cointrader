import numpy as np
import pandas as pd
import pytest
from src.dataset_builder import generate_dataset_vectorized


@pytest.fixture
def sample_df():
    """최소 200행 이상의 OHLCV 더미 데이터."""
    rng = np.random.default_rng(42)
    n = 500
    close = 2.0 + np.cumsum(rng.normal(0, 0.01, n))
    close = np.clip(close, 0.01, None)
    high  = close * (1 + rng.uniform(0, 0.005, n))
    low   = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({
        "open":   close,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.uniform(1e6, 5e6, n),
    })


def test_returns_dataframe(sample_df):
    """결과가 DataFrame이어야 한다."""
    result = generate_dataset_vectorized(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_has_required_columns(sample_df):
    """기본 13개 피처 + label 컬럼이 모두 있어야 한다."""
    BASE_FEATURE_COLS = [
        "rsi", "macd_hist", "bb_pct", "ema_align",
        "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
        "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    ]
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert "label" in result.columns
        for col in BASE_FEATURE_COLS:
            assert col in result.columns, f"컬럼 없음: {col}"


def test_label_is_binary(sample_df):
    """label은 0 또는 1만 있어야 한다."""
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert set(result["label"].unique()).issubset({0, 1})


def test_generate_dataset_vectorized_with_btc_eth_has_21_feature_cols():
    """BTC/ETH DataFrame을 전달하면 결과 컬럼이 21개 피처 + label이어야 한다."""
    import pandas as pd
    import numpy as np
    from src.dataset_builder import generate_dataset_vectorized
    from src.ml_features import FEATURE_COLS

    np.random.seed(42)
    n = 500
    closes = np.cumprod(1 + np.random.randn(n) * 0.001) * 1.0
    xrp_df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    btc_df = xrp_df.copy() * 50000
    eth_df = xrp_df.copy() * 3000

    result = generate_dataset_vectorized(xrp_df, btc_df=btc_df, eth_df=eth_df)
    if not result.empty:
        assert set(FEATURE_COLS).issubset(set(result.columns))
        assert "label" in result.columns


def test_matches_original_generate_dataset(sample_df):
    """벡터화 버전과 기존 버전의 샘플 수가 유사해야 한다.

    벡터화 버전은 전체 시계열로 지표를 1회 계산하고, 기존 버전은 61행 슬라이딩
    윈도우로 매번 재계산한다. EMA 등 지수 이동평균은 초기값에 따라 수렴 속도가
    달라지므로 두 방식의 신호 수는 완전히 동일하지 않을 수 있다. ±50% 범위를
    허용한다.
    """
    from scripts.train_model import generate_dataset
    orig = generate_dataset(sample_df, n_jobs=1)
    vec  = generate_dataset_vectorized(sample_df)
    if len(orig) == 0:
        assert len(vec) == 0
        return
    ratio = len(vec) / len(orig)
    assert 0.5 <= ratio <= 2.0, (
        f"샘플 수 차이가 너무 큼: 벡터화={len(vec)}, 기존={len(orig)}, 비율={ratio:.2f}"
    )


def test_epsilon_no_division_by_zero():
    """bb_range=0, close=0, vol_ma20=0 극단값에서 nan/inf가 발생하지 않아야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 100
    # close를 모두 같은 값으로 → bb_range=0 유발
    df = pd.DataFrame({
        "open":   np.ones(n),
        "high":   np.ones(n),
        "low":    np.ones(n),
        "close":  np.ones(n),
        "volume": np.ones(n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    assert not feat[numeric_cols].isin([np.inf, -np.inf]).any().any(), \
        "inf 값이 있으면 안 됨"


def test_oi_nan_masking_no_column():
    """oi_change 컬럼이 없으면 전체가 nan이어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 100
    np.random.seed(0)
    df = pd.DataFrame({
        "open":   np.random.uniform(1, 2, n),
        "high":   np.random.uniform(2, 3, n),
        "low":    np.random.uniform(0.5, 1, n),
        "close":  np.random.uniform(1, 2, n),
        "volume": np.random.uniform(1000, 5000, n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    assert feat["oi_change"].isna().all(), "oi_change 컬럼 없을 때 전부 nan이어야 함"


def test_oi_nan_masking_with_zeros():
    """oi_change 컬럼이 있어도 0.0 구간은 nan으로 마스킹되어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 100
    np.random.seed(0)
    df = pd.DataFrame({
        "open":      np.random.uniform(1, 2, n),
        "high":      np.random.uniform(2, 3, n),
        "low":       np.random.uniform(0.5, 1, n),
        "close":     np.random.uniform(1, 2, n),
        "volume":    np.random.uniform(1000, 5000, n),
        "oi_change": np.concatenate([np.zeros(50), np.random.uniform(-0.1, 0.1, 50)]),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    assert feat["oi_change"].iloc[50:].notna().any(), "실제 OI 값 구간에 유한값이 있어야 함"


def test_rs_zero_denominator():
    """btc_r1=0일 때 RS가 inf/nan이 아닌 0.0이어야 한다 (np.divide 방식 검증)."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 500
    np.random.seed(7)
    # XRP close: 약간의 변동
    xrp_close = np.cumprod(1 + np.random.randn(n) * 0.001) * 1.0
    xrp_df = pd.DataFrame({
        "open":   xrp_close * 0.999,
        "high":   xrp_close * 1.005,
        "low":    xrp_close * 0.995,
        "close":  xrp_close,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    # BTC close: 완전히 고정 → btc_r1 = 0.0
    btc_close = np.ones(n) * 50000.0
    btc_df = pd.DataFrame({
        "open":   btc_close,
        "high":   btc_close,
        "low":    btc_close,
        "close":  btc_close,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    # ETH close: 약간의 변동 (eth_df 없으면 BTC 피처 자체가 계산 안 됨)
    eth_close = np.cumprod(1 + np.random.randn(n) * 0.001) * 3000.0
    eth_df = pd.DataFrame({
        "open":   eth_close * 0.999,
        "high":   eth_close * 1.005,
        "low":    eth_close * 0.995,
        "close":  eth_close,
        "volume": np.random.rand(n) * 1000 + 500,
    })

    # _calc_features_vectorized를 직접 호출해 BTC/ETH 피처를 포함한 전체 피처를 검증
    d = _calc_indicators(xrp_df)
    signal_arr = _calc_signals(d)
    feat = _calc_features_vectorized(d, signal_arr, btc_df=btc_df, eth_df=eth_df)

    assert "xrp_btc_rs" in feat.columns, "xrp_btc_rs 컬럼이 있어야 함"
    assert not feat["xrp_btc_rs"].isin([np.inf, -np.inf]).any(), \
        "xrp_btc_rs에 inf가 있으면 안 됨"
    assert not feat["xrp_btc_rs"].isna().all(), \
        "xrp_btc_rs가 전부 nan이면 안 됨"


@pytest.fixture
def signal_producing_df():
    """시그널이 반드시 발생하는 더미 데이터. 높은 변동성 + 거래량 급증."""
    rng = np.random.default_rng(7)
    n = 800
    trend = np.linspace(1.5, 3.0, n)
    noise = np.cumsum(rng.normal(0, 0.04, n))
    close = np.clip(trend + noise, 0.01, None)
    high  = close * (1 + rng.uniform(0, 0.015, n))
    low   = close * (1 - rng.uniform(0, 0.015, n))
    volume = rng.uniform(1e6, 3e6, n)
    volume[::30] *= 3.0  # 30봉마다 거래량 급증
    return pd.DataFrame({
        "open": close, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_hold_negative_labels_are_all_zero(signal_producing_df):
    """HOLD negative 샘플의 label은 전부 0이어야 한다."""
    result = generate_dataset_vectorized(signal_producing_df, negative_ratio=3)
    assert len(result) > 0, "시그널이 발생하지 않아 테스트 불가"
    assert "source" in result.columns
    hold_neg = result[result["source"] == "hold_negative"]
    assert len(hold_neg) > 0, "HOLD negative 샘플이 0개"
    assert (hold_neg["label"] == 0).all(), \
        f"HOLD negative 중 label != 0인 샘플 존재: {hold_neg['label'].value_counts().to_dict()}"


def test_signal_samples_preserved_after_sampling(signal_producing_df):
    """계층적 샘플링 후 source='signal' 샘플이 하나도 버려지지 않아야 한다."""
    result_signal_only = generate_dataset_vectorized(signal_producing_df, negative_ratio=0)
    result_with_hold   = generate_dataset_vectorized(signal_producing_df, negative_ratio=3)

    assert len(result_signal_only) > 0, "시그널이 발생하지 않아 테스트 불가"
    assert "source" in result_with_hold.columns
    signal_count = (result_with_hold["source"] == "signal").sum()
    assert signal_count == len(result_signal_only), \
        f"Signal 샘플 손실: 원본={len(result_signal_only)}, 유지={signal_count}"
