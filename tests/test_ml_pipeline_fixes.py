import numpy as np
import pandas as pd
import pytest
from src.dataset_builder import generate_dataset_vectorized, _calc_labels_vectorized


@pytest.fixture
def signal_df():
    """시그널이 발생하는 데이터."""
    rng = np.random.default_rng(7)
    n = 800
    trend = np.linspace(1.5, 3.0, n)
    noise = np.cumsum(rng.normal(0, 0.04, n))
    close = np.clip(trend + noise, 0.01, None)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    volume = rng.uniform(1e6, 3e6, n)
    volume[::30] *= 3.0
    return pd.DataFrame({
        "open": close, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_sltp_params_are_passed_through(signal_df):
    """SL/TP 배수가 generate_dataset_vectorized에 전달되어야 한다."""
    # 파라미터가 수용되는지(TypeError 없이) 확인하는 것이 핵심
    r1 = generate_dataset_vectorized(
        signal_df, atr_sl_mult=1.5, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    r2 = generate_dataset_vectorized(
        signal_df, atr_sl_mult=2.0, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    # 두 결과 모두 DataFrame이어야 한다
    assert isinstance(r1, pd.DataFrame)
    assert isinstance(r2, pd.DataFrame)
    # 신호가 충분히 많을 경우, 다른 SL 배수는 레이블 분포에 영향을 줄 수 있다
    if len(r1) > 10 and len(r2) > 10:
        assert not (r1["label"].values == r2["label"].values).all() or len(r1) != len(r2), \
            "SL 배수가 다르면 레이블이 달라져야 한다"


def test_default_sltp_backward_compatible(signal_df):
    """SL/TP 파라미터 미지정 시 기존 기본값(1.5, 2.0)으로 동작해야 한다."""
    r_default = generate_dataset_vectorized(
        signal_df, adx_threshold=0, volume_multiplier=1.5,
    )
    r_explicit = generate_dataset_vectorized(
        signal_df, atr_sl_mult=1.5, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    if len(r_default) > 0:
        assert len(r_default) == len(r_explicit)
        assert (r_default["label"].values == r_explicit["label"].values).all()
