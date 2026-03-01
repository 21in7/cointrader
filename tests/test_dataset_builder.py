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
    """FEATURE_COLS + label 컬럼이 모두 있어야 한다."""
    from src.ml_features import FEATURE_COLS
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert "label" in result.columns
        for col in FEATURE_COLS:
            assert col in result.columns, f"컬럼 없음: {col}"


def test_label_is_binary(sample_df):
    """label은 0 또는 1만 있어야 한다."""
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert set(result["label"].unique()).issubset({0, 1})


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
