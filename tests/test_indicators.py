import pandas as pd
import numpy as np
import pytest
from src.indicators import Indicators


@pytest.fixture
def sample_df():
    """100개 캔들 샘플 데이터"""
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 0.5
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.001),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    })
    return df


def test_rsi_range(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "rsi" in df.columns
    valid = df["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_columns(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_bollinger_bands(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns
    valid = df.dropna()
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_adx_column_exists(sample_df):
    """calculate_all()이 adx 컬럼을 생성하는지 확인."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "adx" in df.columns
    valid = df["adx"].dropna()
    assert (valid >= 0).all()


def test_adx_low_does_not_block_signal(sample_df):
    """ADX < 25여도 시그널이 차단되지 않는다 (ML에 위임)."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    # 강한 LONG 신호가 나오도록 지표 조작
    df.loc[df.index[-1], "rsi"] = 20
    df.loc[df.index[-2], "macd"] = -1
    df.loc[df.index[-2], "macd_signal"] = 0
    df.loc[df.index[-1], "macd"] = 1
    df.loc[df.index[-1], "macd_signal"] = 0
    df.loc[df.index[-1], "volume"] = df.loc[df.index[-1], "vol_ma20"] * 2
    df["adx"] = 15.0
    signal = ind.get_signal(df)
    # ADX 낮아도 지표 조건 충족 시 LONG 반환 (ML이 최종 판단)
    assert signal == "LONG"


def test_adx_nan_falls_through(sample_df):
    """ADX가 NaN(초기 캔들)이면 기존 가중치 로직으로 폴백해야 한다."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    df["adx"] = float("nan")
    signal = ind.get_signal(df)
    # NaN이면 차단하지 않고 기존 로직 실행 → LONG/SHORT/HOLD 중 하나
    assert signal in ("LONG", "SHORT", "HOLD")


def test_signal_returns_direction(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    signal = ind.get_signal(df)
    assert signal in ("LONG", "SHORT", "HOLD")
