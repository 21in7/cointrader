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


def test_adx_filter_blocks_low_adx(sample_df):
    """ADX < adx_threshold이면 HOLD 반환."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    df.loc[df.index[-1], "rsi"] = 20
    df.loc[df.index[-2], "macd"] = -1
    df.loc[df.index[-2], "macd_signal"] = 0
    df.loc[df.index[-1], "macd"] = 1
    df.loc[df.index[-1], "macd_signal"] = 0
    df.loc[df.index[-1], "volume"] = df.loc[df.index[-1], "vol_ma20"] * 3
    df["adx"] = 15.0
    # 기본 adx_threshold=25이므로 ADX=15은 HOLD
    signal, detail = ind.get_signal(df)
    assert signal == "HOLD"
    assert "ADX" in detail["hold_reason"]
    # adx_threshold=0이면 ADX 필터 비활성화 → LONG
    signal, detail = ind.get_signal(df, adx_threshold=0)
    assert signal == "LONG"


def test_adx_nan_falls_through(sample_df):
    """ADX가 NaN(초기 캔들)이면 기존 가중치 로직으로 폴백해야 한다."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    df["adx"] = float("nan")
    signal, detail = ind.get_signal(df)
    # NaN이면 차단하지 않고 기존 로직 실행 → LONG/SHORT/HOLD 중 하나
    assert signal in ("LONG", "SHORT", "HOLD")
    assert detail["adx"] is None


def test_signal_returns_direction(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    signal, detail = ind.get_signal(df)
    assert signal in ("LONG", "SHORT", "HOLD")
    assert "long" in detail and "short" in detail
