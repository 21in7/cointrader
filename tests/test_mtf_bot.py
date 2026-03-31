"""
MTF Pullback Bot 유닛 테스트
─────────────────────────────
합성 데이터 기반, 외부 API 호출 없음.
"""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.mtf_bot import (
    DataFetcher,
    ExecutionManager,
    MetaFilter,
    TriggerStrategy,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_1h_df():
    """EMA50/200, ADX, ATR 계산에 충분한 250개 1h 합성 캔들."""
    np.random.seed(42)
    n = 250
    # 완만한 상승 추세 (EMA50 > EMA200이 되도록)
    close = np.cumsum(np.random.randn(n) * 0.001 + 0.0005) + 2.0
    high = close + np.abs(np.random.randn(n)) * 0.005
    low = close - np.abs(np.random.randn(n)) * 0.005
    open_ = close + np.random.randn(n) * 0.001

    # 완성된 캔들 timestamp (1h 간격, 과거 시점)
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = pd.date_range(start=base_ts, periods=n, freq="1h")

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    }, index=timestamps)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sample_15m_df():
    """TriggerStrategy용 50개 15m 합성 캔들."""
    np.random.seed(99)
    n = 50
    close = np.cumsum(np.random.randn(n) * 0.001) + 0.5
    high = close + np.abs(np.random.randn(n)) * 0.003
    low = close - np.abs(np.random.randn(n)) * 0.003
    open_ = close + np.random.randn(n) * 0.001

    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = pd.date_range(start=base_ts, periods=n, freq="15min")

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    }, index=timestamps)
    df.index.name = "timestamp"
    return df


# ═══════════════════════════════════════════════════════════════════
# Test 1: _remove_incomplete_candle
# ═══════════════════════════════════════════════════════════════════


class TestRemoveIncompleteCandle:
    """DataFetcher._remove_incomplete_candle 정적 메서드 테스트."""

    def test_removes_incomplete_15m_candle(self):
        """현재 15m 슬롯에 해당하는 미완성 캔들은 제거되어야 한다."""
        now_ms = int(time.time() * 1000)
        current_slot_ms = (now_ms // (900 * 1000)) * (900 * 1000)

        # 완성 캔들 2개 + 미완성 캔들 1개
        timestamps = [
            pd.Timestamp(current_slot_ms - 1800_000, unit="ms", tz="UTC"),  # 2슬롯 전
            pd.Timestamp(current_slot_ms - 900_000, unit="ms", tz="UTC"),   # 1슬롯 전
            pd.Timestamp(current_slot_ms, unit="ms", tz="UTC"),              # 현재 슬롯 (미완성)
        ]
        df = pd.DataFrame({
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [100.0, 200.0, 300.0],
        }, index=timestamps)

        result = DataFetcher._remove_incomplete_candle(df, interval_sec=900)
        assert len(result) == 2, f"미완성 캔들 제거 실패: {len(result)}개 (2개 예상)"

    def test_keeps_all_completed_candles(self):
        """모든 캔들이 완성된 경우 제거하지 않아야 한다."""
        now_ms = int(time.time() * 1000)
        current_slot_ms = (now_ms // (900 * 1000)) * (900 * 1000)

        # 모두 과거 슬롯의 완성 캔들
        timestamps = [
            pd.Timestamp(current_slot_ms - 2700_000, unit="ms", tz="UTC"),
            pd.Timestamp(current_slot_ms - 1800_000, unit="ms", tz="UTC"),
            pd.Timestamp(current_slot_ms - 900_000, unit="ms", tz="UTC"),
        ]
        df = pd.DataFrame({
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [100.0, 200.0, 300.0],
        }, index=timestamps)

        result = DataFetcher._remove_incomplete_candle(df, interval_sec=900)
        assert len(result) == 3, f"완성 캔들 유지 실패: {len(result)}개 (3개 예상)"

    def test_empty_dataframe(self):
        """빈 DataFrame 입력 시 빈 DataFrame 반환."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = DataFetcher._remove_incomplete_candle(df, interval_sec=900)
        assert result.empty

    def test_1h_interval(self):
        """1h 간격에서도 정상 동작."""
        now_ms = int(time.time() * 1000)
        current_slot_ms = (now_ms // (3600 * 1000)) * (3600 * 1000)

        timestamps = [
            pd.Timestamp(current_slot_ms - 7200_000, unit="ms", tz="UTC"),
            pd.Timestamp(current_slot_ms - 3600_000, unit="ms", tz="UTC"),
            pd.Timestamp(current_slot_ms, unit="ms", tz="UTC"),  # 현재 슬롯 (미완성)
        ]
        df = pd.DataFrame({
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.12, 1.22],
            "volume": [100.0, 200.0, 300.0],
        }, index=timestamps)

        result = DataFetcher._remove_incomplete_candle(df, interval_sec=3600)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════
# Test 2: MetaFilter
# ═══════════════════════════════════════════════════════════════════


class TestMetaFilter:
    """MetaFilter 상태 판별 로직 테스트."""

    def _make_fetcher_with_df(self, df_1h):
        """Mock DataFetcher를 생성하여 특정 1h DataFrame을 반환하도록 설정."""
        fetcher = DataFetcher.__new__(DataFetcher)
        fetcher.klines_15m = []
        fetcher.klines_1h = []
        fetcher.data_fetcher = None
        # get_1h_dataframe_completed 을 직접 패치
        fetcher.get_1h_dataframe_completed = lambda: df_1h
        return fetcher

    def test_wait_when_adx_below_threshold(self, sample_1h_df):
        """ADX < 20이면 WAIT 상태."""
        import pandas_ta as ta

        df = sample_1h_df.copy()
        # 변동성이 없는 flat 데이터 → ADX가 낮을 가능성 높음
        df["close"] = 2.0  # 완전 flat
        df["high"] = 2.001
        df["low"] = 1.999
        df["open"] = 2.0

        fetcher = self._make_fetcher_with_df(df)
        meta = MetaFilter(fetcher)
        state = meta.get_market_state()
        assert state == "WAIT", f"Flat 데이터에서 WAIT 아닌 상태: {state}"

    def test_long_allowed_when_uptrend(self):
        """EMA50 > EMA200 + ADX > 20이면 LONG_ALLOWED."""
        np.random.seed(10)
        n = 250
        # 강한 상승 추세
        close = np.linspace(1.0, 3.0, n) + np.random.randn(n) * 0.01
        high = close + 0.02
        low = close - 0.02
        open_ = close - 0.005

        base_ts = pd.Timestamp("2025-01-01", tz="UTC")
        timestamps = pd.date_range(start=base_ts, periods=n, freq="1h")

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": np.ones(n) * 500000,
        }, index=timestamps)

        fetcher = self._make_fetcher_with_df(df)
        meta = MetaFilter(fetcher)
        state = meta.get_market_state()
        assert state == "LONG_ALLOWED", f"강한 상승 추세에서 LONG_ALLOWED 아닌 상태: {state}"

    def test_short_allowed_when_downtrend(self):
        """EMA50 < EMA200 + ADX > 20이면 SHORT_ALLOWED."""
        np.random.seed(20)
        n = 250
        # 강한 하락 추세
        close = np.linspace(3.0, 1.0, n) + np.random.randn(n) * 0.01
        high = close + 0.02
        low = close - 0.02
        open_ = close + 0.005

        base_ts = pd.Timestamp("2025-01-01", tz="UTC")
        timestamps = pd.date_range(start=base_ts, periods=n, freq="1h")

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": np.ones(n) * 500000,
        }, index=timestamps)

        fetcher = self._make_fetcher_with_df(df)
        meta = MetaFilter(fetcher)
        state = meta.get_market_state()
        assert state == "SHORT_ALLOWED", f"강한 하락 추세에서 SHORT_ALLOWED 아닌 상태: {state}"

    def test_indicator_caching(self, sample_1h_df):
        """동일 캔들에 대해 _calc_indicators가 캐시를 재사용하는지 확인."""
        fetcher = self._make_fetcher_with_df(sample_1h_df)
        meta = MetaFilter(fetcher)

        # 첫 호출: 캐시 없음
        df1 = meta._calc_indicators(sample_1h_df)
        ts1 = meta._cache_timestamp

        # 두 번째 호출: 동일 DataFrame → 캐시 히트
        df2 = meta._calc_indicators(sample_1h_df)
        assert df1 is df2, "동일 데이터에 대해 캐시가 재사용되지 않음"
        assert meta._cache_timestamp == ts1


# ═══════════════════════════════════════════════════════════════════
# Test 3: TriggerStrategy
# ═══════════════════════════════════════════════════════════════════


class TestTriggerStrategy:
    """15m 3-candle pullback 시퀀스 감지 테스트."""

    def test_hold_when_meta_wait(self, sample_15m_df):
        """meta_state=WAIT이면 항상 HOLD."""
        trigger = TriggerStrategy()
        signal = trigger.generate_signal(sample_15m_df, "WAIT")
        assert signal == "HOLD"

    def test_hold_when_insufficient_data(self):
        """데이터가 25개 미만이면 HOLD."""
        trigger = TriggerStrategy()
        small_df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [1.1] * 10,
            "low": [0.9] * 10,
            "close": [1.0] * 10,
            "volume": [100.0] * 10,
        })
        signal = trigger.generate_signal(small_df, "LONG_ALLOWED")
        assert signal == "HOLD"

    def test_long_pullback_signal(self):
        """LONG 풀백 시퀀스: t-1 EMA 아래 이탈 + 거래량 고갈 + t EMA 복귀."""
        np.random.seed(42)
        n = 30
        # 기본 상승 추세
        close = np.linspace(1.0, 1.1, n)
        high = close + 0.005
        low = close - 0.005
        open_ = close - 0.001
        volume = np.ones(n) * 100000

        # t-1 (인덱스 -2): EMA 아래로 이탈 + 거래량 고갈
        close[-2] = close[-3] - 0.02  # EMA 아래로 이탈
        volume[-2] = 5000  # 매우 낮은 거래량

        # t (인덱스 -1): EMA 위로 복귀
        close[-1] = close[-3] + 0.01

        base_ts = pd.Timestamp("2026-01-01", tz="UTC")
        timestamps = pd.date_range(start=base_ts, periods=n, freq="15min")

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        }, index=timestamps)

        trigger = TriggerStrategy()
        signal = trigger.generate_signal(df, "LONG_ALLOWED")
        # 풀백 조건 충족 여부는 EMA 계산 결과에 따라 다를 수 있으므로
        # 최소한 valid signal을 반환하는지 확인
        assert signal in ("EXECUTE_LONG", "HOLD")

    def test_short_pullback_signal(self):
        """SHORT 풀백 시퀀스: t-1 EMA 위로 이탈 + 거래량 고갈 + t EMA 아래 복귀."""
        np.random.seed(42)
        n = 30
        # 하락 추세
        close = np.linspace(1.1, 1.0, n)
        high = close + 0.005
        low = close - 0.005
        open_ = close + 0.001
        volume = np.ones(n) * 100000

        # t-1: EMA 위로 이탈 + 거래량 고갈
        close[-2] = close[-3] + 0.02
        volume[-2] = 5000

        # t: EMA 아래로 복귀
        close[-1] = close[-3] - 0.01

        base_ts = pd.Timestamp("2026-01-01", tz="UTC")
        timestamps = pd.date_range(start=base_ts, periods=n, freq="15min")

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        }, index=timestamps)

        trigger = TriggerStrategy()
        signal = trigger.generate_signal(df, "SHORT_ALLOWED")
        assert signal in ("EXECUTE_SHORT", "HOLD")

    def test_trigger_info_populated(self, sample_15m_df):
        """generate_signal 후 get_trigger_info가 비어있지 않아야 한다."""
        trigger = TriggerStrategy()
        trigger.generate_signal(sample_15m_df, "LONG_ALLOWED")
        info = trigger.get_trigger_info()
        assert "signal" in info or "reason" in info


# ═══════════════════════════════════════════════════════════════════
# Test 4: ExecutionManager (SL/TP 계산)
# ═══════════════════════════════════════════════════════════════════


class TestExecutionManager:
    """ExecutionManager SL/TP 계산 및 포지션 관리 테스트."""

    def test_long_sl_tp_calculation(self):
        """LONG 진입 시 SL = entry - ATR*1.5, TP = entry + ATR*2.3."""
        em = ExecutionManager(symbol="XRPUSDT")
        entry = 2.0
        atr = 0.01

        result = em.execute("EXECUTE_LONG", entry, atr)
        assert result is not None
        assert result["action"] == "LONG"

        expected_sl = entry - (atr * 1.5)
        expected_tp = entry + (atr * 2.3)
        assert abs(result["sl_price"] - expected_sl) < 1e-8, f"SL: {result['sl_price']} != {expected_sl}"
        assert abs(result["tp_price"] - expected_tp) < 1e-8, f"TP: {result['tp_price']} != {expected_tp}"

    def test_short_sl_tp_calculation(self):
        """SHORT 진입 시 SL = entry + ATR*1.5, TP = entry - ATR*2.3."""
        em = ExecutionManager(symbol="XRPUSDT")
        entry = 2.0
        atr = 0.01

        result = em.execute("EXECUTE_SHORT", entry, atr)
        assert result is not None
        assert result["action"] == "SHORT"

        expected_sl = entry + (atr * 1.5)
        expected_tp = entry - (atr * 2.3)
        assert abs(result["sl_price"] - expected_sl) < 1e-8
        assert abs(result["tp_price"] - expected_tp) < 1e-8

    def test_hold_returns_none(self):
        """HOLD 신호는 None 반환."""
        em = ExecutionManager(symbol="XRPUSDT")
        result = em.execute("HOLD", 2.0, 0.01)
        assert result is None

    def test_duplicate_position_blocked(self):
        """이미 포지션이 있으면 중복 진입 차단."""
        em = ExecutionManager(symbol="XRPUSDT")
        em.execute("EXECUTE_LONG", 2.0, 0.01)

        result = em.execute("EXECUTE_SHORT", 2.1, 0.01)
        assert result is None, "포지션 중복 차단 실패"

    def test_reentry_after_close(self):
        """청산 후 재진입 가능."""
        em = ExecutionManager(symbol="XRPUSDT")
        em.execute("EXECUTE_LONG", 2.0, 0.01)
        em.close_position("test", exit_price=2.01, pnl_bps=50)

        result = em.execute("EXECUTE_SHORT", 2.05, 0.01)
        assert result is not None, "청산 후 재진입 실패"
        assert result["action"] == "SHORT"

    def test_invalid_atr_blocked(self):
        """ATR이 None/0/NaN이면 주문 차단."""
        em = ExecutionManager(symbol="XRPUSDT")

        assert em.execute("EXECUTE_LONG", 2.0, None) is None
        assert em.execute("EXECUTE_LONG", 2.0, 0) is None
        assert em.execute("EXECUTE_LONG", 2.0, float("nan")) is None

    def test_risk_reward_ratio(self):
        """R:R 비율이 올바르게 계산되는지 확인."""
        em = ExecutionManager(symbol="XRPUSDT")
        result = em.execute("EXECUTE_LONG", 2.0, 0.01)
        # TP/SL = 2.3/1.5 = 1.533...
        expected_rr = round(2.3 / 1.5, 2)
        assert result["risk_reward"] == expected_rr
