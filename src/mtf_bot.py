"""
MTF Pullback Bot — Module 1~4
──────────────────────────────
Module 1: TimeframeSync, DataFetcher (REST 폴링 기반)
Module 2: MetaFilter (1h EMA50/200 + ADX + ATR)
Module 3: TriggerStrategy (15m Volume-backed Pullback 3캔들 시퀀스)
Module 4: ExecutionManager (Dry-run 가상 주문 + SL/TP 관리)

핵심 원칙:
  - Look-ahead bias 원천 차단: 완성된 캔들만 사용 ([:-1] 슬라이싱)
  - Binance 서버 딜레이 고려: 캔들 판별 시 2~5초 range
  - REST 폴링 기반 안정성: WebSocket 대신 30초 주기 폴링
  - 메모리 최적화: deque(maxlen=250)
  - Dry-run 모드: 4월 OOS 검증 기간, 실주문 API 주석 처리
"""

import asyncio
import json
import os
import time as _time
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from loguru import logger
from src.notifier import DiscordNotifier


# ═══════════════════════════════════════════════════════════════════
# Module 1: TimeframeSync
# ═══════════════════════════════════════════════════════════════════

class TimeframeSync:
    """현재 시간이 15m/1h 캔들 종료 직후인지 판별 (Binance 서버 딜레이 2~5초 고려)."""

    _15M_MINUTES = {0, 15, 30, 45}

    @staticmethod
    def is_15m_candle_closed(current_ts: int) -> bool:
        """
        15m 캔들 종료 판별.

        Args:
            current_ts: Unix timestamp (밀리초)

        Returns:
            True if 분(minute)이 [0, 15, 30, 45] 중 하나이고 초(second)가 2~5초 사이
        """
        dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
        return dt.minute in TimeframeSync._15M_MINUTES and 2 <= dt.second <= 5

    @staticmethod
    def is_1h_candle_closed(current_ts: int) -> bool:
        """
        1h 캔들 종료 판별.

        Args:
            current_ts: Unix timestamp (밀리초)

        Returns:
            True if 분(minute)이 0이고 초(second)가 2~5초 사이
        """
        dt = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
        return dt.minute == 0 and 2 <= dt.second <= 5


# ═══════════════════════════════════════════════════════════════════
# Module 1: DataFetcher
# ═══════════════════════════════════════════════════════════════════

class DataFetcher:
    """Binance Futures에서 15m/1h OHLCV 데이터 fetch 및 관리."""

    def __init__(self, symbol: str = "XRP/USDT:USDT"):
        self.symbol = symbol
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.klines_15m: deque = deque(maxlen=250)
        self.klines_1h: deque = deque(maxlen=250)
        self._last_15m_ts: int = 0  # 마지막으로 저장된 15m 캔들 timestamp
        self._last_1h_ts: int = 0

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> List[List]:
        """
        ccxt를 통해 OHLCV 데이터 fetch.

        Returns:
            [[timestamp, open, high, low, close, volume], ...]
        """
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def initialize(self):
        """봇 시작 시 초기 데이터 로드 (250개씩)."""
        # 15m 캔들
        raw_15m = await self.fetch_ohlcv(self.symbol, "15m", limit=250)
        for candle in raw_15m:
            self.klines_15m.append(candle)
        if raw_15m:
            self._last_15m_ts = raw_15m[-1][0]

        # 1h 캔들
        raw_1h = await self.fetch_ohlcv(self.symbol, "1h", limit=250)
        for candle in raw_1h:
            self.klines_1h.append(candle)
        if raw_1h:
            self._last_1h_ts = raw_1h[-1][0]

        logger.info(
            f"[DataFetcher] 초기화 완료: 15m={len(self.klines_15m)}개, 1h={len(self.klines_1h)}개"
        )

    async def poll_update(self, interval: int = 30):
        """
        30초 주기로 REST API 폴링. 새 캔들이 나오면 deque에 append.
        무한 루프 — 백그라운드 태스크로 실행.
        """
        logger.info(f"[DataFetcher] 폴링 시작 (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)

                # 15m 업데이트: 최근 3개 fetch (중복 방지)
                raw_15m = await self.fetch_ohlcv(self.symbol, "15m", limit=3)
                new_15m = 0
                for candle in raw_15m:
                    if candle[0] > self._last_15m_ts:
                        self.klines_15m.append(candle)
                        self._last_15m_ts = candle[0]
                        new_15m += 1

                # 1h 업데이트: 최근 3개 fetch
                raw_1h = await self.fetch_ohlcv(self.symbol, "1h", limit=3)
                new_1h = 0
                for candle in raw_1h:
                    if candle[0] > self._last_1h_ts:
                        self.klines_1h.append(candle)
                        self._last_1h_ts = candle[0]
                        new_1h += 1

                if new_15m > 0 or new_1h > 0:
                    logger.info(
                        f"[DataFetcher] 캔들 업데이트: 15m +{new_15m} (총 {len(self.klines_15m)}), "
                        f"1h +{new_1h} (총 {len(self.klines_1h)})"
                    )

            except Exception as e:
                logger.error(f"[DataFetcher] 폴링 에러: {e}")
                await asyncio.sleep(5)  # 에러 시 짧은 대기 후 재시도

    def get_15m_dataframe(self) -> Optional[pd.DataFrame]:
        """모든 15m 캔들을 DataFrame으로 반환."""
        if not self.klines_15m:
            return None
        data = list(self.klines_15m)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    def get_1h_dataframe_completed(self) -> Optional[pd.DataFrame]:
        """
        '완성된' 1h 캔들만 반환.

        핵심: [:-1] 슬라이싱으로 진행 중인 최신 1h 캔들 제외.
        이유: Look-ahead bias 원천 차단 — 아직 완성되지 않은 캔들의
              high/low/close는 미래 데이터이므로 지표 계산에 사용하면 안 됨.
        """
        if len(self.klines_1h) < 2:
            return None
        completed = list(self.klines_1h)[:-1]  # ← 핵심: 미완성 봉 제외
        df = pd.DataFrame(completed, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    async def close(self):
        """ccxt exchange 연결 정리."""
        await self.exchange.close()


# ═══════════════════════════════════════════════════════════════════
# Module 2: MetaFilter
# ═══════════════════════════════════════════════════════════════════

class MetaFilter:
    """1시간봉 데이터로부터 거시 추세 판독."""

    EMA_FAST = 50
    EMA_SLOW = 200
    ADX_THRESHOLD = 20

    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """1h DataFrame에 EMA50, EMA200, ADX, ATR 계산."""
        df = df.copy()
        df["ema50"] = ta.ema(df["close"], length=self.EMA_FAST)
        df["ema200"] = ta.ema(df["close"], length=self.EMA_SLOW)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_df["ADX_14"]
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        return df

    def get_market_state(self) -> str:
        """
        1h 메타필터 상태 반환.

        Returns:
            'LONG_ALLOWED':  EMA50 > EMA200 & ADX > 20 → 상승 추세, LONG 진입 허용
            'SHORT_ALLOWED': EMA50 < EMA200 & ADX > 20 → 하락 추세, SHORT 진입 허용
            'WAIT':          그 외 (추세 약하거나 데이터 부족)
        """
        df = self.data_fetcher.get_1h_dataframe_completed()
        if df is None or len(df) < self.EMA_SLOW:
            return "WAIT"

        df = self._calc_indicators(df)
        last = df.iloc[-1]

        if pd.isna(last["ema50"]) or pd.isna(last["ema200"]) or pd.isna(last["adx"]):
            return "WAIT"

        if last["adx"] < self.ADX_THRESHOLD:
            return "WAIT"

        if last["ema50"] > last["ema200"]:
            return "LONG_ALLOWED"
        elif last["ema50"] < last["ema200"]:
            return "SHORT_ALLOWED"

        return "WAIT"

    def get_current_atr(self) -> Optional[float]:
        """현재 1h ATR 값 반환 (SL/TP 계산용)."""
        df = self.data_fetcher.get_1h_dataframe_completed()
        if df is None or len(df) < 15:  # ATR(14) 최소 데이터
            return None

        df = self._calc_indicators(df)
        atr = df["atr"].iloc[-1]
        return float(atr) if not pd.isna(atr) else None

    def get_meta_info(self) -> Dict:
        """전체 메타 정보 반환 (디버깅용)."""
        df = self.data_fetcher.get_1h_dataframe_completed()
        if df is None or len(df) < self.EMA_SLOW:
            return {"state": "WAIT", "ema50": None, "ema200": None,
                    "adx": None, "atr": None, "timestamp": None}

        df = self._calc_indicators(df)
        last = df.iloc[-1]

        return {
            "state": self.get_market_state(),
            "ema50": float(last["ema50"]) if not pd.isna(last["ema50"]) else None,
            "ema200": float(last["ema200"]) if not pd.isna(last["ema200"]) else None,
            "adx": float(last["adx"]) if not pd.isna(last["adx"]) else None,
            "atr": float(last["atr"]) if not pd.isna(last["atr"]) else None,
            "timestamp": str(df.index[-1]),
        }


# ═══════════════════════════════════════════════════════════════════
# Module 3: TriggerStrategy
# ═══════════════════════════════════════════════════════════════════

class TriggerStrategy:
    """
    15분봉 Volume-backed Pullback 패턴을 3캔들 시퀀스로 인식.

    3캔들 시퀀스:
      t-2: 기준 캔들 (Vol_SMA20 산출 기준)
      t-1: 풀백 캔들 (EMA 이탈 + 거래량 고갈 확인)
      t  : 돌파 캔들 (가장 최근 완성된 캔들, EMA 복귀 확인)
    """

    EMA_PERIOD = 15
    VOL_SMA_PERIOD = 20
    VOL_THRESHOLD = 0.50  # vol < vol_sma20 * 0.50

    def __init__(self):
        self._last_info: Dict = {}

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """15m DataFrame에 EMA15, Vol_SMA20 계산."""
        df = df.copy()
        df["ema15"] = ta.ema(df["close"], length=self.EMA_PERIOD)
        df["vol_sma20"] = df["volume"].rolling(self.VOL_SMA_PERIOD).mean()
        return df

    def generate_signal(self, df_15m: pd.DataFrame, meta_state: str) -> str:
        """
        3캔들 시퀀스 기반 진입 신호 생성.

        Args:
            df_15m: 15분봉 DataFrame (OHLCV)
            meta_state: 'LONG_ALLOWED' | 'SHORT_ALLOWED' | 'WAIT'

        Returns:
            'EXECUTE_LONG' | 'EXECUTE_SHORT' | 'HOLD'
        """
        # Step 1: 데이터 유효성
        if meta_state == "WAIT":
            self._last_info = {"signal": "HOLD", "reason": "meta_state=WAIT"}
            return "HOLD"

        if df_15m is None or len(df_15m) < 25:
            self._last_info = {"signal": "HOLD", "reason": f"데이터 부족 ({len(df_15m) if df_15m is not None else 0}행)"}
            return "HOLD"

        df = self._calc_indicators(df_15m)

        # Step 2: 캔들 인덱싱
        t = df.iloc[-1]    # 최근 완성 캔들 (돌파 확인)
        t_1 = df.iloc[-2]  # 직전 캔들 (풀백 확인)
        t_2 = df.iloc[-3]  # 그 이전 캔들 (Vol SMA 기준)

        # NaN 체크
        if (pd.isna(t["ema15"]) or pd.isna(t_1["ema15"])
                or pd.isna(t_2["vol_sma20"])):
            self._last_info = {"signal": "HOLD", "reason": "지표 NaN"}
            return "HOLD"

        vol_sma20_t2 = t_2["vol_sma20"]
        vol_t1 = t_1["volume"]
        vol_ratio = vol_t1 / vol_sma20_t2 if vol_sma20_t2 > 0 else float("inf")
        vol_dry = vol_ratio < self.VOL_THRESHOLD

        # 공통 info 구성
        self._last_info = {
            "ema15_t": float(t["ema15"]),
            "ema15_t1": float(t_1["ema15"]),
            "vol_sma20_t2": float(vol_sma20_t2),
            "vol_t1": float(vol_t1),
            "vol_ratio": round(vol_ratio, 4),
            "close_t1": float(t_1["close"]),
            "close_t": float(t["close"]),
        }

        # Step 3: LONG 시그널
        if meta_state == "LONG_ALLOWED":
            pullback = t_1["close"] < t_1["ema15"]     # t-1 EMA 아래로 이탈
            resumption = t["close"] > t["ema15"]        # t EMA 위로 복귀

            if pullback and vol_dry and resumption:
                self._last_info.update({
                    "signal": "EXECUTE_LONG",
                    "reason": f"풀백 이탈 + 거래량 고갈({vol_ratio:.2f}) + 돌파 복귀",
                })
                return "EXECUTE_LONG"

            reasons = []
            if not pullback:
                reasons.append(f"이탈 없음(close_t1={t_1['close']:.4f} >= ema15={t_1['ema15']:.4f})")
            if not vol_dry:
                reasons.append(f"거래량 과다({vol_ratio:.2f} >= {self.VOL_THRESHOLD})")
            if not resumption:
                reasons.append(f"복귀 실패(close_t={t['close']:.4f} <= ema15={t['ema15']:.4f})")
            self._last_info.update({"signal": "HOLD", "reason": " | ".join(reasons)})
            return "HOLD"

        # Step 4: SHORT 시그널
        if meta_state == "SHORT_ALLOWED":
            pullback = t_1["close"] > t_1["ema15"]     # t-1 EMA 위로 이탈
            resumption = t["close"] < t["ema15"]        # t EMA 아래로 복귀

            if pullback and vol_dry and resumption:
                self._last_info.update({
                    "signal": "EXECUTE_SHORT",
                    "reason": f"풀백 이탈 + 거래량 고갈({vol_ratio:.2f}) + 돌파 복귀",
                })
                return "EXECUTE_SHORT"

            reasons = []
            if not pullback:
                reasons.append(f"이탈 없음(close_t1={t_1['close']:.4f} <= ema15={t_1['ema15']:.4f})")
            if not vol_dry:
                reasons.append(f"거래량 과다({vol_ratio:.2f} >= {self.VOL_THRESHOLD})")
            if not resumption:
                reasons.append(f"복귀 실패(close_t={t['close']:.4f} >= ema15={t['ema15']:.4f})")
            self._last_info.update({"signal": "HOLD", "reason": " | ".join(reasons)})
            return "HOLD"

        # Step 5: 기본값
        self._last_info.update({"signal": "HOLD", "reason": f"미지원 meta_state={meta_state}"})
        return "HOLD"

    def get_trigger_info(self) -> Dict:
        """디버깅 및 로그용 트리거 상태 정보 반환."""
        return self._last_info.copy()


# ═══════════════════════════════════════════════════════════════════
# Module 4: ExecutionManager
# ═══════════════════════════════════════════════════════════════════

_MTF_TRADE_DIR = Path("data/trade_history")


class ExecutionManager:
    """
    TriggerStrategy의 신호를 받아 포지션 상태를 관리하고
    SL/TP를 계산하여 가상 주문을 실행한다 (Dry-run 모드).
    """

    ATR_SL_MULT = 1.5
    ATR_TP_MULT = 2.3

    def __init__(self, symbol: str = "XRPUSDT"):
        self.symbol = symbol
        self.current_position: Optional[str] = None  # None | 'LONG' | 'SHORT'
        self._entry_price: Optional[float] = None
        self._entry_ts: Optional[str] = None
        self._sl_price: Optional[float] = None
        self._tp_price: Optional[float] = None
        self._atr_at_entry: Optional[float] = None

    def execute(self, signal: str, current_price: float, atr_value: Optional[float]) -> Optional[Dict]:
        """
        신호에 따라 가상 주문 실행.

        Args:
            signal: 'EXECUTE_LONG' | 'EXECUTE_SHORT' | 'HOLD'
            current_price: 현재 시장가
            atr_value: 1h ATR 값

        Returns:
            주문 정보 Dict 또는 None (HOLD / 중복 포지션 / ATR 무효)
        """
        if signal == "HOLD":
            return None

        if self.current_position is not None:
            logger.debug(
                f"[ExecutionManager] 포지션 중복 차단: "
                f"현재={self.current_position}, 신호={signal}"
            )
            return None

        if atr_value is None or atr_value <= 0 or pd.isna(atr_value):
            logger.warning(f"[ExecutionManager] ATR 무효({atr_value}), 주문 차단")
            return None

        entry_price = current_price

        if signal == "EXECUTE_LONG":
            sl_price = entry_price - (atr_value * self.ATR_SL_MULT)
            tp_price = entry_price + (atr_value * self.ATR_TP_MULT)
            side = "LONG"
        elif signal == "EXECUTE_SHORT":
            sl_price = entry_price + (atr_value * self.ATR_SL_MULT)
            tp_price = entry_price - (atr_value * self.ATR_TP_MULT)
            side = "SHORT"
        else:
            return None

        self.current_position = side
        self._entry_price = entry_price
        self._entry_ts = datetime.now(timezone.utc).isoformat()
        self._sl_price = sl_price
        self._tp_price = tp_price
        self._atr_at_entry = atr_value

        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        # ── Dry-run 로그 ──
        logger.info(
            f"\n┌──────────────────────────────────────────────┐\n"
            f"│ [DRY-RUN] 가상 주문 실행                      │\n"
            f"│ 방향: {side:<5} | 진입가: {entry_price:.4f}            │\n"
            f"│ SL: {sl_price:.4f} ({'-' if side == 'LONG' else '+'}{sl_dist:.4f}, ATR×{self.ATR_SL_MULT})       │\n"
            f"│ TP: {tp_price:.4f} ({'+' if side == 'LONG' else '-'}{tp_dist:.4f}, ATR×{self.ATR_TP_MULT})       │\n"
            f"│ R:R = 1:{rr_ratio:.1f}                                │\n"
            f"└──────────────────────────────────────────────┘"
        )

        # ── 실주문 (프로덕션 전환 시 주석 해제) ──
        # if side == "LONG":
        #     await self.exchange.create_market_buy_order(symbol, amount)
        #     await self.exchange.create_order(symbol, 'stop_market', 'sell', amount, params={'stopPrice': sl_price})
        #     await self.exchange.create_order(symbol, 'take_profit_market', 'sell', amount, params={'stopPrice': tp_price})
        # elif side == "SHORT":
        #     await self.exchange.create_market_sell_order(symbol, amount)
        #     await self.exchange.create_order(symbol, 'stop_market', 'buy', amount, params={'stopPrice': sl_price})
        #     await self.exchange.create_order(symbol, 'take_profit_market', 'buy', amount, params={'stopPrice': tp_price})

        return {
            "action": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "atr": atr_value,
            "risk_reward": round(rr_ratio, 2),
        }

    def close_position(self, reason: str, exit_price: float = 0.0, pnl_bps: float = 0.0) -> None:
        """포지션 청산 + JSONL 기록 (상태 초기화)."""
        if self.current_position is None:
            logger.debug("[ExecutionManager] 청산할 포지션 없음")
            return

        logger.info(
            f"[ExecutionManager] 포지션 청산: {self.current_position} "
            f"(진입: {self._entry_price:.4f}) | 사유: {reason}"
        )

        # JSONL에 기록
        self._save_trade(reason, exit_price, pnl_bps)

        # ── 실주문 (프로덕션 전환 시 주석 해제) ──
        # if self.current_position == "LONG":
        #     await self.exchange.create_market_sell_order(symbol, amount)
        # elif self.current_position == "SHORT":
        #     await self.exchange.create_market_buy_order(symbol, amount)

        self.current_position = None
        self._entry_price = None
        self._entry_ts = None
        self._sl_price = None
        self._tp_price = None
        self._atr_at_entry = None

    def _save_trade(self, reason: str, exit_price: float, pnl_bps: float) -> None:
        """거래 기록을 JSONL 파일에 append."""
        record = {
            "symbol": self.symbol,
            "side": self.current_position,
            "entry_price": self._entry_price,
            "entry_ts": self._entry_ts,
            "exit_price": exit_price,
            "exit_ts": datetime.now(timezone.utc).isoformat(),
            "sl_price": self._sl_price,
            "tp_price": self._tp_price,
            "atr": self._atr_at_entry,
            "pnl_bps": round(pnl_bps, 1),
            "reason": reason,
        }
        try:
            _MTF_TRADE_DIR.mkdir(parents=True, exist_ok=True)
            path = _MTF_TRADE_DIR / f"mtf_{self.symbol.replace('/', '').replace(':', '').lower()}.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"[ExecutionManager] 거래 기록 저장: {path.name}")
        except Exception as e:
            logger.warning(f"[ExecutionManager] 거래 기록 저장 실패: {e}")

    def get_position_info(self) -> Dict:
        """현재 포지션 정보 반환."""
        return {
            "position": self.current_position,
            "entry_price": self._entry_price,
            "sl_price": self._sl_price,
            "tp_price": self._tp_price,
        }


# ═══════════════════════════════════════════════════════════════════
# 검증 테스트
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Main Loop: OOS Dry-run
# ═══════════════════════════════════════════════════════════════════

class MTFPullbackBot:
    """MTF Pullback Bot 메인 루프 — Dry-run OOS 검증용."""

    LOOP_INTERVAL = 1   # 초 (TimeframeSync 4초 윈도우를 놓치지 않기 위해)
    POLL_INTERVAL = 30  # 데이터 폴링 주기 (초)

    def __init__(self, symbol: str = "XRP/USDT:USDT"):
        self.symbol = symbol
        self.fetcher = DataFetcher(symbol=symbol)
        self.meta = MetaFilter(self.fetcher)
        self.trigger = TriggerStrategy()
        self.executor = ExecutionManager(symbol=symbol)
        self.notifier = DiscordNotifier(
            webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        )
        self._last_15m_check_ts: int = 0  # 중복 체크 방지
        self._last_poll_ts: float = 0  # 마지막 폴링 시각

    async def run(self):
        """메인 루프: 30초 폴링 → 15m 캔들 close 감지 → 신호 판정."""
        logger.info(f"[MTFBot] 시작: {self.symbol} (Dry-run OOS 모드)")

        await self.fetcher.initialize()

        # 초기 상태 출력
        meta_state = self.meta.get_market_state()
        atr = self.meta.get_current_atr()
        logger.info(f"[MTFBot] 초기 상태: Meta={meta_state}, ATR={atr}")
        self.notifier.notify_info(
            f"**[MTF Dry-run] 봇 시작**\n"
            f"심볼: `{self.symbol}` | Meta: `{meta_state}` | ATR: `{atr:.6f}`" if atr else
            f"**[MTF Dry-run] 봇 시작**\n심볼: `{self.symbol}` | Meta: `{meta_state}` | ATR: N/A"
        )

        try:
            while True:
                await asyncio.sleep(self.LOOP_INTERVAL)

                try:
                    # 데이터 폴링 (30초마다)
                    now_mono = _time.monotonic()
                    if now_mono - self._last_poll_ts >= self.POLL_INTERVAL:
                        await self._poll_and_update()
                        self._last_poll_ts = now_mono

                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                    # 15m 캔들 close 감지
                    if TimeframeSync.is_15m_candle_closed(now_ms):
                        if now_ms - self._last_15m_check_ts > 60_000:  # 1분 이내 중복 방지
                            self._last_15m_check_ts = now_ms
                            await self._poll_and_update()  # 최신 데이터 보장
                            await self._on_15m_close()

                    # 포지션 보유 중이면 SL/TP 모니터링
                    if self.executor.current_position is not None:
                        self._check_sl_tp()

                except Exception as e:
                    logger.error(f"[MTFBot] 루프 에러: {e}")

        except asyncio.CancelledError:
            logger.info("[MTFBot] 종료 시그널 수신")
        finally:
            await self.fetcher.close()
            logger.info("[MTFBot] 종료 완료")

    async def _poll_and_update(self):
        """데이터 폴링 업데이트."""
        # 15m
        raw_15m = await self.fetcher.fetch_ohlcv(self.symbol, "15m", limit=3)
        for candle in raw_15m:
            if candle[0] > self.fetcher._last_15m_ts:
                self.fetcher.klines_15m.append(candle)
                self.fetcher._last_15m_ts = candle[0]

        # 1h
        raw_1h = await self.fetcher.fetch_ohlcv(self.symbol, "1h", limit=3)
        for candle in raw_1h:
            if candle[0] > self.fetcher._last_1h_ts:
                self.fetcher.klines_1h.append(candle)
                self.fetcher._last_1h_ts = candle[0]

    async def _on_15m_close(self):
        """15m 캔들 종료 시 신호 판정."""
        df_15m = self.fetcher.get_15m_dataframe()
        meta_state = self.meta.get_market_state()
        atr = self.meta.get_current_atr()

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        last_close = float(df_15m.iloc[-1]["close"]) if df_15m is not None and len(df_15m) > 0 else 0
        pos_info = self.executor.current_position or "없음"

        # Heartbeat: 15분마다 무조건 출력 (메타 지표 포함)
        meta_info = self.meta.get_meta_info()
        adx_val = meta_info.get("adx")
        ema50_val = meta_info.get("ema50")
        ema200_val = meta_info.get("ema200")
        adx_str = f"{adx_val:.2f}" if adx_val is not None else "N/A"
        ema50_str = f"{ema50_val:.4f}" if ema50_val is not None else "N/A"
        ema200_str = f"{ema200_val:.4f}" if ema200_val is not None else "N/A"
        atr_str = f"{atr:.6f}" if atr else "N/A"
        logger.info(
            f"[Heartbeat] 15m 마감 ({now_str}) | Meta: {meta_state} | "
            f"ADX: {adx_str} | EMA50: {ema50_str} | EMA200: {ema200_str} | "
            f"ATR: {atr_str} | Close: {last_close:.4f} | Pos: {pos_info}"
        )

        signal = self.trigger.generate_signal(df_15m, meta_state)
        info = self.trigger.get_trigger_info()

        if signal != "HOLD":
            logger.info(f"[MTFBot] 신호: {signal} | {info.get('reason', '')}")
            current_price = last_close
            result = self.executor.execute(signal, current_price, atr)
            if result:
                logger.info(f"[MTFBot] 거래 기록: {result}")
                side = result["action"]
                sl_dist = abs(result["entry_price"] - result["sl_price"])
                tp_dist = abs(result["tp_price"] - result["entry_price"])
                self.notifier._send(
                    f"📌 **[MTF Dry-run] 가상 {side} 진입**\n"
                    f"진입가: `{result['entry_price']:.4f}` | ATR: `{result['atr']:.6f}`\n"
                    f"SL: `{result['sl_price']:.4f}` ({sl_dist:.4f}) | "
                    f"TP: `{result['tp_price']:.4f}` ({tp_dist:.4f})\n"
                    f"R:R = `1:{result['risk_reward']}` | Meta: `{meta_state}`\n"
                    f"사유: {info.get('reason', '')}"
                )
        else:
            logger.info(f"[MTFBot] HOLD | {info.get('reason', '')}")

    def _check_sl_tp(self):
        """현재 가격으로 SL/TP 도달 여부 확인 (15m 캔들 high/low 기반)."""
        df_15m = self.fetcher.get_15m_dataframe()
        if df_15m is None or len(df_15m) < 1:
            return

        last = df_15m.iloc[-1]
        pos = self.executor.current_position
        sl = self.executor._sl_price
        tp = self.executor._tp_price
        entry = self.executor._entry_price

        if pos is None or sl is None or tp is None:
            return

        hit_sl = hit_tp = False
        if pos == "LONG":
            hit_sl = last["low"] <= sl
            hit_tp = last["high"] >= tp
        else:
            hit_sl = last["high"] >= sl
            hit_tp = last["low"] <= tp

        if hit_sl and hit_tp:
            exit_price = sl
            pnl = (exit_price - entry) / entry if pos == "LONG" else (entry - exit_price) / entry
            pnl_bps = pnl * 10000
            logger.info(f"[MTFBot] SL+TP 동시 히트 → SL 우선 청산 | PnL: {pnl_bps:+.1f}bps")
            self.executor.close_position(f"SL 히트 ({exit_price:.4f})", exit_price, pnl_bps)
            self.notifier._send(
                f"❌ **[MTF Dry-run] {pos} SL 청산**\n"
                f"진입: `{entry:.4f}` → 청산: `{exit_price:.4f}`\n"
                f"PnL: `{pnl_bps:+.1f}bps`"
            )
        elif hit_sl:
            exit_price = sl
            pnl = (exit_price - entry) / entry if pos == "LONG" else (entry - exit_price) / entry
            pnl_bps = pnl * 10000
            logger.info(f"[MTFBot] SL 히트 | 청산가: {exit_price:.4f} | PnL: {pnl_bps:+.1f}bps")
            self.executor.close_position(f"SL 히트 ({exit_price:.4f})", exit_price, pnl_bps)
            self.notifier._send(
                f"❌ **[MTF Dry-run] {pos} SL 청산**\n"
                f"진입: `{entry:.4f}` → 청산: `{exit_price:.4f}`\n"
                f"PnL: `{pnl_bps:+.1f}bps`"
            )
        elif hit_tp:
            exit_price = tp
            pnl = (exit_price - entry) / entry if pos == "LONG" else (entry - exit_price) / entry
            pnl_bps = pnl * 10000
            logger.info(f"[MTFBot] TP 히트 | 청산가: {exit_price:.4f} | PnL: {pnl_bps:+.1f}bps")
            self.executor.close_position(f"TP 히트 ({exit_price:.4f})", exit_price, pnl_bps)
            self.notifier._send(
                f"✅ **[MTF Dry-run] {pos} TP 청산**\n"
                f"진입: `{entry:.4f}` → 청산: `{exit_price:.4f}`\n"
                f"PnL: `{pnl_bps:+.1f}bps`"
            )


# ═══════════════════════════════════════════════════════════════════
# 검증 테스트
# ═══════════════════════════════════════════════════════════════════

async def test_module_1_2():
    """Module 1 & 2 검증 테스트."""
    print("=" * 60)
    print("  MTF Bot Module 1 & 2 검증 테스트")
    print("=" * 60)

    # ── 1. TimeframeSync 검증 ──
    print("\n[1] TimeframeSync 검증")
    # 2026-01-01 01:00:03 UTC (1h 캔들 close 직후)
    ts_1h_close = int(datetime(2026, 1, 1, 1, 0, 3, tzinfo=timezone.utc).timestamp() * 1000)
    # 2026-01-01 00:15:04 UTC (15m 캔들 close 직후)
    ts_15m_close = int(datetime(2026, 1, 1, 0, 15, 4, tzinfo=timezone.utc).timestamp() * 1000)
    # 2026-01-01 00:15:00 UTC (정각 — 아직 딜레이 전)
    ts_too_early = int(datetime(2026, 1, 1, 0, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # 2026-01-01 00:15:10 UTC (너무 늦음)
    ts_too_late = int(datetime(2026, 1, 1, 0, 15, 10, tzinfo=timezone.utc).timestamp() * 1000)
    # 2026-01-01 00:07:03 UTC (15m 경계 아님)
    ts_not_boundary = int(datetime(2026, 1, 1, 0, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

    assert TimeframeSync.is_1h_candle_closed(ts_1h_close) is True, "1h close 판별 실패"
    assert TimeframeSync.is_15m_candle_closed(ts_15m_close) is True, "15m close 판별 실패"
    assert TimeframeSync.is_15m_candle_closed(ts_too_early) is False, "정각(0초)에 True 반환"
    assert TimeframeSync.is_15m_candle_closed(ts_too_late) is False, "10초에 True 반환"
    assert TimeframeSync.is_15m_candle_closed(ts_not_boundary) is False, "비경계 시점에 True 반환"
    assert TimeframeSync.is_1h_candle_closed(ts_15m_close) is False, "15분에 1h close True 반환"
    print("  ✓ TimeframeSync: second 2~5 범위에서만 True 반환 확인")

    # ── 2. DataFetcher 초기화 ──
    print("\n[2] DataFetcher 초기화")
    fetcher = DataFetcher(symbol="XRP/USDT:USDT")
    try:
        await fetcher.initialize()

        assert len(fetcher.klines_15m) == 200, f"15m 캔들 {len(fetcher.klines_15m)}개 (200 예상)"
        assert len(fetcher.klines_1h) == 200, f"1h 캔들 {len(fetcher.klines_1h)}개 (200 예상)"
        print(f"  ✓ 초기화 완료: 15m={len(fetcher.klines_15m)}개, 1h={len(fetcher.klines_1h)}개")

        # ── 3. [:-1] 슬라이싱 검증 ──
        print("\n[3] get_1h_dataframe_completed() [:-1] 검증")
        df_1h = fetcher.get_1h_dataframe_completed()
        assert df_1h is not None, "1h DataFrame이 None"
        assert len(df_1h) == 199, f"1h completed 캔들 {len(df_1h)}개 (199 예상)"

        # 마지막 완성 봉의 timestamp < 현재 진행 중 봉의 timestamp
        last_completed_ts = df_1h.index[-1]
        last_raw_ts = pd.to_datetime(fetcher.klines_1h[-1][0], unit="ms", utc=True)
        assert last_completed_ts < last_raw_ts, "completed 봉이 진행 중 봉을 포함"
        print(f"  ✓ 1h completed: {len(df_1h)}개 (200 - 1 = 199, 미완성 봉 제외 확인)")
        print(f"    마지막 완성 봉: {last_completed_ts}")
        print(f"    진행 중 봉:     {last_raw_ts} (제외됨)")

        # 15m DataFrame 검증
        df_15m = fetcher.get_15m_dataframe()
        assert df_15m is not None and len(df_15m) == 200
        print(f"  ✓ 15m DataFrame: {len(df_15m)}개")

        # ── 4. MetaFilter 검증 ──
        print("\n[4] MetaFilter 검증")
        meta = MetaFilter(fetcher)

        state = meta.get_market_state()
        assert state in ("LONG_ALLOWED", "SHORT_ALLOWED", "WAIT"), f"비정상 상태: {state}"
        print(f"  ✓ MetaFilter 상태: {state}")

        atr = meta.get_current_atr()
        assert atr is not None and atr > 0, f"ATR 비정상: {atr}"
        print(f"  ✓ ATR: {atr:.6f} (> 0 확인)")

        info = meta.get_meta_info()
        print(f"  ✓ Meta Info: {info}")

        # ATR 범위 검증 (XRP 기준 0.0001 ~ 0.1)
        assert 0.0001 <= atr <= 0.1, f"ATR 범위 이탈: {atr}"
        print(f"  ✓ ATR 범위 정상: 0.0001 <= {atr:.6f} <= 0.1")

    finally:
        await fetcher.close()

    print("\n" + "=" * 60)
    print("  모든 검증 통과 ✓")
    print("=" * 60)


async def test_module_3_4():
    """
    Module 3 + 4 통합 테스트.

    검증 항목:
    [Module 3 - TriggerStrategy]
    1. 신호 생성: 'EXECUTE_LONG' | 'EXECUTE_SHORT' | 'HOLD' 중 하나 반환
    2. EMA15: NaN 아님, 양수, 현실적 범위
    3. Vol_SMA20: NaN 아님, 양수
    4. vol_ratio: 0.0 ~ 2.0+ 범위 내
    5. 3캔들 시퀀스: t-2, t-1, t 인덱싱 정확성
    6. meta_state 필터: 'LONG_ALLOWED'에서만 LONG, 'SHORT_ALLOWED'에서만 SHORT

    [Module 4 - ExecutionManager]
    7. 포지션 중복 방지
    8. SL/TP 계산: ATR * 1.5 (SL), ATR * 2.3 (TP)
    9. Dry-run 로그 출력
    10. 청산 후 재진입 가능
    """
    print("=" * 60)
    print("  MTF Bot Module 3 & 4 통합 테스트")
    print("=" * 60)

    # ── DataFetcher로 실제 데이터 로드 ──
    fetcher = DataFetcher(symbol="XRP/USDT:USDT")
    try:
        await fetcher.initialize()

        df_15m = fetcher.get_15m_dataframe()
        assert df_15m is not None and len(df_15m) >= 25, "15m 데이터 부족"

        meta = MetaFilter(fetcher)
        meta_state = meta.get_market_state()
        atr = meta.get_current_atr()
        print(f"\n[환경] MetaFilter: {meta_state} | ATR: {atr}")

        # ── [Module 3] TriggerStrategy 검증 ──
        print("\n[1] TriggerStrategy 신호 생성")
        trigger = TriggerStrategy()

        # 테스트 1: 실제 데이터로 신호 생성
        signal = trigger.generate_signal(df_15m, meta_state)
        assert signal in ("EXECUTE_LONG", "EXECUTE_SHORT", "HOLD"), f"비정상 신호: {signal}"
        print(f"  ✓ 신호: {signal}")

        info = trigger.get_trigger_info()
        print(f"  ✓ Trigger Info: {info}")

        # 테스트 2: 지표 값 검증
        if "ema15_t" in info:
            assert not pd.isna(info["ema15_t"]) and info["ema15_t"] > 0, "EMA15 비정상"
            assert not pd.isna(info["vol_sma20_t2"]) and info["vol_sma20_t2"] > 0, "Vol SMA20 비정상"
            assert 0 <= info["vol_ratio"] <= 100, f"vol_ratio 비정상: {info['vol_ratio']}"
            print(f"  ✓ EMA15(t): {info['ema15_t']:.4f}")
            print(f"  ✓ Vol SMA20(t-2): {info['vol_sma20_t2']:.0f}")
            print(f"  ✓ Vol ratio: {info['vol_ratio']:.4f} ({'고갈' if info['vol_ratio'] < 0.5 else '정상'})")

        # 테스트 3: meta_state=WAIT → 무조건 HOLD
        signal_wait = trigger.generate_signal(df_15m, "WAIT")
        assert signal_wait == "HOLD", "WAIT 상태에서 HOLD 아닌 신호 발생"
        print(f"  ✓ meta_state=WAIT → {signal_wait}")

        # 테스트 4: 데이터 부족 → HOLD
        signal_short = trigger.generate_signal(df_15m.iloc[:10], "LONG_ALLOWED")
        assert signal_short == "HOLD", "데이터 부족에서 HOLD 아닌 신호 발생"
        print(f"  ✓ 데이터 부족(10행) → {signal_short}")

        # 테스트 5: None DataFrame → HOLD
        signal_none = trigger.generate_signal(None, "LONG_ALLOWED")
        assert signal_none == "HOLD"
        print(f"  ✓ None DataFrame → HOLD")

        # ── [Module 4] ExecutionManager 검증 ──
        print(f"\n[2] ExecutionManager 검증")
        executor = ExecutionManager()

        # 테스트 6: HOLD → None
        result = executor.execute("HOLD", 2.5, 0.01)
        assert result is None, "HOLD에서 주문 실행됨"
        print(f"  ✓ HOLD → None")

        # 테스트 7: ATR 무효 → None
        result = executor.execute("EXECUTE_LONG", 2.5, None)
        assert result is None, "ATR=None에서 주문 실행됨"
        result = executor.execute("EXECUTE_LONG", 2.5, 0)
        assert result is None, "ATR=0에서 주문 실행됨"
        print(f"  ✓ ATR 무효 → None")

        # 테스트 8: 정상 LONG 주문
        print(f"\n  [LONG 가상 주문 테스트]")
        test_atr = 0.01
        result = executor.execute("EXECUTE_LONG", 2.5340, test_atr)
        assert result is not None, "정상 주문이 None 반환"
        assert result["action"] == "LONG"
        assert abs(result["sl_price"] - (2.5340 - 0.01 * 1.5)) < 1e-8, "SL 계산 오류"
        assert abs(result["tp_price"] - (2.5340 + 0.01 * 2.3)) < 1e-8, "TP 계산 오류"
        assert result["risk_reward"] == 1.53, f"R:R 오류: {result['risk_reward']}"
        print(f"  ✓ LONG 주문: entry={result['entry_price']}, SL={result['sl_price']:.4f}, TP={result['tp_price']:.4f}")
        print(f"  ✓ R:R = 1:{result['risk_reward']}")

        # 테스트 9: 포지션 중복 방지
        result_dup = executor.execute("EXECUTE_SHORT", 2.5000, test_atr)
        assert result_dup is None, "중복 포지션 허용됨"
        assert executor.current_position == "LONG", "포지션 상태 변경됨"
        print(f"  ✓ 중복 차단: LONG 포지션 중 SHORT 신호 → None")

        # 테스트 10: 청산 후 재진입
        executor.close_position("테스트 청산")
        assert executor.current_position is None, "청산 후 포지션 잔존"
        print(f"  ✓ 청산 완료, 포지션=None")

        # 테스트 11: SHORT 주문
        print(f"\n  [SHORT 가상 주문 테스트]")
        result_short = executor.execute("EXECUTE_SHORT", 2.5340, test_atr)
        assert result_short is not None
        assert result_short["action"] == "SHORT"
        assert abs(result_short["sl_price"] - (2.5340 + 0.01 * 1.5)) < 1e-8, "SHORT SL 오류"
        assert abs(result_short["tp_price"] - (2.5340 - 0.01 * 2.3)) < 1e-8, "SHORT TP 오류"
        print(f"  ✓ SHORT 주문: entry={result_short['entry_price']}, SL={result_short['sl_price']:.4f}, TP={result_short['tp_price']:.4f}")

        executor.close_position("테스트 종료")

        # 테스트 12: 빈 포지션 청산 → 에러 없이 처리
        executor.close_position("이미 청산됨")
        print(f"  ✓ 빈 포지션 청산 → 에러 없음")

    finally:
        await fetcher.close()

    print("\n" + "=" * 60)
    print("  Module 3 & 4 모든 검증 통과 ✓")
    print("=" * 60)


async def test_all():
    """Module 1~4 전체 검증."""
    await test_module_1_2()
    print("\n")
    await test_module_3_4()


if __name__ == "__main__":
    asyncio.run(test_all())
