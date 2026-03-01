import asyncio
from collections import deque
from typing import Callable
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

# 15분봉 기준 EMA50 안정화에 필요한 최소 캔들 수.
# EMA50=50, StochRSI(14,14,3,3)=44, MACD(12,26,9)=33 중 최댓값에 여유분 추가.
_MIN_CANDLES_FOR_SIGNAL = 100

# 초기 구동 시 REST API로 가져올 과거 캔들 수.
# 15분봉 200개 = 50시간치 — EMA50(12.5h) 대비 4배 여유.
_PRELOAD_LIMIT = 200



class KlineStream:
    def __init__(
        self,
        symbol: str,
        interval: str = "15m",
        buffer_size: int = 200,
        on_candle: Callable = None,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self.buffer: deque = deque(maxlen=buffer_size)
        self.on_candle = on_candle

    def parse_kline(self, msg: dict) -> dict:
        k = msg["k"]
        return {
            "timestamp": k["t"],
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
            "is_closed": k["x"],
        }

    def handle_message(self, msg: dict):
        candle = self.parse_kline(msg)
        if candle["is_closed"]:
            self.buffer.append(candle)
            if self.on_candle:
                self.on_candle(candle)

    def get_dataframe(self) -> pd.DataFrame | None:
        if len(self.buffer) < _MIN_CANDLES_FOR_SIGNAL:
            return None
        df = pd.DataFrame(list(self.buffer))
        df.set_index("timestamp", inplace=True)
        return df

    async def _preload_history(self, client: AsyncClient, limit: int = _PRELOAD_LIMIT):
        """REST API로 과거 캔들 데이터를 버퍼에 미리 채운다."""
        logger.info(f"과거 캔들 {limit}개 로드 중...")
        klines = await client.futures_klines(
            symbol=self.symbol.upper(),
            interval=self.interval,
            limit=limit,
        )
        # 마지막 캔들은 아직 닫히지 않았을 수 있으므로 제외
        for k in klines[:-1]:
            self.buffer.append({
                "timestamp": k[0],
                "open":      float(k[1]),
                "high":      float(k[2]),
                "low":       float(k[3]),
                "close":     float(k[4]),
                "volume":    float(k[5]),
                "is_closed": True,
            })
        logger.info(f"과거 캔들 {len(self.buffer)}개 로드 완료 — 즉시 신호 계산 가능")

    async def start(self, api_key: str, api_secret: str):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
        await self._preload_history(client)
        bm = BinanceSocketManager(client)
        stream_name = f"{self.symbol}@kline_{self.interval}"
        logger.info(f"WebSocket 스트림 시작: {stream_name}")
        try:
            async with bm.kline_futures_socket(
                symbol=self.symbol.upper(), interval=self.interval
            ) as stream:
                while True:
                    msg = await stream.recv()
                    self.handle_message(msg)
        finally:
            await client.close_connection()


class MultiSymbolStream:
    """
    바이낸스 Combined WebSocket으로 여러 심볼의 캔들을 단일 연결로 수신한다.
    XRP 캔들이 닫힐 때 on_candle 콜백을 호출한다.
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "15m",
        buffer_size: int = 200,
        on_candle: Callable = None,
    ):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.on_candle = on_candle
        self.buffers: dict[str, deque] = {
            s: deque(maxlen=buffer_size) for s in self.symbols
        }
        # 첫 번째 심볼이 주 심볼 (XRP)
        self.primary_symbol = self.symbols[0]

    def parse_kline(self, msg: dict) -> dict:
        k = msg["k"]
        return {
            "timestamp": k["t"],
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
            "is_closed": k["x"],
        }

    def handle_message(self, msg: dict):
        # Combined stream 메시지는 {"stream": "...", "data": {...}} 형태
        if "stream" in msg:
            data = msg["data"]
        else:
            data = msg

        if data.get("e") != "kline":
            return

        symbol = data["s"].lower()
        candle = self.parse_kline(data)

        if candle["is_closed"] and symbol in self.buffers:
            self.buffers[symbol].append(candle)
            if symbol == self.primary_symbol and self.on_candle:
                self.on_candle(candle)

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        key = symbol.lower()
        buf = self.buffers.get(key)
        if buf is None or len(buf) < _MIN_CANDLES_FOR_SIGNAL:
            return None
        df = pd.DataFrame(list(buf))
        df.set_index("timestamp", inplace=True)
        return df

    async def _preload_history(self, client: AsyncClient, limit: int = _PRELOAD_LIMIT):
        """REST API로 모든 심볼의 과거 캔들을 버퍼에 미리 채운다."""
        for symbol in self.symbols:
            logger.info(f"{symbol.upper()} 과거 캔들 {limit}개 로드 중...")
            klines = await client.futures_klines(
                symbol=symbol.upper(),
                interval=self.interval,
                limit=limit,
            )
            for k in klines[:-1]:
                self.buffers[symbol].append({
                    "timestamp": k[0],
                    "open":      float(k[1]),
                    "high":      float(k[2]),
                    "low":       float(k[3]),
                    "close":     float(k[4]),
                    "volume":    float(k[5]),
                    "is_closed": True,
                })
            logger.info(f"{symbol.upper()} {len(self.buffers[symbol])}개 로드 완료")

    async def start(self, api_key: str, api_secret: str):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
        await self._preload_history(client)
        bm = BinanceSocketManager(client)
        streams = [
            f"{s}@kline_{self.interval}" for s in self.symbols
        ]
        logger.info(f"Combined WebSocket 시작: {streams}")
        try:
            async with bm.futures_multiplex_socket(streams) as stream:
                while True:
                    msg = await stream.recv()
                    self.handle_message(msg)
        finally:
            await client.close_connection()
