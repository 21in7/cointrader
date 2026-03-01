import asyncio
from collections import deque
from typing import Callable
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from loguru import logger


class KlineStream:
    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
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
        if len(self.buffer) < 50:
            return None
        df = pd.DataFrame(list(self.buffer))
        df.set_index("timestamp", inplace=True)
        return df

    async def _preload_history(self, client: AsyncClient, limit: int = 200):
        """REST API로 과거 캔들 데이터를 버퍼에 미리 채운다."""
        logger.info(f"과거 캔들 {limit}개 로드 중...")
        loop = asyncio.get_event_loop()
        klines = await loop.run_in_executor(
            None,
            lambda: client.futures_klines(
                symbol=self.symbol.upper(),
                interval=self.interval,
                limit=limit,
            ),
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
