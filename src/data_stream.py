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

    async def start(self, api_key: str, api_secret: str):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
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
