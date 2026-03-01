import asyncio
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from src.config import Config


class BinanceFuturesClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
        )

    def calculate_quantity(self, balance: float, price: float, leverage: int) -> float:
        """리스크 기반 포지션 크기 계산"""
        risk_amount = balance * self.config.risk_per_trade
        notional = risk_amount * leverage
        quantity = notional / price
        return round(quantity, 1)

    async def set_leverage(self, leverage: int) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.futures_change_leverage(
                symbol=self.config.symbol, leverage=leverage
            ),
        )

    async def get_balance(self) -> float:
        loop = asyncio.get_event_loop()
        balances = await loop.run_in_executor(
            None, self.client.futures_account_balance
        )
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    async def place_order(
        self,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False,
    ) -> dict:
        loop = asyncio.get_event_loop()
        params = dict(
            symbol=self.config.symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            reduceOnly=reduce_only,
        )
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if stop_price:
            params["stopPrice"] = stop_price
        try:
            return await loop.run_in_executor(
                None, lambda: self.client.futures_create_order(**params)
            )
        except BinanceAPIException as e:
            logger.error(f"주문 실패: {e}")
            raise

    async def get_position(self) -> dict | None:
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(
            None,
            lambda: self.client.futures_position_information(
                symbol=self.config.symbol
            ),
        )
        for p in positions:
            if float(p["positionAmt"]) != 0:
                return p
        return None

    async def cancel_all_orders(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.futures_cancel_all_open_orders(
                symbol=self.config.symbol
            ),
        )
