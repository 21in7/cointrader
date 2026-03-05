import asyncio
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from src.config import Config


class BinanceFuturesClient:
    def __init__(self, config: Config, symbol: str = None):
        self.config = config
        self.symbol = symbol or config.symbol
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
        )

    MIN_NOTIONAL = 5.0  # 바이낸스 선물 최소 명목금액 (USDT)

    def calculate_quantity(self, balance: float, price: float, leverage: int, margin_ratio: float) -> float:
        """동적 증거금 비율 기반 포지션 크기 계산 (최소 명목금액 $5 보장)"""
        notional = balance * margin_ratio * leverage
        if notional < self.MIN_NOTIONAL:
            notional = self.MIN_NOTIONAL
        quantity = notional / price
        qty_rounded = round(quantity, 1)
        if qty_rounded * price < self.MIN_NOTIONAL:
            qty_rounded = round(self.MIN_NOTIONAL / price + 0.05, 1)
        return qty_rounded

    async def set_leverage(self, leverage: int) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.futures_change_leverage(
                symbol=self.symbol, leverage=leverage
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

    _ALGO_ORDER_TYPES = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT", "TRAILING_STOP_MARKET"}

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

        if order_type in self._ALGO_ORDER_TYPES:
            return await self._place_algo_order(
                side=side,
                quantity=quantity,
                order_type=order_type,
                stop_price=stop_price,
                reduce_only=reduce_only,
            )

        params = dict(
            symbol=self.symbol,
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

    async def _place_algo_order(
        self,
        side: str,
        quantity: float,
        order_type: str,
        stop_price: float = None,
        reduce_only: bool = False,
    ) -> dict:
        """STOP_MARKET / TAKE_PROFIT_MARKET 등 Algo Order API(/fapi/v1/algoOrder)로 전송."""
        loop = asyncio.get_event_loop()
        params = dict(
            symbol=self.symbol,
            side=side,
            algoType="CONDITIONAL",
            type=order_type,
            quantity=quantity,
            reduceOnly="true" if reduce_only else "false",
        )
        if stop_price:
            params["triggerPrice"] = stop_price
        try:
            return await loop.run_in_executor(
                None, lambda: self.client.futures_create_algo_order(**params)
            )
        except BinanceAPIException as e:
            logger.error(f"Algo 주문 실패: {e}")
            raise

    async def get_position(self) -> dict | None:
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(
            None,
            lambda: self.client.futures_position_information(
                symbol=self.symbol
            ),
        )
        for p in positions:
            if float(p["positionAmt"]) != 0:
                return p
        return None

    async def cancel_all_orders(self):
        """일반 오픈 주문과 Algo 오픈 주문을 모두 취소한다."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.futures_cancel_all_open_orders(
                symbol=self.symbol
            ),
        )
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.futures_cancel_all_algo_open_orders(
                    symbol=self.symbol
                ),
            )
        except Exception as e:
            logger.warning(f"Algo 주문 전체 취소 실패 (무시): {e}")

    async def get_open_interest(self) -> float | None:
        """현재 미결제약정(OI)을 조회한다. 오류 시 None 반환."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.client.futures_open_interest(symbol=self.symbol),
            )
            return float(result["openInterest"])
        except Exception as e:
            logger.warning(f"OI 조회 실패 (무시): {e}")
            return None

    async def get_funding_rate(self) -> float | None:
        """현재 펀딩비를 조회한다. 오류 시 None 반환."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.client.futures_mark_price(symbol=self.symbol),
            )
            return float(result["lastFundingRate"])
        except Exception as e:
            logger.warning(f"펀딩비 조회 실패 (무시): {e}")
            return None

    async def get_oi_history(self, limit: int = 5) -> list[float]:
        """최근 OI 변화율 히스토리를 조회한다 (봇 초기화용). 실패 시 빈 리스트."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.client.futures_open_interest_hist(
                    symbol=self.symbol, period="15m", limit=limit + 1,
                ),
            )
            if len(result) < 2:
                return []
            oi_values = [float(r["sumOpenInterest"]) for r in result]
            changes = []
            for i in range(1, len(oi_values)):
                if oi_values[i - 1] > 0:
                    changes.append((oi_values[i] - oi_values[i - 1]) / oi_values[i - 1])
                else:
                    changes.append(0.0)
            return changes
        except Exception as e:
            logger.warning(f"OI 히스토리 조회 실패 (무시): {e}")
            return []

    async def create_listen_key(self) -> str:
        """POST /fapi/v1/listenKey — listenKey 신규 발급"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.futures_stream_get_listen_key(),
        )
        return result

    async def keepalive_listen_key(self, listen_key: str) -> None:
        """PUT /fapi/v1/listenKey — listenKey 만료 연장 (60분 → 리셋)"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.futures_stream_keepalive(listenKey=listen_key),
        )

    async def delete_listen_key(self, listen_key: str) -> None:
        """DELETE /fapi/v1/listenKey — listenKey 삭제 (정상 종료 시)"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.futures_stream_close(listenKey=listen_key),
            )
        except Exception as e:
            logger.warning(f"listenKey 삭제 실패 (무시): {e}")
