import asyncio
import math
import threading
import time as _time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from src.config import Config


class BinanceFuturesClient:
    # 클래스 레벨 exchange info 캐시 (TTL 24시간)
    _exchange_info_cache: dict | None = None
    _exchange_info_time: float = 0.0
    _EXCHANGE_INFO_TTL: float = 86400.0  # 24시간

    def __init__(self, config: Config, symbol: str = None):
        self.config = config
        self.symbol = symbol or config.symbol
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
        )
        self._qty_precision: int | None = None
        self._price_precision: int | None = None
        self._api_lock = threading.Lock()  # requests.Session 스레드 안전성 보장

    MIN_NOTIONAL = 5.0  # 바이낸스 선물 최소 명목금액 (USDT)

    async def _run_api(self, func):
        """동기 API 호출을 스레드 풀에서 실행하되, _api_lock으로 직렬화한다."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._call_with_lock(func),
        )

    def _call_with_lock(self, func):
        with self._api_lock:
            return func()

    @classmethod
    def _get_exchange_info(cls, client: Client) -> dict | None:
        """exchange info를 클래스 레벨로 캐시한다 (TTL 24시간)."""
        now = _time.monotonic()
        if cls._exchange_info_cache is None or (now - cls._exchange_info_time) > cls._EXCHANGE_INFO_TTL:
            try:
                cls._exchange_info_cache = client.futures_exchange_info()
                cls._exchange_info_time = now
            except Exception as e:
                logger.warning(f"exchange info 조회 실패: {e}")
                return cls._exchange_info_cache  # 만료돼도 기존 캐시 반환
        return cls._exchange_info_cache

    def _load_symbol_precision(self) -> None:
        """바이낸스 exchange info에서 심볼별 수량/가격 정밀도를 로드한다."""
        info = self._get_exchange_info(self.client)
        if info is not None:
            for s in info["symbols"]:
                if s["symbol"] == self.symbol:
                    self._qty_precision = s.get("quantityPrecision", 1)
                    self._price_precision = s.get("pricePrecision", 2)
                    logger.info(
                        f"[{self.symbol}] 정밀도 로드: qty={self._qty_precision}, price={self._price_precision}"
                    )
                    return
            logger.warning(f"[{self.symbol}] exchange info에서 심볼 미발견, 기본 정밀도 사용")
        self._qty_precision = 1
        self._price_precision = 2

    @property
    def qty_precision(self) -> int:
        if self._qty_precision is None:
            self._load_symbol_precision()
        return self._qty_precision

    @property
    def price_precision(self) -> int:
        if self._price_precision is None:
            self._load_symbol_precision()
        return self._price_precision

    def _round_qty(self, qty: float) -> float:
        """심볼의 quantityPrecision에 맞춰 수량을 내림(truncate)한다."""
        p = self.qty_precision
        factor = 10 ** p
        return math.floor(qty * factor) / factor

    def _round_price(self, price: float) -> float:
        """심볼의 pricePrecision에 맞춰 가격을 반올림한다."""
        return round(price, self.price_precision)

    def calculate_quantity(self, balance: float, price: float, leverage: int, margin_ratio: float) -> float:
        """동적 증거금 비율 기반 포지션 크기 계산 (최소 명목금액 $5 보장)"""
        notional = balance * margin_ratio * leverage
        if notional < self.MIN_NOTIONAL:
            notional = self.MIN_NOTIONAL
        quantity = notional / price
        qty_rounded = self._round_qty(quantity)
        if qty_rounded * price < self.MIN_NOTIONAL:
            qty_rounded = self._round_qty(self.MIN_NOTIONAL / price + 10 ** -self.qty_precision)
        return qty_rounded

    async def set_leverage(self, leverage: int) -> dict:
        return await self._run_api(
            lambda: self.client.futures_change_leverage(
                symbol=self.symbol, leverage=leverage
            ),
        )

    async def get_balance(self) -> float:
        balances = await self._run_api(self.client.futures_account_balance)
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
        params = dict(
            symbol=self.symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            reduceOnly=reduce_only,
        )
        if price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if stop_price is not None:
            params["stopPrice"] = stop_price
        try:
            return await self._run_api(
                lambda: self.client.futures_create_order(**params)
            )
        except BinanceAPIException as e:
            logger.error(f"주문 실패: {e}")
            raise

    async def get_position(self) -> dict | None:
        positions = await self._run_api(
            lambda: self.client.futures_position_information(
                symbol=self.symbol
            ),
        )
        for p in positions:
            if float(p["positionAmt"]) != 0:
                return p
        return None

    async def get_open_orders(self) -> list[dict]:
        """현재 심볼의 오픈 주문 목록을 조회한다."""
        return await self._run_api(
            lambda: self.client.futures_get_open_orders(symbol=self.symbol),
        )

    async def cancel_all_orders(self):
        """오픈 주문을 모두 취소한다."""
        await self._run_api(
            lambda: self.client.futures_cancel_all_open_orders(
                symbol=self.symbol
            ),
        )

    async def get_recent_income(self, limit: int = 5, start_time: int | None = None) -> tuple[list[dict], list[dict]]:
        """최근 REALIZED_PNL + COMMISSION 내역을 조회한다.

        Args:
            limit: 최대 조회 건수
            start_time: 밀리초 단위 시작 시각. 지정 시 해당 시각 이후 데이터만 반환.
        """
        try:
            pnl_params = dict(symbol=self.symbol, incomeType="REALIZED_PNL", limit=limit)
            comm_params = dict(symbol=self.symbol, incomeType="COMMISSION", limit=limit)
            if start_time is not None:
                pnl_params["startTime"] = start_time
                comm_params["startTime"] = start_time

            rows = await self._run_api(
                lambda: self.client.futures_income_history(**pnl_params),
            )
            commissions = await self._run_api(
                lambda: self.client.futures_income_history(**comm_params),
            )
            return rows, commissions
        except Exception as e:
            logger.warning(f"[{self.symbol}] 수익 내역 조회 실패: {e}")
            return [], []

    async def get_open_interest(self) -> float | None:
        """현재 미결제약정(OI)을 조회한다. 오류 시 None 반환."""
        try:
            result = await self._run_api(
                lambda: self.client.futures_open_interest(symbol=self.symbol),
            )
            return float(result["openInterest"])
        except Exception as e:
            logger.warning(f"OI 조회 실패 (무시): {e}")
            return None

    async def get_funding_rate(self) -> float | None:
        """현재 펀딩비를 조회한다. 오류 시 None 반환."""
        try:
            result = await self._run_api(
                lambda: self.client.futures_mark_price(symbol=self.symbol),
            )
            return float(result["lastFundingRate"])
        except Exception as e:
            logger.warning(f"펀딩비 조회 실패 (무시): {e}")
            return None

    async def get_oi_history(self, limit: int = 5) -> list[float]:
        """최근 OI 변화율 히스토리를 조회한다 (봇 초기화용). 실패 시 빈 리스트."""
        try:
            result = await self._run_api(
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
        return await self._run_api(self.client.futures_stream_get_listen_key)

    async def keepalive_listen_key(self, listen_key: str) -> None:
        """PUT /fapi/v1/listenKey — listenKey 만료 연장 (60분 → 리셋)"""
        await self._run_api(
            lambda: self.client.futures_stream_keepalive(listenKey=listen_key),
        )

    async def delete_listen_key(self, listen_key: str) -> None:
        """DELETE /fapi/v1/listenKey — listenKey 삭제 (정상 종료 시)"""
        try:
            await self._run_api(
                lambda: self.client.futures_stream_close(listenKey=listen_key),
            )
        except Exception as e:
            logger.warning(f"listenKey 삭제 실패 (무시): {e}")
