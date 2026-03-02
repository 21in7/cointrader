import asyncio
from typing import Callable
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

_KEEPALIVE_INTERVAL = 30 * 60   # 30분 (listenKey 만료 60분의 절반)
_RECONNECT_DELAY    = 5         # 재연결 대기 초

_CLOSE_ORDER_TYPES = {"TAKE_PROFIT_MARKET", "STOP_MARKET"}


class UserDataStream:
    """
    Binance Futures User Data Stream을 구독하여 주문 체결 이벤트를 처리한다.

    - listenKey 30분 keepalive 백그라운드 태스크
    - 네트워크 단절 시 무한 재연결 루프
    - ORDER_TRADE_UPDATE 이벤트에서 청산 주문만 필터링하여 콜백 호출
    """

    def __init__(
        self,
        exchange,                        # BinanceFuturesClient 인스턴스
        on_order_filled: Callable,       # bot._on_position_closed 콜백
    ):
        self._exchange = exchange
        self._on_order_filled = on_order_filled
        self._listen_key: str | None = None
        self._keepalive_task: asyncio.Task | None = None

    async def start(self, api_key: str, api_secret: str) -> None:
        """User Data Stream 메인 루프 — 봇 종료 시까지 실행."""
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
        bm = BinanceSocketManager(client)
        try:
            await self._run_loop(bm)
        finally:
            await client.close_connection()

    async def _run_loop(self, bm: BinanceSocketManager) -> None:
        """listenKey 발급 → 연결 → 재연결 무한 루프."""
        while True:
            try:
                self._listen_key = await self._exchange.create_listen_key()
                logger.info(f"User Data Stream listenKey 발급: {self._listen_key[:8]}...")

                self._keepalive_task = asyncio.create_task(
                    self._keepalive_loop(self._listen_key)
                )

                async with bm.futures_user_socket(self._listen_key) as stream:
                    logger.info("User Data Stream 연결 완료")
                    async for msg in stream:
                        await self._handle_message(msg)

            except asyncio.CancelledError:
                logger.info("User Data Stream 정상 종료")
                if self._listen_key:
                    await self._exchange.delete_listen_key(self._listen_key)
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                break

            except Exception as e:
                logger.warning(
                    f"User Data Stream 끊김: {e} — "
                    f"{_RECONNECT_DELAY}초 후 재연결"
                )
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                    self._keepalive_task = None
                await asyncio.sleep(_RECONNECT_DELAY)

    async def _keepalive_loop(self, listen_key: str) -> None:
        """30분마다 listenKey를 갱신한다."""
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            try:
                await self._exchange.keepalive_listen_key(listen_key)
                logger.debug("listenKey 갱신 완료")
            except Exception as e:
                logger.warning(f"listenKey 갱신 실패: {e} — 재연결 루프가 처리")
                break

    async def _handle_message(self, msg: dict) -> None:
        """ORDER_TRADE_UPDATE 이벤트에서 청산 주문을 필터링하여 콜백을 호출한다."""
        if msg.get("e") != "ORDER_TRADE_UPDATE":
            return

        order = msg.get("o", {})

        # x: Execution Type, X: Order Status
        if order.get("x") != "TRADE" or order.get("X") != "FILLED":
            return

        order_type   = order.get("o", "")
        is_reduce    = order.get("R", False)
        realized_pnl = float(order.get("rp", "0"))

        # 청산 주문 판별: reduceOnly이거나, TP/SL 타입이거나, rp != 0
        is_close = is_reduce or order_type in _CLOSE_ORDER_TYPES or realized_pnl != 0
        if not is_close:
            return

        commission = abs(float(order.get("n", "0")))
        net_pnl    = realized_pnl - commission
        exit_price = float(order.get("ap", "0"))

        if order_type == "TAKE_PROFIT_MARKET":
            close_reason = "TP"
        elif order_type == "STOP_MARKET":
            close_reason = "SL"
        else:
            close_reason = "MANUAL"

        logger.info(
            f"청산 감지({close_reason}): exit={exit_price:.4f}, "
            f"rp={realized_pnl:+.4f}, commission={commission:.4f}, "
            f"net_pnl={net_pnl:+.4f}"
        )

        await self._on_order_filled(
            net_pnl=net_pnl,
            close_reason=close_reason,
            exit_price=exit_price,
        )
