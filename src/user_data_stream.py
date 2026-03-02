import asyncio
from typing import Callable
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

_RECONNECT_DELAY = 5  # 재연결 대기 초

_CLOSE_ORDER_TYPES = {"TAKE_PROFIT_MARKET", "STOP_MARKET"}


class UserDataStream:
    """
    Binance Futures User Data Stream을 구독하여 주문 체결 이벤트를 처리한다.

    - python-binance BinanceSocketManager의 내장 keepalive 활용
    - 네트워크 단절 시 무한 재연결 루프
    - ORDER_TRADE_UPDATE 이벤트에서 지정 심볼의 청산 주문만 필터링하여 콜백 호출
    """

    def __init__(
        self,
        symbol: str,                     # 감시할 심볼 (예: "XRPUSDT")
        on_order_filled: Callable,       # bot._on_position_closed 콜백
    ):
        self._symbol = symbol.upper()
        self._on_order_filled = on_order_filled

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
        """연결 → 재연결 무한 루프. BinanceSocketManager가 listenKey keepalive를 내부 처리한다."""
        while True:
            try:
                async with bm.futures_user_socket() as stream:
                    logger.info(f"User Data Stream 연결 완료 (심볼 필터: {self._symbol})")
                    async for msg in stream:
                        await self._handle_message(msg)

            except asyncio.CancelledError:
                logger.info("User Data Stream 정상 종료")
                raise

            except Exception as e:
                logger.warning(
                    f"User Data Stream 끊김: {e} — "
                    f"{_RECONNECT_DELAY}초 후 재연결"
                )
                await asyncio.sleep(_RECONNECT_DELAY)

    async def _handle_message(self, msg: dict) -> None:
        """ORDER_TRADE_UPDATE 이벤트에서 청산 주문을 필터링하여 콜백을 호출한다."""
        if msg.get("e") != "ORDER_TRADE_UPDATE":
            return

        order = msg.get("o", {})

        # 심볼 필터링: 봇이 관리하는 심볼만 처리
        if order.get("s", "") != self._symbol:
            return

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
