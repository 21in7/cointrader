import asyncio
from typing import Callable
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

_RECONNECT_DELAY = 5  # 재연결 대기 초

_CLOSE_ORDER_TYPES = {"TAKE_PROFIT_MARKET", "STOP_MARKET"}


class UserDataStream:
    """
    Binance Futures User Data Stream을 구독하여 주문 체결 이벤트를 처리한다.

    - 매 재연결마다 AsyncClient + BinanceSocketManager를 새로 생성 (listenKey 무효화 대응)
    - 네트워크 단절 시 무한 재연결 루프
    - ORDER_TRADE_UPDATE 이벤트에서 지정 심볼의 청산 주문만 필터링하여 콜백 호출
    - 부분 체결(PARTIALLY_FILLED) 시 rp/commission을 누적하여 최종 FILLED에서 합산 콜백
    """

    def __init__(
        self,
        symbol: str,                     # 감시할 심볼 (예: "XRPUSDT")
        on_order_filled: Callable,       # bot._on_position_closed 콜백
    ):
        self._symbol = symbol.upper()
        self._on_order_filled = on_order_filled
        # 부분 체결 누적용: order_id → {rp, commission}
        self._partial_fills: dict[int, dict[str, float]] = {}

    async def start(self, api_key: str, api_secret: str) -> None:
        """User Data Stream 메인 루프 — 봇 종료 시까지 실행."""
        await self._run_loop(api_key, api_secret)

    async def _run_loop(self, api_key: str, api_secret: str) -> None:
        """연결 → 재연결 무한 루프.

        매 재연결마다 AsyncClient + BinanceSocketManager를 새로 생성한다.
        keepalive ping timeout 후 기존 BinanceSocketManager의 listenKey가
        무효화되면 재사용 시 이벤트를 수신하지 못하는 "조용한 실패"가 발생하므로,
        반드시 새 인스턴스를 만들어야 한다.
        """
        while True:
            client = await AsyncClient.create(
                api_key=api_key,
                api_secret=api_secret,
            )
            try:
                bm = BinanceSocketManager(client)
                async with bm.futures_user_socket() as stream:
                    logger.info(f"User Data Stream 연결 완료 (심볼 필터: {self._symbol})")
                    while True:
                        msg = await stream.recv()

                        if isinstance(msg, dict) and msg.get("e") == "error":
                            logger.warning(
                                f"웹소켓 내부 에러 수신: {msg.get('m', msg)} — "
                                f"재연결을 위해 연결 종료"
                            )
                            break

                        await self._handle_message(msg)

            except asyncio.CancelledError:
                logger.info("User Data Stream 정상 종료")
                try:
                    await client.close_connection()
                except Exception:
                    pass
                raise

            except Exception as e:
                logger.warning(
                    f"User Data Stream 끊김: {e} — "
                    f"{_RECONNECT_DELAY}초 후 재연결"
                )
            finally:
                try:
                    await client.close_connection()
                except Exception:
                    pass

            await asyncio.sleep(_RECONNECT_DELAY)

    async def _handle_message(self, msg: dict) -> None:
        """ORDER_TRADE_UPDATE 이벤트에서 청산 주문을 필터링하여 콜백을 호출한다."""
        if msg.get("e") != "ORDER_TRADE_UPDATE":
            return

        order = msg.get("o", {})

        # 심볼 필터링: 봇이 관리하는 심볼만 처리
        if order.get("s", "") != self._symbol:
            return

        # x: Execution Type — TRADE만 처리
        if order.get("x") != "TRADE":
            return

        order_status = order.get("X", "")
        order_type   = order.get("o", "")
        is_reduce    = order.get("R", False)
        order_id     = order.get("i", 0)

        # 청산 주문 판별: reduceOnly이거나 TP/SL 타입
        is_close = is_reduce or order_type in _CLOSE_ORDER_TYPES
        if not is_close:
            return

        fill_rp         = float(order.get("rp", "0"))
        fill_commission  = abs(float(order.get("n", "0")))

        if order_status == "PARTIALLY_FILLED":
            # 부분 체결: rp와 commission을 누적
            if order_id not in self._partial_fills:
                self._partial_fills[order_id] = {"rp": 0.0, "commission": 0.0}
            self._partial_fills[order_id]["rp"] += fill_rp
            self._partial_fills[order_id]["commission"] += fill_commission
            logger.debug(
                f"[{self._symbol}] 부분 체결 누적 (order_id={order_id}): "
                f"rp={fill_rp:+.4f}, commission={fill_commission:.4f}"
            )
            return

        if order_status != "FILLED":
            return

        # 최종 체결: 이전 부분 체결분 합산
        accumulated = self._partial_fills.pop(order_id, {"rp": 0.0, "commission": 0.0})
        realized_pnl = accumulated["rp"] + fill_rp
        commission   = accumulated["commission"] + fill_commission

        net_pnl    = realized_pnl - commission
        exit_price = float(order.get("ap", "0"))

        if exit_price == 0.0:
            logger.warning(
                f"[{self._symbol}] 청산 이벤트에서 exit_price=0.0 — "
                f"ap 필드 누락 가능. 청산 처리 스킵 (rp={realized_pnl:+.4f})"
            )
            return

        if order_type == "TAKE_PROFIT_MARKET":
            close_reason = "TP"
        elif order_type == "STOP_MARKET":
            close_reason = "SL"
        else:
            close_reason = "MANUAL"

        logger.info(
            f"[{self._symbol}] 청산 감지({close_reason}): exit={exit_price:.4f}, "
            f"rp={realized_pnl:+.4f}, commission={commission:.4f}, "
            f"net_pnl={net_pnl:+.4f}"
        )

        await self._on_order_filled(
            net_pnl=net_pnl,
            close_reason=close_reason,
            exit_price=exit_price,
        )
