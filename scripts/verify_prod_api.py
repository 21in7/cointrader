"""실전 API SL/TP 콜백 검증 스크립트.

검증 항목:
  1. SL/TP 주문 응답에 orderId vs algoId 확인
  2. SL 트리거 시 UDS 콜백의 o, ot 필드 값
  3. futures_cancel_order(orderId=...)로 TP 취소 가능 여부

사용법:
  1. 바이낸스 앱/웹에서 XRPUSDT 소액 LONG 포지션 수동 진입
  2. python scripts/verify_prod_api.py 실행
     → 자동으로 SL/TP 배치 + UDS 리스닝
  3. SL이 트리거되면 콜백 로그 확인 + TP 자동 취소 시도

환경변수: BINANCE_API_KEY, BINANCE_API_SECRET (실전 키)
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

from src.exchange import BinanceFuturesClient
from src.config import Config

# .env에서 실전 키 로드 (BINANCE_TESTNET이 설정되어 있으면 해제)
os.environ.pop("BINANCE_TESTNET", None)
load_dotenv()

SYMBOL = "XRPUSDT"


async def main():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error("BINANCE_API_KEY / BINANCE_API_SECRET 환경변수 필요")
        return

    # Exchange 클라이언트 (실전)
    config = Config()
    config.testnet = False
    config.api_key = api_key
    config.api_secret = api_secret
    config.symbol = SYMBOL

    exchange = BinanceFuturesClient(config, symbol=SYMBOL)

    # ── Step 1: 현재 포지션 확인 ──
    position = await exchange.get_position()
    if position is None:
        logger.error(
            f"[{SYMBOL}] 포지션 없음. 먼저 바이낸스 앱/웹에서 소액 포지션을 수동으로 진입하세요."
        )
        return

    pos_amt = float(position["positionAmt"])
    entry_price = float(position["entryPrice"])
    mark_price = float(position.get("markPrice", entry_price))
    side = "LONG" if pos_amt > 0 else "SHORT"
    quantity = abs(pos_amt)

    logger.info(f"[{SYMBOL}] 포지션 확인: {side} qty={quantity}, entry={entry_price}, mark={mark_price}")

    # ── Step 2: 기존 오픈 주문 확인/정리 ──
    open_orders = await exchange.get_open_orders()
    if open_orders:
        logger.info(f"[{SYMBOL}] 기존 오픈 주문 {len(open_orders)}개 — 전체 취소")
        await exchange.cancel_all_orders()
        await asyncio.sleep(1)

    # ── Step 3: SL/TP 주문 배치 (현재가 기준 가까운 값) ──
    # SL: 현재가에서 0.15% 떨어진 곳 (빨리 트리거되도록)
    # TP: 현재가에서 2% 떨어진 곳 (트리거 안 되도록)
    sl_side = "SELL" if side == "LONG" else "BUY"

    if side == "LONG":
        stop_loss = exchange._round_price(mark_price * 0.9985)   # -0.15%
        take_profit = exchange._round_price(mark_price * 1.02)   # +2%
    else:
        stop_loss = exchange._round_price(mark_price * 1.0015)   # +0.15%
        take_profit = exchange._round_price(mark_price * 0.98)   # -2%

    logger.info(f"[{SYMBOL}] SL/TP 배치 예정: SL={stop_loss}, TP={take_profit}, side={sl_side}")

    # SL 배치
    sl_result = await exchange.place_order(
        side=sl_side,
        quantity=quantity,
        order_type="STOP_MARKET",
        stop_price=stop_loss,
        reduce_only=True,
    )
    logger.success(f"[검증1] SL 주문 응답 전체:\n{json.dumps(sl_result, indent=2)}")
    sl_order_id = sl_result.get("orderId")
    sl_algo_id = sl_result.get("algoId")
    logger.info(f"  → orderId={sl_order_id}, algoId={sl_algo_id}")

    # TP 배치
    tp_result = await exchange.place_order(
        side=sl_side,
        quantity=quantity,
        order_type="TAKE_PROFIT_MARKET",
        stop_price=take_profit,
        reduce_only=True,
    )
    logger.success(f"[검증1] TP 주문 응답 전체:\n{json.dumps(tp_result, indent=2)}")
    tp_order_id = tp_result.get("orderId")
    tp_algo_id = tp_result.get("algoId")
    logger.info(f"  → orderId={tp_order_id}, algoId={tp_algo_id}")

    # ── Step 4: UDS 리스닝 — SL 트리거 대기 ──
    logger.info(f"[{SYMBOL}] UDS 리스닝 시작 — SL 트리거 대기 중 (mark={mark_price}, SL={stop_loss})")
    logger.info("  SL이 트리거되면 자동으로 TP 취소를 시도합니다.")
    logger.info("  Ctrl+C로 중단 가능 (중단 시 잔여 주문 정리)")

    sl_triggered = asyncio.Event()

    async def on_uds_message(msg: dict):
        if msg.get("e") != "ORDER_TRADE_UPDATE":
            return

        order = msg.get("o", {})
        if order.get("s") != SYMBOL:
            return

        # 모든 이벤트 원본 로깅
        logger.info(
            f"[검증2] UDS 원본: "
            f"s={order.get('s')} "
            f"o={order.get('o')} "
            f"ot={order.get('ot')} "
            f"x={order.get('x')} "
            f"X={order.get('X')} "
            f"R={order.get('R')} "
            f"S={order.get('S')} "
            f"i={order.get('i')} "
            f"ap={order.get('ap')} "
            f"rp={order.get('rp')} "
            f"n={order.get('n')}"
        )

        # FILLED된 SL 감지
        if order.get("x") == "TRADE" and order.get("X") == "FILLED":
            ot = order.get("ot", "")
            if ot == "STOP_MARKET":
                logger.success(
                    f"[검증2] SL FILLED 확인! "
                    f"o={order.get('o')}, ot={ot}, "
                    f"orderId={order.get('i')}, "
                    f"exit_price={order.get('ap')}, rp={order.get('rp')}"
                )
                sl_triggered.set()

    # UDS 연결
    client = await AsyncClient.create(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        bm = BinanceSocketManager(client)
        async with bm.futures_user_socket() as stream:
            logger.info("UDS 연결 완료")

            while True:
                try:
                    msg = await asyncio.wait_for(stream.recv(), timeout=1.0)
                    await on_uds_message(msg)
                except asyncio.TimeoutError:
                    pass

                if sl_triggered.is_set():
                    break

        # ── Step 5: TP 취소 검증 ──
        cancel_id = tp_order_id or tp_algo_id
        logger.info(f"[검증3] TP 취소 시도: futures_cancel_order(orderId={cancel_id})")

        try:
            cancel_result = await exchange.cancel_order(cancel_id)
            logger.success(f"[검증3] TP 취소 성공:\n{json.dumps(cancel_result, indent=2)}")
        except Exception as e:
            logger.error(f"[검증3] TP 취소 실패: {e}")

            # cancel_all_orders 폴백
            logger.info("[검증3] cancel_all_orders 폴백 시도")
            try:
                fallback_result = await exchange.cancel_all_orders()
                logger.success(f"[검증3] cancel_all_orders 결과: {fallback_result}")
            except Exception as e2:
                logger.error(f"[검증3] cancel_all_orders도 실패: {e2}")

        # 최종 오픈 주문 확인
        remaining = await exchange.get_open_orders()
        if remaining:
            logger.warning(f"[검증3] 잔여 오픈 주문 {len(remaining)}개:")
            for o in remaining:
                logger.warning(f"  id={o.get('orderId')}, type={o.get('type')}, status={o.get('status')}")
        else:
            logger.success("[검증3] 잔여 오픈 주문 없음 — 고아주문 없음 확인!")

    except KeyboardInterrupt:
        logger.info("중단 — 잔여 주문 정리 중...")
        try:
            await exchange.cancel_all_orders()
            logger.info("잔여 주문 전체 취소 완료")
        except Exception as e:
            logger.warning(f"잔여 주문 취소 실패: {e}")
    finally:
        await client.close_connection()

    # ── 결과 요약 ──
    logger.info("=" * 60)
    logger.info("검증 결과 요약")
    logger.info("=" * 60)
    logger.info(f"[1] SL orderId={sl_order_id}, algoId={sl_algo_id}")
    logger.info(f"[1] TP orderId={tp_order_id}, algoId={tp_algo_id}")
    logger.info(f"[2] SL 트리거 감지: {'YES' if sl_triggered.is_set() else 'NO (타임아웃/중단)'}")
    logger.info(f"[3] 위 로그에서 TP 취소 성공 여부 확인")


if __name__ == "__main__":
    asyncio.run(main())
