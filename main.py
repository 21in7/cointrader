import asyncio
import signal
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from loguru import logger
from src.config import Config
from src.bot import TradingBot
from src.risk_manager import RiskManager
from src.logger_setup import setup_logger

load_dotenv()


async def _daily_reset_loop(risk: RiskManager):
    """매일 UTC 자정에 daily_pnl을 초기화한다."""
    while True:
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        await asyncio.sleep((next_midnight - now).total_seconds())
        await risk.reset_daily()


async def _graceful_shutdown(bots: list[TradingBot], tasks: list[asyncio.Task]):
    """모든 봇의 오픈 주문 취소 후 태스크를 정리한다."""
    logger.info("Graceful shutdown 시작 — 오픈 주문 취소 중...")
    for bot in bots:
        try:
            await asyncio.wait_for(bot.exchange.cancel_all_orders(), timeout=5)
            logger.info(f"[{bot.symbol}] 오픈 주문 취소 완료")
        except Exception as e:
            logger.warning(f"[{bot.symbol}] 오픈 주문 취소 실패 (무시): {e}")

    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
            logger.warning(f"태스크 종료 중 예외: {r}")
    logger.info("Graceful shutdown 완료")


async def main():
    setup_logger(log_level="INFO")
    config = Config()
    risk = RiskManager(config)

    # 기준 잔고를 main에서 한 번만 설정 (경쟁 조건 방지)
    from src.exchange import BinanceFuturesClient
    exchange = BinanceFuturesClient(config, symbol=config.symbols[0])
    balance = await exchange.get_balance()
    risk.set_base_balance(balance)
    logger.info(f"기준 잔고 설정: {balance:.2f} USDT")

    bots = []
    for symbol in config.symbols:
        bot = TradingBot(config, symbol=symbol, risk=risk)
        bots.append(bot)

    logger.info(f"멀티심볼 봇 시작: {config.symbols} ({len(bots)}개 인스턴스)")

    # 시그널 핸들러 등록
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.warning("종료 시그널 수신 (SIGTERM/SIGINT)")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    tasks = [
        asyncio.create_task(bot.run(), name=f"bot-{bot.symbol}")
        for bot in bots
    ]
    tasks.append(asyncio.create_task(_daily_reset_loop(risk), name="daily-reset"))

    # 종료 시그널 대기 vs 태스크 완료 (먼저 발생하는 쪽)
    shutdown_task = asyncio.create_task(shutdown_event.wait(), name="shutdown-wait")
    done, pending = await asyncio.wait(
        tasks + [shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 시그널이든 태스크 종료든 graceful shutdown 수행
    shutdown_task.cancel()
    await _graceful_shutdown(bots, tasks)


if __name__ == "__main__":
    asyncio.run(main())
