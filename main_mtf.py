"""MTF Pullback Bot — OOS Dry-run Entry Point."""

import asyncio
import signal as sig
from loguru import logger
from src.mtf_bot import MTFPullbackBot
from src.logger_setup import setup_logger


async def main():
    setup_logger(log_level="INFO")
    logger.info("MTF Pullback Bot 시작 (Dry-run OOS 모드)")

    bot = MTFPullbackBot(symbol="XRP/USDT:USDT")

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    def _on_signal():
        logger.warning("종료 시그널 수신 (SIGTERM/SIGINT)")
        shutdown.set()

    for s in (sig.SIGTERM, sig.SIGINT):
        loop.add_signal_handler(s, _on_signal)

    bot_task = asyncio.create_task(bot.run(), name="mtf-bot")
    shutdown_task = asyncio.create_task(shutdown.wait(), name="shutdown-wait")

    done, pending = await asyncio.wait(
        [bot_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    bot_task.cancel()
    shutdown_task.cancel()
    await asyncio.gather(bot_task, shutdown_task, return_exceptions=True)
    logger.info("MTF Pullback Bot 종료")


if __name__ == "__main__":
    asyncio.run(main())
