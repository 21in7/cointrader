import asyncio
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
        risk.reset_daily()


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
    await asyncio.gather(
        *[bot.run() for bot in bots],
        _daily_reset_loop(risk),
    )


if __name__ == "__main__":
    asyncio.run(main())
