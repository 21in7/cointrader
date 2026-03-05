import asyncio
from dotenv import load_dotenv
from loguru import logger
from src.config import Config
from src.bot import TradingBot
from src.risk_manager import RiskManager
from src.logger_setup import setup_logger

load_dotenv()


async def main():
    setup_logger(log_level="INFO")
    config = Config()
    risk = RiskManager(config)

    bots = []
    for symbol in config.symbols:
        bot = TradingBot(config, symbol=symbol, risk=risk)
        bots.append(bot)

    logger.info(f"멀티심볼 봇 시작: {config.symbols} ({len(bots)}개 인스턴스)")
    await asyncio.gather(*[bot.run() for bot in bots])


if __name__ == "__main__":
    asyncio.run(main())
