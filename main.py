import asyncio
from dotenv import load_dotenv
from src.config import Config
from src.bot import TradingBot
from src.logger_setup import setup_logger

load_dotenv()


async def main():
    setup_logger(log_level="INFO")
    config = Config()
    bot = TradingBot(config)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
