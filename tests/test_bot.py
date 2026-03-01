import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np
import os
from src.bot import TradingBot
from src.config import Config


@pytest.fixture
def config():
    os.environ.update({
        "BINANCE_API_KEY": "k",
        "BINANCE_API_SECRET": "s",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
        "RISK_PER_TRADE": "0.02",
        "NOTION_TOKEN": "secret_test",
        "NOTION_DATABASE_ID": "db_test",
    })
    return Config()


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 0.5
    return pd.DataFrame({
        "open":   close,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    })


@pytest.mark.asyncio
async def test_bot_processes_signal(config, sample_df):
    with patch("src.bot.BinanceFuturesClient") as MockExchange:
        MockExchange.return_value = AsyncMock()
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_balance = AsyncMock(return_value=1000.0)
    bot.exchange.get_position = AsyncMock(return_value=None)
    bot.exchange.place_order = AsyncMock(return_value={"orderId": "123"})
    bot.exchange.set_leverage = AsyncMock(return_value={})
    bot.exchange.calculate_quantity = MagicMock(return_value=100.0)
    bot.exchange.MIN_NOTIONAL = 5.0

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = "LONG"
        mock_ind.get_atr_stop.return_value = (0.48, 0.56)
        MockInd.return_value = mock_ind
        await bot.process_candle(sample_df)
