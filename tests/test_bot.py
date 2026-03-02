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


def test_bot_uses_multi_symbol_stream(config):
    from src.data_stream import MultiSymbolStream
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    assert isinstance(bot.stream, MultiSymbolStream)

def test_bot_stream_has_btc_eth_buffers(config):
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    assert "btcusdt" in bot.stream.buffers
    assert "ethusdt" in bot.stream.buffers


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


@pytest.mark.asyncio
async def test_close_and_reenter_calls_open_when_ml_passes(config, sample_df):
    """반대 시그널 + ML 필터 통과 시 청산 후 재진입해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.risk = MagicMock()
    bot.risk.can_open_new_position.return_value = True
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = True

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_awaited_once_with("SHORT", sample_df)


@pytest.mark.asyncio
async def test_close_and_reenter_skips_open_when_ml_blocks(config, sample_df):
    """ML 필터 차단 시 청산만 하고 재진입하지 않아야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = False

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_not_called()


@pytest.mark.asyncio
async def test_close_and_reenter_skips_open_when_max_positions_reached(config, sample_df):
    """최대 포지션 수 도달 시 청산만 하고 재진입하지 않아야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.risk = MagicMock()
    bot.risk.can_open_new_position.return_value = False

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_not_called()


@pytest.mark.asyncio
async def test_process_candle_calls_close_and_reenter_on_reverse_signal(config, sample_df):
    """반대 시그널 시 process_candle이 _close_and_reenter를 호출해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_position = AsyncMock(return_value={
        "positionAmt": "100",
        "entryPrice": "0.5",
        "markPrice": "0.52",
    })
    bot._close_and_reenter = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = False
    bot.ml_filter.should_enter.return_value = True

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = "SHORT"  # 현재 LONG 포지션에 반대 시그널
        MockInd.return_value = mock_ind
        await bot.process_candle(sample_df)

    bot._close_and_reenter.assert_awaited_once()
    call_args = bot._close_and_reenter.call_args
    assert call_args.args[1] == "SHORT"


@pytest.mark.asyncio
async def test_process_candle_passes_raw_signal_to_close_and_reenter_even_if_ml_loaded(config, sample_df):
    """포지션 보유 시 ML 필터가 로드되어 있어도 process_candle은 raw signal을 _close_and_reenter에 전달한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_position = AsyncMock(return_value={
        "positionAmt": "100",
        "entryPrice": "0.5",
        "markPrice": "0.52",
    })
    bot._close_and_reenter = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True  # 모델 로드됨
    bot.ml_filter.should_enter.return_value = False    # ML이 차단하더라도

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = "SHORT"
        MockInd.return_value = mock_ind
        await bot.process_candle(sample_df)

    # ML 필터가 차단해도 _close_and_reenter는 호출되어야 한다 (ML 재평가는 내부에서)
    bot._close_and_reenter.assert_awaited_once()
    call_args = bot._close_and_reenter.call_args
    assert call_args.args[1] == "SHORT"
    # process_candle에서 ml_filter.should_enter가 호출되지 않아야 한다
    bot.ml_filter.should_enter.assert_not_called()


@pytest.mark.asyncio
async def test_process_candle_fetches_oi_and_funding(config, sample_df):
    """process_candle()이 OI와 펀딩비를 조회하고 build_features에 전달하는지 확인."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_balance = AsyncMock(return_value=1000.0)
    bot.exchange.get_position = AsyncMock(return_value=None)
    bot.exchange.place_order = AsyncMock(return_value={"orderId": "1"})
    bot.exchange.set_leverage = AsyncMock()
    bot.exchange.get_open_interest = AsyncMock(return_value=5000000.0)
    bot.exchange.get_funding_rate = AsyncMock(return_value=0.0001)

    with patch("src.bot.build_features") as mock_build:
        from src.ml_features import FEATURE_COLS
        mock_build.return_value = pd.Series({col: 0.0 for col in FEATURE_COLS})
        bot.ml_filter.is_model_loaded = MagicMock(return_value=False)
        await bot.process_candle(sample_df)

    assert mock_build.called
    call_kwargs = mock_build.call_args.kwargs
    assert "oi_change" in call_kwargs
    assert "funding_rate" in call_kwargs
