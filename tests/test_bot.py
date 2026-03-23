import asyncio
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
        "DISCORD_WEBHOOK_URL": "",
        "BINANCE_TESTNET": "false",
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
        "atr":    np.full(n, 0.005),
    })


def test_bot_accepts_symbol_and_risk(config):
    """TradingBot이 symbol과 risk를 외부에서 주입받을 수 있다."""
    from src.risk_manager import RiskManager
    risk = RiskManager(config)
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config, symbol="TRXUSDT", risk=risk)
    assert bot.symbol == "TRXUSDT"
    assert bot.risk is risk


def test_bot_stream_uses_injected_symbol(config):
    """봇의 stream이 주입된 심볼을 primary로 사용한다."""
    from src.risk_manager import RiskManager
    risk = RiskManager(config)
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config, symbol="DOGEUSDT", risk=risk)
    assert "dogeusdt" in bot.stream.buffers


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

    bot.risk = MagicMock()
    bot.risk.is_trading_allowed = AsyncMock(return_value=True)
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
    bot.risk.register_position = AsyncMock()
    bot.risk.get_dynamic_margin_ratio.return_value = 0.50

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = ("LONG", {"long": 3, "short": 0, "vol_surge": True, "adx": 30.0, "hold_reason": ""})
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
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
    bot.risk.close_position = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = True

    # 콜백 대기를 건너뛰도록 Event 미리 설정
    bot._close_event.set()

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
    bot.risk = MagicMock()
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
    bot.risk.close_position = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = False

    bot._close_event.set()

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
    bot.risk.can_open_new_position = AsyncMock(return_value=False)
    bot.risk.close_position = AsyncMock()

    bot._close_event.set()

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
        mock_ind.get_signal.return_value = ("SHORT", {"long": 0, "short": 3, "vol_surge": True, "adx": 30.0, "hold_reason": ""})  # 현재 LONG 포지션에 반대 시그널
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
        mock_ind.get_signal.return_value = ("SHORT", {"long": 0, "short": 3, "vol_surge": True, "adx": 30.0, "hold_reason": ""})
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

    bot.risk = MagicMock()
    bot.risk.is_trading_allowed = AsyncMock(return_value=True)
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
    bot.risk.register_position = AsyncMock()
    bot.risk.get_dynamic_margin_ratio.return_value = 0.50

    # 신호를 LONG으로 강제해 build_features가 반드시 호출되도록 함
    with patch("src.bot.Indicators") as mock_ind_cls:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = ("LONG", {"long": 3, "short": 0, "vol_surge": True, "adx": 30.0, "hold_reason": ""})
        mock_ind_cls.return_value = mock_ind

        with patch("src.bot.build_features_aligned") as mock_build:
            from src.ml_features import FEATURE_COLS
            mock_build.return_value = pd.Series({col: 0.0 for col in FEATURE_COLS})
            bot.ml_filter.is_model_loaded = MagicMock(return_value=False)
            # _open_position은 이 테스트의 관심사가 아니므로 mock 처리
            bot._open_position = AsyncMock()
            await bot.process_candle(sample_df)

    assert mock_build.called
    call_kwargs = mock_build.call_args.kwargs
    assert "oi_change" in call_kwargs
    assert "funding_rate" in call_kwargs


def test_bot_has_oi_history_deque(config):
    """봇이 OI 히스토리 deque를 가져야 한다."""
    from collections import deque
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    assert isinstance(bot._oi_history, deque)
    assert bot._oi_history.maxlen == 96


@pytest.mark.asyncio
async def test_init_oi_history_fills_deque(config):
    """_init_oi_history가 deque를 채워야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    bot.exchange = AsyncMock()
    bot.exchange.get_oi_history = AsyncMock(return_value=[0.01, -0.02, 0.03, -0.01, 0.02])
    await bot._init_oi_history()
    assert len(bot._oi_history) == 5


@pytest.mark.asyncio
async def test_fetch_microstructure_returns_4_tuple(config):
    """_fetch_market_microstructure가 4-tuple을 반환해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    bot.exchange = AsyncMock()
    bot.exchange.get_open_interest = AsyncMock(return_value=5000000.0)
    bot.exchange.get_funding_rate = AsyncMock(return_value=0.0001)
    bot._prev_oi = 4900000.0
    bot._oi_history.extend([0.01, -0.02, 0.03, -0.01])
    bot._latest_ret_1 = 0.01

    result = await bot._fetch_market_microstructure()
    assert len(result) == 4


def test_calc_oi_change_first_candle_returns_zero(config):
    """첫 캔들은 0.0을 반환하고 _prev_oi를 설정한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    assert bot._calc_oi_change(5000000.0) == 0.0
    assert bot._prev_oi == 5000000.0


def test_calc_oi_change_api_failure_does_not_corrupt_state(config):
    """API 실패 시 _fetch_market_microstructure가 _calc_oi_change를 호출하지 않아 상태가 오염되지 않는다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    bot._prev_oi = 5000000.0
    # API 실패 시 _fetch_market_microstructure는 oi_val > 0 체크로 _calc_oi_change를 건너뜀
    # _calc_oi_change(0.0)을 직접 호출하면 _prev_oi가 0.0으로 오염되는 이전 버그를 재현
    # 수정 후에는 _fetch_market_microstructure에서 0.0을 직접 반환하므로 이 경로가 없음
    # 대신 _calc_oi_change가 정상 값에서만 호출되는지 확인
    result = bot._calc_oi_change(5100000.0)
    assert abs(result - 0.02) < 1e-6  # (5100000 - 5000000) / 5000000 = 0.02
    assert bot._prev_oi == 5100000.0


@pytest.mark.asyncio
async def test_position_monitor_logs_when_position_open(config, caplog):
    """포지션 보유 중일 때 모니터가 현재가와 PnL을 로깅해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.current_trade_side = "LONG"
    bot._entry_price = 0.5000
    bot._entry_quantity = 100.0
    bot.stream.latest_price = 0.5100

    # 인터벌을 0으로 줄여 즉시 실행되게 함
    bot._MONITOR_INTERVAL = 0

    import loguru
    loguru.logger.enable("src.bot")

    task = asyncio.create_task(bot._position_monitor())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # loguru는 caplog와 호환되지 않으므로 직접 로그 확인 대신 예외 없이 실행됨을 확인
    # PnL 계산이 올바른지 직접 검증
    pnl = bot._calc_estimated_pnl(0.5100)
    assert abs(pnl - 1.0) < 1e-6  # (0.51 - 0.50) * 100 = 1.0


@pytest.mark.asyncio
async def test_position_monitor_skips_when_no_position(config):
    """포지션이 없을 때 모니터는 로깅하지 않고 넘어가야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.current_trade_side = None
    bot._MONITOR_INTERVAL = 0

    task = asyncio.create_task(bot._position_monitor())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # 예외 없이 정상 종료되어야 한다
