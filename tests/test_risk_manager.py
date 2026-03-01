import pytest
import os
from src.risk_manager import RiskManager
from src.config import Config


@pytest.fixture
def config():
    os.environ.update({
        "BINANCE_API_KEY": "k",
        "BINANCE_API_SECRET": "s",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
        "RISK_PER_TRADE": "0.02",
    })
    return Config()


def test_max_drawdown_check(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    rm.daily_pnl = -60.0
    rm.initial_balance = 1000.0
    assert rm.is_trading_allowed() is False


def test_trading_allowed_normal(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    rm.daily_pnl = -10.0
    rm.initial_balance = 1000.0
    assert rm.is_trading_allowed() is True


def test_position_size_capped(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    rm.open_positions = ["pos1", "pos2", "pos3"]
    assert rm.can_open_new_position() is False
