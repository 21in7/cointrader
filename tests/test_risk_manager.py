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


# --- 동적 증거금 비율 테스트 ---

@pytest.fixture
def dynamic_config():
    c = Config()
    c.margin_max_ratio = 0.50
    c.margin_min_ratio = 0.20
    c.margin_decay_rate = 0.0006
    return c


@pytest.fixture
def risk(dynamic_config):
    r = RiskManager(dynamic_config)
    r.set_base_balance(22.0)
    return r


def test_set_base_balance(risk):
    assert risk.initial_balance == 22.0


def test_ratio_at_base_balance(risk):
    """기준 잔고에서 최대 비율(50%) 반환"""
    ratio = risk.get_dynamic_margin_ratio(22.0)
    assert ratio == pytest.approx(0.50, abs=1e-6)


def test_ratio_decreases_as_balance_grows(risk):
    """잔고가 늘수록 비율 감소"""
    ratio_100 = risk.get_dynamic_margin_ratio(100.0)
    ratio_300 = risk.get_dynamic_margin_ratio(300.0)
    assert ratio_100 < 0.50
    assert ratio_300 < ratio_100


def test_ratio_clamped_at_min(risk):
    """잔고가 매우 커도 최소 비율(20%) 이하로 내려가지 않음"""
    ratio = risk.get_dynamic_margin_ratio(10000.0)
    assert ratio == pytest.approx(0.20, abs=1e-6)


def test_ratio_clamped_at_max(risk):
    """잔고가 기준보다 작아도 최대 비율(50%) 초과하지 않음"""
    ratio = risk.get_dynamic_margin_ratio(5.0)
    assert ratio == pytest.approx(0.50, abs=1e-6)
