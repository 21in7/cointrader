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


@pytest.mark.asyncio
async def test_position_size_capped(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    await rm.register_position("XRPUSDT", "LONG")
    await rm.register_position("TRXUSDT", "SHORT")
    await rm.register_position("DOGEUSDT", "LONG")
    assert await rm.can_open_new_position("SOLUSDT", "SHORT") is False


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


# --- 멀티심볼 공유 RiskManager 테스트 ---

@pytest.fixture
def shared_risk(config):
    config.max_same_direction = 2
    return RiskManager(config)


@pytest.mark.asyncio
async def test_can_open_new_position_async(shared_risk):
    """비동기 포지션 오픈 허용 체크."""
    assert await shared_risk.can_open_new_position("XRPUSDT", "LONG") is True


@pytest.mark.asyncio
async def test_register_and_close_position(shared_risk):
    """포지션 등록 후 닫기."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    assert "XRPUSDT" in shared_risk.open_positions
    await shared_risk.close_position("XRPUSDT", pnl=1.5)
    assert "XRPUSDT" not in shared_risk.open_positions
    assert shared_risk.daily_pnl == 1.5


@pytest.mark.asyncio
async def test_same_symbol_blocked(shared_risk):
    """같은 심볼 중복 진입 차단."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    assert await shared_risk.can_open_new_position("XRPUSDT", "SHORT") is False


@pytest.mark.asyncio
async def test_max_same_direction_limit(shared_risk):
    """같은 방향 2개 초과 차단."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    await shared_risk.register_position("TRXUSDT", "LONG")
    # 3번째 LONG 차단
    assert await shared_risk.can_open_new_position("DOGEUSDT", "LONG") is False
    # SHORT은 허용
    assert await shared_risk.can_open_new_position("DOGEUSDT", "SHORT") is True


@pytest.mark.asyncio
async def test_max_positions_global_limit(shared_risk):
    """전체 포지션 수 한도 초과 차단."""
    shared_risk.config.max_positions = 2
    await shared_risk.register_position("XRPUSDT", "LONG")
    await shared_risk.register_position("TRXUSDT", "SHORT")
    assert await shared_risk.can_open_new_position("DOGEUSDT", "LONG") is False
