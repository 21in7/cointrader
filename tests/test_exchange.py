import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.exchange import BinanceFuturesClient
from src.config import Config
import os


@pytest.fixture
def config():
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
    })
    return Config()


@pytest.fixture
def client():
    config = Config()
    config.leverage = 10
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.config = config
    return c


@pytest.mark.asyncio
async def test_set_leverage(config):
    with patch("src.exchange.Client") as MockClient:
        mock_binance = MagicMock()
        MockClient.return_value = mock_binance
        mock_binance.futures_change_leverage.return_value = {"leverage": 10}
        client = BinanceFuturesClient(config)
        result = await client.set_leverage(10)
        assert result is not None


def test_calculate_quantity_basic(client):
    """잔고 22, 비율 50%, 레버리지 10배 → 명목금액 110, XRP 가격 2.5 → 수량 44.0"""
    qty = client.calculate_quantity(balance=22.0, price=2.5, leverage=10, margin_ratio=0.50)
    # 명목금액 = 22 * 0.5 * 10 = 110, 수량 = 110 / 2.5 = 44.0
    assert qty == pytest.approx(44.0, abs=0.1)


def test_calculate_quantity_min_notional(client):
    """명목금액이 최소(5 USDT) 미만이면 최소값으로 올림"""
    qty = client.calculate_quantity(balance=1.0, price=2.5, leverage=1, margin_ratio=0.01)
    # 명목금액 = 1 * 0.01 * 1 = 0.01 < 5 → 최소 5 USDT
    assert qty * 2.5 >= 5.0


def test_calculate_quantity_zero_balance(client):
    """잔고 0이면 최소 명목금액 기반 수량 반환"""
    qty = client.calculate_quantity(balance=0.0, price=2.5, leverage=10, margin_ratio=0.50)
    assert qty > 0
