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
        "RISK_PER_TRADE": "0.02",
    })
    return Config()


@pytest.mark.asyncio
async def test_set_leverage(config):
    client = BinanceFuturesClient(config)
    with patch.object(
        client.client,
        "futures_change_leverage",
        return_value={"leverage": 10},
    ):
        result = await client.set_leverage(10)
        assert result is not None


def test_calculate_quantity(config):
    client = BinanceFuturesClient(config)
    # 잔고 1000 USDT, 리스크 2%, 레버리지 10, 가격 0.5
    qty = client.calculate_quantity(balance=1000.0, price=0.5, leverage=10)
    # 1000 * 0.02 * 10 / 0.5 = 400
    assert qty == pytest.approx(400.0, rel=0.01)
