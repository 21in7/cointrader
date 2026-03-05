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
    c.symbol = config.symbol
    return c


@pytest.fixture
def exchange():
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
    })
    config = Config()
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.config = config
    c.symbol = config.symbol
    c.client = MagicMock()
    return c


def test_exchange_uses_own_symbol():
    """Exchange 클라이언트가 config.symbol 대신 생성자의 symbol을 사용한다."""
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SYMBOL": "XRPUSDT",
    })
    config = Config()
    with patch("src.exchange.Client"):
        client = BinanceFuturesClient(config, symbol="TRXUSDT")
    assert client.symbol == "TRXUSDT"


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


@pytest.mark.asyncio
async def test_get_open_interest(exchange):
    """get_open_interest()가 float을 반환하는지 확인."""
    exchange.client.futures_open_interest = MagicMock(
        return_value={"openInterest": "123456.789"}
    )
    result = await exchange.get_open_interest()
    assert isinstance(result, float)
    assert result == pytest.approx(123456.789)


@pytest.mark.asyncio
async def test_get_funding_rate(exchange):
    """get_funding_rate()가 float을 반환하는지 확인."""
    exchange.client.futures_mark_price = MagicMock(
        return_value={"lastFundingRate": "0.0001"}
    )
    result = await exchange.get_funding_rate()
    assert isinstance(result, float)
    assert result == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_get_open_interest_error_returns_none(exchange):
    """API 오류 시 None 반환 확인."""
    from binance.exceptions import BinanceAPIException
    exchange.client.futures_open_interest = MagicMock(
        side_effect=BinanceAPIException(MagicMock(status_code=400), 400, '{"code":-1121,"msg":"Invalid symbol"}')
    )
    result = await exchange.get_open_interest()
    assert result is None


@pytest.mark.asyncio
async def test_get_funding_rate_error_returns_none(exchange):
    """API 오류 시 None 반환 확인."""
    from binance.exceptions import BinanceAPIException
    exchange.client.futures_mark_price = MagicMock(
        side_effect=BinanceAPIException(MagicMock(status_code=400), 400, '{"code":-1121,"msg":"Invalid symbol"}')
    )
    result = await exchange.get_funding_rate()
    assert result is None


@pytest.mark.asyncio
async def test_get_oi_history_returns_changes(exchange):
    """get_oi_history()가 OI 변화율 리스트를 반환하는지 확인."""
    exchange.client.futures_open_interest_hist = MagicMock(
        return_value=[
            {"sumOpenInterest": "1000000"},
            {"sumOpenInterest": "1010000"},
            {"sumOpenInterest": "1005000"},
            {"sumOpenInterest": "1020000"},
            {"sumOpenInterest": "1015000"},
            {"sumOpenInterest": "1030000"},
        ]
    )
    result = await exchange.get_oi_history(limit=5)
    assert len(result) == 5
    assert isinstance(result[0], float)
    # 첫 번째 변화율: (1010000 - 1000000) / 1000000 = 0.01
    assert abs(result[0] - 0.01) < 1e-6


@pytest.mark.asyncio
async def test_get_oi_history_error_returns_empty(exchange):
    """API 오류 시 빈 리스트 반환 확인."""
    exchange.client.futures_open_interest_hist = MagicMock(
        side_effect=Exception("API error")
    )
    result = await exchange.get_oi_history(limit=5)
    assert result == []


@pytest.mark.asyncio
async def test_get_oi_history_insufficient_data_returns_empty(exchange):
    """데이터가 부족하면 빈 리스트 반환 확인."""
    exchange.client.futures_open_interest_hist = MagicMock(
        return_value=[{"sumOpenInterest": "1000000"}]
    )
    result = await exchange.get_oi_history(limit=5)
    assert result == []
