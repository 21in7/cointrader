import pytest
from unittest.mock import MagicMock, patch
from src.database import TradeRepository


@pytest.fixture
def mock_repo():
    with patch("src.database.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        repo = TradeRepository(token="secret_test", database_id="db_test")
        repo.client = mock_client
        yield repo


def test_save_trade(mock_repo):
    mock_repo.client.pages.create.return_value = {
        "id": "abc123",
        "properties": {},
    }
    result = mock_repo.save_trade(
        symbol="XRPUSDT",
        side="LONG",
        entry_price=0.5,
        quantity=400.0,
        leverage=10,
        signal_data={"rsi": 32, "macd_hist": 0.001},
    )
    assert result["id"] == "abc123"


def test_close_trade(mock_repo):
    mock_repo.client.pages.update.return_value = {
        "id": "abc123",
        "properties": {
            "Status": {"select": {"name": "CLOSED"}},
        },
    }
    result = mock_repo.close_trade(
        trade_id="abc123", exit_price=0.55, pnl=20.0
    )
    assert result["id"] == "abc123"
