import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.data_stream import KlineStream


@pytest.mark.asyncio
async def test_kline_stream_parses_message():
    stream = KlineStream(symbol="XRPUSDT", interval="1m")
    raw_msg = {
        "k": {
            "t": 1700000000000,
            "o": "0.5000",
            "h": "0.5100",
            "l": "0.4900",
            "c": "0.5050",
            "v": "100000",
            "x": True,
        }
    }
    candle = stream.parse_kline(raw_msg)
    assert candle["close"] == 0.5050
    assert candle["is_closed"] is True


@pytest.mark.asyncio
async def test_callback_called_on_closed_candle():
    received = []
    stream = KlineStream(
        symbol="XRPUSDT",
        interval="1m",
        on_candle=lambda c: received.append(c),
    )
    raw_msg = {
        "k": {
            "t": 1700000000000,
            "o": "0.5",
            "h": "0.51",
            "l": "0.49",
            "c": "0.505",
            "v": "100000",
            "x": True,
        }
    }
    stream.handle_message(raw_msg)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_preload_history_fills_buffer():
    stream = KlineStream(symbol="XRPUSDT", interval="1m", buffer_size=200)

    # REST API 응답 형식: [open_time, open, high, low, close, volume, ...]
    fake_klines = [
        [1700000000000 + i * 60000, "0.5", "0.51", "0.49", "0.505", "100000",
         0, "0", "0", "0", "0", "0"]
        for i in range(201)  # 201개 반환 → 마지막 1개 제외 → 200개 버퍼
    ]

    mock_client = MagicMock()
    mock_client.futures_klines.return_value = fake_klines

    await stream._preload_history(mock_client, limit=200)

    assert len(stream.buffer) == 200
    assert stream.get_dataframe() is not None
