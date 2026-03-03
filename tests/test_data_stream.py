import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.data_stream import KlineStream
from src.data_stream import MultiSymbolStream


def test_multi_symbol_stream_has_three_buffers():
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
    )
    assert "xrpusdt" in stream.buffers
    assert "btcusdt" in stream.buffers
    assert "ethusdt" in stream.buffers

def test_multi_symbol_stream_get_dataframe_returns_none_when_empty():
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
    )
    assert stream.get_dataframe("XRPUSDT") is None

def test_multi_symbol_stream_get_dataframe_returns_df_when_full():
    import pandas as pd
    from src.data_stream import _MIN_CANDLES_FOR_SIGNAL
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
        buffer_size=200,
    )
    candle = {
        "timestamp": 1000, "open": 1.0, "high": 1.1,
        "low": 0.9, "close": 1.05, "volume": 100.0, "is_closed": True,
    }
    for i in range(_MIN_CANDLES_FOR_SIGNAL):
        c = candle.copy()
        c["timestamp"] = 1000 + i
        stream.buffers["xrpusdt"].append(c)
    df = stream.get_dataframe("XRPUSDT")
    assert df is not None
    assert len(df) == _MIN_CANDLES_FOR_SIGNAL


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
    callback = AsyncMock()
    stream = KlineStream(
        symbol="XRPUSDT",
        interval="1m",
        on_candle=callback,
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
    await stream.handle_message(raw_msg)
    assert callback.call_count == 1


@pytest.mark.asyncio
async def test_multi_symbol_stream_updates_latest_price_on_every_message():
    """미종료 캔들 메시지도 primary symbol의 latest_price를 업데이트해야 한다."""
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="15m",
    )
    assert stream.latest_price is None

    # 미종료 캔들 메시지 (is_closed=False)
    msg = {
        "stream": "xrpusdt@kline_15m",
        "data": {
            "e": "kline",
            "s": "XRPUSDT",
            "k": {
                "t": 1700000000000, "o": "0.5", "h": "0.51",
                "l": "0.49", "c": "0.5050", "v": "100000", "x": False,
            },
        },
    }
    await stream.handle_message(msg)
    assert stream.latest_price == 0.5050
    # 미종료 캔들은 버퍼에 추가되지 않아야 한다
    assert len(stream.buffers["xrpusdt"]) == 0

    # BTC 메시지는 latest_price를 변경하지 않아야 한다
    btc_msg = {
        "stream": "btcusdt@kline_15m",
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1700000000000, "o": "60000", "h": "61000",
                "l": "59000", "c": "60500", "v": "500", "x": False,
            },
        },
    }
    await stream.handle_message(btc_msg)
    assert stream.latest_price == 0.5050  # 변경 없음


@pytest.mark.asyncio
async def test_preload_history_fills_buffer():
    stream = KlineStream(symbol="XRPUSDT", interval="1m", buffer_size=200)

    # REST API 응답 형식: [open_time, open, high, low, close, volume, ...]
    fake_klines = [
        [1700000000000 + i * 60000, "0.5", "0.51", "0.49", "0.505", "100000",
         0, "0", "0", "0", "0", "0"]
        for i in range(201)  # 201개 반환 → 마지막 1개 제외 → 200개 버퍼
    ]

    mock_client = AsyncMock()
    mock_client.futures_klines.return_value = fake_klines

    await stream._preload_history(mock_client, limit=200)

    assert len(stream.buffer) == 200
    assert stream.get_dataframe() is not None
