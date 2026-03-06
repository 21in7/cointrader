import sys
import os
import sqlite3
import tempfile
import pytest

# dashboard/api를 import path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard", "api"))


@pytest.fixture
def parser():
    """임시 DB로 LogParser 인스턴스 생성."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    import log_parser as lp
    lp.DB_PATH = db_path
    p = lp.LogParser()
    yield p
    p.conn.close()
    os.unlink(db_path)


def test_parse_signal_with_symbol(parser):
    """[SYMBOL] 프리픽스가 있는 신호 로그를 파싱한다."""
    line = "2026-03-06 00:15:00 | INFO | [XRPUSDT] 신호: LONG | 현재가: 2.3456 USDT"
    parser._parse_line(line)
    row = parser.conn.execute("SELECT * FROM candles WHERE symbol='XRPUSDT'").fetchone()
    assert row is not None
    assert row["price"] == 2.3456
    assert row["signal"] == "LONG"


def test_parse_entry_with_symbol(parser):
    """[SYMBOL] 프리픽스가 있는 진입 로그를 파싱한다."""
    line = (
        "2026-03-06 00:15:00 | SUCCESS | [TRXUSDT] SHORT 진입: "
        "가격=0.2345, 수량=1000.0, SL=0.2380, TP=0.2240, "
        "RSI=72.31, MACD_H=-0.001234, ATR=0.005678"
    )
    parser._parse_line(line)
    row = parser.conn.execute("SELECT * FROM trades WHERE symbol='TRXUSDT'").fetchone()
    assert row is not None
    assert row["direction"] == "SHORT"
    assert row["entry_price"] == 0.2345


def test_parse_close_with_symbol(parser):
    """[SYMBOL] 프리픽스가 있는 청산 로그를 심볼별로 처리한다."""
    # 먼저 두 심볼의 포지션을 열어놓음
    entry1 = "2026-03-06 00:00:00 | SUCCESS | [XRPUSDT] LONG 진입: 가격=2.3000, 수량=100.0, SL=2.2600, TP=2.4000"
    entry2 = "2026-03-06 00:00:00 | SUCCESS | [TRXUSDT] SHORT 진입: 가격=0.2345, 수량=1000.0, SL=0.2380, TP=0.2240"
    parser._parse_line(entry1)
    parser._parse_line(entry2)

    # XRPUSDT만 청산
    close_line = (
        "2026-03-06 01:00:00 | INFO | [XRPUSDT] 청산 감지(TP): "
        "exit=2.4000, rp=+10.0000, commission=0.1000, net_pnl=+9.9000"
    )
    parser._parse_line(close_line)

    # XRPUSDT는 CLOSED, TRXUSDT는 여전히 OPEN
    xrp = parser.conn.execute("SELECT status FROM trades WHERE symbol='XRPUSDT'").fetchone()
    trx = parser.conn.execute("SELECT status FROM trades WHERE symbol='TRXUSDT'").fetchone()
    assert xrp["status"] == "CLOSED"
    assert trx["status"] == "OPEN"


def test_parse_bot_start_multi_symbol(parser):
    """멀티심볼 봇 시작 로그를 각각 파싱한다."""
    lines = [
        "2026-03-06 00:04:54 | INFO | [XRPUSDT] 봇 시작, 레버리지 10x",
        "2026-03-06 00:04:54 | INFO | [TRXUSDT] 봇 시작, 레버리지 10x",
        "2026-03-06 00:04:54 | INFO | [DOGEUSDT] 봇 시작, 레버리지 10x",
    ]
    for line in lines:
        parser._parse_line(line)

    symbols = parser.conn.execute(
        "SELECT value FROM bot_status WHERE key LIKE '%:last_start'"
    ).fetchall()
    assert len(symbols) == 3


def test_candles_table_has_symbol_column(parser):
    """candles 테이블에 symbol 컬럼이 있어야 한다."""
    info = parser.conn.execute("PRAGMA table_info(candles)").fetchall()
    col_names = [row[1] for row in info]
    assert "symbol" in col_names


def test_daily_pnl_table_has_symbol_column(parser):
    """daily_pnl 테이블에 symbol 컬럼이 있어야 한다."""
    info = parser.conn.execute("PRAGMA table_info(daily_pnl)").fetchall()
    col_names = [row[1] for row in info]
    assert "symbol" in col_names


def test_balance_log_with_symbol(parser):
    """[SYMBOL] 프리픽스가 있는 잔고 로그를 파싱한다."""
    line = "2026-03-06 00:04:54 | INFO | [XRPUSDT] 기준 잔고 설정: 44.81 USDT (동적 증거금 비율 기준점)"
    parser._parse_line(line)
    row = parser.conn.execute("SELECT value FROM bot_status WHERE key='balance'").fetchone()
    assert row is not None
    assert row["value"] == "44.81"


def test_position_recover_with_symbol(parser):
    """[SYMBOL] 프리픽스가 있는 포지션 복구 로그를 파싱한다."""
    line = "2026-03-06 00:04:54 | INFO | [DOGEUSDT] 기존 포지션 복구: LONG | 진입가=0.1800 | 수량=500.0"
    parser._parse_line(line)
    row = parser.conn.execute("SELECT * FROM trades WHERE symbol='DOGEUSDT'").fetchone()
    assert row is not None
    assert row["direction"] == "LONG"
    assert row["entry_price"] == 0.1800
