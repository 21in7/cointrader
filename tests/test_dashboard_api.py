import sys
import os
import sqlite3
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard", "api"))

# DB_PATH를 테스트용 임시 파일로 설정 (import 전에)
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DB_PATH"] = _tmp_db.name
_tmp_db.close()

import dashboard_api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def setup_db():
    """각 테스트 전에 DB를 초기화하고 테스트 데이터를 삽입."""
    db_path = os.environ["DB_PATH"]
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS trades;
        DROP TABLE IF EXISTS candles;
        DROP TABLE IF EXISTS daily_pnl;
        DROP TABLE IF EXISTS bot_status;
        DROP TABLE IF EXISTS parse_state;

        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL,
            leverage INTEGER DEFAULT 10,
            sl REAL, tp REAL,
            rsi REAL, macd_hist REAL, atr REAL, adx REAL,
            expected_pnl REAL, actual_pnl REAL,
            commission REAL, net_pnl REAL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            close_reason TEXT, extra TEXT
        );
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            price REAL NOT NULL,
            signal TEXT, adx REAL, oi REAL, oi_change REAL, funding_rate REAL,
            UNIQUE(symbol, ts)
        );
        CREATE TABLE daily_pnl (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            cumulative_pnl REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY(symbol, date)
        );
        CREATE TABLE bot_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        );
        CREATE TABLE parse_state (filepath TEXT PRIMARY KEY, position INTEGER DEFAULT 0);
    """)

    # 테스트 데이터
    conn.execute(
        "INSERT INTO trades(symbol,direction,entry_time,entry_price,quantity,status) VALUES(?,?,?,?,?,?)",
        ("XRPUSDT", "LONG", "2026-03-06 00:00:00", 2.30, 100.0, "OPEN"),
    )
    conn.execute(
        "INSERT INTO trades(symbol,direction,entry_time,entry_price,exit_time,exit_price,quantity,net_pnl,commission,status,close_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("TRXUSDT", "SHORT", "2026-03-05 12:00:00", 0.23, "2026-03-05 14:00:00", 0.22, 1000.0, 10.0, 0.1, "CLOSED", "TP"),
    )
    conn.execute("INSERT INTO bot_status(key,value,updated_at) VALUES(?,?,?)", ("XRPUSDT:last_start", "2026-03-06 00:00:00", "2026-03-06 00:00:00"))
    conn.execute("INSERT INTO bot_status(key,value,updated_at) VALUES(?,?,?)", ("TRXUSDT:last_start", "2026-03-06 00:00:00", "2026-03-06 00:00:00"))
    conn.execute("INSERT INTO bot_status(key,value,updated_at) VALUES(?,?,?)", ("XRPUSDT:current_price", "2.35", "2026-03-06 00:00:00"))
    conn.execute(
        "INSERT INTO candles(symbol,ts,price,signal) VALUES(?,?,?,?)",
        ("XRPUSDT", "2026-03-06 00:00:00", 2.35, "LONG"),
    )
    conn.execute(
        "INSERT INTO candles(symbol,ts,price,signal) VALUES(?,?,?,?)",
        ("TRXUSDT", "2026-03-06 00:00:00", 0.23, "SHORT"),
    )
    conn.commit()
    conn.close()
    yield


client = TestClient(dashboard_api.app)


def test_get_symbols():
    r = client.get("/api/symbols")
    assert r.status_code == 200
    data = r.json()
    assert set(data["symbols"]) == {"XRPUSDT", "TRXUSDT"}


def test_get_position_all():
    r = client.get("/api/position")
    assert r.status_code == 200
    data = r.json()
    assert len(data["positions"]) == 1
    assert data["positions"][0]["symbol"] == "XRPUSDT"


def test_get_position_by_symbol():
    r = client.get("/api/position?symbol=XRPUSDT")
    assert r.status_code == 200
    assert len(r.json()["positions"]) == 1


def test_get_trades_by_symbol():
    r = client.get("/api/trades?symbol=TRXUSDT")
    assert r.status_code == 200
    assert len(r.json()["trades"]) == 1
    assert r.json()["trades"][0]["symbol"] == "TRXUSDT"


def test_get_candles_by_symbol():
    r = client.get("/api/candles?symbol=XRPUSDT")
    assert r.status_code == 200
    assert len(r.json()["candles"]) == 1
    assert r.json()["candles"][0]["symbol"] == "XRPUSDT"


def test_get_stats_all():
    r = client.get("/api/stats")
    assert r.status_code == 200


def test_get_stats_by_symbol():
    r = client.get("/api/stats?symbol=TRXUSDT")
    assert r.status_code == 200
    assert r.json()["total_trades"] == 1
