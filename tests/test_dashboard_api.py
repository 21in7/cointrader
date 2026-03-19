import sys
import os
import sqlite3
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard", "api"))

# DB_PATH와 DASHBOARD_RESET_KEY를 테스트용으로 설정 (import 전에)
_tmp_dir = tempfile.mkdtemp()
_tmp_db_path = os.path.join(_tmp_dir, "test_dashboard.db")
os.environ["DB_PATH"] = _tmp_db_path
os.environ["DASHBOARD_RESET_KEY"] = "test-reset-key"

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
    conn.execute(
        "INSERT INTO daily_pnl(symbol,date,cumulative_pnl,trade_count,wins,losses) VALUES(?,?,?,?,?,?)",
        ("TRXUSDT", "2026-03-05", 10.0, 1, 1, 0),
    )
    conn.commit()
    conn.close()
    yield
    # cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


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
    data = r.json()
    assert len(data["trades"]) == 1
    assert data["trades"][0]["symbol"] == "TRXUSDT"
    assert data["total"] == 1


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


# ── M6: 누락된 테스트 추가 ──────────────────────────────────────

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["candles_count"] >= 0


def test_daily():
    r = client.get("/api/daily?symbol=TRXUSDT")
    assert r.status_code == 200
    data = r.json()
    assert len(data["daily"]) == 1
    assert data["daily"][0]["net_pnl"] == 10.0


def test_daily_all():
    r = client.get("/api/daily")
    assert r.status_code == 200
    assert "daily" in r.json()


def test_reset_requires_api_key():
    """C1: API key 없이 reset 호출 시 403."""
    r = client.post("/api/reset")
    assert r.status_code == 403


def test_reset_wrong_api_key():
    """C1: 잘못된 API key로 reset 호출 시 403."""
    r = client.post("/api/reset", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 403


def test_reset_with_valid_key():
    """C1+C2: 올바른 API key로 reset 호출 시 성공."""
    r = client.post("/api/reset", headers={"X-API-Key": "test-reset-key"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # DB가 비워졌는지 확인
    r2 = client.get("/api/trades")
    assert r2.json()["total"] == 0


def test_trades_offset_validation():
    """I2: 음수 offset은 422 에러."""
    r = client.get("/api/trades?offset=-1")
    assert r.status_code == 422


def test_trades_pagination():
    """M6: 페이지네이션 동작 확인."""
    r = client.get("/api/trades?limit=1&offset=0")
    assert r.status_code == 200
    data = r.json()
    assert len(data["trades"]) <= 1
    assert "total" in data


def test_health_error_no_detail_leak():
    """I6: health에서 에러 시 내부 경로 미노출."""
    # 일시적으로 DB 경로를 존재하지 않는 곳으로 설정
    original = dashboard_api.DB_PATH
    dashboard_api.DB_PATH = "/nonexistent/path/db.sqlite"
    r = client.get("/api/health")
    dashboard_api.DB_PATH = original
    data = r.json()
    assert data["status"] == "error"
    assert "/nonexistent" not in data.get("detail", "")
