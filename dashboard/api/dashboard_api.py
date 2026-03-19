"""
dashboard_api.py — 멀티심볼 대시보드 API
"""

import sqlite3
import os
import signal
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "/app/data/dashboard.db")
PARSER_PID_FILE = os.environ.get("PARSER_PID_FILE", "/tmp/parser.pid")
DASHBOARD_RESET_KEY = os.environ.get("DASHBOARD_RESET_KEY", "")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",") if os.environ.get("CORS_ORIGINS") else ["*"]

app = FastAPI(title="Trading Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


@app.get("/api/symbols")
def get_symbols():
    """활성 심볼 목록 반환."""
    with get_db() as db:
        rows = db.execute(
            "SELECT DISTINCT key FROM bot_status WHERE key LIKE '%:%'"
        ).fetchall()
    symbols = {r["key"].split(":")[0] for r in rows}
    return {"symbols": sorted(symbols)}


@app.get("/api/position")
def get_position(symbol: Optional[str] = None):
    with get_db() as db:
        if symbol:
            rows = db.execute(
                "SELECT * FROM trades WHERE status='OPEN' AND symbol=? ORDER BY id DESC",
                (symbol,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC"
            ).fetchall()
        status_rows = db.execute("SELECT key, value FROM bot_status").fetchall()
        bot = {r["key"]: r["value"] for r in status_rows}
    return {"positions": [dict(r) for r in rows], "bot": bot}


@app.get("/api/trades")
def get_trades(
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with get_db() as db:
        if symbol:
            rows = db.execute(
                "SELECT * FROM trades WHERE status='CLOSED' AND symbol=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (symbol, limit, offset),
            ).fetchall()
            total = db.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status='CLOSED' AND symbol=?",
                (symbol,),
            ).fetchone()["cnt"]
        else:
            rows = db.execute(
                "SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status='CLOSED'").fetchone()["cnt"]
    return {"trades": [dict(r) for r in rows], "total": total}


@app.get("/api/daily")
def get_daily(symbol: Optional[str] = None, days: int = Query(30, ge=1, le=365)):
    with get_db() as db:
        if symbol:
            rows = db.execute("""
                SELECT date,
                    SUM(trade_count) as total_trades,
                    SUM(wins) as wins,
                    SUM(losses) as losses,
                    ROUND(SUM(cumulative_pnl), 4) as net_pnl
                FROM daily_pnl
                WHERE symbol=?
                GROUP BY date ORDER BY date DESC LIMIT ?
            """, (symbol, days)).fetchall()
        else:
            rows = db.execute("""
                SELECT date,
                    SUM(trade_count) as total_trades,
                    SUM(wins) as wins,
                    SUM(losses) as losses,
                    ROUND(SUM(cumulative_pnl), 4) as net_pnl
                FROM daily_pnl
                GROUP BY date ORDER BY date DESC LIMIT ?
            """, (days,)).fetchall()
    return {"daily": [dict(r) for r in rows]}


@app.get("/api/stats")
def get_stats(symbol: Optional[str] = None):
    with get_db() as db:
        if symbol:
            row = db.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                    COALESCE(SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
                    COALESCE(SUM(net_pnl), 0) as total_pnl,
                    COALESCE(SUM(commission), 0) as total_fees,
                    COALESCE(AVG(net_pnl), 0) as avg_pnl,
                    COALESCE(MAX(net_pnl), 0) as best_trade,
                    COALESCE(MIN(net_pnl), 0) as worst_trade
                FROM trades WHERE status='CLOSED' AND symbol=?
            """, (symbol,)).fetchone()
        else:
            row = db.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                    COALESCE(SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
                    COALESCE(SUM(net_pnl), 0) as total_pnl,
                    COALESCE(SUM(commission), 0) as total_fees,
                    COALESCE(AVG(net_pnl), 0) as avg_pnl,
                    COALESCE(MAX(net_pnl), 0) as best_trade,
                    COALESCE(MIN(net_pnl), 0) as worst_trade
                FROM trades WHERE status='CLOSED'
            """).fetchone()
        status_rows = db.execute("SELECT key, value FROM bot_status").fetchall()
        bot = {r["key"]: r["value"] for r in status_rows}
    result = dict(row)
    if symbol:
        result["current_price"] = bot.get(f"{symbol}:current_price")
    result["balance"] = bot.get("balance")
    return result


@app.get("/api/candles")
def get_candles(symbol: str = Query(...), limit: int = Query(96, ge=1, le=1000)):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM candles WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return {"candles": [dict(r) for r in reversed(rows)]}


@app.get("/api/health")
def health():
    try:
        with get_db() as db:
            cnt = db.execute("SELECT COUNT(*) as c FROM candles").fetchone()["c"]
        return {"status": "ok", "candles_count": cnt}
    except Exception:
        return {"status": "error", "detail": "database unavailable"}


@app.post("/api/reset")
def reset_db(x_api_key: Optional[str] = Header(None)):
    """DB 초기화 + 파서에 SIGHUP으로 재파싱 요청."""
    # C1: API key 인증 (DASHBOARD_RESET_KEY가 설정된 경우)
    if DASHBOARD_RESET_KEY and x_api_key != DASHBOARD_RESET_KEY:
        raise HTTPException(status_code=403, detail="invalid api key")

    with get_db() as db:
        for table in ["trades", "daily_pnl", "parse_state", "bot_status", "candles"]:
            db.execute(f"DELETE FROM {table}")
        db.commit()

    # C2: PID file + SIGHUP으로 파서에 재파싱 요청 (프로세스 재시작 불필요)
    try:
        with open(PARSER_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGHUP)
    except (FileNotFoundError, ValueError, ProcessLookupError, OSError):
        pass

    return {"status": "ok", "message": "DB 초기화 완료, 파서 재파싱 시작"}
