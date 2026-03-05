"""
dashboard_api.py — 로그 파서가 채운 SQLite DB를 읽어서 대시보드 API 제공
"""

import sqlite3
import os
import signal
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "/app/data/dashboard.db")

app = FastAPI(title="Trading Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@app.get("/api/position")
def get_position():
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        status_rows = db.execute("SELECT key, value FROM bot_status").fetchall()
        bot = {r["key"]: r["value"] for r in status_rows}
    return {"position": dict(row) if row else None, "bot": bot}

@app.get("/api/trades")
def get_trades(limit: int = Query(50, ge=1, le=500), offset: int = 0):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status='CLOSED'").fetchone()["cnt"]
    return {"trades": [dict(r) for r in rows], "total": total}

@app.get("/api/daily")
def get_daily(days: int = Query(30, ge=1, le=365)):
    with get_db() as db:
        rows = db.execute("""
            SELECT
                date(exit_time) as date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(net_pnl), 4) as net_pnl,
                ROUND(SUM(commission), 4) as total_fees
            FROM trades
            WHERE status='CLOSED' AND exit_time IS NOT NULL
            GROUP BY date(exit_time)
            ORDER BY date DESC
            LIMIT ?
        """, (days,)).fetchall()
    return {"daily": [dict(r) for r in rows]}

@app.get("/api/stats")
def get_stats():
    with get_db() as db:
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
    result["current_price"] = bot.get("current_price")
    result["balance"] = bot.get("balance")
    return result

@app.get("/api/candles")
def get_candles(limit: int = Query(96, ge=1, le=1000)):
    with get_db() as db:
        rows = db.execute("SELECT * FROM candles ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return {"candles": [dict(r) for r in reversed(rows)]}

@app.get("/api/health")
def health():
    try:
        with get_db() as db:
            cnt = db.execute("SELECT COUNT(*) as c FROM candles").fetchone()["c"]
        return {"status": "ok", "candles_count": cnt}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/reset")
def reset_db():
    """DB 전체 초기화 후 파서 재시작 (로그를 처음부터 다시 파싱)"""
    with get_db() as db:
        for table in ["trades", "daily_pnl", "parse_state", "bot_status", "candles"]:
            db.execute(f"DELETE FROM {table}")
        db.commit()

    # 파서 프로세스 재시작 (entrypoint.sh의 백그라운드 프로세스)
    import subprocess, os, signal
    # 기존 파서 종료 (pkill 대신 Python-native 방식)
    for pid_str in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid_str.isdigit():
            continue
        try:
            with open(f"/proc/{pid_str}/cmdline", "r") as f:
                cmdline = f.read()
            if "log_parser.py" in cmdline and str(os.getpid()) != pid_str:
                os.kill(int(pid_str), signal.SIGTERM)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass
    # 새 파서 시작
    subprocess.Popen(["python", "log_parser.py"])

    return {"status": "ok", "message": "DB 초기화 완료, 파서 재시작됨"}


