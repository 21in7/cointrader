"""
log_parser.py — 봇 로그 파일을 감시하고 파싱하여 SQLite에 저장
봇 코드 수정 없이 동작. logs/ 디렉토리만 마운트하면 됨.

실행: python log_parser.py
"""

import re
import sqlite3
import time
import glob
import os
import json
import threading
from datetime import datetime, date
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
DB_PATH = os.environ.get("DB_PATH", "/app/data/dashboard.db")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))  # 초

# ── 정규식 패턴 (멀티심볼 [SYMBOL] 프리픽스 포함) ─────────────────
PATTERNS = {
    "signal": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 신호: (?P<signal>\w+) \| 현재가: (?P<price>[\d.]+) USDT"
    ),

    "adx": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] ADX: (?P<adx>[\d.]+)"
    ),

    "microstructure": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] OI=(?P<oi>[\d.]+), OI변화율=(?P<oi_change>[-\d.]+), 펀딩비=(?P<funding>[-\d.]+)"
    ),

    "position_recover": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 기존 포지션 복구: (?P<direction>\w+) \| 진입가=(?P<entry_price>[\d.]+) \| 수량=(?P<qty>[\d.]+)"
    ),

    "entry": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] (?P<direction>SHORT|LONG) 진입: "
        r"가격=(?P<entry_price>[\d.]+), "
        r"수량=(?P<qty>[\d.]+), "
        r"SL=(?P<sl>[\d.]+), "
        r"TP=(?P<tp>[\d.]+)"
        r"(?:, RSI=(?P<rsi>[\d.]+))?"
        r"(?:, MACD_H=(?P<macd_hist>[+\-\d.]+))?"
        r"(?:, ATR=(?P<atr>[\d.]+))?"
    ),

    "close_detect": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 청산 감지\((?P<reason>\w+)\):\s*"
        r"exit=(?P<exit_price>[\d.]+),\s*"
        r"rp=(?P<expected>[+\-\d.]+),\s*"
        r"commission=(?P<commission>[\d.]+),\s*"
        r"net_pnl=(?P<net_pnl>[+\-\d.]+)"
    ),

    "daily_pnl": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 오늘 누적 PnL: (?P<pnl>[+\-\d.]+) USDT"
    ),

    "position_monitor": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 포지션 모니터 \| (?P<direction>\w+) \| "
        r"현재가=(?P<price>[\d.]+) \| PnL=(?P<pnl>[+\-\d.]+) USDT \((?P<pnl_pct>[+\-\d.]+)%\)"
    ),

    "bot_start": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 봇 시작, 레버리지 (?P<leverage>\d+)x"
    ),

    "balance": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*\[(?P<symbol>\w+)\] 기준 잔고 설정: (?P<balance>[\d.]+) USDT"
    ),

    "ml_filter": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*ML 필터 로드.*임계값=(?P<threshold>[\d.]+)"
    ),
}


class LogParser:
    def __init__(self):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

        self._file_positions = {}
        self._current_positions = {}       # {symbol: position_dict}
        self._pending_candles = {}         # {symbol: {ts_key: {data}}}
        self._balance = 0

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                direction       TEXT    NOT NULL,
                entry_time      TEXT    NOT NULL,
                exit_time       TEXT,
                entry_price     REAL    NOT NULL,
                exit_price      REAL,
                quantity         REAL,
                leverage        INTEGER DEFAULT 10,
                sl              REAL,
                tp              REAL,
                rsi             REAL,
                macd_hist       REAL,
                atr             REAL,
                adx             REAL,
                expected_pnl    REAL,
                actual_pnl      REAL,
                commission      REAL,
                net_pnl         REAL,
                status          TEXT    NOT NULL DEFAULT 'OPEN',
                close_reason    TEXT,
                extra           TEXT
            );

            CREATE TABLE IF NOT EXISTS candles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                ts              TEXT    NOT NULL,
                price           REAL    NOT NULL,
                signal          TEXT,
                adx             REAL,
                oi              REAL,
                oi_change       REAL,
                funding_rate    REAL,
                UNIQUE(symbol, ts)
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                symbol          TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                cumulative_pnl  REAL    DEFAULT 0,
                trade_count     INTEGER DEFAULT 0,
                wins            INTEGER DEFAULT 0,
                losses          INTEGER DEFAULT 0,
                last_updated    TEXT,
                PRIMARY KEY(symbol, date)
            );

            CREATE TABLE IF NOT EXISTS bot_status (
                key             TEXT    PRIMARY KEY,
                value           TEXT,
                updated_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS parse_state (
                filepath        TEXT    PRIMARY KEY,
                position        INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, ts);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        """)
        self.conn.commit()
        self._load_state()

    def _load_state(self):
        rows = self.conn.execute("SELECT filepath, position FROM parse_state").fetchall()
        self._file_positions = {r["filepath"]: r["position"] for r in rows}

        # 심볼별 열린 포지션 복원
        open_trades = self.conn.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC"
        ).fetchall()
        for row in open_trades:
            sym = row["symbol"]
            if sym not in self._current_positions:
                self._current_positions[sym] = dict(row)

    def _save_position(self, filepath, pos):
        self.conn.execute(
            "INSERT INTO parse_state(filepath, position) VALUES(?,?) "
            "ON CONFLICT(filepath) DO UPDATE SET position=?",
            (filepath, pos, pos)
        )
        self.conn.commit()

    def _set_status(self, key, value):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO bot_status(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
            (key, str(value), now, str(value), now)
        )
        self.conn.commit()

    # ── 메인 루프 ────────────────────────────────────────────────
    def run(self):
        print(f"[LogParser] 시작 — LOG_DIR={LOG_DIR}, DB={DB_PATH}, 폴링={POLL_INTERVAL}s")
        while True:
            try:
                self._scan_logs()
            except Exception as e:
                print(f"[LogParser] 에러: {e}")
            time.sleep(POLL_INTERVAL)

    def _scan_logs(self):
        log_files = sorted(glob.glob(os.path.join(LOG_DIR, "bot*.log")))
        main_log = os.path.join(LOG_DIR, "bot.log")
        if os.path.exists(main_log):
            log_files.append(main_log)
        for filepath in log_files:
            self._parse_file(filepath)

    def _parse_file(self, filepath):
        last_pos = self._file_positions.get(filepath, 0)

        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            return

        if file_size < last_pos:
            last_pos = 0

        if file_size == last_pos:
            return

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            new_pos = f.tell()

        for line in new_lines:
            self._parse_line(line.strip())

        self._file_positions[filepath] = new_pos
        self._save_position(filepath, new_pos)

    # ── 한 줄 파싱 ──────────────────────────────────────────────
    def _parse_line(self, line):
        if not line:
            return

        # 봇 시작
        m = PATTERNS["bot_start"].search(line)
        if m:
            symbol = m.group("symbol")
            self._set_status(f"{symbol}:leverage", m.group("leverage"))
            self._set_status(f"{symbol}:last_start", m.group("ts"))
            return

        # 잔고
        m = PATTERNS["balance"].search(line)
        if m:
            self._balance = float(m.group("balance"))
            self._set_status("balance", m.group("balance"))
            return

        # ML 필터
        m = PATTERNS["ml_filter"].search(line)
        if m:
            self._set_status("ml_threshold", m.group("threshold"))
            return

        # 포지션 모니터 (5분 간격 현재가·PnL 갱신)
        m = PATTERNS["position_monitor"].search(line)
        if m:
            symbol = m.group("symbol")
            self._set_status(f"{symbol}:current_price", m.group("price"))
            self._set_status(f"{symbol}:unrealized_pnl", m.group("pnl"))
            self._set_status(f"{symbol}:unrealized_pnl_pct", m.group("pnl_pct"))
            return

        # 포지션 복구 (재시작 시)
        m = PATTERNS["position_recover"].search(line)
        if m:
            self._handle_entry(
                ts=m.group("ts"),
                symbol=m.group("symbol"),
                direction=m.group("direction"),
                entry_price=float(m.group("entry_price")),
                qty=float(m.group("qty")),
                is_recovery=True,
            )
            return

        # 포지션 진입
        m = PATTERNS["entry"].search(line)
        if m:
            self._handle_entry(
                ts=m.group("ts"),
                symbol=m.group("symbol"),
                direction=m.group("direction"),
                entry_price=float(m.group("entry_price")),
                qty=float(m.group("qty")),
                sl=float(m.group("sl")),
                tp=float(m.group("tp")),
                rsi=float(m.group("rsi")) if m.group("rsi") else None,
                macd_hist=float(m.group("macd_hist")) if m.group("macd_hist") else None,
                atr=float(m.group("atr")) if m.group("atr") else None,
            )
            return

        # OI/펀딩비 (캔들 데이터에 합침)
        m = PATTERNS["microstructure"].search(line)
        if m:
            symbol = m.group("symbol")
            ts_key = m.group("ts")[:16]
            if symbol not in self._pending_candles:
                self._pending_candles[symbol] = {}
            if ts_key not in self._pending_candles[symbol]:
                self._pending_candles[symbol][ts_key] = {}
            self._pending_candles[symbol][ts_key].update({
                "oi": float(m.group("oi")),
                "oi_change": float(m.group("oi_change")),
                "funding": float(m.group("funding")),
            })
            return

        # ADX
        m = PATTERNS["adx"].search(line)
        if m:
            symbol = m.group("symbol")
            ts_key = m.group("ts")[:16]
            if symbol not in self._pending_candles:
                self._pending_candles[symbol] = {}
            if ts_key not in self._pending_candles[symbol]:
                self._pending_candles[symbol][ts_key] = {}
            self._pending_candles[symbol][ts_key]["adx"] = float(m.group("adx"))
            return

        # 신호 + 현재가 → 캔들 저장
        m = PATTERNS["signal"].search(line)
        if m:
            symbol = m.group("symbol")
            ts = m.group("ts")
            ts_key = ts[:16]
            price = float(m.group("price"))
            signal = m.group("signal")
            extra = self._pending_candles.get(symbol, {}).pop(ts_key, {})

            self._set_status(f"{symbol}:current_price", str(price))
            self._set_status(f"{symbol}:current_signal", signal)
            self._set_status(f"{symbol}:last_candle_time", ts)

            try:
                self.conn.execute(
                    """INSERT INTO candles(symbol, ts, price, signal, adx, oi, oi_change, funding_rate)
                       VALUES(?,?,?,?,?,?,?,?)
                       ON CONFLICT(symbol, ts) DO UPDATE SET
                         price=?, signal=?, adx=?, oi=?, oi_change=?, funding_rate=?""",
                    (symbol, ts, price, signal,
                     extra.get("adx"), extra.get("oi"), extra.get("oi_change"), extra.get("funding"),
                     price, signal,
                     extra.get("adx"), extra.get("oi"), extra.get("oi_change"), extra.get("funding")),
                )
                self.conn.commit()
            except Exception as e:
                print(f"[LogParser] 캔들 저장 에러: {e}")
            return

        # 청산 감지
        m = PATTERNS["close_detect"].search(line)
        if m:
            self._handle_close(
                ts=m.group("ts"),
                symbol=m.group("symbol"),
                exit_price=float(m.group("exit_price")),
                expected_pnl=float(m.group("expected")),
                commission=float(m.group("commission")),
                net_pnl=float(m.group("net_pnl")),
                reason=m.group("reason"),
            )
            return

        # 일일 누적 PnL
        m = PATTERNS["daily_pnl"].search(line)
        if m:
            symbol = m.group("symbol")
            ts = m.group("ts")
            day = ts[:10]
            pnl = float(m.group("pnl"))
            self.conn.execute(
                """INSERT INTO daily_pnl(symbol, date, cumulative_pnl, last_updated)
                   VALUES(?,?,?,?)
                   ON CONFLICT(symbol, date) DO UPDATE SET cumulative_pnl=?, last_updated=?""",
                (symbol, day, pnl, ts, pnl, ts)
            )
            self.conn.commit()
            self._set_status(f"{symbol}:daily_pnl", str(pnl))
            return

    # ── 포지션 진입 핸들러 ───────────────────────────────────────
    def _handle_entry(self, ts, symbol, direction, entry_price, qty,
                      leverage=None, sl=None, tp=None, is_recovery=False,
                      rsi=None, macd_hist=None, atr=None):
        if leverage is None:
            leverage = 10

        # 중복 체크 — 같은 심볼+방향의 OPEN 포지션이 이미 있으면 스킵
        current = self._current_positions.get(symbol)
        if current and current.get("direction") == direction:
            return

        existing = self.conn.execute(
            "SELECT id, entry_price FROM trades WHERE status='OPEN' AND symbol=? AND direction=?",
            (symbol, direction),
        ).fetchone()
        if existing:
            self._current_positions[symbol] = {
                "id": existing["id"],
                "direction": direction,
                "entry_price": existing["entry_price"],
                "entry_time": ts,
            }
            return

        cur = self.conn.execute(
            """INSERT INTO trades(symbol, direction, entry_time, entry_price,
               quantity, leverage, sl, tp, status, extra, rsi, macd_hist, atr)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, direction, ts,
             entry_price, qty, leverage, sl, tp, "OPEN",
             json.dumps({"recovery": is_recovery}),
             rsi, macd_hist, atr),
        )
        self.conn.commit()
        self._current_positions[symbol] = {
            "id": cur.lastrowid,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": ts,
        }
        self._set_status(f"{symbol}:position_status", "OPEN")
        self._set_status(f"{symbol}:position_direction", direction)
        self._set_status(f"{symbol}:position_entry_price", str(entry_price))
        print(f"[LogParser] {symbol} 포지션 진입: {direction} @ {entry_price} (recovery={is_recovery})")

    # ── 포지션 청산 핸들러 ───────────────────────────────────────
    def _handle_close(self, ts, symbol, exit_price, expected_pnl, commission, net_pnl, reason):
        # 해당 심볼의 OPEN 거래만 닫음
        open_trades = self.conn.execute(
            "SELECT id FROM trades WHERE status='OPEN' AND symbol=? ORDER BY id DESC",
            (symbol,),
        ).fetchall()

        if not open_trades:
            print(f"[LogParser] 경고: {symbol} 청산 감지했으나 열린 포지션 없음")
            return

        primary_id = open_trades[0]["id"]
        self.conn.execute(
            """UPDATE trades SET
               exit_time=?, exit_price=?, expected_pnl=?,
               actual_pnl=?, commission=?, net_pnl=?,
               status='CLOSED', close_reason=?
               WHERE id=?""",
            (ts, exit_price, expected_pnl,
             expected_pnl, commission, net_pnl,
             reason, primary_id)
        )

        if len(open_trades) > 1:
            stale_ids = [r["id"] for r in open_trades[1:]]
            self.conn.execute(
                f"DELETE FROM trades WHERE id IN ({','.join('?' * len(stale_ids))})",
                stale_ids,
            )
            print(f"[LogParser] {symbol} 중복 OPEN 거래 {len(stale_ids)}건 삭제")

        # 심볼별 일별 요약
        day = ts[:10]
        win = 1 if net_pnl > 0 else 0
        loss = 1 if net_pnl <= 0 else 0
        self.conn.execute(
            """INSERT INTO daily_pnl(symbol, date, cumulative_pnl, trade_count, wins, losses, last_updated)
               VALUES(?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(symbol, date) DO UPDATE SET
                 trade_count = trade_count + 1,
                 wins = wins + ?,
                 losses = losses + ?,
                 last_updated = ?""",
            (symbol, day, net_pnl, win, loss, ts, win, loss, ts)
        )
        self.conn.commit()

        self._set_status(f"{symbol}:position_status", "NONE")
        print(f"[LogParser] {symbol} 포지션 청산: {reason} @ {exit_price}, PnL={net_pnl}")
        self._current_positions.pop(symbol, None)


if __name__ == "__main__":
    parser = LogParser()
    parser.run()
