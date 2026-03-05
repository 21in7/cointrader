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

# ── 정규식 패턴 (실제 봇 로그 형식 기준) ──────────────────────────
PATTERNS = {
    # 신호: HOLD | 현재가: 1.3889 USDT
    "signal": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*신호: (?P<signal>\w+) \| 현재가: (?P<price>[\d.]+) USDT"
    ),

    # ADX: 24.4
    "adx": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*ADX: (?P<adx>[\d.]+)"
    ),

    # OI=261103765.6, OI변화율=0.000692, 펀딩비=0.000039
    "microstructure": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*OI=(?P<oi>[\d.]+), OI변화율=(?P<oi_change>[-\d.]+), 펀딩비=(?P<funding>[-\d.]+)"
    ),

    # 기존 포지션 복구: SHORT | 진입가=1.4126 | 수량=86.8
    "position_recover": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*기존 포지션 복구: (?P<direction>\w+) \| 진입가=(?P<entry_price>[\d.]+) \| 수량=(?P<qty>[\d.]+)"
    ),

    # SHORT 진입: 가격=1.3940, 수량=86.8, SL=1.4040, TP=1.3840
    "entry": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*(?P<direction>SHORT|LONG) 진입: "
        r"가격=(?P<entry_price>[\d.]+), "
        r"수량=(?P<qty>[\d.]+), "
        r"SL=(?P<sl>[\d.]+), "
        r"TP=(?P<tp>[\d.]+)"
    ),

    # 청산 감지(MANUAL): exit=1.3782, rp=+2.9859, commission=0.0598, net_pnl=+2.9261
    "close_detect": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*청산 감지\((?P<reason>\w+)\):\s*"
        r"exit=(?P<exit_price>[\d.]+),\s*"
        r"rp=(?P<expected>[+\-\d.]+),\s*"
        r"commission=(?P<commission>[\d.]+),\s*"
        r"net_pnl=(?P<net_pnl>[+\-\d.]+)"
    ),

    # 오늘 누적 PnL: 2.9261 USDT
    "daily_pnl": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*오늘 누적 PnL: (?P<pnl>[+\-\d.]+) USDT"
    ),

    # 봇 시작: XRPUSDT, 레버리지 10x
    "bot_start": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*봇 시작: (?P<symbol>\w+), 레버리지 (?P<leverage>\d+)x"
    ),

    # 기준 잔고 설정: 24.46 USDT
    "balance": re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r".*기준 잔고 설정: (?P<balance>[\d.]+) USDT"
    ),

    # ML 필터 로드
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

        # 상태 추적
        self._file_positions = {}          # {파일경로: 마지막 읽은 위치}
        self._current_position = None      # 현재 열린 포지션 정보
        self._pending_candle = {}          # 타임스탬프 기준으로 지표를 모아두기
        self._bot_config = {"symbol": "XRPUSDT", "leverage": 10}
        self._balance = 0

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL DEFAULT 'XRPUSDT',
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
                ts              TEXT    NOT NULL UNIQUE,
                price           REAL    NOT NULL,
                signal          TEXT,
                adx             REAL,
                oi              REAL,
                oi_change       REAL,
                funding_rate    REAL
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date            TEXT    PRIMARY KEY,
                cumulative_pnl  REAL    DEFAULT 0,
                trade_count     INTEGER DEFAULT 0,
                wins            INTEGER DEFAULT 0,
                losses          INTEGER DEFAULT 0,
                last_updated    TEXT
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

            CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """)
        self.conn.commit()
        self._load_state()

    def _load_state(self):
        """이전 파싱 위치 복원"""
        rows = self.conn.execute("SELECT filepath, position FROM parse_state").fetchall()
        self._file_positions = {r["filepath"]: r["position"] for r in rows}

        # 현재 열린 포지션 복원
        row = self.conn.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            self._current_position = dict(row)

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
        """로그 파일 목록을 가져와서 새 줄 파싱"""
        # 날짜 형식 (bot_2026-03-01.log) + 현재 형식 (bot.log) 모두 스캔
        log_files = sorted(glob.glob(os.path.join(LOG_DIR, "bot_*.log")))
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

        # 파일이 줄었으면 (로테이션) 처음부터
        if file_size < last_pos:
            last_pos = 0

        if file_size == last_pos:
            return  # 새 내용 없음

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
            self._bot_config["symbol"] = m.group("symbol")
            self._bot_config["leverage"] = int(m.group("leverage"))
            self._set_status("symbol", m.group("symbol"))
            self._set_status("leverage", m.group("leverage"))
            self._set_status("last_start", m.group("ts"))
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

        # 포지션 복구 (재시작 시)
        m = PATTERNS["position_recover"].search(line)
        if m:
            self._handle_entry(
                ts=m.group("ts"),
                direction=m.group("direction"),
                entry_price=float(m.group("entry_price")),
                qty=float(m.group("qty")),
                is_recovery=True,
            )
            return

        # 포지션 진입: SHORT 진입: 가격=X, 수량=Y, SL=Z, TP=W
        m = PATTERNS["entry"].search(line)
        if m:
            self._handle_entry(
                ts=m.group("ts"),
                direction=m.group("direction"),
                entry_price=float(m.group("entry_price")),
                qty=float(m.group("qty")),
                sl=float(m.group("sl")),
                tp=float(m.group("tp")),
            )
            return

        # OI/펀딩비 (캔들 데이터에 합침)
        m = PATTERNS["microstructure"].search(line)
        if m:
            ts_key = m.group("ts")[:16]  # 분 단위로 그룹
            if ts_key not in self._pending_candle:
                self._pending_candle[ts_key] = {}
            self._pending_candle[ts_key].update({
                "oi": float(m.group("oi")),
                "oi_change": float(m.group("oi_change")),
                "funding": float(m.group("funding")),
            })
            return

        # ADX
        m = PATTERNS["adx"].search(line)
        if m:
            ts_key = m.group("ts")[:16]
            if ts_key not in self._pending_candle:
                self._pending_candle[ts_key] = {}
            self._pending_candle[ts_key]["adx"] = float(m.group("adx"))
            return

        # 신호 + 현재가 → 캔들 저장
        m = PATTERNS["signal"].search(line)
        if m:
            ts = m.group("ts")
            ts_key = ts[:16]
            price = float(m.group("price"))
            signal = m.group("signal")
            extra = self._pending_candle.pop(ts_key, {})

            self._set_status("current_price", str(price))
            self._set_status("current_signal", signal)
            self._set_status("last_candle_time", ts)

            try:
                self.conn.execute(
                    """INSERT INTO candles(ts, price, signal, adx, oi, oi_change, funding_rate)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(ts) DO UPDATE SET
                         price=?, signal=?, adx=?, oi=?, oi_change=?, funding_rate=?""",
                    (ts, price, signal,
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
            ts = m.group("ts")
            day = ts[:10]
            pnl = float(m.group("pnl"))
            self.conn.execute(
                """INSERT INTO daily_pnl(date, cumulative_pnl, last_updated)
                   VALUES(?,?,?)
                   ON CONFLICT(date) DO UPDATE SET cumulative_pnl=?, last_updated=?""",
                (day, pnl, ts, pnl, ts)
            )
            self.conn.commit()
            self._set_status("daily_pnl", str(pnl))
            return

    # ── 포지션 진입 핸들러 ───────────────────────────────────────
    def _handle_entry(self, ts, direction, entry_price, qty,
                      leverage=None, sl=None, tp=None, is_recovery=False):
        if leverage is None:
            leverage = self._bot_config.get("leverage", 10)

        # 중복 체크 — 같은 방향의 OPEN 포지션이 이미 있으면 스킵
        # (봇은 동시에 같은 방향 포지션을 2개 이상 열지 않음)
        if self._current_position and self._current_position.get("direction") == direction:
            return

        existing = self.conn.execute(
            "SELECT id, entry_price FROM trades WHERE status='OPEN' AND direction=?",
            (direction,),
        ).fetchone()
        if existing:
            self._current_position = {
                "id": existing["id"],
                "direction": direction,
                "entry_price": existing["entry_price"],
                "entry_time": ts,
            }
            return

        cur = self.conn.execute(
            """INSERT INTO trades(symbol, direction, entry_time, entry_price,
               quantity, leverage, sl, tp, status, extra)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (self._bot_config.get("symbol", "XRPUSDT"), direction, ts,
             entry_price, qty, leverage, sl, tp, "OPEN",
             json.dumps({"recovery": is_recovery})),
        )
        self.conn.commit()
        self._current_position = {
            "id": cur.lastrowid,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": ts,
        }
        self._set_status("position_status", "OPEN")
        self._set_status("position_direction", direction)
        self._set_status("position_entry_price", str(entry_price))
        print(f"[LogParser] 포지션 진입: {direction} @ {entry_price} (recovery={is_recovery})")

    # ── 포지션 청산 핸들러 ───────────────────────────────────────
    def _handle_close(self, ts, exit_price, expected_pnl, commission, net_pnl, reason):
        # 모든 OPEN 거래를 닫음 (봇은 동시에 1개 포지션만 보유)
        open_trades = self.conn.execute(
            "SELECT id FROM trades WHERE status='OPEN' ORDER BY id DESC"
        ).fetchall()

        if not open_trades:
            print(f"[LogParser] 경고: 청산 감지했으나 열린 포지션 없음")
            return

        # 가장 최근 OPEN에 실제 PnL 기록
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

        # 나머지 OPEN 거래는 중복이므로 삭제
        if len(open_trades) > 1:
            stale_ids = [r["id"] for r in open_trades[1:]]
            self.conn.execute(
                f"DELETE FROM trades WHERE id IN ({','.join('?' * len(stale_ids))})",
                stale_ids,
            )
            print(f"[LogParser] 중복 OPEN 거래 {len(stale_ids)}건 삭제")

        # 일별 요약 갱신
        day = ts[:10]
        win = 1 if net_pnl > 0 else 0
        loss = 1 if net_pnl <= 0 else 0
        self.conn.execute(
            """INSERT INTO daily_pnl(date, cumulative_pnl, trade_count, wins, losses, last_updated)
               VALUES(?, ?, 1, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 trade_count = trade_count + 1,
                 wins = wins + ?,
                 losses = losses + ?,
                 last_updated = ?""",
            (day, net_pnl, win, loss, ts, win, loss, ts)
        )
        self.conn.commit()

        self._set_status("position_status", "NONE")
        print(f"[LogParser] 포지션 청산: {reason} @ {exit_price}, PnL={net_pnl}")
        self._current_position = None


if __name__ == "__main__":
    parser = LogParser()
    parser.run()
