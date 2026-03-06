# Multi-Symbol Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 대시보드(파서/API/UI)를 멀티심볼(XRP, TRX, DOGE) 동시 지원으로 업그레이드

**Architecture:** 봇 로그에 `[SYMBOL]` 프리픽스 일관 추가 → 파서가 심볼별 상태 추적 → DB에 symbol 컬럼 추가 → API에 symbol 쿼리 파라미터 → UI에 심볼 필터 탭

**Tech Stack:** Python (loguru, FastAPI, SQLite), React (recharts), 기존 스택 유지

**Design Doc:** `docs/plans/2026-03-06-multi-symbol-dashboard-design.md`

---

## Task 1: 봇 로그에 `[SYMBOL]` 프리픽스 일관 추가

**Files:**
- Modify: `src/bot.py` (로그 메시지에 `[{self.symbol}]` 추가)
- Modify: `src/user_data_stream.py` (청산 감지 로그에 심볼 추가)
- Modify: `tests/test_bot.py` (기존 테스트가 깨지지 않는지 확인)

**Step 1: `src/bot.py` 로그 메시지 수정**

아래 로그 라인들에 `[{self.symbol}]` 프리픽스 추가 (이미 있는 것은 그대로):

```python
# line 67: 포지션 복구
logger.info(
    f"[{self.symbol}] 기존 포지션 복구: {self.current_trade_side} | "
    f"진입가={entry:.4f} | 수량={abs(amt)}"
)

# line 75: 포지션 없음
logger.info(f"[{self.symbol}] 기존 포지션 없음 - 신규 진입 대기")

# line 85: OI 히스토리
logger.info(f"[{self.symbol}] OI 히스토리 초기화: {len(self._oi_history)}개")

# line 109: OI/펀딩비 debug 로그
logger.debug(
    f"[{self.symbol}] OI={oi_val}, OI변화율={oi_change:.6f}, 펀딩비={fr_float:.6f}, "
    f"OI_MA5={oi_ma5:.6f}, OI_Price_Spread={oi_price_spread:.6f}"
)

# line 137: 리스크 한도
logger.warning(f"[{self.symbol}] 리스크 한도 초과 - 거래 중단")

# line 145: 신호
logger.info(f"[{self.symbol}] 신호: {raw_signal} | 현재가: {current_price:.4f} USDT")

# line 163: ML 필터 차단
logger.info(f"[{self.symbol}] ML 필터 차단: {signal} 신호 무시")

# line 223-228: 진입
logger.success(
    f"[{self.symbol}] {signal} 진입: 가격={price}, 수량={quantity}, "
    f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
    f"RSI={signal_snapshot['rsi']:.2f}, "
    f"MACD_H={signal_snapshot['macd_hist']:.6f}, "
    f"ATR={signal_snapshot['atr']:.6f}"
)

# line 277-279: 포지션 청산
logger.success(
    f"[{self.symbol}] 포지션 청산({close_reason}): 예상={estimated_pnl:+.4f}, "
    f"순수익={net_pnl:+.4f}, 차이={diff:+.4f} USDT"
)

# line 305-308: 포지션 모니터
logger.info(
    f"[{self.symbol}] 포지션 모니터 | {self.current_trade_side} | "
    f"현재가={price:.4f} | PnL={pnl:+.4f} USDT ({pnl_pct:+.2f}%) | "
    f"진입가={self._entry_price:.4f}"
)

# line 317: 청산 주문
logger.info(f"[{self.symbol}] 청산 주문 전송 완료 (side={side}, qty={amt})")

# line 349: ML 필터 재진입 차단
logger.info(f"[{self.symbol}] ML 필터 차단: {signal} 재진입 무시")

# line 362: 기준 잔고
logger.info(f"[{self.symbol}] 기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")
```

**Step 2: `src/user_data_stream.py` 로그 메시지 수정**

```python
# line 104-107: 청산 감지 로그에 심볼 추가
logger.info(
    f"[{self._symbol}] 청산 감지({close_reason}): exit={exit_price:.4f}, "
    f"rp={realized_pnl:+.4f}, commission={commission:.4f}, "
    f"net_pnl={net_pnl:+.4f}"
)
```

**Step 3: 기존 테스트 실행**

Run: `bash scripts/run_tests.sh -k "bot"`
Expected: 모든 테스트 PASS (로그 메시지 변경은 테스트에 영향 없음)

**Step 4: 커밋**

```bash
git add src/bot.py src/user_data_stream.py
git commit -m "feat: add [SYMBOL] prefix to all bot log messages for multi-symbol dashboard"
```

---

## Task 2: Log Parser 멀티심볼 대응

**Files:**
- Modify: `dashboard/api/log_parser.py` (정규식, 상태 추적, 핸들러)
- Create: `tests/test_log_parser.py` (파서 단위 테스트)

**Step 1: 파서 테스트 작성**

```python
# tests/test_log_parser.py
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
```

**Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_log_parser.py -v`
Expected: FAIL (아직 파서 수정 전)

**Step 3: `log_parser.py` 수정 — 정규식에 `[SYMBOL]` 프리픽스 추가**

모든 정규식 패턴에 `\[(?P<symbol>\w+)\]` 추가:

```python
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
```

**Step 4: DB 스키마 변경**

`_init_db()` 메서드의 CREATE TABLE 문 수정:

```python
def _init_db(self):
    # 기존 테이블 삭제 후 재생성 (데이터는 로그 재파싱으로 복구)
    self.conn.executescript("""
        DROP TABLE IF EXISTS trades;
        DROP TABLE IF EXISTS candles;
        DROP TABLE IF EXISTS daily_pnl;
        DROP TABLE IF EXISTS bot_status;
        DROP TABLE IF EXISTS parse_state;

        CREATE TABLE trades (
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

        CREATE TABLE candles (
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

        CREATE TABLE daily_pnl (
            symbol          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            cumulative_pnl  REAL    DEFAULT 0,
            trade_count     INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            losses          INTEGER DEFAULT 0,
            last_updated    TEXT,
            PRIMARY KEY(symbol, date)
        );

        CREATE TABLE bot_status (
            key             TEXT    PRIMARY KEY,
            value           TEXT,
            updated_at      TEXT
        );

        CREATE TABLE parse_state (
            filepath        TEXT    PRIMARY KEY,
            position        INTEGER DEFAULT 0
        );

        CREATE INDEX idx_candles_symbol_ts ON candles(symbol, ts);
        CREATE INDEX idx_trades_status ON trades(status);
        CREATE INDEX idx_trades_symbol ON trades(symbol);
    """)
    self.conn.commit()
    self._load_state()
```

**Step 5: 상태 추적 멀티심볼 대응**

`__init__` 수정:

```python
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
```

`_load_state` 수정:

```python
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
```

**Step 6: `_parse_line` 핸들러 수정**

`bot_start` 핸들러 — 심볼별 bot_status:

```python
m = PATTERNS["bot_start"].search(line)
if m:
    symbol = m.group("symbol")
    self._set_status(f"{symbol}:leverage", m.group("leverage"))
    self._set_status(f"{symbol}:last_start", m.group("ts"))
    return
```

`balance` 핸들러 — 전역 잔고 유지:

```python
m = PATTERNS["balance"].search(line)
if m:
    self._balance = float(m.group("balance"))
    self._set_status("balance", m.group("balance"))
    return
```

`position_recover` 핸들러:

```python
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
```

`entry` 핸들러:

```python
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
```

`microstructure` 핸들러:

```python
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
```

`adx` 핸들러:

```python
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
```

`signal` 핸들러:

```python
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
```

`close_detect` 핸들러:

```python
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
```

`daily_pnl` 핸들러:

```python
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
```

**Step 7: `_handle_entry` 수정**

```python
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
```

**Step 8: `_handle_close` 수정**

```python
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
```

**Step 9: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_log_parser.py -v`
Expected: 모든 테스트 PASS

**Step 10: 커밋**

```bash
git add dashboard/api/log_parser.py tests/test_log_parser.py
git commit -m "feat: update log parser for multi-symbol support"
```

---

## Task 3: API 멀티심볼 대응

**Files:**
- Modify: `dashboard/api/dashboard_api.py`
- Create: `tests/test_dashboard_api.py`

**Step 1: API 테스트 작성**

```python
# tests/test_dashboard_api.py
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
    os.unlink(db_path) if os.path.exists(db_path) else None


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
```

**Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_dashboard_api.py -v`
Expected: FAIL

**Step 3: `dashboard_api.py` 수정**

```python
"""
dashboard_api.py — 멀티심볼 대시보드 API
"""

import sqlite3
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

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


@app.get("/api/symbols")
def get_symbols():
    """활성 심볼 목록 반환."""
    with get_db() as db:
        rows = db.execute(
            "SELECT key FROM bot_status WHERE key LIKE '%:last_start'"
        ).fetchall()
    symbols = [r["key"].split(":")[0] for r in rows]
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
    offset: int = 0,
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
        where = "WHERE status='CLOSED'" + (f" AND symbol='{symbol}'" if symbol else "")
        row = db.execute(f"""
            SELECT
                COUNT(*) as total_trades,
                COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                COALESCE(SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
                COALESCE(SUM(net_pnl), 0) as total_pnl,
                COALESCE(SUM(commission), 0) as total_fees,
                COALESCE(AVG(net_pnl), 0) as avg_pnl,
                COALESCE(MAX(net_pnl), 0) as best_trade,
                COALESCE(MIN(net_pnl), 0) as worst_trade
            FROM trades {where}
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
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/reset")
def reset_db():
    with get_db() as db:
        for table in ["trades", "daily_pnl", "parse_state", "bot_status", "candles"]:
            db.execute(f"DELETE FROM {table}")
        db.commit()

    import subprocess, signal
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
    subprocess.Popen(["python", "log_parser.py"])

    return {"status": "ok", "message": "DB 초기화 완료, 파서 재시작됨"}
```

> 주의: `/api/stats`의 `symbol` 파라미터는 쿼리 파라미터이므로 SQL injection 위험이 있음. 실제 구현 시 파라미터 바인딩 사용. 위 코드에서는 f-string을 사용했지만, 구현 시 반드시 `?` 바인딩으로 교체할 것.

**Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_dashboard_api.py -v`
Expected: 모든 테스트 PASS

**Step 5: 커밋**

```bash
git add dashboard/api/dashboard_api.py tests/test_dashboard_api.py
git commit -m "feat: add multi-symbol support to dashboard API"
```

---

## Task 4: UI 멀티심볼 대응

**Files:**
- Modify: `dashboard/ui/src/App.jsx`

**Step 1: 상태 및 데이터 페칭에 심볼 지원 추가**

주요 변경사항:

1. `symbols` 상태 추가, `/api/symbols`에서 로드
2. `selectedSymbol` 상태 추가 (기본값 `null` = ALL)
3. `fetchAll`에서 선택된 심볼을 쿼리 파라미터로 전달
4. `position` → `positions` (배열)로 변경

```jsx
const [symbols, setSymbols] = useState([]);
const [selectedSymbol, setSelectedSymbol] = useState(null); // null = ALL
const [positions, setPositions] = useState([]);
```

`fetchAll` 수정:

```jsx
const fetchAll = useCallback(async () => {
  const sym = selectedSymbol ? `?symbol=${selectedSymbol}` : "";
  const symRequired = selectedSymbol || symbols[0] || "XRPUSDT";

  const [symRes, sRes, pRes, tRes, dRes, cRes] = await Promise.all([
    api("/symbols"),
    api(`/stats${sym}`),
    api(`/position${sym}`),
    api(`/trades${sym}&limit=50`.replace("?&", "?")),
    api(`/daily${sym}`),
    api(`/candles?symbol=${symRequired}&limit=96`),
  ]);

  if (symRes?.symbols) setSymbols(symRes.symbols);
  if (sRes && sRes.total_trades !== undefined) {
    setStats(sRes);
    setIsLive(true);
    setLastUpdate(new Date());
  }
  if (pRes) {
    setPositions(pRes.positions || []);
    if (pRes.bot) setBotStatus(pRes.bot);
  }
  if (tRes?.trades) setTrades(tRes.trades);
  if (dRes?.daily) setDaily(dRes.daily);
  if (cRes?.candles) setCandles(cRes.candles);
}, [selectedSymbol, symbols]);
```

**Step 2: 심볼 필터 탭 추가**

기존 탭(Overview/Trades/Chart) 위에 심볼 필터 추가:

```jsx
{/* 심볼 필터 */}
<div style={{
  display: "flex", gap: 4, marginBottom: 12,
  background: "rgba(255,255,255,0.02)", borderRadius: 12,
  padding: 4, width: "fit-content",
}}>
  <button
    onClick={() => setSelectedSymbol(null)}
    style={{
      background: selectedSymbol === null ? "rgba(99,102,241,0.15)" : "transparent",
      border: "none",
      color: selectedSymbol === null ? S.indigo : S.text3,
      padding: "6px 14px", borderRadius: 8, cursor: "pointer",
      fontSize: 11, fontWeight: 600, fontFamily: S.mono,
    }}
  >ALL</button>
  {symbols.map((sym) => (
    <button
      key={sym}
      onClick={() => setSelectedSymbol(sym)}
      style={{
        background: selectedSymbol === sym ? "rgba(99,102,241,0.15)" : "transparent",
        border: "none",
        color: selectedSymbol === sym ? S.indigo : S.text3,
        padding: "6px 14px", borderRadius: 8, cursor: "pointer",
        fontSize: 11, fontWeight: 600, fontFamily: S.mono,
      }}
    >{sym.replace("USDT", "")}</button>
  ))}
</div>
```

**Step 3: 헤더 동적 변경**

```jsx
{/* "Live · XRP/USDT" → "Live · 3 symbols" 또는 "Live · XRP/USDT" */}
<span style={{ ... }}>
  {isLive ? "Live" : "Connecting…"}
  {selectedSymbol
    ? ` · ${selectedSymbol.replace("USDT", "/USDT")}`
    : ` · ${symbols.length} symbols`}
  {selectedSymbol && botStatus[`${selectedSymbol}:current_price`] && (
    <span style={{ color: "rgba(255,255,255,0.5)", marginLeft: 8 }}>
      {fmt(botStatus[`${selectedSymbol}:current_price`])}
    </span>
  )}
</span>
```

**Step 4: 오픈 포지션 복수 표시**

```jsx
{/* 오픈 포지션 — 복수 표시 */}
{positions.length > 0 && (
  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
    {positions.map((pos) => (
      <div key={pos.id} style={{
        background: "linear-gradient(135deg,rgba(99,102,241,0.08) 0%,rgba(99,102,241,0.02) 100%)",
        border: "1px solid rgba(99,102,241,0.15)", borderRadius: 14,
        padding: "12px 18px",
      }}>
        <div style={{ fontSize: 9, color: S.text3, letterSpacing: 1.2, fontFamily: S.mono, marginBottom: 4 }}>
          {(pos.symbol || "").replace("USDT", "/USDT")}
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <Badge
            bg={pos.direction === "SHORT" ? "rgba(239,68,68,0.12)" : "rgba(52,211,153,0.12)"}
            color={pos.direction === "SHORT" ? S.red : S.green}
          >
            {pos.direction} {pos.leverage || 10}x
          </Badge>
          <span style={{ fontSize: 14, fontWeight: 700, fontFamily: S.mono }}>
            {fmt(pos.entry_price)}
          </span>
        </div>
      </div>
    ))}
  </div>
)}
```

**Step 5: Chart 탭 — ALL일 때 첫 번째 심볼 사용**

```jsx
{/* Chart 탭 제목 */}
<ChartBox title={`${(selectedSymbol || symbols[0] || "XRP").replace("USDT", "")}/USDT 15m 가격`}>
```

**Step 6: 수동 확인**

- `npm run dev` 또는 Docker 빌드 후 UI 확인
- 심볼 탭 전환 시 데이터가 올바르게 필터링되는지
- ALL 탭에서 전체 통계가 합산되는지
- 오픈 포지션이 복수 표시되는지

**Step 7: 커밋**

```bash
git add dashboard/ui/src/App.jsx
git commit -m "feat: add multi-symbol UI with symbol filter tabs"
```

---

## Task 5: 전체 통합 테스트 및 마무리

**Files:**
- Verify: 전체 테스트 스위트

**Step 1: 전체 테스트 실행**

Run: `bash scripts/run_tests.sh`
Expected: 모든 테스트 PASS

**Step 2: 기존 봇 테스트가 깨지지 않는지 확인**

Run: `bash scripts/run_tests.sh -k "bot"`
Expected: 모든 테스트 PASS

**Step 3: Jenkins CI/CD 변경 확인**

`Jenkinsfile`의 변경 감지 로직이 `dashboard/` 디렉토리와 `src/bot.py`, `src/user_data_stream.py` 변경을 인식하는지 확인. 봇 이미지와 대시보드 이미지 모두 재빌드 트리거 필요.

**Step 4: 운영 배포 후 확인**

1. Docker 이미지 재빌드 (봇 + dashboard-api + dashboard-ui)
2. 운영 서버에서 `docker compose down && docker compose up -d`
3. 대시보드 UI에서 심볼 탭 확인
4. DB 초기화 (Reset DB 버튼) → 로그 재파싱 → 데이터 확인

**Step 5: 최종 커밋 및 CLAUDE.md 업데이트**

`CLAUDE.md`의 plan 테이블에서 `multi-symbol-dashboard` status를 `Completed`로 변경.

```bash
git add CLAUDE.md
git commit -m "docs: mark multi-symbol-dashboard plan as completed"
```
