# Multi-Symbol Dashboard Design

## 배경

멀티심볼 트레이딩(XRP, TRX, DOGE) 지원 이후 대시보드가 단일 심볼 기준으로 되어있어 수정 필요. 봇 로그 형식 통일, 파서/DB/API/UI 전체 레이어 변경.

## 접근 방식

**A안 채택**: 기존 단일 DB에 `symbol` 컬럼 추가. 대시보드 DB는 로그 파싱으로 재생성 가능하므로 초기화 비용 없음.

## 1. 봇 로그 수정

모든 핵심 로그에 `[SYMBOL]` 프리픽스를 일관되게 추가.

변경 대상 (`src/bot.py`):
- `신호: {signal} | 현재가:` → `[{self.symbol}] 신호: ...`
- `{signal} 진입: 가격=` → `[{self.symbol}] {signal} 진입: ...`
- `기존 포지션 복구:` → `[{self.symbol}] 기존 포지션 복구: ...`
- `기준 잔고 설정:` → `[{self.symbol}] 기준 잔고 설정: ...`
- `포지션 청산(...)` → `[{self.symbol}] 포지션 청산(...)`
- `OI=..., OI변화율=...` → `[{self.symbol}] OI=...` (debug→info로 변경 또는 그대로 debug 유지)

변경 대상 (`src/user_data_stream.py`):
- `청산 감지({reason}):` → `[{self.symbol}] 청산 감지({reason}): ...`

이미 `[{self.symbol}]`이 있는 로그는 그대로 유지.

## 2. Log Parser (`log_parser.py`)

### 정규식 변경

모든 패턴에 `\[(?P<symbol>\w+)\]` 프리픽스 추가:

```python
"signal": re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r".*\[(?P<symbol>\w+)\] 신호: (?P<signal>\w+) \| 현재가: (?P<price>[\d.]+) USDT"
),
```

### 상태 추적 멀티심볼 대응

- `_current_position: dict` → `_current_positions: dict[str, dict]` (심볼별)
- `_pending_candle: dict` → `_pending_candles: dict[str, dict[str, dict]]` (심볼별 타임스탬프별)
- `_bot_config["symbol"]` 제거, 정규식에서 심볼 직접 파싱

### 핸들러 변경

**`_handle_entry`**: symbol을 정규식에서 직접 받음. 중복 체크를 `symbol+direction` 기준으로.

**`_handle_close`**: `WHERE status='OPEN' AND symbol=?`로 해당 심볼만 닫음.

### bot_status 키 형식

- 심볼별: `{symbol}:current_price`, `{symbol}:position_status`, `{symbol}:current_signal` 등
- 전역: `balance`, `ml_threshold` 그대로

## 3. DB 스키마 변경

### candles 테이블

```sql
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
CREATE INDEX idx_candles_symbol_ts ON candles(symbol, ts);
```

### daily_pnl 테이블

```sql
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
```

### trades 테이블

기존 `symbol` 컬럼 있음. `DEFAULT 'XRPUSDT'` 제거, 파서에서 항상 명시적으로 심볼 전달.

### bot_status 테이블

스키마 변경 없음. 키 네이밍만 `{symbol}:{key}` 형태로 변경.

### 마이그레이션

`_init_db()`에서 `DROP TABLE IF EXISTS` → 재생성. 기존 데이터는 로그 재파싱으로 복구.

## 4. API (`dashboard_api.py`)

모든 엔드포인트에 `symbol` 쿼리 파라미터 추가. 없으면 전체.

### 변경 엔드포인트

| 엔드포인트 | 변경 |
|-----------|------|
| `GET /api/position` | 심볼별 OPEN 포지션 목록 반환. `{"positions": [...], "bot": {...}}` |
| `GET /api/trades` | `?symbol=` 필터 추가 |
| `GET /api/stats` | `?symbol=` 필터 추가 |
| `GET /api/daily` | `?symbol=` 필터 추가 |
| `GET /api/candles` | `?symbol=` 필수 파라미터 |

### 새 엔드포인트

```
GET /api/symbols → {"symbols": ["XRPUSDT", "TRXUSDT", "DOGEUSDT"]}
```

`bot_status`에서 `{symbol}:last_start` 키가 있는 심볼 목록 반환.

## 5. UI (`App.jsx`)

### 헤더

- "XRP/USDT" 하드코딩 제거 → `Live · 3 symbols`
- 오픈 포지션 카드를 심볼별 복수 표시 (가로 나열)

### 심볼 필터 탭

기존 탭(Overview/Trades/Chart) 위에 심볼 필터 추가: `ALL | XRP | TRX | DOGE`
- `/api/symbols`에서 동적 생성
- `ALL`: 전체 합산, 개별 심볼: 해당 심볼만

### Overview 탭

- `ALL`: 전체 합산 StatCard + 일별 PnL + 최근 거래(심볼 뱃지 표시)
- 개별 심볼: 해당 심볼만

### Trades 탭

- 선택된 심볼로 필터링

### Chart 탭

- `ALL` 선택 시 첫 번째 심볼 자동 선택 (캔들은 심볼별)
- 차트 제목 동적: `{SYMBOL}/USDT 15m 가격`

### 데이터 페칭

- `fetchAll`에서 선택된 심볼을 쿼리 파라미터로 전달
- 심볼 변경 시 즉시 리페치

## 6. 변경 범위 요약

| 레이어 | 파일 | 변경 |
|--------|------|------|
| 봇 | `src/bot.py` | 로그에 `[SYMBOL]` 프리픽스 추가 |
| 봇 | `src/user_data_stream.py` | 청산 로그에 `[SYMBOL]` 프리픽스 추가 |
| 파서 | `dashboard/api/log_parser.py` | 정규식, 상태 추적, 핸들러 멀티심볼 대응 |
| API | `dashboard/api/dashboard_api.py` | `symbol` 파라미터, `/api/symbols` |
| UI | `dashboard/ui/src/App.jsx` | 심볼 필터 탭, 복수 포지션, 동적 헤더 |

봇 이미지와 대시보드 이미지 모두 재빌드 필요.
