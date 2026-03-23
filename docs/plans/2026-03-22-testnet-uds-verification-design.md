# Testnet UDS 검증 설계

**일자**: 2026-03-22
**상태**: 설계 완료, 구현 대기

---

## 목적

Binance Futures Testnet에서 User Data Stream(UDS)의 reconnect 동작을 검증한다. 현재 프로덕션 15분봉 설정 그대로 testnet에 연결하여, UDS 연결 → ~30분 후 reconnect → ORDER_TRADE_UPDATE 수신까지 전체 경로가 정상 작동하는지 확인한다.

**이것은 UDS 검증 전용이다.** 1분봉 전환, 125x 레버리지, ML 파이프라인 변경은 포함하지 않는다. 기존 설계(`2026-03-03-testnet-1m-125x`)는 ML OFF 확정 후 전제가 바뀌었으므로 별도 취급한다.

---

## 접근 방식

python-binance 1.0.35에서 `testnet=True` 파라미터가 REST API와 WebSocket(kline + User Data Stream) 모두 자동 라우팅한다. 별도 URL 오버라이드 불필요.

**검증된 라우팅 경로 (python-binance 소스 확인):**
- REST API: `https://testnet.binancefuture.com`
- Kline WebSocket: `wss://stream.binancefuture.com/` (`BinanceSocketManager._get_futures_socket()`에서 `self.testnet` 체크)
- User Data Stream WebSocket: `wss://stream.binancefuture.com/` (`futures_user_socket()`에서 `self.testnet` 체크)

`AsyncClient.create(testnet=True)` → `BinanceSocketManager(client)` → `client.testnet` 플래그가 자동 전파.

---

## 수정 대상 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/config.py` | `testnet: bool` 필드 추가, `BINANCE_TESTNET` env var 파싱, testnet이면 testnet API key 사용 |
| `src/exchange.py` | `Client(..., testnet=config.testnet)` 전달 |
| `src/user_data_stream.py` | `AsyncClient.create(..., testnet=testnet)` 전달 |
| `src/data_stream.py` | `AsyncClient.create(..., testnet=testnet)` 전달 (KlineStream + MultiSymbolStream) |
| `src/notifier.py` | testnet일 때 Discord 메시지에 `[TESTNET]` 접두사 추가 |
| `src/bot.py` | testnet 플래그를 각 스트림/notifier에 전달 + trade_history 경로 분리 + 시작 시 TESTNET 경고 로그 |

### 변경하지 않는 것

- 지표 계산 (`src/indicators.py`) — 그대로
- ML 필터 (`src/ml_filter.py`) — NO_ML_FILTER=true 상태 그대로
- 학습 파이프라인 — 변경 없음
- 리스크 매니저 — 그대로
- Discord 알림 — testnet일 때 메시지에 `[TESTNET]` 접두사 추가 (아래 상세 변경 참조)
- `.env` 프로덕션 설정 — 변경 없음 (BINANCE_TESTNET 추가만)

---

## 상세 변경

### 1. Config (`src/config.py`)

```python
# 필드 추가
testnet: bool = False

# __post_init__에서:
self.testnet = os.getenv("BINANCE_TESTNET", "").lower() in ("true", "1", "yes")

if self.testnet:
    self.api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
    self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
else:
    self.api_key = os.getenv("BINANCE_API_KEY", "")
    self.api_secret = os.getenv("BINANCE_API_SECRET", "")
```

- testnet이면 `BINANCE_TESTNET_API_KEY/SECRET` 사용
- 나머지 설정(SYMBOLS, LEVERAGE 등)은 동일하게 적용

### 2. Exchange (`src/exchange.py`)

```python
# 현재:
self.client = Client(
    api_key=config.api_key,
    api_secret=config.api_secret,
)

# 변경:
self.client = Client(
    api_key=config.api_key,
    api_secret=config.api_secret,
    testnet=config.testnet,
)
```

### 3. UserDataStream (`src/user_data_stream.py`)

`_run_loop()` 시그니처에 `testnet` 파라미터 추가:

```python
# start()에 testnet 파라미터 추가
async def start(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
    ...
    await self._run_loop(api_key, api_secret, testnet)

# _run_loop()에서 AsyncClient.create에 전달
async def _run_loop(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
    while True:
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
```

### 4. DataStream (`src/data_stream.py`)

KlineStream.start()과 MultiSymbolStream.start() 모두 동일 패턴:

```python
async def start(self, api_key: str, api_secret: str, testnet: bool = False):
    client = await AsyncClient.create(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
```

MultiSymbolStream._run_loop()에서도 reconnect 시 AsyncClient.create에 testnet 전달.

### 5. Notifier (`src/notifier.py`)

testnet일 때 Discord 메시지에 `[TESTNET]` 접두사를 추가하여 프로덕션 알림과 구분:

```python
# Notifier.__init__()에 testnet 파라미터 추가
def __init__(self, webhook_url: str, testnet: bool = False):
    self.webhook_url = webhook_url
    self.testnet = testnet

# 메시지 전송 시 접두사 추가
async def _send(self, content: str):
    if self.testnet:
        content = f"[TESTNET] {content}"
    ...
```

Bot에서 Notifier 생성 시 `testnet=self.config.testnet` 전달.

### 6. Bot (`src/bot.py`)

**시작 로그에 TESTNET 명시 (warning 레벨):**

```python
async def run(self):
    if self.config.testnet:
        logger.warning("⚠️ TESTNET MODE ENABLED — 실제 자금이 아닌 테스트넷에서 실행 중")
    ...
```

**stream.start()와 user_stream.start()에 testnet 전달:**

```python
await asyncio.gather(
    self.stream.start(
        api_key=self.config.api_key,
        api_secret=self.config.api_secret,
        testnet=self.config.testnet,
    ),
    user_stream.start(
        api_key=self.config.api_key,
        api_secret=self.config.api_secret,
        testnet=self.config.testnet,
    ),
    self._position_monitor(),
)
```

**trade_history 경로 분리:**

```python
# 현재 (line 24):
_TRADE_HISTORY_DIR = Path("data/trade_history")

# 변경 — _trade_history_path() 메서드에서 분기:
def _trade_history_path(self) -> Path:
    base = Path("data/trade_history")
    if self.config.testnet:
        base = base / "testnet"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{self.symbol.lower()}.jsonl"
```

- testnet: `data/trade_history/testnet/xrpusdt.jsonl`
- production: `data/trade_history/xrpusdt.jsonl` (기존과 동일)
- Kill Switch 판정이 testnet 트레이드로 오염되지 않음

---

## .env 설정

```bash
# 기존 프로덕션 설정 유지 + 아래 추가
BINANCE_TESTNET=true          # testnet 모드 활성화
BINANCE_TESTNET_API_KEY=xxx   # testnet.binancefuture.com에서 발급
BINANCE_TESTNET_API_SECRET=xxx
```

- `BINANCE_TESTNET=true`를 설정하면 testnet 모드로 전환
- 프로덕션 복귀 시 `BINANCE_TESTNET=false` 또는 줄 삭제

**주의**: .env에 이미 `BINANCE_TESTNET_API_KEY`/`BINANCE_TESTNET_API_SECRET` 자리가 마련되어 있음.

---

## 검증 절차

### 1단계: Testnet API 키 발급

- `testnet.binancefuture.com` 접속 → API 키 발급
- `.env`에 설정

### 2단계: 봇 실행 + UDS 검증

```bash
# .env에 BINANCE_TESTNET=true 설정 후
python main.py
```

확인 사항:
1. 시작 로그에 testnet 표시 확인
2. User Data Stream 연결 로그 확인
3. ~30분 대기 → reconnect 발생하는지 확인
4. reconnect 후 ORDER_TRADE_UPDATE 수신되는지 확인
5. trade_history가 `data/trade_history/testnet/` 에 기록되는지 확인

### 3단계: Kill Switch 경로 확인

- testnet 트레이드가 `data/trade_history/testnet/xrpusdt.jsonl`에만 기록되는지 확인
- 프로덕션 `data/trade_history/xrpusdt.jsonl`이 변경되지 않았는지 확인

---

## 주의사항

- **테스트넷 가격은 실제 시장과 다름**: 전략 성과 판단 불가, UDS 동작 검증만 목적
- **trade_history 분리 필수**: testnet 트레이드가 프로덕션 Kill Switch를 오염시키면 안 됨
- **프로덕션 배포 시 BINANCE_TESTNET 제거 확인**: `.env`에 `BINANCE_TESTNET=true`가 남아있으면 프로덕션이 testnet으로 연결됨
