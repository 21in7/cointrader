# Testnet 1분봉 125x 트레이딩 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 바이낸스 테스트넷에서 XRPUSDT 1분봉, 125x 레버리지로 ML 기반 자동매매를 실행한다.

**Architecture:** Config에 `testnet` 플래그를 추가하고, Exchange/DataStream/UserDataStream에 `testnet=True`를 전달한다. 학습 파이프라인은 LOOKAHEAD=60(1시간)으로 조정하여 1분봉 데이터로 새 모델을 학습한다.

**Tech Stack:** python-binance (testnet=True), LightGBM, asyncio

---

## Task 1: Config에 testnet 지원 추가

**Files:**
- Modify: `src/config.py:8-33`
- Test: `tests/test_config.py` (기존 테스트 수정 필요시)

**Step 1: Config에 testnet, interval 필드 추가**

`src/config.py`에서 `Config` dataclass에 `testnet`, `interval` 필드를 추가하고, `__post_init__`에서 `BINANCE_TESTNET=true`이면 테스트넷 키를 사용하도록 변경:

```python
@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "XRPUSDT"
    leverage: int = 10
    testnet: bool = False
    interval: str = "15m"
    max_positions: int = 3
    stop_loss_pct: float = 0.015    # 1.5%
    take_profit_pct: float = 0.045  # 4.5% (3:1 RR)
    trailing_stop_pct: float = 0.01  # 1%
    discord_webhook_url: str = ""
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006
    ml_threshold: float = 0.55

    def __post_init__(self):
        self.testnet = os.getenv("BINANCE_TESTNET", "").lower() in ("true", "1", "yes")
        self.interval = os.getenv("INTERVAL", "15m")

        if self.testnet:
            self.api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            self.api_key = os.getenv("BINANCE_API_KEY", "")
            self.api_secret = os.getenv("BINANCE_API_SECRET", "")

        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.margin_max_ratio = float(os.getenv("MARGIN_MAX_RATIO", "0.50"))
        self.margin_min_ratio = float(os.getenv("MARGIN_MIN_RATIO", "0.20"))
        self.margin_decay_rate = float(os.getenv("MARGIN_DECAY_RATE", "0.0006"))
        self.ml_threshold = float(os.getenv("ML_THRESHOLD", "0.55"))
```

**Step 2: 테스트 실행**

Run: `pytest tests/ -v --tb=short -x`
Expected: 기존 테스트 모두 PASS (testnet 미설정 시 기존 동작 유지)

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add testnet and interval support to Config"
```

---

## Task 2: Exchange에 testnet 전달

**Files:**
- Modify: `src/exchange.py:9-14`

**Step 1: Client 생성자에 testnet 전달**

`src/exchange.py`에서 `BinanceFuturesClient.__init__`을 수정:

```python
class BinanceFuturesClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
        )
```

**Step 2: 테스트 실행**

Run: `pytest tests/test_exchange.py -v --tb=short -x`
Expected: PASS

**Step 3: Commit**

```bash
git add src/exchange.py
git commit -m "feat: pass testnet flag to Binance Client"
```

---

## Task 3: DataStream에 testnet 전달

**Files:**
- Modify: `src/data_stream.py:78-82,185-189`

**Step 1: KlineStream.start()에 testnet 파라미터 추가**

`src/data_stream.py`의 `KlineStream.start()` (line 78)을 수정:

```python
    async def start(self, api_key: str, api_secret: str, testnet: bool = False):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
```

**Step 2: MultiSymbolStream.start()에 testnet 파라미터 추가**

`src/data_stream.py`의 `MultiSymbolStream.start()` (line 185)을 수정:

```python
    async def start(self, api_key: str, api_secret: str, testnet: bool = False):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
```

**Step 3: 테스트 실행**

Run: `pytest tests/test_data_stream.py -v --tb=short -x`
Expected: PASS (testnet 기본값 False이므로 기존 동작 유지)

**Step 4: Commit**

```bash
git add src/data_stream.py
git commit -m "feat: pass testnet flag to AsyncClient in data streams"
```

---

## Task 4: UserDataStream에 testnet 전달

**Files:**
- Modify: `src/user_data_stream.py:28-33`

**Step 1: start()에 testnet 파라미터 추가**

```python
    async def start(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        """User Data Stream 메인 루프 — 봇 종료 시까지 실행."""
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
```

**Step 2: 테스트 실행**

Run: `pytest tests/test_user_data_stream.py -v --tb=short -x`
Expected: PASS

**Step 3: Commit**

```bash
git add src/user_data_stream.py
git commit -m "feat: pass testnet flag to AsyncClient in user data stream"
```

---

## Task 5: Bot에서 testnet/interval 전달

**Files:**
- Modify: `src/bot.py:27-31,300-322`

**Step 1: MultiSymbolStream에 config.interval 전달**

`src/bot.py` line 27-31을 수정:

```python
        self.stream = MultiSymbolStream(
            symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
            interval=config.interval,
            on_candle=self._on_candle_closed,
        )
```

**Step 2: run()에서 testnet 전달**

`src/bot.py` line 300-322의 `run()` 메서드에서 stream.start() 호출 시 testnet 전달:

```python
    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x, 테스트넷={self.config.testnet}")
        await self._recover_position()
        balance = await self.exchange.get_balance()
        self.risk.set_base_balance(balance)
        logger.info(f"기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")

        user_stream = UserDataStream(
            symbol=self.config.symbol,
            on_order_filled=self._on_position_closed,
        )

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

**Step 3: 테스트 실행**

Run: `pytest tests/test_bot.py -v --tb=short -x`
Expected: PASS

**Step 4: Commit**

```bash
git add src/bot.py
git commit -m "feat: pass testnet and interval from config to streams"
```

---

## Task 6: .env 설정 및 전체 테스트

**Files:**
- Modify: `.env`
- Modify: `.env.example`

**Step 1: .env.example 업데이트**

`.env.example`에 새 변수 추가:

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=false
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
SYMBOL=XRPUSDT
LEVERAGE=10
INTERVAL=15m
RISK_PER_TRADE=0.02
DISCORD_WEBHOOK_URL=
ML_THRESHOLD=0.55
```

**Step 2: 워크트리의 .env에 테스트넷 설정**

`.env` 파일에 테스트넷 키와 설정 적용:

```
BINANCE_TESTNET=true
BINANCE_TESTNET_API_KEY=<사용자의_테스트넷_키>
BINANCE_TESTNET_API_SECRET=<사용자의_테스트넷_시크릿>
SYMBOL=XRPUSDT
LEVERAGE=125
INTERVAL=1m
ML_THRESHOLD=0.55
```

**Step 3: 전체 테스트 실행**

Run: `bash scripts/run_tests.sh`
Expected: 모든 테스트 PASS

**Step 4: Commit**

```bash
git add .env.example
git commit -m "feat: add testnet and interval env vars to .env.example"
```

---

## Task 7: 학습 파이프라인 — LOOKAHEAD 조정 및 1분봉 데이터 수집

**Files:**
- Modify: `src/dataset_builder.py:14` (LOOKAHEAD 변경)
- Modify: `scripts/train_model.py:56` (LOOKAHEAD 변경)
- Modify: `scripts/train_and_deploy.sh:32-50` (1분봉 데이터 경로)

**Step 1: dataset_builder.py LOOKAHEAD 변경**

`src/dataset_builder.py` line 14:

```python
# 변경 전:
LOOKAHEAD    = 24   # 15분봉 × 24 = 6시간 뷰

# 변경 후:
LOOKAHEAD    = 60   # 1분봉 × 60 = 1시간 뷰
```

**Step 2: train_model.py LOOKAHEAD 변경**

`scripts/train_model.py` line 56:

```python
# 변경 전:
LOOKAHEAD = 24  # 15분봉 × 24 = 6시간 (dataset_builder.py와 동기화)

# 변경 후:
LOOKAHEAD = 60  # 1분봉 × 60 = 1시간 (dataset_builder.py와 동기화)
```

**Step 3: train_and_deploy.sh 수정 — 1분봉 파이프라인**

`scripts/train_and_deploy.sh`의 데이터 경로와 수집 파라미터를 1분봉으로 변경:

```bash
# line 32: 파일명 변경
PARQUET_FILE="data/combined_1m.parquet"

# line 46-50: --interval 1m으로 변경
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 1m \
    --days "$FETCH_DAYS" \
    $UPSERT_FLAG \
    --output "$PARQUET_FILE"

# line 57, 60: --data 경로 변경
python scripts/train_mlx_model.py --data data/combined_1m.parquet --decay "$DECAY"
# ...
python scripts/train_model.py --data data/combined_1m.parquet --decay "$DECAY"

# walk-forward 섹션도 동일하게 --data data/combined_1m.parquet로 변경
```

**Step 4: Commit**

```bash
git add src/dataset_builder.py scripts/train_model.py scripts/train_and_deploy.sh
git commit -m "feat: adjust LOOKAHEAD to 60 for 1m candles, update training pipeline"
```

---

## Task 8: 1분봉 데이터 수집

**Step 1: 1분봉 데이터 수집 (30일)**

Run:
```bash
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 1m \
    --days 30 \
    --no-oi \
    --no-upsert \
    --output data/combined_1m.parquet
```

Expected: `data/combined_1m.parquet` 생성 (약 43,000행 × 15컬럼)

> Note: 테스트넷 학습용이므로 OI/펀딩비는 건너뜀 (--no-oi). 1분봉 30일 데이터면 약 43,200개 캔들.

---

## Task 9: ML 모델 학습

**Step 1: LightGBM 모델 학습**

Run:
```bash
python scripts/train_model.py --data data/combined_1m.parquet --decay 2.0
```

Expected: `models/lgbm_filter.pkl` 생성, AUC 출력

**Step 2: 학습 결과 확인**

학습 결과의 AUC가 0.50 이상인지 확인. 모델이 생성되었는지 확인:

Run: `ls -la models/lgbm_filter.pkl`
Expected: 파일 존재

---

## Task 10: 테스트넷 봇 실행

**Step 1: 최종 확인**

`.env`에 테스트넷 설정이 올바른지 확인:
- `BINANCE_TESTNET=true`
- `BINANCE_TESTNET_API_KEY` 설정됨
- `BINANCE_TESTNET_API_SECRET` 설정됨
- `LEVERAGE=125`
- `INTERVAL=1m`

**Step 2: 봇 실행**

Run: `python main.py`

Expected output:
```
봇 시작: XRPUSDT, 레버리지 125x, 테스트넷=True
```

봇이 정상 시작되면 1분봉 캔들을 수신하고 ML 필터를 통해 거래 신호를 처리한다.
