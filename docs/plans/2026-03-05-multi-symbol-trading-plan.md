# Multi-Symbol Trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** XRP 단일 심볼 거래 봇을 TRX·DOGE 등 다중 심볼 동시 거래로 확장한다.

**Architecture:** 심볼별 독립 TradingBot 인스턴스를 `asyncio.gather()`로 병렬 실행. RiskManager만 공유 싱글턴으로 글로벌 리스크(일일 손실 한도, 동일 방향 제한)를 관리한다. 각 봇은 자기 심볼을 직접 소유하고, `config.symbol` 의존을 완전 제거한다.

**Tech Stack:** Python asyncio, LightGBM, ONNX, Binance Futures API

**Design Doc:** `docs/plans/2026-03-05-multi-symbol-trading-design.md`

---

## Task 1: Config — `symbols` 리스트 추가, `symbol` 필드 유지(하위호환)

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

**Step 1: Write the failing tests**

`tests/test_config.py`에 다음 테스트를 추가:

```python
def test_config_loads_symbols_list():
    """SYMBOLS 환경변수로 쉼표 구분 리스트를 로드한다."""
    os.environ["SYMBOLS"] = "XRPUSDT,TRXUSDT,DOGEUSDT"
    os.environ.pop("SYMBOL", None)
    cfg = Config()
    assert cfg.symbols == ["XRPUSDT", "TRXUSDT", "DOGEUSDT"]


def test_config_fallback_to_symbol():
    """SYMBOLS 미설정 시 SYMBOL에서 1개짜리 리스트로 변환한다."""
    os.environ.pop("SYMBOLS", None)
    os.environ["SYMBOL"] = "XRPUSDT"
    cfg = Config()
    assert cfg.symbols == ["XRPUSDT"]


def test_config_correlation_symbols():
    """상관관계 심볼 로드."""
    os.environ["CORRELATION_SYMBOLS"] = "BTCUSDT,ETHUSDT"
    cfg = Config()
    assert cfg.correlation_symbols == ["BTCUSDT", "ETHUSDT"]


def test_config_max_same_direction_default():
    """동일 방향 최대 수 기본값 2."""
    cfg = Config()
    assert cfg.max_same_direction == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `Config` has no `symbols`, `correlation_symbols`, `max_same_direction` attributes

**Step 3: Implement Config changes**

`src/config.py`를 수정:

```python
@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "XRPUSDT"
    symbols: list = None          # NEW
    correlation_symbols: list = None  # NEW
    leverage: int = 10
    max_positions: int = 3
    max_same_direction: int = 2   # NEW
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.045
    trailing_stop_pct: float = 0.01
    discord_webhook_url: str = ""
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006
    ml_threshold: float = 0.55

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.margin_max_ratio = float(os.getenv("MARGIN_MAX_RATIO", "0.50"))
        self.margin_min_ratio = float(os.getenv("MARGIN_MIN_RATIO", "0.20"))
        self.margin_decay_rate = float(os.getenv("MARGIN_DECAY_RATE", "0.0006"))
        self.ml_threshold = float(os.getenv("ML_THRESHOLD", "0.55"))
        self.max_same_direction = int(os.getenv("MAX_SAME_DIRECTION", "2"))

        # symbols: SYMBOLS 환경변수 우선, 없으면 SYMBOL에서 변환
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols_env:
            self.symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
        else:
            self.symbols = [self.symbol]

        # correlation_symbols
        corr_env = os.getenv("CORRELATION_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.correlation_symbols = [s.strip() for s in corr_env.split(",") if s.strip()]
```

`.env.example`에 추가:

```
SYMBOLS=XRPUSDT
CORRELATION_SYMBOLS=BTCUSDT,ETHUSDT
MAX_SAME_DIRECTION=2
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Run full test suite to verify no regressions**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS (기존 코드는 `config.symbol`을 여전히 사용 가능하므로 깨지지 않음)

**Step 6: Commit**

```bash
git add src/config.py tests/test_config.py .env.example
git commit -m "feat: add multi-symbol config (symbols list, correlation_symbols, max_same_direction)"
```

---

## Task 2: RiskManager — 공유 싱글턴, asyncio.Lock, 동일 방향 제한

**Files:**
- Modify: `src/risk_manager.py`
- Modify: `tests/test_risk_manager.py`

**Step 1: Write the failing tests**

`tests/test_risk_manager.py`에 추가:

```python
import asyncio


@pytest.fixture
def shared_risk(config):
    config.max_same_direction = 2
    return RiskManager(config)


@pytest.mark.asyncio
async def test_can_open_new_position_async(shared_risk):
    """비동기 포지션 오픈 허용 체크."""
    assert await shared_risk.can_open_new_position("XRPUSDT", "LONG") is True


@pytest.mark.asyncio
async def test_register_and_close_position(shared_risk):
    """포지션 등록 후 닫기."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    assert "XRPUSDT" in shared_risk.open_positions
    await shared_risk.close_position("XRPUSDT", pnl=1.5)
    assert "XRPUSDT" not in shared_risk.open_positions
    assert shared_risk.daily_pnl == 1.5


@pytest.mark.asyncio
async def test_same_symbol_blocked(shared_risk):
    """같은 심볼 중복 진입 차단."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    assert await shared_risk.can_open_new_position("XRPUSDT", "SHORT") is False


@pytest.mark.asyncio
async def test_max_same_direction_limit(shared_risk):
    """같은 방향 2개 초과 차단."""
    await shared_risk.register_position("XRPUSDT", "LONG")
    await shared_risk.register_position("TRXUSDT", "LONG")
    # 3번째 LONG 차단
    assert await shared_risk.can_open_new_position("DOGEUSDT", "LONG") is False
    # SHORT은 허용
    assert await shared_risk.can_open_new_position("DOGEUSDT", "SHORT") is True


@pytest.mark.asyncio
async def test_max_positions_global_limit(shared_risk):
    """전체 포지션 수 한도 초과 차단."""
    shared_risk.config.max_positions = 2
    await shared_risk.register_position("XRPUSDT", "LONG")
    await shared_risk.register_position("TRXUSDT", "SHORT")
    assert await shared_risk.can_open_new_position("DOGEUSDT", "LONG") is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_risk_manager.py -v -k "async or register or same_direction"`
Expected: FAIL — `can_open_new_position`이 sync이고 파라미터가 없음

**Step 3: Implement RiskManager changes**

`src/risk_manager.py` 전체 교체:

```python
import asyncio
from loguru import logger
from src.config import Config


class RiskManager:
    def __init__(self, config: Config, max_daily_loss_pct: float = 0.05):
        self.config = config
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_pnl: float = 0.0
        self.initial_balance: float = 0.0
        self.open_positions: dict[str, str] = {}  # {symbol: side}
        self._lock = asyncio.Lock()

    def is_trading_allowed(self) -> bool:
        """일일 최대 손실 초과 시 거래 중단"""
        if self.initial_balance <= 0:
            return True
        loss_pct = abs(self.daily_pnl) / self.initial_balance
        if self.daily_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            logger.warning(
                f"일일 손실 한도 초과: {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}"
            )
            return False
        return True

    async def can_open_new_position(self, symbol: str, side: str) -> bool:
        """포지션 오픈 가능 여부 (전체 한도 + 중복 진입 + 동일 방향 제한)"""
        async with self._lock:
            if len(self.open_positions) >= self.config.max_positions:
                logger.info(f"최대 포지션 수 도달: {len(self.open_positions)}/{self.config.max_positions}")
                return False
            if symbol in self.open_positions:
                logger.info(f"{symbol} 이미 포지션 보유 중")
                return False
            same_dir = sum(1 for s in self.open_positions.values() if s == side)
            if same_dir >= self.config.max_same_direction:
                logger.info(f"동일 방향({side}) 한도 도달: {same_dir}/{self.config.max_same_direction}")
                return False
            return True

    async def register_position(self, symbol: str, side: str):
        """포지션 등록"""
        async with self._lock:
            self.open_positions[symbol] = side
            logger.info(f"포지션 등록: {symbol} {side} (현재 {len(self.open_positions)}개)")

    async def close_position(self, symbol: str, pnl: float):
        """포지션 닫기 + PnL 기록"""
        async with self._lock:
            self.open_positions.pop(symbol, None)
            self.daily_pnl += pnl
            logger.info(f"포지션 종료: {symbol}, PnL={pnl:+.4f}, 누적={self.daily_pnl:+.4f}")

    def record_pnl(self, pnl: float):
        self.daily_pnl += pnl
        logger.info(f"오늘 누적 PnL: {self.daily_pnl:.4f} USDT")

    def reset_daily(self):
        """매일 자정 초기화"""
        self.daily_pnl = 0.0
        logger.info("일일 PnL 초기화")

    def set_base_balance(self, balance: float) -> None:
        """봇 시작 시 기준 잔고 설정"""
        self.initial_balance = balance

    def get_dynamic_margin_ratio(self, balance: float) -> float:
        """잔고에 따라 선형 감소하는 증거금 비율 반환"""
        ratio = self.config.margin_max_ratio - (
            (balance - self.initial_balance) * self.config.margin_decay_rate
        )
        return max(self.config.margin_min_ratio, min(self.config.margin_max_ratio, ratio))
```

주요 변경:
- `open_positions: list` → `dict[str, str]` (심볼→방향 매핑)
- `can_open_new_position()` → `async` + `symbol`, `side` 파라미터
- `register_position()`, `close_position()` 새 메서드 추가
- `asyncio.Lock()` 동시성 보호

**Step 4: Fix existing tests that break**

기존 테스트에서 `can_open_new_position()` 호출 방식이 바뀌었으므로 수정:

`tests/test_risk_manager.py`의 `test_position_size_capped`를 수정:

```python
@pytest.mark.asyncio
async def test_position_size_capped(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    await rm.register_position("XRPUSDT", "LONG")
    await rm.register_position("TRXUSDT", "SHORT")
    await rm.register_position("DOGEUSDT", "LONG")
    assert await rm.can_open_new_position("SOLUSDT", "SHORT") is False
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_risk_manager.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: `test_bot.py`에서 `can_open_new_position()` 호출이 깨질 수 있음 — Task 4에서 수정할 것이므로 지금은 bot 테스트 실패 허용

**Step 7: Commit**

```bash
git add src/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: shared RiskManager with async lock, same-direction limit, per-symbol tracking"
```

---

## Task 3: Exchange — `config.symbol` → `self.symbol` 분리

**Files:**
- Modify: `src/exchange.py`
- Modify: `tests/test_exchange.py`

**Step 1: Write the failing test**

`tests/test_exchange.py`에 추가:

```python
def test_exchange_uses_own_symbol():
    """Exchange 클라이언트가 config.symbol 대신 생성자의 symbol을 사용한다."""
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SYMBOL": "XRPUSDT",
    })
    config = Config()
    with patch("src.exchange.Client"):
        client = BinanceFuturesClient(config, symbol="TRXUSDT")
    assert client.symbol == "TRXUSDT"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_exchange.py::test_exchange_uses_own_symbol -v`
Expected: FAIL — `__init__` doesn't accept `symbol` parameter

**Step 3: Implement Exchange changes**

`src/exchange.py` 생성자 변경:

```python
class BinanceFuturesClient:
    def __init__(self, config: Config, symbol: str = None):
        self.config = config
        self.symbol = symbol or config.symbol
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
        )
```

모든 `self.config.symbol` 참조를 `self.symbol`로 교체 (9곳):
- Line 34: `set_leverage` → `symbol=self.symbol`
- Line 71: `place_order` params → `symbol=self.symbol`
- Line 101: `_place_algo_order` params → `symbol=self.symbol`
- Line 123: `get_position` → `symbol=self.symbol`
- Line 137: `cancel_all_orders` 일반 → `symbol=self.symbol`
- Line 144: `cancel_all_orders` algo → `symbol=self.symbol`
- Line 156: `get_open_interest` → `symbol=self.symbol`
- Line 169: `get_funding_rate` → `symbol=self.symbol`
- Line 183: `get_oi_history` → `symbol=self.symbol`

**Step 4: Fix existing test fixtures**

기존 `exchange` 픽스처에서 `BinanceFuturesClient.__new__`를 사용하는 곳에 `c.symbol` 설정 추가:

```python
@pytest.fixture
def client():
    config = Config()
    config.leverage = 10
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.config = config
    c.symbol = config.symbol  # NEW
    return c


@pytest.fixture
def exchange():
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
    })
    config = Config()
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.config = config
    c.symbol = config.symbol  # NEW
    c.client = MagicMock()
    return c
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_exchange.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/exchange.py tests/test_exchange.py
git commit -m "feat: exchange client accepts explicit symbol parameter, removes config.symbol dependency"
```

---

## Task 4: TradingBot — 생성자에 `symbol`, `risk` 주입

**Files:**
- Modify: `src/bot.py`
- Modify: `tests/test_bot.py`

**Step 1: Write the failing tests**

`tests/test_bot.py`에 추가:

```python
def test_bot_accepts_symbol_and_risk(config):
    """TradingBot이 symbol과 risk를 외부에서 주입받을 수 있다."""
    from src.risk_manager import RiskManager
    risk = RiskManager(config)
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config, symbol="TRXUSDT", risk=risk)
    assert bot.symbol == "TRXUSDT"
    assert bot.risk is risk


def test_bot_stream_uses_injected_symbol(config):
    """봇의 stream이 주입된 심볼을 primary로 사용한다."""
    from src.risk_manager import RiskManager
    risk = RiskManager(config)
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config, symbol="DOGEUSDT", risk=risk)
    assert "dogeusdt" in bot.stream.buffers


def test_bot_ml_filter_uses_symbol_model_dir(config):
    """봇의 MLFilter가 심볼별 모델 디렉토리를 사용한다."""
    from src.risk_manager import RiskManager
    risk = RiskManager(config)
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config, symbol="TRXUSDT", risk=risk)
    assert "trxusdt" in str(bot.ml_filter._onnx_path)
    assert "trxusdt" in str(bot.ml_filter._lgbm_path)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bot.py -v -k "accepts_symbol or injected_symbol or symbol_model_dir"`
Expected: FAIL — `TradingBot.__init__` doesn't accept `symbol` or `risk`

**Step 3: Implement TradingBot changes**

`src/bot.py`의 `__init__` 변경:

```python
class TradingBot:
    def __init__(self, config: Config, symbol: str = None, risk: RiskManager = None):
        self.config = config
        self.symbol = symbol or config.symbol
        self.exchange = BinanceFuturesClient(config, symbol=self.symbol)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = risk or RiskManager(config)
        self.ml_filter = MLFilter(
            onnx_path=f"models/{self.symbol.lower()}/mlx_filter.weights.onnx",
            lgbm_path=f"models/{self.symbol.lower()}/lgbm_filter.pkl",
            threshold=config.ml_threshold,
        )
        self.current_trade_side: str | None = None
        self._entry_price: float | None = None
        self._entry_quantity: float | None = None
        self._is_reentering: bool = False
        self._prev_oi: float | None = None
        self._oi_history: deque = deque(maxlen=5)
        self._latest_ret_1: float = 0.0
        self.stream = MultiSymbolStream(
            symbols=[self.symbol] + config.correlation_symbols,
            interval="15m",
            on_candle=self._on_candle_closed,
        )
```

`_on_candle_closed` 변경:

```python
    async def _on_candle_closed(self, candle: dict):
        primary_df = self.stream.get_dataframe(self.symbol)
        btc_df = self.stream.get_dataframe("BTCUSDT")
        eth_df = self.stream.get_dataframe("ETHUSDT")
        if primary_df is not None:
            await self.process_candle(primary_df, btc_df=btc_df, eth_df=eth_df)
```

`process_candle`에서 `can_open_new_position` 호출 변경 (2곳):

```python
# Line ~138 (신규 진입):
if not await self.risk.can_open_new_position(self.symbol, raw_signal):
    logger.info(f"[{self.symbol}] 포지션 오픈 불가")
    return

# Line ~322 (_close_and_reenter 내):
if not await self.risk.can_open_new_position(self.symbol, signal):
    logger.info(f"[{self.symbol}] 최대 포지션 수 도달 — 재진입 건너뜀")
    return
```

`_open_position`에서 `register_position` 호출 추가:

```python
    async def _open_position(self, signal: str, df):
        balance = await self.exchange.get_balance()
        # 심볼 수로 마진 균등 배분
        num_symbols = len(self.config.symbols)
        per_symbol_balance = balance / num_symbols
        price = df["close"].iloc[-1]
        margin_ratio = self.risk.get_dynamic_margin_ratio(balance)
        quantity = self.exchange.calculate_quantity(
            balance=per_symbol_balance, price=price,
            leverage=self.config.leverage, margin_ratio=margin_ratio,
        )
        # ... 기존 로직 ...

        # 포지션 등록
        await self.risk.register_position(self.symbol, signal)
        self.current_trade_side = signal
        # ... 나머지 동일 ...
```

`_on_position_closed`에서 `close_position` 호출:

```python
    async def _on_position_closed(self, net_pnl, close_reason, exit_price):
        # ... 기존 PnL 계산 로직 ...
        await self.risk.close_position(self.symbol, net_pnl)
        # record_pnl 제거 (close_position 내에서 처리)
        # ... 나머지 동일 ...
```

모든 `self.config.symbol` 참조를 `self.symbol`로 교체 (6곳):
- Line 31 → `self.symbol` (stream symbols)
- Line 37 → `self.symbol` (get_dataframe)
- Line 197 → `self.symbol` (notify_open)
- Line 251 → `self.symbol` (notify_close)
- Line 340 → `self.symbol` (run 로그)
- Line 348 → `self.symbol` (UserDataStream)

`run()` 메서드의 로그도 변경:

```python
    async def run(self):
        logger.info(f"[{self.symbol}] 봇 시작, 레버리지 {self.config.leverage}x")
        # ... 나머지 동일, self.config.symbol → self.symbol ...
```

**Step 4: Fix existing bot tests**

기존 `tests/test_bot.py`의 모든 `TradingBot(config)` 호출은 하위호환되므로 그대로 동작.
단, `risk.can_open_new_position` 호출이 async로 바뀌었으므로 mock 수정 필요:

`test_close_and_reenter_calls_open_when_ml_passes`:
```python
    bot.risk = MagicMock()
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
```

`test_close_and_reenter_skips_open_when_max_positions_reached`:
```python
    bot.risk = MagicMock()
    bot.risk.can_open_new_position = AsyncMock(return_value=False)
```

`test_bot_processes_signal`에서 `bot.risk`도 mock:
```python
    bot.risk = MagicMock()
    bot.risk.is_trading_allowed.return_value = True
    bot.risk.can_open_new_position = AsyncMock(return_value=True)
    bot.risk.register_position = AsyncMock()
    bot.risk.get_dynamic_margin_ratio.return_value = 0.50
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_bot.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: TradingBot accepts symbol and shared RiskManager, removes config.symbol dependency"
```

---

## Task 5: main.py — 심볼별 봇 인스턴스 생성 + asyncio.gather

**Files:**
- Modify: `main.py`

**Step 1: Implement main.py changes**

```python
import asyncio
from dotenv import load_dotenv
from src.config import Config
from src.bot import TradingBot
from src.risk_manager import RiskManager
from src.logger_setup import setup_logger
from loguru import logger

load_dotenv()


async def main():
    setup_logger(log_level="INFO")
    config = Config()
    risk = RiskManager(config)

    bots = []
    for symbol in config.symbols:
        bot = TradingBot(config, symbol=symbol, risk=risk)
        bots.append(bot)

    logger.info(f"멀티심볼 봇 시작: {config.symbols} ({len(bots)}개 인스턴스)")
    await asyncio.gather(*[bot.run() for bot in bots])


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Run full test suite to verify no regressions**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main.py spawns per-symbol TradingBot instances with shared RiskManager"
```

---

## Task 6: MLFilter — 심볼별 모델 디렉토리 폴백

**Files:**
- Modify: `src/ml_filter.py`

**Step 1: Implement MLFilter path fallback**

MLFilter는 이미 `onnx_path`/`lgbm_path`를 생성자에서 받으므로, bot.py에서 심볼별 경로를 주입하면 된다 (Task 4에서 완료).

다만 기존 `models/lgbm_filter.pkl` 경로에 모델이 있는 경우(단일 심볼 환경)에도 동작하도록, 심볼별 디렉토리에 모델이 없으면 루트 `models/`에서 폴백하는 로직을 `bot.py`에 추가:

`src/bot.py`의 `__init__`에서:

```python
        # 심볼별 모델 디렉토리. 없으면 기존 models/ 루트로 폴백
        symbol_model_dir = Path(f"models/{self.symbol.lower()}")
        if symbol_model_dir.exists():
            onnx_path = str(symbol_model_dir / "mlx_filter.weights.onnx")
            lgbm_path = str(symbol_model_dir / "lgbm_filter.pkl")
        else:
            onnx_path = "models/mlx_filter.weights.onnx"
            lgbm_path = "models/lgbm_filter.pkl"
        self.ml_filter = MLFilter(
            onnx_path=onnx_path,
            lgbm_path=lgbm_path,
            threshold=config.ml_threshold,
        )
```

**Step 2: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/bot.py
git commit -m "feat: MLFilter falls back to models/ root if symbol-specific dir not found"
```

---

## Task 7: 학습 스크립트 — `--symbol` / `--all` CLI 통일

**Files:**
- Modify: `scripts/fetch_history.py`
- Modify: `scripts/train_model.py`
- Modify: `scripts/tune_hyperparams.py`
- Modify: `scripts/train_and_deploy.sh`
- Modify: `scripts/deploy_model.sh`

### Step 1: fetch_history.py — `--symbol` 단일 심볼 + 출력 경로 자동 결정

`scripts/fetch_history.py`의 argparse에 `--symbol` 추가 및 `--output` 자동 결정:

현재 사용법: `--symbols XRPUSDT BTCUSDT ETHUSDT --output data/combined_15m.parquet`

변경 후:
```python
parser.add_argument("--symbol", type=str, default=None,
                    help="단일 거래 심볼 (예: TRXUSDT). 상관관계 심볼 자동 추가")
```

`--symbol TRXUSDT` 지정 시:
- `symbols = ["TRXUSDT", "BTCUSDT", "ETHUSDT"]`
- `output = "data/trxusdt/combined_15m.parquet"` (자동)

`--symbols XRPUSDT BTCUSDT ETHUSDT` (기존 방식)도 유지.

### Step 2: train_model.py — `--symbol` 추가

```python
parser.add_argument("--symbol", type=str, default=None,
                    help="학습 대상 심볼 (예: TRXUSDT). data/{symbol}/ 에서 데이터 로드, models/{symbol}/ 에 저장")
```

`--symbol TRXUSDT` 지정 시:
- 데이터: `data/trxusdt/combined_15m.parquet`
- 모델: `models/trxusdt/lgbm_filter.pkl`
- 로그: `models/trxusdt/training_log.json`

`--data` 옵션이 명시되면 그것을 우선.

### Step 3: tune_hyperparams.py — `--symbol` 추가

train_model.py와 동일한 패턴. `--symbol`이 지정되면:
- 데이터: `data/{symbol}/combined_15m.parquet`
- 결과: `models/{symbol}/tune_results_*.json`
- active params: `models/{symbol}/active_lgbm_params.json`

### Step 4: train_and_deploy.sh — `--symbol` / `--all` 지원

```bash
# 사용법:
#   bash scripts/train_and_deploy.sh [mlx|lgbm] [--symbol TRXUSDT]
#   bash scripts/train_and_deploy.sh [mlx|lgbm] --all
#   bash scripts/train_and_deploy.sh              # --all과 동일 (기본값)
```

`--symbol` 지정 시: 해당 심볼만 fetch → train → deploy
`--all` 또는 인자 없음: `SYMBOLS` 환경변수의 모든 심볼 순차 처리

핵심 로직:
```bash
if [ -n "$SYMBOL_ARG" ]; then
    TARGETS=("$SYMBOL_ARG")
else
    # .env에서 SYMBOLS 로드
    TARGETS=($(python -c "from src.config import Config; c=Config(); print(' '.join(c.symbols))"))
fi

for SYM in "${TARGETS[@]}"; do
    SYM_LOWER=$(echo "$SYM" | tr '[:upper:]' '[:lower:]')
    mkdir -p "data/$SYM_LOWER" "models/$SYM_LOWER"

    # fetch
    python scripts/fetch_history.py --symbol "$SYM" ...

    # train
    python scripts/train_model.py --symbol "$SYM" ...

    # deploy
    bash scripts/deploy_model.sh "$BACKEND" --symbol "$SYM"
done
```

### Step 5: deploy_model.sh — `--symbol` 지원

```bash
# 사용법: bash scripts/deploy_model.sh [lgbm|mlx] [--symbol TRXUSDT]
```

`--symbol` 지정 시:
- 로컬: `models/{symbol}/lgbm_filter.pkl`
- 원격: `$LXC_MODELS_PATH/{symbol}/lgbm_filter.pkl`

### Step 6: Run full test suite

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS (스크립트 변경은 unit test에 영향 없음)

### Step 7: Smoke test 스크립트

```bash
# fetch만 소량 테스트
python scripts/fetch_history.py --symbol TRXUSDT --interval 15m --days 1
ls data/trxusdt/combined_15m.parquet  # 파일 존재 확인
```

### Step 8: Commit

```bash
git add scripts/fetch_history.py scripts/train_model.py scripts/tune_hyperparams.py scripts/train_and_deploy.sh scripts/deploy_model.sh
git commit -m "feat: add --symbol/--all CLI to all training scripts for per-symbol pipeline"
```

---

## Task 8: 디렉토리 구조 생성 + .env.example 업데이트

**Files:**
- Create: `models/xrpusdt/.gitkeep`
- Create: `models/trxusdt/.gitkeep`
- Create: `models/dogeusdt/.gitkeep`
- Create: `data/xrpusdt/.gitkeep`
- Create: `data/trxusdt/.gitkeep`
- Create: `data/dogeusdt/.gitkeep`
- Modify: `.env.example`

**Step 1: Create directory structure**

```bash
mkdir -p models/{xrpusdt,trxusdt,dogeusdt}
mkdir -p data/{xrpusdt,trxusdt,dogeusdt}
touch models/{xrpusdt,trxusdt,dogeusdt}/.gitkeep
touch data/{xrpusdt,trxusdt,dogeusdt}/.gitkeep
```

**Step 2: Update .env.example**

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
SYMBOLS=XRPUSDT
CORRELATION_SYMBOLS=BTCUSDT,ETHUSDT
LEVERAGE=10
RISK_PER_TRADE=0.02
DISCORD_WEBHOOK_URL=
ML_THRESHOLD=0.55
MAX_SAME_DIRECTION=2
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
```

**Step 3: Commit**

```bash
git add models/ data/ .env.example
git commit -m "feat: add per-symbol model/data directories and update .env.example"
```

---

## Task 9: 기존 모델 마이그레이션 안내 + 문서 업데이트

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Architecture 섹션에 멀티심볼 관련 내용 추가:

- `main.py` → `Config` → 심볼별 `TradingBot` 인스턴스 → `asyncio.gather()`
- `RiskManager` 공유 싱글턴 (글로벌 일일 손실 + 동일 방향 제한)
- 모델/데이터 디렉토리: `models/{symbol}/`, `data/{symbol}/`

Common Commands 섹션 업데이트:

```bash
# 단일 심볼 학습
bash scripts/train_and_deploy.sh --symbol TRXUSDT

# 전체 심볼 학습
bash scripts/train_and_deploy.sh --all
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update architecture and commands for multi-symbol trading"
```

---

## 구현 순서 요약

| Task | 내용 | 의존성 |
|------|------|--------|
| 1 | Config: `symbols`, `correlation_symbols`, `max_same_direction` | 없음 |
| 2 | RiskManager: 공유 싱글턴, async Lock, 동일 방향 제한 | Task 1 |
| 3 | Exchange: `self.symbol` 분리 | 없음 (Task 1과 병렬 가능) |
| 4 | TradingBot: `symbol`, `risk` 주입, `config.symbol` 제거 | Task 1, 2, 3 |
| 5 | main.py: 심볼별 봇 생성 + gather | Task 4 |
| 6 | MLFilter: 심볼별 모델 디렉토리 폴백 | Task 4 |
| 7 | 학습 스크립트: `--symbol` / `--all` CLI | Task 1 |
| 8 | 디렉토리 구조 + .env.example | 없음 |
| 9 | 문서 업데이트 | 전체 완료 후 |

각 태스크 완료 후 기존 XRP 단일 모드에서 전체 테스트를 통과해야 한다.
