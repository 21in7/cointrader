# 동적 증거금 비율 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 잔고의 50%를 증거금으로 사용하되, 잔고가 늘수록 비율이 선형으로 감소하는 동적 포지션 크기 계산 도입

**Architecture:** `RiskManager`에 `get_dynamic_margin_ratio(balance)` 메서드를 추가하고, `bot.py`에서 포지션 진입 전 호출한다. `exchange.py`의 `calculate_quantity`는 `margin_ratio` 파라미터를 받아 기존 `risk_per_trade` 로직을 대체한다. 봇 시작 시 바이낸스 API로 실제 잔고를 조회하여 기준값(`base_balance`)으로 저장한다.

**Tech Stack:** Python 3.11, python-binance, loguru, pytest, python-dotenv

---

## 사전 확인

- 현재 `.env`: `RISK_PER_TRADE=0.02` 존재
- 현재 `Config.risk_per_trade: float = 0.02` 존재
- 현재 `calculate_quantity`는 `balance * risk_per_trade * leverage` 로직 사용
- 테스트 파일 위치: `tests/` 디렉토리 (없으면 생성)

---

### Task 1: Config에 동적 증거금 파라미터 추가

**Files:**
- Modify: `src/config.py`
- Modify: `.env`

**Step 1: `.env`에 새 파라미터 추가**

`.env` 파일 하단에 추가:

```
MARGIN_MAX_RATIO=0.50
MARGIN_MIN_RATIO=0.20
MARGIN_DECAY_RATE=0.0006
```

기존 `RISK_PER_TRADE=0.02` 줄은 삭제.

**Step 2: `src/config.py` 수정**

`Config` 데이터클래스에 필드 추가, `risk_per_trade` 필드 제거:

```python
@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "XRPUSDT"
    leverage: int = 10
    max_positions: int = 3
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.045
    trailing_stop_pct: float = 0.01
    discord_webhook_url: str = ""
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.margin_max_ratio = float(os.getenv("MARGIN_MAX_RATIO", "0.50"))
        self.margin_min_ratio = float(os.getenv("MARGIN_MIN_RATIO", "0.20"))
        self.margin_decay_rate = float(os.getenv("MARGIN_DECAY_RATE", "0.0006"))
```

**Step 3: Commit**

```bash
git add src/config.py .env
git commit -m "feat: add dynamic margin ratio config params"
```

---

### Task 2: RiskManager에 동적 비율 메서드 추가

**Files:**
- Modify: `src/risk_manager.py`
- Create: `tests/test_risk_manager.py`

**Step 1: 실패하는 테스트 작성**

`tests/test_risk_manager.py` 생성:

```python
import pytest
from src.config import Config
from src.risk_manager import RiskManager


@pytest.fixture
def config():
    c = Config()
    c.margin_max_ratio = 0.50
    c.margin_min_ratio = 0.20
    c.margin_decay_rate = 0.0006
    return c


@pytest.fixture
def risk(config):
    r = RiskManager(config)
    r.set_base_balance(22.0)
    return r


def test_set_base_balance(risk):
    assert risk.initial_balance == 22.0


def test_ratio_at_base_balance(risk):
    """기준 잔고에서 최대 비율(50%) 반환"""
    ratio = risk.get_dynamic_margin_ratio(22.0)
    assert ratio == pytest.approx(0.50, abs=1e-6)


def test_ratio_decreases_as_balance_grows(risk):
    """잔고가 늘수록 비율 감소"""
    ratio_100 = risk.get_dynamic_margin_ratio(100.0)
    ratio_300 = risk.get_dynamic_margin_ratio(300.0)
    assert ratio_100 < 0.50
    assert ratio_300 < ratio_100


def test_ratio_clamped_at_min(risk):
    """잔고가 매우 커도 최소 비율(20%) 이하로 내려가지 않음"""
    ratio = risk.get_dynamic_margin_ratio(10000.0)
    assert ratio == pytest.approx(0.20, abs=1e-6)


def test_ratio_clamped_at_max(risk):
    """잔고가 기준보다 작아도 최대 비율(50%) 초과하지 않음"""
    ratio = risk.get_dynamic_margin_ratio(5.0)
    assert ratio == pytest.approx(0.50, abs=1e-6)
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_risk_manager.py -v
```

Expected: `AttributeError: 'RiskManager' object has no attribute 'set_base_balance'`

**Step 3: `src/risk_manager.py` 수정**

기존 코드에 메서드 2개 추가:

```python
def set_base_balance(self, balance: float) -> None:
    """봇 시작 시 기준 잔고 설정 (동적 비율 계산 기준점)"""
    self.initial_balance = balance

def get_dynamic_margin_ratio(self, balance: float) -> float:
    """잔고에 따라 선형 감소하는 증거금 비율 반환"""
    ratio = self.config.margin_max_ratio - (
        (balance - self.initial_balance) * self.config.margin_decay_rate
    )
    return max(self.config.margin_min_ratio, min(self.config.margin_max_ratio, ratio))
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_risk_manager.py -v
```

Expected: 5개 테스트 모두 PASS

**Step 5: Commit**

```bash
git add src/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: add get_dynamic_margin_ratio to RiskManager"
```

---

### Task 3: exchange.py의 calculate_quantity 수정

**Files:**
- Modify: `src/exchange.py:18-29`
- Create: `tests/test_exchange.py`

**Step 1: 실패하는 테스트 작성**

`tests/test_exchange.py` 생성:

```python
import pytest
from unittest.mock import MagicMock
from src.config import Config
from src.exchange import BinanceFuturesClient


@pytest.fixture
def client():
    config = Config()
    config.leverage = 10
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c.config = config
    return c


def test_calculate_quantity_basic(client):
    """잔고 22, 비율 50%, 레버리지 10배 → 명목금액 110, XRP 가격 2.5 → 수량 44.0"""
    qty = client.calculate_quantity(balance=22.0, price=2.5, leverage=10, margin_ratio=0.50)
    # 명목금액 = 22 * 0.5 * 10 = 110, 수량 = 110 / 2.5 = 44.0
    assert qty == pytest.approx(44.0, abs=0.1)


def test_calculate_quantity_min_notional(client):
    """명목금액이 최소(5 USDT) 미만이면 최소값으로 올림"""
    qty = client.calculate_quantity(balance=1.0, price=2.5, leverage=1, margin_ratio=0.01)
    # 명목금액 = 1 * 0.01 * 1 = 0.01 < 5 → 최소 5 USDT
    assert qty * 2.5 >= 5.0


def test_calculate_quantity_zero_balance(client):
    """잔고 0이면 최소 명목금액 기반 수량 반환"""
    qty = client.calculate_quantity(balance=0.0, price=2.5, leverage=10, margin_ratio=0.50)
    assert qty > 0
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_exchange.py -v
```

Expected: `TypeError: calculate_quantity() got an unexpected keyword argument 'margin_ratio'`

**Step 3: `src/exchange.py` 수정**

`calculate_quantity` 메서드를 아래로 교체:

```python
def calculate_quantity(self, balance: float, price: float, leverage: int, margin_ratio: float) -> float:
    """동적 증거금 비율 기반 포지션 크기 계산 (최소 명목금액 $5 보장)"""
    notional = balance * margin_ratio * leverage
    if notional < self.MIN_NOTIONAL:
        notional = self.MIN_NOTIONAL
    quantity = notional / price
    qty_rounded = round(quantity, 1)
    if qty_rounded * price < self.MIN_NOTIONAL:
        qty_rounded = round(self.MIN_NOTIONAL / price + 0.05, 1)
    return qty_rounded
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_exchange.py -v
```

Expected: 3개 테스트 모두 PASS

**Step 5: Commit**

```bash
git add src/exchange.py tests/test_exchange.py
git commit -m "feat: replace risk_per_trade with margin_ratio in calculate_quantity"
```

---

### Task 4: bot.py 연결

**Files:**
- Modify: `src/bot.py:85-99` (`_open_position`)
- Modify: `src/bot.py:165-172` (`run`)

**Step 1: `run()` 메서드에 `set_base_balance` 호출 추가**

`run()` 메서드를 아래로 교체:

```python
async def run(self):
    logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
    await self._recover_position()
    balance = await self.exchange.get_balance()
    self.risk.set_base_balance(balance)
    logger.info(f"기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")
    await self.stream.start(
        api_key=self.config.api_key,
        api_secret=self.config.api_secret,
    )
```

**Step 2: `_open_position()` 메서드에 동적 비율 적용**

`_open_position()` 내부 `quantity` 계산 부분을 수정:

```python
async def _open_position(self, signal: str, df):
    balance = await self.exchange.get_balance()
    price = df["close"].iloc[-1]
    margin_ratio = self.risk.get_dynamic_margin_ratio(balance)
    quantity = self.exchange.calculate_quantity(
        balance=balance, price=price, leverage=self.config.leverage, margin_ratio=margin_ratio
    )
    logger.info(f"포지션 크기: 잔고={balance:.2f} USDT, 증거금비율={margin_ratio:.1%}, 수량={quantity}")
    # 이하 기존 코드 유지 (stop_loss, take_profit, place_order 등)
```

**Step 3: 전체 테스트 실행**

```bash
pytest tests/ -v
```

Expected: 전체 PASS

**Step 4: Commit**

```bash
git add src/bot.py
git commit -m "feat: apply dynamic margin ratio in bot position sizing"
```

---

### Task 5: 기존 risk_per_trade 참조 정리

**Files:**
- Search: 프로젝트 전체에서 `risk_per_trade` 참조 확인

**Step 1: 잔여 참조 검색**

```bash
grep -r "risk_per_trade" src/ tests/ .env
```

Expected: 결과 없음 (이미 모두 제거됨)

남아있는 경우 해당 파일에서 제거.

**Step 2: 전체 테스트 최종 확인**

```bash
pytest tests/ -v
```

Expected: 전체 PASS

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove unused risk_per_trade references"
```

---

## 검증 체크리스트

- [ ] `pytest tests/test_risk_manager.py` — 5개 PASS
- [ ] `pytest tests/test_exchange.py` — 3개 PASS
- [ ] `pytest tests/` — 전체 PASS
- [ ] `.env`에 `MARGIN_MAX_RATIO`, `MARGIN_MIN_RATIO`, `MARGIN_DECAY_RATE` 존재
- [ ] `.env`에 `RISK_PER_TRADE` 없음
- [ ] 봇 시작 로그에 "기준 잔고 설정: XX USDT" 출력
- [ ] 포지션 진입 로그에 "증거금비율=50.0%" 출력 (잔고 22 USDT 기준)
