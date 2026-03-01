# XRP 선물 자동매매 시스템 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 바이낸스 API를 이용해 XRP/USDT 선물 시장에서 다중 기술 지표 기반 공격적 자동매매 봇을 구축한다.

**Architecture:** WebSocket으로 실시간 가격/캔들 데이터를 수신하고, RSI·MACD·볼린저밴드·EMA·ATR 등 복합 지표를 계산해 진입/청산 신호를 생성한다. 레버리지 5~20배, 분할 진입/청산, 손절/익절 자동화를 포함하며 Supabase에 거래 이력을 저장한다.

**Tech Stack:** Python 3.11+, python-binance, pandas, pandas-ta, asyncio, websockets, supabase-py, python-dotenv, pytest, pytest-asyncio

---

## 사전 준비

### 환경 변수 설정 (`.env`)
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true          # 처음엔 테스트넷으로
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SYMBOL=XRPUSDT
LEVERAGE=10
RISK_PER_TRADE=0.02           # 계좌의 2%
```

---

## Task 1: 프로젝트 초기 설정

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: requirements.txt 작성**

```
python-binance==1.0.19
pandas==2.2.0
pandas-ta==0.3.14b
python-dotenv==1.0.0
supabase==2.3.4
pytest==8.0.0
pytest-asyncio==0.23.4
aiohttp==3.9.3
websockets==12.0
loguru==0.7.2
```

**Step 2: 디렉토리 구조 생성**

```bash
mkdir -p src tests
touch src/__init__.py tests/__init__.py
```

**Step 3: .gitignore 작성**

```
.env
__pycache__/
*.pyc
.pytest_cache/
logs/
*.log
```

**Step 4: .env.example 작성**

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=true
SUPABASE_URL=
SUPABASE_KEY=
SYMBOL=XRPUSDT
LEVERAGE=10
RISK_PER_TRADE=0.02
```

**Step 5: 의존성 설치**

```bash
pip install -r requirements.txt
```
Expected: 모든 패키지 설치 성공

**Step 6: Commit**

```bash
git init
git add .
git commit -m "chore: 프로젝트 초기 설정 및 의존성 추가"
```

---

## Task 2: 설정(Config) 모듈

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_config.py
import os
import pytest
from src.config import Config

def test_config_loads_symbol():
    os.environ["SYMBOL"] = "XRPUSDT"
    os.environ["LEVERAGE"] = "10"
    os.environ["RISK_PER_TRADE"] = "0.02"
    cfg = Config()
    assert cfg.symbol == "XRPUSDT"
    assert cfg.leverage == 10
    assert cfg.risk_per_trade == 0.02

def test_config_testnet_default_true():
    os.environ["BINANCE_TESTNET"] = "true"
    cfg = Config()
    assert cfg.testnet is True
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL - "cannot import name 'Config'"

**Step 3: Config 구현**

```python
# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    symbol: str = "XRPUSDT"
    leverage: int = 10
    risk_per_trade: float = 0.02
    max_positions: int = 3
    stop_loss_pct: float = 0.015    # 1.5%
    take_profit_pct: float = 0.045  # 4.5% (3:1 RR)
    trailing_stop_pct: float = 0.01 # 1%

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_config.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: Config 모듈 추가"
```

---

## Task 3: 바이낸스 클라이언트 래퍼

**Files:**
- Create: `src/exchange.py`
- Create: `tests/test_exchange.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_exchange.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.exchange import BinanceFuturesClient
from src.config import Config

@pytest.fixture
def config():
    import os
    os.environ.update({
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "BINANCE_TESTNET": "true",
        "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10",
        "RISK_PER_TRADE": "0.02",
    })
    return Config()

@pytest.mark.asyncio
async def test_set_leverage(config):
    client = BinanceFuturesClient(config)
    with patch.object(client.client.futures_change_leverage, '__call__',
                      return_value={"leverage": 10}):
        result = await client.set_leverage(10)
        assert result is not None

def test_calculate_quantity(config):
    client = BinanceFuturesClient(config)
    # 잔고 1000 USDT, 리스크 2%, 레버리지 10, 가격 0.5
    qty = client.calculate_quantity(
        balance=1000.0, price=0.5, leverage=10
    )
    # 1000 * 0.02 * 10 / 0.5 = 400
    assert qty == pytest.approx(400.0, rel=0.01)
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_exchange.py -v
```
Expected: FAIL

**Step 3: BinanceFuturesClient 구현**

```python
# src/exchange.py
import asyncio
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from src.config import Config


class BinanceFuturesClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
        )

    def calculate_quantity(self, balance: float, price: float, leverage: int) -> float:
        """리스크 기반 포지션 크기 계산"""
        risk_amount = balance * self.config.risk_per_trade
        notional = risk_amount * leverage
        quantity = notional / price
        return round(quantity, 1)

    async def set_leverage(self, leverage: int) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.futures_change_leverage(
                symbol=self.config.symbol, leverage=leverage
            ),
        )

    async def get_balance(self) -> float:
        loop = asyncio.get_event_loop()
        balances = await loop.run_in_executor(
            None, self.client.futures_account_balance
        )
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    async def place_order(
        self,
        side: str,          # "BUY" | "SELL"
        quantity: float,
        order_type: str = "MARKET",
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False,
    ) -> dict:
        loop = asyncio.get_event_loop()
        params = dict(
            symbol=self.config.symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            reduceOnly=reduce_only,
        )
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if stop_price:
            params["stopPrice"] = stop_price
        try:
            return await loop.run_in_executor(
                None, lambda: self.client.futures_create_order(**params)
            )
        except BinanceAPIException as e:
            logger.error(f"주문 실패: {e}")
            raise

    async def get_position(self) -> dict | None:
        loop = asyncio.get_event_loop()
        positions = await loop.run_in_executor(
            None, lambda: self.client.futures_position_information(
                symbol=self.config.symbol
            )
        )
        for p in positions:
            if float(p["positionAmt"]) != 0:
                return p
        return None

    async def cancel_all_orders(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.futures_cancel_all_open_orders(
                symbol=self.config.symbol
            )
        )
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_exchange.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/exchange.py tests/test_exchange.py
git commit -m "feat: BinanceFuturesClient 구현"
```

---

## Task 4: 기술 지표 계산 모듈

**Files:**
- Create: `src/indicators.py`
- Create: `tests/test_indicators.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_indicators.py
import pandas as pd
import numpy as np
import pytest
from src.indicators import Indicators

@pytest.fixture
def sample_df():
    """100개 캔들 샘플 데이터"""
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 0.5
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.001),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    })
    return df

def test_rsi_range(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "rsi" in df.columns
    valid = df["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()

def test_macd_columns(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns

def test_bollinger_bands(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns
    valid = df.dropna()
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()

def test_signal_returns_direction(sample_df):
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    signal = ind.get_signal(df)
    assert signal in ("LONG", "SHORT", "HOLD")
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_indicators.py -v
```
Expected: FAIL

**Step 3: Indicators 구현**

```python
# src/indicators.py
import pandas as pd
import pandas_ta as ta
from loguru import logger


class Indicators:
    """
    복합 기술 지표 계산 및 매매 신호 생성.
    공격적 전략: 여러 지표가 동시에 같은 방향을 가리킬 때 진입.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_all(self) -> pd.DataFrame:
        df = self.df

        # RSI (14)
        df["rsi"] = ta.rsi(df["close"], length=14)

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"]        = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"]   = macd["MACDh_12_26_9"]

        # 볼린저 밴드 (20, 2)
        bb = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_mid"]   = bb["BBM_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]

        # EMA (9, 21, 50)
        df["ema9"]  = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)

        # ATR (14) - 변동성 기반 손절 계산용
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Stochastic RSI
        stoch = ta.stochrsi(df["close"], length=14)
        df["stoch_k"] = stoch["STOCHRSIk_14_14_3_3"]
        df["stoch_d"] = stoch["STOCHRSId_14_14_3_3"]

        # 거래량 이동평균
        df["vol_ma20"] = ta.sma(df["volume"], length=20)

        return df

    def get_signal(self, df: pd.DataFrame) -> str:
        """
        복합 지표 기반 매매 신호 생성.
        공격적 전략: 3개 이상 지표 일치 시 진입.
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]

        long_signals  = 0
        short_signals = 0

        # 1. RSI
        if last["rsi"] < 35:
            long_signals += 1
        elif last["rsi"] > 65:
            short_signals += 1

        # 2. MACD 크로스
        if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
            long_signals += 2  # 크로스는 강한 신호
        elif prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
            short_signals += 2

        # 3. 볼린저 밴드 돌파
        if last["close"] < last["bb_lower"]:
            long_signals += 1
        elif last["close"] > last["bb_upper"]:
            short_signals += 1

        # 4. EMA 정배열/역배열
        if last["ema9"] > last["ema21"] > last["ema50"]:
            long_signals += 1
        elif last["ema9"] < last["ema21"] < last["ema50"]:
            short_signals += 1

        # 5. Stochastic RSI 과매도/과매수
        if last["stoch_k"] < 20 and last["stoch_k"] > last["stoch_d"]:
            long_signals += 1
        elif last["stoch_k"] > 80 and last["stoch_k"] < last["stoch_d"]:
            short_signals += 1

        # 6. 거래량 확인 (신호 강화)
        vol_surge = last["volume"] > last["vol_ma20"] * 1.5

        threshold = 3
        if long_signals >= threshold and (vol_surge or long_signals >= 4):
            return "LONG"
        elif short_signals >= threshold and (vol_surge or short_signals >= 4):
            return "SHORT"
        return "HOLD"

    def get_atr_stop(self, df: pd.DataFrame, side: str, entry_price: float) -> tuple[float, float]:
        """ATR 기반 손절/익절 가격 반환 (stop_loss, take_profit)"""
        atr = df["atr"].iloc[-1]
        multiplier_sl = 1.5
        multiplier_tp = 3.0
        if side == "LONG":
            stop_loss   = entry_price - atr * multiplier_sl
            take_profit = entry_price + atr * multiplier_tp
        else:
            stop_loss   = entry_price + atr * multiplier_sl
            take_profit = entry_price - atr * multiplier_tp
        return stop_loss, take_profit
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_indicators.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/indicators.py tests/test_indicators.py
git commit -m "feat: 복합 기술 지표 모듈 구현 (RSI/MACD/BB/EMA/ATR/StochRSI)"
```

---

## Task 5: Supabase 거래 이력 저장 모듈

**Files:**
- Create: `src/database.py`
- Create: `tests/test_database.py`
- Create: `supabase/migrations/001_trades.sql`

**Step 1: Supabase 테이블 마이그레이션 SQL 작성**

```sql
-- supabase/migrations/001_trades.sql
create table if not exists trades (
  id          uuid primary key default gen_random_uuid(),
  symbol      text not null,
  side        text not null,          -- 'LONG' | 'SHORT'
  entry_price numeric(18,8) not null,
  exit_price  numeric(18,8),
  quantity    numeric(18,4) not null,
  leverage    int not null,
  pnl         numeric(18,4),
  status      text not null default 'OPEN',  -- 'OPEN' | 'CLOSED' | 'CANCELLED'
  signal_data jsonb,                  -- 진입 시 지표 스냅샷
  opened_at   timestamptz not null default now(),
  closed_at   timestamptz
);

create index on trades (symbol, status);
create index on trades (opened_at desc);
```

Supabase 대시보드 SQL 에디터에서 위 SQL을 실행한다.

**Step 2: 실패 테스트 작성**

```python
# tests/test_database.py
import pytest
from unittest.mock import MagicMock, patch
from src.database import TradeRepository

@pytest.fixture
def mock_repo():
    with patch("src.database.create_client") as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        repo = TradeRepository(url="http://test", key="test_key")
        repo.client = mock_client
        yield repo

def test_save_trade(mock_repo):
    mock_repo.client.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": "abc123"}]
    )
    result = mock_repo.save_trade(
        symbol="XRPUSDT", side="LONG",
        entry_price=0.5, quantity=400.0, leverage=10,
        signal_data={"rsi": 32, "macd_hist": 0.001}
    )
    assert result["id"] == "abc123"

def test_close_trade(mock_repo):
    mock_repo.client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": "abc123", "status": "CLOSED"}]
    )
    result = mock_repo.close_trade(trade_id="abc123", exit_price=0.55, pnl=20.0)
    assert result["status"] == "CLOSED"
```

**Step 3: 테스트 실패 확인**

```bash
pytest tests/test_database.py -v
```
Expected: FAIL

**Step 4: TradeRepository 구현**

```python
# src/database.py
from datetime import datetime, timezone
from supabase import create_client, Client
from loguru import logger


class TradeRepository:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def save_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        signal_data: dict = None,
    ) -> dict:
        data = {
            "symbol":       symbol,
            "side":         side,
            "entry_price":  entry_price,
            "quantity":     quantity,
            "leverage":     leverage,
            "signal_data":  signal_data or {},
            "status":       "OPEN",
            "opened_at":    datetime.now(timezone.utc).isoformat(),
        }
        result = self.client.table("trades").insert(data).execute()
        logger.info(f"거래 저장: {result.data[0]['id']}")
        return result.data[0]

    def close_trade(self, trade_id: str, exit_price: float, pnl: float) -> dict:
        data = {
            "exit_price": exit_price,
            "pnl":        pnl,
            "status":     "CLOSED",
            "closed_at":  datetime.now(timezone.utc).isoformat(),
        }
        result = (
            self.client.table("trades")
            .update(data)
            .eq("id", trade_id)
            .execute()
        )
        logger.info(f"거래 종료: {trade_id}, PnL: {pnl:.4f}")
        return result.data[0]

    def get_open_trades(self, symbol: str) -> list[dict]:
        result = (
            self.client.table("trades")
            .select("*")
            .eq("symbol", symbol)
            .eq("status", "OPEN")
            .execute()
        )
        return result.data
```

**Step 5: 테스트 통과 확인**

```bash
pytest tests/test_database.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/database.py tests/test_database.py supabase/
git commit -m "feat: Supabase 거래 이력 저장 모듈 구현"
```

---

## Task 6: 포지션 관리 및 리스크 매니저

**Files:**
- Create: `src/risk_manager.py`
- Create: `tests/test_risk_manager.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_risk_manager.py
import pytest
from src.risk_manager import RiskManager
from src.config import Config
import os

@pytest.fixture
def config():
    os.environ.update({
        "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
        "BINANCE_TESTNET": "true", "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10", "RISK_PER_TRADE": "0.02",
    })
    return Config()

def test_max_drawdown_check(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    rm.daily_pnl = -60.0
    rm.initial_balance = 1000.0
    assert rm.is_trading_allowed() is False

def test_trading_allowed_normal(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    rm.daily_pnl = -10.0
    rm.initial_balance = 1000.0
    assert rm.is_trading_allowed() is True

def test_position_size_capped(config):
    rm = RiskManager(config, max_daily_loss_pct=0.05)
    # 최대 포지션 수 초과 시 False
    rm.open_positions = ["pos1", "pos2", "pos3"]
    assert rm.can_open_new_position() is False
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_risk_manager.py -v
```
Expected: FAIL

**Step 3: RiskManager 구현**

```python
# src/risk_manager.py
from loguru import logger
from src.config import Config


class RiskManager:
    def __init__(self, config: Config, max_daily_loss_pct: float = 0.05):
        self.config = config
        self.max_daily_loss_pct = max_daily_loss_pct  # 일일 최대 손실 5%
        self.daily_pnl: float = 0.0
        self.initial_balance: float = 0.0
        self.open_positions: list = []

    def is_trading_allowed(self) -> bool:
        """일일 최대 손실 초과 시 거래 중단"""
        if self.initial_balance <= 0:
            return True
        loss_pct = abs(self.daily_pnl) / self.initial_balance
        if self.daily_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            logger.warning(f"일일 손실 한도 초과: {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}")
            return False
        return True

    def can_open_new_position(self) -> bool:
        """최대 동시 포지션 수 체크"""
        return len(self.open_positions) < self.config.max_positions

    def record_pnl(self, pnl: float):
        self.daily_pnl += pnl
        logger.info(f"오늘 누적 PnL: {self.daily_pnl:.4f} USDT")

    def reset_daily(self):
        """매일 자정 초기화"""
        self.daily_pnl = 0.0
        logger.info("일일 PnL 초기화")
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_risk_manager.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: 리스크 매니저 구현 (일일 손실 한도, 포지션 수 제한)"
```

---

## Task 7: 실시간 데이터 스트림 (WebSocket)

**Files:**
- Create: `src/data_stream.py`
- Create: `tests/test_data_stream.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_data_stream.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.data_stream import KlineStream

@pytest.mark.asyncio
async def test_kline_stream_parses_message():
    stream = KlineStream(symbol="XRPUSDT", interval="1m")
    raw_msg = {
        "k": {
            "t": 1700000000000,
            "o": "0.5000", "h": "0.5100",
            "l": "0.4900", "c": "0.5050",
            "v": "100000", "x": True,
        }
    }
    candle = stream.parse_kline(raw_msg)
    assert candle["close"] == 0.5050
    assert candle["is_closed"] is True

@pytest.mark.asyncio
async def test_callback_called_on_closed_candle():
    received = []
    stream = KlineStream(symbol="XRPUSDT", interval="1m",
                         on_candle=lambda c: received.append(c))
    raw_msg = {
        "k": {
            "t": 1700000000000,
            "o": "0.5", "h": "0.51",
            "l": "0.49", "c": "0.505",
            "v": "100000", "x": True,
        }
    }
    stream.handle_message(raw_msg)
    assert len(received) == 1
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_data_stream.py -v
```
Expected: FAIL

**Step 3: KlineStream 구현**

```python
# src/data_stream.py
import asyncio
import json
from collections import deque
from typing import Callable
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from loguru import logger


class KlineStream:
    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        buffer_size: int = 200,
        on_candle: Callable = None,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self.buffer: deque = deque(maxlen=buffer_size)
        self.on_candle = on_candle

    def parse_kline(self, msg: dict) -> dict:
        k = msg["k"]
        return {
            "timestamp": k["t"],
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
            "is_closed": k["x"],
        }

    def handle_message(self, msg: dict):
        candle = self.parse_kline(msg)
        if candle["is_closed"]:
            self.buffer.append(candle)
            if self.on_candle:
                self.on_candle(candle)

    def get_dataframe(self) -> pd.DataFrame | None:
        if len(self.buffer) < 50:
            return None
        df = pd.DataFrame(list(self.buffer))
        df.set_index("timestamp", inplace=True)
        return df

    async def start(self, api_key: str, api_secret: str, testnet: bool = True):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        bm = BinanceSocketManager(client)
        stream_name = f"{self.symbol}@kline_{self.interval}"
        logger.info(f"WebSocket 스트림 시작: {stream_name}")
        async with bm.futures_kline_socket(
            symbol=self.symbol.upper(), interval=self.interval
        ) as stream:
            while True:
                msg = await stream.recv()
                self.handle_message(msg)
        await client.close_connection()
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_data_stream.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/data_stream.py tests/test_data_stream.py
git commit -m "feat: WebSocket 실시간 캔들 스트림 구현"
```

---

## Task 8: 트레이딩 봇 메인 루프

**Files:**
- Create: `src/bot.py`
- Create: `tests/test_bot.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_bot.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np
import os
from src.bot import TradingBot
from src.config import Config

@pytest.fixture
def config():
    os.environ.update({
        "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
        "BINANCE_TESTNET": "true", "SYMBOL": "XRPUSDT",
        "LEVERAGE": "10", "RISK_PER_TRADE": "0.02",
    })
    return Config()

@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 0.5
    return pd.DataFrame({
        "open": close, "high": close * 1.005,
        "low": close * 0.995, "close": close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    })

@pytest.mark.asyncio
async def test_bot_processes_signal(config, sample_df):
    bot = TradingBot(config)
    bot.exchange = AsyncMock()
    bot.exchange.get_balance = AsyncMock(return_value=1000.0)
    bot.exchange.get_position = AsyncMock(return_value=None)
    bot.exchange.place_order = AsyncMock(return_value={"orderId": "123"})
    bot.exchange.set_leverage = AsyncMock(return_value={})
    bot.db = MagicMock()
    bot.db.save_trade = MagicMock(return_value={"id": "trade1"})

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = "LONG"
        mock_ind.get_atr_stop.return_value = (0.48, 0.56)
        MockInd.return_value = mock_ind
        await bot.process_candle(sample_df)
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_bot.py -v
```
Expected: FAIL

**Step 3: TradingBot 구현**

```python
# src/bot.py
import asyncio
import os
from loguru import logger
from src.config import Config
from src.exchange import BinanceFuturesClient
from src.indicators import Indicators
from src.data_stream import KlineStream
from src.database import TradeRepository
from src.risk_manager import RiskManager


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.db = TradeRepository(
            url=os.getenv("SUPABASE_URL", ""),
            key=os.getenv("SUPABASE_KEY", ""),
        )
        self.risk = RiskManager(config)
        self.current_trade_id: str | None = None
        self.stream = KlineStream(
            symbol=config.symbol,
            interval="1m",
            on_candle=self._on_candle_closed,
        )

    def _on_candle_closed(self, candle: dict):
        df = self.stream.get_dataframe()
        if df is not None:
            asyncio.create_task(self.process_candle(df))

    async def process_candle(self, df):
        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        signal = ind.get_signal(df_with_indicators)
        logger.info(f"신호: {signal}")

        position = await self.exchange.get_position()

        # 포지션 없을 때 신규 진입
        if position is None and signal != "HOLD":
            if not self.risk.can_open_new_position():
                logger.info("최대 포지션 수 도달")
                return
            await self._open_position(signal, df_with_indicators)

        # 포지션 있을 때 반대 신호 시 청산
        elif position is not None:
            pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
            if (pos_side == "LONG" and signal == "SHORT") or \
               (pos_side == "SHORT" and signal == "LONG"):
                await self._close_position(position)

    async def _open_position(self, signal: str, df):
        balance = await self.exchange.get_balance()
        price = df["close"].iloc[-1]
        quantity = self.exchange.calculate_quantity(
            balance=balance, price=price, leverage=self.config.leverage
        )
        stop_loss, take_profit = Indicators(df).get_atr_stop(df, signal, price)

        side = "BUY" if signal == "LONG" else "SELL"
        await self.exchange.set_leverage(self.config.leverage)
        order = await self.exchange.place_order(side=side, quantity=quantity)

        last_row = df.iloc[-1]
        signal_snapshot = {
            "rsi":       float(last_row.get("rsi", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "atr":       float(last_row.get("atr", 0)),
        }
        trade = self.db.save_trade(
            symbol=self.config.symbol,
            side=signal,
            entry_price=price,
            quantity=quantity,
            leverage=self.config.leverage,
            signal_data=signal_snapshot,
        )
        self.current_trade_id = trade["id"]
        logger.success(f"{signal} 진입: 가격={price}, 수량={quantity}, SL={stop_loss:.4f}, TP={take_profit:.4f}")

        # 손절/익절 주문
        sl_side = "SELL" if signal == "LONG" else "BUY"
        await self.exchange.place_order(
            side=sl_side, quantity=quantity,
            order_type="STOP_MARKET", stop_price=round(stop_loss, 4),
            reduce_only=True,
        )
        await self.exchange.place_order(
            side=sl_side, quantity=quantity,
            order_type="TAKE_PROFIT_MARKET", stop_price=round(take_profit, 4),
            reduce_only=True,
        )

    async def _close_position(self, position: dict):
        amt = abs(float(position["positionAmt"]))
        side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)

        entry = float(position["entryPrice"])
        mark  = float(position["markPrice"])
        pnl   = (mark - entry) * amt if side == "SELL" else (entry - mark) * amt

        if self.current_trade_id:
            self.db.close_trade(self.current_trade_id, exit_price=mark, pnl=pnl)
        self.risk.record_pnl(pnl)
        self.current_trade_id = None
        logger.success(f"포지션 청산: PnL={pnl:.4f} USDT")

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=self.config.testnet,
        )
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_bot.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: TradingBot 메인 루프 구현 (진입/청산/손절/익절)"
```

---

## Task 9: 엔트리포인트 및 로깅 설정

**Files:**
- Create: `main.py`
- Create: `src/logger_setup.py`

**Step 1: 로거 설정**

```python
# src/logger_setup.py
import sys
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
    )
```

**Step 2: main.py 작성**

```python
# main.py
import asyncio
import os
from dotenv import load_dotenv
from src.config import Config
from src.bot import TradingBot
from src.logger_setup import setup_logger

load_dotenv()

async def main():
    setup_logger(log_level="INFO")
    config = Config()
    bot = TradingBot(config)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: logs 디렉토리 생성**

```bash
mkdir -p logs
echo "logs/*.log" >> .gitignore
```

**Step 4: 전체 테스트 실행**

```bash
pytest tests/ -v --tb=short
```
Expected: 모든 테스트 PASS

**Step 5: Commit**

```bash
git add main.py src/logger_setup.py logs/.gitkeep
git commit -m "feat: 엔트리포인트 및 로깅 설정 완료"
```

---

## Task 10: 테스트넷 통합 테스트 및 검증

**Step 1: .env 설정 확인**

```bash
# .env 파일에서 BINANCE_TESTNET=true 확인
grep BINANCE_TESTNET .env
```

**Step 2: 바이낸스 테스트넷 API 키 발급**

- https://testnet.binancefuture.com 접속
- 계정 생성 후 API 키 발급
- `.env` 파일에 키 입력

**Step 3: 테스트넷 연결 확인**

```bash
python -c "
from src.config import Config
from src.exchange import BinanceFuturesClient
import asyncio

async def check():
    cfg = Config()
    client = BinanceFuturesClient(cfg)
    bal = await client.get_balance()
    print(f'잔고: {bal} USDT')

asyncio.run(check())
"
```
Expected: 잔고 출력 (테스트넷 기본 10,000 USDT)

**Step 4: 레버리지 설정 확인**

```bash
python -c "
from src.config import Config
from src.exchange import BinanceFuturesClient
import asyncio

async def check():
    cfg = Config()
    client = BinanceFuturesClient(cfg)
    result = await client.set_leverage(cfg.leverage)
    print(f'레버리지 설정: {result}')

asyncio.run(check())
"
```

**Step 5: 봇 5분 시범 실행**

```bash
timeout 300 python main.py
```
Expected: 로그에 신호 생성 및 포지션 관리 메시지 출력

**Step 6: 최종 Commit**

```bash
git add .
git commit -m "feat: XRP 선물 자동매매 봇 완성 - 테스트넷 검증 완료"
```

---

## 전체 테스트 실행 명령

```bash
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

## 실제 운영 전 체크리스트

- [ ] 테스트넷에서 최소 48시간 시범 운영
- [ ] `.env`에서 `BINANCE_TESTNET=false` 변경
- [ ] 실제 API 키로 교체
- [ ] `RISK_PER_TRADE=0.01` (초기 운영 시 1%로 낮춤)
- [ ] `LEVERAGE=5` (초기 운영 시 5배로 낮춤)
- [ ] Supabase 거래 이력 정상 저장 확인

## 주의사항

> **경고:** 선물 레버리지 거래는 원금 전액 손실 위험이 있습니다.
> 반드시 테스트넷에서 충분히 검증 후 소액으로 시작하세요.
