# Discord 알림 전환 및 포지션 복구 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Notion 연동을 제거하고 Discord 웹훅으로 거래 알림을 전송하며, 봇 재시작 시 기존 포지션을 감지하여 정상 작동하도록 한다.

**Architecture:**
- `TradeRepository` (Notion 기반)를 `DiscordNotifier` (Discord 웹훅 기반)로 교체한다.
- 거래 상태(현재 포지션 ID 등)는 메모리 대신 로컬 JSON 파일(`state.json`)에 저장하여 재시작 후에도 복구 가능하게 한다.
- 봇 시작 시 바이낸스 API로 실제 포지션을 조회하여 `current_trade_id`를 복구한다.

**Tech Stack:** Python 3.13, httpx (Discord 웹훅 HTTP 요청), python-binance, loguru

---

## Task 1: Discord 웹훅 알림 모듈 생성

**Files:**
- Create: `src/notifier.py`
- Modify: `src/config.py`
- Modify: `.env` 및 `.env.example`

### Step 1: `.env`와 `.env.example`에 Discord 웹훅 URL 추가

`.env`에 아래 줄 추가:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

`.env.example`에 아래 줄 추가:
```
DISCORD_WEBHOOK_URL=
```

### Step 2: `src/config.py`에 `discord_webhook_url` 필드 추가

`notion_token`, `notion_database_id` 필드를 제거하고 `discord_webhook_url` 추가:

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
    symbol: str = "XRPUSDT"
    leverage: int = 10
    risk_per_trade: float = 0.02
    max_positions: int = 3
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.045
    trailing_stop_pct: float = 0.01
    discord_webhook_url: str = ""

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
```

### Step 3: `src/notifier.py` 생성

```python
# src/notifier.py
import httpx
from loguru import logger


class DiscordNotifier:
    """Discord 웹훅으로 거래 알림을 전송하는 노티파이어."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self._enabled = bool(webhook_url)

    def _send(self, content: str) -> None:
        if not self._enabled:
            logger.debug("Discord 웹훅 URL 미설정 - 알림 건너뜀")
            return
        try:
            resp = httpx.post(
                self.webhook_url,
                json={"content": content},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Discord 알림 전송 실패: {e}")

    def notify_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
        signal_data: dict = None,
    ) -> None:
        rsi = (signal_data or {}).get("rsi", 0)
        macd = (signal_data or {}).get("macd_hist", 0)
        atr = (signal_data or {}).get("atr", 0)
        msg = (
            f"**[{symbol}] {side} 진입**\n"
            f"진입가: `{entry_price:.4f}` | 수량: `{quantity}` | 레버리지: `{leverage}x`\n"
            f"SL: `{stop_loss:.4f}` | TP: `{take_profit:.4f}`\n"
            f"RSI: `{rsi:.2f}` | MACD Hist: `{macd:.6f}` | ATR: `{atr:.6f}`"
        )
        self._send(msg)

    def notify_close(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} **[{symbol}] {side} 청산**\n"
            f"청산가: `{exit_price:.4f}` | PnL: `{pnl:+.4f} USDT`"
        )
        self._send(msg)

    def notify_info(self, message: str) -> None:
        self._send(f"ℹ️ {message}")
```

### Step 4: 웹훅 URL 실제 값을 `.env`에 입력

Discord 서버 → 채널 설정 → 연동 → 웹훅 생성 후 URL 복사하여 `.env`의 `DISCORD_WEBHOOK_URL=` 뒤에 붙여넣기.

---

## Task 2: `src/database.py` 제거 및 `src/bot.py` 교체

**Files:**
- Delete: `src/database.py`
- Modify: `src/bot.py`

### Step 1: `src/bot.py`에서 Notion 관련 코드를 `DiscordNotifier`로 교체

`src/bot.py` 전체를 아래로 교체:

```python
# src/bot.py
import asyncio
import os
from loguru import logger
from src.config import Config
from src.exchange import BinanceFuturesClient
from src.indicators import Indicators
from src.data_stream import KlineStream
from src.notifier import DiscordNotifier
from src.risk_manager import RiskManager


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = RiskManager(config)
        self.current_trade_side: str | None = None  # "LONG" | "SHORT"
        self.stream = KlineStream(
            symbol=config.symbol,
            interval="1m",
            on_candle=self._on_candle_closed,
        )

    def _on_candle_closed(self, candle: dict):
        df = self.stream.get_dataframe()
        if df is not None:
            asyncio.create_task(self.process_candle(df))

    async def _recover_position(self) -> None:
        """재시작 시 바이낸스에서 현재 포지션을 조회하여 상태 복구."""
        position = await self.exchange.get_position()
        if position is not None:
            amt = float(position["positionAmt"])
            self.current_trade_side = "LONG" if amt > 0 else "SHORT"
            entry = float(position["entryPrice"])
            logger.info(
                f"기존 포지션 복구: {self.current_trade_side} | "
                f"진입가={entry:.4f} | 수량={abs(amt)}"
            )
            self.notifier.notify_info(
                f"봇 재시작 - 기존 포지션 감지: {self.current_trade_side} "
                f"진입가={entry:.4f} 수량={abs(amt)}"
            )
        else:
            logger.info("기존 포지션 없음 - 신규 진입 대기")

    async def process_candle(self, df):
        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        signal = ind.get_signal(df_with_indicators)
        current_price = df_with_indicators["close"].iloc[-1]
        logger.info(f"신호: {signal} | 현재가: {current_price:.4f} USDT")

        position = await self.exchange.get_position()

        if position is None and signal != "HOLD":
            self.current_trade_side = None
            if not self.risk.can_open_new_position():
                logger.info("최대 포지션 수 도달")
                return
            await self._open_position(signal, df_with_indicators)

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

        notional = quantity * price
        if quantity <= 0 or notional < self.exchange.MIN_NOTIONAL:
            logger.warning(
                f"주문 건너뜀: 명목금액 {notional:.2f} USDT < 최소 {self.exchange.MIN_NOTIONAL} USDT "
                f"(잔고={balance:.2f}, 수량={quantity})"
            )
            return

        side = "BUY" if signal == "LONG" else "SELL"
        await self.exchange.set_leverage(self.config.leverage)
        await self.exchange.place_order(side=side, quantity=quantity)

        last_row = df.iloc[-1]
        signal_snapshot = {
            "rsi":       float(last_row.get("rsi", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "atr":       float(last_row.get("atr", 0)),
        }

        self.current_trade_side = signal
        self.notifier.notify_open(
            symbol=self.config.symbol,
            side=signal,
            entry_price=price,
            quantity=quantity,
            leverage=self.config.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_data=signal_snapshot,
        )
        logger.success(
            f"{signal} 진입: 가격={price}, 수량={quantity}, "
            f"SL={stop_loss:.4f}, TP={take_profit:.4f}"
        )

        sl_side = "SELL" if signal == "LONG" else "BUY"
        await self.exchange.place_order(
            side=sl_side,
            quantity=quantity,
            order_type="STOP_MARKET",
            stop_price=round(stop_loss, 4),
            reduce_only=True,
        )
        await self.exchange.place_order(
            side=sl_side,
            quantity=quantity,
            order_type="TAKE_PROFIT_MARKET",
            stop_price=round(take_profit, 4),
            reduce_only=True,
        )

    async def _close_position(self, position: dict):
        amt = abs(float(position["positionAmt"]))
        side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
        pos_side = "LONG" if side == "SELL" else "SHORT"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)

        entry = float(position["entryPrice"])
        mark  = float(position["markPrice"])
        pnl   = (mark - entry) * amt if side == "SELL" else (entry - mark) * amt

        self.notifier.notify_close(
            symbol=self.config.symbol,
            side=pos_side,
            exit_price=mark,
            pnl=pnl,
        )
        self.risk.record_pnl(pnl)
        self.current_trade_side = None
        logger.success(f"포지션 청산: PnL={pnl:.4f} USDT")

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
```

### Step 2: `src/database.py` 삭제

```bash
rm src/database.py
```

---

## Task 3: 의존성 정리

**Files:**
- Modify: `requirements.txt` (또는 `pyproject.toml`)

### Step 1: 현재 의존성 파일 확인

```bash
cat requirements.txt
# 또는
cat pyproject.toml
```

### Step 2: `notion-client` 제거, `httpx` 추가

`requirements.txt`에서 `notion-client` 줄 삭제 후 `httpx` 추가:

```
httpx>=0.27.0
```

### Step 3: 의존성 재설치

```bash
pip install httpx
pip uninstall notion-client -y
```

또는 venv 사용 시:

```bash
.venv/bin/pip install httpx
.venv/bin/pip uninstall notion-client -y
```

---

## Task 4: 동작 검증

**Files:**
- 수정 없음 (실행 테스트)

### Step 1: 봇 시작 전 환경변수 확인

```bash
grep DISCORD_WEBHOOK_URL .env
```
예상 출력: `DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...` (빈 값이면 알림 비활성화)

### Step 2: 봇 실행

```bash
python main.py
# 또는
.venv/bin/python main.py
```

### Step 3: 재시작 시 포지션 복구 확인

봇 실행 중 바이낸스에 포지션이 있는 경우 로그에서 아래 메시지 확인:
```
기존 포지션 복구: LONG | 진입가=X.XXXX | 수량=X.X
```

포지션이 없는 경우:
```
기존 포지션 없음 - 신규 진입 대기
```

### Step 4: Discord 알림 테스트 (선택)

Discord 채널에서 봇 재시작 알림 메시지 확인:
- 포지션 있을 때: `ℹ️ 봇 재시작 - 기존 포지션 감지: LONG 진입가=X.XXXX 수량=X.X`
- 진입 시: `[XRPUSDT] LONG 진입` 메시지
- 청산 시: `✅ [XRPUSDT] LONG 청산` 메시지

### Step 5: Notion 관련 import 잔재 없는지 확인

```bash
grep -r "notion" src/ --include="*.py"
```
예상 출력: (아무것도 없음)

---

## 주요 변경 요약

| 항목 | 이전 | 이후 |
|------|------|------|
| 알림 수단 | Notion API | Discord 웹훅 |
| 거래 ID 추적 | Notion 페이지 ID | 불필요 (바이낸스 포지션 직접 조회) |
| 재시작 복구 | 없음 | `_recover_position()` 으로 바이낸스 조회 |
| 환경변수 | `NOTION_TOKEN`, `NOTION_DATABASE_ID` | `DISCORD_WEBHOOK_URL` |
| 외부 의존성 | `notion-client` | `httpx` |
