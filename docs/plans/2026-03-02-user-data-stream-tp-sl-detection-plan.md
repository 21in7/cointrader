# User Data Stream TP/SL 감지 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Binance Futures User Data Stream을 도입하여 TP/SL 작동을 실시간 감지하고, 순수익(Net PnL)을 기록하며, Discord에 예상 수익 vs 실제 순수익 비교 알림을 전송한다.

**Architecture:** `python-binance`의 `futures_user_socket(listenKey)`로 User Data Stream에 연결하고, 30분 keepalive 백그라운드 태스크와 `while True: try-except` 무한 재연결 루프로 안정성을 확보한다. `ORDER_TRADE_UPDATE` 이벤트에서 청산 주문을 감지하면 `bot._on_position_closed()` 콜백을 호출하여 PnL 기록과 Discord 알림을 일원화한다.

**Tech Stack:** Python 3.12, python-binance (AsyncClient, BinanceSocketManager), asyncio, loguru

**Design Doc:** `docs/plans/2026-03-02-user-data-stream-tp-sl-detection-design.md`

---

## Task 1: `exchange.py`에 listenKey 관리 메서드 추가

**Files:**
- Modify: `src/exchange.py` (끝에 메서드 추가)

**Step 1: listenKey 3개 메서드 구현**

`src/exchange.py` 끝에 아래 메서드 3개를 추가한다.

```python
async def create_listen_key(self) -> str:
    """POST /fapi/v1/listenKey — listenKey 신규 발급"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: self.client.futures_stream_get_listen_key(),
    )
    return result

async def keepalive_listen_key(self, listen_key: str) -> None:
    """PUT /fapi/v1/listenKey — listenKey 만료 연장 (60분 → 리셋)"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: self.client.futures_stream_keepalive(listenKey=listen_key),
    )

async def delete_listen_key(self, listen_key: str) -> None:
    """DELETE /fapi/v1/listenKey — listenKey 삭제 (정상 종료 시)"""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: self.client.futures_stream_close(listenKey=listen_key),
        )
    except Exception as e:
        logger.warning(f"listenKey 삭제 실패 (무시): {e}")
```

**Step 2: 커밋**

```bash
git add src/exchange.py
git commit -m "feat: add listenKey create/keepalive/delete methods to exchange"
```

---

## Task 2: `notifier.py`의 `notify_close()` 시그니처 확장

**Files:**
- Modify: `src/notifier.py`

**Step 1: `notify_close()` 메서드 교체**

기존 `notify_close()`를 아래로 교체한다. `close_reason`, `estimated_pnl`, `net_pnl`, `diff` 파라미터가 추가된다.

```python
def notify_close(
    self,
    symbol: str,
    side: str,
    close_reason: str,      # "TP" | "SL" | "MANUAL"
    exit_price: float,
    estimated_pnl: float,   # 봇 계산 (entry-exit 기반)
    net_pnl: float,         # 바이낸스 rp - |commission|
    diff: float,            # net_pnl - estimated_pnl (슬리피지+수수료)
) -> None:
    emoji_map = {"TP": "✅", "SL": "❌", "MANUAL": "🔶"}
    emoji = emoji_map.get(close_reason, "🔶")
    msg = (
        f"{emoji} **[{symbol}] {side} {close_reason} 청산**\n"
        f"청산가:               `{exit_price:.4f}`\n"
        f"예상 수익:            `{estimated_pnl:+.4f} USDT`\n"
        f"실제 순수익:          `{net_pnl:+.4f} USDT`\n"
        f"차이(슬리피지+수수료): `{diff:+.4f} USDT`"
    )
    self._send(msg)
```

**Step 2: 커밋**

```bash
git add src/notifier.py
git commit -m "feat: extend notify_close with close_reason, net_pnl, diff fields"
```

---

## Task 3: `src/user_data_stream.py` 신규 생성

**Files:**
- Create: `src/user_data_stream.py`

**Step 1: 파일 전체 작성**

```python
import asyncio
from typing import Callable
from binance import AsyncClient, BinanceSocketManager
from loguru import logger

_KEEPALIVE_INTERVAL = 30 * 60   # 30분 (listenKey 만료 60분의 절반)
_RECONNECT_DELAY    = 5         # 재연결 대기 초

_CLOSE_ORDER_TYPES = {"TAKE_PROFIT_MARKET", "STOP_MARKET"}


class UserDataStream:
    """
    Binance Futures User Data Stream을 구독하여 주문 체결 이벤트를 처리한다.

    - listenKey 30분 keepalive 백그라운드 태스크
    - 네트워크 단절 시 무한 재연결 루프
    - ORDER_TRADE_UPDATE 이벤트에서 청산 주문만 필터링하여 콜백 호출
    """

    def __init__(
        self,
        exchange,                        # BinanceFuturesClient 인스턴스
        on_order_filled: Callable,       # bot._on_position_closed 콜백
    ):
        self._exchange = exchange
        self._on_order_filled = on_order_filled
        self._listen_key: str | None = None
        self._keepalive_task: asyncio.Task | None = None

    async def start(self, api_key: str, api_secret: str) -> None:
        """User Data Stream 메인 루프 — 봇 종료 시까지 실행."""
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
        bm = BinanceSocketManager(client)
        try:
            await self._run_loop(bm)
        finally:
            await client.close_connection()

    async def _run_loop(self, bm: BinanceSocketManager) -> None:
        """listenKey 발급 → 연결 → 재연결 무한 루프."""
        while True:
            try:
                self._listen_key = await self._exchange.create_listen_key()
                logger.info(f"User Data Stream listenKey 발급: {self._listen_key[:8]}...")

                self._keepalive_task = asyncio.create_task(
                    self._keepalive_loop(self._listen_key)
                )

                async with bm.futures_user_socket(self._listen_key) as stream:
                    logger.info("User Data Stream 연결 완료")
                    async for msg in stream:
                        await self._handle_message(msg)

            except asyncio.CancelledError:
                logger.info("User Data Stream 정상 종료")
                if self._listen_key:
                    await self._exchange.delete_listen_key(self._listen_key)
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                break

            except Exception as e:
                logger.warning(
                    f"User Data Stream 끊김: {e} — "
                    f"{_RECONNECT_DELAY}초 후 재연결"
                )
                if self._keepalive_task:
                    self._keepalive_task.cancel()
                    self._keepalive_task = None
                await asyncio.sleep(_RECONNECT_DELAY)

    async def _keepalive_loop(self, listen_key: str) -> None:
        """30분마다 listenKey를 갱신한다."""
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            try:
                await self._exchange.keepalive_listen_key(listen_key)
                logger.debug("listenKey 갱신 완료")
            except Exception as e:
                logger.warning(f"listenKey 갱신 실패: {e} — 재연결 루프가 처리")
                break

    async def _handle_message(self, msg: dict) -> None:
        """ORDER_TRADE_UPDATE 이벤트에서 청산 주문을 필터링하여 콜백을 호출한다."""
        if msg.get("e") != "ORDER_TRADE_UPDATE":
            return

        order = msg.get("o", {})

        # x: Execution Type, X: Order Status
        if order.get("x") != "TRADE" or order.get("X") != "FILLED":
            return

        order_type   = order.get("o", "")
        is_reduce    = order.get("R", False)
        realized_pnl = float(order.get("rp", "0"))

        # 청산 주문 판별: reduceOnly이거나, TP/SL 타입이거나, rp != 0
        is_close = is_reduce or order_type in _CLOSE_ORDER_TYPES or realized_pnl != 0
        if not is_close:
            return

        commission = abs(float(order.get("n", "0")))
        net_pnl    = realized_pnl - commission
        exit_price = float(order.get("ap", "0"))

        if order_type == "TAKE_PROFIT_MARKET":
            close_reason = "TP"
        elif order_type == "STOP_MARKET":
            close_reason = "SL"
        else:
            close_reason = "MANUAL"

        logger.info(
            f"청산 감지({close_reason}): exit={exit_price:.4f}, "
            f"rp={realized_pnl:+.4f}, commission={commission:.4f}, "
            f"net_pnl={net_pnl:+.4f}"
        )

        await self._on_order_filled(
            net_pnl=net_pnl,
            close_reason=close_reason,
            exit_price=exit_price,
        )
```

**Step 2: 커밋**

```bash
git add src/user_data_stream.py
git commit -m "feat: add UserDataStream with keepalive and reconnect loop"
```

---

## Task 4: `bot.py` 수정 — 상태 변수 추가 및 `_open_position()` 저장

**Files:**
- Modify: `src/bot.py`

**Step 1: `__init__`에 상태 변수 추가**

`TradingBot.__init__()` 내부에서 `self.current_trade_side` 선언 바로 아래에 추가한다.

```python
self._entry_price: float | None = None
self._entry_quantity: float | None = None
```

**Step 2: `_open_position()` 내부에서 진입가/수량 저장**

`self.current_trade_side = signal` 바로 아래에 추가한다.

```python
self._entry_price = price
self._entry_quantity = quantity
```

**Step 3: 커밋**

```bash
git add src/bot.py
git commit -m "feat: store entry_price and entry_quantity on position open"
```

---

## Task 5: `bot.py` 수정 — `_on_position_closed()` 콜백 추가

**Files:**
- Modify: `src/bot.py`

**Step 1: `_calc_estimated_pnl()` 헬퍼 메서드 추가**

`_close_position()` 메서드 바로 위에 추가한다.

```python
def _calc_estimated_pnl(self, exit_price: float) -> float:
    """진입가·수량 기반 예상 PnL 계산 (수수료 미반영)."""
    if self._entry_price is None or self._entry_quantity is None:
        return 0.0
    if self.current_trade_side == "LONG":
        return (exit_price - self._entry_price) * self._entry_quantity
    return (self._entry_price - exit_price) * self._entry_quantity
```

**Step 2: `_on_position_closed()` 콜백 추가**

`_calc_estimated_pnl()` 바로 아래에 추가한다.

```python
async def _on_position_closed(
    self,
    net_pnl: float,
    close_reason: str,
    exit_price: float,
) -> None:
    """User Data Stream에서 청산 감지 시 호출되는 콜백."""
    estimated_pnl = self._calc_estimated_pnl(exit_price)
    diff = net_pnl - estimated_pnl

    self.risk.record_pnl(net_pnl)

    self.notifier.notify_close(
        symbol=self.config.symbol,
        side=self.current_trade_side or "UNKNOWN",
        close_reason=close_reason,
        exit_price=exit_price,
        estimated_pnl=estimated_pnl,
        net_pnl=net_pnl,
        diff=diff,
    )

    logger.success(
        f"포지션 청산({close_reason}): 예상={estimated_pnl:+.4f}, "
        f"순수익={net_pnl:+.4f}, 차이={diff:+.4f} USDT"
    )

    # Flat 상태로 초기화
    self.current_trade_side = None
    self._entry_price = None
    self._entry_quantity = None
```

**Step 3: 커밋**

```bash
git add src/bot.py
git commit -m "feat: add _on_position_closed callback with net PnL and discord alert"
```

---

## Task 6: `bot.py` 수정 — `_close_position()`에서 중복 후처리 제거

**Files:**
- Modify: `src/bot.py`

**배경:** 봇이 직접 청산(`_close_and_reenter`)하는 경우에도 User Data Stream의 `ORDER_TRADE_UPDATE`가 발생한다. 중복 방지를 위해 `_close_position()`에서 `notify_close()`와 `record_pnl()` 호출을 제거한다.

**Step 1: `_close_position()` 수정**

기존 코드:
```python
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
```

수정 후 (`notify_close`, `record_pnl`, `current_trade_side = None` 제거 — User Data Stream 콜백이 처리):
```python
async def _close_position(self, position: dict):
    """포지션 청산 주문만 실행한다. PnL 기록/알림은 _on_position_closed 콜백이 담당."""
    amt = abs(float(position["positionAmt"]))
    side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
    await self.exchange.cancel_all_orders()
    await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)
    logger.info(f"청산 주문 전송 완료 (side={side}, qty={amt})")
```

**Step 2: 커밋**

```bash
git add src/bot.py
git commit -m "refactor: remove duplicate pnl/notify from _close_position (handled by callback)"
```

---

## Task 7: `bot.py` 수정 — `run()`에서 UserDataStream 병렬 실행

**Files:**
- Modify: `src/bot.py`

**Step 1: import 추가**

파일 상단 import 블록에 추가한다.

```python
from src.user_data_stream import UserDataStream
```

**Step 2: `run()` 메서드 수정**

기존:
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

수정 후:
```python
async def run(self):
    logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
    await self._recover_position()
    balance = await self.exchange.get_balance()
    self.risk.set_base_balance(balance)
    logger.info(f"기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")

    user_stream = UserDataStream(
        exchange=self.exchange,
        on_order_filled=self._on_position_closed,
    )

    await asyncio.gather(
        self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        ),
        user_stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        ),
    )
```

**Step 3: 커밋**

```bash
git add src/bot.py
git commit -m "feat: run UserDataStream in parallel with candle stream"
```

---

## Task 8: README.md 업데이트

**Files:**
- Modify: `README.md`

**Step 1: 기능 목록에 User Data Stream 항목 추가**

README의 주요 기능 섹션에 아래 내용을 추가한다.

- **실시간 TP/SL 감지**: Binance User Data Stream으로 TP/SL 작동을 즉시 감지 (캔들 마감 대기 없음)
- **순수익(Net PnL) 기록**: 바이낸스 `realizedProfit - commission`으로 정확한 순수익 계산
- **Discord 상세 청산 알림**: 예상 수익 vs 실제 순수익 + 슬리피지/수수료 차이 표시
- **listenKey 자동 갱신**: 30분 keepalive + 네트워크 단절 시 자동 재연결

**Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: update README with User Data Stream TP/SL detection feature"
```

---

## 최종 검증

봇 실행 후 로그에서 아래 메시지가 순서대로 나타나면 정상 동작:

```
INFO  | User Data Stream listenKey 발급: xxxxxxxx...
INFO  | User Data Stream 연결 완료
DEBUG | listenKey 갱신 완료  ← 30분 후
INFO  | 청산 감지(TP): exit=1.3393, rp=+0.4821, commission=0.0209, net_pnl=+0.4612
SUCCESS | 포지션 청산(TP): 예상=+0.4821, 순수익=+0.4612, 차이=-0.0209 USDT
```

Discord에는 아래 형식의 알림이 전송됨:

```
✅ [XRPUSDT] SHORT TP 청산
청산가:               1.3393
예상 수익:            +0.4821 USDT
실제 순수익:          +0.4612 USDT
차이(슬리피지+수수료): -0.0209 USDT
```
