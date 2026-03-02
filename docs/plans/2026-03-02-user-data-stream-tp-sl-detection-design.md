# User Data Stream TP/SL 감지 설계

**날짜:** 2026-03-02  
**목적:** Binance Futures User Data Stream을 도입하여 TP/SL 작동을 실시간 감지하고, 순수익(Net PnL)을 기록하며, Discord에 상세 청산 알림을 전송한다.

---

## 배경 및 문제

기존 봇은 매 캔들 마감마다 `get_position()`을 폴링하여 포지션 소멸 여부를 확인하는 방식이었다. 이 구조의 한계:

1. **TP/SL 작동 후 최대 15분 지연** — 캔들 마감 전까지 감지 불가
2. **청산 원인 구분 불가** — TP인지 SL인지 수동 청산인지 알 수 없음
3. **PnL 기록 누락** — `_close_position()`을 봇이 직접 호출하지 않으면 `record_pnl()` 미실행
4. **Discord 알림 누락** — 동일 이유로 `notify_close()` 미호출

---

## 선택한 접근 방식

**방식 A: `python-binance` 내장 User Data Stream + 30분 수동 keepalive 보강**

- 기존 `BinanceSocketManager` 활용으로 추가 의존성 없음
- `futures_user_socket(listenKey)`로 User Data Stream 연결
- 별도 30분 keepalive 백그라운드 태스크로 안정성 보강
- `while True: try-except` 무한 재연결 루프로 네트워크 단절 복구

---

## 전체 아키텍처

### 파일 변경 목록

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `src/user_data_stream.py` | **신규** | User Data Stream 전담 클래스 |
| `src/bot.py` | 수정 | `UserDataStream` 초기화, `run()` 병렬 실행, `_on_position_closed()` 콜백, `_entry_price`/`_entry_quantity` 상태 추가 |
| `src/exchange.py` | 수정 | `create_listen_key()`, `keepalive_listen_key()`, `delete_listen_key()` 메서드 추가 |
| `src/notifier.py` | 수정 | `notify_close()`에 `close_reason`, `estimated_pnl`, `net_pnl` 파라미터 추가 |
| `src/risk_manager.py` | 수정 | `record_pnl()`이 net_pnl을 받도록 유지 (인터페이스 변경 없음) |

### 실행 흐름

```
bot.run()
  └── AsyncClient 단일 인스턴스 생성
        └── asyncio.gather()
              ├── MultiSymbolStream.start(client)   ← 기존 캔들 스트림
              └── UserDataStream.start()             ← 신규
                    ├── [백그라운드] _keepalive_loop()  30분마다 PUT /listenKey
                    └── [메인루프]  while True:
                                    try:
                                      listenKey 발급
                                      futures_user_socket() 연결
                                      async for msg: _handle_message()
                                    except CancelledError: break
                                    except Exception: sleep(5) → 재연결
```

---

## 섹션 1: UserDataStream 클래스 (`src/user_data_stream.py`)

### 상수

```python
KEEPALIVE_INTERVAL = 30 * 60   # 30분 (listenKey 만료 60분의 절반)
RECONNECT_DELAY    = 5         # 재연결 대기 초
```

### listenKey 생명주기

| 단계 | API | 시점 |
|------|-----|------|
| 발급 | `POST /fapi/v1/listenKey` | 연결 시작 / 재연결 시 |
| 갱신 | `PUT /fapi/v1/listenKey` | 30분마다 (백그라운드 태스크) |
| 삭제 | `DELETE /fapi/v1/listenKey` | 봇 정상 종료 시 (`CancelledError`) |

### 재연결 로직

```python
while True:
    try:
        listen_key = await exchange.create_listen_key()
        keepalive_task = asyncio.create_task(_keepalive_loop(listen_key))
        async with bm.futures_user_socket(listen_key):
            async for msg:
                await _handle_message(msg)
    except asyncio.CancelledError:
        await exchange.delete_listen_key(listen_key)
        keepalive_task.cancel()
        break
    except Exception as e:
        logger.warning(f"User Data Stream 끊김: {e}, {RECONNECT_DELAY}초 후 재연결")
        keepalive_task.cancel()
        await asyncio.sleep(RECONNECT_DELAY)
        # while True 상단으로 돌아가 listenKey 재발급
```

### keepalive 백그라운드 태스크

```python
async def _keepalive_loop(listen_key: str):
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL)
        try:
            await exchange.keepalive_listen_key(listen_key)
            logger.debug("listenKey 갱신 완료")
        except Exception:
            logger.warning("listenKey 갱신 실패 → 재연결 루프가 처리")
            break  # 재연결 루프가 새 태스크 생성
```

---

## 섹션 2: 이벤트 파싱 로직

### 페이로드 구조 (Binance Futures ORDER_TRADE_UPDATE)

주문 상세 정보는 최상위가 아닌 **내부 `"o"` 딕셔너리에 중첩**되어 있다.

```json
{
  "e": "ORDER_TRADE_UPDATE",
  "o": {
    "x": "TRADE",          // Execution Type
    "X": "FILLED",         // Order Status
    "o": "TAKE_PROFIT_MARKET",  // Order Type
    "R": true,             // reduceOnly
    "rp": "0.48210000",    // realizedProfit
    "n": "0.02100000",     // commission
    "ap": "1.3393"         // average price (체결가)
  }
}
```

### 판단 트리

```
msg["e"] == "ORDER_TRADE_UPDATE"?
  └── order = msg["o"]
      order["x"] == "TRADE" AND order["X"] == "FILLED"?
        └── 청산 주문인가?
            (order["R"] == true OR float(order["rp"]) != 0
             OR order["o"] in {"TAKE_PROFIT_MARKET", "STOP_MARKET"})
              ├── NO  → 무시 (진입 주문)
              └── YES → close_reason 판별:
                          "TAKE_PROFIT_MARKET" → "TP"
                          "STOP_MARKET"        → "SL"
                          그 외               → "MANUAL"
                        net_pnl = float(rp) - abs(float(n))
                        exit_price = float(order["ap"])
                        await on_order_filled(net_pnl, close_reason, exit_price)
```

---

## 섹션 3: `_on_position_closed()` 콜백 (`src/bot.py`)

### 진입가 상태 저장

`_open_position()` 내부에서 진입가와 수량을 인스턴스 변수로 저장한다. 청산 시점에는 포지션이 이미 사라져 있으므로 사전 저장이 필수다.

```python
# __init__에 추가
self._entry_price: float | None = None
self._entry_quantity: float | None = None

# _open_position() 내부에서 저장
self._entry_price = price
self._entry_quantity = quantity
```

### 예상 PnL 계산

```python
def _calc_estimated_pnl(self, exit_price: float) -> float:
    if self._entry_price is None or self._entry_quantity is None:
        return 0.0
    if self.current_trade_side == "LONG":
        return (exit_price - self._entry_price) * self._entry_quantity
    else:  # SHORT
        return (self._entry_price - exit_price) * self._entry_quantity
```

### 콜백 전체 흐름

```python
async def _on_position_closed(
    self,
    net_pnl: float,
    close_reason: str,   # "TP" | "SL" | "MANUAL"
    exit_price: float,
):
    estimated_pnl = self._calc_estimated_pnl(exit_price)
    diff = net_pnl - estimated_pnl  # 슬리피지 + 수수료 차이

    # RiskManager에 순수익 기록
    self.risk.record_pnl(net_pnl)

    # Discord 알림
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

    # 봇 상태 초기화 (Flat 상태로 복귀)
    self.current_trade_side = None
    self._entry_price = None
    self._entry_quantity = None
```

### 기존 `_close_position()` 변경

봇이 직접 청산하는 경우(`_close_and_reenter`)에도 User Data Stream의 `ORDER_TRADE_UPDATE`가 발생한다. **중복 처리 방지**를 위해 `_close_position()`에서 `notify_close()`와 `record_pnl()` 호출을 제거한다. 모든 청산 후처리는 `_on_position_closed()` 콜백 하나로 일원화한다.

---

## 섹션 4: Discord 알림 포맷 (`src/notifier.py`)

### `notify_close()` 시그니처 변경

```python
def notify_close(
    self,
    symbol: str,
    side: str,
    close_reason: str,    # "TP" | "SL" | "MANUAL"
    exit_price: float,
    estimated_pnl: float,
    net_pnl: float,
    diff: float,          # net_pnl - estimated_pnl
) -> None:
```

### 알림 포맷

```
✅ [XRPUSDT] SHORT TP 청산
청산가:          `1.3393`
예상 수익:       `+0.4821 USDT`
실제 순수익:     `+0.4612 USDT`
차이(슬리피지+수수료): `-0.0209 USDT`
```

| 청산 원인 | 이모지 |
|----------|--------|
| TP       | ✅     |
| SL       | ❌     |
| MANUAL   | 🔶     |

---

## 섹션 5: `src/exchange.py` 추가 메서드

```python
async def create_listen_key(self) -> str:
    """POST /fapi/v1/listenKey — listenKey 신규 발급"""

async def keepalive_listen_key(self, listen_key: str) -> None:
    """PUT /fapi/v1/listenKey — listenKey 만료 연장"""

async def delete_listen_key(self, listen_key: str) -> None:
    """DELETE /fapi/v1/listenKey — listenKey 삭제 (정상 종료 시)"""
```

---

## 데이터 흐름 요약

```
Binance WebSocket
  → ORDER_TRADE_UPDATE (FILLED, reduceOnly)
  → UserDataStream._handle_message()
  → net_pnl = rp - |commission|
  → bot._on_position_closed(net_pnl, close_reason, exit_price)
      ├── estimated_pnl = (exit - entry) × qty  (봇 계산)
      ├── diff = net_pnl - estimated_pnl
      ├── risk.record_pnl(net_pnl)              → 일일 PnL 누적
      ├── notifier.notify_close(...)             → Discord 알림
      └── 상태 초기화 (current_trade_side, _entry_price, _entry_quantity = None)
```

---

## 제외 범위 (YAGNI)

- DB 영구 저장 (SQLite/Postgres) — 현재 로그 기반으로 충분
- 진입 주문 체결 알림 (`TRADE` + not reduceOnly) — 기존 `notify_open()`으로 커버
- 부분 청산(partial fill) 처리 — 현재 봇은 전량 청산만 사용
