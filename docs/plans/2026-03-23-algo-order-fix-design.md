# Algo Order 호환성 수정 설계

## 배경

실전 바이낸스 API 검증 결과, 조건부 주문(STOP_MARKET, TAKE_PROFIT_MARKET)이 Algo Order로 처리되며 테스트넷과 동작이 다름이 확인됨.

### 검증 결과 요약

| 항목 | 테스트넷 | 실전 |
|------|---------|------|
| SL/TP 응답 | `orderId` 반환 | `algoId`만 반환, orderId=None |
| SL 트리거 UDS | `ot=STOP_MARKET` | `ot=MARKET` |
| SL 후 TP 자동만료 | EXPIRED 이벤트 수신 | 만료 안 됨 → 고아주문 |
| `get_open_orders()` | algo 주문 조회됨 | algo 주문 조회 안 됨 |
| `cancel_all_orders()` | algo 주문 취소됨 | algo 주문 취소 안 됨 |
| UDS `i` 필드 vs 배치 ID | 동일 | `i` ≠ `algoId` (서로 다른 값) |

## 수정 대상 파일

1. `src/exchange.py` — algo API 병행 호출
2. `src/bot.py` — SL/TP 가격 저장, close_reason 판별, 복구 로직
3. `src/user_data_stream.py` — 가격 기반 close_reason 판별
4. `tests/` — 변경사항 반영

## 설계

### 1. exchange.py: Algo API 병행

**`cancel_all_orders()`**: 일반 주문 취소 + algo 주문 전체 취소를 모두 호출.

```python
async def cancel_all_orders(self):
    await self._run_api(
        lambda: self.client.futures_cancel_all_open_orders(symbol=self.symbol)
    )
    try:
        await self._run_api(
            lambda: self.client.futures_cancel_all_algo_open_orders(symbol=self.symbol)
        )
    except Exception:
        pass  # algo 주문 없으면 실패 가능 — 무시
```

**`cancel_order()`**: ID 크기나 타입으로 분기하지 않고, 일반 취소 시도 → 실패 시 algo 취소 (현재와 동일, 이미 올바른 구조).

**`get_open_orders()`**: 일반 주문 + algo 주문을 병합 반환. algo 주문 응답의 필드명이 다르므로 정규화 필요.

```python
async def get_open_orders(self) -> list[dict]:
    orders = await self._run_api(
        lambda: self.client.futures_get_open_orders(symbol=self.symbol)
    )
    try:
        algo_orders = await self._run_api(
            lambda: self.client.futures_get_algo_open_orders(symbol=self.symbol)
        )
        for ao in algo_orders.get("orders", []):
            orders.append({
                "orderId": ao.get("algoId"),
                "type": ao.get("orderType"),   # STOP_MARKET / TAKE_PROFIT_MARKET
                "stopPrice": ao.get("triggerPrice"),
                "side": ao.get("side"),
                "status": ao.get("algoStatus"),
                "_is_algo": True,
            })
    except Exception:
        pass
    return orders
```

### 2. bot.py: SL/TP 가격 저장 + close_reason 판별

**새 필드 추가** (`__init__`):
```python
self._sl_price: float | None = None
self._tp_price: float | None = None
```

**`_open_position()`**: SL/TP 배치 후 가격 저장.
```python
# _place_sl_tp_with_retry 호출 전에 이미 stop_loss, take_profit 계산됨
self._sl_price = stop_loss
self._tp_price = take_profit
```

**`_ensure_sl_tp_orders()` (복구)**: 오픈 주문에서 SL/TP 가격 복원.
```python
for o in open_orders:
    otype = o.get("type", "")
    if otype == "STOP_MARKET":
        self._sl_price = float(o.get("stopPrice", 0))
    elif otype == "TAKE_PROFIT_MARKET":
        self._tp_price = float(o.get("stopPrice", 0))
```

**`_on_position_closed()`**: close_reason이 "MANUAL"일 때 가격 비교로 재판별.
```python
if close_reason == "MANUAL" and self._sl_price and self._tp_price:
    sl_dist = abs(exit_price - self._sl_price)
    tp_dist = abs(exit_price - self._tp_price)
    if sl_dist < tp_dist:
        close_reason = "SL"
    else:
        close_reason = "TP"
```

**상태 초기화**: `_on_position_closed()` 및 `_close_and_reenter()`에서 포지션 Flat 전환 시:
```python
self._sl_price = None
self._tp_price = None
```

### 3. user_data_stream.py: close_reason을 콜백에 위임

UDS의 close_reason 판별 로직은 유지하되, 콜백 시그니처에 `exit_price`가 이미 전달되므로 bot.py에서 재판별 가능. UDS 자체는 변경 최소화.

현재 UDS에서 `ot`로 판별 → 실전에서 `ot=MARKET` → `close_reason="MANUAL"` → bot.py에서 가격 비교로 SL/TP 재판별. 이 흐름이 테스트넷에서도 안전 (테스트넷은 `ot=STOP_MARKET`이 오므로 재판별 자체가 불필요).

### 4. 포지션 모니터 SYNC 경로 — 이미 구현됨

`_position_monitor()`의 SYNC 폴백에서 잔여주문 취소는 **이미 구현되어 있음**. 추가 수정 불필요.

> **참고**: `_place_sl_tp_with_retry()`의 algoId 저장도 이미 구현됨 (bot.py line 539, 550).

### 5. 테스트 계획

- 테스트넷에서 SL 트리거 → TP 고아주문 자동 취소 확인
- 테스트넷에서 TP 트리거 → SL 고아주문 자동 취소 확인
- 테스트넷에서 역방향 재진입 → 기존 SL/TP 취소 확인
- 봇 재시작 → SL/TP 가격 복원 확인
- close_reason이 SL/TP로 정확히 분류되는지 확인
- 위 모든 항목 통과 후 실전 배포
