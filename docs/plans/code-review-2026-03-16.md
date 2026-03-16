# CoinTrader 코드 점검 보고서

> 작성일: 2026-03-16  
> 대상: CoinTrader 전체 소스 코드 (bot.py, exchange.py, risk_manager.py, data_stream.py, user_data_stream.py, ml_filter.py, ml_features.py, config.py)

---

## 요약

| 심각도 | 건수 |
|--------|------|
| 🔴 심각 (버그 / 실제 자금 손실 위험) | 4 (✅ 전부 수정 완료) |
| 🟡 경고 (논리 오류 / 운영 리스크) | 6 (✅ 전부 수정 완료) |
| 🔵 개선 (코드 품질 / 유지보수) | 5 |

아키텍처 설계 자체(멀티심볼 독립 인스턴스, 공유 RiskManager)는 합리적이다. 문제는 멀티심볼 확장 과정에서 공유 상태(`RiskManager`)에 대한 동시성 처리가 불완전하고, 자금 관련 계산 로직(마진 비율, PnL 폴백)에 실제 버그가 존재한다는 점이다.

---

## 🔴 심각 — 버그 / 실제 자금 손실 위험

### 1. 마진 비율 계산 불일치 (`bot.py` L190-196)

**문제:**

```python
per_symbol_balance = balance / num_symbols          # 심볼별로 나눔
margin_ratio = self.risk.get_dynamic_margin_ratio(balance)  # 전체 잔고 기준
quantity = self.exchange.calculate_quantity(
    balance=per_symbol_balance,  # 나눈 값
    margin_ratio=margin_ratio    # 전체 기준 비율 → 불일치
)
```

`margin_ratio`는 전체 잔고 기준으로 계산되었는데, `per_symbol_balance`(나눈 값)에 곱해진다. 결과적으로 마진 비율 감소 효과가 의도한 것의 `num_symbols`배로 증폭된다.

**수정 방향:**

```python
per_symbol_balance = balance / num_symbols
margin_ratio = self.risk.get_dynamic_margin_ratio(per_symbol_balance)  # 나눈 값 기준
```

또는 전체 잔고로 수량을 계산하고 나중에 심볼 수로 나누는 방식으로 통일해야 한다.

---

### 2. `_place_algo_order`의 `algoType="CONDITIONAL"` 하드코딩 (`exchange.py` L149)

**문제:**

```python
params = dict(
    symbol=self.symbol,
    side=side,
    algoType="CONDITIONAL",  # 하드코딩
    type=order_type,
    ...
)
```

Binance FAPI `/fapi/v1/algoOrder`의 `algoType`은 `VP`, `TWAP` 등 실행 알고리즘용이다. `STOP_MARKET` / `TAKE_PROFIT_MARKET` 같은 조건부 주문은 `/fapi/v1/order`에 `reduceOnly=true`로 전송해야 한다. 이 경로가 실제로 동작하지 않으면 SL/TP 주문이 아예 등록되지 않아 무한 손실 가능.

**수정 방향:** 테스트넷에서 즉시 검증. 실패 시 일반 `place_order` 경로로 대체하고 `_place_algo_order` 삭제.

---

### 3. 폴백 PnL 계산 오류 (`bot.py` L328-334)

**문제:**

```python
pnl_rows, comm_rows = await self.exchange.get_recent_income(limit=5)
if pnl_rows:
    realized_pnl = float(pnl_rows[-1].get("income", "0"))  # 마지막 1건만 사용
```

멀티심볼 환경에서 `limit=5` 조회 시 다른 심볼의 PnL이 섞일 수 있다. 마지막 항목 하나만 쓰는 것은 다중 체결 건이 있을 때 틀린 값을 기록한다. SYNC 청산에서 잘못된 PnL이 기록되면 `daily_pnl`이 오염되어 손실 한도 체크 자체가 무의미해진다.

**수정 방향:** 조회 시 `symbol` 파라미터로 필터링하고, 해당 포지션의 거래 ID 범위를 기준으로 합산해야 한다.

---

### 4. `_is_reentering` 타이밍 레이스 컨디션 (`bot.py` L401, L421)

**문제:**

```python
self._is_reentering = True
try:
    await self._close_position(position)   # 청산 주문 전송
    # ← 이 시점에 User Data Stream 콜백 도착 가능
    await self._open_position(signal, df)  # 신규 진입
finally:
    self._is_reentering = False
```

청산 주문 전송 직후 User Data Stream 콜백이 도착하면, `_is_reentering = True`인 상태에서 `risk.close_position`이 호출된다. 그 직후 `_open_position`이 `risk.register_position`을 호출하며 상태가 겹친다. `asyncio`의 단일 스레드 특성 덕분에 `await` 사이에는 안전하지만, 콜백 순서와 타이밍에 따라 포지션 카운트가 틀어질 수 있다.

**수정 방향:** `_close_and_reenter` 내에서 포지션 상태 전환을 명시적으로 관리하고, `_on_position_closed`에서 `_is_reentering` 플래그를 확인하는 것 외에도 명시적인 상태 머신 전환을 추가한다.

---

## 🟡 경고 — 논리 오류 / 운영 리스크

### 5. `reset_daily()` 자동 호출 없음 (`risk_manager.py`)

메서드는 정의되어 있으나 어디서도 호출되지 않는다. 봇이 며칠 연속 실행되면 `daily_pnl`이 계속 누적되어 일일 손실 한도 체크가 무의미해진다.

**수정 방향:**

```python
# main.py 또는 bot.run() 내에서
async def _daily_reset_loop(risk: RiskManager):
    while True:
        now = datetime.utcnow()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        await asyncio.sleep((next_midnight - now).total_seconds())
        risk.reset_daily()
```

---

### 6. 공유 `RiskManager`에서 `set_base_balance` 경쟁 조건 (`bot.py` L429)

`asyncio.gather`로 3개 봇이 거의 동시에 `run()`을 실행하면 각자 `set_base_balance(balance)`를 호출한다. 마지막으로 호출한 봇의 잔고로 덮어씌워지며, Lock이 없어 순서도 보장되지 않는다.

**수정 방향:** `initial_balance` 설정을 `main.py`에서 한 번만 수행하고 공유 RiskManager에 주입하거나, 설정 시 Lock으로 보호한다.

---

### 7. 진입 주문이 청산으로 잘못 판별 가능 (`user_data_stream.py` L89)

```python
is_close = is_reduce or order_type in _CLOSE_ORDER_TYPES or realized_pnl != 0
```

일부 상황에서 진입 주문 체결 시 소액의 `rp`(실현 손익)가 붙는 경우가 있다. `realized_pnl != 0` 단독 조건이 너무 넓어 진입 주문이 청산으로 잘못 처리될 수 있다.

**수정 방향:**

```python
is_close = is_reduce or order_type in _CLOSE_ORDER_TYPES
# realized_pnl != 0 조건 제거
```

---

### 8. 피처 컬럼명이 XRP에 하드코딩 (`ml_features.py` L10-11)

```python
FEATURE_COLS = [
    ...
    "xrp_btc_rs", "xrp_eth_rs",  # XRP 하드코딩
]
```

TRX/DOGE 봇도 동일한 피처명을 사용한다. 학습과 추론 간 컬럼명 불일치는 없지만, 의미가 잘못되어 있고 심볼별 모델 학습 시 혼란을 유발한다.

**수정 방향:** `build_features_aligned` 함수에서 심볼명을 동적으로 포함하거나, 컬럼명을 `primary_btc_rs`, `primary_eth_rs`로 범용화한다.

---

### 9. `asyncio.get_event_loop()` deprecated 패턴 (`exchange.py` 전반)

Python 3.10+에서 실행 중인 루프가 없을 때 `get_event_loop()`은 `DeprecationWarning`을 발생시킨다.

**수정 방향:**

```python
# Before
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, lambda: ...)

# After
await asyncio.to_thread(lambda: ...)
# 또는
loop = asyncio.get_running_loop()
await loop.run_in_executor(None, lambda: ...)
```

---

### 10. 프리로드가 순차적으로 처리됨 (`data_stream.py` L164-183)

```python
for symbol in self.symbols:  # 순차 처리
    klines = await client.futures_klines(...)
```

심볼 3개를 순차 REST 조회하면 시작 시간이 약 3배 길어진다.

**수정 방향:**

```python
async def _preload_one(client, symbol):
    ...

await asyncio.gather(*[_preload_one(client, s) for s in self.symbols])
```

---

## 🔵 개선 — 코드 품질 / 유지보수

### 11. `config.py` 데드 필드

`stop_loss_pct`, `take_profit_pct`, `trailing_stop_pct`가 dataclass 기본값으로만 존재하고 `__post_init__`에서 환경변수로 로드되지 않는다. `atr_sl_mult`/`atr_tp_mult`로 대체되었으나 정리되지 않았다. 혼란을 줄이기 위해 삭제하거나 환경변수 로드를 추가해야 한다.

---

### 12. 매 캔들마다 불필요한 REST API 조회 (`bot.py` L158)

```python
position = await self.exchange.get_position()  # 15분마다 호출
```

`current_trade_side`로 로컬 상태를 이미 관리하고 있다. User Data Stream 콜백과 `_position_monitor` 폴백이 있으므로, `process_candle`에서는 로컬 상태만 확인하면 충분하다. 불필요한 API rate limit을 소비하고 있다.

---

### 13. `main.py` 파일 없음

README와 ARCHITECTURE.md에 진입점으로 언급되지만 실제 파일이 없다. 배포 시 어떻게 봇을 실행하는지 코드로 확인할 수 없다.

---

### 14. `MIN_NOTIONAL = 5.0` 하드코딩 (`exchange.py` L20)

Binance의 최소 명목금액은 심볼마다 다르고 정책 변경이 가능하다. `exchange_info`의 `filters`에서 `MIN_NOTIONAL` 또는 `NOTIONAL` 필터를 읽어야 정확하다.

---

### 15. ML 필터 예측 오류 시 무조건 진입 차단 (`ml_filter.py` L153)

```python
except Exception as e:
    logger.warning(f"ML 필터 예측 오류 (진입 차단): {e}")
    return False  # 모든 거래 차단
```

모델에 버그가 생기면 거래가 전면 중단된다. 오류 유형에 따라 `True`(폴백 허용)를 반환할지 `False`(차단)를 반환할지 구분하고, 오류 횟수를 카운팅하여 Discord 알림을 보내는 것이 바람직하다.

---

## 우선 처리 권장 순서

1. **즉시**: `_place_algo_order` API 경로 테스트넷 검증 (#2)
2. **즉시**: 마진 비율 계산 불일치 수정 (#1)
3. **이번 주**: `reset_daily()` 자동 호출 추가 (#5)
4. **이번 주**: `set_base_balance` 경쟁 조건 수정 (#6)
5. **이번 주**: 폴백 PnL 조회 로직 개선 (#3)
6. **다음 배포 전**: `is_close` 판별 조건 수정 (#7), `asyncio.get_event_loop` 교체 (#9), 프리로드 병렬화 (#10)
