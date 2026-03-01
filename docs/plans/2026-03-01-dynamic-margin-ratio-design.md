# 동적 증거금 비율 설계

**날짜**: 2026-03-01  
**목적**: 잔고의 50%를 증거금으로 사용하되, 잔고가 늘어날수록 비율이 선형으로 감소하는 안전한 포지션 크기 계산 도입

---

## 배경

- 현재 포지션 크기 계산: `risk_per_trade = 0.02` (잔고의 2%) × 레버리지 → 명목금액
- 현재 잔고 22 USDT 기준, 최소 명목금액(5 USDT) 보장 로직으로 5 USDT 포지션만 잡힘
- 목표: 잔고의 50%를 증거금으로 활용하여 실질적인 포지션 크기 확보
- 안전장치: 잔고가 늘수록 비율이 자동으로 줄어들어 과도한 노출 방지

---

## 아키텍처

### 데이터 흐름

```
bot.run()
  └─ balance = await exchange.get_balance()
  └─ risk.set_base_balance(balance)          ← 봇 시작 시 1회

bot._open_position()
  └─ balance = await exchange.get_balance()
  └─ margin_ratio = risk.get_dynamic_margin_ratio(balance)   ← 신규
  └─ exchange.calculate_quantity(balance, price, leverage, margin_ratio)
```

### 비율 계산 공식

```
ratio = MAX_RATIO - (balance - base_balance) × DECAY_RATE
ratio = clamp(ratio, MIN_RATIO, MAX_RATIO)
```

- `base_balance`: 봇 시작 시 바이낸스 API로 조회한 실제 잔고
- `MAX_RATIO`: 잔고가 기준값일 때 최대 비율 (기본 50%)
- `MIN_RATIO`: 잔고가 아무리 늘어도 내려가지 않는 하한 비율 (기본 20%)
- `DECAY_RATE`: 잔고 1 USDT 증가당 비율 감소량 (기본 0.0006)

### 시뮬레이션 (기본 파라미터 기준)

| 잔고 | 증거금 비율 | 증거금 | 명목금액(×10배) |
|---|---|---|---|
| 22 USDT | 50.0% | 11.0 USDT | 110 USDT |
| 100 USDT | 45.3% | 45.3 USDT | 453 USDT |
| 300 USDT | 33.2% | 99.6 USDT | 996 USDT |
| 600 USDT | 20.0% (하한) | 120 USDT | 1,200 USDT |

---

## 변경 파일

### 1. `src/config.py`

`Config` 데이터클래스에 3개 파라미터 추가:

```python
margin_max_ratio: float = 0.50
margin_min_ratio: float = 0.20
margin_decay_rate: float = 0.0006
```

`__post_init__`에서 `.env` 값 읽기:

```python
self.margin_max_ratio = float(os.getenv("MARGIN_MAX_RATIO", "0.50"))
self.margin_min_ratio = float(os.getenv("MARGIN_MIN_RATIO", "0.20"))
self.margin_decay_rate = float(os.getenv("MARGIN_DECAY_RATE", "0.0006"))
```

### 2. `src/risk_manager.py`

메서드 2개 추가:

```python
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

### 3. `src/exchange.py`

`calculate_quantity` 시그니처에 `margin_ratio` 파라미터 추가:

```python
def calculate_quantity(self, balance: float, price: float, leverage: int, margin_ratio: float) -> float:
    notional = balance * margin_ratio * leverage
    if notional < self.MIN_NOTIONAL:
        notional = self.MIN_NOTIONAL
    ...
```

기존 `risk_per_trade` 기반 로직 제거.

### 4. `src/bot.py`

- `run()`: 시작 시 잔고 조회 후 `risk.set_base_balance(balance)` 호출
- `_open_position()`: `margin_ratio = self.risk.get_dynamic_margin_ratio(balance)` 호출 후 `calculate_quantity`에 전달

### 5. `.env`

```
MARGIN_MAX_RATIO=0.50
MARGIN_MIN_RATIO=0.20
MARGIN_DECAY_RATE=0.0006
```

---

## 제거되는 설정

- `RISK_PER_TRADE` — `.env` 및 `Config`에서 제거 (동적 비율로 대체)

---

## 리스크 고려사항

- 잔고 22 USDT × 50% × 10배 레버리지 = 명목금액 110 USDT 노출 (잔고의 5배)
- 손실 시 잔고가 줄어들면 다음 포지션 크기도 자동으로 줄어드는 자연스러운 안전장치 존재
- `MARGIN_DECAY_RATE` 조정으로 감소 속도 제어 가능
