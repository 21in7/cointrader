# 15분봉 타임프레임 업그레이드 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 1분봉 파이프라인 전체를 15분봉으로 전환하고, LOOKAHEAD=24(6시간 뷰)로 조정해 모델 AUC를 0.49~0.50 구간에서 0.53+ 이상으로 개선한다.

**Architecture:** 데이터 수집(fetch_history.py) → 데이터셋 빌더(dataset_builder.py) → 학습 스크립트(train_model.py, train_mlx_model.py) → 실시간 봇(bot.py, data_stream.py) 순서로 파라미터를 변경한다. 각 레이어는 `interval` 문자열과 `LOOKAHEAD` 상수만 수정하면 되며 피처 구조는 그대로 유지한다.

**Tech Stack:** Python, LightGBM, pandas, binance-python-client, pytest

---

## 변경 요약

| 파일 | 변경 내용 |
|------|-----------|
| `src/dataset_builder.py` | `LOOKAHEAD 90→24`, `WARMUP 60→60` (유지) |
| `scripts/train_model.py` | `LOOKAHEAD 60→24`, `--data` 기본값 `combined_1m→combined_15m` |
| `scripts/train_mlx_model.py` | `--data` 기본값 `combined_1m→combined_15m` |
| `scripts/fetch_history.py` | `--interval` 기본값 `1m→15m`, `--output` 기본값 반영 |
| `scripts/train_and_deploy.sh` | `--interval 1m→15m`, 파일명 `1m→15m` |
| `src/bot.py` | `interval="1m"→"15m"` |
| `src/data_stream.py` | `buffer_size` 기본값 `200→200` (유지, 15분봉 200개=50시간 충분) |

---

## Task 1: dataset_builder.py — LOOKAHEAD 상수 변경

**Files:**
- Modify: `src/dataset_builder.py:14-17`

**Step 1: 현재 상수 확인**

```bash
head -20 src/dataset_builder.py
```

Expected: `LOOKAHEAD = 90`, `WARMUP = 60`

**Step 2: 상수 변경**

`src/dataset_builder.py` 14번째 줄:
```python
# 변경 전
LOOKAHEAD    = 90
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 2.0
WARMUP       = 60

# 변경 후
LOOKAHEAD    = 24   # 15분봉 × 24 = 6시간 뷰
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 2.0
WARMUP       = 60   # 15분봉 기준 60캔들 = 15시간 (지표 안정화 충분)
```

**Step 3: 변경 확인**

```bash
head -20 src/dataset_builder.py
```

Expected: `LOOKAHEAD = 24`

---

## Task 2: train_model.py — LOOKAHEAD 상수 및 기본 데이터 경로 변경

**Files:**
- Modify: `scripts/train_model.py:56-61`, `scripts/train_model.py:360`

**Step 1: 현재 상수 확인**

```bash
sed -n '55,62p' scripts/train_model.py
sed -n '358,362p' scripts/train_model.py
```

Expected: `LOOKAHEAD = 60`, `--data default="data/combined_1m.parquet"`

**Step 2: LOOKAHEAD 변경**

`scripts/train_model.py` 56번째 줄:
```python
# 변경 전
LOOKAHEAD = 60

# 변경 후
LOOKAHEAD = 24  # 15분봉 × 24 = 6시간 (dataset_builder.py와 동기화)
```

**Step 3: --data 기본값 변경**

`scripts/train_model.py` 360번째 줄 근처 `argparse` 부분:
```python
# 변경 전
parser.add_argument("--data", default="data/combined_1m.parquet")

# 변경 후
parser.add_argument("--data", default="data/combined_15m.parquet")
```

**Step 4: 변경 확인**

```bash
grep -n "LOOKAHEAD\|combined_" scripts/train_model.py
```

Expected: `LOOKAHEAD = 24`, `combined_15m.parquet`

---

## Task 3: train_mlx_model.py — 기본 데이터 경로 변경

**Files:**
- Modify: `scripts/train_mlx_model.py:149`

**Step 1: 현재 기본값 확인**

```bash
grep -n "combined_" scripts/train_mlx_model.py
```

Expected: `default="data/combined_1m.parquet"`

**Step 2: 기본값 변경**

`scripts/train_mlx_model.py` 149번째 줄:
```python
# 변경 전
parser.add_argument("--data", default="data/combined_1m.parquet")

# 변경 후
parser.add_argument("--data", default="data/combined_15m.parquet")
```

**Step 3: 변경 확인**

```bash
grep -n "combined_" scripts/train_mlx_model.py
```

Expected: `combined_15m.parquet`

---

## Task 4: fetch_history.py — 기본 interval 및 output 변경

**Files:**
- Modify: `scripts/fetch_history.py:114-118`

**Step 1: 현재 argparse 기본값 확인**

```bash
sed -n '112,120p' scripts/fetch_history.py
```

Expected: `--interval default="1m"`, `--output default="data/xrpusdt_1m.parquet"`

**Step 2: 기본값 변경**

```python
# 변경 전
parser.add_argument("--interval", default="1m")
parser.add_argument("--days",     type=int, default=90)
parser.add_argument("--output",   default="data/xrpusdt_1m.parquet")

# 변경 후
parser.add_argument("--interval", default="15m")
parser.add_argument("--days",     type=int, default=365)
parser.add_argument("--output",   default="data/xrpusdt_15m.parquet")
```

**Step 3: 변경 확인**

```bash
grep -n "interval\|output\|days" scripts/fetch_history.py | grep "default"
```

Expected: `default="15m"`, `default=365`, `default="data/xrpusdt_15m.parquet"`

---

## Task 5: train_and_deploy.sh — interval 및 파일명 변경

**Files:**
- Modify: `scripts/train_and_deploy.sh:26-43`

**Step 1: 현재 스크립트 확인**

```bash
cat scripts/train_and_deploy.sh
```

**Step 2: 스크립트 변경**

```bash
# 변경 전 (26~32번째 줄)
echo "=== [1/3] 데이터 수집 (XRP + BTC + ETH 3심볼, 1년치) ==="
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 1m \
    --days 365 \
    --output data/xrpusdt_1m.parquet
# 결과: data/combined_1m.parquet (타임스탬프 기준 병합)

# 변경 후
echo "=== [1/3] 데이터 수집 (XRP + BTC + ETH 3심볼, 1년치) ==="
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/xrpusdt_15m.parquet
# 결과: data/combined_15m.parquet (타임스탬프 기준 병합)
```

```bash
# 변경 전 (38~43번째 줄)
    python scripts/train_mlx_model.py --data data/combined_1m.parquet --decay "$DECAY"
else
    echo "  백엔드: LightGBM (CPU), decay=${DECAY}"
    python scripts/train_model.py --data data/combined_1m.parquet --decay "$DECAY"

# 변경 후
    python scripts/train_mlx_model.py --data data/combined_15m.parquet --decay "$DECAY"
else
    echo "  백엔드: LightGBM (CPU), decay=${DECAY}"
    python scripts/train_model.py --data data/combined_15m.parquet --decay "$DECAY"
```

**Step 3: 변경 확인**

```bash
grep -n "1m\|15m" scripts/train_and_deploy.sh
```

Expected: 모든 `1m` 참조가 `15m`으로 변경됨

---

## Task 6: bot.py — 실시간 스트림 interval 변경

**Files:**
- Modify: `src/bot.py:22-25`

**Step 1: 현재 interval 확인**

```bash
grep -n "interval" src/bot.py
```

Expected: `interval="1m"` (MultiSymbolStream 생성자)

**Step 2: interval 변경**

`src/bot.py` 21~25번째 줄:
```python
# 변경 전
self.stream = MultiSymbolStream(
    symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
    interval="1m",
    on_candle=self._on_candle_closed,
)

# 변경 후
self.stream = MultiSymbolStream(
    symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
    interval="15m",
    on_candle=self._on_candle_closed,
)
```

**Step 3: 변경 확인**

```bash
grep -n "interval" src/bot.py
```

Expected: `interval="15m"`

---

## Task 7: 전체 변경 검증

**Step 1: 모든 `1m` 하드코딩 잔재 확인**

```bash
grep -rn '"1m"' src/ scripts/
```

Expected: 결과 없음 (모두 `"15m"`으로 변경됨)

**Step 2: LOOKAHEAD 동기화 확인**

```bash
grep -rn "LOOKAHEAD" src/ scripts/
```

Expected:
- `src/dataset_builder.py`: `LOOKAHEAD = 24`
- `scripts/train_model.py`: `LOOKAHEAD = 24`

**Step 3: combined 파일명 일관성 확인**

```bash
grep -rn "combined_" src/ scripts/
```

Expected: 모두 `combined_15m` 참조

**Step 4: 파이프라인 드라이런 (데이터 없이 import 테스트)**

```bash
python -c "
from src.dataset_builder import LOOKAHEAD, ATR_SL_MULT, ATR_TP_MULT, WARMUP
assert LOOKAHEAD == 24, f'LOOKAHEAD={LOOKAHEAD}'
print(f'OK: LOOKAHEAD={LOOKAHEAD}, ATR_SL={ATR_SL_MULT}, ATR_TP={ATR_TP_MULT}, WARMUP={WARMUP}')
"
```

Expected: `OK: LOOKAHEAD=24, ATR_SL=1.5, ATR_TP=2.0, WARMUP=60`

---

## Task 8: 데이터 수집 및 Walk-Forward 검증 실행

> 이 태스크는 실제 바이낸스 API 키와 네트워크가 필요합니다.

**Step 1: 15분봉 데이터 수집**

```bash
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/xrpusdt_15m.parquet
```

Expected: `data/combined_15m.parquet` 생성, 약 35,040행 (365일 × 96캔들/일)

**Step 2: Walk-Forward AUC 측정 (기준선 확인)**

```bash
python scripts/train_model.py \
    --data data/combined_15m.parquet \
    --wf \
    --wf-splits 5
```

Expected: Walk-Forward 평균 AUC가 0.53 이상이면 개선 확인

**Step 3: 정식 학습 및 모델 저장**

```bash
python scripts/train_model.py \
    --data data/combined_15m.parquet \
    --decay 2.0
```

Expected: `models/lgbm_filter.pkl` 저장, 기존 모델은 `lgbm_filter_prev.pkl`로 백업

---

## 롤백 방법

15분봉 모델이 기대에 미치지 못할 경우:

```bash
# 기존 1분봉 모델 복원
cp models/lgbm_filter_prev.pkl models/lgbm_filter.pkl

# 코드는 git으로 복원
git checkout src/dataset_builder.py scripts/train_model.py \
    scripts/train_mlx_model.py scripts/fetch_history.py \
    scripts/train_and_deploy.sh src/bot.py
```
