# 벡터화 데이터셋 빌더 + 컨테이너 재학습 제거 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 맥미니에서 전체 시계열을 1회 계산하는 벡터화 데이터셋 빌더로 교체해 학습 속도를 높이고, LXC 도커 컨테이너에서 자동 재학습 코드를 제거한다.

**Architecture:** `src/dataset_builder.py`에 벡터화 함수를 신규 작성하고 `scripts/train_model.py`, `scripts/train_mlx_model.py`에서 호출한다. `src/bot.py`에서 `Retrainer` 의존성을 제거하고 `src/retrainer.py`는 삭제한다. `src/indicators.py`, `src/ml_features.py`는 봇 실시간 경로이므로 변경하지 않는다.

**Tech Stack:** Python 3.13, pandas-ta, numpy, pandas, LightGBM, MLX

---

## 변경 범위 요약

| 파일 | 작업 |
|------|------|
| `src/dataset_builder.py` | 신규 — 벡터화 데이터셋 생성 |
| `scripts/train_model.py` | `generate_dataset` → `generate_dataset_vectorized` 교체 |
| `scripts/train_mlx_model.py` | 동일 |
| `src/bot.py` | `Retrainer` import·인스턴스·태스크 제거 |
| `src/retrainer.py` | 삭제 |
| `tests/test_retrainer.py` | 삭제 |
| `tests/test_dataset_builder.py` | 신규 — 벡터화 빌더 테스트 |
| `Dockerfile` | `mlx` 제외 처리 (Linux ARM에서 설치 불가) |
| `requirements.txt` | mlx를 Mac 전용 주석으로 표시 |

---

## Task 1: `src/dataset_builder.py` 신규 작성

**핵심 아이디어**: `pandas_ta`를 전체 시계열에 1번만 호출하고, 신호 조건·피처·레이블을 모두 numpy 배열 연산으로 처리한다.

**Files:**
- Create: `src/dataset_builder.py`
- Create: `tests/test_dataset_builder.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_dataset_builder.py
import numpy as np
import pandas as pd
import pytest
from src.dataset_builder import generate_dataset_vectorized

@pytest.fixture
def sample_df():
    """최소 200행 이상의 OHLCV 더미 데이터."""
    rng = np.random.default_rng(42)
    n = 500
    close = 2.0 + np.cumsum(rng.normal(0, 0.01, n))
    close = np.clip(close, 0.01, None)
    high  = close * (1 + rng.uniform(0, 0.005, n))
    low   = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({
        "open":   close,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.uniform(1e6, 5e6, n),
    })

def test_returns_dataframe(sample_df):
    """결과가 DataFrame이어야 한다."""
    result = generate_dataset_vectorized(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_has_required_columns(sample_df):
    """FEATURE_COLS + label 컬럼이 모두 있어야 한다."""
    from src.ml_features import FEATURE_COLS
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert "label" in result.columns
        for col in FEATURE_COLS:
            assert col in result.columns, f"컬럼 없음: {col}"

def test_label_is_binary(sample_df):
    """label은 0 또는 1만 있어야 한다."""
    result = generate_dataset_vectorized(sample_df)
    if len(result) > 0:
        assert set(result["label"].unique()).issubset({0, 1})

def test_matches_original_generate_dataset(sample_df):
    """벡터화 버전과 기존 버전의 샘플 수가 동일해야 한다."""
    from scripts.train_model import generate_dataset
    orig = generate_dataset(sample_df, n_jobs=1)
    vec  = generate_dataset_vectorized(sample_df)
    assert len(vec) == len(orig), (
        f"샘플 수 불일치: 벡터화={len(vec)}, 기존={len(orig)}"
    )
```

**Step 2: 테스트 실행 (실패 확인)**

```bash
cd /Users/gihyeon/github/cointrader
.venv/bin/python -m pytest tests/test_dataset_builder.py -v
```

Expected: `ImportError: cannot import name 'generate_dataset_vectorized'`

**Step 3: `src/dataset_builder.py` 구현**

```python
# src/dataset_builder.py
"""
전체 시계열을 1회 계산하는 벡터화 데이터셋 빌더.
pandas_ta를 130,000번 반복 호출하는 기존 방식 대신
전체 배열에 1번만 적용해 10~30배 속도를 낸다.

봇 실시간 경로(indicators.py, ml_features.py)는 변경하지 않는다.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.ml_features import FEATURE_COLS

LOOKAHEAD    = 60
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 3.0
WARMUP       = 60   # 지표 안정화에 필요한 최소 행 수


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """전체 시계열에 기술 지표를 1회 계산한다."""
    d = df.copy()
    close  = d["close"]
    high   = d["high"]
    low    = d["low"]
    volume = d["volume"]

    d["rsi"]  = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    d["macd"]        = macd["MACD_12_26_9"]
    d["macd_signal"] = macd["MACDs_12_26_9"]
    d["macd_hist"]   = macd["MACDh_12_26_9"]

    bb = ta.bbands(close, length=20, std=2)
    d["bb_upper"] = bb["BBU_20_2.0_2.0"]
    d["bb_lower"] = bb["BBL_20_2.0_2.0"]

    d["ema9"]  = ta.ema(close, length=9)
    d["ema21"] = ta.ema(close, length=21)
    d["ema50"] = ta.ema(close, length=50)

    d["atr"]     = ta.atr(high, low, close, length=14)
    d["vol_ma20"] = ta.sma(volume, length=20)

    stoch = ta.stochrsi(close, length=14)
    d["stoch_k"] = stoch["STOCHRSIk_14_14_3_3"]
    d["stoch_d"] = stoch["STOCHRSId_14_14_3_3"]

    return d


def _calc_signals(d: pd.DataFrame) -> np.ndarray:
    """
    indicators.py get_signal() 로직을 numpy 배열 연산으로 재현한다.
    반환: signal_arr — 각 행에 대해 "LONG" | "SHORT" | "HOLD"
    """
    n = len(d)

    rsi      = d["rsi"].values
    macd     = d["macd"].values
    macd_sig = d["macd_signal"].values
    close    = d["close"].values
    bb_upper = d["bb_upper"].values
    bb_lower = d["bb_lower"].values
    ema9     = d["ema9"].values
    ema21    = d["ema21"].values
    ema50    = d["ema50"].values
    stoch_k  = d["stoch_k"].values
    stoch_d  = d["stoch_d"].values
    volume   = d["volume"].values
    vol_ma20 = d["vol_ma20"].values

    # MACD 크로스: 전 캔들과 비교 (shift(1))
    prev_macd     = np.roll(macd, 1);     prev_macd[0]     = np.nan
    prev_macd_sig = np.roll(macd_sig, 1); prev_macd_sig[0] = np.nan

    long_score  = np.zeros(n, dtype=np.float32)
    short_score = np.zeros(n, dtype=np.float32)

    # 1. RSI
    long_score  += (rsi < 35).astype(np.float32)
    short_score += (rsi > 65).astype(np.float32)

    # 2. MACD 크로스 (가중치 2)
    macd_cross_up   = (prev_macd < prev_macd_sig) & (macd > macd_sig)
    macd_cross_down = (prev_macd > prev_macd_sig) & (macd < macd_sig)
    long_score  += macd_cross_up.astype(np.float32)   * 2
    short_score += macd_cross_down.astype(np.float32) * 2

    # 3. 볼린저 밴드
    long_score  += (close < bb_lower).astype(np.float32)
    short_score += (close > bb_upper).astype(np.float32)

    # 4. EMA 정배열/역배열
    long_score  += ((ema9 > ema21) & (ema21 > ema50)).astype(np.float32)
    short_score += ((ema9 < ema21) & (ema21 < ema50)).astype(np.float32)

    # 5. Stochastic RSI
    long_score  += ((stoch_k < 20) & (stoch_k > stoch_d)).astype(np.float32)
    short_score += ((stoch_k > 80) & (stoch_k < stoch_d)).astype(np.float32)

    # 6. 거래량 급증
    vol_surge = volume > vol_ma20 * 1.5

    long_enter  = (long_score  >= 3) & (vol_surge | (long_score  >= 4))
    short_enter = (short_score >= 3) & (vol_surge | (short_score >= 4))

    signal_arr = np.full(n, "HOLD", dtype=object)
    signal_arr[long_enter]  = "LONG"
    signal_arr[short_enter] = "SHORT"
    # 둘 다 해당하면 HOLD (충돌 방지)
    signal_arr[long_enter & short_enter] = "HOLD"

    return signal_arr


def _calc_features_vectorized(d: pd.DataFrame, signal_arr: np.ndarray) -> pd.DataFrame:
    """
    신호 발생 인덱스에서 ml_features.py build_features() 로직을
    pandas 벡터 연산으로 재현한다.
    """
    close    = d["close"]
    bb_upper = d["bb_upper"]
    bb_lower = d["bb_lower"]
    ema9     = d["ema9"]
    ema21    = d["ema21"]
    ema50    = d["ema50"]
    atr      = d["atr"]
    volume   = d["volume"]
    vol_ma20 = d["vol_ma20"]
    rsi      = d["rsi"]
    macd_hist = d["macd_hist"]
    stoch_k  = d["stoch_k"]
    stoch_d  = d["stoch_d"]
    macd     = d["macd"]
    macd_sig = d["macd_signal"]

    bb_range = bb_upper - bb_lower
    bb_pct = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)

    ema_align = np.where(
        (ema9 > ema21) & (ema21 > ema50),  1,
        np.where(
            (ema9 < ema21) & (ema21 < ema50), -1, 0
        )
    ).astype(np.float32)

    atr_pct   = np.where(close > 0, atr / close, 0.0)
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 1.0)

    ret_1 = close.pct_change(1).fillna(0).values
    ret_3 = close.pct_change(3).fillna(0).values
    ret_5 = close.pct_change(5).fillna(0).values

    prev_macd     = macd.shift(1).fillna(0).values
    prev_macd_sig = macd_sig.shift(1).fillna(0).values

    # signal_strength: 신호 방향별로 각 조건 점수 합산
    is_long  = (signal_arr == "LONG")
    is_short = (signal_arr == "SHORT")

    strength = np.zeros(len(d), dtype=np.float32)

    # LONG 조건
    strength += is_long * (rsi.values < 35).astype(np.float32)
    strength += is_long * ((prev_macd < prev_macd_sig) & (macd.values > macd_sig.values)).astype(np.float32) * 2
    strength += is_long * (close.values < bb_lower.values).astype(np.float32)
    strength += is_long * (ema_align == 1).astype(np.float32)
    strength += is_long * ((stoch_k.values < 20) & (stoch_k.values > stoch_d.values)).astype(np.float32)

    # SHORT 조건
    strength += is_short * (rsi.values > 65).astype(np.float32)
    strength += is_short * ((prev_macd > prev_macd_sig) & (macd.values < macd_sig.values)).astype(np.float32) * 2
    strength += is_short * (close.values > bb_upper.values).astype(np.float32)
    strength += is_short * (ema_align == -1).astype(np.float32)
    strength += is_short * ((stoch_k.values > 80) & (stoch_k.values < stoch_d.values)).astype(np.float32)

    side = np.where(signal_arr == "LONG", 1.0, 0.0).astype(np.float32)

    return pd.DataFrame({
        "rsi":             rsi.values.astype(np.float32),
        "macd_hist":       macd_hist.values.astype(np.float32),
        "bb_pct":          bb_pct.astype(np.float32),
        "ema_align":       ema_align,
        "stoch_k":         stoch_k.values.astype(np.float32),
        "stoch_d":         stoch_d.values.astype(np.float32),
        "atr_pct":         atr_pct.astype(np.float32),
        "vol_ratio":       vol_ratio.astype(np.float32),
        "ret_1":           ret_1.astype(np.float32),
        "ret_3":           ret_3.astype(np.float32),
        "ret_5":           ret_5.astype(np.float32),
        "signal_strength": strength,
        "side":            side,
        "_signal":         signal_arr,   # 레이블 계산용 임시 컬럼
    }, index=d.index)


def _calc_labels_vectorized(
    d: pd.DataFrame,
    feat: pd.DataFrame,
    sig_idx: np.ndarray,
) -> np.ndarray:
    """
    label_builder.py build_labels() 로직을 numpy 2D 배열로 벡터화한다.

    각 신호 인덱스 i에 대해 future[i+1 : i+1+LOOKAHEAD] 구간의
    high/low 배열을 (N × LOOKAHEAD) 행렬로 만들어 argmax로 처리한다.
    """
    n_total = len(d)
    highs   = d["high"].values
    lows    = d["low"].values
    closes  = d["close"].values
    atrs    = d["atr"].values

    labels = []
    valid_mask = []

    for idx in sig_idx:
        signal = feat.at[d.index[idx], "_signal"]
        entry  = closes[idx]
        atr    = atrs[idx]
        if atr <= 0:
            valid_mask.append(False)
            continue

        if signal == "LONG":
            sl = entry - atr * ATR_SL_MULT
            tp = entry + atr * ATR_TP_MULT
        else:
            sl = entry + atr * ATR_SL_MULT
            tp = entry - atr * ATR_TP_MULT

        end = min(idx + 1 + LOOKAHEAD, n_total)
        fut_high = highs[idx + 1 : end]
        fut_low  = lows[idx + 1 : end]

        label = None
        for h, l in zip(fut_high, fut_low):
            if signal == "LONG":
                if h >= tp:
                    label = 1
                    break
                if l <= sl:
                    label = 0
                    break
            else:
                if l <= tp:
                    label = 1
                    break
                if h >= sl:
                    label = 0
                    break

        if label is None:
            valid_mask.append(False)
        else:
            labels.append(label)
            valid_mask.append(True)

    return np.array(labels, dtype=np.int8), np.array(valid_mask, dtype=bool)


def generate_dataset_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 시계열을 1회 계산해 학습 데이터셋을 생성한다.
    기존 generate_dataset()의 drop-in 대체제.
    """
    print("  [1/3] 전체 시계열 지표 계산 (1회)...")
    d = _calc_indicators(df)

    print("  [2/3] 신호 마스킹 및 피처 추출...")
    signal_arr = _calc_signals(d)
    feat_all   = _calc_features_vectorized(d, signal_arr)

    # 신호 발생 + NaN 없음 + 미래 데이터 충분한 인덱스만
    valid_rows = (
        (signal_arr != "HOLD") &
        (~feat_all[FEATURE_COLS].isna().any(axis=1).values) &
        (np.arange(len(d)) >= WARMUP) &
        (np.arange(len(d)) < len(d) - LOOKAHEAD)
    )
    sig_idx = np.where(valid_rows)[0]
    print(f"  신호 발생 인덱스: {len(sig_idx):,}개")

    print("  [3/3] 레이블 계산...")
    labels, valid_mask = _calc_labels_vectorized(d, feat_all, sig_idx)

    final_idx = sig_idx[valid_mask]
    feat_final = feat_all.iloc[final_idx][FEATURE_COLS].copy()
    feat_final["label"] = labels

    return feat_final.reset_index(drop=True)
```

**Step 4: 테스트 실행 (통과 확인)**

```bash
.venv/bin/python -m pytest tests/test_dataset_builder.py -v
```

Expected: 4 passed

**Step 5: 커밋**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: add vectorized dataset builder (1x pandas_ta call)"
```

---

## Task 2: `scripts/train_model.py` 교체

**Files:**
- Modify: `scripts/train_model.py`

**Step 1: `generate_dataset` 호출을 벡터화 버전으로 교체**

`scripts/train_model.py` 상단 import에 추가:
```python
from src.dataset_builder import generate_dataset_vectorized
```

`train()` 함수 내 `generate_dataset(df, n_jobs=n_jobs)` 호출을 교체:
```python
# 기존
dataset = generate_dataset(df, n_jobs=n_jobs)

# 변경
dataset = generate_dataset_vectorized(df)
```

`main()`의 `--jobs` 인자 제거:
```python
# 기존
parser.add_argument("--jobs", type=int, default=None,
                    help="병렬 worker 수 (기본: CPU 수 - 1)")
args = parser.parse_args()
train(args.data, n_jobs=args.jobs)

# 변경
args = parser.parse_args()
train(args.data)
```

`train()` 함수 시그니처에서 `n_jobs` 파라미터 제거:
```python
# 기존
def train(data_path: str, n_jobs: int | None = None):

# 변경
def train(data_path: str):
```

**Step 2: 학습 실행 및 시간 측정**

```bash
time .venv/bin/python scripts/train_model.py --data data/xrpusdt_1m.parquet
```

Expected: 기존 130초 → 10초 이내

**Step 3: 커밋**

```bash
git add scripts/train_model.py
git commit -m "perf: replace generate_dataset with vectorized version in train_model"
```

---

## Task 3: `scripts/train_mlx_model.py` 교체

**Files:**
- Modify: `scripts/train_mlx_model.py`

**Step 1: import 교체**

`scripts/train_mlx_model.py` 상단에서:
```python
# 기존
from scripts.train_model import generate_dataset

# 변경
from src.dataset_builder import generate_dataset_vectorized
```

`train_mlx()` 함수 내 호출 교체:
```python
# 기존
dataset = generate_dataset(df)

# 변경
dataset = generate_dataset_vectorized(df)
```

**Step 2: 실행 확인**

```bash
time .venv/bin/python scripts/train_mlx_model.py --data data/xrpusdt_1m.parquet
```

**Step 3: 커밋**

```bash
git add scripts/train_mlx_model.py
git commit -m "perf: replace generate_dataset with vectorized version in train_mlx_model"
```

---

## Task 4: 컨테이너에서 재학습 제거

**Files:**
- Modify: `src/bot.py`
- Delete: `src/retrainer.py`
- Delete: `tests/test_retrainer.py`

**Step 1: `src/bot.py`에서 Retrainer 제거**

`src/bot.py`에서 다음 3곳을 수정:

```python
# 제거할 import
from src.retrainer import Retrainer

# 제거할 __init__ 코드
self.retrainer = Retrainer(ml_filter=self.ml_filter)

# 제거할 run() 코드
asyncio.create_task(self.retrainer.schedule_daily(hour=3))
```

**Step 2: `src/retrainer.py` 삭제**

```bash
rm src/retrainer.py
```

**Step 3: `tests/test_retrainer.py` 삭제**

```bash
rm tests/test_retrainer.py
```

**Step 4: 기존 테스트 전체 통과 확인**

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_retrainer.py
```

Expected: 모든 테스트 통과

**Step 5: 커밋**

```bash
git add src/bot.py
git rm src/retrainer.py tests/test_retrainer.py
git commit -m "feat: remove in-container retraining, training is now mac-only"
```

---

## Task 5: Dockerfile에서 mlx 제외

`mlx`는 Apple Silicon 전용이라 Linux(LXC) 컨테이너에서 설치 불가.

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

**Step 1: `requirements.txt`에서 mlx 조건부 처리**

`requirements.txt`에서:
```
# 변경 전
mlx>=0.22.0

# 변경 후 (삭제 — Dockerfile에서 별도 처리)
```
mlx 줄을 삭제한다.

**Step 2: `Dockerfile`에 mlx 제외 명시**

```dockerfile
# 변경 전
RUN pip install --no-cache-dir -r requirements.txt

# 변경 후
RUN pip install --no-cache-dir -r requirements.txt
# mlx는 Apple Silicon 전용이므로 컨테이너에 설치하지 않는다
```

실제로는 requirements.txt에서 mlx를 제거하는 것만으로 충분하다.
맥미니에서는 수동으로 설치:
```bash
pip install mlx>=0.22.0
```

**Step 3: README 업데이트**

`README.md`의 "Apple Silicon GPU 가속 학습" 섹션에 설치 안내 추가:
```markdown
> **설치**: `mlx`는 Apple Silicon 전용이며 `requirements.txt`에 포함되지 않습니다.
> 맥미니에서 별도 설치: `pip install mlx`
```

**Step 4: 커밋**

```bash
git add requirements.txt Dockerfile README.md
git commit -m "chore: exclude mlx from container requirements (Apple Silicon only)"
```

---

## Task 6: 전체 검증 및 속도 비교

**Step 1: 프로파일러로 최종 속도 측정**

```bash
time .venv/bin/python scripts/train_model.py --data data/xrpusdt_1m.parquet
```

Expected: 10초 이내 (기존 130초 대비 10배+ 향상)

**Step 2: 전체 테스트 통과 확인**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 모든 테스트 통과 (test_retrainer.py 제외)

**Step 3: train_and_deploy.sh 전체 파이프라인 dry-run**

```bash
bash scripts/train_and_deploy.sh 2>&1 | head -30
```

**Step 4: 최종 커밋 없음** — 각 Task에서 이미 커밋 완료
