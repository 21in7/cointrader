# ML 필터 (LightGBM) 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 기존 규칙 기반 신호(LONG/SHORT/HOLD)가 발생했을 때 LightGBM 모델이 수익 확률을 계산해 낮은 확률의 진입을 차단하는 보조 필터를 구현한다.

**Architecture:** 과거 캔들 데이터로 LightGBM을 오프라인 학습시키고, 봇의 `process_candle()`에서 규칙 기반 신호가 나오면 ML 필터가 확률을 계산해 0.60 미만이면 진입을 차단한다. 매일 새벽 3시에 자동 재학습하며 성능이 나빠지면 이전 모델로 롤백한다.

**Tech Stack:** Python 3.11+, lightgbm, scikit-learn, joblib, pandas, asyncio (기존 스택 유지)

---

## Task 1: 의존성 추가

**Files:**
- Modify: `requirements.txt`

**Step 1: requirements.txt에 ML 패키지 추가**

```
lightgbm>=4.3.0
scikit-learn>=1.4.0
joblib>=1.3.0
pyarrow>=15.0.0
```

**Step 2: 패키지 설치**

```bash
pip install lightgbm scikit-learn joblib pyarrow
```

Expected: 설치 완료 메시지 출력

**Step 3: models/ 디렉토리 생성**

```bash
mkdir -p models data scripts
touch models/.gitkeep data/.gitkeep
```

**Step 4: .gitignore에 모델/데이터 파일 추가**

기존 `.gitignore`에 추가:
```
models/*.pkl
data/*.parquet
```

**Step 5: Commit**

```bash
git add requirements.txt .gitignore models/.gitkeep data/.gitkeep
git commit -m "feat: add ML dependencies and directory structure"
```

---

## Task 2: 피처 엔지니어링 모듈 (`src/ml_features.py`)

**Files:**
- Create: `src/ml_features.py`
- Create: `tests/test_ml_features.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/test_ml_features.py
import pandas as pd
import numpy as np
import pytest
from src.ml_features import build_features, FEATURE_COLS

def make_df(n=100):
    """테스트용 최소 DataFrame 생성"""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.uniform(1000, 5000, n),
    })
    return df

def test_build_features_returns_series():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    assert isinstance(features, pd.Series)

def test_build_features_has_all_cols():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    for col in FEATURE_COLS:
        assert col in features.index, f"피처 누락: {col}"

def test_build_features_no_nan():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    features = build_features(df_ind, signal="LONG")
    assert not features.isna().any(), f"NaN 존재: {features[features.isna()]}"

def test_side_encoding():
    from src.indicators import Indicators
    df = make_df(100)
    ind = Indicators(df)
    df_ind = ind.calculate_all()
    long_feat  = build_features(df_ind, signal="LONG")
    short_feat = build_features(df_ind, signal="SHORT")
    assert long_feat["side"] == 1
    assert short_feat["side"] == 0
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_ml_features.py -v
```

Expected: FAIL with "cannot import name 'build_features'"

**Step 3: `src/ml_features.py` 구현**

```python
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
]


def build_features(df: pd.DataFrame, signal: str) -> pd.Series:
    """
    기술 지표가 계산된 DataFrame의 마지막 행에서 ML 피처를 추출한다.
    signal: "LONG" | "SHORT"
    """
    last = df.iloc[-1]
    close = last["close"]

    bb_upper = last.get("bb_upper", close)
    bb_lower = last.get("bb_lower", close)
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    ema9  = last.get("ema9",  close)
    ema21 = last.get("ema21", close)
    ema50 = last.get("ema50", close)
    if ema9 > ema21 > ema50:
        ema_align = 1
    elif ema9 < ema21 < ema50:
        ema_align = -1
    else:
        ema_align = 0

    atr = last.get("atr", 0)
    atr_pct = atr / close if close > 0 else 0

    vol_ma20 = last.get("vol_ma20", last.get("volume", 1))
    vol_ratio = last["volume"] / vol_ma20 if vol_ma20 > 0 else 1.0

    closes = df["close"]
    ret_1 = (close - closes.iloc[-2]) / closes.iloc[-2] if len(closes) >= 2 else 0.0
    ret_3 = (close - closes.iloc[-4]) / closes.iloc[-4] if len(closes) >= 4 else 0.0
    ret_5 = (close - closes.iloc[-6]) / closes.iloc[-6] if len(closes) >= 6 else 0.0

    # 규칙 기반 신호 강도 재계산 (indicators.py get_signal 로직 참조)
    prev = df.iloc[-2] if len(df) >= 2 else last
    strength = 0
    rsi = last.get("rsi", 50)
    macd = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    prev_macd = prev.get("macd", 0)
    prev_macd_sig = prev.get("macd_signal", 0)
    stoch_k = last.get("stoch_k", 50)
    stoch_d = last.get("stoch_d", 50)

    if signal == "LONG":
        if rsi < 35: strength += 1
        if prev_macd < prev_macd_sig and macd > macd_sig: strength += 2
        if close < last.get("bb_lower", close): strength += 1
        if ema_align == 1: strength += 1
        if stoch_k < 20 and stoch_k > stoch_d: strength += 1
    else:
        if rsi > 65: strength += 1
        if prev_macd > prev_macd_sig and macd < macd_sig: strength += 2
        if close > last.get("bb_upper", close): strength += 1
        if ema_align == -1: strength += 1
        if stoch_k > 80 and stoch_k < stoch_d: strength += 1

    return pd.Series({
        "rsi":            float(rsi),
        "macd_hist":      float(last.get("macd_hist", 0)),
        "bb_pct":         float(bb_pct),
        "ema_align":      float(ema_align),
        "stoch_k":        float(stoch_k),
        "stoch_d":        float(last.get("stoch_d", 50)),
        "atr_pct":        float(atr_pct),
        "vol_ratio":      float(vol_ratio),
        "ret_1":          float(ret_1),
        "ret_3":          float(ret_3),
        "ret_5":          float(ret_5),
        "signal_strength": float(strength),
        "side":           1.0 if signal == "LONG" else 0.0,
    })
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_ml_features.py -v
```

Expected: 4개 PASS

**Step 5: Commit**

```bash
git add src/ml_features.py tests/test_ml_features.py
git commit -m "feat: add ML feature engineering module"
```

---

## Task 3: 레이블 생성 유틸리티 (`src/label_builder.py`)

**Files:**
- Create: `src/label_builder.py`
- Create: `tests/test_label_builder.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/test_label_builder.py
import pandas as pd
import numpy as np
import pytest
from src.label_builder import build_labels


def make_signal_df():
    """
    신호 발생 시점 이후 가격이 TP에 도달하는 시나리오
    entry=100, TP=103, SL=98.5
    """
    # 신호 시점 이후 캔들: 점진적으로 상승해서 103 돌파
    future_closes = [100.5, 101.0, 101.8, 102.5, 103.1, 103.5]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    return future_closes, future_highs, future_lows


def test_label_tp_reached():
    closes, highs, lows = make_signal_df()
    label = build_labels(
        future_closes=closes,
        future_highs=highs,
        future_lows=lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label == 1, "TP 먼저 도달해야 레이블 1"


def test_label_sl_reached():
    # 하락해서 SL 먼저 도달
    future_closes = [99.5, 99.0, 98.8, 98.4, 98.0]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label == 0, "SL 먼저 도달해야 레이블 0"


def test_label_neither_reached_returns_none():
    # 아무것도 도달 못함
    future_closes = [100.1, 100.2, 100.3]
    future_highs  = [c + 0.1 for c in future_closes]
    future_lows   = [c - 0.1 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label is None, "미결 시 None 반환"


def test_label_short_tp():
    # SHORT: 가격 하락 → TP 도달
    future_closes = [99.5, 99.0, 98.0, 97.0]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=97.0,
        stop_loss=101.5,
        side="SHORT",
    )
    assert label == 1
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_label_builder.py -v
```

Expected: FAIL with "cannot import name 'build_labels'"

**Step 3: `src/label_builder.py` 구현**

```python
from typing import Optional


def build_labels(
    future_closes: list[float],
    future_highs: list[float],
    future_lows: list[float],
    take_profit: float,
    stop_loss: float,
    side: str,
) -> Optional[int]:
    """
    진입 이후 미래 캔들을 순서대로 확인해 TP/SL 도달 여부를 판단한다.
    LONG: high >= TP → 1, low <= SL → 0
    SHORT: low <= TP → 1, high >= SL → 0
    둘 다 미도달 → None (학습 데이터에서 제외)
    """
    for high, low in zip(future_highs, future_lows):
        if side == "LONG":
            if high >= take_profit:
                return 1
            if low <= stop_loss:
                return 0
        else:  # SHORT
            if low <= take_profit:
                return 1
            if high >= stop_loss:
                return 0
    return None
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_label_builder.py -v
```

Expected: 4개 PASS

**Step 5: Commit**

```bash
git add src/label_builder.py tests/test_label_builder.py
git commit -m "feat: add label builder for TP/SL simulation"
```

---

## Task 4: 과거 데이터 수집 스크립트 (`scripts/fetch_history.py`)

**Files:**
- Create: `scripts/fetch_history.py`
- Create: `scripts/__init__.py`

**Step 1: `scripts/fetch_history.py` 작성**

```python
"""
바이낸스 선물 REST API로 과거 캔들 데이터를 수집해 parquet으로 저장한다.
사용법: python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90
"""
import asyncio
import argparse
from datetime import datetime, timedelta
import pandas as pd
from binance import AsyncClient
from dotenv import load_dotenv
import os

load_dotenv()


async def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    try:
        start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_klines = []
        while True:
            klines = await client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            last_ts = klines[-1][0]
            if last_ts >= int(datetime.utcnow().timestamp() * 1000):
                break
            start_ts = last_ts + 1
            print(f"수집 중... {len(all_klines)}개")
    finally:
        await client.close_connection()

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="XRPUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days",     type=int, default=90)
    parser.add_argument("--output",   default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()

    df = asyncio.run(fetch_klines(args.symbol, args.interval, args.days))
    df.to_parquet(args.output)
    print(f"저장 완료: {args.output} ({len(df)}행)")


if __name__ == "__main__":
    main()
```

**Step 2: 실행 테스트 (실제 API 호출)**

```bash
python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90
```

Expected: `저장 완료: data/xrpusdt_1m.parquet (약 129600행)`

**Step 3: Commit**

```bash
git add scripts/fetch_history.py scripts/__init__.py
git commit -m "feat: add historical data fetcher script"
```

---

## Task 5: 모델 학습 스크립트 (`scripts/train_model.py`)

**Files:**
- Create: `scripts/train_model.py`

**Step 1: `scripts/train_model.py` 작성**

```python
"""
과거 캔들 데이터로 LightGBM 필터 모델을 학습하고 저장한다.
사용법: python scripts/train_model.py --data data/xrpusdt_1m.parquet
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from src.indicators import Indicators
from src.ml_features import build_features, FEATURE_COLS
from src.label_builder import build_labels

LOOKAHEAD = 60       # 최대 60캔들(1시간) 이내 TP/SL 도달 확인
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MODEL_PATH = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH = Path("models/training_log.json")


def generate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """신호 발생 시점마다 피처와 레이블을 생성한다."""
    rows = []
    total = len(df)

    for i in range(60, total - LOOKAHEAD):
        window = df.iloc[i - 60: i + 1].copy()
        ind = Indicators(window)
        df_ind = ind.calculate_all()

        if df_ind.isna().any().any():
            continue

        signal = ind.get_signal(df_ind)
        if signal == "HOLD":
            continue

        entry_price = float(df_ind["close"].iloc[-1])
        atr = float(df_ind["atr"].iloc[-1])
        if atr <= 0:
            continue

        stop_loss   = entry_price - atr * ATR_SL_MULT if signal == "LONG" else entry_price + atr * ATR_SL_MULT
        take_profit = entry_price + atr * ATR_TP_MULT if signal == "LONG" else entry_price - atr * ATR_TP_MULT

        future = df.iloc[i + 1: i + 1 + LOOKAHEAD]
        label = build_labels(
            future_closes=future["close"].tolist(),
            future_highs=future["high"].tolist(),
            future_lows=future["low"].tolist(),
            take_profit=take_profit,
            stop_loss=stop_loss,
            side=signal,
        )
        if label is None:
            continue

        features = build_features(df_ind, signal)
        row = features.to_dict()
        row["label"] = label
        rows.append(row)

        if len(rows) % 500 == 0:
            print(f"  샘플 생성 중: {len(rows)}개 (인덱스 {i}/{total})")

    return pd.DataFrame(rows)


def train(data_path: str):
    print(f"데이터 로드: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df)}")

    print("데이터셋 생성 중...")
    dataset = generate_dataset(df)
    print(f"학습 샘플: {len(dataset)}개 (양성={dataset['label'].sum():.0f}, 음성={(dataset['label']==0).sum():.0f})")

    if len(dataset) < 200:
        raise ValueError(f"학습 샘플 부족: {len(dataset)}개 (최소 200 필요)")

    X = dataset[FEATURE_COLS]
    y = dataset["label"]

    # 시계열 분할: 앞 80% 학습, 뒤 20% 검증
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    print(f"\n검증 AUC: {auc:.4f}")
    print(classification_report(y_val, (val_proba >= 0.60).astype(int)))

    # 기존 모델이 있으면 백업
    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, PREV_MODEL_PATH)
        print(f"기존 모델 백업: {PREV_MODEL_PATH}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    # 학습 이력 기록
    log = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append({
        "date": datetime.now().isoformat(),
        "auc": round(auc, 4),
        "samples": len(dataset),
        "model_path": str(MODEL_PATH),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
```

**Step 2: 학습 실행 테스트**

```bash
python scripts/train_model.py --data data/xrpusdt_1m.parquet
```

Expected: 학습 완료 후 `models/lgbm_filter.pkl` 생성, AUC 출력

**Step 3: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: add LightGBM training script with TP/SL label generation"
```

---

## Task 6: ML 필터 클래스 (`src/ml_filter.py`)

**Files:**
- Create: `src/ml_filter.py`
- Create: `tests/test_ml_filter.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/test_ml_filter.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.ml_filter import MLFilter
from src.ml_features import FEATURE_COLS


def make_features(side="LONG") -> pd.Series:
    return pd.Series({col: 0.5 for col in FEATURE_COLS} | {"side": 1.0 if side == "LONG" else 0.0})


def test_no_model_file_is_not_loaded(tmp_path):
    f = MLFilter(model_path=str(tmp_path / "nonexistent.pkl"))
    assert not f.is_model_loaded()


def test_no_model_should_enter_returns_true(tmp_path):
    """모델 없으면 항상 진입 허용 (폴백)"""
    f = MLFilter(model_path=str(tmp_path / "nonexistent.pkl"))
    features = make_features()
    assert f.should_enter(features) is True


def test_should_enter_above_threshold():
    """확률 >= 0.60 이면 True"""
    f = MLFilter(threshold=0.60)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
    f._model = mock_model
    features = make_features()
    assert f.should_enter(features) is True


def test_should_enter_below_threshold():
    """확률 < 0.60 이면 False"""
    f = MLFilter(threshold=0.60)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.55, 0.45]])
    f._model = mock_model
    features = make_features()
    assert f.should_enter(features) is False


def test_reload_model(tmp_path):
    """reload_model 호출 후 모델 로드 상태 변경"""
    import joblib
    import lightgbm as lgb
    # 더미 모델 저장
    dummy = MagicMock()
    model_path = tmp_path / "lgbm_filter.pkl"
    joblib.dump(dummy, model_path)
    f = MLFilter(model_path=str(model_path))
    f.reload_model()
    assert f.is_model_loaded()
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_ml_filter.py -v
```

Expected: FAIL with "cannot import name 'MLFilter'"

**Step 3: `src/ml_filter.py` 구현**

```python
from pathlib import Path
import joblib
import pandas as pd
from loguru import logger


class MLFilter:
    """
    LightGBM 모델을 로드하고 진입 여부를 판단한다.
    모델 파일이 없으면 항상 진입을 허용한다 (폴백).
    """

    def __init__(self, model_path: str = "models/lgbm_filter.pkl", threshold: float = 0.60):
        self._model_path = Path(model_path)
        self._threshold = threshold
        self._model = None
        self._try_load()

    def _try_load(self):
        if self._model_path.exists():
            try:
                self._model = joblib.load(self._model_path)
                logger.info(f"ML 필터 모델 로드 완료: {self._model_path}")
            except Exception as e:
                logger.warning(f"ML 필터 모델 로드 실패: {e}")
                self._model = None

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def should_enter(self, features: pd.Series) -> bool:
        """
        확률 >= threshold 이면 True (진입 허용).
        모델 없으면 True 반환 (폴백).
        """
        if not self.is_model_loaded():
            return True
        try:
            X = features.to_frame().T
            proba = self._model.predict_proba(X)[0][1]
            logger.debug(f"ML 필터 확률: {proba:.3f} (임계값: {self._threshold})")
            return proba >= self._threshold
        except Exception as e:
            logger.warning(f"ML 필터 예측 오류 (폴백 허용): {e}")
            return True

    def reload_model(self):
        """재학습 후 모델을 핫 리로드한다."""
        self._try_load()
        logger.info("ML 필터 모델 리로드 완료")
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_ml_filter.py -v
```

Expected: 5개 PASS

**Step 5: Commit**

```bash
git add src/ml_filter.py tests/test_ml_filter.py
git commit -m "feat: add MLFilter class with fallback support"
```

---

## Task 7: 자동 재학습 스케줄러 (`src/retrainer.py`)

**Files:**
- Create: `src/retrainer.py`
- Create: `tests/test_retrainer.py`

**Step 1: 실패하는 테스트 작성**

```python
# tests/test_retrainer.py
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.retrainer import Retrainer


@pytest.mark.asyncio
async def test_retrain_calls_train(tmp_path):
    """재학습 시 train 함수가 호출되는지 확인"""
    ml_filter = MagicMock()
    r = Retrainer(ml_filter=ml_filter, data_path=str(tmp_path / "data.parquet"))

    with patch("src.retrainer.fetch_and_save", new_callable=AsyncMock) as mock_fetch, \
         patch("src.retrainer.run_training", return_value=0.72) as mock_train, \
         patch("src.retrainer.get_current_auc", return_value=0.65):
        await r.retrain()

    mock_fetch.assert_called_once()
    mock_train.assert_called_once()


@pytest.mark.asyncio
async def test_retrain_rollback_when_worse(tmp_path):
    """새 모델이 기존보다 나쁘면 롤백"""
    ml_filter = MagicMock()
    r = Retrainer(ml_filter=ml_filter, data_path=str(tmp_path / "data.parquet"))

    with patch("src.retrainer.fetch_and_save", new_callable=AsyncMock), \
         patch("src.retrainer.run_training", return_value=0.55), \
         patch("src.retrainer.get_current_auc", return_value=0.70), \
         patch("src.retrainer.rollback_model") as mock_rollback:
        await r.retrain()

    mock_rollback.assert_called_once()
```

**Step 2: 테스트 실패 확인**

```bash
pytest tests/test_retrainer.py -v
```

Expected: FAIL with "cannot import name 'Retrainer'"

**Step 3: `src/retrainer.py` 구현**

```python
import asyncio
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.ml_filter import MLFilter

MODEL_PATH      = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH        = Path("models/training_log.json")


def get_current_auc() -> float:
    """training_log.json에서 가장 최근 AUC를 읽는다."""
    if not LOG_PATH.exists():
        return 0.0
    with open(LOG_PATH) as f:
        log = json.load(f)
    return log[-1]["auc"] if log else 0.0


def rollback_model():
    """이전 모델로 롤백한다."""
    if PREV_MODEL_PATH.exists():
        import shutil
        shutil.copy(PREV_MODEL_PATH, MODEL_PATH)
        logger.warning("ML 모델 롤백 완료")
    else:
        logger.warning("롤백할 이전 모델 없음")


async def fetch_and_save(data_path: str):
    """증분 데이터 수집 (fetch_history.py 로직 재사용)."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/fetch_history.py", "--output", data_path, "--days", "90"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"데이터 수집 실패: {result.stderr}")
    logger.info(f"데이터 수집 완료: {data_path}")


def run_training(data_path: str) -> float:
    """train_model.py를 실행하고 새 AUC를 반환한다."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/train_model.py", "--data", data_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"학습 실패: {result.stderr}")
    new_auc = get_current_auc()
    return new_auc


class Retrainer:
    def __init__(self, ml_filter: MLFilter, data_path: str = "data/xrpusdt_1m.parquet"):
        self._ml_filter = ml_filter
        self._data_path = data_path

    async def retrain(self):
        logger.info("자동 재학습 시작")
        old_auc = get_current_auc()
        try:
            await fetch_and_save(self._data_path)
            new_auc = run_training(self._data_path)
            logger.info(f"재학습 완료: 이전 AUC={old_auc:.4f} → 새 AUC={new_auc:.4f}")

            if new_auc < old_auc - 0.01:
                logger.warning(f"새 모델 성능 저하 ({new_auc:.4f} < {old_auc:.4f}), 롤백")
                rollback_model()
            else:
                self._ml_filter.reload_model()
                logger.success("새 ML 모델 적용 완료")
        except Exception as e:
            logger.error(f"재학습 실패: {e}")

    async def schedule_daily(self, hour: int = 3):
        """매일 지정 시각(UTC 기준)에 재학습을 실행한다."""
        while True:
            now = datetime.utcnow()
            next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                from datetime import timedelta
                next_run += timedelta(days=1)
            wait_secs = (next_run - now).total_seconds()
            logger.info(f"다음 재학습까지 {wait_secs/3600:.1f}시간 대기")
            await asyncio.sleep(wait_secs)
            await self.retrain()
```

**Step 4: 테스트 통과 확인**

```bash
pytest tests/test_retrainer.py -v
```

Expected: 2개 PASS

**Step 5: Commit**

```bash
git add src/retrainer.py tests/test_retrainer.py
git commit -m "feat: add daily retrainer with rollback support"
```

---

## Task 8: bot.py에 ML 필터 통합

**Files:**
- Modify: `src/bot.py:1-10` (import 추가)
- Modify: `src/bot.py:11-22` (`__init__` 수정)
- Modify: `src/bot.py:47-65` (`process_candle` 수정)
- Modify: `src/bot.py:153-160` (`run` 수정)

**Step 1: `src/bot.py` import 추가**

기존 import 블록 끝에 추가:
```python
from src.ml_filter import MLFilter
from src.ml_features import build_features
from src.retrainer import Retrainer
```

**Step 2: `__init__`에 MLFilter, Retrainer 추가**

```python
def __init__(self, config: Config):
    self.config = config
    self.exchange = BinanceFuturesClient(config)
    self.notifier = DiscordNotifier(config.discord_webhook_url)
    self.risk = RiskManager(config)
    self.ml_filter = MLFilter()                          # 추가
    self.retrainer = Retrainer(ml_filter=self.ml_filter) # 추가
    self.current_trade_side: str | None = None
    self.stream = KlineStream(
        symbol=config.symbol,
        interval="1m",
        on_candle=self._on_candle_closed,
    )
```

**Step 3: `process_candle`에 ML 필터 적용**

`signal = ind.get_signal(df_with_indicators)` 바로 아래에 추가:
```python
        # ML 필터: 모델이 있을 때만 적용, 없으면 폴백(통과)
        if signal != "HOLD" and self.ml_filter.is_model_loaded():
            features = build_features(df_with_indicators, signal)
            if not self.ml_filter.should_enter(features):
                logger.info(f"ML 필터 차단: {signal} 신호 무시")
                signal = "HOLD"
```

**Step 4: `run`에 재학습 스케줄러 추가**

```python
    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        asyncio.create_task(self.retrainer.schedule_daily(hour=3))  # 추가
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
```

**Step 5: 기존 bot 테스트 통과 확인**

```bash
pytest tests/test_bot.py -v
```

Expected: 기존 테스트 모두 PASS (ML 필터는 모델 없으면 폴백)

**Step 6: Commit**

```bash
git add src/bot.py
git commit -m "feat: integrate ML filter into trading bot with fallback"
```

---

## Task 9: 전체 테스트 실행 및 검증

**Step 1: 전체 테스트 실행**

```bash
pytest tests/ -v --tb=short
```

Expected: 모든 테스트 PASS

**Step 2: 린트 확인**

```bash
python -m py_compile src/ml_features.py src/ml_filter.py src/label_builder.py src/retrainer.py scripts/train_model.py scripts/fetch_history.py
```

Expected: 오류 없음

**Step 3: 초기 학습 실행 (실제 데이터)**

```bash
python scripts/fetch_history.py --days 90
python scripts/train_model.py
```

Expected: `models/lgbm_filter.pkl` 생성, AUC 출력

**Step 4: 봇 시작 후 ML 필터 로그 확인**

```bash
python main.py
```

Expected: 로그에 `ML 필터 모델 로드 완료` 메시지 출력

**Step 5: Final Commit**

```bash
git add -A
git commit -m "feat: complete ML filter integration with LightGBM"
```

---

## 파일 구조 최종 요약

```
cointrader/
├── src/
│   ├── ml_features.py     ← 피처 엔지니어링 (신규)
│   ├── ml_filter.py       ← LightGBM 필터 클래스 (신규)
│   ├── label_builder.py   ← TP/SL 레이블 생성 (신규)
│   ├── retrainer.py       ← 자동 재학습 스케줄러 (신규)
│   └── bot.py             ← ML 필터 통합 (수정)
├── scripts/
│   ├── fetch_history.py   ← 과거 데이터 수집 (신규)
│   └── train_model.py     ← LightGBM 학습 (신규)
├── tests/
│   ├── test_ml_features.py  (신규)
│   ├── test_ml_filter.py    (신규)
│   ├── test_label_builder.py (신규)
│   └── test_retrainer.py    (신규)
├── models/
│   ├── lgbm_filter.pkl      ← 현재 모델 (학습 후 생성)
│   ├── lgbm_filter_prev.pkl ← 롤백용
│   └── training_log.json    ← 재학습 이력
└── data/
    └── xrpusdt_1m.parquet   ← 과거 캔들 데이터
```
