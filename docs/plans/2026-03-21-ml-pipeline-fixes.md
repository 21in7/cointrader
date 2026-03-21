# ML Pipeline Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ML 파이프라인의 학습-서빙 불일치(SL/TP 배수, 언더샘플링, 정규화)와 백테스트 정확도 이슈를 수정하여 모델 평가 체계와 실전 환경을 일치시킨다.

**Architecture:** `dataset_builder.py`의 하드코딩 SL/TP 상수를 파라미터화하고, 모든 호출부(train_model, train_mlx_model, tune_hyperparams, backtester)가 동일한 값을 주입하도록 변경. MLX 학습의 이중 정규화 제거. 백테스터의 에퀴티 커브에 미실현 PnL 반영. MLFilter에 factory method 추가.

**Tech Stack:** Python, LightGBM, MLX, pandas, numpy, pytest

---

## File Structure

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `src/dataset_builder.py` | Modify | SL/TP 상수 → 파라미터화 |
| `src/ml_filter.py` | Modify | `from_model()` factory method 추가 |
| `src/mlx_filter.py` | Modify | fit()에 `normalize` 파라미터 추가 |
| `src/backtester.py` | Modify | 에퀴티 미실현 PnL, MLFilter factory, initial_balance |
| `src/backtest_validator.py` | Modify | initial_balance 하드코딩 제거 |
| `scripts/train_model.py` | Modify | 레거시 상수 제거, SL/TP 전달 |
| `scripts/train_mlx_model.py` | Modify | 이중 정규화 제거, stratified_undersample 적용 |
| `scripts/tune_hyperparams.py` | Modify | SL/TP 전달 |
| `tests/test_dataset_builder.py` | Modify | SL/TP 파라미터 테스트 추가 |
| `tests/test_ml_pipeline_fixes.py` | Create | 신규 수정사항 전용 테스트 |

---

### Task 1: SL/TP 배수 파라미터화 — dataset_builder.py

**Files:**
- Modify: `src/dataset_builder.py:14-16, 322-383, 385-494`
- Test: `tests/test_dataset_builder.py`

- [ ] **Step 1: 기존 테스트 통과 확인**

Run: `bash scripts/run_tests.sh -k "dataset_builder"`
Expected: 모든 테스트 PASS

- [ ] **Step 2: 파라미터화 테스트 작성**

`tests/test_ml_pipeline_fixes.py`에 추가:

```python
import numpy as np
import pandas as pd
import pytest
from src.dataset_builder import generate_dataset_vectorized, _calc_labels_vectorized


@pytest.fixture
def signal_df():
    """시그널이 발생하는 데이터."""
    rng = np.random.default_rng(7)
    n = 800
    trend = np.linspace(1.5, 3.0, n)
    noise = np.cumsum(rng.normal(0, 0.04, n))
    close = np.clip(trend + noise, 0.01, None)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    volume = rng.uniform(1e6, 3e6, n)
    volume[::30] *= 3.0
    return pd.DataFrame({
        "open": close, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_sltp_params_are_passed_through(signal_df):
    """SL/TP 배수가 generate_dataset_vectorized에 전달되어야 한다."""
    r1 = generate_dataset_vectorized(
        signal_df, atr_sl_mult=1.5, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    r2 = generate_dataset_vectorized(
        signal_df, atr_sl_mult=2.0, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    # SL이 다르면 레이블 분포가 달라져야 한다
    if len(r1) > 0 and len(r2) > 0:
        # 정확히 같은 분포일 확률은 매우 낮음
        assert not (r1["label"].values == r2["label"].values).all() or len(r1) != len(r2), \
            "SL 배수가 다르면 레이블이 달라져야 한다"


def test_default_sltp_backward_compatible(signal_df):
    """SL/TP 파라미터 미지정 시 기존 기본값(1.5, 2.0)으로 동작해야 한다."""
    r_default = generate_dataset_vectorized(
        signal_df, adx_threshold=0, volume_multiplier=1.5,
    )
    r_explicit = generate_dataset_vectorized(
        signal_df, atr_sl_mult=1.5, atr_tp_mult=2.0,
        adx_threshold=0, volume_multiplier=1.5,
    )
    if len(r_default) > 0:
        assert len(r_default) == len(r_explicit)
        assert (r_default["label"].values == r_explicit["label"].values).all()
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py -v`
Expected: FAIL — `generate_dataset_vectorized() got an unexpected keyword argument 'atr_sl_mult'`

- [ ] **Step 4: dataset_builder.py 수정**

`src/dataset_builder.py` 변경:

1. 모듈 상수 `ATR_SL_MULT`, `ATR_TP_MULT`는 기본값으로 유지 (하위 호환)
2. `_calc_labels_vectorized`에 `atr_sl_mult`, `atr_tp_mult` 파라미터 추가
3. `generate_dataset_vectorized`에 `atr_sl_mult`, `atr_tp_mult` 파라미터 추가하여 `_calc_labels_vectorized`에 전달

```python
# _calc_labels_vectorized 시그니처 변경:
def _calc_labels_vectorized(
    d: pd.DataFrame,
    feat: pd.DataFrame,
    sig_idx: np.ndarray,
    atr_sl_mult: float = ATR_SL_MULT,
    atr_tp_mult: float = ATR_TP_MULT,
) -> tuple[np.ndarray, np.ndarray]:

# 함수 본문 (lines 350-355) 변경:
#   변경 전:
#       sl = entry - atr * ATR_SL_MULT
#       tp = entry + atr * ATR_TP_MULT
#   변경 후:
        if signal == "LONG":
            sl = entry - atr * atr_sl_mult
            tp = entry + atr * atr_tp_mult
        else:
            sl = entry + atr * atr_sl_mult
            tp = entry - atr * atr_tp_mult

# generate_dataset_vectorized 시그니처 변경:
def generate_dataset_vectorized(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    time_weight_decay: float = 0.0,
    negative_ratio: int = 0,
    signal_threshold: int = 3,
    adx_threshold: float = 25,
    volume_multiplier: float = 2.5,
    atr_sl_mult: float = ATR_SL_MULT,   # 추가
    atr_tp_mult: float = ATR_TP_MULT,   # 추가
) -> pd.DataFrame:

# _calc_labels_vectorized 호출 시 전달:
#   labels, valid_mask = _calc_labels_vectorized(
#       d, feat_all, sig_idx,
#       atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
#   )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py tests/test_dataset_builder.py -v`
Expected: ALL PASS

- [ ] **Step 6: 커밋**

```bash
git add src/dataset_builder.py tests/test_ml_pipeline_fixes.py
git commit -m "feat(ml): parameterize SL/TP multipliers in dataset_builder"
```

---

### Task 2: 호출부 SL/TP 전달 — train_model, train_mlx_model, tune_hyperparams, backtester

**Files:**
- Modify: `scripts/train_model.py:57-58, 217-221, 358-362, 448-452`
- Modify: `scripts/train_mlx_model.py:61, 179`
- Modify: `scripts/tune_hyperparams.py:67`
- Modify: `src/backtester.py:739-746`

- [ ] **Step 1: train_model.py 수정**

1. 레거시 모듈 상수 `ATR_SL_MULT=1.5`, `ATR_TP_MULT=3.0` (line 57-58)을 삭제
2. `main()`의 argparse에 `--sl-mult` (기본 2.0), `--tp-mult` (기본 2.0) CLI 인자 추가
3. `train()`, `walk_forward_auc()`, `compare()` 함수에 `atr_sl_mult`, `atr_tp_mult` 파라미터 추가하여 `generate_dataset_vectorized`에 전달

```python
# argparse에 추가:
parser.add_argument("--sl-mult", type=float, default=2.0, help="SL ATR 배수 (기본 2.0)")
parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR 배수 (기본 2.0)")

# train() 시그니처:
def train(data_path, time_weight_decay=2.0, tuned_params_path=None,
          atr_sl_mult=2.0, atr_tp_mult=2.0):

# train() 내:
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=time_weight_decay,
    negative_ratio=5,
    atr_sl_mult=atr_sl_mult,
    atr_tp_mult=atr_tp_mult,
)

# main()에서 호출:
train(args.data, ..., atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult)
```

- [ ] **Step 2: train_mlx_model.py 수정**

동일하게 `--sl-mult`, `--tp-mult` CLI 인자 추가. `train_mlx()`, `walk_forward_auc()` 함수에 파라미터 전달.

- [ ] **Step 3: tune_hyperparams.py 수정**

`--sl-mult`, `--tp-mult` CLI 인자 추가. `load_dataset()` 함수에 파라미터 전달.

- [ ] **Step 4: backtester.py WalkForward 수정**

`WalkForwardBacktester._train_model()` (line 739-746)에서 `generate_dataset_vectorized` 호출 시 `self.cfg.atr_sl_mult`, `self.cfg.atr_tp_mult` 전달:

```python
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=self.cfg.time_weight_decay,
    negative_ratio=self.cfg.negative_ratio,
    signal_threshold=self.cfg.signal_threshold,
    adx_threshold=self.cfg.adx_threshold,
    volume_multiplier=self.cfg.volume_multiplier,
    atr_sl_mult=self.cfg.atr_sl_mult,
    atr_tp_mult=self.cfg.atr_tp_mult,
)
```

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 6: 커밋**

```bash
git add scripts/train_model.py scripts/train_mlx_model.py scripts/tune_hyperparams.py src/backtester.py
git commit -m "fix(ml): pass SL/TP multipliers to dataset generation — align train/serve"
```

---

### Task 3: 백테스터 에퀴티 커브 미실현 PnL 반영

**Files:**
- Modify: `src/backtester.py:571-578`
- Test: `tests/test_ml_pipeline_fixes.py`

- [ ] **Step 1: 테스트 작성**

```python
def test_equity_curve_includes_unrealized_pnl():
    """에퀴티 커브에 미실현 PnL이 반영되어야 한다."""
    from src.backtester import Backtester, BacktestConfig, Position
    import pandas as pd

    cfg = BacktestConfig(symbols=["TEST"], initial_balance=1000.0)
    bt = Backtester.__new__(Backtester)
    bt.cfg = cfg
    bt.balance = 1000.0
    bt._peak_equity = 1000.0
    bt.equity_curve = []

    # LONG 포지션: 진입가 100, 현재가는 candle row로 전달
    bt.positions = {"TEST": Position(
        symbol="TEST", side="LONG", entry_price=100.0,
        quantity=10.0, sl=95.0, tp=110.0,
        entry_time=pd.Timestamp("2026-01-01"), entry_fee=0.4,
    )}

    # candle row에 close=105 → 미실현 PnL = (105-100)*10 = 50
    row = pd.Series({"close": 105.0})
    bt._record_equity(pd.Timestamp("2026-01-01 00:15:00"), current_prices={"TEST": 105.0})

    last = bt.equity_curve[-1]
    assert last["equity"] == 1050.0, f"Expected 1050.0 (1000+50), got {last['equity']}"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py::test_equity_curve_includes_unrealized_pnl -v`
Expected: FAIL

- [ ] **Step 3: _record_equity 수정**

`src/backtester.py`의 `_record_equity` 메서드를 수정:

```python
def _record_equity(self, ts: pd.Timestamp, current_prices: dict[str, float] | None = None):
    unrealized = 0.0
    for sym, pos in self.positions.items():
        price = (current_prices or {}).get(sym)
        if price is not None:
            if pos.side == "LONG":
                unrealized += (price - pos.entry_price) * pos.quantity
            else:
                unrealized += (pos.entry_price - price) * pos.quantity
    equity = self.balance + unrealized
    self.equity_curve.append({"timestamp": str(ts), "equity": round(equity, 4)})
    if equity > self._peak_equity:
        self._peak_equity = equity
```

메인 루프 호출부(`run()` 내 `_record_equity` 호출)도 수정:

```python
# run() 메인 루프 내:
current_prices = {}
for sym in self.cfg.symbols:
    idx = ... # 현재 캔들 인덱스
    current_prices[sym] = float(all_indicators[sym].iloc[...]["close"])
self._record_equity(ts, current_prices=current_prices)
```

메인 루프의 이벤트는 `(ts, sym, candle_idx)` 튜플로, 타임스탬프별로 정렬되어 있다 (line 426: `events.sort(key=lambda x: (x[0], x[1]))`). 같은 타임스탬프에 여러 심볼 이벤트가 올 수 있다.

구현: 이벤트 루프 직전에 `latest_prices: dict[str, float] = {}` 초기화. 각 이벤트에서 `latest_prices[sym] = float(row["close"])` 업데이트. `_record_equity`는 **매 이벤트마다** 호출 (현재 동작 유지). `latest_prices`는 점진적으로 축적되므로, 첫 번째 심볼 이벤트 시점에 다른 심볼은 이전 캔들의 가격이 사용된다. 이는 15분봉 기반에서 미미한 차이이며, 타임스탬프 그룹핑을 도입하면 코드 복잡도가 불필요하게 증가한다.

```python
# run() 메인 루프 변경:
latest_prices: dict[str, float] = {}

for ts, sym, candle_idx in events:
    # ... 기존 로직
    row = df_ind.iloc[candle_idx]
    latest_prices[sym] = float(row["close"])

    self._record_equity(ts, current_prices=latest_prices)
    # ... 나머지 기존 로직 (SL/TP 체크, 진입 등)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py -v`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add src/backtester.py tests/test_ml_pipeline_fixes.py
git commit -m "fix(backtest): include unrealized PnL in equity curve for accurate MDD"
```

---

### Task 4: MLX 이중 정규화 제거

**Files:**
- Modify: `src/mlx_filter.py:139-155`
- Modify: `scripts/train_mlx_model.py:218-240`
- Test: `tests/test_ml_pipeline_fixes.py`

- [ ] **Step 1: 테스트 작성**

```python
def test_mlx_no_double_normalization():
    """MLXFilter.fit()에 normalize=False를 전달하면 내부 정규화를 건너뛰어야 한다."""
    import numpy as np
    import pandas as pd
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    n_features = len(FEATURE_COLS)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((100, n_features)).astype(np.float32),
        columns=FEATURE_COLS,
    )
    y = pd.Series(rng.integers(0, 2, 100).astype(np.float32))

    model = MLXFilter(input_dim=n_features, hidden_dim=16, epochs=1, batch_size=32)
    model.fit(X, y, normalize=False)

    # normalize=False면 _mean=0, _std=1이어야 한다
    assert np.allclose(model._mean, 0.0), "normalize=False시 mean은 0이어야 한다"
    assert np.allclose(model._std, 1.0, atol=1e-7), "normalize=False시 std는 1이어야 한다"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py::test_mlx_no_double_normalization -v`
Expected: FAIL — `fit() got an unexpected keyword argument 'normalize'`

- [ ] **Step 3: mlx_filter.py 수정**

`MLXFilter.fit()` 시그니처에 `normalize: bool = True` 추가:

```python
def fit(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
    normalize: bool = True,
) -> "MLXFilter":
    X_np = X[FEATURE_COLS].values.astype(np.float32)
    y_np = y.values.astype(np.float32)

    if normalize:
        mean_vals = np.nanmean(X_np, axis=0)
        self._mean = np.nan_to_num(mean_vals, nan=0.0)
        std_vals = np.nanstd(X_np, axis=0)
        self._std = np.nan_to_num(std_vals, nan=1.0) + 1e-8
        X_np = (X_np - self._mean) / self._std
        X_np = np.nan_to_num(X_np, nan=0.0)
    else:
        self._mean = np.zeros(X_np.shape[1], dtype=np.float32)
        self._std = np.ones(X_np.shape[1], dtype=np.float32)
        X_np = np.nan_to_num(X_np, nan=0.0)
    # ... 나머지 동일
```

- [ ] **Step 4: train_mlx_model.py walk-forward 수정**

`walk_forward_auc()` (line 218-240)에서 이중 정규화 해킹을 제거:

```python
# 변경 전 (해킹):
#   mean = X_tr_bal.mean(axis=0)
#   std = X_tr_bal.std(axis=0) + 1e-8
#   X_tr_norm = (X_tr_bal - mean) / std
#   X_val_norm = (X_val_raw - mean) / std
#   ...
#   model.fit(X_tr_df, pd.Series(y_tr_bal), sample_weight=w_tr_bal)
#   model._mean = np.zeros(...)
#   model._std = np.ones(...)

# 변경 후 (깔끔):
X_tr_df = pd.DataFrame(X_tr_bal, columns=FEATURE_COLS)
X_val_df = pd.DataFrame(X_val_raw, columns=FEATURE_COLS)

model = MLXFilter(...)
model.fit(X_tr_df, pd.Series(y_tr_bal), sample_weight=w_tr_bal)
# fit() 내부에서 학습 데이터 기준으로 정규화
# predict_proba()에서 동일한 mean/std 적용

proba = model.predict_proba(X_val_df)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py -v`
Expected: ALL PASS

- [ ] **Step 6: 커밋**

```bash
git add src/mlx_filter.py scripts/train_mlx_model.py tests/test_ml_pipeline_fixes.py
git commit -m "fix(mlx): remove double normalization in walk-forward validation"
```

---

### Task 5: MLX에 stratified_undersample 적용

**Files:**
- Modify: `scripts/train_mlx_model.py:88-104, 207-212`

- [ ] **Step 1: train_mlx_model.py train 함수 수정**

`train_mlx()` (line 88-104)의 단순 언더샘플링을 `stratified_undersample`로 교체:

```python
# 변경 전:
#   pos_idx = np.where(y_train == 1)[0]
#   neg_idx = np.where(y_train == 0)[0]
#   if len(neg_idx) > len(pos_idx):
#       np.random.seed(42)
#       neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
#   balanced_idx = np.concatenate([pos_idx, neg_idx])
#   np.random.shuffle(balanced_idx)

# 변경 후:
from src.dataset_builder import stratified_undersample

source = dataset["source"].values if "source" in dataset.columns else np.full(len(dataset), "signal")
source_train = source[:split]
balanced_idx = stratified_undersample(y_train.values, source_train, seed=42)
```

- [ ] **Step 2: walk_forward_auc도 동일하게 수정**

`walk_forward_auc()` (line 207-212)도 `stratified_undersample`로 교체.

- [ ] **Step 3: negative_ratio 파라미터 추가**

`train_mlx()` 및 `walk_forward_auc()` 내 `generate_dataset_vectorized` 호출 모두에 `negative_ratio=5` 추가 (LightGBM과 동일):

```python
# train_mlx() 내:
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=time_weight_decay,
    negative_ratio=5,
    atr_sl_mult=2.0,
    atr_tp_mult=2.0,
)

# walk_forward_auc() 내 (line 179-181):
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=time_weight_decay,
    negative_ratio=5,
    atr_sl_mult=2.0,
    atr_tp_mult=2.0,
)
```

- [ ] **Step 4: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/train_mlx_model.py
git commit -m "fix(mlx): use stratified_undersample consistent with LightGBM"
```

---

### Task 6: MLFilter factory method + backtest_validator initial_balance

**Files:**
- Modify: `src/ml_filter.py`
- Modify: `src/backtester.py:320-329`
- Modify: `src/backtest_validator.py:123`
- Test: `tests/test_ml_pipeline_fixes.py`

- [ ] **Step 1: MLFilter factory method 테스트**

```python
def test_ml_filter_from_model():
    """MLFilter.from_model()로 LightGBM 모델을 주입할 수 있어야 한다."""
    from src.ml_filter import MLFilter
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]

    mf = MLFilter.from_model(mock_model, threshold=0.55)
    assert mf.is_model_loaded()
    assert mf.active_backend == "LightGBM"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py::test_ml_filter_from_model -v`
Expected: FAIL — `MLFilter has no attribute 'from_model'`

- [ ] **Step 3: ml_filter.py에 from_model 추가**

```python
@classmethod
def from_model(cls, model, threshold: float = 0.55) -> "MLFilter":
    """외부에서 학습된 LightGBM 모델을 주입하여 MLFilter를 생성한다.
    backtester walk-forward에서 사용."""
    instance = cls.__new__(cls)
    instance._disabled = False
    instance._onnx_session = None
    instance._lgbm_model = model
    instance._threshold = threshold
    instance._onnx_path = Path("/dev/null")
    instance._lgbm_path = Path("/dev/null")
    instance._loaded_onnx_mtime = 0.0
    instance._loaded_lgbm_mtime = 0.0
    return instance
```

- [ ] **Step 4: backtester.py에서 factory method 사용**

`backtester.py:320-329`의 직접 조작 코드를 교체:

```python
# 변경 전:
#   mf = MLFilter.__new__(MLFilter)
#   mf._disabled = False
#   mf._onnx_session = None
#   mf._lgbm_model = ml_models[sym]
#   ...

# 변경 후:
mf = MLFilter.from_model(ml_models[sym], threshold=self.cfg.ml_threshold)
self.ml_filters[sym] = mf
```

- [ ] **Step 5: backtest_validator.py initial_balance 수정**

`src/backtest_validator.py:123`:

```python
# 변경 전:
#   balance = 1000.0

# 변경 후 (cfg는 항상 BacktestConfig이므로 hasattr 불필요):
balance = cfg.initial_balance
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py -v && bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 7: 커밋**

```bash
git add src/ml_filter.py src/backtester.py src/backtest_validator.py tests/test_ml_pipeline_fixes.py
git commit -m "refactor(ml): add MLFilter.from_model(), fix validator initial_balance"
```

---

### Task 7: 레거시 코드 정리 + 최종 검증

**Files:**
- Modify: `scripts/train_model.py:56-103` (레거시 `_process_index`, `generate_dataset` 함수)
- Modify: `tests/test_dataset_builder.py:76-93` (레거시 비교 테스트)

- [ ] **Step 1: 레거시 함수 사용 여부 확인**

`scripts/train_model.py`의 `_process_index()`, `generate_dataset()` 함수는 현재 `tests/test_dataset_builder.py:84`에서만 참조됨. 이 테스트는 레거시와 벡터화 버전의 샘플 수 비교인데, 두 버전의 SL/TP가 다르므로 (레거시 TP=3.0 vs 벡터화 TP=2.0) 비교 자체가 무의미.

- [ ] **Step 2: 레거시 비교 테스트 제거**

`tests/test_dataset_builder.py`에서 `test_matches_original_generate_dataset` 함수를 삭제.

- [ ] **Step 3: 레거시 함수에 deprecation 경고 추가**

`scripts/train_model.py`의 `generate_dataset()`, `_process_index()` 함수 상단에:

```python
import warnings

def generate_dataset(df: pd.DataFrame, n_jobs: int | None = None) -> pd.DataFrame:
    """[Deprecated] generate_dataset_vectorized()를 사용할 것."""
    warnings.warn(
        "generate_dataset()는 deprecated. generate_dataset_vectorized()를 사용하세요.",
        DeprecationWarning, stacklevel=2,
    )
    # ... 기존 코드
```

- [ ] **Step 4: 전체 테스트 실행**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/train_model.py tests/test_dataset_builder.py
git commit -m "chore: deprecate legacy dataset generation, remove stale comparison test"
```

---

### Task 8: README/ARCHITECTURE 동기화 + CLAUDE.md 업데이트

**Files:**
- Modify: `CLAUDE.md` (plan history table)
- Modify: `README.md` (필요시)
- Modify: `ARCHITECTURE.md` (필요시)

- [ ] **Step 1: CLAUDE.md plan history 업데이트**

`CLAUDE.md`의 plan history 테이블에 추가:

```markdown
| 2026-03-21 | `ml-pipeline-fixes` (plan) | Completed |
```

- [ ] **Step 2: 최종 전체 테스트**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 3: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: update plan history with ml-pipeline-fixes"
```
