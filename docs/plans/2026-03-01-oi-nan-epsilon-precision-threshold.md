# OI NaN 마스킹 / 분모 epsilon / 정밀도 우선 임계값 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** OI 데이터 결측 구간을 np.nan으로 처리하고, 분모 연산을 1e-8 패턴으로 통일하며, 임계값 탐색을 정밀도 우선(최소 재현율 조건부)으로 변경한다.

**Architecture:**
- `dataset_builder.py`: OI/펀딩비 nan 마스킹 + 분모 epsilon 통일 + `_rolling_zscore`의 nan-safe 처리
- `mlx_filter.py`: `fit()` 정규화 시 `np.nanmean`/`np.nanstd` + `nan_to_num` 적용
- `train_model.py`: 임계값 탐색 함수를 `precision_recall_curve` 기반으로 교체
- `train_mlx_model.py`: 동일한 임계값 탐색 함수 적용

**Tech Stack:** numpy, pandas, scikit-learn(precision_recall_curve), lightgbm, mlx

---

### Task 1: `dataset_builder.py` — OI/펀딩비 nan 마스킹

**Files:**
- Modify: `src/dataset_builder.py:261-268`
- Test: `tests/test_dataset_builder.py`

**Step 1: 기존 테스트 실행 (기준선 확인)**

```bash
python -m pytest tests/test_dataset_builder.py -v
```
Expected: 기존 테스트 전부 PASS (변경 전 기준선)

**Step 2: OI nan 마스킹 테스트 작성**

`tests/test_dataset_builder.py`에 아래 테스트 추가:

```python
def test_oi_nan_masking_no_column():
    """oi_change 컬럼이 없으면 전체가 nan이어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    # 최소한의 OHLCV 데이터 (지표 계산에 충분한 길이)
    n = 100
    np.random.seed(0)
    df = pd.DataFrame({
        "open":   np.random.uniform(1, 2, n),
        "high":   np.random.uniform(2, 3, n),
        "low":    np.random.uniform(0.5, 1, n),
        "close":  np.random.uniform(1, 2, n),
        "volume": np.random.uniform(1000, 5000, n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    # oi_change 컬럼이 없으면 oi_change 피처는 전부 nan이어야 함
    # (rolling zscore 후에도 nan이 전파되어야 함)
    assert feat["oi_change"].isna().all(), "oi_change 컬럼 없을 때 전부 nan이어야 함"


def test_oi_nan_masking_with_zeros():
    """oi_change 컬럼이 있어도 0.0 구간은 nan으로 마스킹되어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 100
    np.random.seed(0)
    df = pd.DataFrame({
        "open":      np.random.uniform(1, 2, n),
        "high":      np.random.uniform(2, 3, n),
        "low":       np.random.uniform(0.5, 1, n),
        "close":     np.random.uniform(1, 2, n),
        "volume":    np.random.uniform(1000, 5000, n),
        "oi_change": np.concatenate([np.zeros(50), np.random.uniform(-0.1, 0.1, 50)]),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    # 앞 50개 구간은 0이었으므로 nan으로 마스킹 → rolling zscore 후에도 nan 전파
    # 뒤 50개 구간은 실제 값이 있으므로 일부는 유한값이어야 함
    assert feat["oi_change"].iloc[50:].notna().any(), "실제 OI 값 구간에 유한값이 있어야 함"
```

**Step 3: 테스트 실행 (FAIL 확인)**

```bash
python -m pytest tests/test_dataset_builder.py::test_oi_nan_masking_no_column tests/test_dataset_builder.py::test_oi_nan_masking_with_zeros -v
```
Expected: FAIL (현재 0.0으로 채우므로 isna().all()이 False)

**Step 4: `dataset_builder.py` 수정**

`src/dataset_builder.py` 261~268줄을 아래로 교체:

```python
    # OI 변화율 / 펀딩비 피처
    # 컬럼 없으면 전체 nan, 있으면 0.0 구간(데이터 미제공 구간)을 nan으로 마스킹
    # LightGBM은 nan을 자체 처리; MLX는 fit()에서 nanmean/nanstd + nan_to_num 처리
    if "oi_change" in d.columns:
        oi_raw = np.where(d["oi_change"].values == 0.0, np.nan, d["oi_change"].values)
    else:
        oi_raw = np.full(len(d), np.nan)

    if "funding_rate" in d.columns:
        fr_raw = np.where(d["funding_rate"].values == 0.0, np.nan, d["funding_rate"].values)
    else:
        fr_raw = np.full(len(d), np.nan)

    result["oi_change"]    = _rolling_zscore(oi_raw.astype(np.float64))
    result["funding_rate"] = _rolling_zscore(fr_raw.astype(np.float64))
```

**Step 5: `_rolling_zscore` nan-safe 처리 확인 및 수정**

`src/dataset_builder.py` `_rolling_zscore` 함수 (118~128줄)를 nan-safe하게 수정:

```python
def _rolling_zscore(arr: np.ndarray, window: int = 288) -> np.ndarray:
    """rolling window z-score 정규화. nan은 전파된다(nan-safe).
    15분봉 기준 3일(288캔들) 윈도우. min_periods=1로 초반 데이터도 활용."""
    s = pd.Series(arr.astype(np.float64))
    r = s.rolling(window=window, min_periods=1)
    mean = r.mean()   # pandas rolling은 nan을 자동으로 건너뜀
    std  = r.std(ddof=0)
    std  = std.where(std >= 1e-8, other=1e-8)
    z = (s - mean) / std
    return z.values.astype(np.float32)
```

> 참고: pandas `rolling().mean()`은 기본적으로 nan을 건너뛰므로 별도 처리 불필요.
> nan 입력 → nan 출력이 자연스럽게 전파됨.

**Step 6: 테스트 재실행 (PASS 확인)**

```bash
python -m pytest tests/test_dataset_builder.py -v
```
Expected: 모든 테스트 PASS

**Step 7: 커밋**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: OI/펀딩비 결측 구간을 np.nan으로 마스킹 (0.0 → nan)"
```

---

### Task 2: `dataset_builder.py` — 분모 epsilon 통일

**Files:**
- Modify: `src/dataset_builder.py:157-168`
- Test: `tests/test_dataset_builder.py`

**Step 1: epsilon 통일 테스트 작성**

`tests/test_dataset_builder.py`에 추가:

```python
def test_epsilon_no_division_by_zero():
    """bb_range=0, close=0, vol_ma20=0 극단값에서 nan/inf가 발생하지 않아야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 100
    # close를 모두 같은 값으로 → bb_range=0 유발
    df = pd.DataFrame({
        "open":   np.ones(n),
        "high":   np.ones(n),
        "low":    np.ones(n),
        "close":  np.ones(n),
        "volume": np.ones(n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    assert not feat[numeric_cols].isin([np.inf, -np.inf]).any().any(), \
        "inf 값이 있으면 안 됨"
```

**Step 2: 테스트 실행 (기준선)**

```bash
python -m pytest tests/test_dataset_builder.py::test_epsilon_no_division_by_zero -v
```

**Step 3: `_calc_features_vectorized` 분모 epsilon 통일**

`src/dataset_builder.py` 157~168줄을 아래로 교체:

```python
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / (bb_range + 1e-8)

    ema_align = np.where(
        (ema9 > ema21) & (ema21 > ema50),  1,
        np.where(
            (ema9 < ema21) & (ema21 < ema50), -1, 0
        )
    ).astype(np.float32)

    atr_pct   = atr / (close + 1e-8)
    vol_ratio = volume / (vol_ma20 + 1e-8)
```

그리고 상대강도 계산 (246~247줄):

```python
        xrp_btc_rs_raw = (xrp_r1 / (btc_r1 + 1e-8)).astype(np.float32)
        xrp_eth_rs_raw = (xrp_r1 / (eth_r1 + 1e-8)).astype(np.float32)
```

**Step 4: 테스트 재실행**

```bash
python -m pytest tests/test_dataset_builder.py -v
```
Expected: 모든 테스트 PASS

**Step 5: 커밋**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "refactor: 분모 연산을 1e-8 epsilon 패턴으로 통일"
```

---

### Task 3: `mlx_filter.py` — nan-safe 정규화

**Files:**
- Modify: `src/mlx_filter.py:140-145`
- Test: `tests/test_mlx_filter.py`

**Step 1: nan-safe 정규화 테스트 작성**

`tests/test_mlx_filter.py`에 추가:

```python
def test_fit_with_nan_features():
    """oi_change 피처에 nan이 포함된 경우 학습이 정상 완료되어야 한다."""
    import numpy as np
    import pandas as pd
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    n = 300
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, len(FEATURE_COLS)).astype(np.float32),
        columns=FEATURE_COLS,
    )
    # oi_change 앞 절반을 nan으로
    X["oi_change"] = np.where(np.arange(n) < n // 2, np.nan, X["oi_change"])
    y = pd.Series((np.random.rand(n) > 0.5).astype(np.float32))

    model = MLXFilter(input_dim=len(FEATURE_COLS), hidden_dim=32, epochs=3)
    model.fit(X, y)  # nan 있어도 예외 없이 완료되어야 함

    proba = model.predict_proba(X)
    assert not np.any(np.isnan(proba)), "예측 확률에 nan이 없어야 함"
    assert proba.min() >= 0.0 and proba.max() <= 1.0
```

**Step 2: 테스트 실행 (FAIL 확인)**

```bash
python -m pytest tests/test_mlx_filter.py::test_fit_with_nan_features -v
```
Expected: FAIL (현재 nan이 그대로 들어가 loss=nan 발생)

**Step 3: `mlx_filter.py` fit() 정규화 수정**

`src/mlx_filter.py` 140~145줄을 아래로 교체:

```python
        X_np = X[FEATURE_COLS].values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # nan-safe 정규화: nanmean/nanstd로 통계 계산 후 nan → 0.0 대치
        # (z-score 후 0.0 = 평균값, 신경망에 줄 수 있는 가장 무난한 결측 대치값)
        self._mean = np.nanmean(X_np, axis=0)
        self._std  = np.nanstd(X_np, axis=0) + 1e-8
        X_np = (X_np - self._mean) / self._std
        X_np = np.nan_to_num(X_np, nan=0.0)
```

**Step 4: `predict_proba`도 nan_to_num 적용**

`src/mlx_filter.py` 185~189줄:

```python
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X[FEATURE_COLS].values.astype(np.float32)
        if self._trained and self._mean is not None:
            X_np = (X_np - self._mean) / self._std
            X_np = np.nan_to_num(X_np, nan=0.0)
```

**Step 5: 테스트 재실행**

```bash
python -m pytest tests/test_mlx_filter.py -v
```
Expected: 모든 테스트 PASS

**Step 6: 커밋**

```bash
git add src/mlx_filter.py tests/test_mlx_filter.py
git commit -m "fix: MLXFilter fit/predict에 nan-safe 정규화 적용 (nanmean + nan_to_num)"
```

---

### Task 4: `train_model.py` — 정밀도 우선 임계값 탐색

**Files:**
- Modify: `scripts/train_model.py:236-246`
- Test: 없음 (스크립트 레벨 변경, 수동 검증)

**Step 1: `train_model.py` 임계값 탐색 교체**

`scripts/train_model.py` 234~246줄을 아래로 교체:

```python
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    # 최적 임계값 탐색: 최소 재현율(0.15) 조건부 정밀도 최대화
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    # precision_recall_curve의 마지막 원소는 (1.0, 0.0)이므로 제외
    precisions, recalls = precisions[:-1], recalls[:-1]

    MIN_RECALL = 0.15
    valid_idx = np.where(recalls >= MIN_RECALL)[0]
    if len(valid_idx) > 0:
        best_idx  = valid_idx[np.argmax(precisions[valid_idx])]
        best_thr  = float(thresholds[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec  = float(recalls[best_idx])
    else:
        best_thr, best_prec, best_rec = 0.50, 0.0, 0.0
        print(f"  [경고] recall >= {MIN_RECALL} 조건 만족 임계값 없음 → 기본값 0.50 사용")

    print(f"\n검증 AUC: {auc:.4f}  |  최적 임계값: {best_thr:.4f} "
          f"(Precision={best_prec:.3f}, Recall={best_rec:.3f})")
    print(classification_report(y_val, (val_proba >= best_thr).astype(int), zero_division=0))
```

그리고 로그 저장 부분 (261~271줄)에 임계값 정보 추가:

```python
    log.append({
        "date": datetime.now().isoformat(),
        "backend": "lgbm",
        "auc": round(auc, 4),
        "best_threshold": round(best_thr, 4),
        "best_precision": round(best_prec, 3),
        "best_recall":    round(best_rec, 3),
        "samples": len(dataset),
        "features": len(actual_feature_cols),
        "time_weight_decay": time_weight_decay,
        "model_path": str(MODEL_PATH),
    })
```

**Step 2: 수동 검증 (dry-run)**

```bash
python scripts/train_model.py --data data/combined_15m.parquet 2>&1 | tail -30
```
Expected: "최적 임계값: X.XXXX (Precision=X.XXX, Recall=X.XXX)" 형태 출력

**Step 3: 커밋**

```bash
git add scripts/train_model.py
git commit -m "feat: LightGBM 임계값 탐색을 정밀도 우선(recall>=0.15 조건부)으로 변경"
```

---

### Task 5: `train_mlx_model.py` — 동일한 임계값 탐색 적용

**Files:**
- Modify: `scripts/train_mlx_model.py:119-122`

**Step 1: `train_mlx_model.py` 임계값 탐색 교체**

`scripts/train_mlx_model.py` 119~122줄을 아래로 교체:

```python
    val_proba = model.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_proba)

    # 최적 임계값 탐색: 최소 재현율(0.15) 조건부 정밀도 최대화
    from sklearn.metrics import precision_recall_curve, classification_report
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    MIN_RECALL = 0.15
    valid_idx = np.where(recalls >= MIN_RECALL)[0]
    if len(valid_idx) > 0:
        best_idx  = valid_idx[np.argmax(precisions[valid_idx])]
        best_thr  = float(thresholds[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec  = float(recalls[best_idx])
    else:
        best_thr, best_prec, best_rec = 0.50, 0.0, 0.0
        print(f"  [경고] recall >= {MIN_RECALL} 조건 만족 임계값 없음 → 기본값 0.50 사용")

    print(f"\n검증 AUC: {auc:.4f}  |  최적 임계값: {best_thr:.4f} "
          f"(Precision={best_prec:.3f}, Recall={best_rec:.3f})")
    print(classification_report(y_val, (val_proba >= best_thr).astype(int), zero_division=0))
```

그리고 로그 저장 부분에 임계값 정보 추가:

```python
    log.append({
        "date": datetime.now().isoformat(),
        "backend": "mlx",
        "auc": round(auc, 4),
        "best_threshold": round(best_thr, 4),
        "best_precision": round(best_prec, 3),
        "best_recall":    round(best_rec, 3),
        "samples": len(dataset),
        "train_sec": round(t3 - t2, 1),
        "time_weight_decay": time_weight_decay,
        "model_path": str(MLX_MODEL_PATH),
    })
```

**Step 2: 커밋**

```bash
git add scripts/train_mlx_model.py
git commit -m "feat: MLX 임계값 탐색을 정밀도 우선(recall>=0.15 조건부)으로 변경"
```

---

### Task 6: 전체 테스트 통과 확인

**Step 1: 전체 테스트 실행**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```
Expected: 모든 테스트 PASS

**Step 2: 최종 커밋 (필요 시)**

```bash
git add -A
git commit -m "chore: OI nan 마스킹 / epsilon 통일 / 정밀도 우선 임계값 전체 통합"
```
