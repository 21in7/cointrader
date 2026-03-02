# HOLD Negative Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** HOLD 캔들을 negative sample로 추가하고 계층적 언더샘플링을 도입하여 ML 학습 데이터를 535 → ~3,200개로 증가시킨다.

**Architecture:** `dataset_builder.py`에서 시그널 캔들 외에 HOLD 캔들을 label=0으로 추가 샘플링하고, `source` 컬럼("signal"/"hold_negative")으로 구분한다. 학습 시 signal 샘플은 전수 유지, HOLD negative에서만 양성 수 만큼 샘플링하는 계층적 언더샘플링을 적용한다.

**Tech Stack:** Python, NumPy, pandas, LightGBM, pytest

---

### Task 1: dataset_builder.py — HOLD Negative Sampling 추가

**Files:**
- Modify: `src/dataset_builder.py:360-421` (generate_dataset_vectorized 함수)
- Test: `tests/test_dataset_builder.py`

**Step 1: Write the failing tests**

`tests/test_dataset_builder.py` 끝에 2개 테스트 추가:

```python
def test_hold_negative_labels_are_all_zero(sample_df):
    """HOLD negative 샘플의 label은 전부 0이어야 한다."""
    result = generate_dataset_vectorized(sample_df, negative_ratio=3)
    if len(result) > 0 and "source" in result.columns:
        hold_neg = result[result["source"] == "hold_negative"]
        if len(hold_neg) > 0:
            assert (hold_neg["label"] == 0).all(), \
                f"HOLD negative 중 label != 0인 샘플 존재: {hold_neg['label'].value_counts().to_dict()}"


def test_signal_samples_preserved_after_sampling(sample_df):
    """계층적 샘플링 후 source='signal' 샘플이 하나도 버려지지 않아야 한다."""
    # negative_ratio=0이면 기존 동작 (signal만), >0이면 HOLD 추가
    result_signal_only = generate_dataset_vectorized(sample_df, negative_ratio=0)
    result_with_hold   = generate_dataset_vectorized(sample_df, negative_ratio=3)

    if len(result_with_hold) > 0 and "source" in result_with_hold.columns:
        signal_count = (result_with_hold["source"] == "signal").sum()
        assert signal_count == len(result_signal_only), \
            f"Signal 샘플 손실: 원본={len(result_signal_only)}, 유지={signal_count}"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset_builder.py::test_hold_negative_labels_are_all_zero tests/test_dataset_builder.py::test_signal_samples_preserved_after_sampling -v`
Expected: FAIL — `generate_dataset_vectorized()` does not accept `negative_ratio` parameter

**Step 3: Implement HOLD negative sampling in generate_dataset_vectorized**

`src/dataset_builder.py`의 `generate_dataset_vectorized()` 함수를 수정한다.
시그니처에 `negative_ratio: int = 0` 파라미터를 추가하고, HOLD 캔들 샘플링 로직을 삽입한다.

수정 대상: `generate_dataset_vectorized` 함수 전체.

```python
def generate_dataset_vectorized(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    time_weight_decay: float = 0.0,
    negative_ratio: int = 0,
) -> pd.DataFrame:
    """
    전체 시계열을 1회 계산해 학습 데이터셋을 생성한다.

    negative_ratio: 시그널 샘플 대비 HOLD negative 샘플 비율.
        0이면 기존 동작 (시그널만). 5면 시그널의 5배만큼 HOLD 샘플 추가.
    """
    print("  [1/3] 전체 시계열 지표 계산 (1회)...")
    d = _calc_indicators(df)

    print("  [2/3] 신호 마스킹 및 피처 추출...")
    signal_arr = _calc_signals(d)
    feat_all   = _calc_features_vectorized(d, signal_arr, btc_df=btc_df, eth_df=eth_df)

    # 신호 발생 + NaN 없음 + 미래 데이터 충분한 인덱스만
    OPTIONAL_COLS = {"oi_change", "funding_rate"}
    available_cols_for_nan_check = [
        c for c in FEATURE_COLS
        if c in feat_all.columns and c not in OPTIONAL_COLS
    ]
    base_valid = (
        (~feat_all[available_cols_for_nan_check].isna().any(axis=1).values) &
        (np.arange(len(d)) >= WARMUP) &
        (np.arange(len(d)) < len(d) - LOOKAHEAD)
    )

    # --- 시그널 캔들 (기존 로직) ---
    sig_valid = base_valid & (signal_arr != "HOLD")
    sig_idx = np.where(sig_valid)[0]
    print(f"  신호 발생 인덱스: {len(sig_idx):,}개")

    print("  [3/3] 레이블 계산...")
    labels, valid_mask = _calc_labels_vectorized(d, feat_all, sig_idx)

    final_sig_idx = sig_idx[valid_mask]
    available_feature_cols = [c for c in FEATURE_COLS if c in feat_all.columns]
    feat_signal = feat_all.iloc[final_sig_idx][available_feature_cols].copy()
    feat_signal["label"] = labels
    feat_signal["source"] = "signal"

    # --- HOLD negative 캔들 ---
    if negative_ratio > 0 and len(final_sig_idx) > 0:
        hold_valid = base_valid & (signal_arr == "HOLD")
        hold_candidates = np.where(hold_valid)[0]
        n_neg = min(len(hold_candidates), len(final_sig_idx) * negative_ratio)

        if n_neg > 0:
            rng = np.random.default_rng(42)
            hold_idx = rng.choice(hold_candidates, size=n_neg, replace=False)
            hold_idx = np.sort(hold_idx)

            feat_hold = feat_all.iloc[hold_idx][available_feature_cols].copy()
            feat_hold["label"] = 0
            feat_hold["source"] = "hold_negative"

            # HOLD 캔들은 시그널이 없으므로 side를 랜덤 할당 (50:50)
            sides = rng.integers(0, 2, size=len(feat_hold)).astype(np.float32)
            feat_hold["side"] = sides
            # signal_strength는 이미 0 (시그널 미발생이므로)

            print(f"  HOLD negative 추가: {len(feat_hold):,}개 "
                  f"(비율 1:{negative_ratio})")

            feat_final = pd.concat([feat_signal, feat_hold], ignore_index=True)
            # 시간 순서 복원 (원본 인덱스 기반 정렬)
            original_order = np.concatenate([final_sig_idx, hold_idx])
            sort_order = np.argsort(original_order)
            feat_final = feat_final.iloc[sort_order].reset_index(drop=True)
        else:
            feat_final = feat_signal.reset_index(drop=True)
    else:
        feat_final = feat_signal.reset_index(drop=True)

    # 시간 가중치
    n = len(feat_final)
    if time_weight_decay > 0 and n > 1:
        weights = np.exp(time_weight_decay * np.linspace(0.0, 1.0, n)).astype(np.float32)
        weights /= weights.mean()
        print(f"  시간 가중치 적용 (decay={time_weight_decay}): "
              f"min={weights.min():.3f}, max={weights.max():.3f}")
    else:
        weights = np.ones(n, dtype=np.float32)

    feat_final["sample_weight"] = weights

    total_sig = (feat_final["source"] == "signal").sum() if "source" in feat_final.columns else len(feat_final)
    total_hold = (feat_final["source"] == "hold_negative").sum() if "source" in feat_final.columns else 0
    print(f"  최종 데이터셋: {n:,}개 (시그널={total_sig:,}, HOLD={total_hold:,})")

    return feat_final
```

**Step 4: Run the new tests to verify they pass**

Run: `pytest tests/test_dataset_builder.py::test_hold_negative_labels_are_all_zero tests/test_dataset_builder.py::test_signal_samples_preserved_after_sampling -v`
Expected: PASS

**Step 5: Run all existing dataset_builder tests to verify no regressions**

Run: `pytest tests/test_dataset_builder.py -v`
Expected: All existing tests PASS (기존 동작은 negative_ratio=0 기본값으로 유지)

**Step 6: Commit**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: add HOLD negative sampling to dataset builder"
```

---

### Task 2: 계층적 언더샘플링 헬퍼 함수

**Files:**
- Modify: `src/dataset_builder.py` (파일 끝에 헬퍼 추가)
- Test: `tests/test_dataset_builder.py`

**Step 1: Write the failing test**

```python
def test_stratified_undersample_preserves_signal():
    """stratified_undersample은 signal 샘플을 전수 유지해야 한다."""
    from src.dataset_builder import stratified_undersample

    y      = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    source = np.array(["signal", "signal", "signal", "hold_negative",
                        "hold_negative", "hold_negative", "hold_negative",
                        "hold_negative", "signal", "signal"])

    idx = stratified_undersample(y, source, seed=42)

    # signal 인덱스: 0, 1, 2, 8, 9 → 전부 포함
    signal_indices = np.where(source == "signal")[0]
    for si in signal_indices:
        assert si in idx, f"signal 인덱스 {si}가 누락됨"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_builder.py::test_stratified_undersample_preserves_signal -v`
Expected: FAIL — `stratified_undersample` 함수 미존재

**Step 3: Implement stratified_undersample**

`src/dataset_builder.py` 끝에 추가:

```python
def stratified_undersample(
    y: np.ndarray,
    source: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Signal 샘플 전수 유지 + HOLD negative만 양성 수 만큼 샘플링.

    Args:
        y: 라벨 배열 (0 or 1)
        source: 소스 배열 ("signal" or "hold_negative")
        seed: 랜덤 시드

    Returns:
        정렬된 인덱스 배열 (학습에 사용할 행 인덱스)
    """
    pos_idx = np.where(y == 1)[0]                                    # Signal Win
    sig_neg_idx = np.where((y == 0) & (source == "signal"))[0]       # Signal Loss
    hold_neg_idx = np.where(source == "hold_negative")[0]             # HOLD negative

    # HOLD negative에서 양성 수 만큼만 샘플링
    n_hold = min(len(hold_neg_idx), len(pos_idx))
    rng = np.random.default_rng(seed)
    if n_hold > 0:
        hold_sampled = rng.choice(hold_neg_idx, size=n_hold, replace=False)
    else:
        hold_sampled = np.array([], dtype=np.intp)

    return np.sort(np.concatenate([pos_idx, sig_neg_idx, hold_sampled]))
```

**Step 4: Run tests**

Run: `pytest tests/test_dataset_builder.py::test_stratified_undersample_preserves_signal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: add stratified_undersample helper function"
```

---

### Task 3: train_model.py — 계층적 언더샘플링 적용

**Files:**
- Modify: `scripts/train_model.py:229-257` (train 함수)
- Modify: `scripts/train_model.py:356-391` (walk_forward_auc 함수)

**Step 1: Update train() function**

`scripts/train_model.py`에서 `dataset_builder`에서 `stratified_undersample`을 import하고,
`train()` 함수의 언더샘플링 블록을 교체한다.

import 수정 (line 25):
```python
from src.dataset_builder import generate_dataset_vectorized, stratified_undersample
```

`train()` 함수에서 데이터셋 생성 호출에 `negative_ratio=5` 추가 (line 217):
```python
    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df,
        time_weight_decay=time_weight_decay,
        negative_ratio=5,
    )
```

source 배열 추출 추가 (line 231 부근, w 다음):
```python
    source = dataset["source"].values if "source" in dataset.columns else np.full(len(X), "signal")
```

언더샘플링 블록 교체 (line 241-257):
```python
    # --- 계층적 샘플링: signal 전수 유지, HOLD negative만 양성 수 만큼 ---
    source_train = source[:split]
    balanced_idx = stratified_undersample(y_train.values, source_train, seed=42)

    X_train = X_train.iloc[balanced_idx]
    y_train = y_train.iloc[balanced_idx]
    w_train = w_train[balanced_idx]

    sig_count = (source_train[balanced_idx] == "signal").sum()
    hold_count = (source_train[balanced_idx] == "hold_negative").sum()
    print(f"\n계층적 샘플링 후 학습 데이터: {len(X_train)}개 "
          f"(Signal={sig_count}, HOLD={hold_count}, "
          f"양성={int(y_train.sum())}, 음성={int((y_train==0).sum())})")
    print(f"검증 데이터: {len(X_val)}개 (양성={int(y_val.sum())}, 음성={int((y_val==0).sum())})")
```

**Step 2: Update walk_forward_auc() function**

`walk_forward_auc()` 함수에서도 동일하게 적용.

dataset 생성 (line 356-358)에 `negative_ratio=5` 추가:
```python
    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df,
        time_weight_decay=time_weight_decay,
        negative_ratio=5,
    )
```

source 배열 추출 (line 362 부근):
```python
    source = dataset["source"].values if "source" in dataset.columns else np.full(n, "signal")
```

폴드 내 언더샘플링 교체 (line 381-386):
```python
        source_tr = source[:tr_end]
        bal_idx = stratified_undersample(y_tr, source_tr, seed=42)
```

**Step 3: Run training to verify**

Run: `python scripts/train_model.py --data data/combined_15m.parquet --decay 2.0`
Expected: 학습 샘플 수 대폭 증가 확인 (기존 ~535 → ~3,200)

**Step 4: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: apply stratified undersampling to training pipeline"
```

---

### Task 4: tune_hyperparams.py — 계층적 언더샘플링 적용

**Files:**
- Modify: `scripts/tune_hyperparams.py:41-81` (load_dataset)
- Modify: `scripts/tune_hyperparams.py:88-144` (_walk_forward_cv)
- Modify: `scripts/tune_hyperparams.py:151-206` (make_objective)
- Modify: `scripts/tune_hyperparams.py:213-244` (measure_baseline)
- Modify: `scripts/tune_hyperparams.py:370-449` (main)

**Step 1: Update load_dataset to return source**

import 수정 (line 34):
```python
from src.dataset_builder import generate_dataset_vectorized, stratified_undersample
```

`load_dataset()` 시그니처와 반환값 수정:
```python
def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

dataset 생성에 `negative_ratio=5` 추가 (line 66):
```python
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=0.0, negative_ratio=5)
```

source 추출 추가 (line 74 부근, w 다음):
```python
    source = dataset["source"].values if "source" in dataset.columns else np.full(len(dataset), "signal")
```

return 수정:
```python
    return X, y, w, source
```

**Step 2: Update _walk_forward_cv to accept and use source**

시그니처에 source 추가:
```python
def _walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    params: dict,
    ...
```

폴드 내 언더샘플링 교체 (line 117-122):
```python
        source_tr = source[:tr_end]
        bal_idx = stratified_undersample(y_tr, source_tr, seed=42)
```

**Step 3: Update make_objective, measure_baseline, main**

`make_objective()`: 클로저에 source 캡처, `_walk_forward_cv` 호출에 source 전달
`measure_baseline()`: source 파라미터 추가, `_walk_forward_cv` 호출에 전달
`main()`: `load_dataset` 반환값 4개로 변경, 하위 함수에 source 전달

**Step 4: Commit**

```bash
git add scripts/tune_hyperparams.py
git commit -m "feat: apply stratified undersampling to hyperparameter tuning"
```

---

### Task 5: 전체 테스트 실행 및 검증

**Step 1: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: All tests PASS

**Step 2: Run training pipeline end-to-end**

Run: `python scripts/train_model.py --data data/combined_15m.parquet --decay 2.0`
Expected:
- 학습 샘플 ~3,200개 (기존 535)
- "계층적 샘플링 후" 로그에 Signal/HOLD 카운트 표시
- AUC 출력 (값 자체보다 실행 완료가 중요)

**Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: verify HOLD negative sampling pipeline end-to-end"
```
