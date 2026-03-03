# ADX ML 피처 마이그레이션 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** ADX 하드필터를 제거하고 ADX를 24번째 ML 피처로 추가하여, 횡보장 판단을 ML 모델에 위임한다. ADX 값을 항상 로그에 남겨 대시보드 끊김도 해소한다.

**Architecture:** `indicators.py`에서 ADX < 25 early-return 삭제, `ml_features.py`에 ADX 피처 추가 (23 → 24개), `dataset_builder.py`에서 ADX 하드필터 삭제 + ADX 피처 추출 추가. 기존 모델과 호환 안 되므로 재학습 필수.

**Tech Stack:** LightGBM, pandas-ta, pytest

---

### Task 1: ML 피처 테스트 업데이트 (24개 피처)

**Files:**
- Modify: `tests/test_ml_features.py:52-54`

**Step 1: Update the test**

`test_feature_cols_has_23_items`를 24개로 변경:

```python
def test_feature_cols_has_24_items():
    from src.ml_features import FEATURE_COLS
    assert len(FEATURE_COLS) == 24
```

`test_build_features_with_btc_eth_has_21_features`의 assert도 변경:

```python
def test_build_features_with_btc_eth_has_24_features():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    assert len(features) == 24
```

`test_build_features_without_btc_eth_has_13_features`도 변경:

```python
def test_build_features_without_btc_eth_has_16_features():
    xrp_df = _make_df(10, base_price=1.0)
    features = build_features(xrp_df, "LONG")
    assert len(features) == 16
```

`_make_df`에 `"adx": [20.0] * n` 컬럼 추가.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml_features.py::test_feature_cols_has_24_items -v`
Expected: FAIL — 현재 23개

---

### Task 2: FEATURE_COLS에 ADX 추가 + build_features() 수정

**Files:**
- Modify: `src/ml_features.py:4-14` (FEATURE_COLS), `src/ml_features.py:98-112` (base dict)

**Step 3: Add ADX to FEATURE_COLS**

```python
FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    "btc_ret_1", "btc_ret_3", "btc_ret_5",
    "eth_ret_1", "eth_ret_3", "eth_ret_5",
    "xrp_btc_rs", "xrp_eth_rs",
    "oi_change", "funding_rate",
    "adx",
]
```

**Step 4: Add ADX extraction in build_features()**

`base` dict 생성 부분 (line 112 이후)에 추가:

```python
    base["adx"] = float(last.get("adx", 0))
```

docstring의 "23개 피처"를 "24개 피처"로 변경.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ml_features.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/ml_features.py tests/test_ml_features.py
git commit -m "feat: add ADX as 24th ML feature"
```

---

### Task 3: indicators.py ADX 하드필터 제거 + 항상 로깅

**Files:**
- Modify: `src/indicators.py:63-67`

**Step 7: Replace ADX hard filter with always-log**

`get_signal()` 메서드에서 기존 ADX 필터 코드:

```python
        # ADX 횡보장 필터: ADX < 25이면 추세 부재로 판단하여 진입 차단
        adx = last.get("adx", None)
        if adx is not None and not pd.isna(adx) and adx < 25:
            logger.debug(f"ADX 필터: {adx:.1f} < 25 — HOLD")
            return "HOLD"
```

를 다음으로 교체:

```python
        # ADX 로깅 (ML 피처로 위임, 하드필터 제거)
        adx = last.get("adx", None)
        if adx is not None and not pd.isna(adx):
            logger.debug(f"ADX: {adx:.1f}")
```

**Step 8: Run ADX-related tests**

Run: `pytest tests/test_indicators.py -k "adx" -v`
Expected: `test_adx_column_exists` PASS, `test_adx_nan_falls_through` PASS, `test_adx_filter_blocks_low_adx` FAIL (필터 제거됨)

---

### Task 4: ADX 필터 테스트 업데이트

**Files:**
- Modify: `tests/test_indicators.py:57-71`

**Step 9: Replace block test with pass-through test**

`test_adx_filter_blocks_low_adx`를 제거하고 새 테스트로 교체:

```python
def test_adx_low_does_not_block_signal(sample_df):
    """ADX < 25여도 시그널이 차단되지 않는다 (ML에 위임)."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    # 강한 LONG 신호가 나오도록 지표 조작
    df.loc[df.index[-1], "rsi"] = 20
    df.loc[df.index[-2], "macd"] = -1
    df.loc[df.index[-2], "macd_signal"] = 0
    df.loc[df.index[-1], "macd"] = 1
    df.loc[df.index[-1], "macd_signal"] = 0
    df.loc[df.index[-1], "volume"] = df.loc[df.index[-1], "vol_ma20"] * 2
    df["adx"] = 15.0
    signal = ind.get_signal(df)
    # ADX 낮아도 지표 조건 충족 시 LONG 반환 (ML이 최종 판단)
    assert signal == "LONG"
```

**Step 10: Run all indicator tests**

Run: `pytest tests/test_indicators.py -v`
Expected: ALL PASS

**Step 11: Commit**

```bash
git add src/indicators.py tests/test_indicators.py
git commit -m "feat: remove ADX hard filter, delegate to ML"
```

---

### Task 5: dataset_builder.py ADX 하드필터 제거 + ADX 피처 추가

**Files:**
- Modify: `src/dataset_builder.py:119-123` (ADX 필터 삭제), `src/dataset_builder.py:215-230` (ADX 피처 추가)

**Step 12: Remove ADX hard filter in _calc_signals()**

`_calc_signals()` 함수에서 다음 코드 삭제 (lines 119-123):

```python
    # ADX 횡보장 필터: ADX < 25이면 추세 부재로 판단하여 진입 차단
    if "adx" in d.columns:
        adx = d["adx"].values
        low_adx = (~np.isnan(adx)) & (adx < 25)
        signal_arr[low_adx] = "HOLD"
```

**Step 13: Add ADX feature to _calc_features_vectorized()**

`_calc_features_vectorized()` 함수의 `result` DataFrame 생성 부분에 `"adx"` 추가:

```python
    # ADX (ML 피처로 제공 — rolling z-score 정규화)
    adx_raw = d["adx"].values.astype(np.float64) if "adx" in d.columns else np.zeros(len(d), dtype=np.float64)
    adx_z = _rolling_zscore(adx_raw)
```

`result` DataFrame에 `"adx": adx_z,` 추가 (side 다음에).

**Step 14: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 15: Commit**

```bash
git add src/dataset_builder.py
git commit -m "feat: remove ADX hard filter from dataset builder, add ADX as ML feature"
```

---

### Task 6: 전체 테스트 + 최종 검증

**Step 16: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

**Step 17: Final commit if needed**

주의: 기존 모델(23 피처)은 24 피처 입력과 호환 안 됨. 배포 전 반드시 `bash scripts/train_and_deploy.sh` 실행하여 재학습 필요.
