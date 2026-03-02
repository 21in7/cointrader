# ADX 횡보장 필터 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** ADX < 25일 때 get_signal()에서 즉시 HOLD를 반환하여 횡보장 진입을 차단한다.

**Architecture:** `calculate_all()`에서 `pandas_ta.adx()`로 ADX 컬럼을 추가하고, `get_signal()`에서 가중치 계산 전 ADX < 25이면 early-return HOLD. NaN(초기 캔들)은 기존 로직으로 폴백.

**Tech Stack:** pandas-ta (이미 사용 중), pytest

---

### Task 1: ADX 계산 테스트 추가

**Files:**
- Test: `tests/test_indicators.py`

**Step 1: Write the failing test**

```python
def test_adx_column_exists(sample_df):
    """calculate_all()이 adx 컬럼을 생성하는지 확인."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    assert "adx" in df.columns
    valid = df["adx"].dropna()
    assert (valid >= 0).all()
```

`tests/test_indicators.py`에 위 테스트 함수를 추가한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators.py::test_adx_column_exists -v`
Expected: FAIL — `"adx" not in df.columns`

---

### Task 2: calculate_all()에 ADX 계산 추가

**Files:**
- Modify: `src/indicators.py:46-48` (vol_ma20 계산 바로 앞에 추가)

**Step 3: Write minimal implementation**

`calculate_all()`의 Stochastic RSI 계산 뒤, `vol_ma20` 계산 앞에 추가:

```python
        # ADX (14) — 횡보장 필터
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_df["ADX_14"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators.py::test_adx_column_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/indicators.py tests/test_indicators.py
git commit -m "feat: add ADX calculation to indicators"
```

---

### Task 3: ADX 필터 테스트 추가 (차단 케이스)

**Files:**
- Test: `tests/test_indicators.py`

**Step 6: Write the failing test**

```python
def test_adx_filter_blocks_low_adx(sample_df):
    """ADX < 25일 때 가중치와 무관하게 HOLD를 반환해야 한다."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    # ADX를 강제로 낮은 값으로 설정
    df["adx"] = 15.0
    signal = ind.get_signal(df)
    assert signal == "HOLD"
```

**Step 7: Run test to verify it fails**

Run: `pytest tests/test_indicators.py::test_adx_filter_blocks_low_adx -v`
Expected: FAIL — signal이 LONG 또는 SHORT 반환 (ADX 필터 미구현)

---

### Task 4: ADX 필터 테스트 추가 (NaN 폴백 케이스)

**Files:**
- Test: `tests/test_indicators.py`

**Step 8: Write the failing test**

```python
def test_adx_nan_falls_through(sample_df):
    """ADX가 NaN(초기 캔들)이면 기존 가중치 로직으로 폴백해야 한다."""
    ind = Indicators(sample_df)
    df = ind.calculate_all()
    df["adx"] = float("nan")
    signal = ind.get_signal(df)
    # NaN이면 차단하지 않고 기존 로직 실행 → LONG/SHORT/HOLD 중 하나
    assert signal in ("LONG", "SHORT", "HOLD")
```

**Step 9: Run test to verify it passes (이 테스트는 현재도 통과)**

Run: `pytest tests/test_indicators.py::test_adx_nan_falls_through -v`
Expected: PASS (ADX 컬럼이 무시되므로 기존 로직 그대로)

---

### Task 5: get_signal()에 ADX early-return 구현

**Files:**
- Modify: `src/indicators.py:51-56` (get_signal 메서드 시작부)

**Step 10: Write minimal implementation**

`get_signal()` 메서드의 `last = df.iloc[-1]` 바로 다음에 추가:

```python
        # ADX 횡보장 필터: ADX < 25이면 추세 부재로 판단하여 진입 차단
        adx = last.get("adx", None)
        if adx is not None and not pd.isna(adx) and adx < 25:
            logger.debug(f"ADX 필터: {adx:.1f} < 25 — HOLD")
            return "HOLD"
```

**Step 11: Run all ADX-related tests**

Run: `pytest tests/test_indicators.py -k "adx" -v`
Expected: 3 tests PASS

**Step 12: Run full test suite to check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 13: Commit**

```bash
git add src/indicators.py tests/test_indicators.py
git commit -m "feat: add ADX filter to block sideways market entries"
```
