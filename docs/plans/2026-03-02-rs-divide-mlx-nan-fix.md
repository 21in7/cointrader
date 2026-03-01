# RS np.divide 복구 / MLX NaN-Safe 통계 저장 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** RS(상대강도) 계산의 epsilon 폭발 이상치를 `np.divide` 방식으로 제거하고, MLXFilter의 `self._mean`/`self._std`에 NaN이 잔류하는 근본 허점을 차단한다.

**Architecture:**
- `src/dataset_builder.py`: `xrp_btc_rs_raw` / `xrp_eth_rs_raw` 계산을 `np.divide(..., where=...)` 방식으로 교체. 분모(btc_r1, eth_r1)가 0이면 결과를 0.0으로 채워 rolling zscore 윈도우 오염을 방지한다.
- `src/mlx_filter.py`: `fit()` 내부에서 `self._mean`/`self._std`를 저장하기 전에 `nan_to_num`을 적용해 전체-NaN 컬럼(OI 초반 구간 등)이 `predict_proba` 시점까지 NaN을 전파하지 않도록 한다.

**Tech Stack:** numpy, pandas, pytest, mlx(Apple Silicon 전용 — MLX 테스트는 Mac에서만 실행)

---

### Task 1: `dataset_builder.py` — RS 계산을 `np.divide` 방식으로 교체

**Files:**
- Modify: `src/dataset_builder.py:245-246`
- Test: `tests/test_dataset_builder.py`

**배경:**
`btc_r1 = 0.0`(15분 동안 BTC 가격 변동 없음)일 때 `xrp_r1 / (btc_r1 + 1e-8)`는 최대 수백만의 이상치를 만든다. 이 이상치가 288캔들 rolling zscore 윈도우에 들어가면 나머지 287개 값이 전부 0에 가깝게 압사된다.

**Step 1: 기존 테스트 실행 (기준선 확인)**

```bash
python -m pytest tests/test_dataset_builder.py -v
```

Expected: 모든 테스트 PASS (변경 전 기준선)

**Step 2: RS 제로-분모 테스트 작성**

`tests/test_dataset_builder.py` 파일 끝에 추가:

```python
def test_rs_zero_denominator():
    """btc_r1=0일 때 RS가 inf/nan이 아닌 0.0이어야 한다 (np.divide 방식 검증)."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 500
    np.random.seed(7)
    # XRP close: 약간의 변동
    xrp_close = np.cumprod(1 + np.random.randn(n) * 0.001) * 1.0
    xrp_df = pd.DataFrame({
        "open":   xrp_close * 0.999,
        "high":   xrp_close * 1.005,
        "low":    xrp_close * 0.995,
        "close":  xrp_close,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    # BTC close: 완전히 고정 → btc_r1 = 0.0
    btc_close = np.ones(n) * 50000.0
    btc_df = pd.DataFrame({
        "open":   btc_close,
        "high":   btc_close,
        "low":    btc_close,
        "close":  btc_close,
        "volume": np.random.rand(n) * 1000 + 500,
    })

    from src.dataset_builder import generate_dataset_vectorized
    result = generate_dataset_vectorized(xrp_df, btc_df=btc_df)

    if result.empty:
        pytest.skip("신호 없음")

    assert "xrp_btc_rs" in result.columns, "xrp_btc_rs 컬럼이 있어야 함"
    assert not result["xrp_btc_rs"].isin([np.inf, -np.inf]).any(), \
        "xrp_btc_rs에 inf가 있으면 안 됨"
    assert not result["xrp_btc_rs"].isna().all(), \
        "xrp_btc_rs가 전부 nan이면 안 됨"
```

**Step 3: 테스트 실행 (FAIL 확인)**

```bash
python -m pytest tests/test_dataset_builder.py::test_rs_zero_denominator -v
```

Expected: FAIL — `xrp_btc_rs에 inf가 있으면 안 됨` (현재 epsilon 방식은 inf 대신 수백만 이상치를 만들어 rolling zscore 후 nan이 될 수 있음)

> 참고: 현재 코드는 inf를 직접 만들지 않을 수도 있다. 하지만 rolling zscore 후 nan이 생기거나 이상치가 남아있는지 확인하는 것이 목적이다. PASS가 나오더라도 Step 4를 진행한다.

**Step 4: `dataset_builder.py` 245~246줄 수정**

`src/dataset_builder.py`의 아래 두 줄을:

```python
        xrp_btc_rs_raw = (xrp_r1 / (btc_r1 + 1e-8)).astype(np.float32)
        xrp_eth_rs_raw = (xrp_r1 / (eth_r1 + 1e-8)).astype(np.float32)
```

다음으로 교체:

```python
        xrp_btc_rs_raw = np.divide(
            xrp_r1, btc_r1,
            out=np.zeros_like(xrp_r1),
            where=(btc_r1 != 0),
        ).astype(np.float32)
        xrp_eth_rs_raw = np.divide(
            xrp_r1, eth_r1,
            out=np.zeros_like(xrp_r1),
            where=(eth_r1 != 0),
        ).astype(np.float32)
```

**Step 5: 전체 테스트 실행 (PASS 확인)**

```bash
python -m pytest tests/test_dataset_builder.py -v
```

Expected: 모든 테스트 PASS

**Step 6: 커밋**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "fix: RS 계산을 np.divide(where=) 방식으로 교체 — epsilon 이상치 폭발 차단"
```

---

### Task 2: `mlx_filter.py` — `self._mean`/`self._std` 저장 전 `nan_to_num` 적용

**Files:**
- Modify: `src/mlx_filter.py:145-146`
- Test: `tests/test_mlx_filter.py` (기존 `test_fit_with_nan_features` 활용)

**배경:**
현재 코드는 `self._mean = np.nanmean(X_np, axis=0)`으로 저장한다. 전체가 NaN인 컬럼(Walk-Forward 초반 11개월의 OI 데이터)이 있으면 `np.nanmean`은 해당 컬럼의 평균으로 NaN을 반환한다. 이 NaN이 `self._mean`에 저장되면 `predict_proba` 시점에 `(X_np - self._mean)`이 NaN이 되어 OI 데이터를 영원히 활용하지 못한다.

**Step 1: 기존 테스트 실행 (기준선 확인)**

```bash
python -m pytest tests/test_mlx_filter.py -v
```

Expected: 모든 테스트 PASS (MLX 없는 환경에서는 전체 SKIP)

**Step 2: `mlx_filter.py` 145~146줄 수정**

`src/mlx_filter.py`의 아래 두 줄을:

```python
        self._mean = np.nanmean(X_np, axis=0)
        self._std  = np.nanstd(X_np, axis=0) + 1e-8
```

다음으로 교체:

```python
        mean_vals  = np.nanmean(X_np, axis=0)
        self._mean = np.nan_to_num(mean_vals, nan=0.0)   # 전체-NaN 컬럼 → 평균 0.0
        std_vals   = np.nanstd(X_np, axis=0)
        self._std  = np.nan_to_num(std_vals, nan=1.0) + 1e-8  # 전체-NaN 컬럼 → std 1.0
```

**Step 3: 테스트 실행 (PASS 확인)**

```bash
python -m pytest tests/test_mlx_filter.py::test_fit_with_nan_features -v
```

Expected: PASS (MLX 없는 환경에서는 SKIP)

**Step 4: 전체 테스트 실행**

```bash
python -m pytest tests/test_mlx_filter.py -v
```

Expected: 모든 테스트 PASS (또는 SKIP)

**Step 5: 커밋**

```bash
git add src/mlx_filter.py
git commit -m "fix: MLXFilter self._mean/std 저장 전 nan_to_num 적용 — 전체-NaN 컬럼 predict_proba 오염 차단"
```

---

### Task 3: 전체 테스트 통과 확인

**Step 1: 전체 테스트 실행**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: 모든 테스트 PASS (MLX 관련은 SKIP 허용)

**Step 2: 최종 커밋 (필요 시)**

```bash
git add -A
git commit -m "chore: RS epsilon 폭발 차단 + MLX NaN-Safe 통계 저장 통합"
```
