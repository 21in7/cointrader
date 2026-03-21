# Training Threshold Relaxation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ML 학습용 신호 임계값을 완화하여 학습 샘플을 5~10배 증가시키고, 모델이 의미 있는 패턴을 학습할 수 있도록 한다.

**Architecture:** `dataset_builder.py`에 학습 전용 상수 블록(`TRAIN_*`)을 추가하고, `generate_dataset_vectorized()`의 기본값을 이 상수로 변경한다. 모든 호출부(train_model, train_mlx_model, tune_hyperparams)는 기본값을 따르므로 호출부 코드 변경 없이 적용된다. 실전 봇(`bot.py`)과 백테스터 시뮬레이션(`Backtester.run`)은 `config.py`의 엄격한 임계값을 별도로 사용하므로 영향 없다.

**Tech Stack:** Python, pandas, numpy, LightGBM, pytest

---

## File Structure

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `src/dataset_builder.py` | Modify | 학습 전용 상수 추가 + 기본값 변경 |
| `scripts/train_model.py` | Modify | 하드코딩된 `negative_ratio=5` → 기본값 사용으로 전환 |
| `scripts/train_mlx_model.py` | Modify | 동일 |
| `scripts/tune_hyperparams.py` | Modify | 동일 |
| `src/backtester.py` | Modify | `WalkForwardConfig.negative_ratio` 기본값 변경 |
| `tests/test_dataset_builder.py` | Modify | 완화된 기본값 반영 |
| `tests/test_ml_pipeline_fixes.py` | Modify | 새 기본값 검증 테스트 추가 |

---

### Task 1: dataset_builder.py에 학습 전용 상수 추가 + 기본값 변경

**Files:**
- Modify: `src/dataset_builder.py:14-17, 387-397`
- Test: `tests/test_ml_pipeline_fixes.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_ml_pipeline_fixes.py`에 추가:

```python
def test_training_defaults_are_relaxed(signal_df):
    """generate_dataset_vectorized의 기본 임계값이 학습용 완화 값이어야 한다."""
    from src.dataset_builder import (
        TRAIN_SIGNAL_THRESHOLD, TRAIN_ADX_THRESHOLD,
        TRAIN_VOLUME_MULTIPLIER, TRAIN_NEGATIVE_RATIO,
    )
    assert TRAIN_SIGNAL_THRESHOLD == 2
    assert TRAIN_ADX_THRESHOLD == 15.0
    assert TRAIN_VOLUME_MULTIPLIER == 1.5
    assert TRAIN_NEGATIVE_RATIO == 3

    # 완화된 기본값으로 샘플이 더 많이 생성되는지 검증
    r_relaxed = generate_dataset_vectorized(signal_df)
    r_strict = generate_dataset_vectorized(
        signal_df, signal_threshold=3, adx_threshold=25, volume_multiplier=2.5,
    )
    assert len(r_relaxed) >= len(r_strict), \
        f"완화된 임계값이 더 많은 샘플을 생성해야 한다: relaxed={len(r_relaxed)}, strict={len(r_strict)}"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py::test_training_defaults_are_relaxed -v`
Expected: FAIL — `ImportError: cannot import name 'TRAIN_SIGNAL_THRESHOLD'`

- [ ] **Step 3: dataset_builder.py 수정**

`src/dataset_builder.py` 상단 상수 블록(line 14-17)을 변경:

```python
LOOKAHEAD    = 24   # 15분봉 × 24 = 6시간 뷰
ATR_SL_MULT  = 2.0  # config.py 기본값과 동일 (서빙 환경 일치)
ATR_TP_MULT  = 2.0
WARMUP       = 60   # 15분봉 기준 60캔들 = 15시간 (지표 안정화 충분)

# ── 학습 전용 기본값 ──────────────────────────────────────────────
# 실전 봇(config.py)보다 완화된 임계값으로 더 많은 신호를 수집한다.
# ML 모델이 약한 신호 중에서 좋은 기회를 구분하는 법을 학습한다.
# 실전 진입은 bot.py의 엄격한 5단 게이트 + ML 필터가 최종 판단.
TRAIN_SIGNAL_THRESHOLD = 2      # 실전: 3 (config.py)
TRAIN_ADX_THRESHOLD    = 15.0   # 실전: 25.0
TRAIN_VOLUME_MULTIPLIER = 1.5   # 실전: 2.5
TRAIN_NEGATIVE_RATIO   = 3     # HOLD 네거티브 비율 (기존: 5)
```

`generate_dataset_vectorized()` 시그니처(line 387-397)의 기본값을 변경:

```python
def generate_dataset_vectorized(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    time_weight_decay: float = 0.0,
    negative_ratio: int = TRAIN_NEGATIVE_RATIO,          # 변경: 0 → 3
    signal_threshold: int = TRAIN_SIGNAL_THRESHOLD,      # 변경: 3 → 2
    adx_threshold: float = TRAIN_ADX_THRESHOLD,          # 변경: 25 → 15
    volume_multiplier: float = TRAIN_VOLUME_MULTIPLIER,  # 변경: 2.5 → 1.5
    atr_sl_mult: float = ATR_SL_MULT,
    atr_tp_mult: float = ATR_TP_MULT,
) -> pd.DataFrame:
```

또한 `_calc_signals()`(line 57-61)의 기본값도 학습 상수로 변경:

```python
def _calc_signals(
    d: pd.DataFrame,
    signal_threshold: int = TRAIN_SIGNAL_THRESHOLD,      # 변경: 3 → 2
    adx_threshold: float = TRAIN_ADX_THRESHOLD,          # 변경: 25 → 15
    volume_multiplier: float = TRAIN_VOLUME_MULTIPLIER,  # 변경: 2.5 → 1.5
) -> np.ndarray:
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_ml_pipeline_fixes.py tests/test_dataset_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add src/dataset_builder.py tests/test_ml_pipeline_fixes.py
git commit -m "feat(ml): add TRAIN_* constants with relaxed thresholds for more training samples"
```

---

### Task 2: 호출부에서 하드코딩된 값 제거

**Files:**
- Modify: `scripts/train_model.py`
- Modify: `scripts/train_mlx_model.py`
- Modify: `scripts/tune_hyperparams.py`
- Modify: `src/backtester.py`

- [ ] **Step 1: train_model.py — 하드코딩 negative_ratio=5 제거**

`train()`, `walk_forward_auc()`, `compare()` 내 `generate_dataset_vectorized()` 호출에서 `negative_ratio=5`를 삭제하여 기본값(`TRAIN_NEGATIVE_RATIO=3`)을 사용하도록 변경.

변경 전 (3곳):
```python
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=time_weight_decay,
    negative_ratio=5,
    atr_sl_mult=atr_sl_mult,
    atr_tp_mult=atr_tp_mult,
)
```

변경 후:
```python
dataset = generate_dataset_vectorized(
    df, btc_df=btc_df, eth_df=eth_df,
    time_weight_decay=time_weight_decay,
    atr_sl_mult=atr_sl_mult,
    atr_tp_mult=atr_tp_mult,
)
```

- [ ] **Step 2: train_mlx_model.py — 동일 변경**

`train_mlx()`와 `walk_forward_auc()` 내 `negative_ratio=5` 삭제 (2곳).

- [ ] **Step 3: tune_hyperparams.py — 동일 변경**

`load_dataset()` 내 `negative_ratio=5` 삭제 (1곳).

- [ ] **Step 4: backtester.py — WalkForwardConfig 기본값 변경**

`WalkForwardConfig` 데이터클래스(~line 601)에서:

변경 전:
```python
negative_ratio: int = 5
```

변경 후:
```python
negative_ratio: int = 3
```

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 6: 커밋**

```bash
git add scripts/train_model.py scripts/train_mlx_model.py scripts/tune_hyperparams.py src/backtester.py
git commit -m "refactor(ml): remove hardcoded negative_ratio=5, use dataset_builder defaults"
```

---

### Task 3: 기존 테스트 기본값 정합성 확인 + 수정

**Files:**
- Modify: `tests/test_dataset_builder.py`

- [ ] **Step 1: 기존 테스트가 기본값 변경에 영향받는지 확인**

`tests/test_dataset_builder.py`의 기존 테스트 중 `generate_dataset_vectorized(sample_df)` 처럼 기본값에 의존하는 호출이 있음. 기본값이 완화되었으므로:
- `signal_threshold=2`에서 더 많은 신호가 발생 → 기존 테스트의 assertion이 깨질 수 있음
- `negative_ratio=3`이 기본값이 되므로, 기본 호출 시 HOLD 네거티브가 포함됨

기존 테스트가 실패하면, **원래 의도를 유지하면서** 명시적 파라미터를 추가:

예: `test_returns_dataframe`이 기본 호출로 충분한 결과를 기대한다면 그대로 동작할 가능성이 높음. 하지만 `test_has_required_columns`에서 "source" 컬럼 유무가 달라질 수 있음 (negative_ratio=3 → source 컬럼 존재).

- [ ] **Step 2: 테스트 실행 및 실패 확인**

Run: `pytest tests/test_dataset_builder.py -v`

실패하는 테스트를 파악하고, 각각 수정:
- 기본값에 의존하는 테스트에 명시적 파라미터 추가 (기존 동작 테스트 시 `signal_threshold=3, adx_threshold=25, volume_multiplier=2.5, negative_ratio=0` 명시)
- 또는 새 기본값에서도 assertion이 유효하면 그대로 둠

- [ ] **Step 3: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 4: 커밋**

```bash
git add tests/test_dataset_builder.py
git commit -m "test: update dataset_builder tests for relaxed training defaults"
```

---

### Task 4: CLAUDE.md 업데이트

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md plan history 업데이트**

plan history 테이블에 추가:

```markdown
| 2026-03-21 | `training-threshold-relaxation` (plan) | Completed |
```

- [ ] **Step 2: 최종 전체 테스트**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 3: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: update plan history with training-threshold-relaxation"
```
