# HOLD Negative Sampling + Stratified Undersampling Design

## Problem

현재 ML 파이프라인의 학습 데이터가 535개로 매우 적음.
`dataset_builder.py`에서 시그널(LONG/SHORT) 발생 캔들만 라벨링하기 때문.
전체 ~35,000개 캔들 중 98.5%가 HOLD로 버려짐.

## Goal

- HOLD 캔들을 negative sample로 활용하여 학습 데이터 증가
- Train-Serve Skew 방지 (학습/추론 데이터 분포 일치)
- 기존 signal 샘플은 하나도 버리지 않는 계층적 샘플링

## Design

### 1. dataset_builder.py — HOLD Negative Sampling

**변경 위치**: `generate_dataset_vectorized()` (line 360-421)

**현재 로직**:
```python
valid_rows = (
    (signal_arr != "HOLD") &  # ← 시그널 캔들만 선택
    ...
)
```

**변경 로직**:
1. 기존 시그널 캔들(LONG/SHORT) 라벨링은 그대로 유지
2. HOLD 캔들 중 랜덤 샘플링 (시그널 수의 NEGATIVE_RATIO배)
3. HOLD 캔들: label=0, side=랜덤(50% LONG / 50% SHORT), signal_strength=0
4. `source` 컬럼 추가: "signal" | "hold_negative" (계층적 샘플링에 사용)

**파라미터**:
```python
NEGATIVE_RATIO = 5    # 시그널 대비 HOLD 샘플 비율
RANDOM_SEED = 42      # 재현성
```

**예상 데이터량**:
- 시그널: ~535개 (Win ~200, Loss ~335)
- HOLD negative: ~2,675개
- 총 학습 데이터: ~3,210개

### 2. train_model.py — Stratified Undersampling

**변경 위치**: `train()` 함수 내 언더샘플링 블록 (line 241-257)

**현재 로직**: 양성:음성 = 1:1 블라인드 언더샘플링
```python
if len(neg_idx) > len(pos_idx):
    neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
```

**변경 로직**: 계층적 3-class 샘플링
```python
# 1. Signal 샘플(source="signal") 전수 유지 (Win + Loss 모두)
# 2. HOLD negative(source="hold_negative")에서만 샘플링
#    → 양성(Win) 수와 동일한 수만큼 샘플링
# 최종: Win ~200 + Signal Loss ~335 + HOLD ~200 = ~735개
```

**효과**:
- Signal 샘플 보존율: 100% (Win/Loss 모두)
- HOLD negative: 적절한 양만 추가
- Train-Serve Skew 없음 (추론 시 signal_strength ≥ 3에서만 호출)

### 3. 런타임 (변경 없음)

- `bot.py`: 시그널 발생 시에만 ML 필터 호출 (기존 동일)
- `ml_filter.py`: `should_enter()` 그대로
- `ml_features.py`: `FEATURE_COLS` 그대로
- `label_builder.py`: 기존 SL/TP 룩어헤드 로직 그대로

## Test Cases

### 필수 테스트
1. **HOLD negative label 검증**: HOLD negative 샘플의 label이 전부 0인지 확인
2. **Signal 보존 검증**: 계층적 샘플링 후 source="signal" 샘플이 하나도 버려지지 않았는지 확인

### 기존 테스트 호환성
- 기존 dataset_builder 관련 테스트가 깨지지 않도록 보장

## File Changes

| File | Change |
|------|--------|
| `src/dataset_builder.py` | HOLD negative sampling, source 컬럼 추가 |
| `scripts/train_model.py` | 계층적 샘플링으로 교체 |
| `tests/test_dataset_builder.py` (or equivalent) | 2개 테스트 케이스 추가 |
