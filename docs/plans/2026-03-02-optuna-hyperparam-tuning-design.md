# Optuna 하이퍼파라미터 자동 튜닝 설계 문서

**작성일:** 2026-03-02  
**목표:** 봇 운영 로그/학습 결과를 바탕으로 LightGBM 하이퍼파라미터를 Optuna로 자동 탐색하고, 사람이 결과를 확인·승인한 후 재학습에 반영하는 수동 트리거 파이프라인 구축

---

## 배경 및 동기

현재 `train_model.py`의 LightGBM 파라미터는 하드코딩되어 있다. 봇 성능이 저하되거나 데이터가 축적될 때마다 사람이 직접 파라미터를 조정해야 한다. 이를 Optuna로 자동화하되, 과적합 위험을 방지하기 위해 **사람이 결과를 먼저 확인하고 승인하는 구조**를 유지한다.

---

## 구현 범위 (2단계)

### 1단계 (현재): LightGBM 하이퍼파라미터 튜닝
- `scripts/tune_hyperparams.py` 신규 생성
- Optuna + Walk-Forward AUC 목적 함수
- 결과를 JSON + 콘솔 리포트로 출력

### 2단계 (추후): 기술 지표 파라미터 확장
- RSI 임계값, MACD 가중치, Stochastic RSI 임계값, 거래량 배수, 진입 점수 임계값 등을 탐색 공간에 추가
- `dataset_builder.py`의 `_calc_signals()` 파라미터화 필요

---

## 아키텍처

```
scripts/tune_hyperparams.py
├── load_dataset()              ← 데이터 로드 + 벡터화 데이터셋 1회 생성 (캐싱)
├── objective(trial, dataset)   ← Optuna trial 함수
│   ├── trial.suggest_*()       ← 하이퍼파라미터 샘플링
│   ├── num_leaves 상한 강제    ← 2^max_depth - 1 제약
│   └── _walk_forward_cv()      ← Walk-Forward 교차검증 → 평균 AUC 반환
├── run_study()                 ← Optuna study 실행 (TPESampler + MedianPruner)
├── print_report()              ← 콘솔 리포트 출력
└── save_results()              ← JSON 저장 (models/tune_results_YYYYMMDD_HHMMSS.json)
```

---

## 탐색 공간 (소규모 데이터셋 보수적 설계)

| 파라미터 | 범위 | 타입 | 근거 |
|---|---|---|---|
| `n_estimators` | 100 ~ 600 | int | 데이터 적을 때 500+ 트리는 과적합 |
| `learning_rate` | 0.01 ~ 0.2 | float (log) | 낮을수록 일반화 유리 |
| `max_depth` | 2 ~ 7 | int | 트리 깊이 상한 강제 |
| `num_leaves` | 7 ~ min(31, 2^max_depth-1) | int | **핵심**: leaf-wise 과적합 방지 |
| `min_child_samples` | 10 ~ 50 | int | 리프당 최소 샘플 수 |
| `subsample` | 0.5 ~ 1.0 | float | 행 샘플링 |
| `colsample_bytree` | 0.5 ~ 1.0 | float | 열 샘플링 |
| `reg_alpha` | 1e-4 ~ 1.0 | float (log) | L1 정규화 |
| `reg_lambda` | 1e-4 ~ 1.0 | float (log) | L2 정규화 |
| `time_weight_decay` | 0.5 ~ 4.0 | float | 시간 가중치 강도 |

### 핵심 제약: `num_leaves <= 2^max_depth - 1`

LightGBM은 leaf-wise 성장 전략을 사용하므로, `num_leaves`가 `2^max_depth - 1`을 초과하면 `max_depth` 제약이 무의미해진다. trial 내에서 `max_depth`를 먼저 샘플링한 후 `num_leaves` 상한을 동적으로 계산하여 강제한다.

```python
max_depth = trial.suggest_int("max_depth", 2, 7)
max_leaves = min(31, 2 ** max_depth - 1)
num_leaves = trial.suggest_int("num_leaves", 7, max_leaves)
```

---

## 목적 함수: Walk-Forward AUC

기존 `train_model.py`의 `walk_forward_auc()` 로직을 재활용한다. 데이터셋은 study 시작 전 1회만 생성하여 모든 trial이 공유한다 (속도 최적화).

```
전체 데이터셋 (N개 샘플)
├── 폴드 1: 학습[0:60%] → 검증[60%:68%]
├── 폴드 2: 학습[0:68%] → 검증[68%:76%]
├── 폴드 3: 학습[0:76%] → 검증[76%:84%]
├── 폴드 4: 학습[0:84%] → 검증[84%:92%]
└── 폴드 5: 학습[0:92%] → 검증[92%:100%]
목적 함수 = 5폴드 평균 AUC (최대화)
```

### Pruning (조기 종료)

`MedianPruner` 적용: 각 폴드 완료 후 중간 AUC를 Optuna에 보고. 이전 trial들의 중앙값보다 낮으면 나머지 폴드를 건너뛰고 trial 종료. 전체 탐색 시간 ~40% 단축 효과.

---

## 출력 형식

### 콘솔 리포트

```
============================================================
  Optuna 튜닝 완료 | 50 trials | 소요: 28분 42초
============================================================
  Best AUC : 0.6234 (Trial #31)
  Baseline : 0.5891 (현재 train_model.py 고정값)
  개선폭   : +0.0343 (+5.8%)
------------------------------------------------------------
  Best Parameters:
    n_estimators      : 320
    learning_rate     : 0.0412
    max_depth         : 4
    num_leaves        : 15
    min_child_samples : 28
    subsample         : 0.72
    colsample_bytree  : 0.81
    reg_alpha         : 0.0023
    reg_lambda         : 0.0891
    time_weight_decay : 2.31
------------------------------------------------------------
  Walk-Forward 폴드별 AUC:
    폴드 1: 0.6102
    폴드 2: 0.6341
    폴드 3: 0.6198
    폴드 4: 0.6287
    폴드 5: 0.6241
    평균: 0.6234 ± 0.0082
------------------------------------------------------------
  결과 저장: models/tune_results_20260302_143022.json
  다음 단계: python scripts/train_model.py --tuned-params models/tune_results_20260302_143022.json
============================================================
```

### JSON 저장 (`models/tune_results_YYYYMMDD_HHMMSS.json`)

```json
{
  "timestamp": "2026-03-02T14:30:22",
  "n_trials": 50,
  "elapsed_sec": 1722,
  "baseline_auc": 0.5891,
  "best_trial": {
    "number": 31,
    "auc": 0.6234,
    "fold_aucs": [0.6102, 0.6341, 0.6198, 0.6287, 0.6241],
    "params": { ... }
  },
  "all_trials": [ ... ]
}
```

---

## 사용법

```bash
# 기본 실행 (50 trials, 5폴드)
python scripts/tune_hyperparams.py

# 빠른 테스트 (10 trials, 3폴드)
python scripts/tune_hyperparams.py --trials 10 --folds 3

# 데이터 경로 지정
python scripts/tune_hyperparams.py --data data/combined_15m.parquet --trials 100
```

---

## 파일 변경 목록

| 파일 | 변경 | 설명 |
|---|---|---|
| `scripts/tune_hyperparams.py` | **신규 생성** | Optuna 튜닝 스크립트 |
| `requirements.txt` | **수정** | `optuna` 의존성 추가 |
| `README.md` | **수정** | 튜닝 사용법 섹션 추가 |

---

## 향후 확장 (2단계)

`dataset_builder.py`의 `_calc_signals()` 함수를 파라미터화하여 기술 지표 임계값도 탐색 공간에 추가:

```python
# 추가될 탐색 공간 예시
rsi_long_threshold  = trial.suggest_int("rsi_long",  25, 40)
rsi_short_threshold = trial.suggest_int("rsi_short", 60, 75)
vol_surge_mult      = trial.suggest_float("vol_surge_mult", 1.2, 2.5)
entry_threshold     = trial.suggest_int("entry_threshold", 3, 5)
stoch_low           = trial.suggest_int("stoch_low",  10, 30)
stoch_high          = trial.suggest_int("stoch_high", 70, 90)
```
