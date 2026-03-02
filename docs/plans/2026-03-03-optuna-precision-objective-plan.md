# Optuna 목적함수를 Precision 중심으로 변경

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 현재 ROC-AUC만 최적화하는 Optuna objective를 **recall >= 0.35 제약 하에서 precision을 최대화**하는 방향으로 변경한다. AUC는 threshold-independent 지표라 실제 운용 시점의 성능(precision)을 반영하지 못하며, 오탐(false positive = 잘못된 진입)이 실제 손실을 발생시키므로 precision 우선 최적화가 필요하다.

**Tech Stack:** Python, LightGBM, Optuna, scikit-learn

---

## 변경 파일
- `scripts/tune_hyperparams.py` (유일한 변경 대상)

---

## 구현 단계

### 1. `_find_best_precision_at_recall` 헬퍼 함수 추가
- `sklearn.metrics.precision_recall_curve`로 recall >= min_recall 조건의 최대 precision과 threshold 반환
- 조건 불만족 시 `(0.0, 0.0, 0.50)` fallback
- train_model.py:277-292와 동일한 로직

### 2. `_walk_forward_cv` 수정
- 기존 반환: `(mean_auc, fold_aucs)` → 신규: `(mean_score, details_dict)`
- `details_dict` 키: `fold_aucs`, `fold_precisions`, `fold_recalls`, `fold_thresholds`, `fold_n_pos`, `mean_auc`, `mean_precision`, `mean_recall`
- **Score 공식**: `precision + auc * 0.001` (AUC는 precision 동률 시 tiebreaker)
- fold 내 양성 < 3개면 해당 fold precision=0.0으로 처리, 평균 계산에서 제외
- 인자 추가: `min_recall: float = 0.35`
- import 추가: `from sklearn.metrics import precision_recall_curve`
- Pruning: 양성 충분한 fold만 report하여 false pruning 방지

### 3. `make_objective` 수정
- `min_recall` 인자 추가 → `_walk_forward_cv`에 전달
- `trial.set_user_attr`로 precision/recall/threshold/n_pos 등 저장
- 반환값: `mean_score` (precision + auc * 0.001)

### 4. `measure_baseline` 수정
- `min_recall` 인자 추가
- 반환값을 `(mean_score, details_dict)` 형태로 변경

### 5. `--min-recall` CLI 인자 추가
- `parser.add_argument("--min-recall", type=float, default=0.35)`
- `make_objective`와 `measure_baseline`에 전달

### 6. `print_report` 수정
- Best Score, Precision, AUC 모두 표시
- 폴드별 AUC + Precision + Recall + Threshold + 양성수 표시
- Baseline과 비교 시 precision 기준 개선폭 표시

### 7. `save_results` 수정
- JSON에 `min_recall_constraint`, precision/recall/threshold 필드 추가
- `best_trial` 내 `score`, `precision`, `recall`, `threshold`, `fold_precisions`, `fold_recalls`, `fold_thresholds`, `fold_n_pos` 추가
- `best_trial.params` 구조는 그대로 유지 (하위호환)

### 8. 비교 로직 및 기타 수정
- line 440: `study.best_value > baseline_auc` → `study.best_value > baseline_score`
- `study_name`: `"lgbm_wf_auc"` → `"lgbm_wf_precision"`
- progress callback: Precision과 AUC 동시 표시
- `n_warmup_steps` 2 → 3 (precision이 AUC보다 노이즈가 크므로)

---

## 검증 방법

```bash
# 기본 실행 (min_recall=0.35)
python scripts/tune_hyperparams.py --trials 10 --folds 3

# min_recall 조절
python scripts/tune_hyperparams.py --trials 10 --min-recall 0.4

# 기존 테스트 통과 확인
bash scripts/run_tests.sh
```

확인 포인트:
- 폴드별 precision/recall/threshold가 리포트에 표시되는지
- recall >= min_recall 제약이 올바르게 동작하는지
- active_lgbm_params.json이 precision 기준으로 갱신되는지
- train_model.py가 새 JSON 포맷을 기존과 동일하게 읽는지
