# Optuna 하이퍼파라미터 자동 튜닝 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `scripts/tune_hyperparams.py`를 신규 생성하여 Optuna + Walk-Forward AUC 기반 LightGBM 하이퍼파라미터 자동 탐색 파이프라인을 구축한다.

**Architecture:** 데이터셋을 study 시작 전 1회만 생성해 캐싱하고, 각 Optuna trial에서 LightGBM 파라미터를 샘플링 → Walk-Forward 5폴드 AUC를 목적 함수로 최대화한다. `num_leaves <= 2^max_depth - 1` 제약을 코드 레벨에서 강제하여 소규모 데이터셋 과적합을 방지한다. 결과는 콘솔 리포트 + JSON 파일로 출력한다.

**Tech Stack:** Python 3.11+, optuna, lightgbm, numpy, pandas, scikit-learn (기존 의존성 재활용)

**설계 문서:** `docs/plans/2026-03-02-optuna-hyperparam-tuning-design.md`

---

## Task 1: optuna 의존성 추가

**Files:**
- Modify: `requirements.txt`

**Step 1: requirements.txt에 optuna 추가**

```
optuna>=3.6.0
```

`requirements.txt` 파일 끝에 추가한다.

**Step 2: 설치 확인 (로컬)**

```bash
pip install optuna
python -c "import optuna; print(optuna.__version__)"
```

Expected: 버전 번호 출력 (예: `3.6.0`)

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add optuna dependency for hyperparameter tuning"
```

---

## Task 2: `scripts/tune_hyperparams.py` 핵심 구조 생성

**Files:**
- Create: `scripts/tune_hyperparams.py`

**Step 1: 파일 생성 — 전체 코드**

아래 코드를 `scripts/tune_hyperparams.py`로 저장한다.

```python
"""
Optuna를 사용한 LightGBM 하이퍼파라미터 자동 탐색.

사용법:
    python scripts/tune_hyperparams.py                          # 기본 (50 trials, 5폴드)
    python scripts/tune_hyperparams.py --trials 10 --folds 3   # 빠른 테스트
    python scripts/tune_hyperparams.py --data data/combined_15m.parquet --trials 100

결과:
    - 콘솔: Best Params + Walk-Forward 리포트
    - JSON: models/tune_results_YYYYMMDD_HHMMSS.json
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score

from src.ml_features import FEATURE_COLS
from src.dataset_builder import generate_dataset_vectorized


# ──────────────────────────────────────────────
# 데이터 로드 및 데이터셋 생성 (1회 캐싱)
# ──────────────────────────────────────────────

def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    parquet 로드 → 벡터화 데이터셋 생성 → (X, y, w) numpy 배열 반환.
    study 시작 전 1회만 호출하여 모든 trial이 공유한다.
    """
    print(f"데이터 로드: {data_path}")
    df_raw = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df_raw):,}, 컬럼: {list(df_raw.columns)}")

    base_cols = ["open", "high", "low", "close", "volume"]
    btc_df = eth_df = None

    if "close_btc" in df_raw.columns:
        btc_df = df_raw[[c + "_btc" for c in base_cols]].copy()
        btc_df.columns = base_cols
        print("BTC 피처 활성화")

    if "close_eth" in df_raw.columns:
        eth_df = df_raw[[c + "_eth" for c in base_cols]].copy()
        eth_df.columns = base_cols
        print("ETH 피처 활성화")

    df = df_raw[base_cols].copy()

    print("\n데이터셋 생성 중 (1회만 실행)...")
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=0.0)

    if dataset.empty or "label" not in dataset.columns:
        raise ValueError("데이터셋 생성 실패: 샘플 0개")

    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[actual_feature_cols].values.astype(np.float32)
    y = dataset["label"].values.astype(np.int8)
    w = dataset["sample_weight"].values.astype(np.float32)

    pos = y.sum()
    neg = (y == 0).sum()
    print(f"데이터셋 완성: {len(dataset):,}개 샘플 (양성={pos:.0f}, 음성={neg:.0f})")
    print(f"사용 피처: {len(actual_feature_cols)}개\n")

    return X, y, w


# ──────────────────────────────────────────────
# Walk-Forward 교차검증
# ──────────────────────────────────────────────

def _walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    params: dict,
    n_splits: int,
    train_ratio: float,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[float]]:
    """
    Walk-Forward 교차검증으로 평균 AUC를 반환한다.
    trial이 제공되면 각 폴드 후 Optuna에 중간 값을 보고하여 Pruning을 활성화한다.
    """
    n = len(X)
    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    fold_aucs = []

    for fold_idx in range(n_splits):
        tr_end = train_end_start + fold_idx * step
        val_end = tr_end + step
        if val_end > n:
            break

        X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
        X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]

        # 클래스 불균형 처리: 언더샘플링 (시간 순서 유지)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        if len(neg_idx) > len(pos_idx) and len(pos_idx) > 0:
            rng = np.random.default_rng(42)
            neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        bal_idx = np.sort(np.concatenate([pos_idx, neg_idx]))

        if len(bal_idx) < 20 or len(np.unique(y_val)) < 2:
            fold_aucs.append(0.5)
            continue

        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr[bal_idx], y_tr[bal_idx], sample_weight=w_tr[bal_idx])

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5
        fold_aucs.append(auc)

        # Optuna Pruning: 중간 값 보고
        if trial is not None:
            trial.report(float(np.mean(fold_aucs)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    return mean_auc, fold_aucs


# ──────────────────────────────────────────────
# Optuna 목적 함수
# ──────────────────────────────────────────────

def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_splits: int,
    train_ratio: float,
):
    """클로저로 데이터셋을 캡처한 목적 함수를 반환한다."""

    def objective(trial: optuna.Trial) -> float:
        # ── 하이퍼파라미터 샘플링 ──
        n_estimators = trial.suggest_int("n_estimators", 100, 600)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 7)

        # 핵심 제약: num_leaves <= 2^max_depth - 1 (leaf-wise 과적합 방지)
        max_leaves_upper = min(31, 2 ** max_depth - 1)
        num_leaves = trial.suggest_int("num_leaves", 7, max(7, max_leaves_upper))

        min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)

        # time_weight_decay는 데이터셋 생성 시 적용되어야 하지만,
        # 데이터셋을 1회 캐싱하는 구조이므로 LightGBM sample_weight 스케일로 근사한다.
        # 실제 decay 효과는 w 배열에 이미 반영되어 있으므로 스케일 파라미터로 활용한다.
        weight_scale = trial.suggest_float("weight_scale", 0.5, 2.0)
        w_scaled = (w * weight_scale).astype(np.float32)

        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }

        mean_auc, fold_aucs = _walk_forward_cv(
            X, y, w_scaled, params,
            n_splits=n_splits,
            train_ratio=train_ratio,
            trial=trial,
        )

        # 폴드별 AUC를 user_attrs에 저장 (결과 리포트용)
        trial.set_user_attr("fold_aucs", fold_aucs)

        return mean_auc

    return objective


# ──────────────────────────────────────────────
# 베이스라인 AUC 측정 (현재 고정 파라미터)
# ──────────────────────────────────────────────

def measure_baseline(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_splits: int,
    train_ratio: float,
) -> tuple[float, list[float]]:
    """train_model.py의 현재 고정 파라미터로 베이스라인 AUC를 측정한다."""
    baseline_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "max_depth": -1,  # 현재 train_model.py는 max_depth 미설정
    }
    print("베이스라인 측정 중 (현재 train_model.py 고정 파라미터)...")
    return _walk_forward_cv(X, y, w, baseline_params, n_splits=n_splits, train_ratio=train_ratio)


# ──────────────────────────────────────────────
# 결과 출력 및 저장
# ──────────────────────────────────────────────

def print_report(
    study: optuna.Study,
    baseline_auc: float,
    baseline_folds: list[float],
    elapsed_sec: float,
    output_path: Path,
) -> None:
    """콘솔에 최종 리포트를 출력한다."""
    best = study.best_trial
    best_auc = best.value
    best_folds = best.user_attrs.get("fold_aucs", [])
    improvement = best_auc - baseline_auc
    improvement_pct = (improvement / baseline_auc * 100) if baseline_auc > 0 else 0.0

    elapsed_min = int(elapsed_sec // 60)
    elapsed_s = int(elapsed_sec % 60)

    sep = "=" * 62
    dash = "-" * 62

    print(f"\n{sep}")
    print(f"  Optuna 튜닝 완료 | {len(study.trials)} trials | 소요: {elapsed_min}분 {elapsed_s}초")
    print(sep)
    print(f"  Best AUC  : {best_auc:.4f}  (Trial #{best.number})")
    print(f"  Baseline  : {baseline_auc:.4f}  (현재 train_model.py 고정값)")
    sign = "+" if improvement >= 0 else ""
    print(f"  개선폭    : {sign}{improvement:.4f} ({sign}{improvement_pct:.1f}%)")
    print(dash)
    print("  Best Parameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k:<22}: {v:.6f}")
        else:
            print(f"    {k:<22}: {v}")
    print(dash)
    print("  Walk-Forward 폴드별 AUC (Best Trial):")
    for i, auc in enumerate(best_folds, 1):
        print(f"    폴드 {i}: {auc:.4f}")
    if best_folds:
        print(f"    평균: {np.mean(best_folds):.4f} ± {np.std(best_folds):.4f}")
    print(dash)
    print("  Baseline 폴드별 AUC:")
    for i, auc in enumerate(baseline_folds, 1):
        print(f"    폴드 {i}: {auc:.4f}")
    if baseline_folds:
        print(f"    평균: {np.mean(baseline_folds):.4f} ± {np.std(baseline_folds):.4f}")
    print(dash)
    print(f"  결과 저장: {output_path}")
    print(f"  다음 단계: python scripts/train_model.py --tuned-params {output_path}")
    print(sep)


def save_results(
    study: optuna.Study,
    baseline_auc: float,
    baseline_folds: list[float],
    elapsed_sec: float,
    data_path: str,
) -> Path:
    """결과를 JSON 파일로 저장하고 경로를 반환한다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"models/tune_results_{timestamp}.json")
    output_path.parent.mkdir(exist_ok=True)

    best = study.best_trial

    all_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "number": t.number,
                "auc": round(t.value, 6),
                "fold_aucs": [round(a, 6) for a in t.user_attrs.get("fold_aucs", [])],
                "params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.params.items()},
            })

    result = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(all_trials),
        "elapsed_sec": round(elapsed_sec, 1),
        "baseline": {
            "auc": round(baseline_auc, 6),
            "fold_aucs": [round(a, 6) for a in baseline_folds],
        },
        "best_trial": {
            "number": best.number,
            "auc": round(best.value, 6),
            "fold_aucs": [round(a, 6) for a in best.user_attrs.get("fold_aucs", [])],
            "params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.params.items()},
        },
        "all_trials": all_trials,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna LightGBM 하이퍼파라미터 튜닝")
    parser.add_argument("--data",   default="data/combined_15m.parquet", help="학습 데이터 경로")
    parser.add_argument("--trials", type=int, default=50,  help="Optuna trial 수 (기본: 50)")
    parser.add_argument("--folds",  type=int, default=5,   help="Walk-Forward 폴드 수 (기본: 5)")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="학습 구간 비율 (기본: 0.6)")
    parser.add_argument("--no-baseline", action="store_true", help="베이스라인 측정 건너뜀")
    args = parser.parse_args()

    # 1. 데이터셋 로드 (1회)
    X, y, w = load_dataset(args.data)

    # 2. 베이스라인 측정
    if args.no_baseline:
        baseline_auc, baseline_folds = 0.0, []
        print("베이스라인 측정 건너뜀 (--no-baseline)")
    else:
        baseline_auc, baseline_folds = measure_baseline(X, y, w, args.folds, args.train_ratio)
        print(f"베이스라인 AUC: {baseline_auc:.4f} (폴드별: {[round(a,4) for a in baseline_folds]})\n")

    # 3. Optuna study 실행
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lgbm_wf_auc",
    )

    objective = make_objective(X, y, w, n_splits=args.folds, train_ratio=args.train_ratio)

    print(f"Optuna 탐색 시작: {args.trials} trials, {args.folds}폴드 Walk-Forward")
    print("(진행 상황은 trial 완료마다 출력됩니다)\n")

    start_time = time.time()

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = study.best_value
            print(
                f"  Trial #{trial.number:3d} | AUC={trial.value:.4f} "
                f"| Best={best_so_far:.4f} "
                f"| {trial.params.get('num_leaves', '?')}leaves "
                f"depth={trial.params.get('max_depth', '?')}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial #{trial.number:3d} | PRUNED")

    study.optimize(
        objective,
        n_trials=args.trials,
        callbacks=[_progress_callback],
        show_progress_bar=False,
    )

    elapsed = time.time() - start_time

    # 4. 결과 저장 및 출력
    output_path = save_results(study, baseline_auc, baseline_folds, elapsed, args.data)
    print_report(study, baseline_auc, baseline_folds, elapsed, output_path)


if __name__ == "__main__":
    main()
```

**Step 2: 문법 오류 확인**

```bash
cd /path/to/cointrader
python -c "import ast; ast.parse(open('scripts/tune_hyperparams.py').read()); print('문법 OK')"
```

Expected: `문법 OK`

**Step 3: Commit**

```bash
git add scripts/tune_hyperparams.py
git commit -m "feat: add Optuna Walk-Forward AUC hyperparameter tuning script"
```

---

## Task 3: 동작 검증 (빠른 테스트)

**Files:**
- Read: `scripts/tune_hyperparams.py`

**Step 1: 빠른 테스트 실행 (10 trials, 3폴드)**

```bash
python scripts/tune_hyperparams.py --trials 10 --folds 3 --no-baseline
```

Expected:
- 오류 없이 10 trials 완료
- `models/tune_results_YYYYMMDD_HHMMSS.json` 생성
- 콘솔에 Best Params 출력

**Step 2: JSON 결과 확인**

```bash
cat models/tune_results_*.json | python -m json.tool | head -40
```

Expected: `best_trial.auc`, `best_trial.params` 등 구조 확인

**Step 3: Commit**

```bash
git add models/tune_results_*.json
git commit -m "test: verify Optuna tuning pipeline with 10 trials"
```

---

## Task 4: README.md 업데이트

**Files:**
- Modify: `README.md`

**Step 1: ML 모델 학습 섹션에 튜닝 사용법 추가**

`README.md`의 `## ML 모델 학습` 섹션 아래에 다음 내용을 추가한다:

```markdown
### 하이퍼파라미터 자동 튜닝 (Optuna)

봇 성능이 저하되거나 데이터가 충분히 축적되었을 때 Optuna로 최적 파라미터를 탐색합니다.
결과를 확인하고 직접 승인한 후 재학습에 반영하는 **수동 트리거** 방식입니다.

```bash
# 기본 실행 (50 trials, 5폴드 Walk-Forward, ~30분)
python scripts/tune_hyperparams.py

# 빠른 테스트 (10 trials, 3폴드, ~5분)
python scripts/tune_hyperparams.py --trials 10 --folds 3

# 결과 확인 후 승인하면 재학습
python scripts/train_model.py
```

결과는 `models/tune_results_YYYYMMDD_HHMMSS.json`에 저장됩니다.
Best Params와 베이스라인 대비 개선폭을 확인하고 직접 판단하세요.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Optuna hyperparameter tuning usage to README"
```

---

## 검증 체크리스트

- [ ] `python -c "import optuna"` 오류 없음
- [ ] `python scripts/tune_hyperparams.py --trials 10 --folds 3 --no-baseline` 오류 없이 완료
- [ ] `models/tune_results_*.json` 파일 생성 확인
- [ ] JSON에 `best_trial.params`, `best_trial.fold_aucs` 포함 확인
- [ ] 콘솔 리포트에 Best AUC, 폴드별 AUC, 파라미터 출력 확인
- [ ] `num_leaves <= 2^max_depth - 1` 제약이 모든 trial에서 지켜지는지 JSON으로 확인

---

## 향후 확장 (2단계 — 별도 플랜)

파이프라인 안정화 후 `dataset_builder.py`의 `_calc_signals()` 함수를 파라미터화하여 기술 지표 임계값(RSI, Stochastic RSI, 거래량 배수, 진입 점수 임계값)을 탐색 공간에 추가한다.
