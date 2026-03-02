#!/usr/bin/env python3
"""
Optuna를 사용한 LightGBM 하이퍼파라미터 자동 탐색.

사용법:
    python scripts/tune_hyperparams.py                          # 기본 (50 trials, 5폴드)
    python scripts/tune_hyperparams.py --trials 10 --folds 3   # 빠른 테스트
    python scripts/tune_hyperparams.py --data data/combined_15m.parquet --trials 100
    python scripts/tune_hyperparams.py --no-baseline            # 베이스라인 측정 건너뜀

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

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"데이터셋 완성: {len(dataset):,}개 샘플 (양성={pos}, 음성={neg})")
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
    trial: "optuna.Trial | None" = None,
) -> tuple[float, list[float]]:
    """
    Walk-Forward 교차검증으로 평균 AUC를 반환한다.
    trial이 제공되면 각 폴드 후 Optuna에 중간 값을 보고하여 Pruning을 활성화한다.
    """
    n = len(X)
    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    fold_aucs: list[float] = []

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
        fold_aucs.append(float(auc))

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
        n_estimators     = trial.suggest_int("n_estimators", 100, 600)
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        max_depth        = trial.suggest_int("max_depth", 2, 7)

        # 핵심 제약: num_leaves <= 2^max_depth - 1 (leaf-wise 과적합 방지)
        # 360개 수준의 소규모 데이터셋에서 num_leaves가 크면 암기 발생
        max_leaves_upper = min(31, 2 ** max_depth - 1)
        num_leaves       = trial.suggest_int("num_leaves", 7, max(7, max_leaves_upper))

        min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
        subsample         = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)

        # weight_scale: 데이터셋을 1회 캐싱하는 구조이므로
        # time_weight_decay 효과를 sample_weight 스케일로 근사한다.
        weight_scale = trial.suggest_float("weight_scale", 0.5, 2.0)
        w_scaled = (w * weight_scale).astype(np.float32)

        params = {
            "n_estimators":     n_estimators,
            "learning_rate":    learning_rate,
            "max_depth":        max_depth,
            "num_leaves":       num_leaves,
            "min_child_samples": min_child_samples,
            "subsample":        subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha":        reg_alpha,
            "reg_lambda":       reg_lambda,
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
        "n_estimators":     500,
        "learning_rate":    0.05,
        "max_depth":        -1,   # train_model.py는 max_depth 미설정
        "num_leaves":       31,
        "min_child_samples": 15,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.05,
        "reg_lambda":       0.1,
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
    elapsed_s   = int(elapsed_sec % 60)

    sep  = "=" * 64
    dash = "-" * 64

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{sep}")
    print(f"  Optuna 튜닝 완료 | {len(study.trials)} trials "
          f"(완료={len(completed)}, 조기종료={len(pruned)}) | "
          f"소요: {elapsed_min}분 {elapsed_s}초")
    print(sep)
    print(f"  Best AUC  : {best_auc:.4f}  (Trial #{best.number})")
    if baseline_auc > 0:
        sign = "+" if improvement >= 0 else ""
        print(f"  Baseline  : {baseline_auc:.4f}  (현재 train_model.py 고정값)")
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
        arr = np.array(best_folds)
        print(f"    평균: {arr.mean():.4f} ± {arr.std():.4f}")
    if baseline_folds:
        print(dash)
        print("  Baseline 폴드별 AUC:")
        for i, auc in enumerate(baseline_folds, 1):
            print(f"    폴드 {i}: {auc:.4f}")
        arr = np.array(baseline_folds)
        print(f"    평균: {arr.mean():.4f} ± {arr.std():.4f}")
    print(dash)
    print(f"  결과 저장: {output_path}")
    print(f"  다음 단계: python scripts/train_model.py  (파라미터 수동 반영 후)")
    print(sep)


def save_results(
    study: optuna.Study,
    baseline_auc: float,
    baseline_folds: list[float],
    elapsed_sec: float,
    data_path: str,
) -> Path:
    """결과를 JSON 파일로 저장하고 경로를 반환한다."""
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"models/tune_results_{timestamp}.json")
    output_path.parent.mkdir(exist_ok=True)

    best = study.best_trial

    all_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "number":    t.number,
                "auc":       round(t.value, 6),
                "fold_aucs": [round(a, 6) for a in t.user_attrs.get("fold_aucs", [])],
                "params":    {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in t.params.items()
                },
            })

    result = {
        "timestamp":        datetime.now().isoformat(),
        "data_path":        data_path,
        "n_trials_total":   len(study.trials),
        "n_trials_complete": len(all_trials),
        "elapsed_sec":      round(elapsed_sec, 1),
        "baseline": {
            "auc":       round(baseline_auc, 6),
            "fold_aucs": [round(a, 6) for a in baseline_folds],
        },
        "best_trial": {
            "number":    best.number,
            "auc":       round(best.value, 6),
            "fold_aucs": [round(a, 6) for a in best.user_attrs.get("fold_aucs", [])],
            "params":    {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in best.params.items()
            },
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
    parser.add_argument("--data",        default="data/combined_15m.parquet", help="학습 데이터 경로")
    parser.add_argument("--trials",      type=int,   default=50,  help="Optuna trial 수 (기본: 50)")
    parser.add_argument("--folds",       type=int,   default=5,   help="Walk-Forward 폴드 수 (기본: 5)")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="학습 구간 비율 (기본: 0.6)")
    parser.add_argument("--no-baseline", action="store_true",     help="베이스라인 측정 건너뜀")
    args = parser.parse_args()

    # 1. 데이터셋 로드 (1회)
    X, y, w = load_dataset(args.data)

    # 2. 베이스라인 측정
    if args.no_baseline:
        baseline_auc, baseline_folds = 0.0, []
        print("베이스라인 측정 건너뜀 (--no-baseline)\n")
    else:
        baseline_auc, baseline_folds = measure_baseline(X, y, w, args.folds, args.train_ratio)
        print(
            f"베이스라인 AUC: {baseline_auc:.4f} "
            f"(폴드별: {[round(a, 4) for a in baseline_folds]})\n"
        )

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
    print("(trial 완료마다 진행 상황 출력)\n")

    start_time = time.time()

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = study.best_value
            leaves  = trial.params.get("num_leaves", "?")
            depth   = trial.params.get("max_depth", "?")
            print(
                f"  Trial #{trial.number:3d} | AUC={trial.value:.4f} "
                f"| Best={best_so_far:.4f} "
                f"| leaves={leaves} depth={depth}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial #{trial.number:3d} | PRUNED (조기 종료)")

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
