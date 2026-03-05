#!/usr/bin/env python3
"""
Optuna를 사용한 LightGBM 하이퍼파라미터 자동 탐색.

사용법:
    python scripts/tune_hyperparams.py                          # 기본 (50 trials, 5폴드)
    python scripts/tune_hyperparams.py --trials 10 --folds 3   # 빠른 테스트
    python scripts/tune_hyperparams.py --data data/combined_15m.parquet --trials 100
    python scripts/tune_hyperparams.py --no-baseline            # 베이스라인 측정 건너뜀
    python scripts/tune_hyperparams.py --min-recall 0.4         # 최소 재현율 제약 조정

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
from sklearn.metrics import roc_auc_score, precision_recall_curve

from src.ml_features import FEATURE_COLS
from src.dataset_builder import generate_dataset_vectorized, stratified_undersample


# ──────────────────────────────────────────────
# 데이터 로드 및 데이터셋 생성 (1회 캐싱)
# ──────────────────────────────────────────────

def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=0.0, negative_ratio=5)

    if dataset.empty or "label" not in dataset.columns:
        raise ValueError("데이터셋 생성 실패: 샘플 0개")

    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[actual_feature_cols].values.astype(np.float32)
    y = dataset["label"].values.astype(np.int8)
    w = dataset["sample_weight"].values.astype(np.float32)
    source = dataset["source"].values if "source" in dataset.columns else np.full(len(dataset), "signal")

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"데이터셋 완성: {len(dataset):,}개 샘플 (양성={pos}, 음성={neg})")
    print(f"사용 피처: {len(actual_feature_cols)}개\n")

    return X, y, w, source


# ──────────────────────────────────────────────
# Precision 헬퍼
# ──────────────────────────────────────────────

def _find_best_precision_at_recall(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_recall: float = 0.35,
) -> tuple[float, float, float]:
    """
    precision_recall_curve에서 recall >= min_recall 조건을 만족하는
    최대 precision과 해당 threshold를 반환한다.

    Returns:
        (best_precision, best_recall, best_threshold)
        조건 불만족 시 (0.0, 0.0, 0.50)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    valid_idx = np.where(recalls >= min_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        return (
            float(precisions[best_idx]),
            float(recalls[best_idx]),
            float(thresholds[best_idx]),
        )
    return (0.0, 0.0, 0.50)


# ──────────────────────────────────────────────
# Walk-Forward 교차검증
# ──────────────────────────────────────────────

def _walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    params: dict,
    n_splits: int,
    train_ratio: float,
    min_recall: float = 0.35,
    trial: "optuna.Trial | None" = None,
) -> tuple[float, dict]:
    """
    Walk-Forward 교차검증으로 precision 기반 복합 점수를 반환한다.
    Score = mean_precision + mean_auc * 0.001 (AUC는 tiebreaker)

    trial이 제공되면 각 폴드 후 Optuna에 중간 값을 보고하여 Pruning을 활성화한다.

    Returns:
        (mean_score, details) where details contains per-fold metrics.
    """
    n = len(X)
    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    fold_aucs: list[float] = []
    fold_precisions: list[float] = []
    fold_recalls: list[float] = []
    fold_thresholds: list[float] = []
    fold_n_pos: list[int] = []
    scores_so_far: list[float] = []

    for fold_idx in range(n_splits):
        tr_end = train_end_start + fold_idx * step
        val_end = tr_end + step
        if val_end > n:
            break

        X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
        X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]

        # 계층적 샘플링: signal 전수 유지, HOLD negative만 양성 수 만큼
        source_tr = source[:tr_end]
        bal_idx = stratified_undersample(y_tr, source_tr, seed=42)

        n_pos = int(y_val.sum())

        if len(bal_idx) < 20 or len(np.unique(y_val)) < 2:
            fold_aucs.append(0.5)
            fold_precisions.append(0.0)
            fold_recalls.append(0.0)
            fold_thresholds.append(0.50)
            fold_n_pos.append(n_pos)
            continue

        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr[bal_idx], y_tr[bal_idx], sample_weight=w_tr[bal_idx])

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5
        fold_aucs.append(float(auc))

        # Precision at recall-constrained threshold
        if n_pos >= 3:
            prec, rec, thr = _find_best_precision_at_recall(y_val, proba, min_recall)
        else:
            prec, rec, thr = 0.0, 0.0, 0.50

        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_thresholds.append(thr)
        fold_n_pos.append(n_pos)

        # Pruning: 양성 충분한 fold의 score만 보고
        score = prec + auc * 0.001
        scores_so_far.append(score)
        if trial is not None and n_pos >= 3:
            valid_scores = [s for s, np_ in zip(scores_so_far, fold_n_pos) if np_ >= 3]
            if valid_scores:
                trial.report(float(np.mean(valid_scores)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    # 양성 충분한 fold만으로 precision 평균 계산
    valid_precs = [p for p, np_ in zip(fold_precisions, fold_n_pos) if np_ >= 3]
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    mean_prec = float(np.mean(valid_precs)) if valid_precs else 0.0
    valid_recs = [r for r, np_ in zip(fold_recalls, fold_n_pos) if np_ >= 3]
    mean_rec = float(np.mean(valid_recs)) if valid_recs else 0.0
    mean_score = mean_prec + mean_auc * 0.001

    details = {
        "fold_aucs":       fold_aucs,
        "fold_precisions": fold_precisions,
        "fold_recalls":    fold_recalls,
        "fold_thresholds": fold_thresholds,
        "fold_n_pos":      fold_n_pos,
        "mean_auc":        mean_auc,
        "mean_precision":  mean_prec,
        "mean_recall":     mean_rec,
    }

    return mean_score, details


# ──────────────────────────────────────────────
# Optuna 목적 함수
# ──────────────────────────────────────────────

def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    n_splits: int,
    train_ratio: float,
    min_recall: float = 0.35,
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

        mean_score, details = _walk_forward_cv(
            X, y, w_scaled, source, params,
            n_splits=n_splits,
            train_ratio=train_ratio,
            min_recall=min_recall,
            trial=trial,
        )

        # 폴드별 상세 메트릭을 user_attrs에 저장 (결과 리포트용)
        trial.set_user_attr("fold_aucs", details["fold_aucs"])
        trial.set_user_attr("fold_precisions", details["fold_precisions"])
        trial.set_user_attr("fold_recalls", details["fold_recalls"])
        trial.set_user_attr("fold_thresholds", details["fold_thresholds"])
        trial.set_user_attr("fold_n_pos", details["fold_n_pos"])
        trial.set_user_attr("mean_auc", details["mean_auc"])
        trial.set_user_attr("mean_precision", details["mean_precision"])
        trial.set_user_attr("mean_recall", details["mean_recall"])

        return mean_score

    return objective


# ──────────────────────────────────────────────
# 베이스라인 측정 (현재 고정 파라미터)
# ──────────────────────────────────────────────

def measure_baseline(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    n_splits: int,
    train_ratio: float,
    min_recall: float = 0.35,
    active_params_path: "Path | None" = None,
) -> tuple[float, dict]:
    """현재 실전 파라미터(active 파일 또는 하드코딩 기본값)로 베이스라인을 측정한다."""
    active_path = active_params_path or Path("models/active_lgbm_params.json")

    if active_path.exists():
        with open(active_path, "r", encoding="utf-8") as f:
            tune_data = json.load(f)
        best_params = dict(tune_data["best_trial"]["params"])
        best_params.pop("weight_scale", None)
        baseline_params = best_params
        print(f"베이스라인 측정 중 (active 파일: {active_path})...")
    else:
        baseline_params = {
            "n_estimators":      434,
            "learning_rate":     0.123659,
            "max_depth":         6,
            "num_leaves":        14,
            "min_child_samples": 10,
            "subsample":         0.929062,
            "colsample_bytree":  0.946330,
            "reg_alpha":         0.573971,
            "reg_lambda":        0.000157,
        }
        print("베이스라인 측정 중 (active 파일 없음 → 코드 내 기본 파라미터)...")

    return _walk_forward_cv(
        X, y, w, source, baseline_params,
        n_splits=n_splits, train_ratio=train_ratio,
        min_recall=min_recall,
    )


# ──────────────────────────────────────────────
# 결과 출력 및 저장
# ──────────────────────────────────────────────

def print_report(
    study: optuna.Study,
    baseline_score: float,
    baseline_details: dict,
    elapsed_sec: float,
    output_path: Path,
    min_recall: float,
) -> None:
    """콘솔에 최종 리포트를 출력한다."""
    best = study.best_trial
    best_score = best.value
    best_prec = best.user_attrs.get("mean_precision", 0.0)
    best_auc = best.user_attrs.get("mean_auc", 0.0)
    best_rec = best.user_attrs.get("mean_recall", 0.0)

    baseline_prec = baseline_details.get("mean_precision", 0.0)
    baseline_auc = baseline_details.get("mean_auc", 0.0)

    prec_improvement = best_prec - baseline_prec
    prec_improvement_pct = (prec_improvement / baseline_prec * 100) if baseline_prec > 0 else 0.0

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
    print(f"  최적화 지표: Precision (recall >= {min_recall} 제약)")
    print(f"  Best Prec : {best_prec:.4f}  (Trial #{best.number})")
    print(f"  Best AUC  : {best_auc:.4f}")
    print(f"  Best Recall: {best_rec:.4f}")
    if baseline_score > 0:
        sign = "+" if prec_improvement >= 0 else ""
        print(dash)
        print(f"  Baseline  : Prec={baseline_prec:.4f}, AUC={baseline_auc:.4f}")
        print(f"  개선폭    : Precision {sign}{prec_improvement:.4f} ({sign}{prec_improvement_pct:.1f}%)")
    print(dash)
    print("  Best Parameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k:<22}: {v:.6f}")
        else:
            print(f"    {k:<22}: {v}")
    print(dash)

    # 폴드별 상세
    fold_aucs = best.user_attrs.get("fold_aucs", [])
    fold_precs = best.user_attrs.get("fold_precisions", [])
    fold_recs = best.user_attrs.get("fold_recalls", [])
    fold_thrs = best.user_attrs.get("fold_thresholds", [])
    fold_npos = best.user_attrs.get("fold_n_pos", [])

    print("  Walk-Forward 폴드별 상세 (Best Trial):")
    for i, (auc, prec, rec, thr, npos) in enumerate(
        zip(fold_aucs, fold_precs, fold_recs, fold_thrs, fold_npos), 1
    ):
        print(f"    폴드 {i}: AUC={auc:.4f} Prec={prec:.3f} Rec={rec:.3f} Thr={thr:.3f} (양성={npos})")
    if fold_precs:
        valid_precs = [p for p, np_ in zip(fold_precs, fold_npos) if np_ >= 3]
        if valid_precs:
            arr_p = np.array(valid_precs)
            print(f"    평균 Precision: {arr_p.mean():.4f} ± {arr_p.std():.4f}")
    if fold_aucs:
        arr_a = np.array(fold_aucs)
        print(f"    평균 AUC: {arr_a.mean():.4f} ± {arr_a.std():.4f}")

    # 베이스라인 폴드별
    bl_folds = baseline_details.get("fold_aucs", [])
    bl_precs = baseline_details.get("fold_precisions", [])
    bl_recs = baseline_details.get("fold_recalls", [])
    bl_thrs = baseline_details.get("fold_thresholds", [])
    bl_npos = baseline_details.get("fold_n_pos", [])
    if bl_folds:
        print(dash)
        print("  Baseline 폴드별 상세:")
        for i, (auc, prec, rec, thr, npos) in enumerate(
            zip(bl_folds, bl_precs, bl_recs, bl_thrs, bl_npos), 1
        ):
            print(f"    폴드 {i}: AUC={auc:.4f} Prec={prec:.3f} Rec={rec:.3f} Thr={thr:.3f} (양성={npos})")

    print(dash)
    print(f"  결과 저장: {output_path}")
    print(f"  다음 단계: python scripts/train_model.py  (파라미터 수동 반영 후)")
    print(sep)


def save_results(
    study: optuna.Study,
    baseline_score: float,
    baseline_details: dict,
    elapsed_sec: float,
    data_path: str,
    min_recall: float,
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
                "score":     round(t.value, 6),
                "auc":       round(t.user_attrs.get("mean_auc", 0.0), 6),
                "precision": round(t.user_attrs.get("mean_precision", 0.0), 6),
                "recall":    round(t.user_attrs.get("mean_recall", 0.0), 6),
                "fold_aucs": [round(a, 6) for a in t.user_attrs.get("fold_aucs", [])],
                "fold_precisions": [round(p, 6) for p in t.user_attrs.get("fold_precisions", [])],
                "params":    {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in t.params.items()
                },
            })

    result = {
        "timestamp":          datetime.now().isoformat(),
        "data_path":          data_path,
        "min_recall_constraint": min_recall,
        "n_trials_total":     len(study.trials),
        "n_trials_complete":  len(all_trials),
        "elapsed_sec":        round(elapsed_sec, 1),
        "baseline": {
            "score":           round(baseline_score, 6),
            "auc":             round(baseline_details.get("mean_auc", 0.0), 6),
            "precision":       round(baseline_details.get("mean_precision", 0.0), 6),
            "recall":          round(baseline_details.get("mean_recall", 0.0), 6),
            "fold_aucs":       [round(a, 6) for a in baseline_details.get("fold_aucs", [])],
            "fold_precisions": [round(p, 6) for p in baseline_details.get("fold_precisions", [])],
            "fold_recalls":    [round(r, 6) for r in baseline_details.get("fold_recalls", [])],
            "fold_thresholds": [round(t, 6) for t in baseline_details.get("fold_thresholds", [])],
        },
        "best_trial": {
            "number":          best.number,
            "score":           round(best.value, 6),
            "auc":             round(best.user_attrs.get("mean_auc", 0.0), 6),
            "precision":       round(best.user_attrs.get("mean_precision", 0.0), 6),
            "recall":          round(best.user_attrs.get("mean_recall", 0.0), 6),
            "fold_aucs":       [round(a, 6) for a in best.user_attrs.get("fold_aucs", [])],
            "fold_precisions": [round(p, 6) for p in best.user_attrs.get("fold_precisions", [])],
            "fold_recalls":    [round(r, 6) for r in best.user_attrs.get("fold_recalls", [])],
            "fold_thresholds": [round(t, 6) for t in best.user_attrs.get("fold_thresholds", [])],
            "fold_n_pos":      best.user_attrs.get("fold_n_pos", []),
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
    parser.add_argument("--data",        default=None, help="학습 데이터 경로")
    parser.add_argument("--symbol",      type=str, default=None,
                        help="튜닝 대상 심볼 (예: TRXUSDT). data/{symbol}/ 에서 데이터 로드, models/{symbol}/ 에 저장")
    parser.add_argument("--trials",      type=int,   default=50,  help="Optuna trial 수 (기본: 50)")
    parser.add_argument("--folds",       type=int,   default=5,   help="Walk-Forward 폴드 수 (기본: 5)")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="학습 구간 비율 (기본: 0.6)")
    parser.add_argument("--min-recall",  type=float, default=0.35, help="최소 재현율 제약 (기본: 0.35)")
    parser.add_argument("--no-baseline", action="store_true",     help="베이스라인 측정 건너뜀")
    args = parser.parse_args()

    # --symbol 모드: 심볼별 디렉토리 경로 자동 결정
    if args.symbol:
        sym_lower = args.symbol.lower()
        if args.data is None:
            args.data = f"data/{sym_lower}/combined_15m.parquet"
    elif args.data is None:
        args.data = "data/combined_15m.parquet"

    # 1. 데이터셋 로드 (1회)
    X, y, w, source = load_dataset(args.data)

    # 2. 베이스라인 측정
    if args.symbol:
        sym_lower = args.symbol.lower()
        _active_params_path = Path(f"models/{sym_lower}/active_lgbm_params.json")
    else:
        _active_params_path = None

    if args.no_baseline:
        baseline_score, baseline_details = 0.0, {}
        print("베이스라인 측정 건너뜀 (--no-baseline)\n")
    else:
        baseline_score, baseline_details = measure_baseline(
            X, y, w, source, args.folds, args.train_ratio, args.min_recall,
            active_params_path=_active_params_path,
        )
        bl_prec = baseline_details.get("mean_precision", 0.0)
        bl_auc = baseline_details.get("mean_auc", 0.0)
        bl_rec = baseline_details.get("mean_recall", 0.0)
        print(
            f"베이스라인: Prec={bl_prec:.4f}, AUC={bl_auc:.4f}, Recall={bl_rec:.4f} "
            f"(recall >= {args.min_recall} 제약)\n"
        )

    # 3. Optuna study 실행
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lgbm_wf_precision",
    )

    objective = make_objective(
        X, y, w, source,
        n_splits=args.folds,
        train_ratio=args.train_ratio,
        min_recall=args.min_recall,
    )

    print(f"Optuna 탐색 시작: {args.trials} trials, {args.folds}폴드 Walk-Forward")
    print(f"최적화 지표: Precision (recall >= {args.min_recall} 제약)")
    print("(trial 완료마다 진행 상황 출력)\n")

    start_time = time.time()

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = study.best_value
            prec = trial.user_attrs.get("mean_precision", 0.0)
            auc  = trial.user_attrs.get("mean_auc", 0.0)
            print(
                f"  Trial #{trial.number:3d} | Prec={prec:.4f} AUC={auc:.4f} "
                f"| Best={best_so_far:.4f} "
                f"| leaves={trial.params.get('num_leaves', '?')} "
                f"depth={trial.params.get('max_depth', '?')}"
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
    import shutil
    output_path = save_results(
        study, baseline_score, baseline_details, elapsed, args.data, args.min_recall,
    )
    # --symbol 모드: 결과 파일을 심볼별 디렉토리로 이동
    if args.symbol:
        sym_lower = args.symbol.lower()
        sym_model_dir = Path(f"models/{sym_lower}")
        sym_model_dir.mkdir(parents=True, exist_ok=True)
        dest = sym_model_dir / output_path.name
        shutil.move(str(output_path), str(dest))
        output_path = dest
    print_report(
        study, baseline_score, baseline_details, elapsed, output_path, args.min_recall,
    )

    # 5. 성능 개선 시 active 파일 자동 갱신
    if args.symbol:
        sym_lower = args.symbol.lower()
        active_path = Path(f"models/{sym_lower}/active_lgbm_params.json")
    else:
        active_path = Path("models/active_lgbm_params.json")
    if not args.no_baseline and study.best_value > baseline_score:
        shutil.copy(output_path, active_path)
        best_prec = study.best_trial.user_attrs.get("mean_precision", 0.0)
        bl_prec = baseline_details.get("mean_precision", 0.0)
        improvement = best_prec - bl_prec
        print(f"[MLOps] Precision +{improvement:.4f} 개선 → {active_path} 자동 갱신 완료")
        print(f"[MLOps] 다음 train_model.py 실행 시 새 파라미터가 자동 적용됩니다.\n")
    elif args.no_baseline:
        print("[MLOps] --no-baseline 모드: 성능 비교 없이 active 파일 유지\n")
    else:
        best_prec = study.best_trial.user_attrs.get("mean_precision", 0.0)
        bl_prec = baseline_details.get("mean_precision", 0.0)
        print(
            f"[MLOps] 성능 개선 없음 (Prec={best_prec:.4f} ≤ Baseline={bl_prec:.4f}) "
            f"→ active 파일 유지\n"
        )


if __name__ == "__main__":
    main()
