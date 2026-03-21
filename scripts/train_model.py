"""
과거 캔들 데이터로 LightGBM 필터 모델을 학습하고 저장한다.
사용법: python scripts/train_model.py --data data/xrpusdt_1m.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

from src.indicators import Indicators
from src.ml_features import build_features, FEATURE_COLS
from src.label_builder import build_labels
from src.dataset_builder import generate_dataset_vectorized, stratified_undersample

def _cgroup_cpu_count() -> int:
    """cgroup v1/v2 쿼터를 읽어 실제 할당된 CPU 수를 반환한다.
    LXC/컨테이너 환경에서 cpu_count()가 호스트 전체 코어를 반환하는 문제를 방지한다.
    쿼터를 읽을 수 없으면 cpu_count()를 그대로 사용한다.
    """
    # cgroup v2
    try:
        quota_path = Path("/sys/fs/cgroup/cpu.max")
        if quota_path.exists():
            parts = quota_path.read_text().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, math.floor(quota / period))
    except Exception:
        pass

    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, math.floor(quota / period))
    except Exception:
        pass

    return cpu_count()


LOOKAHEAD = 24  # 15분봉 × 24 = 6시간 (dataset_builder.py와 동기화)
MODEL_PATH = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH = Path("models/training_log.json")


def _process_index(args: tuple) -> dict | None:
    """단일 인덱스에 대해 피처+레이블을 계산한다. Pool worker 함수."""
    ATR_SL_MULT = 1.5  # legacy values
    ATR_TP_MULT = 3.0
    i, df_values, df_columns = args
    df = pd.DataFrame(df_values, columns=df_columns)

    window = df.iloc[i - 60: i + 1].copy()
    ind = Indicators(window)
    df_ind = ind.calculate_all()

    if df_ind.iloc[-1].isna().any():
        return None

    signal, _ = ind.get_signal(df_ind)
    if signal == "HOLD":
        return None

    entry_price = float(df_ind["close"].iloc[-1])
    atr = float(df_ind["atr"].iloc[-1])
    if atr <= 0:
        return None

    stop_loss   = entry_price - atr * ATR_SL_MULT if signal == "LONG" else entry_price + atr * ATR_SL_MULT
    take_profit = entry_price + atr * ATR_TP_MULT if signal == "LONG" else entry_price - atr * ATR_TP_MULT

    future = df.iloc[i + 1: i + 1 + LOOKAHEAD]
    label = build_labels(
        future_closes=future["close"].tolist(),
        future_highs=future["high"].tolist(),
        future_lows=future["low"].tolist(),
        take_profit=take_profit,
        stop_loss=stop_loss,
        side=signal,
    )
    if label is None:
        return None

    features = build_features(df_ind, signal)
    row = features.to_dict()
    row["label"] = label
    return row


def generate_dataset(df: pd.DataFrame, n_jobs: int | None = None) -> pd.DataFrame:
    """[Deprecated] generate_dataset_vectorized()를 사용할 것."""
    warnings.warn(
        "generate_dataset()는 deprecated. generate_dataset_vectorized()를 사용하세요.",
        DeprecationWarning, stacklevel=2,
    )
    total = len(df)
    indices = range(60, total - LOOKAHEAD)

    # M4 mini: 10코어(P4+E6). 너무 많은 worker는 IPC 오버헤드를 늘리므로 8로 제한
    workers = n_jobs or min(max(1, _cgroup_cpu_count() - 1), 8)
    print(f"  병렬 처리: {workers}코어 사용 (총 {len(indices):,}개 인덱스)")

    # DataFrame을 numpy로 변환해서 worker 간 전달 비용 최소화
    df_values = df.values
    df_columns = list(df.columns)
    task_args = [(i, df_values, df_columns) for i in indices]

    rows = []
    errors = []
    # chunksize를 크게 잡아 IPC 직렬화 횟수를 줄임
    chunk = max(100, len(task_args) // workers)
    with Pool(processes=workers) as pool:
        for idx, result in enumerate(pool.imap(_process_index, task_args, chunksize=chunk)):
            if isinstance(result, dict):
                rows.append(result)
            elif result is not None:
                errors.append(result)
            if (idx + 1) % 10000 == 0:
                print(f"  진행: {idx + 1:,}/{len(task_args):,} | 샘플: {len(rows):,}개")

    if errors:
        print(f"  [경고] worker 오류 {len(errors)}건: {errors[0]}")

    if not rows:
        print("  [오류] 생성된 샘플이 없습니다. worker 예외 여부를 확인합니다...")
        # 단일 프로세스로 첫 번째 인덱스를 직접 실행해서 예외 확인
        try:
            test_result = _process_index(task_args[0])
            print(f"  단일 실행 결과: {test_result}")
        except Exception as e:
            import traceback
            print(f"  단일 실행 예외:\n{traceback.format_exc()}")

    return pd.DataFrame(rows)


ACTIVE_PARAMS_PATH = Path("models/active_lgbm_params.json")


def _load_lgbm_params(tuned_params_path: str | None) -> tuple[dict, float]:
    """기본 LightGBM 파라미터를 반환하고, 튜닝 JSON이 주어지면 덮어쓴다.

    우선순위:
      1. --tuned-params 명시적 인자
      2. models/active_lgbm_params.json (Optuna가 자동 갱신)
      3. 코드 내 하드코딩 기본값 (fallback)
    """
    lgbm_params: dict = {
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
    weight_scale = 1.783105

    # 명시적 인자가 없으면 active 파일 자동 탐색
    resolved_path = tuned_params_path or (
        str(ACTIVE_PARAMS_PATH) if ACTIVE_PARAMS_PATH.exists() else None
    )

    if resolved_path:
        with open(resolved_path, "r", encoding="utf-8") as f:
            tune_data = json.load(f)
        best_params = dict(tune_data["best_trial"]["params"])
        weight_scale = float(best_params.pop("weight_scale", 1.0))
        lgbm_params.update(best_params)
        source = "명시적 인자" if tuned_params_path else "active 파일 자동 로드"
        print(f"\n[Optuna] 튜닝 파라미터 로드 ({source}): {resolved_path}")
        print(f"[Optuna] 적용 파라미터: {lgbm_params}")
        print(f"[Optuna] weight_scale: {weight_scale}\n")
    else:
        print("[Optuna] active 파일 없음 → 코드 내 기본 파라미터 사용\n")

    return lgbm_params, weight_scale


def train(data_path: str, time_weight_decay: float = 2.0, tuned_params_path: str | None = None, atr_sl_mult: float = 2.0, atr_tp_mult: float = 2.0):
    print(f"데이터 로드: {data_path}")
    df_raw = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df_raw)}, 컬럼: {list(df_raw.columns)}")

    # 병합 데이터셋 여부 판별
    btc_df = None
    eth_df = None
    base_cols = ["open", "high", "low", "close", "volume"]

    if "close_btc" in df_raw.columns:
        btc_df = df_raw[[c + "_btc" for c in base_cols]].copy()
        btc_df.columns = base_cols
        print("BTC 피처 활성화")

    if "close_eth" in df_raw.columns:
        eth_df = df_raw[[c + "_eth" for c in base_cols]].copy()
        eth_df.columns = base_cols
        print("ETH 피처 활성화")

    df = df_raw[base_cols].copy()

    print("데이터셋 생성 중...")
    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df,
        time_weight_decay=time_weight_decay,
        negative_ratio=5,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
    )

    if dataset.empty or "label" not in dataset.columns:
        raise ValueError(f"데이터셋 생성 실패: 샘플 0개. 위 오류 메시지를 확인하세요.")

    print(f"학습 샘플: {len(dataset)}개 (양성={dataset['label'].sum():.0f}, 음성={(dataset['label']==0).sum():.0f})")

    if len(dataset) < 200:
        raise ValueError(f"학습 샘플 부족: {len(dataset)}개 (최소 200 필요)")

    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    print(f"사용 피처: {len(actual_feature_cols)}개 {actual_feature_cols}")
    X = dataset[actual_feature_cols]
    y = dataset["label"]
    w = dataset["sample_weight"].values
    source = dataset["source"].values if "source" in dataset.columns else np.full(len(X), "signal")

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # 튜닝 파라미터 로드 (없으면 기본값 사용)
    lgbm_params, weight_scale = _load_lgbm_params(tuned_params_path)
    w_train = (w[:split] * weight_scale).astype(np.float32)

    # --- 계층적 샘플링: signal 전수 유지, HOLD negative만 양성 수 만큼 ---
    source_train = source[:split]
    balanced_idx = stratified_undersample(y_train.values, source_train, seed=42)

    X_train = X_train.iloc[balanced_idx]
    y_train = y_train.iloc[balanced_idx]
    w_train = w_train[balanced_idx]

    sig_count = (source_train[balanced_idx] == "signal").sum()
    hold_count = (source_train[balanced_idx] == "hold_negative").sum()
    print(f"\n계층적 샘플링 후 학습 데이터: {len(X_train)}개 "
          f"(Signal={sig_count}, HOLD={hold_count}, "
          f"양성={int(y_train.sum())}, 음성={int((y_train==0).sum())})")
    print(f"검증 데이터: {len(X_val)}개 (양성={int(y_val.sum())}, 음성={int((y_val==0).sum())})")
    # ---------------------------------------------------------------

    model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(80, first_metric_only=True, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    # 최적 임계값 탐색: 최소 재현율(0.15) 조건부 정밀도 최대화
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    # precision_recall_curve의 마지막 원소는 (1.0, 0.0)이므로 제외
    precisions, recalls = precisions[:-1], recalls[:-1]

    MIN_RECALL = 0.15
    valid_idx = np.where(recalls >= MIN_RECALL)[0]
    if len(valid_idx) > 0:
        best_idx  = valid_idx[np.argmax(precisions[valid_idx])]
        best_thr  = float(thresholds[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec  = float(recalls[best_idx])
    else:
        best_thr, best_prec, best_rec = 0.50, 0.0, 0.0
        print(f"  [경고] recall >= {MIN_RECALL} 조건 만족 임계값 없음 → 기본값 0.50 사용")

    print(f"\n검증 AUC: {auc:.4f}  |  최적 임계값: {best_thr:.4f} "
          f"(Precision={best_prec:.3f}, Recall={best_rec:.3f})")
    print(classification_report(y_val, (val_proba >= best_thr).astype(int), zero_division=0))

    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, PREV_MODEL_PATH)
        print(f"기존 모델 백업: {PREV_MODEL_PATH}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    log = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            log = json.load(f)
    log_entry: dict = {
        "date": datetime.now().isoformat(),
        "backend": "lgbm",
        "auc": round(auc, 4),
        "best_threshold": round(best_thr, 4),
        "best_precision": round(best_prec, 3),
        "best_recall":    round(best_rec, 3),
        "samples": len(dataset),
        "features": len(actual_feature_cols),
        "time_weight_decay": time_weight_decay,
        "model_path": str(MODEL_PATH),
        "tuned_params_path": tuned_params_path,
        "lgbm_params": lgbm_params,
        "weight_scale": weight_scale,
    }
    log.append(log_entry)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def walk_forward_auc(
    data_path: str,
    time_weight_decay: float = 2.0,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    tuned_params_path: str | None = None,
    atr_sl_mult: float = 2.0,
    atr_tp_mult: float = 2.0,
) -> None:
    """Walk-Forward 검증: 슬라이딩 윈도우로 n_splits번 학습/검증 반복.

    시계열 순서를 지키면서 매 폴드마다 학습 구간을 늘려가며 검증한다.
    실제 미래 예측력의 평균 AUC를 측정하는 데 사용한다.
    """
    import warnings

    print(f"\n=== Walk-Forward 검증 ({n_splits}폴드, decay={time_weight_decay}) ===")
    df_raw = pd.read_parquet(data_path)
    base_cols = ["open", "high", "low", "close", "volume"]
    btc_df = eth_df = None
    if "close_btc" in df_raw.columns:
        btc_df = df_raw[[c + "_btc" for c in base_cols]].copy()
        btc_df.columns = base_cols
    if "close_eth" in df_raw.columns:
        eth_df = df_raw[[c + "_eth" for c in base_cols]].copy()
        eth_df.columns = base_cols
    df = df_raw[base_cols].copy()

    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df,
        time_weight_decay=time_weight_decay,
        negative_ratio=5,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
    )
    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[actual_feature_cols].values
    y = dataset["label"].values
    w = dataset["sample_weight"].values
    n = len(dataset)
    source = dataset["source"].values if "source" in dataset.columns else np.full(n, "signal")

    lgbm_params, weight_scale = _load_lgbm_params(tuned_params_path)
    w = (w * weight_scale).astype(np.float32)

    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    aucs = []
    fold_metrics = []
    for i in range(n_splits):
        tr_end = train_end_start + i * step
        val_end = tr_end + step
        if val_end > n:
            break

        X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
        X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]

        source_tr = source[:tr_end]
        idx = stratified_undersample(y_tr, source_tr, seed=42)

        model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr[idx], y_tr[idx], sample_weight=w_tr[idx])

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5
        aucs.append(auc)

        # 폴드별 최적 임계값 (recall >= 0.15 조건부 precision 최대화)
        MIN_RECALL = 0.15
        precs, recs, thrs = precision_recall_curve(y_val, proba)
        precs, recs = precs[:-1], recs[:-1]
        valid_idx = np.where(recs >= MIN_RECALL)[0]
        if len(valid_idx) > 0:
            best_i = valid_idx[np.argmax(precs[valid_idx])]
            f_thr, f_prec, f_rec = float(thrs[best_i]), float(precs[best_i]), float(recs[best_i])
        else:
            f_thr, f_prec, f_rec = 0.50, 0.0, 0.0

        fold_metrics.append({"auc": auc, "precision": f_prec, "recall": f_rec, "threshold": f_thr})
        print(
            f"  폴드 {i+1}/{n_splits}: 학습={tr_end}개, "
            f"검증={tr_end}~{val_end} ({step}개), AUC={auc:.4f}  |  "
            f"Thr={f_thr:.4f}  Prec={f_prec:.3f}  Rec={f_rec:.3f}"
        )

    mean_prec = np.mean([m["precision"] for m in fold_metrics])
    mean_rec = np.mean([m["recall"] for m in fold_metrics])
    mean_thr = np.mean([m["threshold"] for m in fold_metrics])
    print(f"\n  Walk-Forward 평균 AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  평균 Precision: {mean_prec:.3f}  |  평균 Recall: {mean_rec:.3f}  |  평균 Threshold: {mean_thr:.4f}")
    print(f"  폴드별: {[round(a, 4) for a in aucs]}")


def compare(data_path: str, time_weight_decay: float = 2.0, tuned_params_path: str | None = None, atr_sl_mult: float = 2.0, atr_tp_mult: float = 2.0):
    """기존 피처 vs OI 파생 피처 추가 버전 A/B 비교."""
    import warnings

    print("=" * 70)
    print("  OI 파생 피처 A/B 비교 (30일 데이터 기반, 방향성 참고용)")
    print("=" * 70)

    df_raw = pd.read_parquet(data_path)
    base_cols = ["open", "high", "low", "close", "volume"]
    btc_df = eth_df = None
    if "close_btc" in df_raw.columns:
        btc_df = df_raw[[c + "_btc" for c in base_cols]].copy()
        btc_df.columns = base_cols
    if "close_eth" in df_raw.columns:
        eth_df = df_raw[[c + "_eth" for c in base_cols]].copy()
        eth_df.columns = base_cols
    df = df_raw[base_cols].copy()
    if "oi_change" in df_raw.columns:
        df["oi_change"] = df_raw["oi_change"]
    if "funding_rate" in df_raw.columns:
        df["funding_rate"] = df_raw["funding_rate"]

    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df,
        time_weight_decay=time_weight_decay,
        negative_ratio=5,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
    )

    if dataset.empty:
        raise ValueError("데이터셋 생성 실패")

    lgbm_params, weight_scale = _load_lgbm_params(tuned_params_path)

    # Baseline: OI 파생 피처 제외
    BASELINE_EXCLUDE = {"oi_change_ma5", "oi_price_spread"}
    baseline_cols = [c for c in FEATURE_COLS if c in dataset.columns and c not in BASELINE_EXCLUDE]
    new_cols = [c for c in FEATURE_COLS if c in dataset.columns]

    results = {}
    for label, cols in [("Baseline", baseline_cols), ("New", new_cols)]:
        X = dataset[cols]
        y = dataset["label"]
        w = dataset["sample_weight"].values
        source = dataset["source"].values if "source" in dataset.columns else np.full(len(X), "signal")

        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]
        w_tr = (w[:split] * weight_scale).astype(np.float32)
        source_tr = source[:split]

        balanced_idx = stratified_undersample(y_tr.values, source_tr, seed=42)
        X_tr_b = X_tr.iloc[balanced_idx]
        y_tr_b = y_tr.iloc[balanced_idx]
        w_tr_b = w_tr[balanced_idx]

        model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_b, y_tr_b, sample_weight=w_tr_b)

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5

        precs, recs, thrs = precision_recall_curve(y_val, proba)
        precs, recs = precs[:-1], recs[:-1]
        valid_idx = np.where(recs >= 0.15)[0]
        if len(valid_idx) > 0:
            best_i = valid_idx[np.argmax(precs[valid_idx])]
            thr, prec, rec = float(thrs[best_i]), float(precs[best_i]), float(recs[best_i])
        else:
            thr, prec, rec = 0.50, 0.0, 0.0

        # Feature importance
        imp = dict(zip(cols, model.feature_importances_))
        top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]

        results[label] = {
            "auc": auc, "precision": prec, "recall": rec,
            "threshold": thr, "n_val": len(y_val),
            "n_val_pos": int(y_val.sum()), "top10": top10,
        }

    # 비교 테이블 출력
    n_base = len(baseline_cols)
    n_new = len(new_cols)
    print(f"\n{'지표':<20} {f'Baseline({n_base})':>15} {f'New({n_new})':>15} {'Delta':>10}")
    print("-" * 62)
    for metric in ["auc", "precision", "recall", "threshold"]:
        b = results["Baseline"][metric]
        n = results["New"][metric]
        d = n - b
        sign = "+" if d > 0 else ""
        print(f"{metric:<20} {b:>15.4f} {n:>15.4f} {sign}{d:>9.4f}")

    n_val = results["Baseline"]["n_val"]
    n_pos = results["Baseline"]["n_val_pos"]
    print(f"\n검증셋: n={n_val} (양성={n_pos}, 음성={n_val - n_pos})")
    print("⚠ 30일 데이터 기반 — 방향성 참고용\n")

    print("Feature Importance Top 10 (New):")
    for feat_name, imp_val in results["New"]["top10"]:
        marker = " ← NEW" if feat_name in BASELINE_EXCLUDE else ""
        print(f"  {feat_name:<25} {imp_val:>6}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--symbol", type=str, default=None,
                        help="학습 대상 심볼 (예: TRXUSDT). data/{symbol}/ 에서 데이터 로드, models/{symbol}/ 에 저장")
    parser.add_argument(
        "--decay", type=float, default=2.0,
        help="시간 가중치 감쇠 강도 (0=균등, 2.0=최신이 ~7.4배 높음)",
    )
    parser.add_argument("--wf", action="store_true", help="Walk-Forward 검증 실행")
    parser.add_argument("--wf-splits", type=int, default=5, help="Walk-Forward 폴드 수")
    parser.add_argument(
        "--tuned-params", type=str, default=None,
        help="Optuna 튜닝 결과 JSON 경로 (지정 시 기본 파라미터를 덮어씀)",
    )
    parser.add_argument("--compare", action="store_true",
                        help="OI 파생 피처 추가 전후 A/B 성능 비교")
    parser.add_argument("--sl-mult", type=float, default=2.0, help="SL ATR 배수 (기본 2.0)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR 배수 (기본 2.0)")
    args = parser.parse_args()

    # --symbol 모드: 심볼별 디렉토리 경로 자동 결정
    if args.symbol:
        sym_lower = args.symbol.lower()
        if args.data is None:
            args.data = f"data/{sym_lower}/combined_15m.parquet"
        global MODEL_PATH, PREV_MODEL_PATH, LOG_PATH, ACTIVE_PARAMS_PATH
        MODEL_PATH = Path(f"models/{sym_lower}/lgbm_filter.pkl")
        PREV_MODEL_PATH = Path(f"models/{sym_lower}/lgbm_filter_prev.pkl")
        LOG_PATH = Path(f"models/{sym_lower}/training_log.json")
        ACTIVE_PARAMS_PATH = Path(f"models/{sym_lower}/active_lgbm_params.json")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    elif args.data is None:
        args.data = "data/combined_15m.parquet"

    if args.compare:
        compare(args.data, time_weight_decay=args.decay, tuned_params_path=args.tuned_params,
                atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult)
    elif args.wf:
        walk_forward_auc(
            args.data,
            time_weight_decay=args.decay,
            n_splits=args.wf_splits,
            tuned_params_path=args.tuned_params,
            atr_sl_mult=args.sl_mult,
            atr_tp_mult=args.tp_mult,
        )
    else:
        train(args.data, time_weight_decay=args.decay, tuned_params_path=args.tuned_params,
              atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult)


if __name__ == "__main__":
    main()
