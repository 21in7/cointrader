"""
MLX 기반 신경망 필터를 학습하고 저장한다.
M4 통합 GPU(Metal)를 자동으로 사용한다.

사용법: python scripts/train_mlx_model.py --data data/xrpusdt_1m.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

from src.dataset_builder import generate_dataset_vectorized, stratified_undersample
from src.ml_features import FEATURE_COLS
from src.mlx_filter import MLXFilter

MLX_MODEL_PATH = Path("models/mlx_filter.weights")
LOG_PATH = Path("models/training_log.json")


def _split_combined(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """combined parquet에서 XRP/BTC/ETH DataFrame을 분리한다."""
    xrp_cols = ["open", "high", "low", "close", "volume"]
    xrp_df = df[xrp_cols].copy()

    btc_df = None
    eth_df = None
    btc_raw = [c for c in df.columns if c.endswith("_btc")]
    eth_raw = [c for c in df.columns if c.endswith("_eth")]

    if btc_raw:
        btc_df = df[btc_raw].copy()
        btc_df.columns = [c.replace("_btc", "") for c in btc_raw]
    if eth_raw:
        eth_df = df[eth_raw].copy()
        eth_df.columns = [c.replace("_eth", "") for c in eth_raw]

    return xrp_df, btc_df, eth_df


def train_mlx(data_path: str, time_weight_decay: float = 2.0, atr_sl_mult: float = 2.0, atr_tp_mult: float = 2.0) -> float:
    print(f"데이터 로드: {data_path}")
    raw = pd.read_parquet(data_path)
    print(f"캔들 수: {len(raw)}")

    df, btc_df, eth_df = _split_combined(raw)
    if btc_df is not None:
        print(f"  BTC/ETH 피처 활성화 (21개 피처)")
    else:
        print(f"  XRP 단독 데이터 (13개 피처)")

    print("\n데이터셋 생성 중...")
    t0 = time.perf_counter()
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=time_weight_decay,
                                          atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, negative_ratio=5)
    t1 = time.perf_counter()
    print(f"데이터셋 생성 완료: {t1 - t0:.1f}초, {len(dataset)}개 샘플")

    if dataset.empty or "label" not in dataset.columns:
        raise ValueError("데이터셋 생성 실패: 샘플 0개")

    print(f"학습 샘플: {len(dataset)}개 (양성={dataset['label'].sum():.0f}, 음성={(dataset['label']==0).sum():.0f})")

    if len(dataset) < 200:
        raise ValueError(f"학습 샘플 부족: {len(dataset)}개 (최소 200 필요)")

    actual_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    missing = [c for c in FEATURE_COLS if c not in dataset.columns]
    if missing:
        print(f"  경고: 데이터셋에 없는 피처 {missing} → 0으로 채움 (BTC/ETH 데이터 미제공)")
        for col in missing:
            dataset[col] = 0.0
    X = dataset[FEATURE_COLS]
    y = dataset["label"]
    w = dataset["sample_weight"].values

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    w_train = w[:split]

    # --- 클래스 불균형 처리: stratified 언더샘플링 (Signal 전수 유지, HOLD만 샘플링) ---
    source = dataset["source"].values if "source" in dataset.columns else np.full(len(dataset), "signal")
    source_train = source[:split]
    balanced_idx = stratified_undersample(y_train.values, source_train, seed=42)

    X_train = X_train.iloc[balanced_idx]
    y_train = y_train.iloc[balanced_idx]
    w_train = w_train[balanced_idx]

    print(f"\n언더샘플링 적용 후 학습 데이터: {len(X_train)}개 (양성={y_train.sum()}, 음성={(y_train==0).sum()})")
    # --------------------------------------

    print("\nMLX 신경망 학습 시작 (GPU)...")
    t2 = time.perf_counter()
    model = MLXFilter(
        input_dim=len(FEATURE_COLS),
        hidden_dim=128,
        lr=1e-3,
        epochs=100,
        batch_size=256,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    t3 = time.perf_counter()
    print(f"학습 완료: {t3 - t2:.1f}초")

    val_proba = model.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_proba)

    # 최적 임계값 탐색: 최소 재현율(0.15) 조건부 정밀도 최대화
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
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

    MLX_MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MLX_MODEL_PATH)
    print(f"모델 저장: {MLX_MODEL_PATH}")

    log = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append({
        "date": datetime.now().isoformat(),
        "backend": "mlx",
        "auc": round(auc, 4),
        "best_threshold": round(best_thr, 4),
        "best_precision": round(best_prec, 3),
        "best_recall":    round(best_rec, 3),
        "samples": len(dataset),
        "train_sec": round(t3 - t2, 1),
        "time_weight_decay": time_weight_decay,
        "model_path": str(MLX_MODEL_PATH),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def walk_forward_auc(
    data_path: str,
    time_weight_decay: float = 2.0,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    atr_sl_mult: float = 2.0,
    atr_tp_mult: float = 2.0,
) -> None:
    """Walk-Forward 검증: 슬라이딩 윈도우로 n_splits번 학습/검증 반복."""
    print(f"\n=== Walk-Forward 검증 ({n_splits}폴드, decay={time_weight_decay}) ===")
    raw = pd.read_parquet(data_path)
    df, btc_df, eth_df = _split_combined(raw)

    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=time_weight_decay,
        atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, negative_ratio=5,
    )
    missing = [c for c in FEATURE_COLS if c not in dataset.columns]
    for col in missing:
        dataset[col] = 0.0

    X_all = dataset[FEATURE_COLS].values.astype(np.float32)
    y_all = dataset["label"].values.astype(np.float32)
    w_all = dataset["sample_weight"].values.astype(np.float32)
    source_all = dataset["source"].values if "source" in dataset.columns else np.full(len(dataset), "signal")
    n = len(dataset)

    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    aucs = []
    for i in range(n_splits):
        tr_end = train_end_start + i * step
        val_end = tr_end + step
        if val_end > n:
            break

        X_tr_raw = X_all[:tr_end]
        y_tr = y_all[:tr_end]
        w_tr = w_all[:tr_end]
        X_val_raw = X_all[tr_end:val_end]
        y_val = y_all[tr_end:val_end]

        source_tr = source_all[:tr_end]
        bal_idx = stratified_undersample(y_tr, source_tr, seed=42)

        X_tr_bal = X_tr_raw[bal_idx]
        y_tr_bal = y_tr[bal_idx]
        w_tr_bal = w_tr[bal_idx]

        X_tr_df = pd.DataFrame(X_tr_bal, columns=FEATURE_COLS)
        X_val_df = pd.DataFrame(X_val_raw, columns=FEATURE_COLS)

        model = MLXFilter(
            input_dim=len(FEATURE_COLS),
            hidden_dim=128,
            lr=1e-3,
            epochs=100,
            batch_size=256,
        )
        model.fit(X_tr_df, pd.Series(y_tr_bal), sample_weight=w_tr_bal)
        # fit() handles normalization internally, predict_proba() applies same mean/std

        proba = model.predict_proba(X_val_df)
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5
        aucs.append(auc)
        print(
            f"  폴드 {i+1}/{n_splits}: 학습={tr_end}개, "
            f"검증={tr_end}~{val_end} ({step}개), AUC={auc:.4f}"
        )

    print(f"\n  Walk-Forward 평균 AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  폴드별: {[round(a, 4) for a in aucs]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/combined_15m.parquet")
    parser.add_argument(
        "--decay", type=float, default=2.0,
        help="시간 가중치 감쇠 강도 (0=균등, 2.0=최신이 ~7.4배 높음)",
    )
    parser.add_argument("--wf", action="store_true", help="Walk-Forward 검증 실행")
    parser.add_argument("--wf-splits", type=int, default=5, help="Walk-Forward 폴드 수")
    parser.add_argument("--sl-mult", type=float, default=2.0, help="SL ATR 배수 (기본 2.0)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR 배수 (기본 2.0)")
    args = parser.parse_args()

    if args.wf:
        walk_forward_auc(args.data, time_weight_decay=args.decay, n_splits=args.wf_splits,
                         atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult)
    else:
        train_mlx(args.data, time_weight_decay=args.decay,
                  atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult)


if __name__ == "__main__":
    main()
