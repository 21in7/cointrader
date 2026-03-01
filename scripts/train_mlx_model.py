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

from src.dataset_builder import generate_dataset_vectorized
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


def train_mlx(data_path: str, time_weight_decay: float = 2.0) -> float:
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
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=time_weight_decay)
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

    # --- 클래스 불균형 처리: 언더샘플링 (가중치 인덱스 보존) ---
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    if len(neg_idx) > len(pos_idx):
        np.random.seed(42)
        neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)

    balanced_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(balanced_idx)

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
    print(f"\n검증 AUC: {auc:.4f}")
    print(classification_report(y_val, (val_proba >= 0.60).astype(int)))

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
) -> None:
    """Walk-Forward 검증: 슬라이딩 윈도우로 n_splits번 학습/검증 반복."""
    print(f"\n=== Walk-Forward 검증 ({n_splits}폴드, decay={time_weight_decay}) ===")
    raw = pd.read_parquet(data_path)
    df, btc_df, eth_df = _split_combined(raw)

    dataset = generate_dataset_vectorized(
        df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=time_weight_decay
    )
    missing = [c for c in FEATURE_COLS if c not in dataset.columns]
    for col in missing:
        dataset[col] = 0.0

    X_all = dataset[FEATURE_COLS].values.astype(np.float32)
    y_all = dataset["label"].values.astype(np.float32)
    w_all = dataset["sample_weight"].values.astype(np.float32)
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

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        if len(neg_idx) > len(pos_idx):
            np.random.seed(42)
            neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
        bal_idx = np.sort(np.concatenate([pos_idx, neg_idx]))

        X_tr_bal = X_tr_raw[bal_idx]
        y_tr_bal = y_tr[bal_idx]
        w_tr_bal = w_tr[bal_idx]

        # 폴드별 정규화 (학습 데이터 기준으로 계산, 검증에도 동일 적용)
        mean = X_tr_bal.mean(axis=0)
        std = X_tr_bal.std(axis=0) + 1e-8
        X_tr_norm = (X_tr_bal - mean) / std
        X_val_norm = (X_val_raw - mean) / std

        # DataFrame으로 래핑해서 MLXFilter.fit()에 전달
        # fit() 내부 정규화가 덮어쓰지 않도록 이미 정규화된 데이터를 넘기고
        # _mean=0, _std=1로 고정해 이중 정규화를 방지
        X_tr_df = pd.DataFrame(X_tr_norm, columns=FEATURE_COLS)
        X_val_df = pd.DataFrame(X_val_norm, columns=FEATURE_COLS)

        model = MLXFilter(
            input_dim=len(FEATURE_COLS),
            hidden_dim=128,
            lr=1e-3,
            epochs=100,
            batch_size=256,
        )
        model.fit(X_tr_df, pd.Series(y_tr_bal), sample_weight=w_tr_bal)
        # fit()이 내부에서 다시 정규화하므로 저장된 mean/std를 항등 변환으로 교체
        model._mean = np.zeros(len(FEATURE_COLS), dtype=np.float32)
        model._std = np.ones(len(FEATURE_COLS), dtype=np.float32)

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
    args = parser.parse_args()

    if args.wf:
        walk_forward_auc(args.data, time_weight_decay=args.decay, n_splits=args.wf_splits)
    else:
        train_mlx(args.data, time_weight_decay=args.decay)


if __name__ == "__main__":
    main()
