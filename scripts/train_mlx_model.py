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


def train_mlx(data_path: str) -> float:
    print(f"데이터 로드: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df)}")

    print("\n데이터셋 생성 중...")
    t0 = time.perf_counter()
    dataset = generate_dataset_vectorized(df)
    t1 = time.perf_counter()
    print(f"데이터셋 생성 완료: {t1 - t0:.1f}초, {len(dataset)}개 샘플")

    if dataset.empty or "label" not in dataset.columns:
        raise ValueError("데이터셋 생성 실패: 샘플 0개")

    print(f"학습 샘플: {len(dataset)}개 (양성={dataset['label'].sum():.0f}, 음성={(dataset['label']==0).sum():.0f})")

    if len(dataset) < 200:
        raise ValueError(f"학습 샘플 부족: {len(dataset)}개 (최소 200 필요)")

    X = dataset[FEATURE_COLS]
    y = dataset["label"]

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print("\nMLX 신경망 학습 시작 (GPU)...")
    t2 = time.perf_counter()
    model = MLXFilter(
        input_dim=len(FEATURE_COLS),
        hidden_dim=128,
        lr=1e-3,
        epochs=100,
        batch_size=256,
    )
    model.fit(X_train, y_train)
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
        "model_path": str(MLX_MODEL_PATH),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()
    train_mlx(args.data)


if __name__ == "__main__":
    main()
