"""
과거 캔들 데이터로 LightGBM 필터 모델을 학습하고 저장한다.
사용법: python scripts/train_model.py --data data/xrpusdt_1m.parquet
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from src.indicators import Indicators
from src.ml_features import build_features, FEATURE_COLS
from src.label_builder import build_labels

LOOKAHEAD = 60
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MODEL_PATH = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH = Path("models/training_log.json")


def generate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """신호 발생 시점마다 피처와 레이블을 생성한다."""
    rows = []
    total = len(df)

    for i in range(60, total - LOOKAHEAD):
        window = df.iloc[i - 60: i + 1].copy()
        ind = Indicators(window)
        df_ind = ind.calculate_all()

        if df_ind.isna().any().any():
            continue

        signal = ind.get_signal(df_ind)
        if signal == "HOLD":
            continue

        entry_price = float(df_ind["close"].iloc[-1])
        atr = float(df_ind["atr"].iloc[-1])
        if atr <= 0:
            continue

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
            continue

        features = build_features(df_ind, signal)
        row = features.to_dict()
        row["label"] = label
        rows.append(row)

        if len(rows) % 500 == 0:
            print(f"  샘플 생성 중: {len(rows)}개 (인덱스 {i}/{total})")

    return pd.DataFrame(rows)


def train(data_path: str):
    print(f"데이터 로드: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df)}")

    print("데이터셋 생성 중...")
    dataset = generate_dataset(df)
    print(f"학습 샘플: {len(dataset)}개 (양성={dataset['label'].sum():.0f}, 음성={(dataset['label']==0).sum():.0f})")

    if len(dataset) < 200:
        raise ValueError(f"학습 샘플 부족: {len(dataset)}개 (최소 200 필요)")

    X = dataset[FEATURE_COLS]
    y = dataset["label"]

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    print(f"\n검증 AUC: {auc:.4f}")
    print(classification_report(y_val, (val_proba >= 0.60).astype(int)))

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
    log.append({
        "date": datetime.now().isoformat(),
        "auc": round(auc, 4),
        "samples": len(dataset),
        "model_path": str(MODEL_PATH),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
