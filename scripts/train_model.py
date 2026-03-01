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
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

from src.indicators import Indicators
from src.ml_features import build_features, FEATURE_COLS
from src.label_builder import build_labels
from src.dataset_builder import generate_dataset_vectorized

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


LOOKAHEAD = 60
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
MODEL_PATH = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH = Path("models/training_log.json")


def _process_index(args: tuple) -> dict | None:
    """단일 인덱스에 대해 피처+레이블을 계산한다. Pool worker 함수."""
    i, df_values, df_columns = args
    df = pd.DataFrame(df_values, columns=df_columns)

    window = df.iloc[i - 60: i + 1].copy()
    ind = Indicators(window)
    df_ind = ind.calculate_all()

    if df_ind.iloc[-1].isna().any():
        return None

    signal = ind.get_signal(df_ind)
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
    """신호 발생 시점마다 피처와 레이블을 병렬로 생성한다."""
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


def train(data_path: str, time_weight_decay: float = 2.0):
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
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df, time_weight_decay=time_weight_decay)

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
        sample_weight=w_train,
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
        "backend": "lgbm",
        "auc": round(auc, 4),
        "samples": len(dataset),
        "features": len(actual_feature_cols),
        "time_weight_decay": time_weight_decay,
        "model_path": str(MODEL_PATH),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/combined_1m.parquet")
    parser.add_argument(
        "--decay", type=float, default=2.0,
        help="시간 가중치 감쇠 강도 (0=균등, 2.0=최신이 ~7.4배 높음)",
    )
    args = parser.parse_args()
    train(args.data, time_weight_decay=args.decay)


if __name__ == "__main__":
    main()
