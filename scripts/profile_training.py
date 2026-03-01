"""
학습 파이프라인 각 단계의 소요 시간을 측정한다.
사용법: python scripts/profile_training.py --data data/xrpusdt_1m.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import argparse
import pandas as pd
from scripts.train_model import generate_dataset, _cgroup_cpu_count


def profile(data_path: str):
    print(f"데이터 로드: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"캔들 수: {len(df)}")

    workers = max(1, _cgroup_cpu_count() - 1)
    print(f"사용 코어: {workers}")

    t0 = time.perf_counter()
    dataset = generate_dataset(df)
    t1 = time.perf_counter()
    print(f"\n[결과] 데이터셋 생성: {t1 - t0:.1f}초, 샘플 {len(dataset)}개")

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from src.ml_features import FEATURE_COLS

    X = dataset[FEATURE_COLS]
    y = dataset["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    t2 = time.perf_counter()
    model.fit(X_train, y_train)
    t3 = time.perf_counter()
    print(f"[결과] LightGBM 학습: {t3 - t2:.1f}초")
    print(f"[결과] 전체: {t3 - t0:.1f}초")
    print(f"\n[비율] 데이터셋 생성: {(t1-t0)/(t3-t0)*100:.0f}% / LightGBM 학습: {(t3-t2)/(t3-t0)*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()
    profile(args.data)
