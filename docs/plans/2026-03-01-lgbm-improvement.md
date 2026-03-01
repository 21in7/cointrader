# LightGBM 예측력 개선 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 현재 AUC 0.54 수준의 LightGBM 모델을 피처 정규화 + 강한 시간 가중치 + Walk-Forward 검증 세 가지를 순서대로 적용해 AUC 0.57+ 로 끌어올린다.

**Architecture:**
- `src/dataset_builder.py`에 rolling z-score 정규화를 추가해 레짐 변화에 강한 피처를 만든다.
- `scripts/train_model.py`에 Walk-Forward 검증 루프를 추가해 실제 예측력을 정확히 측정한다.
- 1년치 `combined_1m.parquet` 데이터를 decay=4.0 이상의 강한 시간 가중치로 학습해 샘플 수와 최신성을 동시에 확보한다.

**Tech Stack:** LightGBM, pandas, numpy, scikit-learn, Python 3.13

---

## 배경: 현재 문제 진단 결과

| 데이터 | 구간별 독립 AUC | 전체 80/20 AUC |
|--------|----------------|----------------|
| combined 1년 | 0.49~0.51 (전 구간 동일) | 0.49 |
| xrpusdt 3개월 | 0.49~0.58 (구간 편차 큼) | 0.54 |

**핵심 원인 두 가지:**
1. `xrp_btc_rs` 같은 절대값 피처가 Q1=0.86 → Q4=3.68로 4배 변동 → 모델이 스케일 변화에 혼란
2. 학습셋(과거)이 검증셋(최근)을 설명 못 함 → Walk-Forward로 실제 예측력 측정 필요

---

## Task 1: 피처 정규화 개선 (rolling z-score)

**Files:**
- Modify: `src/dataset_builder.py` — `_calc_features_vectorized()` 함수 내부

**목표:** 절대값 피처(`atr_pct`, `vol_ratio`, `xrp_btc_rs`, `xrp_eth_rs`, `ret_1/3/5`, `btc_ret_1/3/5`, `eth_ret_1/3/5`)를 rolling 200 window z-score로 정규화해서 레짐 변화에 무관하게 만든다.

**Step 1: 정규화 헬퍼 함수 추가**

`_calc_features_vectorized()` 함수 시작 부분에 추가:

```python
def _rolling_zscore(arr: np.ndarray, window: int = 200) -> np.ndarray:
    """rolling window z-score 정규화. window 미만 구간은 0으로 채운다."""
    s = pd.Series(arr)
    mean = s.rolling(window, min_periods=window).mean()
    std  = s.rolling(window, min_periods=window).std()
    z = (s - mean) / std.replace(0, np.nan)
    return z.fillna(0).values.astype(np.float32)
```

**Step 2: 절대값 피처에 정규화 적용**

`result` DataFrame 생성 시 다음 피처를 정규화 버전으로 교체:

```python
# 기존
"atr_pct":   atr_pct.astype(np.float32),
"vol_ratio": vol_ratio.astype(np.float32),
"ret_1":     ret_1.astype(np.float32),
"ret_3":     ret_3.astype(np.float32),
"ret_5":     ret_5.astype(np.float32),

# 변경 후
"atr_pct":   _rolling_zscore(atr_pct),
"vol_ratio": _rolling_zscore(vol_ratio),
"ret_1":     _rolling_zscore(ret_1),
"ret_3":     _rolling_zscore(ret_3),
"ret_5":     _rolling_zscore(ret_5),
```

BTC/ETH 피처도 동일하게:
```python
"btc_ret_1": _rolling_zscore(btc_r1), "btc_ret_3": _rolling_zscore(btc_r3), ...
"xrp_btc_rs": _rolling_zscore(xrp_btc_rs), "xrp_eth_rs": _rolling_zscore(xrp_eth_rs),
```

**Step 3: 검증**

```bash
cd /Users/gihyeon/github/cointrader
.venv/bin/python -c "
from src.dataset_builder import generate_dataset_vectorized
import pandas as pd
df = pd.read_parquet('data/combined_1m.parquet')
base = ['open','high','low','close','volume']
btc = df[[c+'_btc' for c in base]].copy(); btc.columns = base
eth = df[[c+'_eth' for c in base]].copy(); eth.columns = base
ds = generate_dataset_vectorized(df[base].copy(), btc_df=btc, eth_df=eth, time_weight_decay=0)
print(ds[['atr_pct','vol_ratio','xrp_btc_rs']].describe())
"
```

기대 결과: `atr_pct`, `vol_ratio`, `xrp_btc_rs` 모두 mean≈0, std≈1 범위

---

## Task 2: Walk-Forward 검증 함수 추가

**Files:**
- Modify: `scripts/train_model.py` — `train()` 함수 뒤에 `walk_forward_auc()` 함수 추가 및 `main()` 에 `--wf` 플래그 추가

**목표:** 시계열 순서를 지키면서 n_splits번 학습/검증을 반복해 실제 미래 예측력의 평균 AUC를 측정한다.

**Step 1: walk_forward_auc 함수 추가**

`train()` 함수 바로 아래에 추가:

```python
def walk_forward_auc(
    data_path: str,
    time_weight_decay: float = 2.0,
    n_splits: int = 5,
    train_ratio: float = 0.6,
) -> None:
    """Walk-Forward 검증: 슬라이딩 윈도우로 n_splits번 학습/검증 반복."""
    import warnings
    from sklearn.metrics import roc_auc_score

    print(f"\n=== Walk-Forward 검증 ({n_splits}폴드) ===")
    df_raw = pd.read_parquet(data_path)
    base_cols = ["open", "high", "low", "close", "volume"]
    btc_df = eth_df = None
    if "close_btc" in df_raw.columns:
        btc_df = df_raw[[c + "_btc" for c in base_cols]].copy(); btc_df.columns = base_cols
    if "close_eth" in df_raw.columns:
        eth_df = df_raw[[c + "_eth" for c in base_cols]].copy(); eth_df.columns = base_cols
    df = df_raw[base_cols].copy()

    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df,
                                          time_weight_decay=time_weight_decay)
    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[actual_feature_cols].values
    y = dataset["label"].values
    w = dataset["sample_weight"].values
    n = len(dataset)

    step = int(n * (1 - train_ratio) / n_splits)
    train_end_start = int(n * train_ratio)

    aucs = []
    for i in range(n_splits):
        tr_end = train_end_start + i * step
        val_end = tr_end + step
        if val_end > n:
            break

        X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
        X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        if len(neg_idx) > len(pos_idx):
            np.random.seed(42)
            neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
        idx = np.sort(np.concatenate([pos_idx, neg_idx]))

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.1, random_state=42, verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr[idx], y_tr[idx], sample_weight=w_tr[idx])

        proba = model.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_val, proba)
        aucs.append(auc)
        print(f"  폴드 {i+1}/{n_splits}: 학습={tr_end}, 검증={tr_end}~{val_end} ({step}개), AUC={auc:.4f}")

    print(f"\n  Walk-Forward 평균 AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  폴드별: {[round(a,4) for a in aucs]}")
```

**Step 2: main()에 --wf 플래그 추가**

```python
parser.add_argument("--wf", action="store_true", help="Walk-Forward 검증 실행")
parser.add_argument("--wf-splits", type=int, default=5)

# args 처리 부분
if args.wf:
    walk_forward_auc(args.data, time_weight_decay=args.decay, n_splits=args.wf_splits)
else:
    train(args.data, time_weight_decay=args.decay)
```

**Step 3: 검증 실행**

```bash
# xrpusdt 3개월 Walk-Forward
.venv/bin/python scripts/train_model.py --data data/xrpusdt_1m.parquet --decay 2.0 --wf

# combined 1년 Walk-Forward
.venv/bin/python scripts/train_model.py --data data/combined_1m.parquet --decay 2.0 --wf
```

기대 결과: 폴드별 AUC가 0.50~0.58 범위, 평균 0.52+

---

## Task 3: 강한 시간 가중치 + 1년 데이터 최적화

**Files:**
- Modify: `scripts/train_model.py` — `train()` 함수 내 `--decay` 기본값 및 권장값 주석

**목표:** `combined_1m.parquet`에서 decay=4.0~5.0으로 최근 3개월에 집중하되 1년치 패턴도 참고한다.

**Step 1: decay 값별 AUC 비교 스크립트 실행**

```bash
for decay in 1.0 2.0 3.0 4.0 5.0; do
    echo "=== decay=$decay ==="
    .venv/bin/python scripts/train_model.py --data data/combined_1m.parquet --decay $decay --wf --wf-splits 3 2>&1 | grep "Walk-Forward 평균"
done
```

**Step 2: 최적 decay 값으로 최종 학습**

Walk-Forward 평균 AUC가 가장 높은 decay 값으로:

```bash
.venv/bin/python scripts/train_model.py --data data/combined_1m.parquet --decay <최적값>
```

**Step 3: 결과 확인**

```bash
.venv/bin/python -c "import json; log=json.load(open('models/training_log.json')); [print(e) for e in log[-3:]]"
```

---

## 예상 결과

| 개선 단계 | 예상 AUC |
|-----------|---------|
| 현재 (3개월, 기본) | 0.54 |
| + rolling z-score 정규화 | 0.54~0.56 |
| + Walk-Forward로 정확한 측정 | 측정 정확도 향상 |
| + decay=4.0, 1년 데이터 | 0.55~0.58 |

---

## 주의사항

- `_rolling_zscore`는 `dataset_builder.py` 내부에서만 사용 (실시간 봇 경로 `ml_features.py`는 건드리지 않음)
- Walk-Forward는 `--wf` 플래그로만 실행, 기본 `train()`은 그대로 유지
- rolling window=200은 약 3~4시간치 1분봉 → 단기 레짐 변화 반영
