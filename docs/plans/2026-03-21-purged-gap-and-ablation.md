# Purged Gap + Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Walk-Forward 검증에 purged gap(embargo)을 추가하여 레이블 누수를 제거하고, feature ablation으로 signal_strength/side 의존도를 진단하여 ML 필터의 실질적 예측력을 검증한다.

**Architecture:** 3개의 walk-forward 함수(train_model.py, train_mlx_model.py, tune_hyperparams.py)의 검증 시작 인덱스에 `LOOKAHEAD` 만큼의 embargo를 추가한다. `train_model.py`에 `--ablation` CLI 플래그를 추가하여 A/B/C 실험을 자동 실행하고 상대 드롭을 출력한다.

**Tech Stack:** Python, LightGBM, numpy, sklearn, pytest

**판단 기준 (합의됨):**
- A→C 드롭 ≤ 0.05: ML 필터 가치 있음
- A→C 드롭 0.05~0.10: 조건부 투입
- A→C 드롭 ≥ 0.10: 재설계 필요

---

## File Structure

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `scripts/train_model.py` | Modify | purged gap + ablation CLI |
| `scripts/train_mlx_model.py` | Modify | purged gap |
| `scripts/tune_hyperparams.py` | Modify | purged gap |
| `tests/test_ml_pipeline_fixes.py` | Modify | purged gap 테스트 |

---

### Task 1: walk-forward에 purged gap(embargo) 추가

**Files:**
- Modify: `scripts/train_model.py:389-396`
- Modify: `scripts/train_mlx_model.py:194-204`
- Modify: `scripts/tune_hyperparams.py:153-160`
- Test: `tests/test_ml_pipeline_fixes.py`

- [ ] **Step 1: purged gap 테스트 작성**

`tests/test_ml_pipeline_fixes.py`에 추가:

```python
def test_walk_forward_purged_gap():
    """Walk-Forward 검증에서 학습/검증 사이에 LOOKAHEAD 만큼의 gap이 존재해야 한다."""
    from src.dataset_builder import LOOKAHEAD
    import numpy as np

    # 시뮬레이션: n=1000, train_ratio=0.6, n_splits=5
    n = 1000
    train_ratio = 0.6
    n_splits = 5
    embargo = LOOKAHEAD  # 24

    step = max(1, int(n * (1 - train_ratio) / n_splits))
    train_end_start = int(n * train_ratio)

    for fold_idx in range(n_splits):
        tr_end = train_end_start + fold_idx * step
        val_start = tr_end + embargo  # purged gap
        val_end = val_start + step
        if val_end > n:
            break

        # 학습 마지막 인덱스와 검증 첫 인덱스 사이에 최소 embargo 캔들 gap
        assert val_start - tr_end >= embargo, \
            f"폴드 {fold_idx}: gap={val_start - tr_end} < embargo={embargo}"
        # 검증 구간이 학습 구간과 겹치지 않아야 한다
        assert val_start > tr_end, \
            f"폴드 {fold_idx}: val_start={val_start} <= tr_end={tr_end}"
```

- [ ] **Step 2: 테스트 통과 확인 (로직 테스트이므로 바로 PASS)**

Run: `pytest tests/test_ml_pipeline_fixes.py::test_walk_forward_purged_gap -v`
Expected: PASS (이 테스트는 로직만 검증하므로 코드 변경 없이도 통과)

- [ ] **Step 3: train_model.py walk_forward_auc() 수정**

`scripts/train_model.py`의 `walk_forward_auc()` 함수 내 폴드 루프(~line 389-396):

변경 전:
```python
for i in range(n_splits):
    tr_end = train_end_start + i * step
    val_end = tr_end + step
    if val_end > n:
        break

    X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
    X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]
```

변경 후:
```python
from src.dataset_builder import LOOKAHEAD

for i in range(n_splits):
    tr_end = train_end_start + i * step
    val_start = tr_end + LOOKAHEAD  # purged gap: 레이블 누수 방지
    val_end = val_start + step
    if val_end > n:
        break

    X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
    X_val, y_val = X[val_start:val_end], y[val_start:val_end]
```

`source_tr`는 기존과 동일하게 `source[:tr_end]`.

출력 문자열도 업데이트:
```python
print(
    f"  폴드 {i+1}/{n_splits}: 학습={tr_end}개, "
    f"검증={val_start}~{val_end} ({step}개, embargo={LOOKAHEAD}), AUC={auc:.4f}  |  "
    f"Thr={f_thr:.4f}  Prec={f_prec:.3f}  Rec={f_rec:.3f}"
)
```

- [ ] **Step 4: train_mlx_model.py walk_forward_auc() 동일 수정**

`scripts/train_mlx_model.py`의 `walk_forward_auc()` 폴드 루프(~line 194-204):

변경 전:
```python
X_val_raw = X_all[tr_end:val_end]
y_val = y_all[tr_end:val_end]
```

변경 후:
```python
from src.dataset_builder import LOOKAHEAD

val_start = tr_end + LOOKAHEAD
val_end = val_start + step
if val_end > n:
    break

X_val_raw = X_all[val_start:val_end]
y_val = y_all[val_start:val_end]
```

- [ ] **Step 5: tune_hyperparams.py _walk_forward_cv() 동일 수정**

`scripts/tune_hyperparams.py`의 `_walk_forward_cv()` 폴드 루프(~line 153-160):

변경 전:
```python
X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]
```

변경 후:
```python
from src.dataset_builder import LOOKAHEAD

val_start = tr_end + LOOKAHEAD
val_end = val_start + step
if val_end > n:
    break

X_val, y_val = X[val_start:val_end], y[val_start:val_end]
```

- [ ] **Step 6: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 7: 커밋**

```bash
git add scripts/train_model.py scripts/train_mlx_model.py scripts/tune_hyperparams.py tests/test_ml_pipeline_fixes.py
git commit -m "fix(ml): add purged gap (embargo=LOOKAHEAD) to walk-forward validation"
```

---

### Task 2: Feature ablation 실험 CLI 추가

**Files:**
- Modify: `scripts/train_model.py`

- [ ] **Step 1: ablation 함수 추가**

`scripts/train_model.py`에 `ablation()` 함수를 추가:

```python
def ablation(
    data_path: str,
    time_weight_decay: float = 2.0,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    tuned_params_path: str | None = None,
    atr_sl_mult: float = 2.0,
    atr_tp_mult: float = 2.0,
) -> None:
    """Feature ablation 실험: signal_strength/side 의존도 진단.

    실험 A: 전체 피처 (baseline)
    실험 B: signal_strength 제거
    실험 C: signal_strength + side 제거

    판단 기준 (절대 AUC 차이):
      A→C ≤ 0.05: ML 필터 가치 있음 (다른 피처가 충분히 기여)
      A→C 0.05~0.10: 조건부 투입 (signal_strength 의존도 높지만 다른 피처도 기여)
      A→C ≥ 0.10: 재설계 필요 (사실상 점수 재확인기)
    """
    import warnings
    from src.dataset_builder import LOOKAHEAD

    print(f"\n{'='*64}")
    print(f"  Feature Ablation 실험 ({n_splits}폴드 Walk-Forward, embargo={LOOKAHEAD})")
    print(f"{'='*64}")

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
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
    )
    actual_feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    y = dataset["label"].values
    w = dataset["sample_weight"].values
    n = len(dataset)
    source = dataset["source"].values if "source" in dataset.columns else np.full(n, "signal")

    lgbm_params, weight_scale = _load_lgbm_params(tuned_params_path)
    w = (w * weight_scale).astype(np.float32)

    # 실험 정의
    experiments = {
        "A (전체 피처)": actual_feature_cols,
        "B (-signal_strength)": [c for c in actual_feature_cols if c != "signal_strength"],
        "C (-signal_strength, -side)": [c for c in actual_feature_cols if c not in ("signal_strength", "side")],
    }

    results = {}
    for exp_name, cols in experiments.items():
        X = dataset[cols].values
        step = max(1, int(n * (1 - train_ratio) / n_splits))
        train_end_start = int(n * train_ratio)

        fold_aucs = []
        fold_importances = []
        for fold_idx in range(n_splits):
            tr_end = train_end_start + fold_idx * step
            val_start = tr_end + LOOKAHEAD
            val_end = val_start + step
            if val_end > n:
                break

            X_tr, y_tr, w_tr = X[:tr_end], y[:tr_end], w[:tr_end]
            X_val, y_val = X[val_start:val_end], y[val_start:val_end]

            source_tr = source[:tr_end]
            idx = stratified_undersample(y_tr, source_tr, seed=42)

            model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr[idx], y_tr[idx], sample_weight=w_tr[idx])

            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else 0.5
            fold_aucs.append(auc)
            fold_importances.append(dict(zip(cols, model.feature_importances_)))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results[exp_name] = {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_aucs": fold_aucs,
            "importances": fold_importances,
        }
        print(f"\n  {exp_name}: AUC={mean_auc:.4f} ± {std_auc:.4f}")
        print(f"    폴드별: {[round(a, 4) for a in fold_aucs]}")

        # 실험 A에서만 feature importance top 10 출력
        if exp_name.startswith("A"):
            avg_imp = {}
            for imp in fold_importances:
                for k, v in imp.items():
                    avg_imp[k] = avg_imp.get(k, 0) + v / len(fold_importances)
            top10 = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"    Feature Importance Top 10:")
            for feat_name, imp_val in top10:
                marker = " ← 주의" if feat_name in ("signal_strength", "side") else ""
                print(f"      {feat_name:<25} {imp_val:>8.1f}{marker}")

    # 드롭 분석
    auc_a = results["A (전체 피처)"]["mean_auc"]
    auc_b = results["B (-signal_strength)"]["mean_auc"]
    auc_c = results["C (-signal_strength, -side)"]["mean_auc"]
    drop_ab = auc_a - auc_b
    drop_ac = auc_a - auc_c

    print(f"\n{'='*64}")
    print(f"  드롭 분석")
    print(f"{'='*64}")
    print(f"  A → B (signal_strength 제거): {drop_ab:+.4f}")
    print(f"  A → C (signal_strength + side 제거): {drop_ac:+.4f}")
    print(f"{'─'*64}")

    if drop_ac <= 0.05:
        verdict = "✅ ML 필터 가치 있음 (다른 피처가 충분히 기여)"
    elif drop_ac <= 0.10:
        verdict = "⚠️ 조건부 투입 (signal_strength 의존도 높지만 다른 피처도 기여)"
    else:
        verdict = "❌ 재설계 필요 (사실상 점수 재확인기)"
    print(f"  판정: {verdict}")
    print(f"{'='*64}\n")
```

- [ ] **Step 2: CLI에 --ablation 플래그 추가**

`scripts/train_model.py`의 `main()` 내 argparse에:

```python
parser.add_argument("--ablation", action="store_true",
                    help="Feature ablation 실험 (signal_strength/side 의존도 진단)")
```

main() 분기에 추가:

```python
if args.ablation:
    ablation(
        args.data, time_weight_decay=args.decay,
        tuned_params_path=args.tuned_params,
        atr_sl_mult=args.sl_mult, atr_tp_mult=args.tp_mult,
    )
```

기존 `elif args.compare:` 앞에 배치.

- [ ] **Step 3: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS

- [ ] **Step 4: 커밋**

```bash
git add scripts/train_model.py
git commit -m "feat(ml): add --ablation CLI for signal_strength/side dependency diagnosis"
```

---

### Task 3: CLAUDE.md 업데이트

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: plan history 업데이트**

```markdown
| 2026-03-21 | `purged-gap-and-ablation` (plan) | Completed |
```

- [ ] **Step 2: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: update plan history with purged-gap-and-ablation"
```

---

## 구현 후 실행 가이드

구현 완료 후 다음 순서로 실행:

```bash
# 1. Purged gap 적용된 Walk-Forward (심볼별)
python scripts/train_model.py --symbol XRPUSDT --wf
python scripts/train_model.py --symbol SOLUSDT --wf
python scripts/train_model.py --symbol DOGEUSDT --wf

# 2. Ablation 실험 (심볼별)
python scripts/train_model.py --symbol XRPUSDT --ablation
python scripts/train_model.py --symbol SOLUSDT --ablation
python scripts/train_model.py --symbol DOGEUSDT --ablation
```

결과를 보고 판단:
- Purged AUC가 0.85+ 유지되면 모델 유효
- A→C 드롭이 0.05 이내면 ML 필터 실전 투입 가치 있음
- 두 조건 모두 충족 시 PF 계산(Task 미포함, 별도 판단 후 추가)으로 진행
