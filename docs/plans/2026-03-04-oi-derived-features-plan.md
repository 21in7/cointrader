# OI 파생 피처 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** OI 파생 피처 2개(`oi_change_ma5`, `oi_price_spread`)를 추가하고, 기존 대비 성능을 자동 비교하며, OI 장기 수집 스크립트를 만든다.

**Architecture:** dataset_builder.py에 파생 피처 계산 추가 → ml_features.py에 FEATURE_COLS/build_features 확장 → train_model.py에 --compare 플래그로 A/B 비교 → bot.py에 OI deque 히스토리 관리 및 cold start → scripts/collect_oi.py 신규

**Tech Stack:** Python, LightGBM, pandas, numpy, Binance REST API

---

### Task 1: dataset_builder.py — OI 파생 피처 계산

**Files:**
- Modify: `src/dataset_builder.py:277-291` (OI/FR 피처 계산 블록)
- Test: `tests/test_dataset_builder.py`

**Step 1: Write failing tests**

`tests/test_dataset_builder.py` 끝에 추가:

```python
def test_oi_derived_features_present():
    """OI 파생 피처 2개가 결과에 포함되어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 300
    np.random.seed(42)
    df = pd.DataFrame({
        "open":      np.random.uniform(1, 2, n),
        "high":      np.random.uniform(2, 3, n),
        "low":       np.random.uniform(0.5, 1, n),
        "close":     np.random.uniform(1, 2, n),
        "volume":    np.random.uniform(1000, 5000, n),
        "oi_change": np.concatenate([np.zeros(100), np.random.uniform(-0.05, 0.05, 200)]),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    assert "oi_change_ma5" in feat.columns, "oi_change_ma5 컬럼이 없음"
    assert "oi_price_spread" in feat.columns, "oi_price_spread 컬럼이 없음"


def test_oi_derived_features_nan_when_no_oi():
    """oi_change 컬럼이 없으면 파생 피처도 nan이어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 200
    np.random.seed(0)
    df = pd.DataFrame({
        "open":   np.random.uniform(1, 2, n),
        "high":   np.random.uniform(2, 3, n),
        "low":    np.random.uniform(0.5, 1, n),
        "close":  np.random.uniform(1, 2, n),
        "volume": np.random.uniform(1000, 5000, n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    assert feat["oi_change_ma5"].isna().all(), "oi_change 컬럼 없을 때 oi_change_ma5는 전부 nan이어야 함"
    assert feat["oi_price_spread"].isna().all(), "oi_change 컬럼 없을 때 oi_price_spread는 전부 nan이어야 함"


def test_oi_price_spread_is_continuous():
    """oi_price_spread는 바이너리가 아닌 연속값이어야 한다."""
    import numpy as np
    import pandas as pd
    from src.dataset_builder import _calc_features_vectorized, _calc_signals, _calc_indicators

    n = 300
    np.random.seed(42)
    df = pd.DataFrame({
        "open":      np.random.uniform(1, 2, n),
        "high":      np.random.uniform(2, 3, n),
        "low":       np.random.uniform(0.5, 1, n),
        "close":     np.random.uniform(1, 2, n),
        "volume":    np.random.uniform(1000, 5000, n),
        "oi_change": np.random.uniform(-0.05, 0.05, n),
    })
    d = _calc_indicators(df)
    sig = _calc_signals(d)
    feat = _calc_features_vectorized(d, sig)

    valid = feat["oi_price_spread"].dropna()
    assert len(valid.unique()) > 2, "oi_price_spread는 연속값이어야 함 (2개 초과 유니크 값)"
```

**Step 2: Run tests to verify they fail**

Run: `bash scripts/run_tests.sh -k "oi_derived"`
Expected: FAIL — `oi_change_ma5`, `oi_price_spread` 컬럼 없음

**Step 3: Implement in dataset_builder.py**

`src/dataset_builder.py:277-291` (기존 OI/FR 블록) 뒤에 파생 피처 추가:

```python
    # OI 변화율 / 펀딩비 피처
    # 컬럼 없으면 전체 nan, 있으면 0.0 구간(데이터 미제공 구간)을 nan으로 마스킹
    if "oi_change" in d.columns:
        oi_raw = np.where(d["oi_change"].values == 0.0, np.nan, d["oi_change"].values)
    else:
        oi_raw = np.full(len(d), np.nan)

    if "funding_rate" in d.columns:
        fr_raw = np.where(d["funding_rate"].values == 0.0, np.nan, d["funding_rate"].values)
    else:
        fr_raw = np.full(len(d), np.nan)

    oi_z = _rolling_zscore(oi_raw.astype(np.float64), window=96)
    result["oi_change"]    = oi_z
    result["funding_rate"] = _rolling_zscore(fr_raw.astype(np.float64), window=96)

    # --- OI 파생 피처 ---
    # 1. oi_change_ma5: OI 변화율의 5캔들 이동평균 (단기 추세)
    oi_series = pd.Series(oi_raw.astype(np.float64))
    oi_ma5_raw = oi_series.rolling(window=5, min_periods=1).mean().values
    result["oi_change_ma5"] = _rolling_zscore(oi_ma5_raw, window=96)

    # 2. oi_price_spread: z-scored OI 변화율 - z-scored 가격 수익률 (연속값)
    #    양수: OI가 가격 대비 강세 (자금 유입)
    #    음수: OI가 가격 대비 약세 (자금 유출)
    result["oi_price_spread"] = oi_z - ret_1_z
```

주의: 기존 `oi_change`와 `funding_rate`의 window도 288→96으로 변경. `oi_z` 변수를 재사용하여 `oi_price_spread` 계산. `ret_1_z`는 이미 위에서 계산됨 (line 181).

**Step 4: Update OPTIONAL_COLS in generate_dataset_vectorized**

`src/dataset_builder.py:387` 수정:

```python
    OPTIONAL_COLS = {"oi_change", "funding_rate", "oi_change_ma5", "oi_price_spread"}
```

**Step 5: Run tests to verify they pass**

Run: `bash scripts/run_tests.sh -k "oi_derived"`
Expected: 3 tests PASS

**Step 6: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: All existing tests PASS (기존 oi_change/funding_rate 테스트 포함)

**Step 7: Commit**

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: add oi_change_ma5 and oi_price_spread derived features to dataset builder"
```

---

### Task 2: ml_features.py — FEATURE_COLS 및 build_features() 확장

**Files:**
- Modify: `src/ml_features.py:4-15` (FEATURE_COLS), `src/ml_features.py:33-139` (build_features)
- Test: `tests/test_ml_features.py`

**Step 1: Write failing tests**

`tests/test_ml_features.py` 끝에 추가:

```python
def test_feature_cols_has_26_items():
    from src.ml_features import FEATURE_COLS
    assert len(FEATURE_COLS) == 26


def test_build_features_with_oi_derived_params():
    """oi_change_ma5, oi_price_spread 파라미터가 피처에 반영된다."""
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(
        xrp_df, "LONG",
        btc_df=btc_df, eth_df=eth_df,
        oi_change=0.05, funding_rate=0.0002,
        oi_change_ma5=0.03, oi_price_spread=0.12,
    )
    assert features["oi_change_ma5"] == pytest.approx(0.03)
    assert features["oi_price_spread"] == pytest.approx(0.12)


def test_build_features_oi_derived_defaults_to_zero():
    """oi_change_ma5, oi_price_spread 미제공 시 0.0으로 채워진다."""
    xrp_df = _make_df(10, base_price=1.0)
    features = build_features(xrp_df, "LONG")
    assert features["oi_change_ma5"] == pytest.approx(0.0)
    assert features["oi_price_spread"] == pytest.approx(0.0)
```

기존 테스트 수정:
- `test_feature_cols_has_24_items` → 삭제 또는 숫자를 26으로 변경
- `test_build_features_with_btc_eth_has_24_features` → `assert len(features) == 26`
- `test_build_features_without_btc_eth_has_16_features` → `assert len(features) == 18`

**Step 2: Run tests to verify they fail**

Run: `bash scripts/run_tests.sh -k "test_feature_cols_has_26 or test_build_features_oi_derived"`
Expected: FAIL

**Step 3: Implement**

`src/ml_features.py` FEATURE_COLS 수정 (line 4-15):

```python
FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    "btc_ret_1", "btc_ret_3", "btc_ret_5",
    "eth_ret_1", "eth_ret_3", "eth_ret_5",
    "xrp_btc_rs", "xrp_eth_rs",
    # 시장 미시구조: OI 변화율(z-score), 펀딩비(z-score)
    "oi_change", "funding_rate",
    # OI 파생 피처
    "oi_change_ma5", "oi_price_spread",
    "adx",
]
```

`build_features()` 시그니처 수정 (line 33-40):

```python
def build_features(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    oi_change: float | None = None,
    funding_rate: float | None = None,
    oi_change_ma5: float | None = None,
    oi_price_spread: float | None = None,
) -> pd.Series:
```

`build_features()` 끝부분 (line 134-138) 수정:

```python
    base["oi_change"]       = float(oi_change)       if oi_change       is not None else 0.0
    base["funding_rate"]    = float(funding_rate)    if funding_rate    is not None else 0.0
    base["oi_change_ma5"]   = float(oi_change_ma5)   if oi_change_ma5   is not None else 0.0
    base["oi_price_spread"] = float(oi_price_spread) if oi_price_spread is not None else 0.0
    base["adx"] = float(last.get("adx", 0))
```

**Step 4: Run tests**

Run: `bash scripts/run_tests.sh -k "test_ml_features"`
Expected: All PASS

**Step 5: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: All PASS (test_dataset_builder의 FEATURE_COLS 참조도 26개로 통과)

**Step 6: Commit**

```bash
git add src/ml_features.py tests/test_ml_features.py
git commit -m "feat: add oi_change_ma5 and oi_price_spread to FEATURE_COLS and build_features"
```

---

### Task 3: train_model.py — --compare A/B 비교 모드

**Files:**
- Modify: `scripts/train_model.py:425-452` (main, argparse)
- Test: 수동 실행 확인 (학습 스크립트는 통합 테스트)

**Step 1: Implement compare function**

`scripts/train_model.py`에 `compare()` 함수 추가 (train() 함수 뒤):

```python
def compare(data_path: str, time_weight_decay: float = 2.0, tuned_params_path: str | None = None):
    """기존 피처 vs OI 파생 피처 추가 버전 A/B 비교."""
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
    )

    if dataset.empty:
        raise ValueError("데이터셋 생성 실패")

    lgbm_params, weight_scale = _load_lgbm_params(tuned_params_path)

    # Baseline: OI 파생 피처 제외
    BASELINE_EXCLUDE = {"oi_change_ma5", "oi_price_spread"}
    baseline_cols = [c for c in FEATURE_COLS if c in dataset.columns and c not in BASELINE_EXCLUDE]
    new_cols = [c for c in FEATURE_COLS if c in dataset.columns]

    results = {}
    for label, cols in [("Baseline (24)", baseline_cols), ("New (26)", new_cols)]:
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

        import warnings
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
    print(f"\n{'지표':<20} {'Baseline (24)':>15} {'New (26)':>15} {'Delta':>10}")
    print("-" * 62)
    for metric in ["auc", "precision", "recall", "threshold"]:
        b = results["Baseline (24)"][metric]
        n = results["New (26)"][metric]
        d = n - b
        sign = "+" if d > 0 else ""
        print(f"{metric:<20} {b:>15.4f} {n:>15.4f} {sign}{d:>9.4f}")

    n_val = results["Baseline (24)"]["n_val"]
    n_pos = results["Baseline (24)"]["n_val_pos"]
    print(f"\n검증셋: n={n_val} (양성={n_pos}, 음성={n_val - n_pos})")
    print("⚠ 30일 데이터 기반 — 방향성 참고용\n")

    print("Feature Importance Top 10 (New):")
    for feat_name, imp_val in results["New (26)"]["top10"]:
        marker = " ← NEW" if feat_name in BASELINE_EXCLUDE else ""
        print(f"  {feat_name:<25} {imp_val:>6}{marker}")
```

**Step 2: Add --compare flag to argparse**

`scripts/train_model.py` main() 함수의 argparse에 추가:

```python
    parser.add_argument("--compare", action="store_true",
                        help="OI 파생 피처 추가 전후 A/B 성능 비교")
```

main() 분기에 추가:

```python
    if args.compare:
        compare(args.data, time_weight_decay=args.decay, tuned_params_path=args.tuned_params)
    elif args.wf:
        ...
```

**Step 3: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: add --compare flag for OI derived features A/B comparison"
```

---

### Task 4: bot.py — OI deque 히스토리 및 실시간 파생 피처 공급

**Files:**
- Modify: `src/bot.py:15-31` (init), `src/bot.py:60-83` (fetch/calc), `src/bot.py:110-114,287-291` (build_features 호출)
- Modify: `src/exchange.py` (get_oi_history 추가)
- Test: `tests/test_bot.py`

**Step 1: Write failing tests**

`tests/test_bot.py` 끝에 추가:

```python
def test_bot_has_oi_history_deque(config):
    """봇이 OI 히스토리 deque를 가져야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    from collections import deque
    assert isinstance(bot._oi_history, deque)
    assert bot._oi_history.maxlen == 5


@pytest.mark.asyncio
async def test_init_oi_history_fills_deque(config):
    """_init_oi_history가 deque를 채워야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    bot.exchange.get_oi_history = AsyncMock(return_value=[0.01, -0.02, 0.03, -0.01, 0.02])
    await bot._init_oi_history()
    assert len(bot._oi_history) == 5


@pytest.mark.asyncio
async def test_fetch_microstructure_returns_derived_features(config):
    """_fetch_market_microstructure가 oi_change_ma5와 oi_price_spread를 반환해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)
    bot.exchange.get_open_interest = AsyncMock(return_value=5000000.0)
    bot.exchange.get_funding_rate = AsyncMock(return_value=0.0001)
    bot._prev_oi = 4900000.0
    bot._oi_history.extend([0.01, -0.02, 0.03, -0.01])
    bot._latest_ret_1 = 0.01

    result = await bot._fetch_market_microstructure()
    assert len(result) == 4  # oi_change, funding_rate, oi_change_ma5, oi_price_spread
```

**Step 2: Run tests to verify they fail**

Run: `bash scripts/run_tests.sh -k "oi_history or fetch_microstructure_returns_derived"`
Expected: FAIL

**Step 3: Implement exchange.get_oi_history()**

`src/exchange.py`에 추가:

```python
    async def get_oi_history(self, limit: int = 5) -> list[float]:
        """최근 OI 변화율 히스토리를 조회한다 (봇 초기화용). 실패 시 빈 리스트."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.client.futures_open_interest_hist(
                    symbol=self.config.symbol, period="15m", limit=limit + 1,
                ),
            )
            if len(result) < 2:
                return []
            oi_values = [float(r["sumOpenInterest"]) for r in result]
            changes = []
            for i in range(1, len(oi_values)):
                if oi_values[i - 1] > 0:
                    changes.append((oi_values[i] - oi_values[i - 1]) / oi_values[i - 1])
                else:
                    changes.append(0.0)
            return changes
        except Exception as e:
            logger.warning(f"OI 히스토리 조회 실패 (무시): {e}")
            return []
```

**Step 4: Implement bot.py changes**

`src/bot.py` `__init__` 수정:

```python
from collections import deque

# __init__에 추가:
        self._oi_history: deque = deque(maxlen=5)
        self._latest_ret_1: float = 0.0  # 최신 가격 수익률 (oi_price_spread용)
```

`_init_oi_history()` 추가:

```python
    async def _init_oi_history(self) -> None:
        """봇 시작 시 최근 OI 변화율 히스토리를 조회하여 deque를 채운다."""
        try:
            changes = await self.exchange.get_oi_history(limit=5)
            for c in changes:
                self._oi_history.append(c)
            if changes:
                self._prev_oi = None  # 다음 실시간 OI로 갱신
            logger.info(f"OI 히스토리 초기화: {len(self._oi_history)}개")
        except Exception as e:
            logger.warning(f"OI 히스토리 초기화 실패 (무시): {e}")
```

`_fetch_market_microstructure()` 수정 — 4-tuple 반환:

```python
    async def _fetch_market_microstructure(self) -> tuple[float, float, float, float]:
        """OI 변화율, 펀딩비, OI MA5, OI-가격 스프레드를 실시간으로 조회한다."""
        oi_val, fr_val = await asyncio.gather(
            self.exchange.get_open_interest(),
            self.exchange.get_funding_rate(),
            return_exceptions=True,
        )
        if isinstance(oi_val, (int, float)) and oi_val > 0:
            oi_change = self._calc_oi_change(float(oi_val))
        else:
            oi_change = 0.0
        fr_float = float(fr_val) if isinstance(fr_val, (int, float)) else 0.0

        # OI 히스토리 업데이트 및 MA5 계산
        self._oi_history.append(oi_change)
        oi_ma5 = sum(self._oi_history) / len(self._oi_history) if self._oi_history else 0.0

        # OI-가격 스프레드 (단순 차이, 실시간에서는 z-score 없이 raw)
        oi_price_spread = oi_change - self._latest_ret_1

        logger.debug(
            f"OI={oi_val}, OI변화율={oi_change:.6f}, 펀딩비={fr_float:.6f}, "
            f"OI_MA5={oi_ma5:.6f}, OI_Price_Spread={oi_price_spread:.6f}"
        )
        return oi_change, fr_float, oi_ma5, oi_price_spread
```

`process_candle()` 수정:

```python
        # 캔들 마감 시 가격 수익률 계산 (oi_price_spread용)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            curr_close = df["close"].iloc[-1]
            self._latest_ret_1 = (curr_close - prev_close) / prev_close if prev_close != 0 else 0.0

        oi_change, funding_rate, oi_ma5, oi_price_spread = await self._fetch_market_microstructure()
```

모든 `build_features()` 호출에 새 파라미터 추가:

```python
            features = build_features(
                df_with_indicators, signal,
                btc_df=btc_df, eth_df=eth_df,
                oi_change=oi_change, funding_rate=funding_rate,
                oi_change_ma5=oi_ma5, oi_price_spread=oi_price_spread,
            )
```

`_close_and_reenter()` 시그니처도 확장:

```python
    async def _close_and_reenter(
        self,
        position: dict,
        signal: str,
        df,
        btc_df=None,
        eth_df=None,
        oi_change: float = 0.0,
        funding_rate: float = 0.0,
        oi_change_ma5: float = 0.0,
        oi_price_spread: float = 0.0,
    ) -> None:
```

`run()` 수정 — `_init_oi_history()` 호출 추가:

```python
    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        await self._init_oi_history()
        ...
```

**Step 5: Run tests**

Run: `bash scripts/run_tests.sh -k "test_bot"`
Expected: All PASS

**Step 6: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/bot.py src/exchange.py tests/test_bot.py
git commit -m "feat: add OI history deque, cold start init, and derived features to bot runtime"
```

---

### Task 5: scripts/collect_oi.py — OI 장기 수집 스크립트

**Files:**
- Create: `scripts/collect_oi.py`

**Step 1: Implement**

```python
"""
OI 장기 수집 스크립트.
15분마다 cron 실행하여 Binance OI를 data/oi_history.parquet에 누적한다.

사용법:
  python scripts/collect_oi.py
  python scripts/collect_oi.py --symbol XRPUSDT

crontab 예시:
  */15 * * * * cd /path/to/cointrader && .venv/bin/python scripts/collect_oi.py >> logs/collect_oi.log 2>&1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os

load_dotenv()

OI_PATH = Path("data/oi_history.parquet")


def collect(symbol: str = "XRPUSDT"):
    client = Client(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )

    result = client.futures_open_interest(symbol=symbol)
    oi_value = float(result["openInterest"])
    ts = datetime.now(timezone.utc)

    new_row = pd.DataFrame([{
        "timestamp": ts,
        "symbol": symbol,
        "open_interest": oi_value,
    }])

    if OI_PATH.exists():
        existing = pd.read_parquet(OI_PATH)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        OI_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined = new_row

    combined.to_parquet(OI_PATH, index=False)
    print(f"[{ts.isoformat()}] OI={oi_value:.2f} → {OI_PATH}")


def main():
    parser = argparse.ArgumentParser(description="OI 장기 수집")
    parser.add_argument("--symbol", default="XRPUSDT")
    args = parser.parse_args()
    collect(symbol=args.symbol)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/collect_oi.py
git commit -m "feat: add OI long-term collection script for cron-based data accumulation"
```

---

### Task 6: 기존 테스트 수정 및 전체 검증

**Files:**
- Modify: `tests/test_ml_features.py` (피처 수 변경)
- Modify: `tests/test_bot.py` (기존 OI 테스트가 4-tuple 반환에 호환되도록)

**Step 1: Fix test_ml_features.py assertions**

- `test_feature_cols_has_24_items` → 26으로 변경
- `test_build_features_with_btc_eth_has_24_features` → 26
- `test_build_features_without_btc_eth_has_16_features` → 18

**Step 2: Fix test_bot.py**

기존 `test_process_candle_fetches_oi_and_funding` 등에서 `_fetch_market_microstructure` 반환값이 4-tuple이 되므로 mock 반환값 수정:

```python
bot._fetch_market_microstructure = AsyncMock(return_value=(0.02, 0.0001, 0.015, 0.01))
```

또는 `_fetch_market_microstructure`를 mock하지 않는 테스트는 exchange mock이 정상이면 자동 통과.

**Step 3: Run full test suite**

Run: `bash scripts/run_tests.sh`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_ml_features.py tests/test_bot.py
git commit -m "test: update test assertions for 26-feature model and 4-tuple microstructure"
```

---

### Task 7: CLAUDE.md 업데이트

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update plan table**

CLAUDE.md의 plan history 테이블에 추가:

```
| 2026-03-04 | `oi-derived-features` (design + plan) | In Progress |
```

ml_features.py 설명도 24→26개로 갱신.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with OI derived features plan status"
```
