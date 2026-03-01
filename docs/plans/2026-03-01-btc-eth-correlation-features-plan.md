# BTC/ETH 상관관계 피처 추가 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** BTC/ETH 캔들 데이터를 XRP ML 필터의 추가 피처(21개)로 활용해 모델 예측 정확도를 향상시킨다.

**Architecture:** 바이낸스 Combined WebSocket으로 XRP/BTC/ETH 3개 심볼을 단일 연결로 수신하고, XRP 캔들이 닫힐 때 BTC/ETH 버퍼의 수익률·상대강도 8개 피처를 기존 13개 피처에 추가해 LightGBM에 전달한다. 학습 데이터도 3심볼을 타임스탬프 기준으로 병합해 동일한 21개 피처로 재학습한다.

**Tech Stack:** Python 3.12, python-binance (AsyncClient + BinanceSocketManager), LightGBM, pandas, joblib

---

## Task 1: `MultiSymbolStream` — Combined WebSocket으로 3심볼 수신

**Files:**
- Modify: `src/data_stream.py`
- Test: `tests/test_data_stream.py`

### Step 1: 실패하는 테스트 작성

`tests/test_data_stream.py` 파일에 아래 테스트를 추가한다.

```python
from src.data_stream import MultiSymbolStream

def test_multi_symbol_stream_has_three_buffers():
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
    )
    assert "xrpusdt" in stream.buffers
    assert "btcusdt" in stream.buffers
    assert "ethusdt" in stream.buffers

def test_multi_symbol_stream_get_dataframe_returns_none_when_empty():
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
    )
    assert stream.get_dataframe("XRPUSDT") is None

def test_multi_symbol_stream_get_dataframe_returns_df_when_full():
    import pandas as pd
    stream = MultiSymbolStream(
        symbols=["XRPUSDT", "BTCUSDT", "ETHUSDT"],
        interval="1m",
        buffer_size=200,
    )
    candle = {
        "timestamp": 1000, "open": 1.0, "high": 1.1,
        "low": 0.9, "close": 1.05, "volume": 100.0, "is_closed": True,
    }
    for i in range(50):
        c = candle.copy()
        c["timestamp"] = 1000 + i
        stream.buffers["xrpusdt"].append(c)
    df = stream.get_dataframe("XRPUSDT")
    assert df is not None
    assert len(df) == 50
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_data_stream.py::test_multi_symbol_stream_has_three_buffers -v
```

Expected: `FAILED` — `MultiSymbolStream` not defined

### Step 3: `MultiSymbolStream` 구현

`src/data_stream.py` 파일에 기존 `KlineStream` 클래스 아래에 추가한다.

```python
class MultiSymbolStream:
    """
    바이낸스 Combined WebSocket으로 여러 심볼의 캔들을 단일 연결로 수신한다.
    XRP 캔들이 닫힐 때 on_candle 콜백을 호출한다.
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "1m",
        buffer_size: int = 200,
        on_candle: Callable = None,
    ):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.on_candle = on_candle
        self.buffers: dict[str, deque] = {
            s: deque(maxlen=buffer_size) for s in self.symbols
        }
        # 첫 번째 심볼이 주 심볼 (XRP)
        self.primary_symbol = self.symbols[0]

    def parse_kline(self, msg: dict) -> dict:
        k = msg["k"]
        return {
            "timestamp": k["t"],
            "open":      float(k["o"]),
            "high":      float(k["h"]),
            "low":       float(k["l"]),
            "close":     float(k["c"]),
            "volume":    float(k["v"]),
            "is_closed": k["x"],
        }

    def handle_message(self, msg: dict):
        # Combined stream 메시지는 {"stream": "...", "data": {...}} 형태
        if "stream" in msg:
            data = msg["data"]
        else:
            data = msg

        if data.get("e") != "kline":
            return

        symbol = data["s"].lower()
        candle = self.parse_kline(data)

        if candle["is_closed"] and symbol in self.buffers:
            self.buffers[symbol].append(candle)
            if symbol == self.primary_symbol and self.on_candle:
                self.on_candle(candle)

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        key = symbol.lower()
        buf = self.buffers.get(key)
        if buf is None or len(buf) < 50:
            return None
        df = pd.DataFrame(list(buf))
        df.set_index("timestamp", inplace=True)
        return df

    async def _preload_history(self, client: AsyncClient, limit: int = 200):
        """REST API로 모든 심볼의 과거 캔들을 버퍼에 미리 채운다."""
        for symbol in self.symbols:
            logger.info(f"{symbol.upper()} 과거 캔들 {limit}개 로드 중...")
            klines = await client.futures_klines(
                symbol=symbol.upper(),
                interval=self.interval,
                limit=limit,
            )
            for k in klines[:-1]:
                self.buffers[symbol].append({
                    "timestamp": k[0],
                    "open":      float(k[1]),
                    "high":      float(k[2]),
                    "low":       float(k[3]),
                    "close":     float(k[4]),
                    "volume":    float(k[5]),
                    "is_closed": True,
                })
            logger.info(f"{symbol.upper()} {len(self.buffers[symbol])}개 로드 완료")

    async def start(self, api_key: str, api_secret: str):
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
        )
        await self._preload_history(client)
        bm = BinanceSocketManager(client)
        streams = [
            f"{s}@kline_{self.interval}" for s in self.symbols
        ]
        logger.info(f"Combined WebSocket 시작: {streams}")
        try:
            async with bm.futures_multiplex_socket(streams) as stream:
                while True:
                    msg = await stream.recv()
                    self.handle_message(msg)
        finally:
            await client.close_connection()
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_data_stream.py -v
```

Expected: 모든 테스트 PASS

### Step 5: 커밋

```bash
git add src/data_stream.py tests/test_data_stream.py
git commit -m "feat: add MultiSymbolStream for combined BTC/ETH/XRP WebSocket"
```

---

## Task 2: `build_features` — BTC/ETH 피처 8개 추가

**Files:**
- Modify: `src/ml_features.py`
- Test: `tests/test_ml_features.py`

### Step 1: 실패하는 테스트 작성

`tests/test_ml_features.py`에 아래 테스트를 추가한다.

```python
import pandas as pd
import numpy as np
from src.ml_features import build_features, FEATURE_COLS

def _make_df(n=10, base_price=1.0):
    """테스트용 더미 캔들 DataFrame 생성."""
    closes = [base_price * (1 + i * 0.001) for i in range(n)]
    return pd.DataFrame({
        "close": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "volume": [1000.0] * n,
        "rsi": [50.0] * n, "macd": [0.0] * n, "macd_signal": [0.0] * n,
        "macd_hist": [0.0] * n, "bb_upper": [c * 1.02 for c in closes],
        "bb_lower": [c * 0.98 for c in closes], "ema9": closes,
        "ema21": closes, "ema50": closes, "atr": [0.01] * n,
        "stoch_k": [50.0] * n, "stoch_d": [50.0] * n,
        "vol_ma20": [1000.0] * n,
    })

def test_build_features_with_btc_eth_has_21_features():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    assert len(features) == 21

def test_build_features_without_btc_eth_has_13_features():
    xrp_df = _make_df(10, base_price=1.0)
    features = build_features(xrp_df, "LONG")
    assert len(features) == 13

def test_build_features_btc_ret_1_correct():
    xrp_df = _make_df(10, base_price=1.0)
    btc_df = _make_df(10, base_price=50000.0)
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    btc_closes = btc_df["close"]
    expected_btc_ret_1 = (btc_closes.iloc[-1] - btc_closes.iloc[-2]) / btc_closes.iloc[-2]
    assert abs(features["btc_ret_1"] - expected_btc_ret_1) < 1e-6

def test_build_features_rs_zero_when_btc_ret_zero():
    xrp_df = _make_df(10, base_price=1.0)
    # BTC 가격이 변하지 않으면 ret=0, RS=0
    btc_df = _make_df(10, base_price=50000.0)
    btc_df["close"] = 50000.0  # 모든 캔들 동일
    eth_df = _make_df(10, base_price=3000.0)
    features = build_features(xrp_df, "LONG", btc_df=btc_df, eth_df=eth_df)
    assert features["xrp_btc_rs"] == 0.0

def test_feature_cols_has_21_items():
    from src.ml_features import FEATURE_COLS
    assert len(FEATURE_COLS) == 21
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_ml_features.py -v
```

Expected: 여러 테스트 FAIL

### Step 3: `ml_features.py` 수정

`src/ml_features.py` 전체를 아래로 교체한다.

```python
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    "btc_ret_1", "btc_ret_3", "btc_ret_5",
    "eth_ret_1", "eth_ret_3", "eth_ret_5",
    "xrp_btc_rs", "xrp_eth_rs",
]


def _calc_ret(closes: pd.Series, n: int) -> float:
    """n캔들 전 대비 수익률. 데이터 부족 시 0.0."""
    if len(closes) < n + 1:
        return 0.0
    prev = closes.iloc[-(n + 1)]
    return (closes.iloc[-1] - prev) / prev if prev != 0 else 0.0


def _calc_rs(xrp_ret: float, other_ret: float) -> float:
    """상대강도 = xrp_ret / other_ret. 분모 0이면 0.0."""
    if other_ret == 0.0:
        return 0.0
    return xrp_ret / other_ret


def build_features(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    기술 지표가 계산된 DataFrame의 마지막 행에서 ML 피처를 추출한다.
    btc_df, eth_df가 제공되면 21개 피처를, 없으면 13개 피처를 반환한다.
    signal: "LONG" | "SHORT"
    """
    last = df.iloc[-1]
    close = last["close"]

    bb_upper = last.get("bb_upper", close)
    bb_lower = last.get("bb_lower", close)
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    ema9  = last.get("ema9",  close)
    ema21 = last.get("ema21", close)
    ema50 = last.get("ema50", close)
    if ema9 > ema21 > ema50:
        ema_align = 1
    elif ema9 < ema21 < ema50:
        ema_align = -1
    else:
        ema_align = 0

    atr = last.get("atr", 0)
    atr_pct = atr / close if close > 0 else 0

    vol_ma20 = last.get("vol_ma20", last.get("volume", 1))
    vol_ratio = last["volume"] / vol_ma20 if vol_ma20 > 0 else 1.0

    closes = df["close"]
    ret_1 = _calc_ret(closes, 1)
    ret_3 = _calc_ret(closes, 3)
    ret_5 = _calc_ret(closes, 5)

    prev = df.iloc[-2] if len(df) >= 2 else last
    strength = 0
    rsi = last.get("rsi", 50)
    macd = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    prev_macd = prev.get("macd", 0)
    prev_macd_sig = prev.get("macd_signal", 0)
    stoch_k = last.get("stoch_k", 50)
    stoch_d = last.get("stoch_d", 50)

    if signal == "LONG":
        if rsi < 35: strength += 1
        if prev_macd < prev_macd_sig and macd > macd_sig: strength += 2
        if close < last.get("bb_lower", close): strength += 1
        if ema_align == 1: strength += 1
        if stoch_k < 20 and stoch_k > stoch_d: strength += 1
    else:
        if rsi > 65: strength += 1
        if prev_macd > prev_macd_sig and macd < macd_sig: strength += 2
        if close > last.get("bb_upper", close): strength += 1
        if ema_align == -1: strength += 1
        if stoch_k > 80 and stoch_k < stoch_d: strength += 1

    base = {
        "rsi":            float(rsi),
        "macd_hist":      float(last.get("macd_hist", 0)),
        "bb_pct":         float(bb_pct),
        "ema_align":      float(ema_align),
        "stoch_k":        float(stoch_k),
        "stoch_d":        float(last.get("stoch_d", 50)),
        "atr_pct":        float(atr_pct),
        "vol_ratio":      float(vol_ratio),
        "ret_1":          float(ret_1),
        "ret_3":          float(ret_3),
        "ret_5":          float(ret_5),
        "signal_strength": float(strength),
        "side":           1.0 if signal == "LONG" else 0.0,
    }

    if btc_df is not None and eth_df is not None:
        btc_ret_1 = _calc_ret(btc_df["close"], 1)
        btc_ret_3 = _calc_ret(btc_df["close"], 3)
        btc_ret_5 = _calc_ret(btc_df["close"], 5)
        eth_ret_1 = _calc_ret(eth_df["close"], 1)
        eth_ret_3 = _calc_ret(eth_df["close"], 3)
        eth_ret_5 = _calc_ret(eth_df["close"], 5)

        base.update({
            "btc_ret_1":  float(btc_ret_1),
            "btc_ret_3":  float(btc_ret_3),
            "btc_ret_5":  float(btc_ret_5),
            "eth_ret_1":  float(eth_ret_1),
            "eth_ret_3":  float(eth_ret_3),
            "eth_ret_5":  float(eth_ret_5),
            "xrp_btc_rs": float(_calc_rs(ret_1, btc_ret_1)),
            "xrp_eth_rs": float(_calc_rs(ret_1, eth_ret_1)),
        })

    return pd.Series(base)
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_ml_features.py -v
```

Expected: 모든 테스트 PASS

### Step 5: 커밋

```bash
git add src/ml_features.py tests/test_ml_features.py
git commit -m "feat: extend build_features to 21 features with BTC/ETH correlation"
```

---

## Task 3: `dataset_builder.py` — BTC/ETH 피처 벡터화 추가

**Files:**
- Modify: `src/dataset_builder.py`
- Test: `tests/test_dataset_builder.py`

### Step 1: 실패하는 테스트 작성

`tests/test_dataset_builder.py`에 아래 테스트를 추가한다.

```python
def test_generate_dataset_vectorized_with_btc_eth_has_21_feature_cols():
    """BTC/ETH DataFrame을 전달하면 결과 컬럼이 21개 피처 + label이어야 한다."""
    import pandas as pd
    import numpy as np
    from src.dataset_builder import generate_dataset_vectorized
    from src.ml_features import FEATURE_COLS

    np.random.seed(42)
    n = 500
    closes = np.cumprod(1 + np.random.randn(n) * 0.001) * 1.0
    xrp_df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes,
        "volume": np.random.rand(n) * 1000 + 500,
    })
    btc_df = xrp_df.copy() * 50000
    eth_df = xrp_df.copy() * 3000

    result = generate_dataset_vectorized(xrp_df, btc_df=btc_df, eth_df=eth_df)
    if not result.empty:
        assert set(FEATURE_COLS).issubset(set(result.columns))
        assert len(result.columns) == len(FEATURE_COLS) + 1  # +1 for label
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_dataset_builder.py::test_generate_dataset_vectorized_with_btc_eth_has_21_feature_cols -v
```

Expected: FAIL — `generate_dataset_vectorized()` does not accept btc_df/eth_df

### Step 3: `dataset_builder.py` 수정

`_calc_features_vectorized` 함수와 `generate_dataset_vectorized` 함수를 수정한다.

`_calc_features_vectorized` 시그니처와 반환부에 BTC/ETH 피처 추가:

```python
def _calc_features_vectorized(
    d: pd.DataFrame,
    signal_arr: np.ndarray,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # ... 기존 코드 유지 ...

    # BTC/ETH 피처 계산 (제공된 경우)
    if btc_df is not None and eth_df is not None:
        btc_ret_1 = btc_df["close"].pct_change(1).fillna(0).values
        btc_ret_3 = btc_df["close"].pct_change(3).fillna(0).values
        btc_ret_5 = btc_df["close"].pct_change(5).fillna(0).values
        eth_ret_1 = eth_df["close"].pct_change(1).fillna(0).values
        eth_ret_3 = eth_df["close"].pct_change(3).fillna(0).values
        eth_ret_5 = eth_df["close"].pct_change(5).fillna(0).values

        # 타임스탬프 정렬: XRP 인덱스 기준으로 BTC/ETH 값을 맞춤
        # 길이가 다를 경우 짧은 쪽에 맞춰 앞을 0으로 패딩
        def _align(arr: np.ndarray, target_len: int) -> np.ndarray:
            if len(arr) >= target_len:
                return arr[-target_len:]
            return np.concatenate([np.zeros(target_len - len(arr)), arr])

        n = len(d)
        btc_r1 = _align(btc_ret_1, n).astype(np.float32)
        btc_r3 = _align(btc_ret_3, n).astype(np.float32)
        btc_r5 = _align(btc_ret_5, n).astype(np.float32)
        eth_r1 = _align(eth_ret_1, n).astype(np.float32)
        eth_r3 = _align(eth_ret_3, n).astype(np.float32)
        eth_r5 = _align(eth_ret_5, n).astype(np.float32)

        xrp_r1 = ret_1.astype(np.float32)
        xrp_btc_rs = np.where(btc_r1 != 0, xrp_r1 / btc_r1, 0.0).astype(np.float32)
        xrp_eth_rs = np.where(eth_r1 != 0, xrp_r1 / eth_r1, 0.0).astype(np.float32)

        extra = pd.DataFrame({
            "btc_ret_1": btc_r1, "btc_ret_3": btc_r3, "btc_ret_5": btc_r5,
            "eth_ret_1": eth_r1, "eth_ret_3": eth_r3, "eth_ret_5": eth_r5,
            "xrp_btc_rs": xrp_btc_rs, "xrp_eth_rs": xrp_eth_rs,
        }, index=d.index)
        result = pd.concat([result, extra], axis=1)  # result는 기존 13개 피처 DataFrame

    return result
```

`generate_dataset_vectorized` 시그니처 변경:

```python
def generate_dataset_vectorized(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # ...
    feat_all = _calc_features_vectorized(d, signal_arr, btc_df=btc_df, eth_df=eth_df)
    # ...
    feat_final = feat_all.iloc[final_idx][FEATURE_COLS].copy()
    # ...
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_dataset_builder.py -v
```

Expected: 모든 테스트 PASS

### Step 5: 커밋

```bash
git add src/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: add BTC/ETH features to vectorized dataset builder"
```

---

## Task 4: `fetch_history.py` — 3심볼 동시 수집 및 병합

**Files:**
- Modify: `scripts/fetch_history.py`

### Step 1: 수정 내용

`fetch_history.py`의 `main()` 함수를 수정해 `--symbols` 인자로 여러 심볼을 받고, 타임스탬프 기준 inner join으로 병합 후 저장한다.

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["XRPUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days",     type=int, default=90)
    parser.add_argument("--output",   default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()

    if len(args.symbols) == 1:
        # 단일 심볼: 기존 동작 유지
        df = asyncio.run(fetch_klines(args.symbols[0], args.interval, args.days))
        df.to_parquet(args.output)
        print(f"저장 완료: {args.output} ({len(df)}행)")
    else:
        # 멀티 심볼: 각각 수집 후 병합
        dfs = {}
        for symbol in args.symbols:
            print(f"{symbol} 수집 중...")
            dfs[symbol] = asyncio.run(fetch_klines(symbol, args.interval, args.days))

        # 타임스탬프 기준 inner join
        primary = args.symbols[0]
        merged = dfs[primary].copy()
        for symbol in args.symbols[1:]:
            suffix = "_" + symbol.lower().replace("usdt", "")
            merged = merged.join(
                dfs[symbol].add_suffix(suffix),
                how="inner",
            )

        output = args.output.replace("xrpusdt", "combined")
        merged.to_parquet(output)
        print(f"병합 저장 완료: {output} ({len(merged)}행, {len(merged.columns)}컬럼)")
```

### Step 2: 동작 확인 (dry run — API 키 없이 구조만 확인)

```bash
python scripts/fetch_history.py --help
```

Expected: `--symbols` 인자가 출력됨

### Step 3: 커밋

```bash
git add scripts/fetch_history.py
git commit -m "feat: fetch_history supports multi-symbol collection and merge"
```

---

## Task 5: `train_model.py` — 병합 데이터셋으로 21피처 학습

**Files:**
- Modify: `scripts/train_model.py`

### Step 1: 수정 내용

`train()` 함수가 병합된 parquet을 받아 BTC/ETH 컬럼을 분리해 `generate_dataset_vectorized`에 전달하도록 수정한다.

```python
def train(data_path: str):
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
    dataset = generate_dataset_vectorized(df, btc_df=btc_df, eth_df=eth_df)

    # ... 이하 기존 학습 코드 동일 (X = dataset[FEATURE_COLS] 부분에서 자동으로 21개 사용) ...
```

### Step 2: 커밋

```bash
git add scripts/train_model.py
git commit -m "feat: train_model uses merged dataset with BTC/ETH features"
```

---

## Task 6: `bot.py` — `MultiSymbolStream` 연결 및 피처 전달

**Files:**
- Modify: `src/bot.py`
- Test: `tests/test_bot.py`

### Step 1: 실패하는 테스트 작성

`tests/test_bot.py`에 아래 테스트를 추가한다.

```python
def test_bot_uses_multi_symbol_stream():
    from src.bot import TradingBot
    from src.config import Config
    from src.data_stream import MultiSymbolStream

    config = Config()
    bot = TradingBot(config)
    assert isinstance(bot.stream, MultiSymbolStream)

def test_bot_stream_has_btc_eth_buffers():
    from src.bot import TradingBot
    from src.config import Config

    config = Config()
    bot = TradingBot(config)
    assert "btcusdt" in bot.stream.buffers
    assert "ethusdt" in bot.stream.buffers
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_bot.py::test_bot_uses_multi_symbol_stream -v
```

Expected: FAIL

### Step 3: `bot.py` 수정

`__init__` 에서 `KlineStream` → `MultiSymbolStream`으로 교체하고, `process_candle`에 BTC/ETH df를 전달한다.

```python
# import 변경
from src.data_stream import MultiSymbolStream  # KlineStream 대신

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = RiskManager(config)
        self.ml_filter = MLFilter()
        self.current_trade_side: str | None = None
        self.stream = MultiSymbolStream(
            symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
            interval="1m",
            on_candle=self._on_candle_closed,
        )

    def _on_candle_closed(self, candle: dict):
        xrp_df = self.stream.get_dataframe(self.config.symbol)
        btc_df = self.stream.get_dataframe("BTCUSDT")
        eth_df = self.stream.get_dataframe("ETHUSDT")
        if xrp_df is not None:
            asyncio.create_task(self.process_candle(xrp_df, btc_df=btc_df, eth_df=eth_df))

    async def process_candle(
        self,
        df,
        btc_df=None,
        eth_df=None,
    ):
        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        signal = ind.get_signal(df_with_indicators)

        if signal != "HOLD" and self.ml_filter.is_model_loaded():
            features = build_features(df_with_indicators, signal, btc_df=btc_df, eth_df=eth_df)
            if not self.ml_filter.should_enter(features):
                logger.info(f"ML 필터 차단: {signal} 신호 무시")
                signal = "HOLD"

        # ... 이하 기존 코드 동일 ...

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_bot.py -v
```

Expected: 모든 테스트 PASS

### Step 5: 커밋

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: bot uses MultiSymbolStream and passes BTC/ETH df to build_features"
```

---

## Task 7: 전체 테스트 통과 및 재학습 실행

### Step 1: 전체 테스트 실행

```bash
pytest tests/ -v
```

Expected: 모든 테스트 PASS

### Step 2: 3심볼 데이터 수집

```bash
python scripts/fetch_history.py \
  --symbols XRPUSDT BTCUSDT ETHUSDT \
  --days 90 \
  --output data/xrpusdt_1m.parquet
```

Expected: `data/combined_1m.parquet` 생성

### Step 3: 21피처 모델 재학습

```bash
python scripts/train_model.py --data data/combined_1m.parquet
```

Expected: `models/lgbm_filter.pkl` 교체, AUC 출력

### Step 4: 최종 커밋

```bash
git add models/training_log.json
git commit -m "chore: retrain model with 21 BTC/ETH correlation features"
```

---

## 실행 순서 요약

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7
(Stream)  (피처)  (데이터셋)  (수집)   (학습)    (봇)    (검증)
```

각 Task는 독립적으로 테스트 가능하며, Task 7 이전까지는 기존 봇이 정상 동작한다.
