# 실시간 OI/펀딩비 피처 수집 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 실시간 봇에서 캔들 마감 시 바이낸스 REST API로 현재 OI와 펀딩비를 수집해 ML 피처에 실제 값을 넣어 학습-추론 불일치(train-serve skew)를 해소한다.

**Architecture:**
- `exchange.py`에 `get_open_interest()`, `get_funding_rate()` 메서드 추가 (REST 호출)
- `bot.py`의 `process_candle()`에서 캔들 마감 시 두 값을 조회하고 `build_features()` 호출 시 전달
- `ml_features.py`의 `build_features()`가 `oi_change`, `funding_rate` 파라미터를 받아 실제 값으로 채우도록 수정

**Tech Stack:** python-binance AsyncClient, aiohttp (이미 사용 중), pytest-asyncio

---

## Task 1: exchange.py — OI / 펀딩비 조회 메서드 추가

**Files:**
- Modify: `src/exchange.py`
- Test: `tests/test_exchange.py`

### Step 1: 실패 테스트 작성

`tests/test_exchange.py` 파일에 아래 테스트를 추가한다.

```python
@pytest.mark.asyncio
async def test_get_open_interest(exchange):
    """get_open_interest()가 float을 반환하는지 확인."""
    exchange.client.futures_open_interest = MagicMock(
        return_value={"openInterest": "123456.789"}
    )
    result = await exchange.get_open_interest()
    assert isinstance(result, float)
    assert result == pytest.approx(123456.789)


@pytest.mark.asyncio
async def test_get_funding_rate(exchange):
    """get_funding_rate()가 float을 반환하는지 확인."""
    exchange.client.futures_mark_price = MagicMock(
        return_value={"lastFundingRate": "0.0001"}
    )
    result = await exchange.get_funding_rate()
    assert isinstance(result, float)
    assert result == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_get_open_interest_error_returns_none(exchange):
    """API 오류 시 None 반환 확인."""
    from binance.exceptions import BinanceAPIException
    exchange.client.futures_open_interest = MagicMock(
        side_effect=BinanceAPIException(MagicMock(status_code=400), 400, '{"code":-1121,"msg":"Invalid symbol"}')
    )
    result = await exchange.get_open_interest()
    assert result is None


@pytest.mark.asyncio
async def test_get_funding_rate_error_returns_none(exchange):
    """API 오류 시 None 반환 확인."""
    from binance.exceptions import BinanceAPIException
    exchange.client.futures_mark_price = MagicMock(
        side_effect=BinanceAPIException(MagicMock(status_code=400), 400, '{"code":-1121,"msg":"Invalid symbol"}')
    )
    result = await exchange.get_funding_rate()
    assert result is None
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_exchange.py::test_get_open_interest tests/test_exchange.py::test_get_funding_rate -v
```

Expected: `FAILED` — `AttributeError: 'BinanceFuturesClient' object has no attribute 'get_open_interest'`

### Step 3: exchange.py에 메서드 구현

`src/exchange.py`의 `cancel_all_orders()` 메서드 아래에 추가한다.

```python
async def get_open_interest(self) -> float | None:
    """현재 미결제약정(OI)을 조회한다. 오류 시 None 반환."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: self.client.futures_open_interest(symbol=self.config.symbol),
        )
        return float(result["openInterest"])
    except Exception as e:
        logger.warning(f"OI 조회 실패 (무시): {e}")
        return None

async def get_funding_rate(self) -> float | None:
    """현재 펀딩비를 조회한다. 오류 시 None 반환."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: self.client.futures_mark_price(symbol=self.config.symbol),
        )
        return float(result["lastFundingRate"])
    except Exception as e:
        logger.warning(f"펀딩비 조회 실패 (무시): {e}")
        return None
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_exchange.py -v
```

Expected: 기존 테스트 포함 전체 PASS

### Step 5: 커밋

```bash
git add src/exchange.py tests/test_exchange.py
git commit -m "feat: add get_open_interest and get_funding_rate to BinanceFuturesClient"
```

---

## Task 2: ml_features.py — build_features()에 oi/funding 파라미터 추가

**Files:**
- Modify: `src/ml_features.py`
- Test: `tests/test_ml_features.py`

### Step 1: 실패 테스트 작성

`tests/test_ml_features.py`에 아래 테스트를 추가한다.

```python
def test_build_features_uses_provided_oi_funding(sample_df_with_indicators):
    """oi_change, funding_rate 파라미터가 제공되면 실제 값이 피처에 반영된다."""
    from src.ml_features import build_features
    feat = build_features(
        sample_df_with_indicators,
        signal="LONG",
        oi_change=0.05,
        funding_rate=0.0002,
    )
    assert feat["oi_change"] == pytest.approx(0.05)
    assert feat["funding_rate"] == pytest.approx(0.0002)


def test_build_features_defaults_to_zero_when_not_provided(sample_df_with_indicators):
    """oi_change, funding_rate 파라미터 미제공 시 0.0으로 채워진다."""
    from src.ml_features import build_features
    feat = build_features(sample_df_with_indicators, signal="LONG")
    assert feat["oi_change"] == pytest.approx(0.0)
    assert feat["funding_rate"] == pytest.approx(0.0)
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_ml_features.py::test_build_features_uses_provided_oi_funding -v
```

Expected: `FAILED` — `TypeError: build_features() got an unexpected keyword argument 'oi_change'`

### Step 3: ml_features.py 수정

`build_features()` 시그니처와 마지막 부분을 수정한다.

```python
def build_features(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    oi_change: float | None = None,
    funding_rate: float | None = None,
) -> pd.Series:
```

그리고 함수 끝의 `setdefault` 부분을 아래로 교체한다.

```python
    # 실시간에서 실제 값이 제공되면 사용, 없으면 0으로 채운다
    base["oi_change"]    = float(oi_change)    if oi_change    is not None else 0.0
    base["funding_rate"] = float(funding_rate) if funding_rate is not None else 0.0

    return pd.Series(base)
```

기존 코드:
```python
    # 실시간에서는 OI/펀딩비를 수집하지 않으므로 0으로 채워 학습 피처(23개)와 일치시킨다
    base.setdefault("oi_change", 0.0)
    base.setdefault("funding_rate", 0.0)

    return pd.Series(base)
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_ml_features.py -v
```

Expected: 전체 PASS

### Step 5: 커밋

```bash
git add src/ml_features.py tests/test_ml_features.py
git commit -m "feat: build_features accepts oi_change and funding_rate params"
```

---

## Task 3: bot.py — 캔들 마감 시 OI/펀딩비 조회 후 피처에 전달

**Files:**
- Modify: `src/bot.py`
- Test: `tests/test_bot.py`

### Step 1: 실패 테스트 작성

`tests/test_bot.py`에 아래 테스트를 추가한다.

```python
@pytest.mark.asyncio
async def test_process_candle_fetches_oi_and_funding(config, sample_df):
    """process_candle()이 OI와 펀딩비를 조회하고 build_features에 전달하는지 확인."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_balance = AsyncMock(return_value=1000.0)
    bot.exchange.get_position = AsyncMock(return_value=None)
    bot.exchange.place_order = AsyncMock(return_value={"orderId": "1"})
    bot.exchange.set_leverage = AsyncMock()
    bot.exchange.get_open_interest = AsyncMock(return_value=5000000.0)
    bot.exchange.get_funding_rate = AsyncMock(return_value=0.0001)

    with patch("src.bot.build_features") as mock_build:
        mock_build.return_value = pd.Series({col: 0.0 for col in __import__("src.ml_features", fromlist=["FEATURE_COLS"]).FEATURE_COLS})
        # ML 필터는 비활성화
        bot.ml_filter.is_model_loaded = MagicMock(return_value=False)
        await bot.process_candle(sample_df)

    # build_features가 oi_change, funding_rate 키워드 인자와 함께 호출됐는지 확인
    assert mock_build.called
    call_kwargs = mock_build.call_args.kwargs
    assert "oi_change" in call_kwargs
    assert "funding_rate" in call_kwargs
```

### Step 2: 테스트 실패 확인

```bash
pytest tests/test_bot.py::test_process_candle_fetches_oi_and_funding -v
```

Expected: `FAILED` — `AssertionError: assert 'oi_change' in {}`

### Step 3: bot.py 수정

`process_candle()` 메서드에서 OI/펀딩비를 조회하고 `build_features()`에 전달한다.

`process_candle()` 메서드 시작 부분에 OI/펀딩비 조회를 추가한다:

```python
async def process_candle(self, df, btc_df=None, eth_df=None):
    self.ml_filter.check_and_reload()

    if not self.risk.is_trading_allowed():
        logger.warning("리스크 한도 초과 - 거래 중단")
        return

    # 캔들 마감 시 OI/펀딩비 실시간 조회 (실패해도 0으로 폴백)
    oi_change, funding_rate = await self._fetch_market_microstructure()

    ind = Indicators(df)
    df_with_indicators = ind.calculate_all()
    raw_signal = ind.get_signal(df_with_indicators)
    # ... (이하 동일)
```

그리고 `build_features()` 호출 부분 두 곳을 모두 수정한다:

```python
features = build_features(
    df_with_indicators, signal,
    btc_df=btc_df, eth_df=eth_df,
    oi_change=oi_change, funding_rate=funding_rate,
)
```

`_fetch_market_microstructure()` 메서드를 추가한다:

```python
async def _fetch_market_microstructure(self) -> tuple[float, float]:
    """OI 변화율과 펀딩비를 실시간으로 조회한다. 실패 시 0.0으로 폴백."""
    oi_val, fr_val = await asyncio.gather(
        self.exchange.get_open_interest(),
        self.exchange.get_funding_rate(),
        return_exceptions=True,
    )
    oi_float = float(oi_val) if isinstance(oi_val, (int, float)) else 0.0
    fr_float = float(fr_val) if isinstance(fr_val, (int, float)) else 0.0

    # OI는 절대값이므로 이전 값 대비 변화율로 변환
    oi_change = self._calc_oi_change(oi_float)
    logger.debug(f"OI={oi_float:.0f}, OI변화율={oi_change:.6f}, 펀딩비={fr_float:.6f}")
    return oi_change, fr_float
```

`_calc_oi_change()` 메서드와 `_prev_oi` 상태를 추가한다:

`__init__()` 에 추가:
```python
self._prev_oi: float | None = None  # OI 변화율 계산용 이전 값
```

메서드 추가:
```python
def _calc_oi_change(self, current_oi: float) -> float:
    """이전 OI 대비 변화율을 계산한다. 첫 캔들은 0.0 반환."""
    if self._prev_oi is None or self._prev_oi == 0.0:
        self._prev_oi = current_oi
        return 0.0
    change = (current_oi - self._prev_oi) / self._prev_oi
    self._prev_oi = current_oi
    return change
```

### Step 4: 테스트 통과 확인

```bash
pytest tests/test_bot.py -v
```

Expected: 전체 PASS

### Step 5: 커밋

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: fetch realtime OI and funding rate on candle close for ML features"
```

---

## Task 4: 전체 테스트 통과 확인 및 README 업데이트

### Step 1: 전체 테스트 실행

```bash
bash scripts/run_tests.sh
```

Expected: 전체 PASS (새 테스트 포함)

### Step 2: README.md 업데이트

`README.md`의 "주요 기능" 섹션에서 ML 피처 설명을 수정한다.

기존:
```
- **23개 ML 피처**: XRP 기술 지표 13개 + BTC/ETH 수익률·상대강도 8개 + OI 변화율·펀딩비 2개 (실시간 미수집 항목은 0으로 채움)
```

변경:
```
- **23개 ML 피처**: XRP 기술 지표 13개 + BTC/ETH 수익률·상대강도 8개 + OI 변화율·펀딩비 2개 (캔들 마감 시 REST API로 실시간 수집)
```

### Step 3: 최종 커밋

```bash
git add README.md
git commit -m "docs: update README to reflect realtime OI/funding rate collection"
```

---

## 구현 후 검증 포인트

1. 봇 실행 로그에서 `OI=xxx, OI변화율=xxx, 펀딩비=xxx` 라인이 15분마다 출력되는지 확인
2. API 오류(네트워크 단절 등) 시 `WARNING: OI 조회 실패 (무시)` 로그 후 0.0으로 폴백해 봇이 정상 동작하는지 확인
3. `build_features()` 호출 시 `oi_change`, `funding_rate`가 실제 값으로 채워지는지 로그 확인

---

## 다음 단계: 접근법 B (OI/펀딩비 누적 저장)

A 완료 후 진행할 계획:
- `scripts/fetch_history.py` 실행 시 기존 parquet에 새 30일치를 **append(중복 제거)** 방식으로 저장
- 시간이 지날수록 OI/펀딩비 학습 데이터가 누적되어 모델 품질 향상
- 별도 플랜 문서로 작성 예정
