# 반대 시그널 시 청산 후 즉시 재진입 구현 플랜

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 반대 방향 시그널이 오면 기존 포지션을 청산하고 ML 필터 통과 시 즉시 반대 방향으로 재진입한다.

**Architecture:** `src/bot.py`에 `_close_and_reenter` 메서드를 추가하고, `process_candle`의 반대 시그널 분기에서 이를 호출한다. 기존 `_close_position`과 `_open_position`을 그대로 재사용하므로 중복 없음.

**Tech Stack:** Python 3.12, pytest, unittest.mock

---

## 테스트 스크립트

각 태스크 단계마다 아래 스크립트로 테스트를 실행한다.

```bash
# Task 1 — 신규 테스트 실행 (구현 전, FAIL 확인용)
bash scripts/test_reverse_reenter.sh 1

# Task 2 — _close_and_reenter 메서드 테스트 (구현 후, PASS 확인)
bash scripts/test_reverse_reenter.sh 2

# Task 3 — process_candle 분기 테스트 (수정 후, PASS 확인)
bash scripts/test_reverse_reenter.sh 3

# test_bot.py 전체
bash scripts/test_reverse_reenter.sh bot

# 전체 테스트 스위트
bash scripts/test_reverse_reenter.sh all
```

---

## 참고 파일

- 설계 문서: `docs/plans/2026-03-02-reverse-signal-reenter-design.md`
- 구현 대상: `src/bot.py`
- 기존 테스트: `tests/test_bot.py`

---

## Task 1: `_close_and_reenter` 테스트 작성

**Files:**
- Modify: `tests/test_bot.py`

### Step 1: 테스트 3개 추가

`tests/test_bot.py` 맨 아래에 다음 테스트를 추가한다.

```python
@pytest.mark.asyncio
async def test_close_and_reenter_calls_open_when_ml_passes(config, sample_df):
    """반대 시그널 + ML 필터 통과 시 청산 후 재진입해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = True

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_awaited_once_with("SHORT", sample_df)


@pytest.mark.asyncio
async def test_close_and_reenter_skips_open_when_ml_blocks(config, sample_df):
    """ML 필터 차단 시 청산만 하고 재진입하지 않아야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = True
    bot.ml_filter.should_enter.return_value = False

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_not_called()


@pytest.mark.asyncio
async def test_close_and_reenter_skips_open_when_max_positions_reached(config, sample_df):
    """최대 포지션 수 도달 시 청산만 하고 재진입하지 않아야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot._close_position = AsyncMock()
    bot._open_position = AsyncMock()
    bot.risk = MagicMock()
    bot.risk.can_open_new_position.return_value = False

    position = {"positionAmt": "100", "entryPrice": "0.5", "markPrice": "0.52"}
    await bot._close_and_reenter(position, "SHORT", sample_df)

    bot._close_position.assert_awaited_once_with(position)
    bot._open_position.assert_not_called()
```

### Step 2: 테스트 실행 — 실패 확인

```bash
bash scripts/test_reverse_reenter.sh 1
```

예상 결과: `AttributeError: 'TradingBot' object has no attribute '_close_and_reenter'` 로 3개 FAIL

---

## Task 2: `_close_and_reenter` 메서드 구현

**Files:**
- Modify: `src/bot.py:148` (`_close_position` 메서드 바로 아래에 추가)

### Step 1: `_close_position` 다음에 메서드 추가

`src/bot.py`에서 `_close_position` 메서드(148~167번째 줄) 바로 뒤에 다음을 추가한다.

```python
    async def _close_and_reenter(
        self,
        position: dict,
        signal: str,
        df,
        btc_df=None,
        eth_df=None,
    ) -> None:
        """기존 포지션을 청산하고, ML 필터 통과 시 반대 방향으로 즉시 재진입한다."""
        await self._close_position(position)

        if not self.risk.can_open_new_position():
            logger.info("최대 포지션 수 도달 — 재진입 건너뜀")
            return

        if self.ml_filter.is_model_loaded():
            features = build_features(df, signal, btc_df=btc_df, eth_df=eth_df)
            if not self.ml_filter.should_enter(features):
                logger.info(f"ML 필터 차단: {signal} 재진입 무시")
                return

        await self._open_position(signal, df)
```

### Step 2: 테스트 실행 — 통과 확인

```bash
bash scripts/test_reverse_reenter.sh 2
```

예상 결과: 3개 PASS

### Step 3: 커밋

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: add _close_and_reenter method for reverse signal handling"
```

---

## Task 3: `process_candle` 분기 수정

**Files:**
- Modify: `src/bot.py:83-85`

### Step 1: 기존 분기 테스트 추가

`tests/test_bot.py`에 다음 테스트를 추가한다.

```python
@pytest.mark.asyncio
async def test_process_candle_calls_close_and_reenter_on_reverse_signal(config, sample_df):
    """반대 시그널 시 process_candle이 _close_and_reenter를 호출해야 한다."""
    with patch("src.bot.BinanceFuturesClient"):
        bot = TradingBot(config)

    bot.exchange = AsyncMock()
    bot.exchange.get_position = AsyncMock(return_value={
        "positionAmt": "100",
        "entryPrice": "0.5",
        "markPrice": "0.52",
    })
    bot._close_and_reenter = AsyncMock()
    bot.ml_filter = MagicMock()
    bot.ml_filter.is_model_loaded.return_value = False
    bot.ml_filter.should_enter.return_value = True

    with patch("src.bot.Indicators") as MockInd:
        mock_ind = MagicMock()
        mock_ind.calculate_all.return_value = sample_df
        mock_ind.get_signal.return_value = "SHORT"  # 현재 LONG 포지션에 반대 시그널
        MockInd.return_value = mock_ind
        await bot.process_candle(sample_df)

    bot._close_and_reenter.assert_awaited_once()
    call_args = bot._close_and_reenter.call_args
    assert call_args.args[1] == "SHORT"
```

### Step 2: 테스트 실행 — 실패 확인

```bash
bash scripts/test_reverse_reenter.sh 3
```

예상 결과: FAIL (`_close_and_reenter`가 아직 호출되지 않음)

### Step 3: `process_candle` 수정

`src/bot.py`에서 아래 부분을 찾아 수정한다.

```python
# 변경 전 (81~85번째 줄 근처)
        elif position is not None:
            pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
            if (pos_side == "LONG" and signal == "SHORT") or \
               (pos_side == "SHORT" and signal == "LONG"):
                await self._close_position(position)

# 변경 후
        elif position is not None:
            pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
            if (pos_side == "LONG" and signal == "SHORT") or \
               (pos_side == "SHORT" and signal == "LONG"):
                await self._close_and_reenter(
                    position, signal, df_with_indicators, btc_df=btc_df, eth_df=eth_df
                )
```

### Step 4: 전체 테스트 실행 — 통과 확인

```bash
bash scripts/test_reverse_reenter.sh bot
```

예상 결과: 전체 PASS (기존 테스트 포함)

### Step 5: 커밋

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: call _close_and_reenter on reverse signal in process_candle"
```

---

## Task 4: 전체 테스트 스위트 확인

### Step 1: 전체 테스트 실행

```bash
bash scripts/test_reverse_reenter.sh all
```

예상 결과: 모든 테스트 PASS

### Step 2: 실패 테스트 있으면 수정 후 재실행

실패가 있으면 원인을 파악하고 수정한다. 기존 테스트를 깨뜨리지 않도록 주의.
