# Code Review Fixes Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 9 issues from code review re-evaluation (2 Critical, 3 Important, 4 Minor)

**Architecture:** Targeted fixes across risk_manager, exchange, bot, config, ml_filter. No new files — all modifications to existing modules.

**Tech Stack:** Python 3.12, asyncio, python-binance, LightGBM, ONNX Runtime

---

### Task 1: #2 Critical — Balance reservation lock for concurrent entry

**Files:**
- Modify: `src/risk_manager.py` — add `_entry_lock` to serialize entry flow
- Modify: `src/bot.py:405-413` — acquire entry lock around balance read → order
- Test: `tests/test_risk_manager.py`

The simplest fix: add an asyncio.Lock in RiskManager that serializes the entire _open_position flow across all bots. This prevents two bots from reading the same balance simultaneously.

- [ ] Add `_entry_lock = asyncio.Lock()` to RiskManager
- [ ] Add `async def entry_lock(self)` context manager
- [ ] In bot.py `_open_position`, wrap balance read + order under `async with self.risk.entry_lock()`
- [ ] Add test for concurrent entry serialization
- [ ] Run tests

### Task 2: #3 Critical — SYNC PnL startTime + single query

**Files:**
- Modify: `src/exchange.py:166-185` — add `start_time` param to `get_recent_income`
- Modify: `src/bot.py:75-82` — record `_entry_time` on position open
- Modify: `src/bot.py:620-629` — pass `start_time` to income query
- Test: `tests/test_exchange.py`

- [ ] Add `_entry_time: int | None = None` to TradingBot
- [ ] Set `_entry_time = int(time.time() * 1000)` on entry and recovery
- [ ] Add `start_time` parameter to `get_recent_income()`
- [ ] Use start_time in SYNC fallback
- [ ] Add test
- [ ] Run tests

### Task 3: #1 Important — Thread-safe Client access

**Files:**
- Modify: `src/exchange.py` — add `threading.Lock` per instance

- [ ] Add `self._api_lock = threading.Lock()` in `__init__`
- [ ] Wrap all `run_in_executor` lambdas with lock acquisition
- [ ] Add test
- [ ] Run tests

### Task 4: #4 Important — reset_daily async with lock

**Files:**
- Modify: `src/risk_manager.py:61-64` — make async + lock
- Modify: `main.py:22` — await reset_daily
- Test: `tests/test_risk_manager.py`

- [ ] Convert `reset_daily` to async, add lock
- [ ] Update `_daily_reset_loop` call
- [ ] Add test
- [ ] Run tests

### Task 5: #8 Important — exchange_info cache TTL

**Files:**
- Modify: `src/exchange.py:25-34` — add TTL (24h)

- [ ] Add `_exchange_info_time: float = 0.0`
- [ ] Check TTL in `_get_exchange_info`
- [ ] Add test
- [ ] Run tests

### Task 6: #7 Minor — Pass pre-computed indicators to _open_position

**Files:**
- Modify: `src/bot.py:392,415,736` — pass df_with_indicators

- [ ] Add `df_with_indicators` parameter to `_open_position`
- [ ] Use passed df instead of re-creating Indicators
- [ ] Run tests

### Task 7: #11 Minor — Config input validation

**Files:**
- Modify: `src/config.py:39` — add range checks
- Test: `tests/test_config.py`

- [ ] Add validation for LEVERAGE, MARGIN ratios, ML_THRESHOLD
- [ ] Add test for invalid values
- [ ] Run tests

### Task 8: #12 Minor — Dynamic correlation symbol access

**Files:**
- Modify: `src/bot.py:196-198` — iterate dynamically

- [ ] Replace hardcoded [0]/[1] with dict-based access
- [ ] Run tests

### Task 9: #14 Minor — Normalize NaN handling for LightGBM

**Files:**
- Modify: `src/ml_filter.py:144-147` — apply nan_to_num for LightGBM too

- [ ] Add `np.nan_to_num` to LightGBM path
- [ ] Run tests
