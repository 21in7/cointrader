# Critical Bugfixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 critical bugs identified in code review (C5, C1, C3, C8)

**Architecture:** Direct fixes to backtester.py, bot.py, main.py — no new files needed

**Tech Stack:** Python asyncio, signal handling

---

## Task 1: C5 — Backtester double fee deduction + atr≤0 fee leak

**Files:**
- Modify: `src/backtester.py:494-501`

- [x] Remove `self.balance -= entry_fee` at L496. The fee is already deducted in `_close_position` via `net_pnl = gross_pnl - entry_fee - exit_fee`.
- [x] This also fixes the atr≤0 early return bug — since balance is no longer modified before ATR check, early return doesn't leak fees.

## Task 2: C1 — SL/TP atomicity with retry and emergency close

**Files:**
- Modify: `src/bot.py:461-475`

- [x] Wrap SL/TP placement in `_place_sl_tp_with_retry()` with 3 retries and 1s backoff
- [x] Track `sl_placed` and `tp_placed` independently to avoid re-placing successful orders
- [x] On final failure, call `_emergency_close()` which market-closes the position and notifies via Discord
- [x] `_emergency_close` also handles its own failure with critical log + Discord alert

## Task 3: C3 — PnL double recording race condition

**Files:**
- Modify: `src/bot.py` (init, _on_position_closed, _position_monitor)

- [x] Add `self._close_lock = asyncio.Lock()` to `__init__`
- [x] Wrap `_on_position_closed` body with `async with self._close_lock`
- [x] Wrap SYNC path in `_position_monitor` with `async with self._close_lock`
- [x] Add double-check after lock acquisition in monitor (callback may have already processed)

## Task 4: C8 — Graceful shutdown with signal handler

**Files:**
- Modify: `main.py`

- [x] Add `signal.SIGTERM` and `signal.SIGINT` handlers via `loop.add_signal_handler()`
- [x] Use `asyncio.Event` + `asyncio.wait(FIRST_COMPLETED)` pattern
- [x] `_graceful_shutdown()`: cancel all open orders per bot (with 5s timeout), then cancel tasks
- [x] Log shutdown progress for each symbol

## Verification

- [x] All 138 existing tests pass (0 failures)
