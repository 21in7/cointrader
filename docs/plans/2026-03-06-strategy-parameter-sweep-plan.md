# Strategy Parameter Sweep Plan

**Date**: 2026-03-06
**Status**: Completed

## Goal

Find profitable parameter combinations for the base technical indicator strategy (ML OFF) using walk-forward backtesting, targeting PF >= 1.0 as foundation for ML redesign.

## Background

Walk-forward backtest revealed the current XRP strategy is unprofitable (PF 0.71, -641 PnL). The strategy parameter sweep systematically tests 324 combinations of 5 parameters to find profitable regimes.

## Parameters Swept


| Parameter           | Values        | Description                               |
| ------------------- | ------------- | ----------------------------------------- |
| `atr_sl_mult`       | 1.0, 1.5, 2.0 | Stop-loss ATR multiplier                  |
| `atr_tp_mult`       | 2.0, 3.0, 4.0 | Take-profit ATR multiplier                |
| `signal_threshold`  | 3, 4, 5       | Min weighted indicator score for entry    |
| `adx_threshold`     | 0, 20, 25, 30 | ADX filter (0=disabled, N=require ADX>=N) |
| `volume_multiplier` | 1.5, 2.0, 2.5 | Volume surge detection multiplier         |


Total combinations: 3 x 3 x 3 x 4 x 3 = **324**

## Implementation

### Files Modified

- `src/indicators.py` — `get_signal()` accepts `signal_threshold`, `adx_threshold`, `volume_multiplier` params
- `src/dataset_builder.py` — `_calc_signals()` accepts same params for vectorized computation
- `src/backtester.py` — `BacktestConfig` includes strategy params; `WalkForwardBacktester` propagates them to test folds

### Files Created

- `scripts/strategy_sweep.py` — CLI tool for parameter grid sweep

### Bug Fix

- `WalkForwardBacktester` was not passing `signal_threshold`, `adx_threshold`, `volume_multiplier`, or `use_ml` to fold `BacktestConfig`. All signal params were silently using defaults, making ADX/volume/threshold sweeps have zero effect.

## Results (XRPUSDT, Walk-Forward 3/1)

### Top 10 Combinations


| Rank | SL×ATR | TP×ATR | Signal | ADX | Vol | Trades | WinRate | PF   | MDD   | PnL  | Sharpe |
| ---- | ------ | ------ | ------ | --- | --- | ------ | ------- | ---- | ----- | ---- | ------ |
| 1    | 1.5    | 4.0    | 3      | 30  | 2.5 | 19     | 52.6%   | 2.39 | 7.0%  | +469 | 61.0   |
| 2    | 1.5    | 2.0    | 3      | 30  | 2.5 | 19     | 68.4%   | 2.23 | 6.5%  | +282 | 61.2   |
| 3    | 1.0    | 2.0    | 3      | 30  | 2.5 | 19     | 57.9%   | 1.98 | 5.0%  | +213 | 50.8   |
| 4    | 1.0    | 4.0    | 3      | 30  | 2.5 | 19     | 36.8%   | 1.80 | 7.7%  | +248 | 37.1   |
| 5    | 1.5    | 3.0    | 3      | 30  | 2.5 | 19     | 52.6%   | 1.76 | 10.1% | +258 | 40.9   |
| 6    | 1.5    | 4.0    | 3      | 25  | 2.5 | 28     | 42.9%   | 1.75 | 13.1% | +381 | 36.8   |
| 7    | 2.0    | 4.0    | 3      | 30  | 1.5 | 39     | 48.7%   | 1.67 | 16.9% | +572 | 35.3   |
| 8    | 1.0    | 2.0    | 3      | 25  | 2.5 | 28     | 50.0%   | 1.64 | 5.8%  | +205 | 35.7   |
| 9    | 1.5    | 2.0    | 3      | 25  | 2.5 | 28     | 57.1%   | 1.62 | 10.3% | +229 | 35.7   |
| 10   | 2.0    | 2.0    | 3      | 25  | 2.5 | 27     | 66.7%   | 1.57 | 12.0% | +217 | 33.3   |


### Current Production (Rank 93/324)


| SL×ATR | TP×ATR | Signal | ADX | Vol | Trades | WinRate | PF   | MDD   | PnL  |
| ------ | ------ | ------ | --- | --- | ------ | ------- | ---- | ----- | ---- |
| 1.5    | 3.0    | 3      | 0   | 1.5 | 118    | 30.5%   | 0.71 | 65.9% | -641 |


### Key Findings

1. **ADX filter is the single most impactful parameter.** All top 10 results use ADX >= 25, with ADX=30 dominating the top 5. This filters out sideways/ranging markets where signals are noise.
2. **Volume multiplier 2.5 dominates.** Higher volume thresholds ensure entries only on strong conviction (genuine breakouts vs. noise).
3. **Signal threshold 3 is optimal.** Higher thresholds (4, 5) produced too few trades or zero trades in most ADX-filtered regimes.
4. **SL/TP ratios matter less than entry filters.** The top results span all SL/TP combos, but all share ADX=25-30 + Vol=2.5.
5. **Trade count drops significantly with filters.** Top combos have 19-39 trades vs. 118 for current. Fewer but higher quality entries.
6. **41 combinations achieved PF >= 1.0** out of 324 total (12.7%).

## Recommended Next Steps

1. **Update production defaults**: ADX=25, volume_multiplier=2.0 as a conservative choice (more trades than ADX=30)
2. **Validate on TRXUSDT and DOGEUSDT** to confirm ADX filter is not XRP-specific
3. **Retrain ML models** with updated strategy params — the ML filter should now have a profitable base to improve upon
4. **Fine-tune sweep** around the profitable zone: ADX [25-35], Vol [2.0-3.0]

