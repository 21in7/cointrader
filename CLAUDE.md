# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoinTrader is a Python asyncio-based automated cryptocurrency trading bot for Binance Futures. It supports multi-symbol simultaneous trading (XRP, TRX, DOGE etc.) on 15-minute candles, using BTC/ETH as correlation features. The system has 5 layers: Data (WebSocket streams) → Signal (technical indicators) → ML Filter (ONNX/LightGBM) → Execution & Risk → Event/Alert (Discord).

## Common Commands

```bash
# venv 
source .venv/bin/activate

# Run the bot
python main.py

# Run full test suite
bash scripts/run_tests.sh

# Run filtered tests
bash scripts/run_tests.sh -k "bot"

# Run pytest directly
pytest tests/ -v --tb=short

# ML training pipeline (all symbols)
bash scripts/train_and_deploy.sh

# Single symbol training
bash scripts/train_and_deploy.sh --symbol TRXUSDT

# MLX GPU training (macOS Apple Silicon)
bash scripts/train_and_deploy.sh mlx --symbol XRPUSDT

# Hyperparameter tuning (50 trials, 5-fold walk-forward)
python scripts/tune_hyperparams.py --symbol XRPUSDT

# Weekly strategy report (manual, skip data fetch)
python scripts/weekly_report.py --skip-fetch

# Weekly report with data refresh
python scripts/weekly_report.py

# Fetch historical data (single symbol with auto correlation)
python scripts/fetch_history.py --symbol TRXUSDT --interval 15m --days 365

# Fetch historical data (explicit symbols)
python scripts/fetch_history.py --symbols XRPUSDT BTCUSDT ETHUSDT --interval 15m --days 365

# Deploy models to production
bash scripts/deploy_model.sh --symbol XRPUSDT
```

## Architecture

**Entry point**: `main.py` → creates `Config` → shared `RiskManager` → per-symbol `TradingBot` instances → `asyncio.gather()`

**Multi-symbol architecture**: Each symbol gets its own `TradingBot` instance with independent `Exchange`, `MLFilter`, and `DataStream`. The `RiskManager` is shared as a singleton across all bots, enforcing global daily loss limits and same-direction position limits via `asyncio.Lock`.

**5-layer data flow on each 15m candle close:**
1. `src/data_stream.py` — Combined WebSocket for primary+correlation symbols, deque buffers (200 candles each)
2. `src/indicators.py` — RSI, MACD, BB, EMA, StochRSI, ATR; weighted signal aggregation → LONG/SHORT/HOLD
3. `src/ml_filter.py` + `src/ml_features.py` — 26-feature extraction (ADX + OI 파생 피처 포함), ONNX priority > LightGBM fallback, threshold ≥ 0.55
4. `src/exchange.py` + `src/risk_manager.py` — Dynamic margin, MARKET orders with SL/TP, daily loss limit (5%), same-direction limit
5. `src/user_data_stream.py` + `src/notifier.py` — Real-time TP/SL detection via WebSocket, Discord webhooks

**Dual-layer kill switch** (per-symbol, in `src/bot.py`): Fast Kill (8 consecutive net losses) + Slow Kill (last 15 trades PF < 0.75). Trade history persisted to `data/trade_history/{symbol}.jsonl`. Blocks new entries only; existing SL/TP exits work normally. Manual reset via `RESET_KILL_SWITCH_{SYMBOL}=True` env var + restart.

**Parallel execution**: Per-symbol bots run independently via `asyncio.gather()`. Each bot's `user_data_stream` also runs in parallel.

**Model/data directories**: `models/{symbol}/` and `data/{symbol}/` for per-symbol models. Falls back to `models/` root if symbol dir doesn't exist.

## Key Patterns

- **Async-first**: All I/O operations use `async/await`; parallel tasks via `asyncio.gather()`
- **Reverse signal re-entry**: While holding LONG, if SHORT signal appears → close position, cancel SL/TP, open SHORT. `_is_reentering` flag prevents race conditions with User Data Stream
- **ML hot reload**: `ml_filter.check_and_reload()` compares file mtime on every candle, reloads model without restart
- **Active Config pattern**: Best hyperparams stored in `models/active_lgbm_params.json`, must be manually approved before retraining
- **Graceful degradation**: Missing model → all signals pass; API failure → use fallback values (0.0 for OI/funding)
- **Walk-forward validation**: Time-series CV with undersampling (1:1 class balance, preserving time order)
- **Label generation**: Binary labels based on 24-candle (6h) lookahead — check SL hit first (conservative), then TP

## Testing

- All external APIs (Binance, Discord) are mocked with `unittest.mock.AsyncMock`
- Async tests use `@pytest.mark.asyncio`
- 14 test files, 80+ test cases covering all layers
- Testing is done in actual terminal, not IDE sandbox

## Configuration

Environment variables via `.env` file (see `.env.example`). Key vars: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `SYMBOLS` (comma-separated, e.g. `XRPUSDT,TRXUSDT`), `CORRELATION_SYMBOLS` (default `BTCUSDT,ETHUSDT`), `LEVERAGE`, `DISCORD_WEBHOOK_URL`, `MARGIN_MAX_RATIO`, `MARGIN_MIN_RATIO`, `MAX_SAME_DIRECTION` (default 2), `NO_ML_FILTER`.

`src/config.py` uses `@dataclass` with `__post_init__` to load and validate all env vars. Per-symbol strategy params supported via `SymbolStrategyParams` — override with `ATR_SL_MULT_{SYMBOL}`, `ATR_TP_MULT_{SYMBOL}`, `SIGNAL_THRESHOLD_{SYMBOL}`, `ADX_THRESHOLD_{SYMBOL}`, `VOL_MULTIPLIER_{SYMBOL}`. Access via `config.get_symbol_params(symbol)`.

## Deployment

- **Docker**: `Dockerfile` (Python 3.12-slim) + `docker-compose.yml`
- **CI/CD**: Jenkins pipeline (Gitea → Docker registry → LXC production server)
- Models stored in `models/{symbol}/`, data cache in `data/{symbol}/`, logs in `logs/`

## Design & Implementation Plans

All design documents and implementation plans are stored in `docs/plans/` with the naming convention `YYYY-MM-DD-feature-name.md`. Design docs (`-design.md`) describe architecture decisions; implementation plans (`-plan.md`) contain step-by-step tasks for Claude to execute.

**Chronological plan history:**

| Date | Plan | Status |
|------|------|--------|
| 2026-03-01 | `xrp-futures-autotrader` | Completed |
| 2026-03-01 | `discord-notifier-and-position-recovery` | Completed |
| 2026-03-01 | `upload-to-gitea` | Completed |
| 2026-03-01 | `dockerfile-and-docker-compose` | Completed |
| 2026-03-01 | `fix-pandas-ta-python312` | Completed |
| 2026-03-01 | `jenkins-gitea-registry-cicd` | Completed |
| 2026-03-01 | `ml-filter-design` / `ml-filter-implementation` | Completed |
| 2026-03-01 | `train-on-mac-deploy-to-lxc` | Completed |
| 2026-03-01 | `m4-accelerated-training` | Completed |
| 2026-03-01 | `vectorized-dataset-builder` | Completed |
| 2026-03-01 | `btc-eth-correlation-features` (design + plan) | Completed |
| 2026-03-01 | `dynamic-margin-ratio` (design + plan) | Completed |
| 2026-03-01 | `lgbm-improvement` | Completed |
| 2026-03-01 | `15m-timeframe-upgrade` | Completed |
| 2026-03-01 | `oi-nan-epsilon-precision-threshold` | Completed |
| 2026-03-02 | `rs-divide-mlx-nan-fix` | Completed |
| 2026-03-02 | `reverse-signal-reenter` (design + plan) | Completed |
| 2026-03-02 | `realtime-oi-funding-features` | Completed |
| 2026-03-02 | `oi-funding-accumulation` | Completed |
| 2026-03-02 | `optuna-hyperparam-tuning` (design + plan) | Completed |
| 2026-03-02 | `user-data-stream-tp-sl-detection` (design + plan) | Completed |
| 2026-03-02 | `adx-filter-design` | Completed |
| 2026-03-02 | `hold-negative-sampling` (design + plan) | Completed |
| 2026-03-03 | `position-monitor-logging` | Completed |
| 2026-03-03 | `adx-ml-feature-migration` (design + plan) | Completed |
| 2026-03-03 | `optuna-precision-objective-plan` | Completed |
| 2026-03-03 | `demo-1m-125x` (design + plan) | In Progress |
| 2026-03-04 | `oi-derived-features` (design + plan) | Completed |
| 2026-03-05 | `multi-symbol-trading` (design + plan) | Completed |
| 2026-03-06 | `multi-symbol-dashboard` (design + plan) | Completed |
| 2026-03-06 | `strategy-parameter-sweep` (plan) | Completed |
| 2026-03-07 | `weekly-report` (plan) | Completed |
| 2026-03-07 | `code-review-improvements` | Partial (#1,#2,#4,#5,#6,#8 완료) |
