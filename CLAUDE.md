# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoinTrader is a Python asyncio-based automated cryptocurrency trading bot for Binance Futures. It trades XRPUSDT on 15-minute candles, using BTC/ETH as correlation features. The system has 5 layers: Data (WebSocket streams) → Signal (technical indicators) → ML Filter (ONNX/LightGBM) → Execution & Risk → Event/Alert (Discord).

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

# ML training pipeline (LightGBM default)
bash scripts/train_and_deploy.sh

# MLX GPU training (macOS Apple Silicon)
bash scripts/train_and_deploy.sh mlx

# Hyperparameter tuning (50 trials, 5-fold walk-forward)
python scripts/tune_hyperparams.py

# Fetch historical data
python scripts/fetch_history.py --symbols XRPUSDT BTCUSDT ETHUSDT --interval 15m --days 365

# Deploy models to production
bash scripts/deploy_model.sh
```

## Architecture

**Entry point**: `main.py` → creates `Config` (dataclass from env vars) → runs `TradingBot`

**5-layer data flow on each 15m candle close:**
1. `src/data_stream.py` — Combined WebSocket for XRP/BTC/ETH, deque buffers (200 candles each)
2. `src/indicators.py` — RSI, MACD, BB, EMA, StochRSI, ATR; weighted signal aggregation → LONG/SHORT/HOLD
3. `src/ml_filter.py` + `src/ml_features.py` — 23-feature extraction, ONNX priority > LightGBM fallback, threshold ≥ 0.60
4. `src/exchange.py` + `src/risk_manager.py` — Dynamic margin, MARKET orders with SL/TP, daily loss limit (5%)
5. `src/user_data_stream.py` + `src/notifier.py` — Real-time TP/SL detection via WebSocket, Discord webhooks

**Parallel execution**: `user_data_stream` runs independently via `asyncio.gather()` alongside candle processing.

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

Environment variables via `.env` file (see `.env.example`). Key vars: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `SYMBOL` (default XRPUSDT), `LEVERAGE`, `DISCORD_WEBHOOK_URL`, `MARGIN_MAX_RATIO`, `MARGIN_MIN_RATIO`, `NO_ML_FILTER`.

`src/config.py` uses `@dataclass` with `__post_init__` to load and validate all env vars.

## Deployment

- **Docker**: `Dockerfile` (Python 3.12-slim) + `docker-compose.yml`
- **CI/CD**: Jenkins pipeline (Gitea → Docker registry → LXC production server)
- Models stored in `models/`, data cache in `data/`, logs in `logs/`
