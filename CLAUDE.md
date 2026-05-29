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

**Dual-layer kill switch** (per-symbol, in `src/bot.py` and `src/mtf_bot.py`): Fast Kill (8 consecutive net losses) + Slow Kill (last 15 trades PF < 0.75). Trade history persisted to `data/trade_history/{symbol}.jsonl`. Blocks new entries only; existing SL/TP exits work normally. Manual reset via `RESET_KILL_SWITCH_{SYMBOL}=True` (main bot) or `RESET_KILL_SWITCH_MTF_{SYMBOL}=True` (MTF bot) env var + restart. MTF bot uses bps-based PnL for kill switch decisions.

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

Environment variables via `.env` file (see `.env.example`). Key vars: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `SYMBOLS` (comma-separated, currently `XRPUSDT` only — SOL/DOGE/TRX removed due to PF < 1.0), `CORRELATION_SYMBOLS` (default `BTCUSDT,ETHUSDT`), `LEVERAGE`, `DISCORD_WEBHOOK_URL`, `MARGIN_MAX_RATIO`, `MARGIN_MIN_RATIO`, `MAX_SAME_DIRECTION` (default 2), `NO_ML_FILTER` (default `true` — ML disabled due to insufficient feature alpha).

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
| 2026-03-19 | `critical-bugfixes` (C5,C1,C3,C8) | Completed |
| 2026-03-21 | `dashboard-code-review-r2` (#14,#19) | Completed |
| 2026-03-21 | `code-review-fixes-r2` (9 issues) | Completed |
| 2026-03-21 | `ml-pipeline-fixes` (C1,C3,I1,I3,I4,I5) | Completed |
| 2026-03-21 | `training-threshold-relaxation` (plan) | Completed |
| 2026-03-21 | `purged-gap-and-ablation` (plan) | Completed |
| 2026-03-21 | `ml-validation-result` | ML OFF > ML ON 확정, SOL/DOGE/TRX 제외, XRP 단독 운영 |
| 2026-03-21 | `ml-validation-pipeline` (plan) | Completed |
| 2026-03-22 | `backtest-market-context` (design) | 설계 완료, 구현 대기 |
| 2026-03-22 | `testnet-uds-verification` (design) | 설계 완료, 구현 대기 |
| 2026-03-30 | `ls-ratio-backtest` (design + result) | Edge 없음 확정, 폐기 |
| 2026-03-30 | `fr-oi-backtest` (result) | SHORT PF=1.88이나 대칭성 실패(Case2), 폐기 |
| 2026-03-30 | `public-api-research-closed` | Binance 공개 API 전수 테스트 완료, 단독 edge 없음 |
| 2026-03-30 | `mtf-pullback-bot` | MTF Pullback Bot — **최종 폐기** (OOS+BTC필터 모두 실패) |
| 2026-04-21 | `mtf-oos-dryrun-result` | 중간 보고 — 24건 Raw PF 0.98 |
| 2026-05-04 | `mtf-oos-final-result` | **FAIL, 폐기** — 30건 fees_only PF 0.84, SHORT 대칭성 실패 |
| 2026-05-04 | `mtf-btc-filter` (design + result) | **FAIL, 최종 폐기** — BTC 필터 추가해도 OOS PF 0.90, 베이스라인보다 악화 |
| 2026-05-04 | `strategy-post-mortem` | 7전 7패 분석 — 공개 시그널 방향 예측 패러다임 한계, 다음 방향 제안 |
| 2026-05-16 | `tradingagents-sentiment-fusion` (design+result) | **FAIL, 폐기** — 게이트 B 백테스트: veto/contrarian 무효과(N=19, 0건 필터), confirm 악화(PF 0.77→0.39). 단일심볼 표본 부족 + 뉴스-only modality gap. 코드는 유지(off 기본), 봇 미연결 |
| 2026-05-17 | `funding-carry` (design+result) | **FAIL, 폐기** — 델타중립 펀딩 캐리: 이상화 gross +4.6%/yr이나 TAKER+5%borrow net -0.37%, 3/6심볼만 양, 레짐의존 강세프리미엄(시장중립 아님), 변형B 부호churn 전멸. 사전기준 4개 전부 위반 → **9전9패, from-scratch 재설계 에스컬레이션** |
| 2026-05-17 | `new-listing-microstructure` (design+result) | **FAIL, 폐기 (10전10패)** — N=323. H1 best slip0.6% -2.6bps/slip1.0% -82.6bps, H2 중앙값 -98bps(아웃라이어 의존). 부분데이터 +275bps 힌트는 저N 아티팩트(전체N서 소멸). 사전기준 5개 위반. 알파 추구 라인 종결 → 인프라/학습 자산 전환 재확인 |
| 2026-05-17 | `quant-research-case-study` | `docs/CASE-STUDY-quant-research.md` — 10전10패를 반증-우선 리서치 방법론 증거물(스킬 자산)로 종합. 알파 추구 종결의 capstone 산출물 |
| 2026-05-28 | `maker-fill-killscreen` (result) | **FAIL, maker caveat 종결** — fees_only kill-screen: 전부-maker 상한(왕복 4bps)조차 TOTAL PF 0.95<1.0, SHORT 대칭성 fee와 무관하게 지속(0.56→0.64). edge 부재가 병, 수수료는 증상. Q2(체결 모델링) 불필요 |
| 2026-05-29 | `statarb-cointegration-precheck` (result) | **FAIL, stat-arb 라인 종료** — 방향중립 패러다임(post-mortem §5 1순위) 검증. 반증-우선 precheck: XRP/BTC 6/7게이트 탈락(EG p=0.39, Johansen 미기각, half-life 47일, OOS p=0.41, rolling 9.9%), 크로스-알트 15페어 0/15 PASS(Bonferroni/BH 보정 후 0개, half-life 전부 ≥7.9일=랜덤워크). 크립토 알트는 co-trending이나 공적분 없음 → stat-arb 전제 불성립. precheck/scan 모듈은 임의 유니버스 재사용 자산. `src/statarb/` |
| 2026-05-29 | `carry-precheck-btceth` (result) | **FAIL, 캐리=이자율 재포장 확인** — spot-perp 펀딩 캐리 BTC/ETH 특성화(이전 6알트 연구 미포함분). 게이트 1만 탈락: net 캐리 BTC +4.32%/ETH +4.65%/yr < 10% 바. basis 공적분(β≈1, HL~4h)·리스크(1x 안전레버 ~100x)·레짐(rolling + 92~94%) 전부 PASS. funding(+)은 주로 이자율 베이스라인(프리미엄은 음). borrow 5% 시 net −0.4~−0.7% = 이전 연구 −0.37%와 사실상 동일 → 펀딩수입≈자금조달비, 메이저서도 재현. `src/carry/` |
| 2026-05-29 | `leadlag-directional-precheck` (result) | **FAIL, "진짜지만 거래 불가"** — BTC/ETH→6알트 15m 리드-래그 12쌍. economics-first 게이트가 헤드라인서 0/12(예측 edge 0.2~2.1bps ≪ 15bps 비용임계). BTC/ETH→TRX는 통계적으로 진짜 lead(bootP<0.01, BH생존, 비대칭 11~22, OOS부호일치)지만 edge 1~2bps + OOS 비유의 → 시장이 거래가능분 차익소거, 잔차만 잔존. 15m 유동성 메이저 리드-래그는 비용스케일서 비-엣지. `src/leadlag/` |
