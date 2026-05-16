# CoinTrader — Falsification-First Quant Trading Research

> **📄 Start here: [Case Study — Falsification-First Quantitative Trading Research](docs/CASE-STUDY-quant-research.en.md)** ([한국어](docs/CASE-STUDY-quant-research.md))

A Python asyncio research codebase for Binance Futures, built around a
disciplined research process rather than a single strategy. Over the
project's life, **10 trading hypotheses were systematically falsified
using pre-committed kill criteria and data-first validation — at $0
capital lost.**

The honest finding: **there is no retail alpha in public-data crypto
across the paradigms tested** (directional prediction, ML filtering,
LLM/sentiment, market-neutral carry, event microstructure). That is the
*correct* result for an efficient market — and the deliverable here is
not alpha but a **reusable falsification discipline plus production-grade
execution infrastructure**.

If you only read one thing, read the
[case study](docs/CASE-STUDY-quant-research.en.md): it explains the
methodology and walks through all 10 null results and the engineering
judgment behind them.

---

## What this demonstrates

- **Falsification-first research loop** — hypothesis → kill criteria
  fixed *before* running → cheapest-possible (data-only) validation →
  honest verdict → negative results documented as first-class artifacts
  (`docs/plans/*-result.md`) → paradigm-level escalation, not knob-tweaking.
- **Statistical rigor under pressure** — caught an overfit (+275 bps on a
  partial sample that collapsed to −82 bps on the full N=323), reasoned
  about survivorship bias directionally, decomposed gross-vs-net to kill
  a "structural" edge that was a hidden directional bet.
- **Reproducibility engineering** — made a non-deterministic LLM
  bit-reproducible via `temperature=0` + response-hash caching so a
  backtest could be verified (zero extra dependencies).

## Engineering (production-grade infrastructure)

| Area | Highlights |
|------|-----------|
| Execution | Async multi-symbol bot, `asyncio.gather` per symbol, shared singleton `RiskManager` (`src/bot.py`) |
| Risk | Dual-layer kill switch (Fast: 8 consecutive losses / Slow: 15-trade PF < 0.75), JSONL-persisted, boot-time retroactive check |
| Backtest | Vectorized walk-forward engine, slippage/fee model, deterministic, extensible event studies (`src/backtester.py`, `scripts/*_backtest.py`) |
| Data | Idempotent/resumable Binance fapi pipelines (`scripts/collect_listings.py`, `scripts/build_sentiment_dataset.py`) |
| Tests | Regression suite incl. look-ahead guards & baseline no-op guarantees (`tests/`) |
| Ops | Docker + docker-compose, Discord alerts, monitoring dashboard, weekly strategy report |

## The 10 falsified hypotheses

Directional signals (LS-ratio, Funding+OI, public-API sweep, MTF pullback
×3), ML filtering, LLM/sentiment fusion, delta-neutral funding carry, and
new-listing microstructure — each with a `docs/plans/*-design.md` +
`*-result.md` pair. Consolidated analysis in
[`docs/plans/2026-05-04-strategy-post-mortem.md`](docs/plans/2026-05-04-strategy-post-mortem.md)
and the [case study](docs/CASE-STUDY-quant-research.en.md).

## Status

Alpha research is **concluded** (no edge found; the correct, evidence-based
finding). The repository's value is the documented methodology and the
reusable infrastructure. The bot defaults to safe/disabled states
(`NO_ML_FILTER=true`, sentiment gate `off`); it is not a profitable system
and is not represented as one.

## Repository map

- [`docs/CASE-STUDY-quant-research.en.md`](docs/CASE-STUDY-quant-research.en.md) — **the main artifact** (EN; [KO](docs/CASE-STUDY-quant-research.md))
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 5-layer system design
- [`docs/plans/`](docs/plans/) — every hypothesis: design + result docs
- [`docs/README.ko.md`](docs/README.ko.md) — full bot operation guide (Korean)
- `src/` — bot, backtester, risk, execution, deterministic LLM provider
- `scripts/` — reproducible research/backtest scripts

## Quick start (research / reproduction)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_tests.sh                          # test suite
python scripts/run_backtest.py --symbol XRPUSDT --no-ml   # baseline backtest
python scripts/funding_carry_backtest.py           # a falsified hypothesis, reproducible
```

Operating the bot itself (API keys, Docker, deployment) is documented in
[`docs/README.ko.md`](docs/README.ko.md). It trades real funds when
enabled — validate on Binance Testnet first; past results do not
guarantee future results; you bear all risk.

---

*This project's most honest output is a fast, cheap "no." It proved
"there is no edge here" ten times without losing capital, and built the
discipline and tooling to do so. See the
[case study](docs/CASE-STUDY-quant-research.en.md).*
