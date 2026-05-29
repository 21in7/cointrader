# CoinTrader — Falsification-First Quant Trading Research

> **📄 Start here: [Case Study — Falsification-First Quantitative Trading Research](docs/CASE-STUDY-quant-research.en.md)** ([한국어](docs/CASE-STUDY-quant-research.md))

A Python asyncio research codebase for Binance Futures, built around a
disciplined research process rather than a single strategy. Over the
project's life, **13 trading hypotheses were systematically falsified or
characterized as non-deployable using pre-committed kill criteria and
data-first validation — at $0 capital lost.**

The honest finding: **there is no deployable retail alpha in public-data
crypto across the paradigms tested** (directional prediction, ML
filtering, LLM/sentiment, market-neutral carry, statistical arbitrage /
cointegration, cross-asset lead-lag, event microstructure, and
low-frequency momentum / crisis-alpha). That is the *correct* result for
an efficient market — and the deliverable here is not alpha but a
**reusable falsification discipline plus production-grade execution
infrastructure**.

The latest research wave added an even cheaper kill — **data-only
"precheck" gates** that decide PASS/FAIL on structural properties
(cointegration, funding economics, predictable-edge-vs-cost, walk-forward
survival) *before a single line of backtest is written* — and an
**adversarial-verification pass that caught a bug in our own analysis**
before it reached a commit (see the case study).

If you only read one thing, read the
[case study](docs/CASE-STUDY-quant-research.en.md): it explains the
methodology and walks through the null results and the engineering
judgment behind them.

---

## Why this might be useful to you

If you are researching crypto trading strategies, this repo can save you
time and money by showing — with reproducible code — what *does not* work
and exactly why:

- **A documented falsification loop you can copy** — hypothesis → kill
  criteria fixed *before* running → cheapest-possible (data-only)
  validation → honest verdict → negative results kept as first-class
  artifacts (`docs/plans/*-result.md`) → paradigm-level escalation, not
  knob-tweaking.
- **Worked examples of statistical traps** — an overfit caught (+275 bps
  on a partial sample that collapsed to −82 bps on the full N=323),
  survivorship bias reasoned about directionally, a "market-neutral"
  carry shown to be a hidden directional bet via regime decomposition, a
  single-asset momentum "pass" exposed as non-portable by multi-asset
  walk-forward, and a bug in our *own* portfolio aggregation caught by an
  adversarial-verification pass before it reached a commit.
- **Data-only precheck gates** — structural PASS/FAIL tests
  (cointegration via Engle-Granger/Johansen with BH correction, OU
  half-life, funding economics, predictable-edge-vs-cost, walk-forward
  survival) that kill a hypothesis *before* any backtest is written
  (`src/{statarb,carry,leadlag,momentum}/`).
- **A reproducibility pattern worth reusing** — a non-deterministic LLM
  made bit-reproducible via `temperature=0` + response-hash caching, with
  zero extra dependencies (`src/sentiment_provider.py`).

## Engineering (production-grade infrastructure)

| Area | Highlights |
|------|-----------|
| Execution | Async multi-symbol bot, `asyncio.gather` per symbol, shared singleton `RiskManager` (`src/bot.py`) |
| Risk | Dual-layer kill switch (Fast: 8 consecutive losses / Slow: 15-trade PF < 0.75), JSONL-persisted, boot-time retroactive check |
| Backtest | Vectorized walk-forward engine, slippage/fee model, deterministic, extensible event studies (`src/backtester.py`, `scripts/*_backtest.py`) |
| Prechecks | Data-only falsification gates reusing the cost model: cointegration (EG/Johansen, BH-corrected), funding-carry economics, lead-lag edge-vs-cost, walk-forward survival, crisis-alpha regime decomposition, block-bootstrap significance (`src/{statarb,carry,leadlag,momentum}/`) |
| Data | Idempotent/resumable Binance fapi pipelines (`scripts/collect_listings.py`, `scripts/build_sentiment_dataset.py`) |
| Tests | Regression suite incl. look-ahead guards & baseline no-op guarantees (`tests/`) |
| Ops | Docker + docker-compose, Discord alerts, monitoring dashboard, weekly strategy report |

## The 13 falsified / non-deployable hypotheses

**First wave (1–10):** directional signals (LS-ratio, Funding+OI,
public-API sweep, MTF pullback ×3), ML filtering, LLM/sentiment fusion,
delta-neutral funding carry, new-listing microstructure.

**Second wave (11–13), data-only prechecks:** statistical arbitrage /
cointegration (no cointegration among crypto alts — EG/Johansen agree,
half-lives are random-walk scale), BTC/ETH→alt lead-lag (statistically
real for TRX but 1–2 bps ≪ ~15 bps cost — the tradeable part is
arbitraged away), and low-frequency momentum (TSMOM/XSMOM — no portable
edge: the lone 2/56 "pass" was ETH-only and shown non-portable by
multi-asset walk-forward; "crisis-alpha" proved crash-morphology-dependent
— LUNA caught, FTX whipsawed 7/8 — not a general property). The
funding-carry finding was reconfirmed on BTC/ETH (carry ≈ financing cost).

Each has a `docs/plans/*-design.md` / `*-result.md` pair. Consolidated
analysis in
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
- `src/{statarb,carry,leadlag,momentum}/` — data-only research prechecks (second wave)
- `scripts/` — reproducible research/backtest scripts
- `results/` — precheck JSON outputs + plots (reproducible)

## Quick start (research / reproduction)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_tests.sh                          # test suite
python scripts/run_backtest.py --symbol XRPUSDT --no-ml   # baseline backtest
python scripts/funding_carry_backtest.py           # a falsified hypothesis, reproducible
python -m src.statarb.precheck                     # a data-only precheck gate (FAIL ⇒ stop before backtest)
python -m src.momentum.crisis_alpha                # multi-asset crisis-alpha generality test
```

Operating the bot itself (API keys, Docker, deployment) is documented in
[`docs/README.ko.md`](docs/README.ko.md). It trades real funds when
enabled — validate on Binance Testnet first; past results do not
guarantee future results; you bear all risk.

## Contributing

Contributions are welcome — but the scope is unusual (this is a
negative-results research project, not a product). Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first. The most valuable
contributions: reproduce or dispute a documented result, or add a new
falsification experiment *with the same pre-committed-kill-criteria
discipline*. There are tagged good-first-issues in `CONTRIBUTING.md`.

## License

[MIT](LICENSE). Research project documenting negative results — not a
profitable or maintained trading system; not financial advice; trades
real funds when enabled; you bear all risk.

---

*This project's most honest output is a fast, cheap "no." It proved
"there is no edge here" thirteen times without losing capital, and built
the discipline and tooling to do so. See the
[case study](docs/CASE-STUDY-quant-research.en.md).*
