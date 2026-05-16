# Case Study — Falsification-First Quantitative Trading Research

> One line: **A systematic research program that falsified 10 trading
> hypotheses using pre-committed kill criteria and data-first validation,
> proving — at $0 capital lost — that there is no retail alpha in
> public-data crypto.** The deliverable is not alpha; it is a *reusable
> falsification discipline plus execution infrastructure*.

This document is not a record of "10 failures." It is a portfolio asset
showing **a methodology for killing bad ideas cheaply before risking
capital, with 10 traceable examples.** Evidence of negative results
handled with rigor is exactly what most candidates cannot show.

---

## 1. Methodology (this is the actual asset)

Every hypothesis was forced through the same loop:

1. **Structure the hypothesis** into a measurable claim.
2. **Pre-commit kill criteria *before* running** — PF / Sharpe / N /
   symmetry / yearly-consistency thresholds fixed in advance and never
   moved after seeing results.
3. **Cheapest validation first** — data-only backtests before any
   capital or live trading; data *availability itself* is a gate
   (if data can't support the test, kill it there).
4. **Honest verdict** — scripts emit numbers only; the pass/fail
   judgment is made by a human against the pre-committed criteria.
   Negative results are first-class artifacts (`docs/plans/*-result.md`).
5. **Paradigm-level escalation** — no endless parameter tweaking. When
   a paradigm dies, change the *category* (directional → structural
   carry → event-driven → non-alpha), not the knobs.

Core principle: **minimize the cost of validation before maximizing
confidence in the result.** All 10 were falsified within hours-to-a-day
at $0 cost.

---

## 2. The 10 Experiments (design + result docs in `docs/plans/`)

| # | Hypothesis | Paradigm | Why it was falsified | Transferable lesson |
|---|------------|----------|----------------------|---------------------|
| 1 | LS-ratio signal | Directional | No edge confirmed | Public positioning is non-differentiating |
| 2 | Funding+OI standalone | Directional | SHORT-only profit, LONG/SHORT asymmetric | One-sided profit = overfit signal |
| 3 | Binance public-API sweep | Directional | No standalone edge | Public data is arbitraged away |
| 4 | ML filter (LGBM/ONNX) | ML | ML OFF > ML ON | ML is futile on alpha-free features |
| 5 | MTF pullback bot | Directional | OOS failure | IS overfit; OOS is the truth |
| 6 | MTF OOS re-test | Directional | fees-only PF 0.84, asymmetric | Net-of-fees is what counts |
| 7 | MTF + BTC filter | Directional | Filter made baseline *worse* | A filter cannot turn −EV into +EV |
| 8 | TradingAgents sentiment fusion | LLM/sentiment | N=19 sample-poverty; news-only modality gap | Backtest ≠ live signal source ⇒ unverifiable |
| 9 | Delta-neutral funding carry | Structural / market-neutral | Idealized gross +4.6%/yr but net −0.37%; actually regime-dependent bull-premium | gross ≠ net; verify "structural" claims via regime decomposition |
| 10 | New-listing microstructure | Asymmetry / event | Net negative at N=323; a +275 bps hint on partial data vanished on the full sample | Low-N, cherry-pick and survivorship caught by discipline |

(1–7 are consolidated in `2026-05-04-strategy-post-mortem.md`; 8–10 are
this cycle. Each has traceable `*-design.md` / `*-result.md`.)

---

## 3. Meta-Lessons — Evidence of Senior Judgment

Each item is a concrete interview/review example of *judgment*:

- **Sample-size discipline**: in #8, discovered a single-symbol
  entry-filter produced ≤22 trades over 25 months (N=19) → accepted
  "statistically unverifiable" as the result instead of forcing a
  conclusion. In #10, deliberately designed an N=323 event study for
  statistical power.
- **Catching overfit in real time (best evidence)**: in #10 a partial
  sample (95 listings) showed a +275 bps "hint" — instead of excitement,
  flagged it as a single-cell, low-N artifact → on the full N=323 it
  collapsed to −82 bps. Pre-committed criteria + full-sample discipline
  prevented an expensive live failure.
- **Reasoning about survivorship bias directionally**: in #10 the data
  contained only currently-listed symbols → explicitly stated the bias
  *inflates* apparent edge, with the logic "if it's negative even so,
  it's a confirmed kill."
- **gross ≠ net**: in #9 decomposed an idealized +4.6%/yr gross that
  failed to clear spot-financing cost (~5%/yr) via a cost matrix.
- **Exposing hidden exposure**: in #9 broke the "market-neutral" claim
  via regime decomposition, revealing a directional bull-premium bet.
- **Reproducibility engineering**: in #8 made a non-deterministic LLM
  *bit-reproducible* via `temperature=0` + response-hash cache so the
  backtest was verifiable (zero dependencies, no LangChain, direct httpx).
- **Negative results as first-class**: 10 `*-result.md` files + plan
  history + memory permanently record "why it died + do-not-retry
  conditions" so the same loop is never re-run.

---

## 4. Engineering Artifacts (reusable infrastructure)

- **Async multi-symbol execution** — `src/bot.py`: per-symbol bots via
  `asyncio.gather`, shared singleton `RiskManager` (global loss limit /
  same-direction cap, `asyncio.Lock`).
- **Dual-layer kill switch** — Fast (8 consecutive losses) / Slow
  (15-trade PF < 0.75), JSONL-persisted, boot-time retroactive check —
  keeps a −EV system from dying slowly.
- **Vectorized backtester** — `src/backtester.py`: walk-forward,
  slippage/fee model, deterministic, extended for custom event studies
  (`funding_carry_backtest.py`, `listing_microstructure_backtest.py`).
- **Deterministic local-LLM integration** — `src/sentiment_provider.py`:
  direct OpenAI-compatible local-MLX calls, response cache for
  reproducibility, graceful degradation.
- **Data pipelines** — Binance fapi collection, cached, idempotent/
  resumable (`collect_listings.py`, `build_sentiment_dataset.py`).
- **Test discipline** — gate/provider regression tests
  (`tests/test_sentiment_*`), baseline no-op guarantee, look-ahead
  guard unit tests.

---

## 5. Honest Conclusion

Across directional prediction, structural carry, and event asymmetry —
**every paradigm was falsified once costs and biases were faced
honestly.** This is not failure; it is the *correct finding* about an
efficient market. The value delivered:

> **A fast, cheap NO instead of a slow, expensive one.**
> Proved "there is no edge here" 10 times at $0 capital lost, and built
> a reusable falsification discipline and execution stack along the way.

---

## 6. Using, Reproducing & Extending This Work

This is open source (MIT) so others don't repeat the same dead ends.

**Reproduce a result**: every hypothesis has a `docs/plans/*-design.md`
(with pre-committed kill criteria) + `*-result.md`, and a runnable
script. Example:
```bash
python scripts/funding_carry_backtest.py            # #9, fully reproducible
python scripts/run_backtest.py --symbol XRPUSDT --sentiment-mode off  # #8 baseline
```

**Dispute a result**: the most valuable contribution here is a
well-argued "your kill was wrong because X." The kill criteria are
written down precisely so they can be checked against.

**Reuse a component** (each is reasonably self-contained):
- deterministic LLM provider — `temperature=0` + response-hash cache,
  zero deps (`src/sentiment_provider.py`)
- vectorized walk-forward backtester with cost model (`src/backtester.py`)
- idempotent/resumable Binance fapi pipelines (`scripts/collect_*.py`,
  `scripts/build_sentiment_dataset.py`)
- dual-layer kill switch (`src/bot.py`)

**Add a new falsification experiment**: follow the same discipline —
fix kill criteria *before* running, data-only validation first, honest
`*-result.md` whatever the outcome. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).

**The methodology generalizes** beyond trading: pre-committed kill
criteria + cheapest-possible validation + negative results as first-class
artifacts applies to any empirical/ML research where it is easy to fool
yourself with overfit, survivorship bias, or gross-vs-net confusion.

**Where the evidence lives**:
`docs/plans/2026-05-04-strategy-post-mortem.md` (experiments 1–7),
`docs/plans/2026-05-16~17-*-result.md` (8–10), `CLAUDE.md` plan history,
`scripts/*_backtest.py` (reproducible code).
