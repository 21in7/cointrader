# Contributing

Thanks for your interest. Please read this first — the scope here is
unusual.

## What this project is

A **research project that documents negative results.** Across its life,
10 trading hypotheses were falsified using pre-committed kill criteria
and data-first validation, at $0 capital lost. See
[`docs/CASE-STUDY-quant-research.en.md`](docs/CASE-STUDY-quant-research.en.md).

It is **not** a profitable bot and is **not** maintained as a product.
The value to the community is honest, reproducible "this does not work
and here is exactly why" — rare in a space full of overfit, survivorship-
biased "look at my backtest" repos.

## Contributions that fit

- **Reproduce / dispute a documented result.** Run the scripts, check
  the numbers, find a methodology flaw. A well-argued "your kill was
  wrong because X" is the most valuable contribution possible here.
- **Add a new falsification experiment** — but only with the same
  discipline: a `docs/plans/YYYY-MM-DD-<name>-design.md` that fixes
  **kill criteria before running**, a reproducible data-only script, and
  an honest `*-result.md`. Cherry-picked positive backtests without
  pre-committed criteria, OOS separation, and cost modeling will be
  declined — that is the exact failure mode this project documents.
- **Extract reusable components** — the deterministic LLM-cache pattern
  (`src/sentiment_provider.py`), the walk-forward backtester
  (`src/backtester.py`), idempotent Binance pipelines, the dual-layer
  kill switch. Clean library extraction is welcome.
- **Correctness / tests / docs.** Look-ahead-bias guards, reproducibility
  fixes, clearer docs.

## Contributions that do NOT fit

- "I made it profitable" PRs without pre-committed kill criteria,
  out-of-sample separation, realistic fees/slippage, and reproducible
  code. Extraordinary claims need the same rigor used to kill the other
  10 ideas.
- Turning this into a maintained trading product. That is out of scope.

## Good first issues

- **Fix the brittle regression test.**
  `tests/test_evaluate_oos.py::test_regression_fees_only_cum_pnl`
  hard-codes expectations against a live-growing JSONL file and fails as
  data accumulates. Replace with a fixed fixture so the suite is
  deterministic. (Known issue; see git history / result docs.)
- Pin requirements / add a clean `requirements.txt` reproduction path.
- Survivorship-bias note: the new-listing study uses only currently
  listed symbols. A delisted-symbol data source would strengthen #10.

## Process

1. Open an issue describing the change (especially for new experiments —
   discuss the kill criteria *before* running).
2. Keep PRs small and focused. Tests must pass: `bash scripts/run_tests.sh`.
3. For research PRs, include the design + result docs, not just code.

## Tooling

Python 3.11+, `pytest`. External APIs (Binance, Discord) are mocked in
tests. No real network or credentials in the test suite.
