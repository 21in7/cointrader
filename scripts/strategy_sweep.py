#!/usr/bin/env python3
"""
전략 파라미터 스윕: 기존 백테스터를 활용하여 파라미터 조합별 성능을 비교한다.
ML 필터 OFF 상태에서 순수 전략 성능만 측정한다.

사용법:
  python scripts/strategy_sweep.py --symbol XRPUSDT
  python scripts/strategy_sweep.py --symbol XRPUSDT --train-months 3 --test-months 1
  python scripts/strategy_sweep.py --symbols XRPUSDT,TRXUSDT,DOGEUSDT
  python scripts/strategy_sweep.py --symbols XRPUSDT,TRXUSDT,DOGEUSDT --combined
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import itertools
from datetime import datetime

import numpy as np
from loguru import logger

from src.backtester import Backtester, BacktestConfig, WalkForwardBacktester, WalkForwardConfig


# ── 스윕 파라미터 정의 ────────────────────────────────────────────────
PARAM_GRID = {
    "atr_sl_mult":       [1.0, 1.5, 2.0],
    "atr_tp_mult":       [2.0, 3.0, 4.0],
    "signal_threshold":  [3, 4, 5],
    "adx_threshold":     [0, 20, 25, 30],
    "volume_multiplier": [1.5, 2.0, 2.5],
}

# 현재 프로덕션 파라미터
CURRENT_PARAMS = {
    "atr_sl_mult": 2.0,
    "atr_tp_mult": 2.0,
    "signal_threshold": 3,
    "adx_threshold": 25,
    "volume_multiplier": 2.5,
}

EMPTY_SUMMARY = {
    "total_trades": 0, "total_pnl": 0, "return_pct": 0, "win_rate": 0,
    "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
    "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_fees": 0, "close_reasons": {},
}


def generate_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_single_backtest(symbols: list[str], params: dict, train_months: int, test_months: int) -> dict:
    """단일 파라미터 조합으로 walk-forward 백테스트 실행."""
    cfg = WalkForwardConfig(
        symbols=symbols,
        use_ml=False,
        train_months=train_months,
        test_months=test_months,
        atr_sl_mult=params["atr_sl_mult"],
        atr_tp_mult=params["atr_tp_mult"],
        signal_threshold=params["signal_threshold"],
        adx_threshold=params["adx_threshold"],
        volume_multiplier=params["volume_multiplier"],
    )
    wf = WalkForwardBacktester(cfg)
    result = wf.run()
    return result["summary"]


def run_combined_backtest(symbols: list[str], params: dict, train_months: int, test_months: int) -> dict:
    """심볼별 독립 walk-forward 실행 후 합산 결과 반환."""
    per_symbol = {}
    total_gross_profit = 0.0
    total_gross_loss = 0.0
    total_trades = 0
    total_pnl = 0.0

    for sym in symbols:
        try:
            summary = run_single_backtest([sym], params, train_months, test_months)
        except Exception as e:
            logger.warning(f"  {sym} 실패: {e}")
            summary = EMPTY_SUMMARY.copy()

        per_symbol[sym] = summary

        # gross profit/loss 역산
        n = summary["total_trades"]
        if n > 0:
            wr = summary["win_rate"] / 100.0
            n_wins = round(wr * n)
            n_losses = n - n_wins
            gp = summary["avg_win"] * n_wins if n_wins > 0 else 0.0
            gl = abs(summary["avg_loss"]) * n_losses if n_losses > 0 else 0.0
            total_gross_profit += gp
            total_gross_loss += gl
        total_trades += n
        total_pnl += summary["total_pnl"]

    combined_pf = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else float("inf")

    return {
        "params": params,
        "combined_pf": round(combined_pf, 2),
        "combined_trades": total_trades,
        "combined_pnl": round(total_pnl, 2),
        "per_symbol": per_symbol,
    }


def print_results_table(results: list[dict], symbols: list[str], train_months: int, test_months: int):
    sym_str = ",".join(symbols)
    print(f"\n{'=' * 100}")
    print(f"  Strategy Parameter Sweep Results ({sym_str}, Walk-Forward {train_months}/{test_months})")
    print(f"{'=' * 100}")
    print(f"  {'Rank':>4}  {'SL×ATR':>6}  {'TP×ATR':>6}  {'Signal':>6}  {'ADX':>4}  {'Vol':>4}  "
          f"{'Trades':>6}  {'WinRate':>7}  {'PF':>6}  {'MDD':>5}  {'PnL':>10}  {'Sharpe':>6}")
    print(f"  {'-' * 94}")

    for i, r in enumerate(results):
        p = r["params"]
        s = r["summary"]
        pf = s["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"

        is_current = all(p[k] == CURRENT_PARAMS[k] for k in CURRENT_PARAMS)
        marker = " ← CURRENT" if is_current else ""

        print(f"  {i+1:>4}  {p['atr_sl_mult']:>6.1f}  {p['atr_tp_mult']:>6.1f}  "
              f"{p['signal_threshold']:>6}  {p['adx_threshold']:>4.0f}  {p['volume_multiplier']:>4.1f}  "
              f"{s['total_trades']:>6}  {s['win_rate']:>6.1f}%  {pf_str:>6}  {s['max_drawdown_pct']:>4.1f}%  "
              f"{s['total_pnl']:>+10.2f}  {s['sharpe_ratio']:>6.1f}{marker}")

    print(f"{'=' * 100}")


def print_combined_results_table(results: list[dict], symbols: list[str],
                                  train_months: int, test_months: int,
                                  min_pf_count: int = 2, min_pf: float = 0.9):
    sym_str = ",".join(symbols)
    # 심볼 약칭
    short = {s: s.replace("USDT", "") for s in symbols}

    print(f"\n{'=' * 130}")
    print(f"  Combined Strategy Sweep ({sym_str}, WF {train_months}/{test_months})")
    print(f"  Filter: {min_pf_count}+ symbols with PF >= {min_pf}")
    print(f"{'=' * 130}")

    # 헤더
    sym_headers = "  ".join(f"{short[s]:>12s}" for s in symbols)
    print(f"  {'Rank':>4}  {'SL':>4}  {'TP':>4}  {'Sig':>3}  {'ADX':>3}  {'Vol':>4}  "
          f"{'Tot':>4}  {'CombPF':>6}  {'PnL':>9}  {sym_headers}")

    # 심볼별 서브헤더
    sub = "  ".join(f"{'PF/WR%/Trd':>12s}" for _ in symbols)
    print(f"  {'':>4}  {'':>4}  {'':>4}  {'':>3}  {'':>3}  {'':>4}  "
          f"{'':>4}  {'':>6}  {'':>9}  {sub}")
    print(f"  {'-' * 124}")

    for i, r in enumerate(results):
        p = r["params"]
        cpf = r["combined_pf"]
        cpf_str = f"{cpf:.2f}" if cpf != float("inf") else "INF"

        is_current = all(p[k] == CURRENT_PARAMS[k] for k in CURRENT_PARAMS)
        marker = " ←CUR" if is_current else ""

        # 심볼별 PF/WR/Trades
        sym_cols = []
        for s in symbols:
            ss = r["per_symbol"][s]
            spf = ss["profit_factor"]
            spf_str = f"{spf:.1f}" if spf != float("inf") else "INF"
            sym_cols.append(f"{spf_str}/{ss['win_rate']:.0f}%/{ss['total_trades']}")

        sym_detail = "  ".join(f"{c:>12s}" for c in sym_cols)

        print(f"  {i+1:>4}  {p['atr_sl_mult']:>4.1f}  {p['atr_tp_mult']:>4.1f}  "
              f"{p['signal_threshold']:>3}  {p['adx_threshold']:>3.0f}  {p['volume_multiplier']:>4.1f}  "
              f"{r['combined_trades']:>4}  {cpf_str:>6}  {r['combined_pnl']:>+9.1f}  "
              f"{sym_detail}{marker}")

    print(f"{'=' * 130}")
    print(f"  표시된 조합: {len(results)}개 / 전체 324개")
    print(f"  심볼별 칼럼: PF/승률%/거래수")


def save_results(results: list[dict], symbols: list[str]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for sym in symbols:
        out_dir = Path(f"results/{sym.lower()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"strategy_sweep_{ts}.json"

    if len(symbols) > 1:
        out_dir = Path("results/combined")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"strategy_sweep_{ts}.json"

    def sanitize(obj):
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, float) and obj == float("inf"):
            return "Infinity"
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(sanitize(results), f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {path}")


def main():
    p = argparse.ArgumentParser(description="Strategy Parameter Sweep")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", type=str)
    group.add_argument("--symbols", type=str)
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--combined", action="store_true",
                   help="심볼별 독립 실행 후 합산 PF 기준 정렬 (--symbols 필수)")
    p.add_argument("--min-pf", type=float, default=0.9,
                   help="심볼별 최소 PF 필터 (기본: 0.9)")
    p.add_argument("--min-pf-count", type=int, default=2,
                   help="최소 PF 충족 심볼 수 (기본: 2)")
    args = p.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else [s.strip().upper() for s in args.symbols.split(",")]

    if args.combined:
        if len(symbols) < 2:
            logger.error("--combined 모드는 --symbols에 2개 이상 심볼 필요")
            sys.exit(1)
        run_combined_sweep(symbols, args)
    else:
        run_single_sweep(symbols, args)


def run_single_sweep(symbols: list[str], args):
    combos = generate_combinations(PARAM_GRID)
    logger.info(f"스윕 시작: {len(combos)}개 조합, 심볼={','.join(symbols)}")

    results = []
    for i, params in enumerate(combos):
        param_str = " | ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"  [{i+1}/{len(combos)}] {param_str}")

        try:
            summary = run_single_backtest(symbols, params, args.train_months, args.test_months)
            results.append({"params": params, "summary": summary})
        except Exception as e:
            logger.warning(f"  실패: {e}")
            results.append({"params": params, "summary": EMPTY_SUMMARY.copy()})

    # PF 기준 내림차순 정렬
    def sort_key(r):
        pf = r["summary"]["profit_factor"]
        return pf if pf != float("inf") else 999
    results.sort(key=sort_key, reverse=True)

    print_results_table(results, symbols, args.train_months, args.test_months)
    save_results(results, symbols)


def run_combined_sweep(symbols: list[str], args):
    combos = generate_combinations(PARAM_GRID)
    total_runs = len(combos) * len(symbols)
    logger.info(f"합산 스윕 시작: {len(combos)}개 조합 × {len(symbols)}심볼 = {total_runs}회")

    results = []
    for i, params in enumerate(combos):
        param_str = " | ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"  [{i+1}/{len(combos)}] {param_str}")

        r = run_combined_backtest(symbols, params, args.train_months, args.test_months)
        results.append(r)

    # 필터: N개 이상 심볼에서 PF >= min_pf
    filtered = []
    for r in results:
        pf_pass = sum(
            1 for s in symbols
            if r["per_symbol"][s]["profit_factor"] >= args.min_pf
            and r["per_symbol"][s]["total_trades"] > 0
        )
        if pf_pass >= args.min_pf_count:
            filtered.append(r)

    # 합산 PF 기준 정렬
    def sort_key(r):
        pf = r["combined_pf"]
        return pf if pf != float("inf") else 999
    filtered.sort(key=sort_key, reverse=True)

    print_combined_results_table(filtered, symbols, args.train_months, args.test_months,
                                  min_pf_count=args.min_pf_count, min_pf=args.min_pf)
    save_results(filtered, symbols)


if __name__ == "__main__":
    main()
