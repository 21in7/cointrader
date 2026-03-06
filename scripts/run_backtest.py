#!/usr/bin/env python3
"""
백테스트 CLI 진입점.

사용법:
  python scripts/run_backtest.py --symbol XRPUSDT
  python scripts/run_backtest.py --symbols XRPUSDT,TRXUSDT,DOGEUSDT
  python scripts/run_backtest.py --symbol XRPUSDT --no-ml
  python scripts/run_backtest.py --symbol XRPUSDT --start 2025-06-01 --end 2026-03-01
  python scripts/run_backtest.py --symbol XRPUSDT --fee 0.04 --slippage 0.02
  python scripts/run_backtest.py --symbol XRPUSDT --walk-forward
  python scripts/run_backtest.py --symbol XRPUSDT --walk-forward --train-months 6 --test-months 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime

import numpy as np

from loguru import logger

from src.backtester import Backtester, BacktestConfig, WalkForwardBacktester, WalkForwardConfig


def parse_args():
    p = argparse.ArgumentParser(description="CoinTrader Backtest Engine")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", type=str, help="단일 심볼 (e.g. XRPUSDT)")
    group.add_argument("--symbols", type=str, help="멀티심볼, 콤마 구분 (e.g. XRPUSDT,TRXUSDT,DOGEUSDT)")

    p.add_argument("--start", type=str, default=None, help="시작일 (e.g. 2025-06-01)")
    p.add_argument("--end", type=str, default=None, help="종료일 (e.g. 2026-03-01)")
    p.add_argument("--balance", type=float, default=1000.0, help="초기 잔고 (기본: 1000)")
    p.add_argument("--leverage", type=int, default=10, help="레버리지 (기본: 10)")
    p.add_argument("--fee", type=float, default=0.04, help="taker 수수료 %% (기본: 0.04)")
    p.add_argument("--slippage", type=float, default=0.01, help="슬리피지 %% (기본: 0.01)")
    p.add_argument("--no-ml", action="store_true", help="ML 필터 비활성화")
    p.add_argument("--ml-threshold", type=float, default=0.55, help="ML 임계값 (기본: 0.55)")

    # Strategy params
    p.add_argument("--sl-atr", type=float, default=1.5, help="SL ATR 배수 (기본: 1.5)")
    p.add_argument("--tp-atr", type=float, default=3.0, help="TP ATR 배수 (기본: 3.0)")
    p.add_argument("--signal-threshold", type=int, default=3, help="신호 임계값 (기본: 3)")
    p.add_argument("--adx-threshold", type=float, default=0, help="ADX 필터 (0=비활성화, 기본: 0)")
    p.add_argument("--vol-multiplier", type=float, default=1.5, help="거래량 급증 배수 (기본: 1.5)")

    # Walk-Forward
    p.add_argument("--walk-forward", action="store_true", help="Walk-Forward 백테스트 (기간별 모델 학습/검증)")
    p.add_argument("--train-months", type=int, default=6, help="WF 학습 윈도우 개월 (기본: 6)")
    p.add_argument("--test-months", type=int, default=1, help="WF 검증 윈도우 개월 (기본: 1)")
    return p.parse_args()


def print_summary(summary: dict, cfg, mode: str = "standard"):
    print("\n" + "=" * 60)
    title = "WALK-FORWARD BACKTEST RESULT" if mode == "walk_forward" else "BACKTEST RESULT"
    print(f"  {title}")
    print("=" * 60)
    print(f"  심볼:          {', '.join(cfg.symbols)}")
    print(f"  기간:          {cfg.start or '전체'} ~ {cfg.end or '전체'}")
    print(f"  초기 잔고:     {cfg.initial_balance:,.2f} USDT")
    print(f"  레버리지:      {cfg.leverage}x")
    print(f"  수수료:        {cfg.fee_pct}% | 슬리피지: {cfg.slippage_pct}%")
    if mode == "walk_forward":
        print(f"  학습/검증:     {cfg.train_months}개월 / {cfg.test_months}개월")
    else:
        print(f"  ML 필터:       {'OFF' if not cfg.use_ml else f'ON (threshold={cfg.ml_threshold})'}")
    print("-" * 60)
    print(f"  총 거래:       {summary['total_trades']}건")
    print(f"  총 PnL:        {summary['total_pnl']:+,.4f} USDT")
    print(f"  수익률:        {summary['return_pct']:+.2f}%")
    print(f"  승률:          {summary['win_rate']:.1f}%")
    print(f"  평균 수익:     {summary['avg_win']:+.4f} USDT")
    print(f"  평균 손실:     {summary['avg_loss']:+.4f} USDT")
    pf = summary['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"
    print(f"  Profit Factor: {pf_str}")
    print(f"  최대 낙폭:     {summary['max_drawdown_pct']:.2f}%")
    print(f"  샤프비율:      {summary['sharpe_ratio']:.2f}")
    print(f"  총 수수료:     {summary['total_fees']:,.4f} USDT")
    print("-" * 60)
    print("  청산 사유:")
    for reason, count in summary.get("close_reasons", {}).items():
        pct = count / summary["total_trades"] * 100 if summary["total_trades"] > 0 else 0
        print(f"    {reason:20s} {count:4d}건 ({pct:.1f}%)")
    print("=" * 60)


def print_fold_table(folds: list[dict]):
    print("\n" + "=" * 90)
    print("  FOLD DETAILS")
    print("=" * 90)
    print(f"  {'Fold':>4}  {'Test Period':>25}  {'Trades':>6}  {'PnL':>10}  {'WinRate':>7}  {'PF':>6}  {'MDD':>6}")
    print("-" * 90)
    for f in folds:
        s = f["summary"]
        pf = s["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"
        print(f"  {f['fold']:>4}  {f['test_period']:>25}  {s['total_trades']:>6}  "
              f"{s['total_pnl']:>+10.2f}  {s['win_rate']:>6.1f}%  {pf_str:>6}  {s['max_drawdown_pct']:>5.1f}%")
    print("=" * 90)


def save_result(result: dict, cfg):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = result.get("mode", "standard")
    prefix = "wf_backtest" if mode == "walk_forward" else "backtest"

    for sym in cfg.symbols:
        out_dir = Path(f"results/{sym.lower()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{ts}.json"

    if len(cfg.symbols) > 1:
        out_dir = Path("results/combined")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{ts}.json"

    def sanitize(obj):
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            if isinstance(obj, float):
                if obj == float("inf"):
                    return "Infinity"
                if obj == float("-inf"):
                    return "-Infinity"
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, "w") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {path}")
    return path


def main():
    args = parse_args()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.walk_forward:
        cfg = WalkForwardConfig(
            symbols=symbols,
            start=args.start,
            end=args.end,
            initial_balance=args.balance,
            leverage=args.leverage,
            fee_pct=args.fee,
            slippage_pct=args.slippage,
            use_ml=not args.no_ml,
            ml_threshold=args.ml_threshold,
            atr_sl_mult=args.sl_atr,
            atr_tp_mult=args.tp_atr,
            signal_threshold=args.signal_threshold,
            adx_threshold=args.adx_threshold,
            volume_multiplier=args.vol_multiplier,
            train_months=args.train_months,
            test_months=args.test_months,
        )
        logger.info(f"Walk-Forward 백테스트 시작: {', '.join(symbols)} "
                     f"(학습 {cfg.train_months}개월, 검증 {cfg.test_months}개월)")
        wf = WalkForwardBacktester(cfg)
        result = wf.run()
        print_summary(result["summary"], cfg, mode="walk_forward")
        if result.get("folds"):
            print_fold_table(result["folds"])
        save_result(result, cfg)
    else:
        cfg = BacktestConfig(
            symbols=symbols,
            start=args.start,
            end=args.end,
            initial_balance=args.balance,
            leverage=args.leverage,
            fee_pct=args.fee,
            slippage_pct=args.slippage,
            use_ml=not args.no_ml,
            ml_threshold=args.ml_threshold,
            atr_sl_mult=args.sl_atr,
            atr_tp_mult=args.tp_atr,
            signal_threshold=args.signal_threshold,
            adx_threshold=args.adx_threshold,
            volume_multiplier=args.vol_multiplier,
        )
        logger.info(f"백테스트 시작: {', '.join(symbols)}")
        bt = Backtester(cfg)
        result = bt.run()
        print_summary(result["summary"], cfg)
        save_result(result, cfg)


if __name__ == "__main__":
    main()
