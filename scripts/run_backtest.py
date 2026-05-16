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
import pandas as pd
import pandas_ta as ta

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

    # 센티먼트 게이트 (게이트 B)
    p.add_argument("--sentiment-mode", choices=["off", "veto", "contrarian", "confirm"],
                   default="off", help="센티먼트 게이트 모드 (기본: off=베이스라인)")
    p.add_argument("--sentiment-threshold", type=float, default=0.5,
                   help="veto/confirm 충돌 판정 |score| (기본: 0.5)")
    p.add_argument("--sentiment-extreme-band", type=float, default=1.0,
                   help="contrarian 극단 판정 |score| (기본: 1.0)")

    # Walk-Forward
    p.add_argument("--walk-forward", action="store_true", help="Walk-Forward 백테스트 (기간별 모델 학습/검증)")
    p.add_argument("--compare-ml", action="store_true",
                   help="ML on vs off Walk-Forward 비교 (--walk-forward 자동 활성화)")
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


def _classify_regime(btc_return: float, btc_avg_adx: float) -> str:
    """BTC ADX와 수익률 기반 시장 레짐 분류."""
    if btc_avg_adx >= 25:
        return "상승 추세" if btc_return > 0 else "하락 추세"
    return "횡보"


def _calc_fold_market_context(
    raw_df: pd.DataFrame, test_start: str, test_end: str
) -> dict:
    """폴드 기간의 BTC/ETH 수익률과 시장 레짐 계산."""
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)

    idx = raw_df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if ts_start.tz is not None:
        ts_start = ts_start.tz_localize(None)
    if ts_end.tz is not None:
        ts_end = ts_end.tz_localize(None)

    fold_df = raw_df[(idx >= ts_start) & (idx < ts_end)]
    if len(fold_df) < 20:
        return None

    # BTC return
    btc_start = fold_df["close_btc"].iloc[0]
    btc_end = fold_df["close_btc"].iloc[-1]
    btc_return = (btc_end - btc_start) / btc_start * 100

    # ETH return
    eth_start = fold_df["close_eth"].iloc[0]
    eth_end = fold_df["close_eth"].iloc[-1]
    eth_return = (eth_end - eth_start) / eth_start * 100

    # BTC ADX (period average)
    adx_df = ta.adx(fold_df["high_btc"], fold_df["low_btc"], fold_df["close_btc"], length=14)
    btc_avg_adx = adx_df["ADX_14"].mean()
    if np.isnan(btc_avg_adx):
        btc_avg_adx = 0.0

    regime = _classify_regime(btc_return, btc_avg_adx)

    return {
        "btc_return_pct": round(btc_return, 1),
        "eth_return_pct": round(eth_return, 1),
        "btc_avg_adx": round(btc_avg_adx, 1),
        "market_regime": regime,
    }


def _load_ls_ratio(symbol: str, test_start: str, test_end: str) -> dict | None:
    """폴드 기간의 L/S ratio 평균값 로드. 데이터 없으면 None."""
    path = Path(f"data/{symbol.lower()}/ls_ratio_15m.parquet")
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)

    # tz 맞추기
    if df["timestamp"].dt.tz is not None:
        if ts_start.tz is None:
            ts_start = ts_start.tz_localize("UTC")
        if ts_end.tz is None:
            ts_end = ts_end.tz_localize("UTC")

    mask = (df["timestamp"] >= ts_start) & (df["timestamp"] < ts_end)
    period_df = df[mask]

    if period_df.empty:
        return None

    return {
        "top_acct_avg": round(period_df["top_acct_ls_ratio"].mean(), 2),
        "global_avg": round(period_df["global_ls_ratio"].mean(), 2),
    }


def calc_market_context(folds: list[dict], symbols: list[str]) -> list[dict]:
    """각 폴드에 대한 시장 컨텍스트 계산."""
    # XRP parquet에서 BTC/ETH 데이터 로드 (임베딩됨)
    primary_sym = symbols[0].lower()
    raw_path = Path(f"data/{primary_sym}/combined_15m.parquet")
    if not raw_path.exists():
        logger.warning(f"데이터 파일 없음: {raw_path}")
        return []

    raw_df = pd.read_parquet(raw_path)
    if "close_btc" not in raw_df.columns or "close_eth" not in raw_df.columns:
        logger.warning("BTC/ETH 상관 데이터 없음")
        return []

    contexts = []
    for fold in folds:
        test_start = fold.get("test_start")
        test_end = fold.get("test_end")
        if not test_start or not test_end:
            contexts.append({"fold": fold["fold"], "market_context": None})
            continue

        ctx = _calc_fold_market_context(raw_df, test_start, test_end)
        if ctx is None:
            contexts.append({"fold": fold["fold"], "market_context": None})
            continue

        # L/S ratio (XRP, BTC, ETH)
        ls_data = {}
        for ls_sym in ["xrpusdt", "btcusdt", "ethusdt"]:
            ls = _load_ls_ratio(ls_sym, test_start, test_end)
            if ls:
                ls_data[ls_sym.replace("usdt", "")] = ls

        ctx["ls_ratio"] = ls_data if ls_data else None
        contexts.append({"fold": fold["fold"], "market_context": ctx})

    return contexts


def print_market_context(contexts: list[dict]):
    """시장 컨텍스트 테이블 출력."""
    if not contexts:
        return

    # Market Regime 테이블
    print("\n📊 Market Context per Fold")
    print(f"{'─' * 80}")
    print(f"  {'Fold':>4}  {'BTC Return':>12}  {'ETH Return':>12}  {'Market Regime':<32}")
    print(f"{'─' * 80}")

    for c in contexts:
        ctx = c.get("market_context")
        if ctx is None:
            print(f"  {c['fold']:>4}  {'N/A':>12}  {'N/A':>12}  {'N/A':<32}")
        else:
            regime_str = f"{ctx['market_regime']} (BTC ADX {ctx['btc_avg_adx']:.0f})"
            print(f"  {c['fold']:>4}  {ctx['btc_return_pct']:>+11.1f}%  "
                  f"{ctx['eth_return_pct']:>+11.1f}%  {regime_str:<32}")
    print(f"{'─' * 80}")

    # L/S Ratio 테이블 (데이터 있는 폴드가 하나라도 있으면)
    has_ls = any(
        c.get("market_context") and c["market_context"].get("ls_ratio")
        for c in contexts
    )

    if has_ls:
        print("\n📊 L/S Ratio Context per Fold (period avg)")
        print(f"{'─' * 80}")
        print(f"  {'Fold':>4}  {'XRP Top/Global':>18}  {'BTC Top/Global':>18}  {'ETH Top/Global':>18}")
        print(f"{'─' * 80}")
        for c in contexts:
            ctx = c.get("market_context")
            ls = ctx.get("ls_ratio") if ctx else None
            parts = []
            for sym in ["xrp", "btc", "eth"]:
                if ls and sym in ls:
                    parts.append(f"{ls[sym]['top_acct_avg']:.2f} / {ls[sym]['global_avg']:.2f}")
                else:
                    parts.append("N/A")
            print(f"  {c['fold']:>4}  {parts[0]:>18}  {parts[1]:>18}  {parts[2]:>18}")
        print(f"{'─' * 80}")
    else:
        print("  ℹ️ L/S ratio 데이터 없음 — collector 데이터 축적 후 표시됩니다")


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


def compare_ml(symbols: list[str], args):
    """ML on vs ML off Walk-Forward 백테스트 비교."""
    base_kwargs = dict(
        symbols=symbols,
        start=args.start,
        end=args.end,
        initial_balance=args.balance,
        leverage=args.leverage,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
        ml_threshold=args.ml_threshold,
        atr_sl_mult=args.sl_atr,
        atr_tp_mult=args.tp_atr,
        signal_threshold=args.signal_threshold,
        adx_threshold=args.adx_threshold,
        volume_multiplier=args.vol_multiplier,
        train_months=args.train_months,
        test_months=args.test_months,
    )

    results = {}
    for label, use_ml in [("ML OFF", False), ("ML ON", True)]:
        print(f"\n{'='*60}")
        print(f"  Walk-Forward 백테스트: {label}")
        print(f"{'='*60}")

        cfg = WalkForwardConfig(**base_kwargs, use_ml=use_ml)
        wf = WalkForwardBacktester(cfg)
        result = wf.run()
        results[label] = result
        print_summary(result["summary"], cfg, mode="walk_forward")
        if result.get("folds"):
            print_fold_table(result["folds"])
            # 시장 컨텍스트는 첫 번째 실행에서만 출력 (동일 데이터)
            if label == "ML OFF":
                contexts = calc_market_context(result["folds"], symbols)
                if contexts:
                    print_market_context(contexts)

    _print_comparison(results, symbols)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(symbols) == 1:
        out_dir = Path(f"results/{symbols[0].lower()}")
    else:
        out_dir = Path("results/combined")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ml_comparison_{ts}.json"

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "ml_off": results["ML OFF"]["summary"],
        "ml_on": results["ML ON"]["summary"],
    }

    def sanitize(obj):
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            if isinstance(obj, float) and obj == float("inf"):
                return "Infinity"
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(sanitize(comparison), f, indent=2, ensure_ascii=False)
    print(f"\n비교 결과 저장: {path}")


def _print_comparison(results: dict, symbols: list[str]):
    """ML on vs off 비교 리포트 출력."""
    off = results["ML OFF"]["summary"]
    on = results["ML ON"]["summary"]

    print(f"\n{'='*64}")
    print(f"  ML ON vs OFF 비교 ({', '.join(symbols)})")
    print(f"{'='*64}")
    print(f"  {'지표':<20} {'ML OFF':>12} {'ML ON':>12} {'Delta':>12}")
    print(f"{'─'*64}")

    metrics = [
        ("총 거래", "total_trades", "d"),
        ("총 PnL (USDT)", "total_pnl", ".2f"),
        ("수익률 (%)", "return_pct", ".2f"),
        ("승률 (%)", "win_rate", ".1f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("MDD (%)", "max_drawdown_pct", ".2f"),
        ("Sharpe", "sharpe_ratio", ".2f"),
    ]

    for label, key, fmt in metrics:
        v_off = off.get(key, 0)
        v_on = on.get(key, 0)
        if v_off == float("inf"):
            v_off_str = "INF"
        else:
            v_off_str = f"{v_off:{fmt}}"
        if v_on == float("inf"):
            v_on_str = "INF"
        else:
            v_on_str = f"{v_on:{fmt}}"

        if isinstance(v_off, (int, float)) and isinstance(v_on, (int, float)) \
                and v_off != float("inf") and v_on != float("inf"):
            delta = v_on - v_off
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:{fmt}}"
        else:
            delta_str = "N/A"

        print(f"  {label:<20} {v_off_str:>12} {v_on_str:>12} {delta_str:>12}")

    pf_off = off.get("profit_factor", 0)
    pf_on = on.get("profit_factor", 0)
    wr_off = off.get("win_rate", 0)
    wr_on = on.get("win_rate", 0)
    mdd_off = off.get("max_drawdown_pct", 0)
    mdd_on = on.get("max_drawdown_pct", 0)

    print(f"{'─'*64}")

    if pf_off == float("inf") or pf_on == float("inf"):
        print(f"  판정: PF=INF — 한쪽 모드에서 손실 거래 없음 (거래 수 부족 가능), 판단 보류")
    elif pf_off == 0:
        print(f"  판정: ML OFF PF=0 — baseline 거래 없음, 판단 불가")
    else:
        pf_improvement = pf_on - pf_off
        wr_improvement = wr_on - wr_off
        mdd_improvement = mdd_off - mdd_on

        improvements = []
        if pf_improvement > 0.1:
            improvements.append(f"PF +{pf_improvement:.2f}")
        if wr_improvement > 2.0:
            improvements.append(f"승률 +{wr_improvement:.1f}%p")
        if mdd_improvement > 1.0:
            improvements.append(f"MDD -{mdd_improvement:.1f}%p")

        if len(improvements) >= 2:
            verdict = f"ML 필터 투입 가치 있음 ({', '.join(improvements)})"
        elif len(improvements) == 1:
            verdict = f"ML 필터 조건부 투입 ({improvements[0]}, 다른 지표 변화 미미)"
        else:
            verdict = f"ML 필터 기여 미미 (PF {pf_improvement:+.2f}, 승률 {wr_improvement:+.1f}%p)"
        print(f"  판정: {verdict}")

    print(f"{'='*64}\n")


def main():
    args = parse_args()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.compare_ml:
        if args.no_ml:
            logger.warning("--no-ml is ignored when using --compare-ml")
        compare_ml(symbols, args)
        return

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
            sentiment_mode=args.sentiment_mode,
            sentiment_threshold=args.sentiment_threshold,
            sentiment_extreme_band=args.sentiment_extreme_band,
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
            contexts = calc_market_context(result["folds"], symbols)
            if contexts:
                print_market_context(contexts)
                # JSON에 market_context 추가
                for fold, ctx in zip(result["folds"], contexts):
                    fold["market_context"] = ctx.get("market_context")
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
            sentiment_mode=args.sentiment_mode,
            sentiment_threshold=args.sentiment_threshold,
            sentiment_extreme_band=args.sentiment_extreme_band,
        )
        logger.info(f"백테스트 시작: {', '.join(symbols)}")
        bt = Backtester(cfg)
        result = bt.run()
        print_summary(result["summary"], cfg)
        save_result(result, cfg)


if __name__ == "__main__":
    main()
