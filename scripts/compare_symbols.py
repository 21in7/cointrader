#!/usr/bin/env python3
"""
종목 비교 백테스트: 후보 심볼별 파라미터 sweep → 최적 파라미터 기준 비교표 출력.

사용법:
  python scripts/compare_symbols.py --symbols SOLUSDT LINKUSDT AVAXUSDT
  python scripts/compare_symbols.py --symbols SOLUSDT LINKUSDT AVAXUSDT --skip-fetch
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import subprocess
from datetime import date

from loguru import logger

from src.backtester import WalkForwardBacktester, WalkForwardConfig
from scripts.strategy_sweep import generate_combinations, PARAM_GRID


TRAIN_MONTHS = 3
TEST_MONTHS = 1
FETCH_DAYS = 365


def fetch_data(symbols: list[str], days: int = FETCH_DAYS) -> None:
    script = str(Path(__file__).parent / "fetch_history.py")
    for sym in symbols:
        cmd = [
            sys.executable, script,
            "--symbol", sym,
            "--interval", "15m",
            "--days", str(days),
        ]
        logger.info(f"데이터 수집: {sym} (최근 {days}일)")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"  {sym} 수집 실패: {result.stderr[:300]}")
        else:
            logger.info(f"  {sym} 수집 완료")


def run_backtest(symbol: str, params: dict) -> dict:
    cfg = WalkForwardConfig(
        symbols=[symbol],
        use_ml=False,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        **params,
    )
    wf = WalkForwardBacktester(cfg)
    return wf.run()


def sweep_symbol(symbol: str) -> dict:
    """심볼별 파라미터 sweep 실행 → 최적 조합 반환."""
    combos = generate_combinations(PARAM_GRID)
    logger.info(f"[{symbol}] 파라미터 sweep 시작: {len(combos)}개 조합")

    best = None
    best_params = None

    for i, params in enumerate(combos):
        try:
            result = run_backtest(symbol, params)
            summary = result["summary"]

            # 거래 5건 미만은 스킵
            if summary["total_trades"] < 5:
                continue

            # PF 기준으로 최적 선택 (동률 시 승률 → 손익비 순)
            if best is None or _is_better(summary, best):
                best = summary
                best_params = params

        except Exception as e:
            logger.warning(f"  [{symbol}] 조합 {i+1} 실패: {e}")

        if (i + 1) % 50 == 0:
            logger.info(f"  [{symbol}] {i+1}/{len(combos)} 완료")

    logger.info(f"[{symbol}] sweep 완료 → 최적 PF: {best['profit_factor'] if best else 'N/A'}")
    return {"symbol": symbol, "best_params": best_params, "summary": best}


def _is_better(new: dict, old: dict) -> bool:
    """PF → 손익비 → 승률 순으로 비교."""
    new_pf = new["profit_factor"] if new["profit_factor"] != float("inf") else 999
    old_pf = old["profit_factor"] if old["profit_factor"] != float("inf") else 999

    if new_pf != old_pf:
        return new_pf > old_pf
    new_pr = new.get("payoff_ratio", 0) or 0
    old_pr = old.get("payoff_ratio", 0) or 0
    if new_pr != old_pr:
        return new_pr > old_pr
    return new["win_rate"] > old["win_rate"]


def print_comparison(results: list[dict]) -> None:
    header = (
        f"{'심볼':<10} {'파라미터':^30} {'거래수':>6} {'승률':>7} "
        f"{'손익비':>7} {'연속손실':>8} {'PF':>6} {'수익률':>8} {'MDD':>6} {'총PnL':>10}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("종목 비교 백테스트 결과 (심볼별 최적 파라미터)")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in results:
        s = r["summary"]
        p = r["best_params"]
        if not s or not p:
            print(f"{r['symbol'].replace('USDT', ''):<10} {'데이터 부족 또는 sweep 실패':^30}")
            continue

        short = r["symbol"].replace("USDT", "")
        param_str = f"SL={p['atr_sl_mult']}/TP={p['atr_tp_mult']}/ADX={p['adx_threshold']}"
        pf = s["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"

        print(
            f"{short:<10} {param_str:^30} {s['total_trades']:>6} "
            f"{s['win_rate']:>6.1f}% "
            f"{s.get('payoff_ratio', 0):>7.2f} "
            f"{s.get('max_consecutive_losses', 0):>8} "
            f"{pf_str:>6} "
            f"{s['return_pct']:>7.2f}% "
            f"{s['max_drawdown_pct']:>5.1f}% "
            f"{s['total_pnl']:>+10.2f}"
        )

    print("-" * len(header))
    print("\n[판정 기준]")
    print("  - 승률 50%+ & 손익비 1.0+ → 실전 지속 가능")
    print("  - 연속 손실 5회 이하 → 멘탈 관리 가능")
    print("  - 거래 20건+ → 통계적 유의성 있음")
    print()

    # 상세 파라미터 출력
    print("[심볼별 최적 파라미터 상세]")
    for r in results:
        if r["best_params"]:
            p = r["best_params"]
            print(f"  {r['symbol']}: SL={p['atr_sl_mult']}, TP={p['atr_tp_mult']}, "
                  f"Signal={p['signal_threshold']}, ADX={p['adx_threshold']}, "
                  f"Vol={p['volume_multiplier']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="종목 비교 백테스트 (심볼별 파라미터 sweep)")
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="비교할 심볼 리스트 (e.g., SOLUSDT LINKUSDT AVAXUSDT)",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="데이터 수집 스킵")
    parser.add_argument("--days", type=int, default=FETCH_DAYS, help="데이터 수집 기간 (일)")
    args = parser.parse_args()

    # 1) 데이터 수집
    if not args.skip_fetch:
        fetch_data(args.symbols, args.days)

    # 2) 심볼별 sweep
    results = []
    for sym in args.symbols:
        try:
            result = sweep_symbol(sym)
            results.append(result)
        except Exception as e:
            logger.error(f"  {sym} sweep 실패: {e}")
            results.append({"symbol": sym, "best_params": None, "summary": None})

    # 3) 비교표
    if results:
        print_comparison(results)

        # 4) JSON 저장
        out_dir = Path("results/compare")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"compare_{date.today().isoformat()}.json"
        with open(out_path, "w") as f:
            json.dump(
                [{
                    "symbol": r["symbol"],
                    "best_params": r["best_params"],
                    "summary": r["summary"],
                } for r in results],
                f, indent=2, ensure_ascii=False,
                default=lambda x: str(x) if isinstance(x, float) and x == float("inf") else x,
            )
        logger.info(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
