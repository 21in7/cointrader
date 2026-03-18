#!/usr/bin/env python3
"""
포지션 사이징 분석: Robust Monte Carlo 방식.

핵심: 백테스트 31건의 승률/손익비를 고정값으로 믿지 않고,
불확실성 범위(승률 30~45%, 손익비 3.0~5.0)를 넣어
worst-case 조합에서도 파산하지 않는 리스크 비중을 산출한다.

사용법:
  python scripts/position_sizing_analysis.py --symbol SOLUSDT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from loguru import logger

from src.backtester import WalkForwardBacktester, WalkForwardConfig


def run_backtest(symbol: str, params: dict) -> dict:
    cfg = WalkForwardConfig(
        symbols=[symbol],
        use_ml=False,
        train_months=3,
        test_months=1,
        **params,
    )
    wf = WalkForwardBacktester(cfg)
    return wf.run()


def extract_r_multiples(trades: list[dict]) -> np.ndarray:
    """각 트레이드의 R-multiple을 추출 (1R = SL 히트 시 손실)."""
    r_multiples = []
    for t in trades:
        sl_distance = abs(t["entry_price"] - t["sl"])
        sl_loss = sl_distance * t["quantity"]
        if sl_loss <= 0:
            continue
        r_multiples.append(t["net_pnl"] / sl_loss)
    return np.array(r_multiples)


def kelly_criterion(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    """Kelly: f* = (W * avg_win_R - (1-W) * |avg_loss_R|) / avg_win_R"""
    if avg_win_r <= 0:
        return 0.0
    expectancy = win_rate * avg_win_r - (1 - win_rate) * abs(avg_loss_r)
    if expectancy <= 0:
        return 0.0
    return expectancy / avg_win_r


def consecutive_loss_survival(risk_pct: float, n: int) -> float:
    """n연패 후 잔고 비율(%)."""
    return (1 - risk_pct) ** n * 100


def robust_monte_carlo(
    risk_pct: float,
    win_rate_range: tuple[float, float],
    payoff_range: tuple[float, float],
    loss_r_range: tuple[float, float],
    n_simulations: int = 10000,
    n_trades: int = 200,
    initial_balance: float = 1000.0,
    ruin_threshold: float = 0.20,
) -> dict:
    """Robust Monte Carlo: 매 시뮬레이션마다 승률/손익비를 범위 내에서 샘플링.

    각 시뮬레이션:
      1) 승률을 win_rate_range에서 uniform 추출
      2) 승리 R-multiple을 payoff_range에서 uniform 추출
      3) 패배 R-multiple을 loss_r_range에서 uniform 추출
      4) 해당 파라미터로 n_trades건을 생성하여 에퀴티 시뮬레이션
    """
    rng = np.random.default_rng(42)
    final_balances = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0

    for sim in range(n_simulations):
        # 파라미터 샘플링
        wr = rng.uniform(*win_rate_range)
        win_r = rng.uniform(*payoff_range)
        loss_r = rng.uniform(*loss_r_range)

        # 트레이드 생성
        outcomes = rng.random(n_trades)
        r_multiples = np.where(outcomes < wr, win_r, loss_r)

        # 에퀴티 시뮬레이션
        balance = initial_balance
        peak = balance
        max_dd = 0.0
        ruined = False

        for r in r_multiples:
            pnl = balance * risk_pct * r
            balance += pnl

            if balance <= initial_balance * ruin_threshold:
                ruined = True
                break

            peak = max(peak, balance)
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)

        if ruined:
            ruin_count += 1
            final_balances[sim] = 0
            max_drawdowns[sim] = 1.0
        else:
            final_balances[sim] = balance
            max_drawdowns[sim] = max_dd

    return {
        "risk_pct": risk_pct,
        "ruin_probability": round(ruin_count / n_simulations * 100, 2),
        "median_return": round((np.median(final_balances) - initial_balance) / initial_balance * 100, 1),
        "p5_return": round((np.percentile(final_balances, 5) - initial_balance) / initial_balance * 100, 1),
        "p25_return": round((np.percentile(final_balances, 25) - initial_balance) / initial_balance * 100, 1),
        "p75_return": round((np.percentile(final_balances, 75) - initial_balance) / initial_balance * 100, 1),
        "p95_return": round((np.percentile(final_balances, 95) - initial_balance) / initial_balance * 100, 1),
        "median_max_dd": round(np.median(max_drawdowns) * 100, 1),
        "p95_max_dd": round(np.percentile(max_drawdowns, 95) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="포지션 사이징 분석 (Robust Monte Carlo)")
    parser.add_argument("--symbol", required=True, type=str)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--tp-mult", type=float, default=4.0)
    parser.add_argument("--signal-threshold", type=int, default=3)
    parser.add_argument("--adx", type=float, default=20)
    parser.add_argument("--vol-mult", type=float, default=2.5)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    params = {
        "atr_sl_mult": args.sl_mult,
        "atr_tp_mult": args.tp_mult,
        "signal_threshold": args.signal_threshold,
        "adx_threshold": args.adx,
        "volume_multiplier": args.vol_mult,
    }

    # 1) 백테스트로 기준값 추출
    logger.info(f"[{symbol}] 백테스트 실행")
    result = run_backtest(symbol, params)
    trades = result.get("trades", [])
    summary = result["summary"]

    if len(trades) < 5:
        logger.error(f"트레이드 {len(trades)}건 — 분석 불가")
        return

    r_multiples = extract_r_multiples(trades)
    wins = r_multiples[r_multiples > 0]
    losses = r_multiples[r_multiples <= 0]
    obs_wr = len(wins) / len(r_multiples)
    obs_win_r = float(np.mean(wins)) if len(wins) > 0 else 0
    obs_loss_r = float(np.mean(losses)) if len(losses) > 0 else 0
    obs_expectancy = obs_wr * obs_win_r + (1 - obs_wr) * obs_loss_r
    obs_kelly = kelly_criterion(obs_wr, obs_win_r, abs(obs_loss_r))

    # 불확실성 범위 설정 (관측값 기준 ±보정)
    # 승률: 관측 38.7% → 30~45% (하방으로 더 넓게)
    wr_lo = max(0.25, obs_wr - 0.10)
    wr_hi = min(0.55, obs_wr + 0.07)
    # 승리 R: 관측 3.85R → 3.0~5.0
    win_r_lo = max(2.0, obs_win_r - 1.0)
    win_r_hi = obs_win_r + 1.2
    # 패배 R: 관측 -1.18R → -1.5 ~ -0.9
    loss_r_lo = min(-0.8, obs_loss_r + 0.3)
    loss_r_hi = obs_loss_r - 0.3

    print(f"\n{'=' * 85}")
    print(f"  포지션 사이징 분석: {symbol} (Robust Monte Carlo)")
    print(f"  파라미터: SL={args.sl_mult}x ATR, TP={args.tp_mult}x ATR, ADX={args.adx}")
    print(f"{'=' * 85}")

    # 관측값
    print(f"\n[백테스트 관측값] ({len(trades)}건)")
    print(f"  승률: {obs_wr*100:.1f}% | 승리 평균: +{obs_win_r:.2f}R | 패배 평균: {obs_loss_r:.2f}R")
    print(f"  기대값: {obs_expectancy:.2f}R | Kelly: {obs_kelly*100:.1f}%")
    print(f"  최대 연속 손실: {summary.get('max_consecutive_losses', 'N/A')}회")

    # R-multiple 분포 (간략)
    print(f"\n  R 분포: 패배 {obs_loss_r:.2f}R (SL+수수료) | 승리 +{obs_win_r:.2f}R (TP-수수료)")
    print(f"  → 거의 바이너리: SL 아니면 TP, 중간 청산 없음")

    # 불확실성 범위
    print(f"\n[불확실성 범위] (실전 괴리 반영)")
    print(f"  승률:    {wr_lo*100:.0f}% ~ {wr_hi*100:.0f}%  (관측: {obs_wr*100:.1f}%)")
    print(f"  승리 R:  +{win_r_lo:.1f}R ~ +{win_r_hi:.1f}R  (관측: +{obs_win_r:.2f}R)")
    print(f"  패배 R:  {loss_r_hi:.1f}R ~ {loss_r_lo:.1f}R  (관측: {obs_loss_r:.2f}R)")

    # Worst-case Kelly
    worst_kelly = kelly_criterion(wr_lo, win_r_lo, abs(loss_r_hi))
    best_kelly = kelly_criterion(wr_hi, win_r_hi, abs(loss_r_lo))
    print(f"\n  Worst-case Kelly: {worst_kelly*100:.1f}% | Best-case Kelly: {best_kelly*100:.1f}%")

    # 연속 손실 생존 테이블
    print(f"\n[연속 손실 생존 테이블]")
    print(f"  {'리스크%':>8}  {'4연패':>7}  {'6연패':>7}  {'8연패':>7}  {'10연패':>7}  {'12연패':>7}")
    print(f"  {'-' * 50}")
    for rp in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        cols = [f"{consecutive_loss_survival(rp, n):.1f}%" for n in [4, 6, 8, 10, 12]]
        print(f"  {rp*100:>7.1f}%  {'  '.join(f'{c:>7}' for c in cols)}")

    # Robust Monte Carlo
    risk_levels = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07]

    print(f"\n[Robust Monte Carlo (10,000회 × 200건, 파라미터 매회 랜덤 샘플링)]")
    print(f"  파산 기준: 잔고 ≤ 20%")
    print(f"  {'리스크%':>8} {'파산%':>6} {'하위5%':>9} {'하위25%':>9} {'중위':>9} "
          f"{'상위75%':>9} {'상위95%':>10} {'중위MDD':>7} {'95%MDD':>7}")
    print(f"  {'-' * 85}")

    best_risk = None
    best_score = -999
    mc_results = []

    for rp in risk_levels:
        mc = robust_monte_carlo(
            risk_pct=rp,
            win_rate_range=(wr_lo, wr_hi),
            payoff_range=(win_r_lo, win_r_hi),
            loss_r_range=(loss_r_hi, loss_r_lo),  # hi is more negative
        )
        mc_results.append(mc)

        # 숫자 포맷
        def fmt_ret(v):
            if abs(v) >= 10000:
                return f"{v/1000:>+7.0f}k%"
            return f"{v:>+8.1f}%"

        print(f"  {rp*100:>7.1f}% {mc['ruin_probability']:>5.1f}% "
              f"{fmt_ret(mc['p5_return'])} {fmt_ret(mc['p25_return'])} "
              f"{fmt_ret(mc['median_return'])} {fmt_ret(mc['p75_return'])} "
              f"{fmt_ret(mc['p95_return']):>10} {mc['median_max_dd']:>6.1f}% "
              f"{mc['p95_max_dd']:>6.1f}%")

        # 선정 기준: 파산 <1% AND 95%MDD ≤ 30% 에서 중위 수익 최대
        if (mc["ruin_probability"] <= 1.0
            and mc["p95_max_dd"] <= 30.0
            and mc["median_return"] > best_score):
            best_score = mc["median_return"]
            best_risk = rp

    # Worst-case 전용 MC (승률 30%, 손익비 3.0 고정)
    print(f"\n[Worst-Case 시나리오 (승률={wr_lo*100:.0f}%, 승리R=+{win_r_lo:.1f}, 패배R={loss_r_hi:.1f})]")
    print(f"  {'리스크%':>8} {'파산%':>6} {'중위수익':>9} {'95%MDD':>7}")
    print(f"  {'-' * 35}")

    worst_best_risk = None
    worst_best_score = -999

    for rp in risk_levels:
        mc = robust_monte_carlo(
            risk_pct=rp,
            win_rate_range=(wr_lo, wr_lo + 0.001),  # 거의 고정
            payoff_range=(win_r_lo, win_r_lo + 0.001),
            loss_r_range=(loss_r_hi, loss_r_hi + 0.001),
        )

        def fmt_ret(v):
            if abs(v) >= 10000:
                return f"{v/1000:>+7.0f}k%"
            return f"{v:>+8.1f}%"

        print(f"  {rp*100:>7.1f}% {mc['ruin_probability']:>5.1f}% "
              f"{fmt_ret(mc['median_return'])} {mc['p95_max_dd']:>6.1f}%")

        if (mc["ruin_probability"] <= 1.0
            and mc["p95_max_dd"] <= 30.0
            and mc["median_return"] > worst_best_score):
            worst_best_score = mc["median_return"]
            worst_best_risk = rp

    # 최종 권장
    print(f"\n{'=' * 85}")
    print(f"  최종 권장")
    print(f"{'=' * 85}")

    # 가장 보수적인 값: worst-case MC 최적과 robust MC 최적 중 작은 값
    candidates = [r for r in [best_risk, worst_best_risk, worst_kelly / 2] if r and r > 0]
    recommended = min(candidates) if candidates else 0.01
    recommended = max(0.005, min(recommended, 0.05))

    print(f"  Robust MC 최적 (파산<1%, 95%MDD≤30%): {best_risk*100:.1f}%" if best_risk else "  Robust MC: 조건 충족 없음")
    print(f"  Worst-Case MC 최적:                   {worst_best_risk*100:.1f}%" if worst_best_risk else "  Worst-Case MC: 조건 충족 없음")
    print(f"  Worst-Case Half Kelly:                {worst_kelly/2*100:.1f}%")

    print(f"\n  >>> 실전 권장: 1회 리스크 = 계좌의 {recommended*100:.1f}%")
    print(f"      근거: worst-case에서도 파산하지 않는 가장 보수적 기준")
    survival_6 = consecutive_loss_survival(recommended, 6)
    survival_10 = consecutive_loss_survival(recommended, 10)
    print(f"      6연패 후 잔고: {survival_6:.1f}% | 10연패 후: {survival_10:.1f}%")

    print(f"\n  [.env 설정 가이드]")
    print(f"      ATR_SL_MULT_SOLUSDT={args.sl_mult}")
    print(f"      ATR_TP_MULT_SOLUSDT={args.tp_mult}")
    print(f"      ADX_THRESHOLD_SOLUSDT={args.adx}")
    print(f"      SIGNAL_THRESHOLD_SOLUSDT={args.signal_threshold}")
    for atr_pct in [0.01, 0.012, 0.015]:
        margin_ratio = recommended / (10 * atr_pct)
        margin_ratio = min(margin_ratio, 0.50)
        print(f"      ATR≈{atr_pct*100:.1f}% → MARGIN_MAX_RATIO_SOLUSDT ≈ {margin_ratio:.2f}")
    print()


if __name__ == "__main__":
    main()
