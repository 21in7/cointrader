"""델타중립 펀딩 캐리 백테스트 (구조적 엣지 리서치, 데이터 전용).

설계: docs/plans/2026-05-17-funding-carry-design.md

방향 예측 없음. perp + 반대 spot 델타중립으로 펀딩 현금흐름만 수확.
가격 leg는 완전 헤지 가정(basis PnL=0) — *상한* 추정. 이상화조차 net
음수면 실제는 확실히 죽음 → 싸게 폐기(edge-first).

변형:
  A 정적(always-short-perp + long spot): 수확 Σ(signed funding), 턴오버 최소
  B 부호추종: 수확 Σ|funding|, 부호전환 시 양 leg 반전 비용

비용: 선물 fee(taker 0.04% / maker 0.018%) + 슬리피지 0.01% + spot 차입비
스트레스 {0,5,10}%/yr. 결과는 stdout 구조화 출력 — PASS/FAIL 판정은
사람이 사전 폐기기준에 대조(스크립트가 판정하지 않음).

사용법:
  python scripts/funding_carry_backtest.py
  python scripts/funding_carry_backtest.py --symbols XRPUSDT,SOLUSDT --notional 1000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd

_SETTLE_PER_YEAR = 3 * 365  # 8h 정산 → 1095/yr
_DEFAULT_SYMBOLS = ["XRPUSDT", "SOLUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT"]


def _parse_args():
    p = argparse.ArgumentParser(description="델타중립 펀딩 캐리 백테스트")
    p.add_argument("--symbols", default=",".join(_DEFAULT_SYMBOLS))
    p.add_argument("--notional", type=float, default=1000.0)
    p.add_argument("--taker-bps", type=float, default=4.0, help="taker fee bps/leg")
    p.add_argument("--maker-bps", type=float, default=1.8, help="maker fee bps/leg")
    p.add_argument("--slip-bps", type=float, default=1.0, help="슬리피지 bps/fill")
    p.add_argument("--borrow-scenarios", default="0,5,10", help="spot 차입비 %/yr")
    p.add_argument("--regime-window", type=int, default=90, help="추세 레짐 윈도우(일)")
    p.add_argument("--regime-band", type=float, default=10.0, help="상승/하락 판정 %")
    return p.parse_args()


def _load_settlements(symbol: str) -> pd.DataFrame | None:
    path = Path(f"data/{symbol.lower()}/combined_15m.parquet")
    if not path.exists():
        print(f"  [skip] {symbol}: parquet 없음")
        return None
    df = pd.read_parquet(path, columns=["close", "funding_rate"])
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df = df.sort_index()
    s = df[(df.index.hour.isin([0, 8, 16])) & (df.index.minute == 0)].copy()
    s = s[s["funding_rate"].notna()]
    return s


def _regime(close: pd.Series, window_days: int, band_pct: float) -> pd.Series:
    """정산 시점별 추세 레짐: 직전 window_days 가격 변화율로 분류."""
    # 정산 간격 8h → window_days*3 정산 = 룩백
    n = max(window_days * 3, 1)
    chg = close.pct_change(n) * 100
    out = pd.Series("chop", index=close.index)
    out[chg > band_pct] = "up"
    out[chg < -band_pct] = "down"
    return out


def _carry_metrics(pnl_per_settle: pd.Series, notional: float) -> dict:
    """정산별 PnL 시계열 → 연환산 캐리/Sharpe/MaxDD."""
    if len(pnl_per_settle) == 0:
        return {"net_yr_pct": 0.0, "sharpe": 0.0, "maxdd_pct": 0.0, "n": 0}
    ret = pnl_per_settle / notional  # 정산당 수익률
    years = len(pnl_per_settle) / _SETTLE_PER_YEAR
    total = pnl_per_settle.sum()
    net_yr_pct = (total / notional) / years * 100 if years > 0 else 0.0
    mu, sd = ret.mean(), ret.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(_SETTLE_PER_YEAR)) if sd > 0 else 0.0
    equity = notional + pnl_per_settle.cumsum()
    peak = equity.cummax()
    maxdd = ((equity - peak) / peak).min() * 100
    return {"net_yr_pct": net_yr_pct, "sharpe": float(sharpe),
            "maxdd_pct": float(maxdd), "n": int(len(pnl_per_settle))}


def _run_symbol(symbol: str, s: pd.DataFrame, args, fee_bps: float,
                borrow_yr_pct: float) -> dict | None:
    f = s["funding_rate"]
    notional = args.notional
    leg_cost = (fee_bps + args.slip_bps) / 1e4 * notional  # 1 leg 1회 비용

    years = len(f) / _SETTLE_PER_YEAR
    borrow_total = borrow_yr_pct / 100.0 * years * notional  # spot 차입 누적
    onetime = 2 * leg_cost * 2  # 진입+청산 × 2 leg (perp+spot)

    # 변형 A: always-short-perp → 수취 = +signed funding
    pnl_A = f * notional
    cost_A = onetime + borrow_total
    netA = pnl_A.sum() - cost_A
    mA = _carry_metrics(pnl_A - cost_A / len(f), notional)  # 비용 균등배분
    mA["net_yr_pct"] = (netA / notional) / years * 100 if years > 0 else 0.0

    # 변형 B: 부호추종 → 수취 |funding|, 부호전환 시 양 leg 반전
    side = np.sign(f).replace(0, np.nan).ffill().fillna(1.0)
    flips = int((side != side.shift()).sum())
    flip_cost = flips * (2 * 2 * leg_cost)  # 전환당 perp+spot 각 round-trip
    pnl_B = f.abs() * notional
    cost_B = onetime + borrow_total + flip_cost
    netB = pnl_B.sum() - cost_B
    mB = _carry_metrics(pnl_B - cost_B / len(f), notional)
    mB["net_yr_pct"] = (netB / notional) / years * 100 if years > 0 else 0.0

    return {
        "symbol": symbol, "n": len(f), "years": round(years, 2),
        "gross_signed_yr": (f.sum() / years * 100) if years else 0,
        "gross_abs_yr": (f.abs().sum() / years * 100) if years else 0,
        "flips": flips, "flips_per_yr": round(flips / years, 1) if years else 0,
        "A": mA, "B": mB,
        "pnl_A": pnl_A, "pnl_B": pnl_B,  # 포트폴리오 집계용
    }


def main():
    args = _parse_args()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    borrows = [float(x) for x in args.borrow_scenarios.split(",")]

    loaded = {}
    for sym in symbols:
        s = _load_settlements(sym)
        if s is not None and len(s) > 0:
            loaded[sym] = s
    if not loaded:
        print("[ERR] 데이터 없음", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*78}\n델타중립 펀딩 캐리 백테스트 — {len(loaded)}심볼\n{'='*78}")
    print(f"notional={args.notional} | taker={args.taker_bps}bps maker={args.maker_bps}bps "
          f"slip={args.slip_bps}bps | borrow scenarios={borrows}%/yr\n")

    for fee_label, fee_bps in [("TAKER", args.taker_bps), ("MAKER", args.maker_bps)]:
        for bw in borrows:
            print(f"\n----- 시나리오: {fee_label} fee + borrow {bw}%/yr -----")
            print(f"{'SYMBOL':9s} {'yrs':>4s} {'grossA%':>8s} {'grossB%':>8s} "
                  f"{'flips/yr':>8s} | {'netA%':>7s} {'shA':>5s} {'ddA%':>6s} | "
                  f"{'netB%':>7s} {'shB':>5s} {'ddB%':>6s}")
            agg_A, agg_B = [], []
            rows = []
            for sym, s in loaded.items():
                r = _run_symbol(sym, s, args, fee_bps, bw)
                rows.append(r)
                agg_A.append(r["pnl_A"]); agg_B.append(r["pnl_B"])
                print(f"{sym:9s} {r['years']:4.1f} {r['gross_signed_yr']:8.2f} "
                      f"{r['gross_abs_yr']:8.2f} {r['flips_per_yr']:8.1f} | "
                      f"{r['A']['net_yr_pct']:7.2f} {r['A']['sharpe']:5.2f} "
                      f"{r['A']['maxdd_pct']:6.1f} | {r['B']['net_yr_pct']:7.2f} "
                      f"{r['B']['sharpe']:5.2f} {r['B']['maxdd_pct']:6.1f}")
            # 동일가중 포트폴리오 (정산 타임스탬프 union, 가용 심볼 평균)
            pA = pd.concat(agg_A, axis=1).mean(axis=1).dropna()
            pB = pd.concat(agg_B, axis=1).mean(axis=1).dropna()
            yrsP = len(pA) / _SETTLE_PER_YEAR
            # 포트폴리오 비용: 심볼 평균이므로 1심볼분 비용 근사 차감
            lc = (fee_bps + args.slip_bps) / 1e4 * args.notional
            ot = 2 * lc * 2
            bt = bw / 100 * yrsP * args.notional
            mPA = _carry_metrics(pA - (ot + bt) / max(len(pA), 1), args.notional)
            mPA["net_yr_pct"] = ((pA.sum() - ot - bt) / args.notional / yrsP * 100) if yrsP else 0
            # B 포트폴리오 flip 비용은 심볼별 평균 반영(근사: 평균 flips)
            avg_flip_cost = np.mean([r["flips"] for r in rows]) * (2 * 2 * lc)
            mPB = _carry_metrics(pB - (ot + bt + avg_flip_cost) / max(len(pB), 1), args.notional)
            mPB["net_yr_pct"] = ((pB.sum() - ot - bt - avg_flip_cost) / args.notional / yrsP * 100) if yrsP else 0
            print(f"{'PORT(eq)':9s} {yrsP:4.1f} {'':8s} {'':8s} {'':8s} | "
                  f"{mPA['net_yr_pct']:7.2f} {mPA['sharpe']:5.2f} {mPA['maxdd_pct']:6.1f} | "
                  f"{mPB['net_yr_pct']:7.2f} {mPB['sharpe']:5.2f} {mPB['maxdd_pct']:6.1f}")

    # 레짐 분해 (베이스라인: taker + 5%/yr, 변형 A signed)
    print(f"\n----- 레짐 분해 (변형 A, gross signed, 비용 전) -----")
    print(f"{'SYMBOL':9s} {'up%':>8s} {'down%':>8s} {'chop%':>8s}  (연환산 gross signed)")
    for sym, s in loaded.items():
        reg = _regime(s["close"], args.regime_window, args.regime_band)
        f = s["funding_rate"]
        line = f"{sym:9s}"
        for rg in ["up", "down", "chop"]:
            m = reg == rg
            if m.sum() > 0:
                yr = f[m].sum() / (m.sum() / _SETTLE_PER_YEAR) * 100
                line += f" {yr:8.2f}"
            else:
                line += f" {'--':>8s}"
        print(line)
    print(f"\n{'='*78}\n해석은 사전 폐기기준(design §5)에 대조 — 스크립트는 판정하지 않음\n{'='*78}")


if __name__ == "__main__":
    main()
