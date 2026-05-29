"""
Spot-perp 베이시스 / 펀딩 캐리 적합성 precheck (BTC/ETH). 백테스트 아님.

성격: 엣지의 '존재'는 펀딩 메커니즘상 거의 구조적 → '죽이기'보다 '특성화'.
가장 싸고 결정적인 킬 = 순 캐리가 비용 차감 후 ≤0 (게이트 1).
포지션 구조: long spot + short perp, 델타중립, 1x (funding>0이면 숏 perp가 수취).

PASS/FAIL 임계값은 데이터 보기 전 확정(상단 상수). 사후 변경 금지.
이전 funding-carry 연구(2026-05-17, 6알트, FAIL: net -0.37%@5%borrow)는 BTC/ETH
미포함 + 가격leg 이상화. 본 precheck은 BTC/ETH 실측 basis·risk까지 특성화.

데이터: src/carry/data.py 가 생성한 data/{sym}/carry_15m.parquet
재사용: src.statarb.precheck(검정), src.backtester(비용)

실행:  python -m src.carry.precheck
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.carry import setup_korean_font  # noqa: E402
from src.statarb.precheck import adf, engle_granger, half_life, johansen, ols_beta_alpha  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) PASS/FAIL — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
MIN_NET_CARRY_APR = 0.10        # 순 연율 캐리 > 10% (수수료·슬리피지 차감 후, borrow 제외)
MIN_FUNDING_POS_RATIO = 0.60    # 펀딩 양(+) 비율 ≥ 60%
BASIS_EG_P_MAX = 0.01           # 베이시스 EG 공적분 p < 0.01
BASIS_HALFLIFE_MAX_DAYS = 3.0   # 베이시스 half-life < 3일
ROLLING_POS_RATIO_MIN = 0.60    # rolling 연율 캐리 > 0 인 윈도우 비율 ≥ 60%

# ----- 자유 파라미터 (잠금) -----
ROLLING_WINDOW_DAYS = 90
IS_FRACTION = 0.70
MAINT_MARGIN_RATE = 0.005       # 청산 임계 근사 (perp 유지증거금률)

# ----- 비용 모델 (src.backtester 와 동일 상수) -----
FEE_PCT_PER_SIDE = 0.04         # taker %
SLIPPAGE_PCT_PER_SIDE = 0.01    # slip % per fill
N_FILLS_ROUNDTRIP = 4           # 2레그(spot·perp) × (진입+청산)
BORROW_SCENARIOS = [0.0, 5.0, 10.0]   # ★참고용(게이트 아님) — 이전 연구 대조

UNIVERSE = ["BTCUSDT", "ETHUSDT"]
OUTDIR = ROOT / "results" / "carry"
YEAR_DAYS = 365.25


def _settlements(df: pd.DataFrame) -> pd.Series:
    """ffill된 funding_rate에서 정산점만 추출(값 변화 지점). index=정산시각."""
    fr = df["funding_rate"].dropna()
    return fr[fr.ne(fr.shift())]


def _runs(mask: np.ndarray) -> list[int]:
    """불리언 시퀀스에서 True 연속 구간 길이들."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _onetime_cost_frac() -> float:
    return N_FILLS_ROUNDTRIP * (FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE) / 100.0


def run_symbol(symbol: str) -> dict:
    path = ROOT / "data" / symbol.lower() / "carry_15m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 python -m src.carry.data")
    df = pd.read_parquet(path)
    df = df[df["funding_rate"].notna()].copy()  # 선두 NaN trim

    bar_min = pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds() / 60
    settle = _settlements(df)
    cadence_h = pd.Series(settle.index).diff().dropna().median().total_seconds() / 3600
    settles_per_yr = YEAR_DAYS * 24 / cadence_h
    years = (settle.index[-1] - settle.index[0]).total_seconds() / (YEAR_DAYS * 24 * 3600)

    # ---- 게이트 1: 펀딩 경제성 ----
    gross_sum = float(settle.sum())                       # 변형 A: Σ signed funding
    gross_apr = gross_sum / years                         # 연율 (fraction)
    net_apr = (gross_sum - _onetime_cost_frac()) / years  # 일회성 비용 amortize
    pos_ratio = float((settle > 0).mean())
    g1 = (net_apr > MIN_NET_CARRY_APR) and (pos_ratio >= MIN_FUNDING_POS_RATIO)

    # ---- 게이트 2: 베이시스 공적분 ----
    lp, ls = np.log(df["perp_close"].values), np.log(df["spot_close"].values)
    eg = engle_granger(lp, ls)
    joh = johansen(np.column_stack([lp, ls]))
    basis_log = pd.Series(lp - ls, index=df.index)
    b_adf = adf(basis_log.values)
    hl = half_life(basis_log.values)
    hl_days = (hl["hl_bars"] * bar_min / (60 * 24)) if hl["valid"] else float("inf")
    g2 = (eg["p"] < BASIS_EG_P_MAX) and hl["valid"] and (hl_days < BASIS_HALFLIFE_MAX_DAYS)

    # ---- 게이트 3: 리스크 (안전 레버리지) ----
    cum = settle.cumsum()
    max_dd = float((cum.cummax() - cum).max())            # 음펀딩 누적 드로다운(fraction)
    neg_runs = _runs((settle < 0).values)
    worst_neg_run = max(neg_runs) if neg_runs else 0
    basis_rel = df["basis_rel"]
    # short-perp+long-spot 적: basis 상승(perp가 spot대비 상승). 표본 내 최대 역이격 swing.
    max_adverse = float(basis_rel.max() - basis_rel.min())
    safe_lev = (1.0 - MAINT_MARGIN_RATE) / max_adverse if max_adverse > 0 else float("inf")
    g3 = safe_lev > 1.0                                    # 1x 풀마진 미청산

    # ---- 게이트 4: 안정성/레짐 (rolling) ----
    daily = settle.resample("1D").sum()
    win = ROLLING_WINDOW_DAYS
    roll_apr = daily.rolling(f"{win}D").sum() * (YEAR_DAYS / win)
    roll_apr = roll_apr.dropna()
    roll_apr = roll_apr.iloc[win:] if len(roll_apr) > win else roll_apr  # 워밍업 제거
    roll_pos_ratio = float((roll_apr > 0).mean())
    g4 = roll_pos_ratio >= ROLLING_POS_RATIO_MIN

    # ---- [추가 진단 1] 조건부 캐리 (funding>0 구간만) ----
    pos = settle[settle > 0]
    cond_carry_apr = float(pos.mean() * settles_per_yr) if len(pos) else 0.0

    # ---- [추가 진단 2] 양(+)펀딩 레짐 지속시간 분포 ----
    pos_runs = _runs((settle > 0).values)
    buckets = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16-31": 0, "32+": 0}
    for r in pos_runs:
        if r == 1:
            buckets["1"] += 1
        elif r <= 3:
            buckets["2-3"] += 1
        elif r <= 7:
            buckets["4-7"] += 1
        elif r <= 15:
            buckets["8-15"] += 1
        elif r <= 31:
            buckets["16-31"] += 1
        else:
            buckets["32+"] += 1
    run_median = float(np.median(pos_runs)) if pos_runs else 0.0
    run_max = max(pos_runs) if pos_runs else 0

    # ---- borrow 민감도 (참고용, 게이트 아님) ----
    borrow_rows = {f"{b:.0f}%": net_apr - b / 100.0 for b in BORROW_SCENARIOS}

    passed = g1 and g2 and g3 and g4
    checklist = [
        (f"펀딩 경제성 (net>{MIN_NET_CARRY_APR:.0%} & 양≥{MIN_FUNDING_POS_RATIO:.0%})", g1),
        (f"베이시스 공적분 (EG p<{BASIS_EG_P_MAX} & HL<{BASIS_HALFLIFE_MAX_DAYS:.0f}일)", g2),
        ("리스크 (1x 풀마진 미청산)", g3),
        (f"레짐 안정성 (rolling +비중 ≥{ROLLING_POS_RATIO_MIN:.0%})", g4),
    ]

    # ---- 콘솔 리포트 ----
    print("\n" + "=" * 78)
    print(f"  {symbol}  spot-perp 캐리 precheck")
    print("=" * 78)
    print(f"  {len(df):,} bars, {years:.2f}년, funding 주기 {cadence_h:.0f}h ({settles_per_yr:.0f}/yr)")
    print(f"\n[1] 펀딩 경제성")
    print(f"  gross 연율 캐리 = {gross_apr*100:+.2f}%   net(일회성비용 후) = {net_apr*100:+.2f}%   "
          f"(필요 >{MIN_NET_CARRY_APR*100:.0f}%)")
    print(f"  펀딩 양(+)비율 = {pos_ratio*100:.1f}%  (필요 ≥{MIN_FUNDING_POS_RATIO*100:.0f}%)")
    print(f"  → {'PASS' if g1 else 'FAIL'}")
    print(f"\n[2] 베이시스 공적분 (같은 자산 → 강하게 통과 기대, β≈1)")
    print(f"  EG p={eg['p']:.2e}  Johansen r=0기각={joh['reject_r0']}  β(OLS)={ols_beta_alpha(lp, ls)[0]:.4f}")
    print(f"  basis ADF p={b_adf['p']:.2e}  half-life={hl_days*24:.1f}h (={hl_days:.2f}일, 필요<{BASIS_HALFLIFE_MAX_DAYS:.0f}일)")
    print(f"  → {'PASS' if g2 else 'FAIL'}")
    print(f"\n[3] 리스크 / 안전 레버리지")
    print(f"  음펀딩 누적 드로다운 = {max_dd*100:.3f}% (notional), 최장 음구간 {worst_neg_run}정산({worst_neg_run*cadence_h/24:.1f}일)")
    print(f"  표본 내 최대 역이격(basis swing) = {max_adverse*1e4:.1f}bps → 안전 최대 레버리지 ≈ {safe_lev:.0f}x")
    print(f"  → 1x 미청산 {'PASS' if g3 else 'FAIL'}")
    print(f"\n[4] 레짐 안정성 (rolling {win}일 연율 캐리)")
    print(f"  rolling 캐리 >0 윈도우 비율 = {roll_pos_ratio*100:.1f}%  (필요 ≥{ROLLING_POS_RATIO_MIN*100:.0f}%)")
    print(f"  rolling 연율 캐리: 중앙 {roll_apr.median()*100:+.2f}%  [min {roll_apr.min()*100:+.1f}%, max {roll_apr.max()*100:+.1f}%]")
    print(f"  → {'PASS' if g4 else 'FAIL'}")

    print(f"\n[진단 +1] 조건부 캐리 (funding>0 구간만의 평균 연율) = {cond_carry_apr*100:+.2f}%/yr")
    print(f"           (양펀딩 강도; 전체 net {net_apr*100:+.2f}% 와 비교 → 음구간 drag 크기)")
    print(f"[진단 +2] 양(+)펀딩 레짐 지속시간 분포 (정산 횟수 기준; 1정산={cadence_h:.0f}h)")
    print(f"           runs={len(pos_runs)}개, 중앙 {run_median:.0f}정산({run_median*cadence_h:.0f}h), 최장 {run_max}정산({run_max*cadence_h/24:.1f}일)")
    print("           " + "  ".join(f"{k}:{v}" for k, v in buckets.items()))

    print(f"\n[참고] borrow(spot 자금조달) 민감도 — 게이트 아님, 이전 6알트 연구 대조용")
    print("           " + "  ".join(f"borrow {k}→net {v*100:+.2f}%/yr" for k, v in borrow_rows.items()))

    print(f"\n  체크리스트:")
    for name, p in checklist:
        print(f"   {'✓' if p else '✗'}  {name}")
    print(f"  최종: {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "symbol": symbol, "years": years, "cadence_hours": cadence_h,
        "gross_apr": gross_apr, "net_apr": net_apr, "pos_ratio": pos_ratio,
        "basis_eg_p": eg["p"], "basis_adf_p": b_adf["p"], "basis_halflife_days": hl_days,
        "basis_beta": ols_beta_alpha(lp, ls)[0], "johansen_reject_r0": joh["reject_r0"],
        "max_dd_pct": max_dd * 100, "worst_neg_run": worst_neg_run,
        "max_adverse_basis_bps": max_adverse * 1e4, "safe_leverage": safe_lev,
        "rolling_pos_ratio": roll_pos_ratio, "rolling_median_apr": float(roll_apr.median()),
        "cond_carry_apr": cond_carry_apr,
        "pos_run_count": len(pos_runs), "pos_run_median": run_median, "pos_run_max": run_max,
        "pos_run_buckets": buckets,
        "borrow_sensitivity": borrow_rows,
        "gates": {name: bool(p) for name, p in checklist},
        "PASS": bool(passed),
        "_series": {"settle": settle, "basis_rel": basis_rel, "roll_apr": roll_apr},
    }


def run_precheck(symbols=UNIVERSE) -> dict:
    setup_korean_font()
    print("=" * 78)
    print("  Spot-perp 펀딩 캐리 precheck — BTC/ETH 특성화 (백테스트 아님)")
    print("=" * 78)
    results = {s: run_symbol(s) for s in symbols}

    # ---- 랭킹 (순 캐리 내림차순) ----
    rank = sorted(results.values(), key=lambda r: r["net_apr"], reverse=True)
    print("\n" + "=" * 78)
    print("  코인 랭킹 (net 연율 캐리 내림차순)")
    print("=" * 78)
    print(f"  {'coin':6s} {'net%':>7s} {'gross%':>7s} {'cond%':>7s} {'양비율':>6s} "
          f"{'basisHL':>8s} {'safeLev':>8s} {'roll+%':>7s} {'PASS':>5s}")
    for r in rank:
        print(f"  {r['symbol'][:-4]:6s} {r['net_apr']*100:>+7.2f} {r['gross_apr']*100:>+7.2f} "
              f"{r['cond_carry_apr']*100:>+7.2f} {r['pos_ratio']*100:>5.0f}% "
              f"{r['basis_halflife_days']*24:>6.1f}h {r['safe_leverage']:>7.0f}x "
              f"{r['rolling_pos_ratio']*100:>6.0f}% {'PASS' if r['PASS'] else 'FAIL':>5s}")

    _plots(results)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dump = {s: {k: v for k, v in r.items() if k != "_series"} for s, r in results.items()}
    dump["_meta"] = {"generated": today, "universe": symbols,
                     "note": "carry는 long-spot/short-perp 1x 델타중립. borrow는 참고용."}
    (OUTDIR / f"carry_precheck_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/carry/carry_precheck_{today}.json + 플롯 3")
    return results


def _plots(results: dict):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # (a) 누적 펀딩 수익 곡선
    fig, ax = plt.subplots(figsize=(12, 5))
    for s, r in results.items():
        c = r["_series"]["settle"].cumsum() * 100
        ax.plot(c.index, c.values, lw=1.0, label=f"{s[:-4]} 누적펀딩")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("(a) 누적 펀딩 수익 (% of notional, 변형A long-spot/short-perp)")
    ax.legend()
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"carry_cumfunding_{today}.png", dpi=110)
    plt.close(fig)

    # (b) 베이시스 + 평균 ± 2σ
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), squeeze=False)
    for ax, (s, r) in zip(axes[:, 0], results.items()):
        b = r["_series"]["basis_rel"] * 1e4
        m, sd = b.mean(), b.std()
        ax.plot(b.index, b.values, lw=0.5, color="steelblue")
        ax.axhline(m, color="black", lw=0.8)
        ax.axhline(m + 2 * sd, color="firebrick", ls="--", lw=0.8)
        ax.axhline(m - 2 * sd, color="firebrick", ls="--", lw=0.8)
        ax.set_title(f"(b) {s[:-4]} 베이시스(perp−spot, bps) + 평균 ±2σ")
        ax.set_ylabel("bps")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"carry_basis_{today}.png", dpi=110)
    plt.close(fig)

    # (c) rolling 연율 캐리
    fig, ax = plt.subplots(figsize=(12, 5))
    for s, r in results.items():
        ra = r["_series"]["roll_apr"] * 100
        ax.plot(ra.index, ra.values, lw=1.0, label=f"{s[:-4]}")
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(MIN_NET_CARRY_APR * 100, color="firebrick", ls="--", lw=0.9,
               label=f"{MIN_NET_CARRY_APR*100:.0f}% 바")
    ax.set_title(f"(c) rolling {ROLLING_WINDOW_DAYS}일 연율 캐리 (%)")
    ax.legend()
    ax.set_ylabel("%/yr")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"carry_rolling_{today}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run_precheck()
