"""온체인 거래소 netflow directional precheck. 백테스트 아님.

질문: 거래소 일일 netflow(FlowIn−FlowOut)가 다음날 BTC/ETH 방향성 수익률을 비용 넘게
예측해 거래 가능한가? 산출물 = (자산/변형)별 PASS/FAIL.

성격: falsification-first. 알파 추구 10전10패는 전부 거래소 *내부* 신호였고, 온체인
플로우는 정보 차원이 다른 유일한 미탐색 데이터. 예상은 FAIL(플로우도 차익소거 진행)이나
가격과 다른 차원이라 잔존 edge 가능성 0은 아님. 가장 싸고 결정적인 킬 = 경제성 게이트 먼저.

신호(사전등록, 잠금): netflow z-score(90d) → pos = −sign(z), 발행지연 LAG=1.
  유입(z>0)=매도압력→숏, 유출=축적→롱. always-in ±1.
정렬: ret_t = log(price_t/price_{t-1}). 포지션은 close(t−1)에 확정되며 그 신호는
  최소 LAG일 전(z_{t−1−LAG})만 사용 → pos.shift(1+LAG)로 lookahead·flash개정 차단.

게이트(사전등록): 1)경제성 2)통계(boot+BH) 3)진위/레짐(top1%제거+연도) 4)안정성(IS/OOS)
  + 대칭성(롱/숏 레그) + 포트폴리오 확증(EW). PASS = EW + (BTC or ETH) 전 게이트 통과.

재사용: src.backtester(비용), src.statarb.scan(BH), src.carry(폰트).
데이터: src/onchain/data.py 가 받은 data/{btc,eth}usdt/onchain_daily.parquet.

실행:  python -m src.onchain.precheck
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

from src.backtester import _apply_slippage, _calc_fee  # noqa: E402
from src.carry import setup_korean_font  # noqa: E402
from src.statarb.scan import _benjamini_hochberg  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) 상수 — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
Z_WINDOW = 90                     # 1차 rolling z-score lookback (days)
LAG_DAYS = 1                      # 1차 발행지연 (flash 개정 누수 차단)
FEE_PCT_PER_SIDE = 0.04           # taker %
SLIPPAGE_PCT_PER_SIDE = 0.01      # slip % per fill
N_FILLS_PER_FLIP = 2             # 플립 = close + open
COST_MARGIN = 1.5

BOOTSTRAP_P_MAX = 0.05
BH_ALPHA = 0.05
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20                   # days (자기상관·변동성클러스터 보존)
IS_FRACTION = 0.70
REGIME_TOPK_PCT = 0.01            # 상위 1% |PnL|일 제거 후 부호 생존
REGIME_YEAR_MIN_FRAC = 0.60       # 연도별 양(+) 비율 최소
SEED = 42

ASSETS = ["btc", "eth"]
# robustness 그리드 (BH에 포함). (z_window, lag). 첫 항목 = PRIMARY.
GRID = [(90, 1), (30, 1), (90, 2), (30, 2)]
PRIMARY = (90, 1)
ANN = 365.0                       # 크립토 연율화

OUTDIR = ROOT / "results" / "onchain"
COST_BPS = None                   # 런타임 설정
EDGE_THRESHOLD_BPS = None


def _roundtrip_cost_bps() -> float:
    """플립(close+open) 비용 bps. leadlag 패턴 재사용."""
    fee = _calc_fee(1.0, 1.0, FEE_PCT_PER_SIDE)
    slip = abs(_apply_slippage(1.0, "BUY", SLIPPAGE_PCT_PER_SIDE) - 1.0)
    return N_FILLS_PER_FLIP * (fee + slip) * 1e4


def _load(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{asset}usdt" / "onchain_daily.parquet")
    df = df.dropna(subset=["FlowInExNtv", "FlowOutExNtv", "PriceUSD"]).sort_index()
    return df


def build_series(df: pd.DataFrame, z_window: int, lag: int) -> pd.DataFrame:
    """신호 → 포지션 → PnL. 반환: ret/pos/gross/net/flip date-indexed."""
    netflow = df["FlowInExNtv"] - df["FlowOutExNtv"]
    mu = netflow.rolling(z_window).mean()
    sd = netflow.rolling(z_window).std()
    z = (netflow - mu) / sd
    sig = -np.sign(z)                                   # 유입=숏, 유출=롱
    pos = sig.shift(1 + lag)                            # close(t-1) 확정 + 발행지연
    ret = np.log(df["PriceUSD"]).diff()                 # ret_t = t-1→t
    flip = (pos != pos.shift(1)) & pos.notna() & pos.shift(1).notna()
    cost = (COST_BPS * 1e-4) * flip.astype(float)
    gross = pos * ret
    out = pd.DataFrame({
        "ret": ret, "pos": pos, "gross": gross,
        "net": gross - cost, "flip": flip.astype(float),
    }).dropna(subset=["pos", "ret"])
    return out


def _block_bootstrap_mean_p(x: np.ndarray, rng) -> float:
    """이동 블록 부트스트랩: 평균>0 단측 p (자기상관 보존)."""
    n = len(x)
    if n < BLOCK_SIZE * 2:
        return 1.0
    nb = int(np.ceil(n / BLOCK_SIZE))
    max_start = n - BLOCK_SIZE
    offsets = np.arange(BLOCK_SIZE)
    boots = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        starts = rng.integers(0, max_start, nb)
        idx = (starts[:, None] + offsets).ravel()[:n]
        boots[b] = x[idx].mean()
    return float((boots <= 0).mean())             # 평균이 0 이하일 확률


def _sharpe(x: np.ndarray) -> float:
    s = x.std()
    return float(x.mean() / s * np.sqrt(ANN)) if s > 1e-12 else 0.0


def evaluate(series: pd.DataFrame, rng) -> dict:
    """단일 (자산/변형) 전 게이트 평가."""
    net = series["net"].values
    gross = series["gross"].values
    pos = series["pos"].values
    flip = series["flip"].values
    idx = series.index
    n = len(net)

    # ---- 경제성 (가장 싸고 결정적) ----
    gross_mean_bps = float(gross.mean() * 1e4)
    flip_rate = float(flip.mean())
    avg_hold = (1.0 / flip_rate) if flip_rate > 1e-9 else float(n)
    gross_per_hold_bps = gross_mean_bps * avg_hold
    g_econ = bool(gross_per_hold_bps > EDGE_THRESHOLD_BPS)

    net_mean_bps = float(net.mean() * 1e4)
    net_sharpe = _sharpe(net)
    net_total = float(np.exp(net.sum()) - 1.0)         # 누적수익률(로그합)

    # ---- 통계 (block bootstrap + BH는 호출측에서) ----
    boot_p = _block_bootstrap_mean_p(net, rng)
    g_stat_raw = bool(net.mean() > 0 and boot_p < BOOTSTRAP_P_MAX)

    # ---- 진위/레짐 ----
    k = max(1, int(n * REGIME_TOPK_PCT))
    order = np.argsort(np.abs(net))[::-1]
    keep = np.ones(n, dtype=bool)
    keep[order[:k]] = False
    net_ex = net[keep]
    regime_topk_survive = bool(net_ex.mean() > 0 and net.mean() > 0)
    yearly = pd.Series(net, index=idx).groupby(idx.year).sum()
    year_pos_frac = float((yearly > 0).mean()) if len(yearly) else 0.0
    g_regime = bool(regime_topk_survive and year_pos_frac >= REGIME_YEAR_MIN_FRAC)

    # ---- 안정성 (IS/OOS) ----
    cut = int(n * IS_FRACTION)
    is_net, oos_net = net[:cut], net[cut:]
    is_mean, oos_mean = is_net.mean(), oos_net.mean()
    oos_p = _block_bootstrap_mean_p(oos_net, rng)
    oos_sign_match = bool(np.sign(oos_mean) == np.sign(is_mean) and is_mean > 0)
    g_stab = bool(oos_sign_match and oos_p < BOOTSTRAP_P_MAX)

    # ---- 대칭성 (롱/숏 레그 각각) ----
    long_m = float(net[pos > 0].mean() * 1e4) if (pos > 0).any() else 0.0
    short_m = float(net[pos < 0].mean() * 1e4) if (pos < 0).any() else 0.0
    g_symm = bool(long_m > 0 and short_m > 0)

    # ---- 벤치마크 buy&hold ----
    bh = series["ret"].values
    bh_sharpe = _sharpe(bh)
    bh_total = float(np.exp(bh.sum()) - 1.0)

    return {
        "n": n, "years": round((idx[-1] - idx[0]).days / 365.25, 2),
        "gross_mean_bps": gross_mean_bps, "flip_rate": flip_rate,
        "avg_hold_days": avg_hold, "gross_per_hold_bps": gross_per_hold_bps,
        "net_mean_bps": net_mean_bps, "net_sharpe": net_sharpe, "net_total": net_total,
        "boot_p": boot_p, "g_stat_raw": g_stat_raw,
        "regime_topk_survive": regime_topk_survive, "year_pos_frac": year_pos_frac,
        "oos_sharpe": _sharpe(oos_net), "oos_sign_match": oos_sign_match, "oos_p": oos_p,
        "long_leg_bps": long_m, "short_leg_bps": short_m,
        "bh_sharpe": bh_sharpe, "bh_total": bh_total,
        "g_econ": g_econ, "g_regime": g_regime, "g_stab": g_stab, "g_symm": g_symm,
        "_yearly": {int(y): float(v) for y, v in yearly.items()},
    }


def run_precheck() -> dict:
    global COST_BPS, EDGE_THRESHOLD_BPS
    setup_korean_font()
    COST_BPS = _roundtrip_cost_bps()
    EDGE_THRESHOLD_BPS = COST_BPS * COST_MARGIN
    rng = np.random.default_rng(SEED)

    print("=" * 96)
    print("  온체인 거래소 netflow directional precheck — BTC/ETH 일봉 (백테스트 아님)")
    print(f"  플립비용 {COST_BPS:.1f}bps × {COST_MARGIN} = 보유기간당 edge 임계 {EDGE_THRESHOLD_BPS:.1f}bps")
    print(f"  1차 사양: z-window={PRIMARY[0]}d, LAG={PRIMARY[1]}d | robustness 그리드 {GRID}")
    print("=" * 96)

    raw = {a: _load(a) for a in ASSETS}

    # ---- 전 (자산 × 그리드) 시리즈 + EW 포트폴리오 ----
    results = {}     # key=(label) → eval dict
    series_store = {}
    for zw, lag in GRID:
        nets = {}
        for a in ASSETS:
            s = build_series(raw[a], zw, lag)
            series_store[(a, zw, lag)] = s
            results[f"{a.upper()}|z{zw}L{lag}"] = evaluate(s, rng)
            nets[a] = s["net"].rename(a)
        # EW 포트폴리오: 공통일자 평균 (net만; ret/pos/flip은 합성 평균)
        common = build_ew(series_store[("btc", zw, lag)], series_store[("eth", zw, lag)])
        series_store[("EW", zw, lag)] = common
        results[f"EW|z{zw}L{lag}"] = evaluate(common, rng)

    # ---- BH 보정 (전 테스트 bootstrap p) ----
    keys = list(results.keys())
    bh = _benjamini_hochberg([results[k]["boot_p"] for k in keys], BH_ALPHA)
    for k, surv in zip(keys, bh):
        r = results[k]
        r["bh_survive"] = bool(surv)
        r["g_stat"] = bool(r["g_stat_raw"] and surv)
        r["PASS"] = bool(r["g_econ"] and r["g_stat"] and r["g_regime"]
                         and r["g_stab"] and r["g_symm"])

    _print_headline(results)
    _print_summary(results)

    # ---- 최종 판정: PRIMARY 사양 EW + (BTC or ETH) ----
    zw, lag = PRIMARY
    ew = results[f"EW|z{zw}L{lag}"]
    btc = results[f"BTC|z{zw}L{lag}"]
    eth = results[f"ETH|z{zw}L{lag}"]
    final_pass = bool(ew["PASS"] and (btc["PASS"] or eth["PASS"]))

    print("\n" + "=" * 96)
    print(f"  최종(1차 사양 z{zw}L{lag}): EW {'PASS' if ew['PASS'] else 'FAIL'}, "
          f"BTC {'PASS' if btc['PASS'] else 'FAIL'}, ETH {'PASS' if eth['PASS'] else 'FAIL'}")
    print(f"  ⇒ {'✅ PASS — 방향성 backtester로 승격' if final_pass else '❌ FAIL — 온체인 netflow directional 라인 종료(예상대로)'}")
    print("=" * 96)

    _plots(raw, series_store, results)
    _dump(results, final_pass)
    return results


def build_ew(s_btc: pd.DataFrame, s_eth: pd.DataFrame) -> pd.DataFrame:
    """EW 포트폴리오: 공통일자 net/gross/ret 평균, flip은 두 레그 합(보수적)."""
    common = s_btc.index.intersection(s_eth.index)
    b, e = s_btc.loc[common], s_eth.loc[common]
    return pd.DataFrame({
        "ret": (b["ret"] + e["ret"]) / 2,
        "pos": np.sign(b["pos"] + e["pos"]),       # 진단용(대칭성 레그 판정)
        "gross": (b["gross"] + e["gross"]) / 2,
        "net": (b["net"] + e["net"]) / 2,
        "flip": (b["flip"] + e["flip"]) / 2,
    }, index=common)


def _print_headline(results: dict):
    print(f"\n[헤드라인 — 경제성 게이트] 보유기간당 gross edge vs {EDGE_THRESHOLD_BPS:.0f}bps 임계")
    print(f"  {'label':14s} {'n':>5s} {'grossbps':>9s} {'hold':>5s} {'per-hold':>9s} {'econ':>5s}")
    print("  " + "-" * 56)
    for k, r in results.items():
        print(f"  {k:14s} {r['n']:>5d} {r['gross_mean_bps']:>+8.2f}b {r['avg_hold_days']:>4.1f}d "
              f"{r['gross_per_hold_bps']:>+8.1f}b {'✓' if r['g_econ'] else '✗':>5s}")
    any_econ = any(r["g_econ"] for r in results.values())
    print("  " + "-" * 56)
    print(f"  경제성 통과: {sum(r['g_econ'] for r in results.values())}/{len(results)}  "
          f"→ {'일부 생존' if any_econ else '★전 변형 미달 = 결정적 킬(전체 리포트는 산출)'}")


def _print_summary(results: dict):
    print("\n" + "=" * 96)
    print("  요약 — 게이트: econ / stat(boot+BH) / regime / stab / symm")
    print("=" * 96)
    print(f"  {'label':14s} {'Sharpe':>7s} {'net%':>7s} {'bh%':>7s} {'bootP':>7s} {'BH':>3s} "
          f"{'yr+':>4s} {'OOSsh':>6s} {'L/Sbps':>11s} {'e':>2s}{'s':>2s}{'r':>2s}{'b':>2s}{'y':>2s} {'PASS':>4s}")
    for k, r in results.items():
        ls = f"{r['long_leg_bps']:+.1f}/{r['short_leg_bps']:+.1f}"
        print(f"  {k:14s} {r['net_sharpe']:>+7.2f} {r['net_total']*100:>+6.1f}% {r['bh_total']*100:>+6.1f}% "
              f"{r['boot_p']:>7.3f} {'✓' if r['bh_survive'] else '·':>3s} {r['year_pos_frac']:>4.2f} "
              f"{r['oos_sharpe']:>+6.2f} {ls:>11s} "
              f"{'✓' if r['g_econ'] else '·':>2s}{'✓' if r['g_stat'] else '·':>2s}"
              f"{'✓' if r['g_regime'] else '·':>2s}{'✓' if r['g_stab'] else '·':>2s}"
              f"{'✓' if r['g_symm'] else '·':>2s} {'PASS' if r['PASS'] else 'FAIL':>4s}")


def _plots(raw: dict, series_store: dict, results: dict):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    zw, lag = PRIMARY

    # (a) BTC netflow z vs 가격
    df = raw["btc"]
    netflow = df["FlowInExNtv"] - df["FlowOutExNtv"]
    z = (netflow - netflow.rolling(zw).mean()) / netflow.rolling(zw).std()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df["PriceUSD"], color="black", lw=0.7, label="BTC price")
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC price (log)")
    ax2 = ax1.twinx()
    ax2.plot(z.index, z.clip(-4, 4), color="firebrick", lw=0.5, alpha=0.5, label="netflow z")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("netflow z (clip±4)")
    ax1.set_title(f"(a) BTC netflow z(z{zw}) vs 가격 — 유입(z>0)=숏 신호")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"onchain_signal_btc_{today}.png", dpi=110)
    plt.close(fig)

    # (b) equity curve (1차 사양 전 시리즈)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in ["BTC", "ETH", "EW"]:
        a = label.lower() if label != "EW" else "EW"
        s = series_store[(a, zw, lag)]
        eq = np.exp(s["net"].cumsum())
        ax.plot(s.index, eq, lw=1.0, label=f"{label} 전략")
        if label == "EW":
            bh = np.exp(s["ret"].cumsum())
            ax.plot(s.index, bh, lw=0.8, ls="--", color="gray", label="EW buy&hold")
    ax.set_yscale("log")
    ax.axhline(1, color="black", lw=0.5)
    ax.set_title(f"(b) 순(net) equity curve — 1차 사양 z{zw}L{lag} (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"onchain_equity_{today}.png", dpi=110)
    plt.close(fig)

    # (c) 연도별 net PnL 바 (EW)
    yearly = results[f"EW|z{zw}L{lag}"]["_yearly"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    yrs = sorted(yearly)
    vals = [yearly[y] * 100 for y in yrs]
    ax.bar([str(y) for y in yrs], vals,
           color=["seagreen" if v > 0 else "firebrick" for v in vals])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("연도 net PnL (로그합 %)")
    ax.set_title(f"(c) EW 연도별 net PnL — 레짐 집중 진단 (z{zw}L{lag})")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"onchain_yearly_{today}.png", dpi=110)
    plt.close(fig)

    # (d) rolling 1년 net Sharpe (EW)
    s = series_store[("EW", zw, lag)]["net"]
    win = 365
    roll = s.rolling(win).mean() / s.rolling(win).std() * np.sqrt(ANN)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(roll.index, roll.values, lw=0.9, color="navy")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"(d) EW rolling 365일 net Sharpe (z{zw}L{lag})")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"onchain_rolling_sharpe_{today}.png", dpi=110)
    plt.close(fig)


def _dump(results: dict, final_pass: bool):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dump = {k: {kk: vv for kk, vv in r.items()} for k, r in results.items()}
    dump["_meta"] = {
        "generated": today, "cost_bps": COST_BPS,
        "edge_threshold_bps": EDGE_THRESHOLD_BPS,
        "primary": f"z{PRIMARY[0]}L{PRIMARY[1]}", "grid": [list(g) for g in GRID],
        "final_pass": final_pass,
        "caveat": "CM 거래소 플로우=휴리스틱 라벨 주소 기반(일부 거래소·노이즈). "
                  "flash 개정은 LAG로 차단하나 완전 배제 불가. 일봉=15m 봇 직접투입 불가.",
    }
    (OUTDIR / f"onchain_precheck_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/onchain/onchain_precheck_{today}.json + 플롯 4")
    print("  ※ caveat: CM 플로우=라벨 주소 휴리스틱. flash 개정은 LAG로 차단(완전배제 불가).")


if __name__ == "__main__":
    run_precheck()
