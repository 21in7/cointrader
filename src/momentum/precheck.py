"""
저빈도 모멘텀 directional precheck (TSMOM + XSMOM). 백테스트 아님.

변형:
  TSMOM_LS  시계열 모멘텀 long/short (추세 부호로 롱/숏)
  TSMOM_LO  시계열 모멘텀 long-only (상승=롱, 하락=현금) ← 가장 현실적 후보
  XSMOM     단면 모멘텀 long/short (랭킹 상위 롱/하위 숏, market-neutral)

성격: FAIL 예단 안 함(크립토 모멘텀은 문서화된 이상현상). 핵심은 **false positive 방지**
— 비용·생존편향·벤치마크를 빡세게. economics-first: 순수익−비용 → 벤치마크 대비가 헤드라인 킬.

헤드라인 검증 단위(사전지정, cherry-pick 방지):
  TSMOM: BTC·ETH 개별 + 전코인 EW 포트폴리오 / XSMOM: 포트폴리오. (per-coin 알트는 진단용)

PASS/FAIL은 데이터 보기 전 확정(상단 상수). 파라미터 사후 최적화 금지.
재사용: src.backtester(비용/로더 컨벤션), src.statarb.scan(BH), src.carry(한글폰트).

실행:  python -m src.momentum.precheck
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
from src.statarb.scan import _benjamini_hochberg  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) PASS/FAIL — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
MIN_SHARPE = 0.5                # 순 Sharpe ≥ 0.5
BOOTSTRAP_P_MAX = 0.05
BH_ALPHA = 0.05
MDD_VS_BENCH = 1.0              # 전략 MDD ≤ buy&hold MDD × 이 배수

# 신호 그리드 (사전등록, 최적화 금지 — 전부 테스트 후 BH 보정)
L_WEEKS = [1, 2, 4, 12]
SKIP_DAYS = [0, 1]
REBALANCE_DAYS = 7             # 주간 리밸런스

# 비용
ONEWAY_COST = 5e-4             # 5bps/leg → 플립(Δpos=2)=10bps 왕복
SHORT_CARRY_SCENARIOS = [0.0, 5.0, 10.0]   # 숏 노출 연 캐리 %/yr (병기, 게이트는 0)

# 자유 파라미터(잠금)
IS_FRACTION = 0.70
N_BOOTSTRAP = 1000
BLOCK_DAYS = 30
SEED = 42
ANN = 365
XSMOM_SIDE = 3                # 상/하위 각 3코인

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "TRXUSDT"]
OUTDIR = ROOT / "results" / "momentum"


# ---------- 데이터 ----------
def _load_logret() -> dict[str, pd.Series]:
    out = {}
    for s in SYMBOLS:
        p = ROOT / "data" / s.lower() / "daily_spot.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 — 먼저 python -m src.momentum.data")
        c = pd.read_parquet(p)["close"]
        out[s[:-4]] = np.log(c).diff().dropna()
    return out


# ---------- 전략 ----------
def _tsmom(r: pd.Series, L_days: int, skip: int, mode: str):
    """반환: (net_before_carry Series, short_frac, pos_eff Series)."""
    v = r.values
    n = len(v)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i >= L_days and (i - L_days) % REBALANCE_DAYS == 0:
            w = v[i - L_days:i - skip].sum() if skip else v[i - L_days:i].sum()
            s = 1.0 if w > 0 else (-1.0 if w < 0 else 0.0)
            cur = max(s, 0.0) if mode == "LO" else s
        pos[i] = cur
    pos_eff = np.concatenate([[0.0], pos[:-1]])          # 결정 다음날부터 적용(룩어헤드 차단)
    gross = pos_eff * v
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos_eff])))
    net = gross - dpos * ONEWAY_COST
    return (pd.Series(net, index=r.index), float((pos_eff < 0).mean()),
            pd.Series(pos_eff, index=r.index))


def _xsmom(R: pd.DataFrame, L_days: int, skip: int):
    """단면 롱숏(상/하위 XSMOM_SIDE). 반환: (net Series, short_frac=~0.5)."""
    idx = R.index
    M = R.values
    n, k = M.shape
    W = np.zeros((n, k))
    cur = np.zeros(k)
    for i in range(n):
        if i >= L_days and (i - L_days) % REBALANCE_DAYS == 0:
            past = M[i - L_days:i - skip].sum(0) if skip else M[i - L_days:i].sum(0)
            order = np.argsort(past)
            cur = np.zeros(k)
            cur[order[-XSMOM_SIDE:]] = 1.0 / XSMOM_SIDE
            cur[order[:XSMOM_SIDE]] = -1.0 / XSMOM_SIDE
        W[i] = cur
    W_eff = np.vstack([np.zeros(k), W[:-1]])
    gross = (W_eff * M).sum(1)
    turn = np.abs(np.diff(np.vstack([np.zeros(k), W_eff]), axis=0)).sum(1)
    net = gross - turn * ONEWAY_COST
    return pd.Series(net, index=idx), float(np.abs(np.minimum(W_eff, 0).sum(1)).mean())


# ---------- 메트릭 ----------
def _metrics(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean(), r.std()
    ann_ret = mu * ANN
    sharpe = (mu / sd * np.sqrt(ANN)) if sd > 0 else 0.0
    dn = r[r < 0].std()
    sortino = (mu / dn * np.sqrt(ANN)) if dn and dn > 0 else 0.0
    eq = np.exp(r.cumsum())
    mdd = float((eq / eq.cummax() - 1).min())
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {"ann_ret": float(ann_ret), "sharpe": float(sharpe), "sortino": float(sortino),
            "mdd": mdd, "calmar": float(calmar)}


def _apply_carry(net: pd.Series, short_frac_series: pd.Series | None, short_frac: float,
                 carry_pct: float) -> pd.Series:
    """숏 노출에 연 carry_pct 적용(일할). short_frac_series 없으면 평균 short_frac 사용."""
    if carry_pct == 0:
        return net
    drag = carry_pct / 100.0 / ANN * short_frac
    return net - drag


def _bootstrap_p(r: np.ndarray, rng) -> float:
    n = len(r)
    nb = int(np.ceil(n / BLOCK_DAYS))
    means = np.empty(N_BOOTSTRAP)
    maxs = n - BLOCK_DAYS
    off = np.arange(BLOCK_DAYS)
    for b in range(N_BOOTSTRAP):
        st = rng.integers(0, maxs, nb)
        idx = (st[:, None] + off).ravel()[:n]
        means[b] = r[idx].mean()
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(min(p, 1.0))


# ---------- 파이프라인 ----------
def run_precheck() -> dict:
    setup_korean_font()
    rng = np.random.default_rng(SEED)
    lr = _load_logret()
    common = pd.concat(lr.values(), axis=1, join="inner").dropna()  # 공통구간 (XSMOM/EW)
    common.columns = list(lr.keys())

    # buy&hold 벤치마크
    bench = {"BTC": _metrics(lr["BTC"]), "ETH": _metrics(lr["ETH"]),
             "EW": _metrics(common.mean(1))}
    bh_ret = {"BTC": lr["BTC"], "ETH": lr["ETH"], "EW": common.mean(1)}

    print("=" * 104)
    print("  저빈도 모멘텀 precheck — TSMOM(LS/LO) + XSMOM | economics-first")
    print("=" * 104)
    print("  buy&hold 벤치마크:  " + "  ".join(
        f"{k} ann{v['ann_ret']*100:+.0f}% Sh{v['sharpe']:.2f} MDD{v['mdd']*100:.0f}%"
        for k, v in bench.items()))
    print("  ⚠ 생존편향: '오늘 살아남은 코인'만의 결과 → 낙관 편향(실제 더 낮음). XSMOM 특히(8코인=얇음, primary는 TSMOM).")

    # 사전지정 단위 × 그리드 (56 테스트)
    tests = []
    for L in L_WEEKS:
        for sk in SKIP_DAYS:
            Ld = L * 7
            for mode in ["LS", "LO"]:
                for unit in ["BTC", "ETH", "EW"]:
                    tests.append({"variant": f"TSMOM_{mode}", "unit": unit,
                                  "L": L, "skip": sk, "mode": mode, "Ld": Ld})
            tests.append({"variant": "XSMOM", "unit": "PORT", "L": L, "skip": sk,
                          "mode": "XS", "Ld": Ld})

    # 전략 수익 계산
    for t in tests:
        Ld, sk = t["Ld"], t["skip"]
        if t["variant"] == "XSMOM":
            net, sfrac = _xsmom(common, Ld, sk)
            t["bench"] = "EW"
        else:
            unit = t["unit"]
            if unit == "EW":
                cols = [_tsmom(lr[c], Ld, sk, t["mode"])[0] for c in lr]
                net = pd.concat(cols, axis=1, join="inner").mean(1)
                sfrac = np.mean([_tsmom(lr[c], Ld, sk, t["mode"])[1] for c in lr])
            else:
                net, sfrac, _ = _tsmom(lr[unit], Ld, sk, t["mode"])
            t["bench"] = unit
        t["net"] = net.dropna()
        t["short_frac"] = sfrac
        m = _metrics(t["net"])
        t.update(m)
        # 숏 캐리 민감도
        t["carry_net_sharpe"] = {}
        for c in SHORT_CARRY_SCENARIOS:
            t["carry_net_sharpe"][f"{c:.0f}%"] = _metrics(_apply_carry(t["net"], None, sfrac, c))["sharpe"]

    # ---- 게이트 1: economics 헤드라인 (가장 싸고 결정적) ----
    print(f"\n[헤드라인 — economics 게이트]  순 Sharpe≥{MIN_SHARPE} & 순수익>0 & (TSMOM)buy&hold Sharpe 우위")
    print(f"  {'variant':9s} {'unit':5s} {'L':>3s} {'sk':>2s} {'netAnn':>7s} {'Sh':>6s} "
          f"{'benchSh':>7s} {'MDD':>6s} {'shFrac':>6s} {'carry5/10 Sh':>13s} {'econ':>5s}")
    print("  " + "-" * 92)
    for t in sorted(tests, key=lambda x: x["sharpe"], reverse=True):
        bsh = bench[t["bench"]]["sharpe"]
        beat = (t["sharpe"] > bsh) if t["variant"] != "XSMOM" else True  # XSMOM은 절대+위험조정
        t["g_econ"] = bool(t["sharpe"] >= MIN_SHARPE and t["ann_ret"] > 0 and beat)
        c5, c10 = t["carry_net_sharpe"]["5%"], t["carry_net_sharpe"]["10%"]
        print(f"  {t['variant']:9s} {t['unit']:5s} {t['L']:>3d} {t['skip']:>2d} "
              f"{t['ann_ret']*100:>+6.1f}% {t['sharpe']:>6.2f} {bsh:>7.2f} {t['mdd']*100:>5.0f}% "
              f"{t['short_frac']*100:>5.0f}% {c5:>6.2f}/{c10:>5.2f} {'✓' if t['g_econ'] else '·':>5s}")

    n_econ = sum(t["g_econ"] for t in tests)
    print("  " + "-" * 92)
    print(f"  economics 통과: {n_econ}/{len(tests)}  "
          f"→ {'생존 → 무거운 검정 진행' if n_econ else '★전 구성 미달 = 결정적 킬(그래도 전체 리포트 산출)'}")

    # ---- 게이트 2~4 ----
    print(f"\n[무거운 검정] block bootstrap(N={N_BOOTSTRAP}, 블록 {BLOCK_DAYS}d) + BH + 리스크 + IS/OOS ...")
    for t in tests:
        t["boot_p"] = _bootstrap_p(t["net"].values, rng)
        t["g_risk"] = bool(t["mdd"] >= bench[t["bench"]]["mdd"] * MDD_VS_BENCH)  # mdd는 음수: ≥ 면 덜 깊음
        # IS/OOS
        net = t["net"]
        cut = int(len(net) * IS_FRACTION)
        is_r, oos_r = net.iloc[:cut], net.iloc[cut:]
        b_oos = bh_ret[t["bench"]].reindex(oos_r.index).dropna()
        oos_strat_sh = _metrics(oos_r)["sharpe"]
        oos_bench_sh = _metrics(b_oos)["sharpe"] if len(b_oos) > 2 else 0.0
        beat_oos = (oos_strat_sh > oos_bench_sh) if t["variant"] != "XSMOM" else (oos_strat_sh >= MIN_SHARPE)
        t["g_stab"] = bool(_metrics(is_r)["sharpe"] >= MIN_SHARPE and oos_r.mean() > 0 and beat_oos)
        t["oos_strat_sharpe"], t["oos_bench_sharpe"] = oos_strat_sh, oos_bench_sh

    bh = _benjamini_hochberg([t["boot_p"] for t in tests], BH_ALPHA)
    for t, s in zip(tests, bh):
        t["bh_survive"] = bool(s)
        t["g_stat"] = bool(t["boot_p"] < BOOTSTRAP_P_MAX and s)
        t["PASS"] = bool(t["g_econ"] and t["g_stat"] and t["g_risk"] and t["g_stab"])

    # ---- 요약 (순 Sharpe 내림차순) ----
    print("\n" + "=" * 104)
    print("  요약 (순 Sharpe 내림차순) — econ/stat/risk/stab.  buy&hold 행 포함")
    print("=" * 104)
    print(f"  {'variant':9s} {'unit':5s} {'L/sk':>5s} {'netAnn':>7s} {'Sh':>6s} {'Sort':>6s} "
          f"{'MDD':>6s} {'Calmar':>6s} {'bootP':>6s} {'BH':>3s} {'OOS✓':>5s} "
          f"{'e':>2s}{'s':>2s}{'r':>2s}{'b':>2s} {'PASS':>4s}")
    for t in sorted(tests, key=lambda x: x["sharpe"], reverse=True):
        print(f"  {t['variant']:9s} {t['unit']:5s} {t['L']:>2d}/{t['skip']:<2d} "
              f"{t['ann_ret']*100:>+6.1f}% {t['sharpe']:>6.2f} {t['sortino']:>6.2f} "
              f"{t['mdd']*100:>5.0f}% {t['calmar']:>6.2f} {t['boot_p']:>6.3f} "
              f"{'✓' if t['bh_survive'] else '·':>3s} {'✓' if t['g_stab'] else '·':>5s} "
              f"{'✓' if t['g_econ'] else '·':>2s}{'✓' if t['g_stat'] else '·':>2s}"
              f"{'✓' if t['g_risk'] else '·':>2s}{'✓' if t['g_stab'] else '·':>2s} "
              f"{'PASS' if t['PASS'] else 'FAIL':>4s}")
    print("  " + "-" * 92)
    for k, v in bench.items():
        print(f"  {'buy&hold':9s} {k:5s} {'—':>5s} {v['ann_ret']*100:>+6.1f}% {v['sharpe']:>6.2f} "
              f"{v['sortino']:>6.2f} {v['mdd']*100:>5.0f}% {v['calmar']:>6.2f}")

    n_pass = sum(t["PASS"] for t in tests)
    print("\n" + "=" * 104)
    print(f"  최종: {n_pass}/{len(tests)} 구성 PASS  "
          f"→ {'생존 구성 → 방향성 backtester 후보' if n_pass else '❌ 전 구성 FAIL — 저빈도 모멘텀 라인 종료'}")
    print("=" * 104)

    _plots(tests, bh_ret, bench)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dump = {"_meta": {"generated": today, "n_pass": n_pass, "n_tests": len(tests),
                      "data": "spot daily, market="
                              "spot(≈perp directional), survivorship-biased universe(낙관 편향)",
                      "benchmark": {k: v for k, v in bench.items()}},
            "tests": [{k: v for k, v in t.items() if k != "net"} for t in tests]}
    (OUTDIR / f"momentum_precheck_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/momentum/momentum_precheck_{today}.json + 플롯 3")
    print("  ※ 캐비엇: 생존편향(낙관), XSMOM 8코인=얇음(넓은 유니버스 재검 필요), spot 일봉 기준.")
    return {"tests": tests, "bench": bench, "n_pass": n_pass}


def _plots(tests, bh_ret, bench):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # 대표 구성: 각 variant에서 순 Sharpe 최고 (EW/PORT 우선)
    reps = {}
    for v in ["TSMOM_LS", "TSMOM_LO", "XSMOM"]:
        cand = [t for t in tests if t["variant"] == v and t["unit"] in ("EW", "PORT")]
        reps[v] = max(cand, key=lambda x: x["sharpe"])

    # (a) 누적수익 vs buy&hold
    fig, ax = plt.subplots(figsize=(12, 6))
    for v, t in reps.items():
        eq = np.exp(t["net"].cumsum())
        ax.plot(eq.index, eq.values, lw=1.1, label=f"{v} (L{t['L']}/sk{t['skip']}) Sh{t['sharpe']:.2f}")
    eqb = np.exp(bh_ret["EW"].cumsum())
    ax.plot(eqb.index, eqb.values, lw=1.4, color="black", ls="--", label=f"buy&hold EW Sh{bench['EW']['sharpe']:.2f}")
    ax.set_yscale("log")
    ax.set_title("(a) 모멘텀 변형 누적수익 vs buy&hold EW (log축)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"momentum_cumret_{today}.png", dpi=110)
    plt.close(fig)

    # (b) rolling 1년 Sharpe (대표 TSMOM_LO EW vs bench)
    t = reps["TSMOM_LO"]
    fig, ax = plt.subplots(figsize=(12, 5))
    win = 365
    rs = t["net"].rolling(win).mean() / t["net"].rolling(win).std() * np.sqrt(ANN)
    bsr = bh_ret[t["bench"]].reindex(t["net"].index)
    rb = bsr.rolling(win).mean() / bsr.rolling(win).std() * np.sqrt(ANN)
    ax.plot(rs.index, rs.values, lw=1.0, label=f"{t['variant']} {t['unit']}")
    ax.plot(rb.index, rb.values, lw=1.0, color="black", ls="--", label=f"buy&hold {t['bench']}")
    ax.axhline(MIN_SHARPE, color="firebrick", ls=":", lw=0.8, label=f"{MIN_SHARPE} 바")
    ax.set_title("(b) rolling 1년 순 Sharpe (대표 TSMOM_LO vs buy&hold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"momentum_rollsharpe_{today}.png", dpi=110)
    plt.close(fig)

    # (c) 드로다운 곡선
    fig, ax = plt.subplots(figsize=(12, 5))
    for v, t in reps.items():
        eq = np.exp(t["net"].cumsum())
        dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, lw=0.9, label=f"{v}")
    eqb = np.exp(bh_ret["EW"].cumsum())
    ddb = (eqb / eqb.cummax() - 1) * 100
    ax.plot(ddb.index, ddb.values, lw=1.2, color="black", ls="--", label="buy&hold EW")
    ax.set_title("(c) 드로다운 곡선 (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"momentum_drawdown_{today}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run_precheck()
