"""
ETH 모멘텀(TSMOM_LS, L=2주) walk-forward 분석 — precheck 유일 통과 신호 판정.

질문: 56테스트 중 유일하게 게이트를 통과한 ETH-TSMOM-LS L2 가 진짜 이식 가능한
엣지인가, 단일구간 아티팩트인가? 예상: 아티팩트. PASS/FAIL은 보기 전 확정.

두 버전:
  (A) 고정 L=2주: 매 fold 동일 파라미터로 OOS 거래. "이 파라미터가 시간에 견고한가."
  (B) 적응형 L:  매 fold 학습구간에서 그리드 {1,2,4,12}주 중 최고를 골라 OOS 거래.
                "'과거로 최적 L 고르기' 방법론 자체가 앞으로 통하나" (더 정직한 질문).
anchored(확장창) 기본 + rolling(고정창) 보조. train=2yr, test=6mo, step=6mo.

신호는 [t-L, t] 과거만, 거래는 전방 — lookahead 차단(_tsmom의 pos_eff 시프트).
재사용: src.momentum.precheck(데이터·신호·비용·메트릭·bootstrap), src.carry(한글폰트).

실행:  python -m src.momentum.walkforward
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
from src.momentum.precheck import (  # noqa: E402
    ANN, SHORT_CARRY_SCENARIOS, _apply_carry, _load_logret, _metrics, _tsmom)

# ==========================================================================
# 사전등록(PRE-REGISTERED) — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
SYMBOL = "ETH"
FIXED_L = 2          # 주
SKIP = 1             # precheck 최고 통과 변형(L2/sk1, Sharpe 1.06)
GRID = [1, 2, 4, 12]  # 적응형 후보
TRAIN_YEARS = 2
TEST_MONTHS = 6
STEP_MONTHS = 6

# 생존 기준
SURVIVE_SHARPE = 0.5
MAX_FOLD_PNL_SHARE = 0.5     # 단일 fold가 총 OOS PnL의 과반이면 실격
MAJORITY = 0.5               # (A) fold 과반 +
ADAPTIVE_L_STABLE_FRAC = 0.60  # (B) 최빈 L이 fold의 ≥60%면 '안정적'

OUTDIR = ROOT / "results" / "momentum"


def _strat_full(r: pd.Series, L_weeks: int) -> pd.Series:
    """전구간 TSMOM_LS 순수익(causal). 슬라이스해서 fold OOS로 사용."""
    return _tsmom(r, L_weeks * 7, SKIP, "LS")[0]


def _pos_full(r: pd.Series, L_weeks: int) -> pd.Series:
    return _tsmom(r, L_weeks * 7, SKIP, "LS")[2]


def _build_folds(idx: pd.DatetimeIndex, anchored: bool):
    start, end = idx[0], idx[-1]
    folds = []
    test_start = start + pd.DateOffset(years=TRAIN_YEARS)
    while test_start + pd.DateOffset(months=TEST_MONTHS) <= end:
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        train_start = start if anchored else test_start - pd.DateOffset(years=TRAIN_YEARS)
        folds.append((train_start, test_start, test_end))
        test_start = test_start + pd.DateOffset(months=STEP_MONTHS)
    return folds


def _trades(pos: pd.Series, lo, hi) -> int:
    p = pos[(pos.index >= lo) & (pos.index < hi)]
    return int((p.diff().fillna(0) != 0).sum())


def _run_mode(r: pd.Series, folds, adaptive: bool):
    """fold별 OOS 수익·선택L·거래수 → stitched OOS."""
    nets_fixed = {L: _strat_full(r, L) for L in (GRID if adaptive else [FIXED_L])}
    pos_cache = {L: _pos_full(r, L) for L in (GRID if adaptive else [FIXED_L])}
    rows, stitched = [], []
    for (tr_s, te_s, te_e) in folds:
        if adaptive:
            best_L, best_sh = FIXED_L, -1e9
            for L in GRID:
                tr = nets_fixed[L][(nets_fixed[L].index >= tr_s) & (nets_fixed[L].index < te_s)]
                sh = _metrics(tr.dropna())["sharpe"] if len(tr.dropna()) > 5 else -1e9
                if sh > best_sh:
                    best_sh, best_L = sh, L
        else:
            best_L = FIXED_L
        net = nets_fixed[best_L]
        oos = net[(net.index >= te_s) & (net.index < te_e)].dropna()
        rows.append({"test_start": str(te_s.date()), "test_end": str(te_e.date()),
                     "selected_L": best_L, "oos_sharpe": _metrics(oos)["sharpe"],
                     "oos_ret_sum": float(oos.sum()), "trades": _trades(pos_cache[best_L], te_s, te_e),
                     "n": len(oos)})
        stitched.append(oos)
    stitched = pd.concat(stitched).sort_index()
    return rows, stitched


def _verdict(rows, stitched, bench, adaptive: bool) -> dict:
    m = _metrics(stitched)
    fold_sums = np.array([r["oos_ret_sum"] for r in rows])
    total = fold_sums.sum()
    max_share = float(fold_sums.max() / total) if total > 0 else float("inf")
    pos_fold_frac = float((fold_sums > 0).mean())
    beat_bh = m["sharpe"] > bench["sharpe"]

    crit = {
        "stitched_sharpe≥0.5 & buy&hold우위": bool(m["sharpe"] >= SURVIVE_SHARPE and beat_bh),
        "단일fold≤50% PnL집중": bool(total > 0 and max_share <= MAX_FOLD_PNL_SHARE),
    }
    if adaptive:
        Ls = [r["selected_L"] for r in rows]
        mode_frac = max(Ls.count(L) for L in set(Ls)) / len(Ls)
        crit["선택L 안정(최빈≥60%) & OOS+"] = bool(mode_frac >= ADAPTIVE_L_STABLE_FRAC and total > 0)
        extra = {"selected_L_mode_frac": mode_frac, "selected_L_list": Ls}
    else:
        crit["fold 과반 +"] = bool(pos_fold_frac > MAJORITY)
        extra = {"pos_fold_frac": pos_fold_frac}
    passed = all(crit.values())
    return {"metrics": m, "max_fold_pnl_share": max_share, "beat_bh": beat_bh,
            "criteria": crit, "PASS": passed, **extra}


def run_walkforward() -> dict:
    setup_korean_font()
    r = _load_logret()[SYMBOL]
    print("=" * 96)
    print(f"  ETH TSMOM_LS L={FIXED_L}주(sk{SKIP}) walk-forward — precheck 유일 통과 신호 판정")
    print(f"  train={TRAIN_YEARS}yr test={TEST_MONTHS}mo step={STEP_MONTHS}mo | "
          f"데이터 {r.index[0].date()}~{r.index[-1].date()} ({(r.index[-1]-r.index[0]).days/365.25:.1f}yr)")
    print("=" * 96)

    out = {}
    for anc_name, anchored in [("anchored", True), ("rolling", False)]:
        folds = _build_folds(r.index, anchored)
        bench = _metrics(r[(r.index >= folds[0][1]) & (r.index < folds[-1][2])])
        out[anc_name] = {"n_folds": len(folds), "bench": bench}
        for mode_name, adaptive in [("A_fixedL2", False), ("B_adaptiveL", True)]:
            rows, stitched = _run_mode(r, folds, adaptive)
            v = _verdict(rows, stitched, bench, adaptive)
            out[anc_name][mode_name] = {"folds": rows, "verdict": v,
                                        "stitched_index": [str(stitched.index[0].date()),
                                                           str(stitched.index[-1].date())]}
            out[anc_name][mode_name]["_stitched"] = stitched
            # 숏캐리 민감도 (LS라 ~절반 숏)
            sf = float((_pos_full(r, FIXED_L) < 0).mean())
            v["carry_sharpe"] = {f"{c:.0f}%": _metrics(_apply_carry(stitched, None, sf, c))["sharpe"]
                                 for c in SHORT_CARRY_SCENARIOS}

    # ---- 콘솔 리포트 ----
    for anc_name in ["anchored", "rolling"]:
        d = out[anc_name]
        bench = d["bench"]
        tag = "★기본" if anc_name == "anchored" else "보조"
        print(f"\n{'='*96}\n  [{anc_name}] {tag}  ({d['n_folds']} folds)  "
              f"buy&hold ETH: Sharpe {bench['sharpe']:.2f}, ann {bench['ann_ret']*100:+.0f}%, MDD {bench['mdd']*100:.0f}%")
        print("=" * 96)
        for mode_name in ["A_fixedL2", "B_adaptiveL"]:
            md = d[mode_name]
            v = md["verdict"]
            m = v["metrics"]
            print(f"\n  ── {mode_name} ──  stitched OOS {md['stitched_index'][0]}~{md['stitched_index'][1]}")
            print(f"    {'fold(test_start)':18s} {'selL':>4s} {'OOS Sh':>7s} {'retSum':>8s} {'trades':>6s}")
            for fr in md["folds"]:
                print(f"    {fr['test_start']:18s} {fr['selected_L']:>4d} {fr['oos_sharpe']:>7.2f} "
                      f"{fr['oos_ret_sum']*100:>+7.1f}% {fr['trades']:>6d}")
            print(f"    stitched OOS: Sharpe {m['sharpe']:.2f}  ann {m['ann_ret']*100:+.1f}%  "
                  f"MDD {m['mdd']*100:.0f}%  Calmar {m['calmar']:.2f}  "
                  f"(carry5/10: Sh {v['carry_sharpe']['5%']:.2f}/{v['carry_sharpe']['10%']:.2f})")
            print(f"    단일fold 최대 PnL비중 {v['max_fold_pnl_share']*100:.0f}%  buy&hold우위 {v['beat_bh']}", end="")
            if "selected_L_mode_frac" in v:
                print(f"  선택L 최빈비중 {v['selected_L_mode_frac']*100:.0f}% {v['selected_L_list']}")
            else:
                print(f"  fold+비율 {v['pos_fold_frac']*100:.0f}%")
            for k, ok in v["criteria"].items():
                print(f"      {'✓' if ok else '✗'}  {k}")
            print(f"    → {mode_name}: {'✅ PASS (walk-forward 생존)' if v['PASS'] else '❌ FAIL (단일구간 아티팩트)'}")

    # ---- 종합 판정 ----
    anc = out["anchored"]
    overall_pass = anc["A_fixedL2"]["verdict"]["PASS"] or anc["B_adaptiveL"]["verdict"]["PASS"]
    print("\n" + "=" * 96)
    print(f"  종합 판정 (anchored 기준): A={'PASS' if anc['A_fixedL2']['verdict']['PASS'] else 'FAIL'}, "
          f"B={'PASS' if anc['B_adaptiveL']['verdict']['PASS'] else 'FAIL'}")
    print(f"  → {'⚠️ 일부 생존 — 추가 검증 가치' if overall_pass else '❌ 단일구간 아티팩트 확정 — ETH-L2 라인 종료'}")
    print("=" * 96)
    print("  ※ 캐비엇: ETH 8.8yr 일봉·주간 리밸런스 → fold·독립거래 수 적음(방향 강하나 통계력 한계).")
    print("            (B)서 선택 L이 fold마다 튀면 그 자체가 'L=2주는 운'의 증거.")

    _plots(out, r)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dump = {"_meta": {"generated": today, "symbol": SYMBOL, "fixed_L": FIXED_L, "skip": SKIP,
                      "grid": GRID, "train_yr": TRAIN_YEARS, "test_mo": TEST_MONTHS,
                      "overall_pass_anchored": overall_pass}}
    for anc_name in ["anchored", "rolling"]:
        dump[anc_name] = {"n_folds": out[anc_name]["n_folds"], "bench": out[anc_name]["bench"]}
        for mn in ["A_fixedL2", "B_adaptiveL"]:
            md = out[anc_name][mn]
            dump[anc_name][mn] = {k: v for k, v in md.items() if k != "_stitched"}
    (OUTDIR / f"eth_walkforward_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/momentum/eth_walkforward_{today}.json + 플롯")
    return out


def _plots(out, r):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anc = out["anchored"]

    # (a) stitched OOS 곡선 vs buy&hold (anchored, A & B)
    fig, ax = plt.subplots(figsize=(12, 6))
    for mn, lab in [("A_fixedL2", "A 고정L2"), ("B_adaptiveL", "B 적응형L")]:
        s = anc[mn]["_stitched"]
        eq = np.exp(s.cumsum())
        ax.plot(eq.index, eq.values, lw=1.1, label=f"{lab} (Sh {anc[mn]['verdict']['metrics']['sharpe']:.2f})")
    s0 = anc["A_fixedL2"]["_stitched"]
    bh = r[(r.index >= s0.index[0]) & (r.index <= s0.index[-1])]
    ax.plot(np.exp(bh.cumsum()).index, np.exp(bh.cumsum()).values, lw=1.4, color="black", ls="--",
            label=f"buy&hold ETH (Sh {anc['bench']['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_title("(a) ETH 모멘텀 walk-forward stitched OOS vs buy&hold (anchored, log축)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"eth_wf_stitched_{today}.png", dpi=110)
    plt.close(fig)

    # (b) fold별 OOS Sharpe (anchored A vs B)
    fig, ax = plt.subplots(figsize=(12, 5))
    fa, fb = anc["A_fixedL2"]["folds"], anc["B_adaptiveL"]["folds"]
    x = np.arange(len(fa))
    ax.bar(x - 0.2, [f["oos_sharpe"] for f in fa], 0.4, label="A 고정L2", color="steelblue")
    ax.bar(x + 0.2, [f["oos_sharpe"] for f in fb], 0.4, label="B 적응형L", color="orange")
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(SURVIVE_SHARPE, color="firebrick", ls=":", lw=0.8, label=f"{SURVIVE_SHARPE} 바")
    ax.set_xticks(x, [f["test_start"][:7] for f in fa], rotation=45, fontsize=7)
    ax.set_title("(b) fold별 OOS Sharpe (anchored)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"eth_wf_foldsharpe_{today}.png", dpi=110)
    plt.close(fig)

    # (c) 적응형 선택 L (anchored)
    fig, ax = plt.subplots(figsize=(12, 4))
    fb = anc["B_adaptiveL"]["folds"]
    ax.plot(range(len(fb)), [f["selected_L"] for f in fb], "o-", color="darkgreen")
    ax.set_yticks(GRID)
    ax.set_xticks(range(len(fb)), [f["test_start"][:7] for f in fb], rotation=45, fontsize=7)
    ax.set_title("(c) 적응형: fold별 선택 L(주) — 튀면 'L=2는 운'의 증거")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"eth_wf_selectedL_{today}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run_walkforward()
