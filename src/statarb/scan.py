"""
크로스-알트 공적분 스캔 — 보유 6심볼의 모든 페어(C(6,2)=15)를 전수 검사.

precheck.py(XRP/BTC 단일 페어)의 검정 함수를 재사용한다.
목적: stat-arb 패러다임에서 '공적분되는 알트 페어'가 존재하는지 1차 스크리닝.

★다중비교 보정 필수: 15개 페어를 p<0.05로 보면 우연히 ~0.75개 거짓양성.
  - Bonferroni: α/m = 0.05/15 = 0.00333 (보수적, family-wise)
  - Benjamini-Hochberg FDR: 순위 기반 (덜 보수적)
  두 기준 모두 리포트하고, '보정 후에도 살아남은' 페어만 PASS로 본다.

스크린 PASS 조건(전부 충족):
  EG p < Bonferroni α  AND  Johansen r=0 기각  AND
  half-life ∈ (30분, 7일)  AND  OOS-ADF p < 0.05

PASS 페어가 있으면 → 그 페어로 full precheck(rolling 포함) 진행.
PASS 페어가 없으면 → stat-arb 알트 라인도 여기서 종료.

실행:  python -m src.statarb.scan
"""
from __future__ import annotations

import itertools
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

from src.backtester import _load_data  # noqa: E402
from src.statarb.precheck import (  # noqa: E402
    HALF_LIFE_MAX_DAYS,
    HALF_LIFE_MIN_MINUTES,
    OOS_ADF_P_MAX,
    IS_FRACTION,
    adf,
    engle_granger,
    half_life,
    johansen,
    ols_beta_alpha,
)

SYMBOLS = ["AVAXUSDT", "DOGEUSDT", "LINKUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]
ALPHA = 0.05
OUTDIR = ROOT / "results" / "statarb"


def _benjamini_hochberg(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """BH-FDR: p_(k) <= (k/m)·α 인 최대 k까지 기각."""
    m = len(pvals)
    order = np.argsort(pvals)
    passed = [False] * m
    thresh_rank = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * alpha:
            thresh_rank = rank
    if thresh_rank > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= thresh_rank:
                passed[idx] = True
    return passed


def scan() -> dict:
    # ---- 종가 로드 + 바 간격 ----
    closes = {}
    bar_min = None
    for sym in SYMBOLS:
        df = _load_data(sym, None, None)
        closes[sym] = np.log(df["close"]).rename(sym)
        if bar_min is None:
            bar_min = pd.Series(df.index).diff().dropna().mode().iloc[0].total_seconds() / 60.0
    bars_per_day = 24 * 60 / bar_min

    print("=" * 86)
    print("  크로스-알트 공적분 스캔  (6심볼 × 15페어)")
    print("=" * 86)
    print(f"  바 간격 {bar_min:.0f}분.  다중비교 보정: Bonferroni α={ALPHA}/15={ALPHA/15:.5f}, BH-FDR")

    pairs = list(itertools.combinations(SYMBOLS, 2))
    rows = []
    for a, b in pairs:
        # 페어별 자체 공통구간(inner join) → 데이터 최대 활용
        j = pd.concat([closes[a], closes[b]], axis=1, join="inner").dropna()
        la, lb = j[a].values, j[b].values
        n = len(j)

        eg_ab = engle_granger(la, lb)["p"]   # A ~ B
        eg_ba = engle_granger(lb, la)["p"]    # B ~ A
        eg_best = min(eg_ab, eg_ba)
        joh = johansen(np.column_stack([la, lb]))

        beta, alpha_, _ = ols_beta_alpha(la, lb)
        spread = la - (alpha_ + beta * lb)
        hl = half_life(spread)
        hl_days = (hl["hl_bars"] * bar_min / (60 * 24)) if hl["valid"] else float("inf")

        cut = int(n * IS_FRACTION)
        beta_is, a_is, _ = ols_beta_alpha(la[:cut], lb[:cut])
        oos_adf_p = adf(la[cut:] - (a_is + beta_is * lb[cut:]))["p"]

        rows.append({
            "pair": f"{a[:-4]}/{b[:-4]}", "a": a, "b": b, "n": n,
            "eg_ab": eg_ab, "eg_ba": eg_ba, "eg_best": eg_best,
            "joh_reject_r0": joh["reject_r0"],
            "joh_trace": joh["trace_r0"], "joh_crit": joh["trace_r0_crit"],
            "half_life_days": hl_days, "hl_valid": hl["valid"],
            "oos_adf_p": oos_adf_p,
        })

    # ---- 다중비교 보정 ----
    eg_list = [r["eg_best"] for r in rows]
    bonf_alpha = ALPHA / len(rows)
    bh_pass = _benjamini_hochberg(eg_list, ALPHA)
    for r, bh in zip(rows, bh_pass):
        r["eg_raw_pass"] = r["eg_best"] < ALPHA
        r["eg_bonf_pass"] = r["eg_best"] < bonf_alpha
        r["eg_bh_pass"] = bool(bh)
        r["hl_pass"] = (r["hl_valid"]
                        and r["half_life_days"] * 24 * 60 > HALF_LIFE_MIN_MINUTES
                        and r["half_life_days"] < HALF_LIFE_MAX_DAYS)
        r["oos_pass"] = r["oos_adf_p"] < OOS_ADF_P_MAX
        r["SCREEN_PASS"] = bool(r["eg_bonf_pass"] and r["joh_reject_r0"]
                                and r["hl_pass"] and r["oos_pass"])

    rows.sort(key=lambda r: r["eg_best"])

    # ---- 테이블 출력 ----
    print(f"\n  {'pair':12s} {'N':>6s} {'EG_best':>8s} {'raw':>4s} {'Bonf':>5s} {'BH':>4s} "
          f"{'Joh':>5s} {'HL(d)':>8s} {'HLok':>5s} {'OOSp':>7s} {'OOSok':>6s} {'SCREEN':>7s}")
    print("  " + "-" * 84)
    for r in rows:
        hl_s = f"{r['half_life_days']:.1f}" if np.isfinite(r["half_life_days"]) else "inf"
        print(f"  {r['pair']:12s} {r['n']:>6d} {r['eg_best']:>8.4f} "
              f"{'✓' if r['eg_raw_pass'] else '·':>4s} {'✓' if r['eg_bonf_pass'] else '·':>5s} "
              f"{'✓' if r['eg_bh_pass'] else '·':>4s} {'✓' if r['joh_reject_r0'] else '·':>5s} "
              f"{hl_s:>8s} {'✓' if r['hl_pass'] else '·':>5s} {r['oos_adf_p']:>7.3f} "
              f"{'✓' if r['oos_pass'] else '·':>6s} {'PASS' if r['SCREEN_PASS'] else 'FAIL':>7s}")

    survivors = [r for r in rows if r["SCREEN_PASS"]]
    raw_survivors = [r for r in rows if r["eg_raw_pass"]]
    print("\n  " + "-" * 84)
    print(f"  raw p<0.05 통과(보정 전): {len(raw_survivors)}개  "
          f"→ 우연 기대 거짓양성 ≈ {len(rows) * ALPHA:.2f}개")
    print(f"  Bonferroni 통과: {sum(r['eg_bonf_pass'] for r in rows)}개   "
          f"BH-FDR 통과: {sum(r['eg_bh_pass'] for r in rows)}개")
    print(f"\n  최종 SCREEN PASS (Bonf 공적분 + Johansen + half-life + OOS): "
          f"{len(survivors)}개")
    if survivors:
        for r in survivors:
            print(f"     ✅ {r['pair']}  (EG p={r['eg_best']:.5f}, HL={r['half_life_days']:.2f}d) "
                  f"→ full precheck 권장")
    else:
        print("     ❌ 없음 — stat-arb 알트 라인도 종료. (XRP/BTC 단일 페어 FAIL과 일관)")
    print("=" * 86)

    result = {
        "generated": date.today().isoformat(),
        "symbols": SYMBOLS, "n_pairs": len(rows), "alpha": ALPHA,
        "bonferroni_alpha": bonf_alpha, "bar_minutes": bar_min,
        "correction_note": "EG는 15회 검정 → Bonferroni/BH 보정 적용. raw p<0.05는 우연 양성 포함.",
        "rows": rows,
        "raw_survivors": [r["pair"] for r in raw_survivors],
        "screen_pass": [r["pair"] for r in survivors],
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"crossalt_scan_{result['generated']}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    _heatmap(rows, result["generated"])
    print(f"\n  저장: {out.relative_to(ROOT)}")
    print(f"  히트맵: results/statarb/crossalt_eg_heatmap_{result['generated']}.png")
    return result


def _heatmap(rows, d):
    syms = [s[:-4] for s in SYMBOLS]
    m = np.full((len(syms), len(syms)), np.nan)
    idx = {s: i for i, s in enumerate(syms)}
    for r in rows:
        i, jx = idx[r["a"][:-4]], idx[r["b"][:-4]]
        m[i, jx] = m[jx, i] = r["eg_best"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(syms)), syms, rotation=45)
    ax.set_yticks(range(len(syms)), syms)
    for i in range(len(syms)):
        for jx in range(len(syms)):
            if not np.isnan(m[i, jx]):
                ax.text(jx, i, f"{m[i, jx]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cross-alt Engle-Granger best p-value (green=lower=cointegrated)")
    fig.colorbar(im, label="EG p-value")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"crossalt_eg_heatmap_{d}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    scan()
