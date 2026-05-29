"""
크래시 포착(crisis-alpha/convexity)이 ETH 고유인가 크립토 전반의 일반 성질인가?

가설(falsification-first): "크래시 포착은 ETH 고유가 아니라 일반 성질이다."
ETH walk-forward와 동일 방법(anchored, 고정 L=2w + 적응형 L)을 8자산으로 확장 +
크래시 레짐 분해 + 자산간/에피소드간 비교. 백테스트 아님 — 검증·특성화.

핵심 방법론 주의(상관 보정): 크립토 크래시는 상관이 높아 8자산이 8 독립 확증이 아니다.
→ fold가 아니라 *구별되는 에피소드* 단위로도 집계한다.

재사용: src.momentum.walkforward(_build_folds,_run_mode,_strat_full,SKIP,FIXED_L 등),
       src.momentum.precheck(_load_logret,_metrics), src.carry(한글폰트).

실행:  python -m src.momentum.crisis_alpha
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
from src.momentum.precheck import _load_logret, _metrics  # noqa: E402
from src.momentum.walkforward import (  # noqa: E402
    FIXED_L, SKIP, _build_folds, _run_mode)

# ==========================================================================
# 사전등록(PRE-REGISTERED) — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
CRASH_DD_THRESHOLD = 0.30      # BTC trailing-peak 드로다운 > 30% = 크래시 레짐(정의 i)
# 정의 ii: 고정 명명 에피소드 (start, end). LUNA/FTX는 2022-bear의 부분집합임을 명시.
NAMED_EPISODES = [
    ("2018-bear",   "2018-01-15", "2018-12-15"),
    ("2021-05crash", "2021-05-10", "2021-07-20"),
    ("LUNA-2022-05", "2022-05-05", "2022-06-30"),
    ("FTX-2022-11",  "2022-11-06", "2022-12-31"),
    ("2022-bear",   "2022-01-01", "2022-12-31"),
]
# "일반 성질" 판정 기준
GENERAL_MIN_ASSETS = 5         # 크래시+ & 집중된 자산 ≥ 5/8
GENERAL_MIN_EPISODES = 3       # 구별 에피소드 ≥ 3개서 자산 전반 +포착
EPISODE_CAPTURE_FRAC = 0.5     # 에피소드서 '자산 전반 포착' = 존재자산 과반 +

ASSETS = ["BTC", "ETH", "XRP", "SOL", "AVAX", "LINK", "DOGE", "TRX"]
OUTDIR = ROOT / "results" / "momentum"


def _crash_mask() -> pd.Series:
    """BTC trailing-peak 드로다운 > 임계 → 크래시 레짐(일자 boolean)."""
    btc = pd.read_parquet(ROOT / "data" / "btcusdt" / "daily_spot.parquet")["close"]
    dd = btc / btc.cummax() - 1.0
    return (dd < -CRASH_DD_THRESHOLD)


def _episode_ranges():
    out = []
    for name, s, e in NAMED_EPISODES:
        out.append((name, pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")))
    return out


def run() -> dict:
    setup_korean_font()
    lr = _load_logret()
    crash = _crash_mask()
    episodes = _episode_ranges()

    print("=" * 100)
    print("  크래시 포착(crisis-alpha): ETH 고유 vs 크립토 일반 성질 — 8자산 walk-forward 확장")
    print(f"  크래시 정의(i): BTC trailing-peak DD > {CRASH_DD_THRESHOLD:.0%} | (ii) 명명 에피소드 {len(episodes)}개")
    print("=" * 100)

    per_asset = {}
    stitched_all = {}
    for a in ASSETS:
        r = lr[a]
        folds = _build_folds(r.index, anchored=True)
        if len(folds) < 2:
            print(f"  {a}: OOS fold 부족(상장 늦음) — skip 분해")
        rows_f, st_f = _run_mode(r, folds, adaptive=False)
        rows_a, st_a = _run_mode(r, folds, adaptive=True)
        stitched_all[a] = st_f
        m = _metrics(st_f)
        # buy&hold (동일 OOS 구간)
        bh = r[(r.index >= st_f.index[0]) & (r.index <= st_f.index[-1])]
        bh_m = _metrics(bh)
        # 크래시 vs 평시 분해 (정의 i)
        cm = crash.reindex(st_f.index).fillna(False)
        crash_pnl = float(st_f[cm].sum())
        normal_pnl = float(st_f[~cm].sum())
        crash_days = int(cm.sum())
        # 에피소드별 (정의 ii)
        ep = {}
        for name, s, e in episodes:
            seg = st_f[(st_f.index >= s) & (st_f.index <= e)]
            ep[name] = float(seg.sum()) if len(seg) else None  # None=OOS 미커버
        sel_L = [fr["selected_L"] for fr in rows_a]
        per_asset[a] = {
            "oos_start": str(st_f.index[0].date()), "oos_end": str(st_f.index[-1].date()),
            "sharpe": m["sharpe"], "ann_ret": m["ann_ret"], "mdd": m["mdd"],
            "bh_sharpe": bh_m["sharpe"], "beat_bh": bool(m["sharpe"] > bh_m["sharpe"]),
            "crash_pnl": crash_pnl, "normal_pnl": normal_pnl, "crash_days": crash_days,
            "total_pnl": float(st_f.sum()),
            "crash_concentrated": bool(crash_pnl > 0 and crash_pnl > normal_pnl),
            "normal_drag": bool(normal_pnl <= 0),
            "episodes": ep, "sel_L_mode_frac": max(sel_L.count(x) for x in set(sel_L)) / len(sel_L),
            "sel_L": sel_L,
        }

    # ---- 자산별 표 ----
    print(f"\n  {'asset':5s} {'OOS구간':23s} {'Sh':>5s} {'MDD':>5s} {'vsBH':>5s} "
          f"{'crashPnL':>9s} {'normPnL':>9s} {'집중':>4s} {'평시drag':>7s} {'selL최빈':>7s}")
    print("  " + "-" * 92)
    for a in ASSETS:
        p = per_asset[a]
        print(f"  {a:5s} {p['oos_start']+'~'+p['oos_end']:23s} {p['sharpe']:>5.2f} {p['mdd']*100:>4.0f}% "
              f"{'✓' if p['beat_bh'] else '·':>5s} {p['crash_pnl']*100:>+8.0f}% {p['normal_pnl']*100:>+8.0f}% "
              f"{'✓' if p['crash_concentrated'] else '·':>4s} {'✓' if p['normal_drag'] else '·':>7s} "
              f"{p['sel_L_mode_frac']*100:>5.0f}%")

    # ---- 에피소드 × 자산 매트릭스 ----
    print(f"\n  에피소드 × 자산 포착 매트릭스 (+ = 추세추종 +포착, − = 손실, · = OOS 미커버)")
    head = "  " + " " * 14 + "".join(f"{a:>6s}" for a in ASSETS)
    print(head)
    ep_capture = {}
    for name, s, e in episodes:
        cells, existed, captured = [], 0, 0
        for a in ASSETS:
            v = per_asset[a]["episodes"][name]
            if v is None:
                cells.append(f"{'·':>6s}")
            else:
                existed += 1
                captured += (v > 0)
                cells.append(f"{('+' if v>0 else '-')+f'{abs(v)*100:.0f}':>6s}")
        frac = (captured / existed) if existed else 0.0
        ep_capture[name] = {"existed": existed, "captured": captured, "frac": frac}
        flag = "✓" if (existed and frac >= EPISODE_CAPTURE_FRAC) else "·"
        print(f"  {name:14s}" + "".join(cells) + f"   [{captured}/{existed} {flag}]")

    # ---- 다자산 EW 추세 포트폴리오 (★진단용, 게이트 아님 — 판정은 per-asset 3게이트만) ----
    # outer-join: 날짜별 가용 자산 평균. inner-join은 늦게상장(AVAX)·일찍끝(TRX) 탓에
    # 2022-09~2025-12로 잘려 LUNA/2021 에피소드를 통째로 누락 → 헤드라인 왜곡(적대검증서 발견).
    ew = pd.concat(stitched_all.values(), axis=1, join="outer").mean(axis=1)
    ew_m = _metrics(ew)
    cm_ew = crash.reindex(ew.index).fillna(False)
    ew_crash, ew_normal = float(ew[cm_ew].sum()), float(ew[~cm_ew].sum())
    bh_ew = pd.concat([lr[a] for a in ASSETS], axis=1, join="outer").mean(axis=1)
    bh_ew = bh_ew[(bh_ew.index >= ew.index[0]) & (bh_ew.index <= ew.index[-1])]
    print(f"\n  다자산 EW 추세 포트폴리오 [★진단용, 게이트 아님] "
          f"({ew.index[0].date()}~{ew.index[-1].date()}, {len(ew)}일):")
    print(f"    Sharpe {ew_m['sharpe']:.2f} (vs buy&hold EW {_metrics(bh_ew)['sharpe']:.2f}), "
          f"MDD {ew_m['mdd']*100:.0f}%, crashPnL {ew_crash*100:+.0f}% / normPnL {ew_normal*100:+.0f}%")
    print("    ※ 이 EW는 per-asset 승자(ETH/AVAX/SOL)+높은 상관에 지배됨 → '일반 성질' 증거 아님(진단 지표).")

    # ---- 사전등록 판정 ----
    n_crash_conc = sum(per_asset[a]["crash_concentrated"] for a in ASSETS)
    n_normal_drag = sum(per_asset[a]["normal_drag"] for a in ASSETS)
    n_ep_capture = sum(1 for v in ep_capture.values() if v["existed"] and v["frac"] >= EPISODE_CAPTURE_FRAC)
    c1 = n_crash_conc >= GENERAL_MIN_ASSETS
    c2 = n_ep_capture >= GENERAL_MIN_EPISODES
    c3 = n_normal_drag >= GENERAL_MIN_ASSETS
    general = c1 and c2 and c3

    print("\n" + "=" * 100)
    print("  사전등록 '일반 성질' 판정")
    print("=" * 100)
    print(f"   {'✓' if c1 else '✗'}  크래시+ & 집중 자산 ≥ {GENERAL_MIN_ASSETS}/8: {n_crash_conc}/8")
    print(f"   {'✓' if c2 else '✗'}  자산전반 +포착 에피소드 ≥ {GENERAL_MIN_EPISODES}: {n_ep_capture} "
          f"({[k for k,v in ep_capture.items() if v['existed'] and v['frac']>=EPISODE_CAPTURE_FRAC]})")
    print(f"   {'✓' if c3 else '✗'}  평시 드래그 부호 일관(≥{GENERAL_MIN_ASSETS}/8 음): {n_normal_drag}/8")
    if general:
        print(f"\n  → ✅ 일반 성질 확정 — 단, 배포 알파 아님. confirmed RISK-OVERLAY(crisis convexity).")
        print(f"     (MDD·평시 드래그 그대로. 예측 알파 아니라 컨벡시티.)")
    else:
        only_eth = per_asset["ETH"]["crash_concentrated"] and n_crash_conc <= 2
        print(f"\n  → ❌ 일반 성질 미확정. {'ETH-specific 경향(아침 아티팩트 의심 일부 옳음)' if only_eth else '부분적/혼재'}.")
    print("=" * 100)
    print("  ※ 캐비엇: 크립토 크래시 상관 높음 → 8자산≠8 독립표본(에피소드 집계로 일부 보정). 진짜 보편성은")
    print("            비-크립토(주식·채권·상품) 필요=후속. 일반성질 확정돼도 '예측 알파'아닌 '리스크 오버레이'.")
    print("            walk-forward 2yr train → 늦게 상장한 자산(SOL/AVAX)은 초기 에피소드 OOS 미커버.")

    _plots(per_asset, ep_capture, episodes, stitched_all, ew, bh_ew)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dump = {"_meta": {"generated": today, "crash_dd_threshold": CRASH_DD_THRESHOLD,
                      "named_episodes": [(n, s, e) for n, s, e in NAMED_EPISODES],
                      "general_property": general, "n_crash_concentrated": n_crash_conc,
                      "n_episode_capture": n_ep_capture, "n_normal_drag": n_normal_drag,
                      "ew_sharpe": ew_m["sharpe"], "ew_mdd": ew_m["mdd"],
                      "ew_crash_pnl": ew_crash, "ew_normal_pnl": ew_normal},
            "per_asset": per_asset, "episode_capture": ep_capture}
    (OUTDIR / f"crisis_alpha_{today}.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/momentum/crisis_alpha_{today}.json + 플롯 3")
    return dump


def _plots(per_asset, ep_capture, episodes, stitched_all, ew, bh_ew):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # (a) 자산별 크래시 vs 평시 PnL
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ASSETS))
    ax.bar(x - 0.2, [per_asset[a]["crash_pnl"] * 100 for a in ASSETS], 0.4, label="크래시 PnL", color="firebrick")
    ax.bar(x + 0.2, [per_asset[a]["normal_pnl"] * 100 for a in ASSETS], 0.4, label="평시 PnL", color="steelblue")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, ASSETS)
    ax.set_title(f"(a) 자산별 stitched OOS PnL 분해: 크래시(BTC DD>{CRASH_DD_THRESHOLD:.0%}) vs 평시")
    ax.set_ylabel("누적 logret %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"crisis_decomp_{today}.png", dpi=110)
    plt.close(fig)

    # (b) 에피소드 × 자산 히트맵
    M = np.full((len(ASSETS), len(episodes)), np.nan)
    for i, a in enumerate(ASSETS):
        for j, (name, _, _) in enumerate(episodes):
            v = per_asset[a]["episodes"][name]
            if v is not None:
                M[i, j] = v * 100
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(episodes)), [e[0] for e in episodes], rotation=30, ha="right")
    ax.set_yticks(range(len(ASSETS)), ASSETS)
    for i in range(len(ASSETS)):
        for j in range(len(episodes)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("(b) 에피소드 × 자산: 추세추종 OOS PnL% (녹=포착, 적=손실, 공백=OOS 미커버)")
    fig.colorbar(im, label="PnL %")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"crisis_episode_matrix_{today}.png", dpi=110)
    plt.close(fig)

    # (c) EW 포트폴리오 vs 단일자산 vs buy&hold
    fig, ax = plt.subplots(figsize=(12, 6))
    for a in ASSETS:
        s = stitched_all[a]
        ax.plot(np.exp(s.cumsum()).index, np.exp(s.cumsum()).values, lw=0.6, alpha=0.45)
    ax.plot(np.exp(ew.cumsum()).index, np.exp(ew.cumsum()).values, lw=2.0, color="black", label="EW 추세 포트폴리오")
    ax.plot(np.exp(bh_ew.cumsum()).index, np.exp(bh_ew.cumsum()).values, lw=1.4, color="gray", ls="--", label="buy&hold EW")
    ax.set_yscale("log")
    ax.set_title("(c) 다자산 EW 추세 포트폴리오 vs 단일자산(얇은선) vs buy&hold (log축)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"crisis_ew_portfolio_{today}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run()
