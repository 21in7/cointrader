"""신규 상장 마이크로구조 이벤트 스터디 (H1 페이드 / H2 돌파).

설계: docs/plans/2026-05-17-new-listing-microstructure-design.md
데이터: scripts/collect_listings.py 가 만든 data/listings/*.parquet
스크립트는 판정하지 않는다 — 사람이 design §5 사전 폐기기준에 대조.

H1 (페이드): 상장+E분 시점 open0 대비 |move|>=T% 면 그 반대로 진입,
             H분 보유 후 청산.
H2 (돌파):   첫 E분 [hi,lo] 레인지. 이후 종가가 hi 상향/lo 하향 돌파하면
             그 방향 추종 진입, H분 보유.
비용: fee 0.04%/leg + slip {0.3,0.6,1.0}%/fill, 진입+청산 2 fill.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd

_DIR = Path("data/listings")
_FEE = 0.0004  # taker/leg
_SLIPS = [0.003, 0.006, 0.010]
_E_GRID = [5, 15, 30, 60]
_T_GRID = [0.05, 0.10, 0.20]
_H_GRID = [30, 60, 240]


def _load_events():
    ev = []
    for p in sorted(_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if len(df) < 65:
            continue
        df = df.sort_index()
        ev.append((p.stem, df.index[0].year, df))
    return ev


def _cost(slip):
    return 2 * _FEE + 2 * slip  # 진입+청산 fee+slip 합 (수익률 차감분)


def _h1(df, E, T, H):
    o0 = df["open"].iloc[0]
    if o0 <= 0 or len(df) <= E + 1:
        return None
    cE = df["close"].iloc[E]
    move = cE / o0 - 1
    if abs(move) < T:
        return None
    direction = -1 if move > 0 else 1  # 페이드
    j = min(E + H, len(df) - 1)
    cX = df["close"].iloc[j]
    return direction * (cX / cE - 1)  # 비용 전 gross


def _h2(df, E, H):
    if len(df) <= E + 2:
        return None
    hi = df["high"].iloc[:E].max()
    lo = df["low"].iloc[:E].min()
    if hi <= 0 or lo <= 0:
        return None
    post = df.iloc[E:]
    for k in range(len(post)):
        c = post["close"].iloc[k]
        d = 1 if c > hi else (-1 if c < lo else 0)
        if d != 0:
            entry = c
            j = min(k + H, len(post) - 1)
            cX = post["close"].iloc[j]
            return d * (cX / entry - 1)
    return None


def _stats(grosses, years, slip):
    c = _cost(slip)
    nets = np.array(grosses) - c
    if len(nets) == 0:
        return None
    yr = {}
    ya = np.array(years)
    for y in sorted(set(years)):
        m = nets[ya == y]
        yr[y] = (len(m), m.mean() * 1e4)
    return {
        "n": len(nets),
        "mean_bps": nets.mean() * 1e4,
        "med_bps": np.median(nets) * 1e4,
        "hit": (nets > 0).mean() * 100,
        "std_bps": nets.std() * 1e4,
        "mu_sd": nets.mean() / nets.std() if nets.std() > 0 else 0.0,
        "by_year": yr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6, help="전략별 상위 그리드 표시")
    args = ap.parse_args()

    ev = _load_events()
    if not ev:
        print("[ERR] data/listings 비어있음 — collect_listings.py 먼저", file=sys.stderr)
        sys.exit(1)
    yrs = sorted(set(y for _, y, _ in ev))
    print(f"\n{'='*82}\n신규 상장 이벤트 스터디 — N={len(ev)} 상장 (연도 {yrs})\n{'='*82}")
    print(f"비용: fee {_FEE*100:.2f}%/leg + slip {[s*100 for s in _SLIPS]}%/fill, 2 fill\n")

    # ── H1 그리드 (slip 0.6% 기준 평균순익 정렬) ──
    h1_cells = []
    for E in _E_GRID:
        for T in _T_GRID:
            for H in _H_GRID:
                g, ys = [], []
                for _, y, df in ev:
                    r = _h1(df, E, T, H)
                    if r is not None:
                        g.append(r); ys.append(y)
                if len(g) < 30:
                    continue
                s06 = _stats(g, ys, 0.006)
                h1_cells.append((E, T, H, g, ys, s06))
    h1_cells.sort(key=lambda x: x[5]["mean_bps"], reverse=True)

    print(f"--- H1 페이드: 상위 {args.top} 그리드 (slip 0.6% 정렬) ---")
    print(f"{'E':>3}{'T%':>5}{'H':>5} {'n':>4} {'mean_bps':>9}{'med_bps':>9}{'hit%':>6}{'mu/sd':>7}")
    for E, T, H, g, ys, s in h1_cells[:args.top]:
        print(f"{E:>3}{T*100:>5.0f}{H:>5} {s['n']:>4} {s['mean_bps']:>9.1f}"
              f"{s['med_bps']:>9.1f}{s['hit']:>6.1f}{s['mu_sd']:>7.3f}")

    if h1_cells:
        bE, bT, bH, bg, bys, _ = h1_cells[0]
        print(f"\n--- H1 최우수셀 E={bE} T={bT*100:.0f}% H={bH} 비용민감도 ---")
        for sl in _SLIPS:
            st = _stats(bg, bys, sl)
            ysig = " ".join(f"{y}:{v[1]:+.0f}({v[0]})" for y, v in st["by_year"].items())
            print(f"slip {sl*100:>4.1f}%  n={st['n']} mean={st['mean_bps']:+.1f}bps "
                  f"med={st['med_bps']:+.1f} hit={st['hit']:.1f}% | 연도 {ysig}")

    # ── H2 그리드 ──
    h2_cells = []
    for E in _E_GRID:
        for H in _H_GRID:
            g, ys = [], []
            for _, y, df in ev:
                r = _h2(df, E, H)
                if r is not None:
                    g.append(r); ys.append(y)
            if len(g) < 30:
                continue
            h2_cells.append((E, H, g, ys, _stats(g, ys, 0.006)))
    h2_cells.sort(key=lambda x: x[4]["mean_bps"], reverse=True)

    print(f"\n--- H2 돌파: 상위 {args.top} 그리드 (slip 0.6%) ---")
    print(f"{'E':>3}{'H':>5} {'n':>4} {'mean_bps':>9}{'med_bps':>9}{'hit%':>6}{'mu/sd':>7}")
    for E, H, g, ys, s in h2_cells[:args.top]:
        print(f"{E:>3}{H:>5} {s['n']:>4} {s['mean_bps']:>9.1f}{s['med_bps']:>9.1f}"
              f"{s['hit']:>6.1f}{s['mu_sd']:>7.3f}")
    if h2_cells:
        bE, bH, bg, bys, _ = h2_cells[0]
        print(f"\n--- H2 최우수셀 E={bE} H={bH} 비용민감도 ---")
        for sl in _SLIPS:
            st = _stats(bg, bys, sl)
            ysig = " ".join(f"{y}:{v[1]:+.0f}({v[0]})" for y, v in st["by_year"].items())
            print(f"slip {sl*100:>4.1f}%  n={st['n']} mean={st['mean_bps']:+.1f}bps "
                  f"med={st['med_bps']:+.1f} hit={st['hit']:.1f}% | 연도 {ysig}")

    print(f"\n{'='*82}\n해석은 design §5 사전 폐기기준에 대조 — 스크립트 판정 안 함\n{'='*82}")


if __name__ == "__main__":
    main()
