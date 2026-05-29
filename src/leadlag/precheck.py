"""
리드-래그 directional 시그널 precheck. 백테스트 아님.

질문: BTC/ETH(leader) 15m 수익률이 알트(alt) 수익률을 k바 선행하고, 그 예측가능
폭이 비용을 넘게 거래 가능한가? 산출물 = (leader, alt) 쌍별 PASS/FAIL.

성격: falsification-first. 유동성 15m 메이저는 이미 차익소거됐을 확률↑ → 예상 FAIL.
가장 싸고 결정적인 킬 = "예측 edge < 비용"(경제성 게이트)을 먼저. PASS면 기존
src/backtester.py 방향성 엔진이 바로 받는 신호.

PASS/FAIL은 데이터 보기 전 확정(상단 상수). 사후 변경 금지.

데이터: 각 알트 combined_15m.parquet 에 close(alt perp)·close_btc·close_eth 가 정렬
내장 → 파일 내에서 leader/alt 수익률이 완벽 정렬. (combined은 perp지만 directional
수익률엔 무관.) 수익률 = close-to-close 로그수익률.

재사용: src.backtester(비용/로더), src.statarb.scan(BH), src.carry(한글폰트).

실행:  python -m src.leadlag.precheck
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

from src.backtester import _apply_slippage, _calc_fee, _load_data  # noqa: E402
from src.carry import setup_korean_font  # noqa: E402
from src.statarb.scan import _benjamini_hochberg  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) PASS/FAIL — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
MAX_LAG = 8                       # k=1..8 (15m×8=2h)
FEE_PCT_PER_SIDE = 0.04           # taker %
SLIPPAGE_PCT_PER_SIDE = 0.01      # slip % per fill
N_FILLS_ROUNDTRIP = 2             # 단일 레그 directional: 진입+청산
COST_MARGIN = 1.5
ASYMMETRY_MIN = 2.0               # (leader→alt)/(alt→leader) ≥ 2
BOOTSTRAP_P_MAX = 0.05
BH_ALPHA = 0.05
LEADER_QUANTILE = 0.90            # 조건부 edge: 상위 10% |leader|

# ----- 자유 파라미터 (잠금) -----
N_BOOTSTRAP = 1000
BLOCK_SIZE = 96                   # 1일 블록 (자기상관·변동성클러스터 보존)
IS_FRACTION = 0.70
SEED = 42

LEADERS = {"BTC": "close_btc", "ETH": "close_eth"}
ALTS = ["XRPUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "TRXUSDT"]
OUTDIR = ROOT / "results" / "leadlag"


def _roundtrip_cost_bps() -> float:
    fee = _calc_fee(1.0, 1.0, FEE_PCT_PER_SIDE)
    slip = abs(_apply_slippage(1.0, "BUY", SLIPPAGE_PCT_PER_SIDE) - 1.0)
    return N_FILLS_ROUNDTRIP * (fee + slip) * 1e4


COST_BPS = None  # 런타임 설정
EDGE_THRESHOLD_BPS = None


def _lag_corr(lead: np.ndarray, alt: np.ndarray, k: int) -> float:
    """corr(lead_t, alt_{t+k}), k>=1 → leader가 alt를 선행."""
    if k == 0:
        return float(np.corrcoef(lead, alt)[0, 1])
    return float(np.corrcoef(lead[:-k], alt[k:])[0, 1])


def _best_lag(lead: np.ndarray, alt: np.ndarray, kmax: int = MAX_LAG):
    cc = {k: _lag_corr(lead, alt, k) for k in range(1, kmax + 1)}
    best = max(cc, key=lambda k: abs(cc[k]))
    return best, cc


def _uncond_edge_bps(corr_best: float, alt: np.ndarray) -> float:
    return abs(corr_best) * float(np.std(alt)) * 1e4


def _conditional_edge_bps(lead: np.ndarray, alt: np.ndarray, k: int) -> float:
    """상위 분위 |leader_t| 구간에서 sign(leader_t)·alt_{t+k} 평균 (lookahead 없음)."""
    lead_t = lead[:-k]
    alt_fwd = alt[k:]
    thr = np.quantile(np.abs(lead_t), LEADER_QUANTILE)
    mask = np.abs(lead_t) >= thr
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(lead_t[mask]) * alt_fwd[mask])) * 1e4


def _block_bootstrap_corr_p(x: np.ndarray, y: np.ndarray, rng) -> tuple[float, float, float]:
    """이동 블록 부트스트랩으로 corr CI/유의도. 자기상관 보존 → 유의도 부풀림 방지.
    반환: (p_value, ci_lo, ci_hi)."""
    n = len(x)
    nb = int(np.ceil(n / BLOCK_SIZE))
    boots = np.empty(N_BOOTSTRAP)
    max_start = n - BLOCK_SIZE
    offsets = np.arange(BLOCK_SIZE)
    for b in range(N_BOOTSTRAP):
        starts = rng.integers(0, max_start, nb)
        idx = (starts[:, None] + offsets).ravel()[:n]
        boots[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(min(p, 1.0)), float(lo), float(hi)


def _load_pairs() -> dict:
    """{(leader, alt): (lead_ret, alt_ret)} 정렬 로그수익률."""
    pairs = {}
    for alt in ALTS:
        df = _load_data(alt, None, None)
        alt_ret = np.log(df["close"]).diff().dropna().values
        for lname, col in LEADERS.items():
            lead_ret = np.log(df[col]).diff().dropna().values
            m = min(len(lead_ret), len(alt_ret))
            pairs[(lname, alt[:-4])] = (lead_ret[-m:], alt_ret[-m:])
    return pairs


def run_precheck() -> dict:
    global COST_BPS, EDGE_THRESHOLD_BPS
    setup_korean_font()
    COST_BPS = _roundtrip_cost_bps()
    EDGE_THRESHOLD_BPS = COST_BPS * COST_MARGIN
    rng = np.random.default_rng(SEED)

    print("=" * 92)
    print("  리드-래그 directional precheck — BTC/ETH → 6알트 (백테스트 아님)")
    print(f"  왕복비용 {COST_BPS:.1f}bps × {COST_MARGIN} = edge 임계 {EDGE_THRESHOLD_BPS:.1f}bps")
    print("=" * 92)

    pairs = _load_pairs()
    rows = {}

    # ---- 게이트 1 먼저: 경제성 헤드라인 (가장 싸고 결정적) ----
    print(f"\n[헤드라인 — 경제성 게이트] 예측 edge vs {EDGE_THRESHOLD_BPS:.0f}bps 임계")
    print(f"  {'pair':12s} {'bestlag':>7s} {'corr':>7s} {'uncond':>8s} {'cond':>8s} {'econ':>6s}")
    print("  " + "-" * 56)
    for (lead_name, alt_name), (lead, alt) in pairs.items():
        best, cc = _best_lag(lead, alt)
        uncond = _uncond_edge_bps(cc[best], alt)
        cond = _conditional_edge_bps(lead, alt, best)
        econ_pass = max(uncond, abs(cond)) > EDGE_THRESHOLD_BPS
        rows[(lead_name, alt_name)] = {
            "leader": lead_name, "alt": alt_name, "n": len(lead),
            "best_lag": best, "ccf": cc, "corr_best": cc[best],
            "uncond_edge_bps": uncond, "cond_edge_bps": cond, "g_econ": econ_pass,
        }
        print(f"  {lead_name+'→'+alt_name:12s} {best:>7d} {cc[best]:>+7.3f} "
              f"{uncond:>7.1f}b {cond:>+7.1f}b {'✓' if econ_pass else '✗':>6s}")

    any_econ = any(r["g_econ"] for r in rows.values())
    print("  " + "-" * 56)
    print(f"  경제성 통과 쌍: {sum(r['g_econ'] for r in rows.values())}/{len(rows)}  "
          f"→ {'일부 생존, 무거운 검정 진행' if any_econ else '★전 쌍 미달 = 결정적 킬 (그래도 전체 리포트 산출)'}")

    # ---- 게이트 2~4: 통계/진위/안정성 (전체 리포트용) ----
    print(f"\n[무거운 검정] block bootstrap(N={N_BOOTSTRAP}, 블록 {BLOCK_SIZE}) + 비대칭 + IS/OOS ...")
    for key, r in rows.items():
        lead, alt = pairs[key]
        k = r["best_lag"]
        x, y = lead[:-k], alt[k:]
        p, lo, hi = _block_bootstrap_corr_p(x, y, rng)
        r["bootstrap_p"], r["ci_lo"], r["ci_hi"] = p, lo, hi

        fwd = abs(_lag_corr(lead, alt, k))
        bwd = abs(_lag_corr(alt, lead, k))
        r["asym_ratio"] = (fwd / bwd) if bwd > 1e-12 else float("inf")

        cut = int(len(lead) * IS_FRACTION)
        b_is, _ = _best_lag(lead[:cut], alt[:cut])
        is_corr = _lag_corr(lead[:cut], alt[:cut], b_is)
        oos_corr = _lag_corr(lead[cut:], alt[cut:], b_is)
        p_oos, _, _ = _block_bootstrap_corr_p(lead[cut:][:-b_is], alt[cut:][b_is:], rng)
        r["is_best_lag"] = b_is
        r["oos_sign_match"] = bool(np.sign(oos_corr) == np.sign(is_corr) and oos_corr != 0)
        r["oos_corr"] = oos_corr
        r["oos_p"] = p_oos
        r["g_stab"] = bool(r["oos_sign_match"] and p_oos < BOOTSTRAP_P_MAX)
        r["g_asym"] = bool(r["asym_ratio"] >= ASYMMETRY_MIN)

    # ---- BH 보정 (전 쌍 best-lag bootstrap p) ----
    keys = list(rows.keys())
    bh = _benjamini_hochberg([rows[k]["bootstrap_p"] for k in keys], BH_ALPHA)
    for k, surv in zip(keys, bh):
        r = rows[k]
        r["bh_survive"] = bool(surv)
        r["g_stat"] = bool(r["bootstrap_p"] < BOOTSTRAP_P_MAX and surv)
        r["PASS"] = bool(r["g_econ"] and r["g_stat"] and r["g_asym"] and r["g_stab"])

    # ---- 요약 테이블 (예측 edge 내림차순) ----
    rank = sorted(rows.values(), key=lambda r: max(r["uncond_edge_bps"], abs(r["cond_edge_bps"])), reverse=True)
    print("\n" + "=" * 92)
    print("  요약 (예측 edge 내림차순) — 게이트: econ / stat(boot+BH) / asym / stab")
    print("=" * 92)
    print(f"  {'pair':12s} {'lag':>3s} {'corr':>7s} {'edge':>7s} {'bootP':>7s} {'BH':>3s} "
          f"{'asym':>6s} {'OOS':>4s} {'econ':>4s}{'stat':>5s}{'asym':>5s}{'stab':>5s}  {'PASS':>4s}")
    for r in rank:
        edge = max(r["uncond_edge_bps"], abs(r["cond_edge_bps"]))
        asym = r["asym_ratio"]
        asym_s = f"{asym:.1f}" if np.isfinite(asym) else "inf"
        print(f"  {r['leader']+'→'+r['alt']:12s} {r['best_lag']:>3d} {r['corr_best']:>+7.3f} "
              f"{edge:>6.1f}b {r['bootstrap_p']:>7.3f} {'✓' if r['bh_survive'] else '·':>3s} "
              f"{asym_s:>6s} {'✓' if r['oos_sign_match'] else '·':>4s} "
              f"{'✓' if r['g_econ'] else '·':>4s}{'✓' if r['g_stat'] else '·':>5s}"
              f"{'✓' if r['g_asym'] else '·':>5s}{'✓' if r['g_stab'] else '·':>5s}  "
              f"{'PASS' if r['PASS'] else 'FAIL':>4s}")

    n_pass = sum(r["PASS"] for r in rows.values())
    print("\n" + "=" * 92)
    print(f"  최종: {n_pass}/{len(rows)} 쌍 PASS  "
          f"→ {'생존 쌍을 방향성 backtester로' if n_pass else '❌ 전 쌍 FAIL — 리드-래그 라인 종료'}")
    print("=" * 92)

    _plots(rows, rank)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dump = {f"{r['leader']}->{r['alt']}": {k: v for k, v in r.items() if k != "ccf"}
            | {"ccf": {str(kk): vv for kk, vv in r["ccf"].items()}} for r in rows.values()}
    dump["_meta"] = {"generated": today, "cost_bps": COST_BPS,
                     "edge_threshold_bps": EDGE_THRESHOLD_BPS,
                     "n_pass": n_pass, "n_pairs": len(rows),
                     "caveat": "stale-price/비동기거래 아티팩트는 15m 메이저서 작으나 tick 없이 완전 배제 불가."}
    (OUTDIR / f"leadlag_precheck_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/leadlag/leadlag_precheck_{today}.json + 플롯 3")
    print("  ※ caveat: 15m close 동기성 가정. tick 없이 stale-price lead 누수 완전 배제 불가.")
    return rows


def _plots(rows: dict, rank: list):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    keys = list(rows.keys())
    labels = [f"{rows[k]['leader']}→{rows[k]['alt']}" for k in keys]

    # (a) CCF 히트맵 (쌍 × lag)
    mat = np.array([[rows[k]["ccf"][lag] for lag in range(1, MAX_LAG + 1)] for k in keys])
    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(MAX_LAG), [f"k={k+1}" for k in range(MAX_LAG)])
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(MAX_LAG):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("(a) 지연 교차상관 CCF: corr(leader_t, alt_{t+k})")
    fig.colorbar(im, label="corr")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"leadlag_ccf_heatmap_{today}.png", dpi=110)
    plt.close(fig)

    # (b) 최고 쌍 CCF + bootstrap CI
    top = rank[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    lags = list(range(1, MAX_LAG + 1))
    ax.bar(lags, [top["ccf"][k] for k in lags], color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhspan(top["ci_lo"], top["ci_hi"], color="firebrick", alpha=0.15,
               label=f"best-lag 95% boot CI [{top['ci_lo']:+.3f},{top['ci_hi']:+.3f}]")
    ax.set_title(f"(b) 최고 쌍 {top['leader']}→{top['alt']} CCF (best lag k={top['best_lag']})")
    ax.set_xlabel("lag k (15m bars)")
    ax.set_ylabel("corr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"leadlag_topccf_{today}.png", dpi=110)
    plt.close(fig)

    # (c) 베스트 lag rolling 상관 (최고 쌍)
    from src.backtester import _load_data as _ld
    alt_sym = top["alt"] + "USDT"
    df = _ld(alt_sym, None, None)
    lead = np.log(df[LEADERS[top["leader"]]]).diff().dropna()
    alt = np.log(df["close"]).diff().dropna()
    k = top["best_lag"]
    s = pd.Series(lead.values[:-k], index=lead.index[:-k])
    a = pd.Series(alt.values[k:], index=lead.index[:-k])
    win = 96 * 30  # 30일
    roll = s.rolling(win).corr(a)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(roll.index, roll.values, lw=0.8, color="darkgreen")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"(c) {top['leader']}→{top['alt']} best-lag(k={k}) rolling 30일 상관")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"leadlag_rolling_{today}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run_precheck()
