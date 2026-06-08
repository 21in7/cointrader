"""온체인 메트릭 바스켓 directional precheck (접근2). 백테스트 아님.

질문: netflow 외 다른 온체인 메트릭(밸류에이션·네트워크활동)에 방향성 edge가 있는가?
netflow directional FAIL(0/12)·overlay FAIL 후, "바스켓도 닫자"는 접근2. 무료로 가능.

메트릭·방향(사전등록, 잠금):
  MVRV(CapMVRVCur)  : 밸류에이션 → 평균회귀, 고평가(level z↑)=숏 (dir −1)
  활성주소(AdrActCnt): 네트워크성장 → 모멘텀, 활동↑(Δlog z↑)=롱 (dir +1)
  Tx수(TxCnt)       : 네트워크사용 → 모멘텀, 사용↑=롱 (dir +1)
구성원칙: 밸류에이션=level 평균회귀 / 활동=변화율(Δlog) 모멘텀(level은 채택과 함께
세속상승 → 베타프록시일 뿐).

게이트는 netflow directional과 동일 — `precheck.py`의 evaluate/build_ew/bootstrap 재사용.
신호 구성만 메트릭별로 교체. BH는 전체(3메트릭×{BTC,ETH,EW}×z{30,90}×LAG{1,2}=36테스트).
PASS = EW + (BTC or ETH)가 어떤 메트릭에서 전 게이트 통과.

사전확률 낮음: netflow(최강 동기)가 이미 실패, MVRV/활동은 가격과 반사적(베타 혼입).
실행:  python -m src.onchain.basket
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

import src.onchain.precheck as pc  # noqa: E402
from src.carry import setup_korean_font  # noqa: E402
from src.statarb.scan import _benjamini_hochberg  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
METRIC_SPECS = [
    {"name": "MVRV", "col": "CapMVRVCur", "mode": "level", "dir": -1},   # 고평가→숏
    {"name": "ActAddr", "col": "AdrActCnt", "mode": "mom", "dir": +1},   # 활동↑→롱
    {"name": "TxCnt", "col": "TxCnt", "mode": "mom", "dir": +1},         # 사용↑→롱
]
GRID = [(90, 1), (30, 1), (90, 2), (30, 2)]   # precheck와 동일
PRIMARY = (90, 1)
ASSETS = ["btc", "eth"]
OUTDIR = ROOT / "results" / "onchain"


def _load_metric(asset: str, col: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{asset}usdt" / "onchain_daily.parquet")
    return df.dropna(subset=[col, "PriceUSD"]).sort_index()


def build_metric_series(df: pd.DataFrame, spec: dict, z_window: int, lag: int) -> pd.DataFrame:
    """메트릭별 신호 → 포지션 → PnL. precheck.build_series와 동일 구조, 신호만 교체."""
    metric = df[spec["col"]].astype(float)
    base = metric if spec["mode"] == "level" else np.log(metric).diff()
    z = (base - base.rolling(z_window).mean()) / base.rolling(z_window).std()
    sig = spec["dir"] * np.sign(z)
    pos = sig.shift(1 + lag)                         # close(t-1) 확정 + 발행지연
    ret = np.log(df["PriceUSD"]).diff()
    flip = (pos != pos.shift(1)) & pos.notna() & pos.shift(1).notna()
    cost = (pc.COST_BPS * 1e-4) * flip.astype(float)
    gross = pos * ret
    return pd.DataFrame({
        "ret": ret, "pos": pos, "gross": gross,
        "net": gross - cost, "flip": flip.astype(float),
    }).dropna(subset=["pos", "ret"])


def run() -> dict:
    setup_korean_font()
    pc.COST_BPS = pc._roundtrip_cost_bps()
    pc.EDGE_THRESHOLD_BPS = pc.COST_BPS * pc.COST_MARGIN
    rng = np.random.default_rng(pc.SEED)

    print("=" * 100)
    print("  온체인 메트릭 바스켓 directional precheck (접근2) — MVRV·활성주소·Tx수 (백테스트 아님)")
    print(f"  플립비용 {pc.COST_BPS:.1f}bps × {pc.COST_MARGIN} = 보유기간당 edge 임계 {pc.EDGE_THRESHOLD_BPS:.1f}bps")
    print(f"  1차 사양 z{PRIMARY[0]}L{PRIMARY[1]} | 36테스트(3메트릭×3계열×4변형) BH 보정")
    print("=" * 100)

    results = {}
    for spec in METRIC_SPECS:
        for zw, lag in GRID:
            series = {}
            for a in ASSETS:
                df = _load_metric(a, spec["col"])
                s = build_metric_series(df, spec, zw, lag)
                series[a] = s
                results[f"{spec['name']}|{a.upper()}|z{zw}L{lag}"] = pc.evaluate(s, rng)
            ew = pc.build_ew(series["btc"], series["eth"])
            results[f"{spec['name']}|EW|z{zw}L{lag}"] = pc.evaluate(ew, rng)

    # ---- BH 보정 (전 36테스트) ----
    keys = list(results.keys())
    bh = _benjamini_hochberg([results[k]["boot_p"] for k in keys], pc.BH_ALPHA)
    for k, surv in zip(keys, bh):
        r = results[k]
        r["bh_survive"] = bool(surv)
        r["g_stat"] = bool(r["g_stat_raw"] and surv)
        r["PASS"] = bool(r["g_econ"] and r["g_stat"] and r["g_regime"]
                         and r["g_stab"] and r["g_symm"])

    _print_headline(results)
    _print_summary(results)

    # ---- 최종 판정: PRIMARY 사양, 메트릭별 EW+(BTC or ETH) ----
    zw, lag = PRIMARY
    metric_pass = {}
    for spec in METRIC_SPECS:
        nm = spec["name"]
        ew = results[f"{nm}|EW|z{zw}L{lag}"]["PASS"]
        btc = results[f"{nm}|BTC|z{zw}L{lag}"]["PASS"]
        eth = results[f"{nm}|ETH|z{zw}L{lag}"]["PASS"]
        metric_pass[nm] = bool(ew and (btc or eth))
    final = any(metric_pass.values())

    print("\n" + "=" * 100)
    for nm, p in metric_pass.items():
        print(f"  {nm:8s} (1차 z{zw}L{lag}): {'PASS' if p else 'FAIL'}")
    print(f"  ⇒ {'✅ PASS — 해당 메트릭 방향성 backtester로' if final else '❌ FAIL — 온체인 바스켓 directional 전멸(예상대로). 온체인 라인 종료'}")
    print("=" * 100)

    _dump(results, metric_pass, final)
    return results


def _print_headline(results: dict):
    print(f"\n[헤드라인 — 경제성] 보유기간당 gross edge vs {pc.EDGE_THRESHOLD_BPS:.0f}bps (1차 z90L1만 발췌)")
    print(f"  {'label':18s} {'grossbps':>9s} {'hold':>5s} {'per-hold':>9s} {'econ':>5s}")
    for k, r in results.items():
        if "z90L1" not in k:
            continue
        print(f"  {k.replace('|z90L1',''):18s} {r['gross_mean_bps']:>+8.2f}b {r['avg_hold_days']:>4.1f}d "
              f"{r['gross_per_hold_bps']:>+8.1f}b {'✓' if r['g_econ'] else '✗':>5s}")
    n_econ = sum(r["g_econ"] for r in results.values())
    print(f"  경제성 통과(전36): {n_econ}/{len(results)}  "
          f"→ {'일부 생존' if n_econ else '★전멸 = 결정적 킬'}")


def _print_summary(results: dict):
    print("\n" + "=" * 100)
    print("  요약 (1차 z90L1) — econ/stat(boot+BH)/regime/stab/symm")
    print("=" * 100)
    print(f"  {'label':14s} {'Sharpe':>7s} {'net%':>8s} {'bootP':>7s} {'BH':>3s} {'yr+':>4s} "
          f"{'L/Sbps':>13s} {'e':>2s}{'s':>2s}{'r':>2s}{'b':>2s}{'y':>2s} {'PASS':>4s}")
    for k, r in results.items():
        if "z90L1" not in k:
            continue
        ls = f"{r['long_leg_bps']:+.1f}/{r['short_leg_bps']:+.1f}"
        print(f"  {k.replace('|z90L1',''):14s} {r['net_sharpe']:>+7.2f} {r['net_total']*100:>+7.1f}% "
              f"{r['boot_p']:>7.3f} {'✓' if r['bh_survive'] else '·':>3s} {r['year_pos_frac']:>4.2f} "
              f"{ls:>13s} {'✓' if r['g_econ'] else '·':>2s}{'✓' if r['g_stat'] else '·':>2s}"
              f"{'✓' if r['g_regime'] else '·':>2s}{'✓' if r['g_stab'] else '·':>2s}"
              f"{'✓' if r['g_symm'] else '·':>2s} {'PASS' if r['PASS'] else 'FAIL':>4s}")


def _dump(results: dict, metric_pass: dict, final: bool):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dump = {k: {kk: vv for kk, vv in r.items() if kk != "_yearly"} for k, r in results.items()}
    dump["_meta"] = {
        "generated": today, "cost_bps": pc.COST_BPS,
        "edge_threshold_bps": pc.EDGE_THRESHOLD_BPS,
        "metric_specs": METRIC_SPECS, "primary": f"z{PRIMARY[0]}L{PRIMARY[1]}",
        "metric_pass": metric_pass, "final_pass": final,
        "caveat": "scan에 가까움(DoF↑) → BH 36테스트 보정. MVRV/활동은 가격과 반사적(베타혼입). "
                  "CM 라벨 휴리스틱. 일봉=15m봇 직접투입 불가.",
    }
    (OUTDIR / f"onchain_basket_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/onchain/onchain_basket_{today}.json")


if __name__ == "__main__":
    run()
