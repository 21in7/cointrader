"""온체인 netflow 디리스크 오버레이 (비방향성) precheck. 백테스트 아님.

질문: 거래소 인플로우 급증(netflow z↑=분산)으로 롱 BTC/ETH 노출을 줄이면 위험조정수익이
buy&hold 대비 개선되며, 그 개선이 (a)단순 노출감소 (b)몇 크래시 에피소드로 설명되지
않는가? directional netflow FAIL(0/12) 후 비방향성 피벗 — netflow를 방향이 아니라 노출
조절에 사용. momentum/crisis-alpha 연구가 반복 시사한 "리스크 오버레이" 가설의 정직검증.

핵심 위험: 리스크 오버레이는 자기기만이 쉽다(아무 디리스크나 고변동자산 Sharpe↑).
→ 설계 핵심 = anti-fooling 벤치마크: B1 변동성관리(빈도매칭), B2 랜덤(부트스트랩 null).

전략(사전등록, 잠금): 기본 롱(exp=1), netflow_z_{t−1−LAG}>THRESHOLD 시 플랫(exp=0).
  숏 없음. z90/LAG1/THR1.0 1차. shift(1+LAG)로 lookahead·flash개정 차단.

게이트(사전등록): 1)개선(Sharpe↑&MDD↓ vs B0) 2)정보(랜덤 p<0.05 & vol 이상)
  3)레짐(크래시구간 제거 후 개선 생존) 4)안정성(OOS) + 포트폴리오(BTC·ETH or EW).

재사용: src.backtester(비용), src.carry(폰트). 데이터: data/{btc,eth}usdt/onchain_daily.parquet.
실행:  python -m src.onchain.overlay
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

# ==========================================================================
# 사전등록(PRE-REGISTERED) 상수 — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
Z_WINDOW = 90
LAG_DAYS = 1
THRESHOLD = 1.0                   # 1차 인플로우 급증 임계 (z)
GRID_THRESHOLD = [0.5, 1.0, 1.5, 2.0]
FEE_PCT_PER_SIDE = 0.04
SLIPPAGE_PCT_PER_SIDE = 0.01
VOL_WINDOW = 30                   # B1 실현변동성 lookback
N_RANDOM = 1000                   # B2 부트스트랩
IS_FRACTION = 0.70
SEED = 42
ANN = 365.0
ASSETS = ["btc", "eth"]
# 사전지정 크래시 구간 (레짐 게이트). (start, end) inclusive.
CRASH_WINDOWS = [
    ("2018-01-15", "2018-02-10"),   # 2018 1월 급락
    ("2020-03-01", "2020-04-30"),   # COVID
    ("2022-05-01", "2022-05-31"),   # LUNA
    ("2022-11-01", "2022-11-30"),   # FTX
]
OUTDIR = ROOT / "results" / "onchain"

FILL_COST = None                  # 런타임 (return 단위, 토글당)


def _fill_cost() -> float:
    """노출 토글 1 fill 비용(return 단위). leadlag 패턴 재사용."""
    fee = _calc_fee(1.0, 1.0, FEE_PCT_PER_SIDE)
    slip = abs(_apply_slippage(1.0, "BUY", SLIPPAGE_PCT_PER_SIDE) - 1.0)
    return fee + slip


def _load(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{asset}usdt" / "onchain_daily.parquet")
    return df.dropna(subset=["FlowInExNtv", "FlowOutExNtv", "PriceUSD"]).sort_index()


def _sharpe(x: np.ndarray) -> float:
    s = x.std()
    return float(x.mean() / s * np.sqrt(ANN)) if s > 1e-12 else 0.0


def _mdd(net: np.ndarray) -> float:
    """최대낙폭(음수). net=일별 로그수익률."""
    eq = np.exp(np.cumsum(net))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def _overlay_net(ret: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """노출 시퀀스 → 비용 차감 일별 net 로그수익률."""
    toggle = np.empty_like(exposure)
    toggle[0] = 0.0
    toggle[1:] = np.abs(np.diff(exposure))
    return exposure * ret - toggle * FILL_COST


def _zscore_netflow(df: pd.DataFrame, z_window: int) -> pd.Series:
    netflow = df["FlowInExNtv"] - df["FlowOutExNtv"]
    return (netflow - netflow.rolling(z_window).mean()) / netflow.rolling(z_window).std()


def build(df: pd.DataFrame, z_window: int, lag: int, threshold: float) -> pd.DataFrame:
    """netflow 오버레이 + buy&hold 정렬. 반환: ret/exp/net/bh/derisk date-indexed."""
    z = _zscore_netflow(df, z_window)
    derisk_raw = (z > threshold).astype(float)            # 인플로우 급증 → 디리스크
    derisk = derisk_raw.shift(1 + lag)                    # close(t-1) 확정 + 발행지연
    exp = 1.0 - derisk                                    # 1=롱, 0=플랫
    ret = np.log(df["PriceUSD"]).diff()
    out = pd.DataFrame({"ret": ret, "exp": exp, "derisk": derisk}).dropna()
    out["net"] = _overlay_net(out["ret"].values, out["exp"].values)
    out["bh"] = out["ret"].values                         # buy&hold(롱 상시)
    return out


def build_vol(df: pd.DataFrame, ret_index: pd.DatetimeIndex, lag: int,
              target_freq: float) -> np.ndarray:
    """B1 변동성관리: 실현변동성 상위 target_freq 구간 디리스크(빈도 매칭)."""
    ret = np.log(df["PriceUSD"]).diff()
    rv = ret.rolling(VOL_WINDOW).std().shift(1 + lag)
    rv = rv.reindex(ret_index)
    thr = np.nanquantile(rv.values, 1.0 - target_freq) if target_freq > 0 else np.inf
    exp = np.where(rv.values >= thr, 0.0, 1.0)
    return exp


def _gain(net: np.ndarray, bh: np.ndarray) -> float:
    return _sharpe(net) - _sharpe(bh)


def random_null(ret: np.ndarray, bh: np.ndarray, n_derisk: int, rng) -> np.ndarray:
    """B2: n_derisk 일을 무작위 플랫 → Sharpe-gain 분포."""
    n = len(ret)
    gains = np.empty(N_RANDOM)
    for b in range(N_RANDOM):
        mask = np.zeros(n)
        mask[rng.choice(n, n_derisk, replace=False)] = 1.0
        exp = 1.0 - mask
        net = _overlay_net(ret, exp)
        gains[b] = _sharpe(net) - _sharpe(bh)
    return gains


def evaluate(ov: pd.DataFrame, df: pd.DataFrame, rng) -> dict:
    """단일 자산 전 게이트."""
    ret = ov["ret"].values
    net = ov["net"].values
    bh = ov["bh"].values
    idx = ov.index
    n = len(net)
    n_derisk = int((ov["exp"].values == 0).sum())
    derisk_freq = n_derisk / n

    ov_sh, bh_sh = _sharpe(net), _sharpe(bh)
    ov_mdd, bh_mdd = _mdd(net), _mdd(bh)
    ov_tot = float(np.exp(net.sum()) - 1)
    bh_tot = float(np.exp(bh.sum()) - 1)

    # ---- 게이트 1: 개선 ----
    g1 = bool(ov_sh > bh_sh and ov_mdd > bh_mdd)          # mdd 음수: 클수록(0에 가까울수록) 양호

    # ---- 게이트 2: 정보 (랜덤 null + vol) ----
    overlay_gain = ov_sh - bh_sh
    rand_gains = random_null(ret, bh, n_derisk, rng) if n_derisk > 0 else np.zeros(N_RANDOM)
    p_random = float((rand_gains >= overlay_gain).mean())
    vol_exp = build_vol(df, idx, LAG_DAYS, derisk_freq)
    vol_net = _overlay_net(ret, vol_exp)
    vol_sh = _sharpe(vol_net)
    g2 = bool(p_random < 0.05 and ov_sh >= vol_sh)

    # ---- 게이트 3: 레짐 (크래시구간 제거 후 개선 생존) ----
    crash_mask = np.zeros(n, dtype=bool)
    for s, e in CRASH_WINDOWS:
        crash_mask |= (idx >= pd.Timestamp(s, tz="UTC")) & (idx <= pd.Timestamp(e, tz="UTC"))
    keep = ~crash_mask
    gain_excl = _gain(net[keep], bh[keep]) if keep.sum() > VOL_WINDOW else -1.0
    g3 = bool(gain_excl > 0)

    # ---- 게이트 4: 안정성 (OOS) ----
    cut = int(n * IS_FRACTION)
    oos_sh = _sharpe(net[cut:])
    oos_bh = _sharpe(bh[cut:])
    g4 = bool(oos_sh > oos_bh)

    return {
        "n": n, "years": round((idx[-1] - idx[0]).days / 365.25, 2),
        "derisk_freq": derisk_freq, "n_derisk": n_derisk,
        "ov_sharpe": ov_sh, "bh_sharpe": bh_sh, "vol_sharpe": vol_sh,
        "ov_mdd": ov_mdd, "bh_mdd": bh_mdd,
        "ov_total": ov_tot, "bh_total": bh_tot,
        "overlay_gain": overlay_gain, "p_random": p_random,
        "rand_gain_p95": float(np.percentile(rand_gains, 95)),
        "gain_excl_crash": gain_excl,
        "oos_sharpe": oos_sh, "oos_bh_sharpe": oos_bh,
        "g1_improve": g1, "g2_info": g2, "g3_regime": g3, "g4_stab": g4,
        "PASS": bool(g1 and g2 and g3 and g4),
    }


def build_ew(ov_b: pd.DataFrame, ov_e: pd.DataFrame) -> pd.DataFrame:
    """EW: 공통일자 net/bh/ret/exp 평균."""
    c = ov_b.index.intersection(ov_e.index)
    b, e = ov_b.loc[c], ov_e.loc[c]
    return pd.DataFrame({
        "ret": (b["ret"] + e["ret"]) / 2,
        "exp": (b["exp"] + e["exp"]) / 2,        # 0/0.5/1 (진단·빈도용)
        "derisk": (b["derisk"] + e["derisk"]) / 2,
        "net": (b["net"] + e["net"]) / 2,
        "bh": (b["bh"] + e["bh"]) / 2,
    }, index=c)


def evaluate_ew(ew: pd.DataFrame, rng) -> dict:
    """EW 전 게이트 (vol 벤치는 BTC/ETH 평균 대용 — 단순화: 자체 ret 변동성)."""
    ret = ew["ret"].values
    net = ew["net"].values
    bh = ew["bh"].values
    idx = ew.index
    n = len(net)
    n_derisk = int((ew["exp"].values < 1.0).sum())        # 한쪽이라도 디리스크

    ov_sh, bh_sh = _sharpe(net), _sharpe(bh)
    ov_mdd, bh_mdd = _mdd(net), _mdd(bh)
    g1 = bool(ov_sh > bh_sh and ov_mdd > bh_mdd)
    overlay_gain = ov_sh - bh_sh
    rand_gains = random_null(ret, bh, max(1, n_derisk), rng)
    p_random = float((rand_gains >= overlay_gain).mean())
    # vol 벤치(EW): 자체 ret 실현변동성 빈도매칭
    rv = pd.Series(ret, index=idx).rolling(VOL_WINDOW).std().shift(1 + LAG_DAYS)
    freq = n_derisk / n
    thr = np.nanquantile(rv.values, 1 - freq) if freq > 0 else np.inf
    vol_exp = np.where(rv.values >= thr, 0.0, 1.0)
    vol_exp[np.isnan(rv.values)] = 1.0
    vol_sh = _sharpe(_overlay_net(ret, vol_exp))
    g2 = bool(p_random < 0.05 and ov_sh >= vol_sh)

    crash_mask = np.zeros(n, dtype=bool)
    for s, e in CRASH_WINDOWS:
        crash_mask |= (idx >= pd.Timestamp(s, tz="UTC")) & (idx <= pd.Timestamp(e, tz="UTC"))
    keep = ~crash_mask
    gain_excl = _gain(net[keep], bh[keep])
    g3 = bool(gain_excl > 0)
    cut = int(n * IS_FRACTION)
    oos_sh, oos_bh = _sharpe(net[cut:]), _sharpe(bh[cut:])
    g4 = bool(oos_sh > oos_bh)
    return {
        "n": n, "derisk_freq": freq, "n_derisk": n_derisk,
        "ov_sharpe": ov_sh, "bh_sharpe": bh_sh, "vol_sharpe": vol_sh,
        "ov_mdd": ov_mdd, "bh_mdd": bh_mdd,
        "ov_total": float(np.exp(net.sum()) - 1), "bh_total": float(np.exp(bh.sum()) - 1),
        "overlay_gain": overlay_gain, "p_random": p_random,
        "rand_gain_p95": float(np.percentile(rand_gains, 95)),
        "gain_excl_crash": gain_excl, "oos_sharpe": oos_sh, "oos_bh_sharpe": oos_bh,
        "g1_improve": g1, "g2_info": g2, "g3_regime": g3, "g4_stab": g4,
        "PASS": bool(g1 and g2 and g3 and g4),
    }


def run() -> dict:
    global FILL_COST
    setup_korean_font()
    FILL_COST = _fill_cost()
    rng = np.random.default_rng(SEED)

    print("=" * 96)
    print("  온체인 netflow 디리스크 오버레이 (비방향성) — BTC/ETH 일봉 (백테스트 아님)")
    print(f"  토글비용 {FILL_COST*1e4:.1f}bps/fill | 1차: z{Z_WINDOW}/LAG{LAG_DAYS}/THR{THRESHOLD}")
    print(f"  게이트: 1)개선 2)정보(랜덤p<0.05&vol이상) 3)레짐(크래시제거생존) 4)OOS + 포트")
    print("=" * 96)

    raw = {a: _load(a) for a in ASSETS}
    ov = {a: build(raw[a], Z_WINDOW, LAG_DAYS, THRESHOLD) for a in ASSETS}
    res = {a.upper(): evaluate(ov[a], raw[a], rng) for a in ASSETS}
    ew = build_ew(ov["btc"], ov["eth"])
    res["EW"] = evaluate_ew(ew, rng)

    _print_main(res)

    # ---- robustness: THRESHOLD 스윕 (개선폭만) ----
    print("\n[robustness] THRESHOLD 스윕 — overlay Sharpe gain (vs B0)")
    print(f"  {'THR':>5s} | " + " ".join(f"{a.upper():>8s}" for a in ASSETS) + f" {'EW':>8s}")
    for thr in GRID_THRESHOLD:
        gains = []
        ovs = {a: build(raw[a], Z_WINDOW, LAG_DAYS, thr) for a in ASSETS}
        for a in ASSETS:
            o = ovs[a]
            gains.append(_sharpe(o["net"].values) - _sharpe(o["bh"].values))
        ewt = build_ew(ovs["btc"], ovs["eth"])
        gains.append(_sharpe(ewt["net"].values) - _sharpe(ewt["bh"].values))
        mark = " ←1차" if thr == THRESHOLD else ""
        print(f"  {thr:>5.1f} | " + " ".join(f"{g:>+8.3f}" for g in gains) + mark)

    # ---- 최종 판정 ----
    final = bool(res["EW"]["PASS"] or (res["BTC"]["PASS"] and res["ETH"]["PASS"]))
    print("\n" + "=" * 96)
    print(f"  최종: BTC {'PASS' if res['BTC']['PASS'] else 'FAIL'}, "
          f"ETH {'PASS' if res['ETH']['PASS'] else 'FAIL'}, EW {'PASS' if res['EW']['PASS'] else 'FAIL'}")
    print(f"  ⇒ {'✅ PASS — netflow 디리스크 오버레이 유효' if final else '❌ FAIL — 비방향성 오버레이도 무효(예상 시나리오 확인)'}")
    print("=" * 96)

    _plots(raw, ov, ew, res)
    _dump(res, final)
    return res


def _print_main(res: dict):
    print(f"\n  {'asset':5s} {'OVsh':>6s} {'BHsh':>6s} {'VOLsh':>6s} {'OVmdd':>7s} {'BHmdd':>7s} "
          f"{'gain':>7s} {'p_rnd':>6s} {'g-crash':>8s} {'OOSov/bh':>10s} {'1234':>5s} {'PASS':>4s}")
    for k, r in res.items():
        gates = f"{int(r['g1_improve'])}{int(r['g2_info'])}{int(r['g3_regime'])}{int(r['g4_stab'])}"
        print(f"  {k:5s} {r['ov_sharpe']:>+6.2f} {r['bh_sharpe']:>+6.2f} {r['vol_sharpe']:>+6.2f} "
              f"{r['ov_mdd']*100:>+6.1f}% {r['bh_mdd']*100:>+6.1f}% {r['overlay_gain']:>+7.3f} "
              f"{r['p_random']:>6.3f} {r['gain_excl_crash']:>+8.3f} "
              f"{r['oos_sharpe']:>+4.2f}/{r['oos_bh_sharpe']:>+4.2f} {gates:>5s} "
              f"{'PASS' if r['PASS'] else 'FAIL':>4s}")
    print(f"\n  디리스크 빈도: " + ", ".join(f"{k} {r['derisk_freq']*100:.1f}%" for k, r in res.items()))


def _plots(raw, ov, ew, res):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # (a) EW equity: overlay vs B0 vs B1(vol)
    ret = ew["ret"].values
    freq = res["EW"]["derisk_freq"]
    rv = pd.Series(ret, index=ew.index).rolling(VOL_WINDOW).std().shift(1 + LAG_DAYS)
    thr = np.nanquantile(rv.values, 1 - freq) if freq > 0 else np.inf
    vol_exp = np.where(rv.values >= thr, 0.0, 1.0)
    vol_exp[np.isnan(rv.values)] = 1.0
    vol_net = _overlay_net(ret, vol_exp)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ew.index, np.exp(ew["net"].cumsum()), lw=1.1, color="seagreen", label="netflow 오버레이")
    ax.plot(ew.index, np.exp(ew["bh"].cumsum()), lw=0.9, color="gray", ls="--", label="B0 buy&hold")
    ax.plot(ew.index, np.exp(np.cumsum(vol_net)), lw=0.9, color="steelblue", ls=":", label="B1 변동성관리")
    ax.set_yscale("log")
    ax.set_title("(a) EW equity — netflow 오버레이 vs buy&hold vs 변동성관리 (log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"overlay_equity_{today}.png", dpi=110)
    plt.close(fig)

    # (b) underwater (drawdown) EW: overlay vs B0
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col, c, lab in [("net", "seagreen", "오버레이"), ("bh", "gray", "buy&hold")]:
        eq = np.exp(ew[col].cumsum())
        dd = eq / np.maximum.accumulate(eq) - 1
        ax.fill_between(ew.index, dd * 100, 0, color=c, alpha=0.4, label=lab)
    ax.set_ylabel("drawdown %")
    ax.set_title("(b) EW underwater — 오버레이 MDD vs buy&hold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"overlay_underwater_{today}.png", dpi=110)
    plt.close(fig)

    # (c) BTC 디리스크일 + 크래시구간 표시
    df = raw["btc"]
    o = ov["btc"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["PriceUSD"], color="black", lw=0.7)
    ax.set_yscale("log")
    drd = o.index[o["derisk"] == 1.0]
    ax.scatter(drd, df["PriceUSD"].reindex(drd), s=4, color="firebrick", alpha=0.4, label="디리스크일")
    for s, e in CRASH_WINDOWS:
        ax.axvspan(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC"), color="orange", alpha=0.2)
    ax.set_title("(c) BTC 디리스크일(빨강) + 사전지정 크래시구간(주황) — 집중도 진단")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"overlay_derisk_days_{today}.png", dpi=110)
    plt.close(fig)

    # (d) 랜덤 null 분포 vs 실제 gain (EW)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    rng2 = np.random.default_rng(SEED + 1)
    rg = random_null(ew["ret"].values, ew["bh"].values, max(1, res["EW"]["n_derisk"]), rng2)
    ax.hist(rg, bins=50, color="lightgray", edgecolor="gray", label="B2 랜덤 null")
    ax.axvline(res["EW"]["overlay_gain"], color="seagreen", lw=2,
               label=f"netflow gain={res['EW']['overlay_gain']:+.3f} (p={res['EW']['p_random']:.3f})")
    ax.axvline(np.percentile(rg, 95), color="firebrick", lw=1, ls="--", label="랜덤 95%ile")
    ax.set_xlabel("Sharpe gain vs buy&hold")
    ax.set_title("(d) EW: netflow 오버레이 vs 랜덤 디리스크 null 분포")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"overlay_random_null_{today}.png", dpi=110)
    plt.close(fig)


def _dump(res: dict, final: bool):
    today = date.today().isoformat()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in res.items()}
    dump["_meta"] = {
        "generated": today, "fill_cost_bps": FILL_COST * 1e4,
        "z_window": Z_WINDOW, "lag": LAG_DAYS, "threshold": THRESHOLD,
        "crash_windows": CRASH_WINDOWS, "final_pass": final,
        "caveat": "리스크 오버레이 자기기만 방지 위해 B1(vol 빈도매칭)·B2(랜덤 null)·레짐(크래시제거) "
                  "삼중 차단. CM 플로우=라벨 휴리스틱. 일봉=15m봇 직접투입 불가.",
    }
    (OUTDIR / f"onchain_overlay_{today}.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False, default=float))
    print(f"\n  저장: results/onchain/onchain_overlay_{today}.json + 플롯 4")


if __name__ == "__main__":
    run()
