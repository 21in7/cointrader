"""
XRP/BTC 통계적 차익거래(stat-arb) 적합성 사전검정 (precheck).

이것은 백테스트가 아니다. 단 하나의 PASS/FAIL 게이트가 최종 산출물이다.
FAIL이면 백테스트 모듈은 작성하지 않고 여기서 멈춘다.

PASS/FAIL 임계값은 데이터를 보기 전에 확정(pre-registered)되었으며,
결과를 본 뒤 변경하지 않는다(p-hacking 금지).

방법론(퀀트 표준, 순서대로):
  1) 적분차수: ADF + KPSS (log-level, Δlog) → I(1) 확인
  2) 공적분: Engle-Granger(양방향) + Johansen
  3) 스프레드: spread = log(XRP) − β·log(BTC) − α
  4) 평균회귀 속도: half-life (Ornstein-Uhlenbeck)
  5) Hurst: R/S + variance-ratio (★보조 진단, 게이트 아님)
  6) 안정성: IS/OOS 의 OOS-ADF + rolling 공적분
  7) 거래가능성: 2σ 진폭 vs (왕복 거래비용 + 보유 캐리)

재사용: src.backtester._load_data / _calc_fee / _apply_slippage

실행:
  python -m src.statarb.precheck --sanity     # 데이터+적분차수+self-test만
  python -m src.statarb.precheck              # 전체 파이프라인
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
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
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.stattools import adfuller, coint, kpss  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402

from src.backtester import _apply_slippage, _calc_fee, _load_data  # noqa: E402

# ==========================================================================
# 사전등록(PRE-REGISTERED) PASS/FAIL 임계값 — 데이터 보기 전 확정. 변경 금지.
# ==========================================================================
ADF_LEVEL_P_MIN = 0.10        # log-level ADF p > 이 값 (단위근 기각 실패 → I(1) 후보)
ADF_DIFF_P_MAX = 0.05         # Δlog ADF p < 이 값 (단위근 기각 → I(0))
EG_P_MAX = 0.05               # Engle-Granger 공적분 p < 이 값 (최소 XRP~BTC 방향)
JOHANSEN_CONF_IDX = 1         # cvt/cvm 컬럼 인덱스: 0=90%, 1=95%, 2=99%
HALF_LIFE_MIN_MINUTES = 30.0  # half-life 하한 (이보다 빠르면 노이즈/비용 과다)
HALF_LIFE_MAX_DAYS = 7.0      # half-life 상한 (이보다 느리면 캐리 부담)
OOS_ADF_P_MAX = 0.05          # IS-β 스프레드의 OOS ADF p < 이 값
ROLLING_EG_P_MAX = 0.05       # rolling 윈도우 EG p < 이 값을 '공적분으로 보임'으로 카운트
ROLLING_PASS_FRACTION = 0.70  # 공적분으로 보인 '달력시간 비중' 하한
TRADEABILITY_MULTIPLE = 3.0   # 2σ 진폭 > 총비용 × 이 배수
ENTRY_SIGMA = 2.0             # 진입 임계 ±2σ → 평균 회귀 시 2σ 포착

# ----- 자유 파라미터 (이번 세션 잠금) -----
IS_FRACTION = 0.70            # 시간순 단일 분할: 앞 70% = IS, 뒤 30% = OOS
ROLLING_WINDOW_DAYS = 90      # rolling 공적분 윈도우
ROLLING_STEP_DAYS = 1         # rolling step

# ----- 비용 모델 -----
FEE_PCT_PER_SIDE = 0.04       # taker 수수료 % (per fill)
SLIPPAGE_PCT_PER_SIDE = 0.01  # 슬리피지 % (per fill) — 사이즈 키우면 재검토 필요
N_FILLS_ROUNDTRIP = 4         # 2레그(XRP·BTC) × (진입+청산) = 4 fills
CARRY_BPS_PER_DAY = 3.0       # ★가정 캐리(perp funding / spot borrow 차익). 실측 funding 미반영

OUTDIR = ROOT / "results" / "statarb"


# ==========================================================================
# 통계 헬퍼
# ==========================================================================
def _bar_interval(idx: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = pd.Series(idx).diff().dropna()
    return diffs.mode().iloc[0]


def adf(series, regression: str = "c") -> dict:
    s = pd.Series(np.asarray(series, dtype=float)).dropna()
    stat, p, usedlag, nobs, crit, _ = adfuller(s, regression=regression, autolag="AIC")
    return {
        "stat": float(stat), "p": float(p), "lags": int(usedlag), "nobs": int(nobs),
        "crit": {k: float(v) for k, v in crit.items()},
    }


def kpss_test(series, regression: str = "c") -> dict:
    """KPSS — 귀무가설 = '정상성'(ADF와 반대). 레벨에서 기각(p작음)이면 비정상 = I(1) 일관."""
    s = pd.Series(np.asarray(series, dtype=float)).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value 보간 경고 억제
        stat, p, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {"stat": float(stat), "p": float(p), "lags": int(lags),
            "crit": {k: float(v) for k, v in crit.items()}}


def engle_granger(y, x) -> dict:
    """coint(y, x): y를 x로 회귀한 잔차의 ADF (EG 임계값 자동)."""
    t, p, crit = coint(np.asarray(y), np.asarray(x), trend="c", autolag="aic")
    return {"stat": float(t), "p": float(p),
            "crit": {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}}


def johansen(mat, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    res = coint_johansen(np.asarray(mat, dtype=float), det_order, k_ar_diff)
    evec0 = res.evec[:, 0]
    beta_hedge = float(-evec0[1] / evec0[0])  # spread = logXRP − β·logBTC
    return {
        "trace_r0": float(res.lr1[0]), "trace_r0_crit": float(res.cvt[0, JOHANSEN_CONF_IDX]),
        "trace_r1": float(res.lr1[1]), "trace_r1_crit": float(res.cvt[1, JOHANSEN_CONF_IDX]),
        "maxeig_r0": float(res.lr2[0]), "maxeig_r0_crit": float(res.cvm[0, JOHANSEN_CONF_IDX]),
        "reject_r0": bool(res.lr1[0] > res.cvt[0, JOHANSEN_CONF_IDX]),
        "reject_r1": bool(res.lr1[1] > res.cvt[1, JOHANSEN_CONF_IDX]),
        "beta_johansen": beta_hedge,
    }


def ols_beta_alpha(y, x):
    """y = α + β·x + resid. 반환: (β, α, resid)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    return beta, alpha, y - (alpha + beta * x)


def half_life(spread) -> dict:
    """OU: Δs_t = λ·s_{t-1} + c + ε. half-life = −ln2/λ (λ<0)."""
    s = pd.Series(np.asarray(spread, dtype=float)).dropna().reset_index(drop=True)
    ds = s.diff()
    s_lag = s.shift(1)
    d = pd.concat([ds, s_lag], axis=1).dropna()
    fit = sm.OLS(d.iloc[:, 0].values, sm.add_constant(d.iloc[:, 1].values)).fit()
    lam = float(fit.params[1])
    if lam >= 0:
        return {"lam": lam, "valid": False, "hl_bars": float("inf")}
    return {"lam": lam, "valid": True, "hl_bars": float(-np.log(2) / lam)}


def hurst_rs(ts, min_n: int = 16) -> float:
    """고전 R/S Hurst. RW→0.5 규약을 가지려면 '증분'에 적용해야 함(레벨에 쓰면 RW에서 ≈1).
    파이프라인에서는 Δspread(증분)에 적용. 평균회귀면 <0.5. (보조 진단)."""
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    N = len(ts)
    ns, rs = [], []
    n = min_n
    while n <= N // 2:
        chunks = N // n
        vals = []
        for i in range(chunks):
            c = ts[i * n:(i + 1) * n]
            dev = np.cumsum(c - c.mean())
            R = dev.max() - dev.min()
            S = c.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            ns.append(n)
            rs.append(np.mean(vals))
        n *= 2
    if len(ns) < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns), np.log(rs), 1)[0])


def hurst_vr(ts, max_lag: int = 200) -> float:
    """Variance-ratio(lag-τ) Hurst: log-σ(Δ_lag) vs log-lag 기울기. RW≈0.5."""
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    lags, taus = [], []
    for lag in range(2, min(max_lag, len(ts) // 2)):
        d = ts[lag:] - ts[:-lag]
        sd = np.std(d)
        if sd > 0:
            lags.append(lag)
            taus.append(sd)
    if len(lags) < 2:
        return float("nan")
    return float(np.polyfit(np.log(lags), np.log(taus), 1)[0])


def rolling_eg(logx: pd.Series, logy: pd.Series, window_bars: int, step_bars: int):
    """겹치는 윈도우라 자기상관 큼 → 비율은 '공적분으로 보인 달력시간 비중'으로만 해석."""
    n = len(logx)
    idx = logx.index
    pvals, centers = [], []
    start = 0
    while start + window_bars <= n:
        yw = logx.iloc[start:start + window_bars].values
        xw = logy.iloc[start:start + window_bars].values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = float(coint(yw, xw, trend="c")[1])
        except Exception:
            p = float("nan")
        pvals.append(p)
        centers.append(idx[start + window_bars // 2])
        start += step_bars
    s = pd.Series(pvals, index=pd.DatetimeIndex(centers))
    valid = s.dropna()
    frac = float((valid < ROLLING_EG_P_MAX).mean()) if len(valid) else 0.0
    return s, frac


def cost_model_bps(half_life_days: float) -> dict:
    """src.backtester 비용함수 재사용 → 왕복 거래비용 + 보유 캐리(bps)."""
    fee_frac = _calc_fee(1.0, 1.0, FEE_PCT_PER_SIDE) / 1.0                 # 1notional당 수수료 비율
    slip_frac = abs(_apply_slippage(1.0, "BUY", SLIPPAGE_PCT_PER_SIDE) - 1.0)
    per_fill_bps = (fee_frac + slip_frac) * 1e4
    txn_bps = N_FILLS_ROUNDTRIP * per_fill_bps
    carry_bps = CARRY_BPS_PER_DAY * max(half_life_days, 0.0)
    return {"per_fill_bps": per_fill_bps, "txn_bps": txn_bps,
            "carry_bps": carry_bps, "total_bps": txn_bps + carry_bps}


# ==========================================================================
# self-test: 합성데이터로 추정기 정확성 검증 (게이트 신뢰성의 전제)
# ==========================================================================
def self_test() -> None:
    print("\n[self-test] 합성데이터로 추정기 검증")
    rng = np.random.default_rng(42)

    # 1) 알려진 half-life의 OU 과정 복원
    lam_true = -0.04
    n = 30_000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + lam_true * x[t - 1] + rng.normal(0, 1)
    hl = half_life(x)
    hl_true = -np.log(2) / lam_true
    print(f"  OU half-life: 복원 {hl['hl_bars']:.2f} bars vs 정답 {hl_true:.2f} bars "
          f"→ {'OK' if abs(hl['hl_bars'] - hl_true) / hl_true < 0.15 else 'WARN'}")

    # 2) Hurst 규약 검증: R/S는 증분(white noise), VR은 레벨(random walk) → 둘 다 ≈0.5
    wn = rng.normal(0, 1, 60_000)        # white noise = RW의 증분
    rw = np.cumsum(wn)
    h_rs, h_vr = hurst_rs(wn), hurst_vr(rw)
    ok = abs(h_rs - 0.5) < 0.12 and abs(h_vr - 0.5) < 0.1
    print(f"  Hurst: R/S(증분) {h_rs:.3f}, VR(레벨) {h_vr:.3f} (둘 다 정답≈0.50) "
          f"→ {'OK' if ok else 'WARN'}")

    # 3) 정상 AR(1) 잔차는 ADF로 강하게 기각되어야
    p_ou = adf(x)["p"]
    print(f"  OU ADF p={p_ou:.4f} (정상 → p<0.05 기대) → {'OK' if p_ou < 0.05 else 'WARN'}")


# ==========================================================================
# 메인 파이프라인
# ==========================================================================
def run_precheck(symbol: str = "XRPUSDT", sanity_only: bool = False) -> dict:
    df = _load_data(symbol, None, None)
    bar = _bar_interval(df.index)
    bar_min = bar.total_seconds() / 60.0
    bars_per_day = 24 * 60 / bar_min

    logx = np.log(df["close"])
    logx.name = "logXRP"
    logy = np.log(df["close_btc"])
    logy.name = "logBTC"

    meta = {
        "symbol": symbol, "pair": "log(XRP) ~ log(BTC)",
        "rows": int(len(df)), "bar_minutes": bar_min,
        "range_start": str(df.index[0]), "range_end": str(df.index[-1]),
        "generated": date.today().isoformat(),
        "carry_warning": "CARRY_BPS_PER_DAY는 가정값. 실측 funding/borrow 미반영.",
    }

    print("=" * 74)
    print(f"  XRP/BTC stat-arb 적합성 precheck  ({symbol})")
    print("=" * 74)
    print(f"  데이터: {meta['rows']:,}행  {meta['range_start']} ~ {meta['range_end']}")
    print(f"  바 간격: {bar_min:.0f}분  (={bars_per_day:.0f} bars/day)")

    # ---- 1) 적분차수 ----
    adf_x_lvl, adf_y_lvl = adf(logx), adf(logy)
    adf_x_dif, adf_y_dif = adf(logx.diff().dropna()), adf(logy.diff().dropna())
    kpss_x_lvl, kpss_y_lvl = kpss_test(logx), kpss_test(logy)
    g_i1 = (adf_x_lvl["p"] > ADF_LEVEL_P_MIN and adf_y_lvl["p"] > ADF_LEVEL_P_MIN
            and adf_x_dif["p"] < ADF_DIFF_P_MAX and adf_y_dif["p"] < ADF_DIFF_P_MAX)

    print("\n[1] 적분차수 (I(1) 확인)")
    print(f"  ADF log-level : XRP p={adf_x_lvl['p']:.3f}  BTC p={adf_y_lvl['p']:.3f}  "
          f"(둘 다 > {ADF_LEVEL_P_MIN} 기대)")
    print(f"  ADF Δlog      : XRP p={adf_x_dif['p']:.4f}  BTC p={adf_y_dif['p']:.4f}  "
          f"(둘 다 < {ADF_DIFF_P_MAX} 기대)")
    print(f"  KPSS log-level: XRP p={kpss_x_lvl['p']:.3f}  BTC p={kpss_y_lvl['p']:.3f}  "
          f"(정상성 기각=비정상 → p작을수록 I(1) 일관)")
    print(f"  → I(1) 게이트: {'PASS' if g_i1 else 'FAIL'}")

    if sanity_only:
        self_test()
        print("\n[sanity] 데이터 로드 + 적분차수 + self-test 완료. 전체 실행은 --sanity 없이.")
        return {"meta": meta, "i1": g_i1}

    # ---- 2) 공적분 ----
    eg_xy = engle_granger(logx, logy)   # XRP ~ BTC
    eg_yx = engle_granger(logy, logx)   # BTC ~ XRP
    joh = johansen(df[["close", "close_btc"]].apply(np.log).values)
    g_eg = eg_xy["p"] < EG_P_MAX
    g_joh = joh["reject_r0"]

    print("\n[2] 공적분")
    print(f"  Engle-Granger XRP~BTC: stat={eg_xy['stat']:.3f}  p={eg_xy['p']:.4f}  "
          f"(crit5%={eg_xy['crit']['5%']:.3f})  (< {EG_P_MAX} 기대)")
    print(f"  Engle-Granger BTC~XRP: stat={eg_yx['stat']:.3f}  p={eg_yx['p']:.4f}  (비대칭 비교용)")
    print(f"  Johansen trace r=0   : {joh['trace_r0']:.2f}  vs 95%crit {joh['trace_r0_crit']:.2f}  "
          f"→ r=0 기각 {joh['reject_r0']}")
    print(f"  Johansen trace r<=1  : {joh['trace_r1']:.2f}  vs 95%crit {joh['trace_r1_crit']:.2f}  "
          f"→ r<=1 기각 {joh['reject_r1']} (rank=1이려면 False 기대)")
    print(f"  β(Johansen)={joh['beta_johansen']:.4f}")
    print(f"  → EG 게이트: {'PASS' if g_eg else 'FAIL'}   Johansen 게이트: {'PASS' if g_joh else 'FAIL'}")

    # ---- 3) 스프레드 (full-sample β: 존재 검정/진단 전용, look-ahead 있음) ----
    beta_full, alpha_full, resid_full = ols_beta_alpha(logx.values, logy.values)
    spread = pd.Series(resid_full, index=df.index, name="spread")
    sigma = float(spread.std())
    print(f"\n[3] 스프레드  spread = logXRP − β·logBTC − α   (β_OLS={beta_full:.4f}, α={alpha_full:.4f})")
    print(f"  σ(spread)={sigma:.5f}   (full-sample β는 look-ahead → 진단 전용, 실거래는 rolling/Kalman β)")

    # ---- 4) half-life ----
    hl = half_life(spread)
    hl_min = hl["hl_bars"] * bar_min
    hl_days = hl_min / (60 * 24)
    g_hl = hl["valid"] and (hl_min > HALF_LIFE_MIN_MINUTES) and (hl_days < HALF_LIFE_MAX_DAYS)
    print(f"\n[4] 평균회귀 속도 (OU half-life)")
    if hl["valid"]:
        print(f"  λ={hl['lam']:.5f}  half-life={hl['hl_bars']:.1f} bars "
              f"= {hl_min:.0f}분 = {hl_days:.2f}일")
        print(f"  → 게이트({HALF_LIFE_MIN_MINUTES:.0f}분 < HL < {HALF_LIFE_MAX_DAYS:.0f}일): "
              f"{'PASS' if g_hl else 'FAIL'}")
    else:
        print(f"  λ={hl['lam']:.5f} ≥ 0 → 평균회귀 아님(발산). FAIL")

    # ---- 5) Hurst (보조 진단, 게이트 아님) ----
    h_rs, h_vr = hurst_rs(np.diff(spread.dropna().values)), hurst_vr(spread.values)
    print(f"\n[5] Hurst (★보조 진단, 게이트 제외; RW≈0.5, <0.5 평균회귀)")
    print(f"  R/S(Δspread)={h_rs:.3f}   VR(spread)={h_vr:.3f}   "
          f"{'※ 두 값 엇갈림 — 신중 해석' if (h_rs - 0.5) * (h_vr - 0.5) < 0 else ''}")

    # ---- 6) 안정성: OOS + rolling ----
    cut = int(len(df) * IS_FRACTION)
    beta_is, alpha_is, _ = ols_beta_alpha(logx.iloc[:cut].values, logy.iloc[:cut].values)
    spread_oos = logx.iloc[cut:].values - (alpha_is + beta_is * logy.iloc[cut:].values)
    oos_adf = adf(spread_oos)
    g_oos = oos_adf["p"] < OOS_ADF_P_MAX

    win_bars = int(ROLLING_WINDOW_DAYS * bars_per_day)
    step_bars = int(ROLLING_STEP_DAYS * bars_per_day)
    roll_p, roll_frac = rolling_eg(logx, logy, win_bars, step_bars)
    g_roll = roll_frac >= ROLLING_PASS_FRACTION

    print(f"\n[6] 안정성 / 레짐")
    print(f"  OOS-ADF (IS-β={beta_is:.4f} 로 만든 뒤 30% 스프레드): p={oos_adf['p']:.4f}  "
          f"(< {OOS_ADF_P_MAX} 기대) → {'PASS' if g_oos else 'FAIL'}")
    print(f"  rolling 공적분({ROLLING_WINDOW_DAYS}일 윈도우, {len(roll_p)}개): "
          f"EG p<{ROLLING_EG_P_MAX}인 달력시간 비중 = {roll_frac*100:.1f}%  "
          f"(≥ {ROLLING_PASS_FRACTION*100:.0f}% 기대) → {'PASS' if g_roll else 'FAIL'}")
    print("    ※ 윈도우 중첩 → 자기상관 큼. 비율은 '독립검정 N회'가 아니라 달력시간 비중으로만 해석.")

    # ---- 7) 거래가능성 (비용 + 캐리) ----
    amp_bps = ENTRY_SIGMA * sigma * 1e4    # 2σ 진폭(log-spread)≈수익률 → bps
    cm = cost_model_bps(hl_days if hl["valid"] else HALF_LIFE_MAX_DAYS)
    threshold = cm["total_bps"] * TRADEABILITY_MULTIPLE
    headroom = amp_bps / threshold if threshold > 0 else float("inf")
    g_trade = amp_bps > threshold

    print(f"\n[7] 거래가능성 (★캐리는 가정값, 실측 funding 미반영)")
    print(f"  2σ 진폭 = {amp_bps:.1f} bps")
    print(f"  왕복 거래비용 = {cm['txn_bps']:.1f} bps  +  캐리 {CARRY_BPS_PER_DAY}bps/day × "
          f"{hl_days:.2f}일 = {cm['carry_bps']:.1f} bps  →  총비용 {cm['total_bps']:.1f} bps")
    print(f"  필요 임계 = 총비용 × {TRADEABILITY_MULTIPLE:.0f} = {threshold:.1f} bps")
    print(f"  여유 배수 = 진폭 / 임계 = {headroom:.2f}×  → {'PASS' if g_trade else 'FAIL'}")
    print("    ※ slippage 1bp/fill은 사이즈 키우면 재검토 필요.")

    # ---- 최종 게이트 ----
    checklist = [
        ("I(1) 적분차수", g_i1),
        ("Engle-Granger 공적분", g_eg),
        ("Johansen 공적분", g_joh),
        (f"half-life ∈ ({HALF_LIFE_MIN_MINUTES:.0f}분, {HALF_LIFE_MAX_DAYS:.0f}일)", g_hl),
        ("OOS 정상성", g_oos),
        (f"rolling 공적분 ≥ {ROLLING_PASS_FRACTION*100:.0f}%", g_roll),
        (f"거래가능성 (진폭 > 비용×{TRADEABILITY_MULTIPLE:.0f})", g_trade),
    ]
    passed = all(p for _, p in checklist)

    print("\n" + "=" * 74)
    print("  사전등록 체크리스트")
    print("=" * 74)
    for name, p in checklist:
        print(f"   {'✓' if p else '✗'}  {name}")
    print(f"\n  최종 판정: {'✅ PASS — 백테스트 단계로 진행 가능' if passed else '❌ FAIL — 여기서 중단 (백테스트 작성 금지)'}")
    print("=" * 74)

    result = {
        "meta": meta,
        "thresholds": {
            "ADF_LEVEL_P_MIN": ADF_LEVEL_P_MIN, "ADF_DIFF_P_MAX": ADF_DIFF_P_MAX,
            "EG_P_MAX": EG_P_MAX, "HALF_LIFE_MIN_MINUTES": HALF_LIFE_MIN_MINUTES,
            "HALF_LIFE_MAX_DAYS": HALF_LIFE_MAX_DAYS, "OOS_ADF_P_MAX": OOS_ADF_P_MAX,
            "ROLLING_PASS_FRACTION": ROLLING_PASS_FRACTION,
            "TRADEABILITY_MULTIPLE": TRADEABILITY_MULTIPLE, "ENTRY_SIGMA": ENTRY_SIGMA,
            "IS_FRACTION": IS_FRACTION, "ROLLING_WINDOW_DAYS": ROLLING_WINDOW_DAYS,
            "CARRY_BPS_PER_DAY": CARRY_BPS_PER_DAY,
        },
        "integration": {"adf_x_level": adf_x_lvl, "adf_y_level": adf_y_lvl,
                        "adf_x_diff": adf_x_dif, "adf_y_diff": adf_y_dif,
                        "kpss_x_level": kpss_x_lvl, "kpss_y_level": kpss_y_lvl},
        "cointegration": {"eg_xrp_btc": eg_xy, "eg_btc_xrp": eg_yx, "johansen": joh},
        "spread": {"beta_ols": beta_full, "alpha_ols": alpha_full, "sigma": sigma},
        "half_life": {**hl, "minutes": hl_min, "days": hl_days},
        "hurst": {"rs_on_diff": h_rs, "vr_on_level": h_vr,
                  "note": "R/S=Δspread, VR=spread levels; RW≈0.5; 보조 진단, 게이트 제외"},
        "stability": {"oos_adf": oos_adf, "beta_is": beta_is,
                      "rolling_frac": roll_frac, "rolling_windows": int(len(roll_p))},
        "tradeability": {"amplitude_bps": amp_bps, **cm,
                         "threshold_bps": threshold, "headroom_mult": headroom},
        "checklist": {name: bool(p) for name, p in checklist},
        "PASS": bool(passed),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTDIR / f"xrpbtc_precheck_{meta['generated']}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    _make_plots(logx, logy, spread, sigma, roll_p, meta)
    print(f"\n  저장: {out_json.relative_to(ROOT)}")
    print(f"  플롯: {OUTDIR.relative_to(ROOT)}/xrpbtc_*_{meta['generated']}.png")
    return result


def _make_plots(logx, logy, spread, sigma, roll_p, meta):
    d = meta["generated"]
    # (a) 정규화 log 오버레이
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(logx.index, (logx - logx.mean()) / logx.std(), label="log(XRP) z", lw=0.7)
    ax.plot(logy.index, (logy - logy.mean()) / logy.std(), label="log(BTC) z", lw=0.7)
    ax.set_title("(a) 정규화 log(XRP) vs log(BTC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"xrpbtc_overlay_{d}.png", dpi=110)
    plt.close(fig)

    # (b) 스프레드 + 평균 + ±2σ
    fig, ax = plt.subplots(figsize=(12, 5))
    m = spread.mean()
    ax.plot(spread.index, spread.values, lw=0.6, color="steelblue")
    for k, ls in [(0, "-"), (ENTRY_SIGMA, "--"), (-ENTRY_SIGMA, "--")]:
        ax.axhline(m + k * sigma, color="firebrick" if k else "black", ls=ls, lw=0.9)
    ax.set_title(f"(b) spread = logXRP − β·logBTC − α  (mean ±{ENTRY_SIGMA:.0f}σ)")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"xrpbtc_spread_{d}.png", dpi=110)
    plt.close(fig)

    # (c) rolling EG p-value
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll_p.index, roll_p.values, lw=0.8, color="darkgreen")
    ax.axhline(ROLLING_EG_P_MAX, color="firebrick", ls="--", lw=0.9, label=f"p={ROLLING_EG_P_MAX}")
    ax.set_ylim(0, 1)
    ax.set_title(f"(c) rolling EG p-value ({ROLLING_WINDOW_DAYS}일 윈도우)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / f"xrpbtc_rolling_{d}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="XRP/BTC stat-arb 적합성 precheck")
    ap.add_argument("--symbol", default="XRPUSDT")
    ap.add_argument("--sanity", action="store_true", help="데이터+적분차수+self-test만")
    args = ap.parse_args()
    run_precheck(symbol=args.symbol, sanity_only=args.sanity)


if __name__ == "__main__":
    main()
