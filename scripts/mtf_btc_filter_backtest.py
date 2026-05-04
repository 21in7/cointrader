"""
MTF Pullback + BTC 추세 필터 백테스트
──────────────────────────────────────
기존 MTF Pullback 전략에 BTC 추세 필터를 추가하여 검증.

메인 가설 (사전 확정):
  BTC 1h + EMA 50/200 + ADX > 20
  sweep 결과와 무관하게 사후 변경하지 않음.

판정 흐름:
  1. 베이스라인(필터 없음) IS/OOS 결과 산출
  2. 12개 sweep IS, 메인 가설 OOS 검증
  3. 나머지 11개 OOS robustness 체크

Usage:
    python scripts/mtf_btc_filter_backtest.py
    python scripts/mtf_btc_filter_backtest.py --symbol xrpusdt
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
import pandas_ta as ta
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import COST_MODEL, COST_SCENARIOS  # noqa: E402

# ─── 설정 ──────────────────────────────────────────────────────────
SYMBOL = "xrpusdt"
DATA_PATH = Path(f"data/{SYMBOL}/combined_15m.parquet")

# XRP 1h 메타필터 (기존 MTF bot 설정 그대로)
MTF_EMA_FAST = 50
MTF_EMA_SLOW = 200
MTF_ADX_THRESHOLD = 20

# 15m Trigger
EMA_PULLBACK_LEN = 20
VOL_DRY_RATIO = 0.5

# SL/TP
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.3

# IS/OOS 분할
IS_RATIO = 0.7

# ─── Sweep 그리드 ─────────────────────────────────────────────────
SWEEP_GRID = {
    "btc_tf": ["1h", "4h", "1D"],
    "btc_ema_fast": [20, 50],
    "btc_ema_slow": [100, 200],
}

# BTC ADX 임계값 — 전 조합 고정
BTC_ADX_THRESHOLD = 20

# 메인 가설 (사전 확정 — commitment device)
MAIN_HYPOTHESIS = {"btc_tf": "1h", "btc_ema_fast": 50, "btc_ema_slow": 200}

# IS 거래 수 최소 기준
MIN_IS_TRADES = 100


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    sl: float
    tp: float
    btc_trend: str = ""
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_bps: float | None = None
    reason: str = ""


def build_xrp_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """XRP 15m → 1h: EMA50, EMA200, ADX, ATR."""
    df_1h = df_15m[["open", "high", "low", "close", "volume"]].resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    df_1h["ema50_1h"] = ta.ema(df_1h["close"], length=MTF_EMA_FAST)
    df_1h["ema200_1h"] = ta.ema(df_1h["close"], length=MTF_EMA_SLOW)
    adx_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["adx_1h"] = adx_df["ADX_14"]
    df_1h["atr_1h"] = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)

    return df_1h[["ema50_1h", "ema200_1h", "adx_1h", "atr_1h"]]


def build_btc_resampled(df_15m: pd.DataFrame, tf: str, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    """BTC 15m → 지정 타임프레임: EMA + ADX."""
    btc_cols = {"open_btc": "open", "high_btc": "high", "low_btc": "low",
                "close_btc": "close", "volume_btc": "volume"}
    df_btc = df_15m[list(btc_cols.keys())].rename(columns=btc_cols)

    df_rs = df_btc.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    df_rs[f"btc_ema_fast"] = ta.ema(df_rs["close"], length=ema_fast)
    df_rs[f"btc_ema_slow"] = ta.ema(df_rs["close"], length=ema_slow)
    adx_df = ta.adx(df_rs["high"], df_rs["low"], df_rs["close"], length=14)
    df_rs["btc_adx"] = adx_df["ADX_14"]

    return df_rs[["btc_ema_fast", "btc_ema_slow", "btc_adx"]]


def merge_higher_tf(df_15m: pd.DataFrame, df_htf: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Look-ahead bias 방지 merge. 1h → +1h shift, 4h → +4h shift, 1d → +1d shift."""
    shift_map = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
                 "1D": pd.Timedelta(days=1)}
    df_shifted = df_htf.copy()
    df_shifted.index = df_shifted.index + shift_map[tf]

    df_15m_r = df_15m.reset_index()
    df_htf_r = df_shifted.reset_index()
    ts_col_15m = df_15m_r.columns[0]
    ts_col_htf = df_htf_r.columns[0]
    df_15m_r.rename(columns={ts_col_15m: "timestamp"}, inplace=True)
    df_htf_r.rename(columns={ts_col_htf: "timestamp"}, inplace=True)

    df_15m_r["timestamp"] = pd.to_datetime(df_15m_r["timestamp"]).astype("datetime64[us]")
    df_htf_r["timestamp"] = pd.to_datetime(df_htf_r["timestamp"]).astype("datetime64[us]")

    merged = pd.merge_asof(
        df_15m_r.sort_values("timestamp"),
        df_htf_r.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")


def get_xrp_meta(row) -> str:
    """XRP 1h 메타필터."""
    ema50 = row.get("ema50_1h")
    ema200 = row.get("ema200_1h")
    adx = row.get("adx_1h")
    if pd.isna(ema50) or pd.isna(ema200) or pd.isna(adx):
        return "HOLD"
    if adx < MTF_ADX_THRESHOLD:
        return "HOLD"
    return "LONG" if ema50 > ema200 else "SHORT"


def get_btc_trend(row) -> str:
    """BTC 추세 필터."""
    ema_f = row.get("btc_ema_fast")
    ema_s = row.get("btc_ema_slow")
    adx = row.get("btc_adx")
    if pd.isna(ema_f) or pd.isna(ema_s) or pd.isna(adx):
        return "NEUTRAL"
    if adx < BTC_ADX_THRESHOLD:
        return "NEUTRAL"
    return "UP" if ema_f > ema_s else "DOWN"


def run_backtest(df: pd.DataFrame, use_btc_filter: bool) -> list[Trade]:
    """MTF Pullback 백테스트 실행."""
    trades: list[Trade] = []
    in_trade = False
    current_trade: Trade | None = None
    pullback_ready = False
    pullback_side = ""

    for i in range(1, len(df)):
        row = df.iloc[i]

        # ── SL/TP 체크 ──
        if in_trade and current_trade is not None:
            hit_sl = hit_tp = False
            if current_trade.side == "LONG":
                hit_sl = row["low"] <= current_trade.sl
                hit_tp = row["high"] >= current_trade.tp
            else:
                hit_sl = row["high"] >= current_trade.sl
                hit_tp = row["low"] <= current_trade.tp

            if hit_sl or hit_tp:
                exit_price = current_trade.sl if hit_sl else current_trade.tp
                if hit_sl and hit_tp:
                    exit_price = current_trade.sl  # 보수적

                if current_trade.side == "LONG":
                    raw_pnl = (exit_price - current_trade.entry_price) / current_trade.entry_price
                else:
                    raw_pnl = (current_trade.entry_price - exit_price) / current_trade.entry_price

                current_trade.exit_time = df.index[i]
                current_trade.exit_price = exit_price
                current_trade.pnl_bps = raw_pnl * 10000  # raw bps (비용 미반영)
                current_trade.reason = "SL" if hit_sl else "TP"
                trades.append(current_trade)
                in_trade = False
                current_trade = None

        if in_trade:
            continue

        # NaN 체크
        if pd.isna(row.get("ema20")) or pd.isna(row.get("vol_ma20")) or pd.isna(row.get("atr_1h")):
            pullback_ready = False
            continue

        # ── XRP 1h Meta ──
        meta = get_xrp_meta(row)
        if meta == "HOLD":
            pullback_ready = False
            continue

        # ── BTC 추세 필터 ──
        btc_trend = get_btc_trend(row) if use_btc_filter else "DISABLED"

        if use_btc_filter:
            # BTC UP → LONG만, BTC DOWN → SHORT만, NEUTRAL → 차단
            if btc_trend == "UP" and meta != "LONG":
                pullback_ready = False
                continue
            elif btc_trend == "DOWN" and meta != "SHORT":
                pullback_ready = False
                continue
            elif btc_trend == "NEUTRAL":
                pullback_ready = False
                continue

        # ── Pullback 감지 → 재개 확인 ──
        if pullback_ready and pullback_side == meta:
            if pullback_side == "LONG" and row["close"] > row["ema20"]:
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    entry_price = next_row["open"]
                    atr = row["atr_1h"]
                    current_trade = Trade(
                        entry_time=df.index[i + 1], entry_price=entry_price,
                        side="LONG", sl=entry_price - atr * ATR_SL_MULT,
                        tp=entry_price + atr * ATR_TP_MULT, btc_trend=btc_trend,
                    )
                    in_trade = True
                    pullback_ready = False
                    continue

            elif pullback_side == "SHORT" and row["close"] < row["ema20"]:
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    entry_price = next_row["open"]
                    atr = row["atr_1h"]
                    current_trade = Trade(
                        entry_time=df.index[i + 1], entry_price=entry_price,
                        side="SHORT", sl=entry_price + atr * ATR_SL_MULT,
                        tp=entry_price - atr * ATR_TP_MULT, btc_trend=btc_trend,
                    )
                    in_trade = True
                    pullback_ready = False
                    continue

        # ── Pullback 감지 ──
        vol_dry = row["volume"] < row["vol_ma20"] * VOL_DRY_RATIO
        if meta == "LONG" and row["close"] < row["ema20"] and vol_dry:
            pullback_ready = True
            pullback_side = "LONG"
        elif meta == "SHORT" and row["close"] > row["ema20"] and vol_dry:
            pullback_ready = True
            pullback_side = "SHORT"
        elif meta != pullback_side:
            pullback_ready = False

    return trades


def apply_cost(trades: list[Trade], scenario_name: str) -> list[float]:
    """거래 리스트에 비용 시나리오 적용, adjusted pnl_bps 리스트 반환."""
    scenario = COST_SCENARIOS[scenario_name]
    fee_per_side = COST_MODEL["taker_fee_bps"]  # 현재 전부 taker
    fee_roundtrip = fee_per_side * 2
    slippage_roundtrip = scenario["slippage_bps_per_side"] * 2

    adjusted = []
    for t in trades:
        # 펀딩비: 보유 시간 중 8h 경계 교차 수
        if t.entry_time is not None and t.exit_time is not None:
            dur_h = (t.exit_time - t.entry_time).total_seconds() / 3600
            funding_events = max(0, int(dur_h / 8))
        else:
            funding_events = 0
        funding_cost = funding_events * scenario["funding_bps_per_8h"]
        total_cost = fee_roundtrip + slippage_roundtrip + funding_cost
        adjusted.append(t.pnl_bps - total_cost)
    return adjusted


def calc_metrics(pnl_list: list[float]) -> dict:
    """pnl_bps 리스트로 메트릭 계산."""
    if not pnl_list:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "cum_pnl": 0.0, "avg_pnl": 0.0}

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": len(pnl_list),
        "win_rate": round(len(wins) / len(pnl_list) * 100, 1),
        "pf": round(pf, 2),
        "cum_pnl": round(sum(pnl_list), 1),
        "avg_pnl": round(sum(pnl_list) / len(pnl_list), 2),
    }


def split_is_oos(trades: list[Trade], split_ts: pd.Timestamp):
    """IS/OOS 분할."""
    is_trades = [t for t in trades if t.entry_time < split_ts]
    oos_trades = [t for t in trades if t.entry_time >= split_ts]
    return is_trades, oos_trades


def print_metrics_row(label: str, raw: dict, fees: dict, realistic: dict):
    """한 줄 메트릭 출력."""
    print(f"  {label:<8} {raw['trades']:>5}  {raw['win_rate']:>5.1f}%  "
          f"{raw['pf']:>5.2f}  {fees['pf']:>5.2f}  {realistic['pf']:>5.2f}  "
          f"{raw['cum_pnl']:>+8.1f}  {fees['cum_pnl']:>+8.1f}")


def print_section(title: str, trades: list[Trade]):
    """섹션별 메트릭 출력."""
    if not trades:
        print(f"\n  [{title}] 거래 없음")
        return

    raw_all = [t.pnl_bps for t in trades]
    fees_all = apply_cost(trades, "fees_only")
    real_all = apply_cost(trades, "realistic")

    long_t = [t for t in trades if t.side == "LONG"]
    short_t = [t for t in trades if t.side == "SHORT"]

    raw_l = [t.pnl_bps for t in long_t]
    raw_s = [t.pnl_bps for t in short_t]
    fees_l = apply_cost(long_t, "fees_only")
    fees_s = apply_cost(short_t, "fees_only")
    real_l = apply_cost(long_t, "realistic")
    real_s = apply_cost(short_t, "realistic")

    print(f"\n  [{title}]")
    print(f"  {'':8} {'N':>5}  {'WR':>6}  {'RawPF':>5}  {'FeePF':>5}  {'RealPF':>5}  {'RawPnL':>8}  {'FeePnL':>8}")
    print(f"  {'-'*62}")
    print_metrics_row("Total", calc_metrics(raw_all), calc_metrics(fees_all), calc_metrics(real_all))
    print_metrics_row("LONG", calc_metrics(raw_l), calc_metrics(fees_l), calc_metrics(real_l))
    print_metrics_row("SHORT", calc_metrics(raw_s), calc_metrics(fees_s), calc_metrics(real_s))


def save_trade_log(trades: list[Trade], filepath: Path, combo_label: str):
    """거래 수준 CSV 로그 저장."""
    rows = []
    for t in trades:
        rows.append({
            "combo": combo_label,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_bps": t.pnl_bps,
            "reason": t.reason,
            "btc_trend": t.btc_trend,
        })
    df = pd.DataFrame(rows)
    mode = "a" if filepath.exists() else "w"
    header = not filepath.exists()
    df.to_csv(filepath, mode=mode, header=header, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTF + BTC 추세 필터 백테스트")
    parser.add_argument("--symbol", default="xrpusdt")
    args = parser.parse_args()

    data_path = Path(f"data/{args.symbol}/combined_15m.parquet")
    print("=" * 72)
    print("  MTF Pullback + BTC 추세 필터 백테스트")
    print(f"  메인 가설: BTC {MAIN_HYPOTHESIS['btc_tf']} EMA{MAIN_HYPOTHESIS['btc_ema_fast']}/{MAIN_HYPOTHESIS['btc_ema_slow']} ADX>{BTC_ADX_THRESHOLD}")
    print("=" * 72)

    # ── 데이터 로드 ──
    df_raw = pd.read_parquet(data_path)
    if df_raw.index.tz is not None:
        df_raw.index = df_raw.index.tz_localize(None)

    # EMA200 워밍업 (200h × 4 + 여유 = 1000 bars)
    warmup_bars = 1000
    df_full = df_raw.iloc[warmup_bars:].copy() if len(df_raw) > warmup_bars else df_raw.copy()
    # 워밍업 포함 전체 데이터로 지표 계산
    df_calc = df_raw.copy()

    print(f"\n데이터: {len(df_raw)} bars total, 분석: {len(df_full)} bars")
    print(f"기간: {df_full.index[0]} ~ {df_full.index[-1]}")
    print(f"일수: {(df_full.index[-1] - df_full.index[0]).days}일")

    # ── IS/OOS 분할 ──
    split_idx = int(len(df_full) * IS_RATIO)
    split_ts = df_full.index[split_idx]
    print(f"IS/OOS 분할: IS ~{split_ts.date()} | OOS {split_ts.date()}~")

    # ── XRP 15m 지표 ──
    df_calc["ema20"] = ta.ema(df_calc["close"], length=EMA_PULLBACK_LEN)
    df_calc["vol_ma20"] = ta.sma(df_calc["volume"], length=20)

    # ── XRP 1h 지표 ──
    df_1h = build_xrp_1h(df_calc)
    df_merged_base = merge_higher_tf(df_calc, df_1h, "1h")

    # 분석 기간만 슬라이스
    df_analysis = df_merged_base[df_merged_base.index >= df_full.index[0]].copy()

    # ── 1. 베이스라인 (BTC 필터 없음) ──
    print("\n" + "=" * 72)
    print("  BASELINE (BTC 필터 없음)")
    print("=" * 72)

    baseline_trades = run_backtest(df_analysis, use_btc_filter=False)
    baseline_is, baseline_oos = split_is_oos(baseline_trades, split_ts)

    print_section("IS (베이스라인)", baseline_is)
    print_section("OOS (베이스라인)", baseline_oos)

    # ── 2. Sweep ──
    print("\n" + "=" * 72)
    print("  SWEEP (12개 조합)")
    print("=" * 72)

    combos = list(product(
        SWEEP_GRID["btc_tf"],
        SWEEP_GRID["btc_ema_fast"],
        SWEEP_GRID["btc_ema_slow"],
    ))

    trade_log_path = Path(f"results/{args.symbol}/mtf_btc_filter_trades.csv")
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    if trade_log_path.exists():
        trade_log_path.unlink()

    results = []

    for btc_tf, ema_f, ema_s in combos:
        if ema_f >= ema_s:
            continue  # fast >= slow는 무의미

        label = f"BTC_{btc_tf}_EMA{ema_f}/{ema_s}"
        is_main = (btc_tf == MAIN_HYPOTHESIS["btc_tf"] and
                   ema_f == MAIN_HYPOTHESIS["btc_ema_fast"] and
                   ema_s == MAIN_HYPOTHESIS["btc_ema_slow"])

        # BTC 지표 계산 + merge
        df_btc = build_btc_resampled(df_calc, btc_tf, ema_f, ema_s)
        df_with_btc = merge_higher_tf(df_analysis, df_btc, btc_tf)

        # 백테스트
        trades = run_backtest(df_with_btc, use_btc_filter=True)
        is_trades, oos_trades = split_is_oos(trades, split_ts)

        # IS 거래 수 체크
        if len(is_trades) < MIN_IS_TRADES:
            status = "SKIP(IS<100)"
        else:
            status = "MAIN" if is_main else "sweep"

        # 메트릭
        is_raw = calc_metrics([t.pnl_bps for t in is_trades])
        is_fees = calc_metrics(apply_cost(is_trades, "fees_only"))
        oos_raw = calc_metrics([t.pnl_bps for t in oos_trades])
        oos_fees = calc_metrics(apply_cost(oos_trades, "fees_only"))
        oos_real = calc_metrics(apply_cost(oos_trades, "realistic"))

        # LONG/SHORT 분리 (OOS)
        oos_long = [t for t in oos_trades if t.side == "LONG"]
        oos_short = [t for t in oos_trades if t.side == "SHORT"]
        oos_fees_l = calc_metrics(apply_cost(oos_long, "fees_only"))
        oos_fees_s = calc_metrics(apply_cost(oos_short, "fees_only"))

        results.append({
            "label": label, "is_main": is_main, "status": status,
            "is_trades": is_raw["trades"], "is_raw_pf": is_raw["pf"],
            "is_fees_pf": is_fees["pf"],
            "oos_trades": oos_raw["trades"], "oos_raw_pf": oos_raw["pf"],
            "oos_fees_pf": oos_fees["pf"], "oos_real_pf": oos_real["pf"],
            "oos_fees_pnl": oos_fees["cum_pnl"],
            "oos_long_fees_pf": oos_fees_l["pf"], "oos_short_fees_pf": oos_fees_s["pf"],
            "oos_long_n": oos_fees_l["trades"], "oos_short_n": oos_fees_s["trades"],
        })

        # 거래 로그 저장
        save_trade_log(trades, trade_log_path, label)

    # ── Sweep 결과 테이블 ──
    print(f"\n  {'Label':<22} {'St':>6} {'IS_N':>5} {'IS_FPF':>6} "
          f"{'OOS_N':>5} {'OOS_RPF':>7} {'OOS_FPF':>7} {'OOS_rPF':>7} "
          f"{'L_FPF':>6} {'S_FPF':>6}")
    print(f"  {'-'*92}")

    for r in results:
        marker = " ★" if r["is_main"] else ""
        print(f"  {r['label']:<22} {r['status']:>6} {r['is_trades']:>5} {r['is_fees_pf']:>6.2f} "
              f"{r['oos_trades']:>5} {r['oos_raw_pf']:>7.2f} {r['oos_fees_pf']:>7.2f} {r['oos_real_pf']:>7.2f} "
              f"{r['oos_long_fees_pf']:>6.2f} {r['oos_short_fees_pf']:>6.2f}{marker}")

    # ── 3. 메인 가설 상세 결과 ──
    main_result = next((r for r in results if r["is_main"]), None)
    if main_result is None:
        print("\n  [ERROR] 메인 가설 결과 없음")
        return

    print("\n" + "=" * 72)
    print(f"  메인 가설 상세: {main_result['label']}")
    print("=" * 72)

    # 메인 가설 재실행하여 상세 출력
    df_btc_main = build_btc_resampled(
        df_calc, MAIN_HYPOTHESIS["btc_tf"],
        MAIN_HYPOTHESIS["btc_ema_fast"], MAIN_HYPOTHESIS["btc_ema_slow"])
    df_main = merge_higher_tf(df_analysis, df_btc_main, MAIN_HYPOTHESIS["btc_tf"])
    main_trades = run_backtest(df_main, use_btc_filter=True)
    main_is, main_oos = split_is_oos(main_trades, split_ts)

    print_section("IS (메인 가설)", main_is)
    print_section("OOS (메인 가설)", main_oos)

    # ── 4. 판정 ──
    print("\n" + "=" * 72)
    print("  판정")
    print("=" * 72)

    # 베이스라인 비교
    bl_oos_fees = calc_metrics(apply_cost(baseline_oos, "fees_only"))
    main_oos_fees = calc_metrics(apply_cost(main_oos, "fees_only"))
    main_oos_real = calc_metrics(apply_cost(main_oos, "realistic"))

    print(f"\n  베이스라인 OOS fees_only PF: {bl_oos_fees['pf']:.2f} ({bl_oos_fees['trades']}건)")
    print(f"  메인 가설  OOS fees_only PF: {main_oos_fees['pf']:.2f} ({main_oos_fees['trades']}건)")
    print(f"  메인 가설  OOS realistic PF: {main_oos_real['pf']:.2f}")
    print(f"  개선폭: fees_only PF {main_oos_fees['pf'] - bl_oos_fees['pf']:+.2f}")

    # 합격 기준 체크
    checks = []
    checks.append(("OOS fees_only PF >= 1.2", main_oos_fees["pf"] >= 1.2, f"{main_oos_fees['pf']:.2f}"))
    checks.append(("OOS realistic PF >= 1.0", main_oos_real["pf"] >= 1.0, f"{main_oos_real['pf']:.2f}"))
    checks.append(("OOS 거래수 >= 50", main_oos_fees["trades"] >= 50, f"{main_oos_fees['trades']}"))

    # LONG/SHORT 대칭성
    oos_long_t = [t for t in main_oos if t.side == "LONG"]
    oos_short_t = [t for t in main_oos if t.side == "SHORT"]
    l_pf = calc_metrics(apply_cost(oos_long_t, "fees_only"))["pf"]
    s_pf = calc_metrics(apply_cost(oos_short_t, "fees_only"))["pf"]
    checks.append(("LONG/SHORT fees PF >= 0.8", l_pf >= 0.8 and s_pf >= 0.8, f"L:{l_pf:.2f} S:{s_pf:.2f}"))

    # IS/OOS 격차
    main_is_fees = calc_metrics(apply_cost(main_is, "fees_only"))
    if main_is_fees["pf"] > 0:
        gap = abs(main_oos_fees["pf"] - main_is_fees["pf"]) / main_is_fees["pf"]
    else:
        gap = 1.0
    checks.append(("IS/OOS PF 격차 < 30%", gap < 0.3, f"{gap*100:.1f}%"))

    # 베이스라인 대비 개선
    improvement = main_oos_fees["pf"] > bl_oos_fees["pf"]
    checks.append(("베이스라인 대비 개선", improvement,
                    f"{main_oos_fees['pf']:.2f} vs {bl_oos_fees['pf']:.2f}"))

    print(f"\n  합격 기준 체크:")
    all_pass = True
    for desc, passed, val in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {desc}: {val}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ★ [최종 판정: PASS] BTC 추세 필터가 유효합니다.")
    else:
        print("  ✗ [최종 판정: FAIL] BTC 추세 필터로도 기준 미달. MTF 전략 폐기.")

    # robustness 요약
    passing_combos = [r for r in results if r["status"] != "SKIP(IS<100)" and r["oos_fees_pf"] >= 1.2]
    print(f"\n  Robustness: {len(passing_combos)}/{len(results)} 조합이 OOS fees_only PF >= 1.2")

    print(f"\n  거래 로그: {trade_log_path}")
    print()


if __name__ == "__main__":
    main()
