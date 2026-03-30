"""
MTF Pullback Backtest
─────────────────────
Trigger: 1h 추세 방향으로 15m 눌림목 진입
  LONG:  1h Meta=LONG + 15m close < EMA20 + vol < SMA20*0.5 → 다음 봉 close > EMA20 시 진입
  SHORT: 1h Meta=SHORT + 15m close > EMA20 + vol < SMA20*0.5 → 다음 봉 close < EMA20 시 진입

SL/TP: 1h ATR 기반 (진입 시점 직전 완성된 1h 캔들)
Look-ahead bias 방지: 1h 지표는 직전 완성 봉만 사용
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ─── 설정 ────────────────────────────────────────────────────────
SYMBOL = "xrpusdt"
DATA_PATH = Path(f"data/{SYMBOL}/combined_15m.parquet")
START = "2026-02-01"
END = "2026-03-30"

ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.3
FEE_RATE = 0.0004  # 0.04% per side

# 1h 메타필터
MTF_EMA_FAST = 50
MTF_EMA_SLOW = 200
MTF_ADX_THRESHOLD = 20

# 15m Trigger
EMA_PULLBACK_LEN = 20
VOL_DRY_RATIO = 0.5  # volume < vol_ma20 * 0.5


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    sl: float
    tp: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None


def build_1h_data(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15m → 1h 리샘플링 + EMA50, EMA200, ADX, ATR."""
    df_1h = df_15m[["open", "high", "low", "close", "volume"]].resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    df_1h["ema50_1h"] = ta.ema(df_1h["close"], length=MTF_EMA_FAST)
    df_1h["ema200_1h"] = ta.ema(df_1h["close"], length=MTF_EMA_SLOW)
    adx_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["adx_1h"] = adx_df["ADX_14"]
    df_1h["atr_1h"] = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)

    return df_1h[["ema50_1h", "ema200_1h", "adx_1h", "atr_1h"]]


def merge_1h_to_15m(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Look-ahead bias 방지: 1h 봉 완성 시점(+1h) 기준 backward merge."""
    df_1h_shifted = df_1h.copy()
    df_1h_shifted.index = df_1h_shifted.index + pd.Timedelta(hours=1)

    df_15m_reset = df_15m.reset_index()
    df_1h_reset = df_1h_shifted.reset_index()
    df_1h_reset.rename(columns={"index": "timestamp"}, inplace=True)
    if "timestamp" not in df_15m_reset.columns:
        df_15m_reset.rename(columns={df_15m_reset.columns[0]: "timestamp"}, inplace=True)

    df_15m_reset["timestamp"] = pd.to_datetime(df_15m_reset["timestamp"]).astype("datetime64[us]")
    df_1h_reset["timestamp"] = pd.to_datetime(df_1h_reset["timestamp"]).astype("datetime64[us]")

    merged = pd.merge_asof(
        df_15m_reset.sort_values("timestamp"),
        df_1h_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")


def get_1h_meta(row) -> str:
    """1h 메타필터: EMA50/200 방향 + ADX > 20."""
    ema50 = row.get("ema50_1h")
    ema200 = row.get("ema200_1h")
    adx = row.get("adx_1h")

    if pd.isna(ema50) or pd.isna(ema200) or pd.isna(adx):
        return "HOLD"
    if adx < MTF_ADX_THRESHOLD:
        return "HOLD"
    if ema50 > ema200:
        return "LONG"
    elif ema50 < ema200:
        return "SHORT"
    return "HOLD"


def calc_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "pf": 0, "pnl_bps": 0, "max_dd_bps": 0,
                "avg_win_bps": 0, "avg_loss_bps": 0, "long_trades": 0, "short_trades": 0}

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "pf": round(pf, 2),
        "pnl_bps": round(sum(pnls) * 10000, 1),
        "max_dd_bps": round(max_dd * 10000, 1),
        "avg_win_bps": round(np.mean(wins) * 10000, 1) if wins else 0,
        "avg_loss_bps": round(np.mean(losses) * 10000, 1) if losses else 0,
        "long_trades": sum(1 for t in trades if t.side == "LONG"),
        "short_trades": sum(1 for t in trades if t.side == "SHORT"),
    }


def main():
    print("=" * 70)
    print("  MTF Pullback Backtest")
    print(f"  {SYMBOL.upper()} | {START} ~ {END}")
    print(f"  SL: 1h ATR×{ATR_SL_MULT} | TP: 1h ATR×{ATR_TP_MULT} | Fee: {FEE_RATE*100:.2f}%/side")
    print(f"  Pullback: EMA{EMA_PULLBACK_LEN} | Vol dry: <{VOL_DRY_RATIO*100:.0f}% of SMA20")
    print("=" * 70)

    # ── 데이터 로드 ──
    df_raw = pd.read_parquet(DATA_PATH)
    if df_raw.index.tz is not None:
        df_raw.index = df_raw.index.tz_localize(None)

    # 1h EMA200 워밍업 (200h = 800 bars)
    warmup_start = pd.Timestamp(START) - pd.Timedelta(hours=250)
    df_full = df_raw[df_raw.index >= warmup_start].copy()
    print(f"\n데이터: {len(df_full)} bars (워밍업 포함)")

    # ── 15m 지표: EMA20, vol_ma20 ──
    df_full["ema20"] = ta.ema(df_full["close"], length=EMA_PULLBACK_LEN)
    df_full["vol_ma20"] = ta.sma(df_full["volume"], length=20)

    # ── 1h 지표 ──
    df_1h = build_1h_data(df_full)
    print(f"1h 캔들: {len(df_1h)} bars")

    # ── 병합 ──
    df_merged = merge_1h_to_15m(df_full, df_1h)

    # ── 분석 기간 ──
    df = df_merged[(df_merged.index >= START) & (df_merged.index <= END)].copy()
    print(f"분석 기간: {len(df)} bars ({df.index.min()} ~ {df.index.max()})")

    # ── 신호 스캔 & 시뮬레이션 ──
    trades: list[Trade] = []
    in_trade = False
    current_trade: Trade | None = None
    pullback_ready = False  # 눌림 감지 상태
    pullback_side = ""

    # 디버그 카운터
    meta_long_count = 0
    meta_short_count = 0
    pullback_detected = 0
    entry_triggered = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── 기존 포지션 SL/TP 체크 ──
        if in_trade and current_trade is not None:
            hit_sl = False
            hit_tp = False

            if current_trade.side == "LONG":
                if row["low"] <= current_trade.sl:
                    hit_sl = True
                if row["high"] >= current_trade.tp:
                    hit_tp = True
            else:
                if row["high"] >= current_trade.sl:
                    hit_sl = True
                if row["low"] <= current_trade.tp:
                    hit_tp = True

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
                current_trade.pnl_pct = raw_pnl - FEE_RATE * 2
                trades.append(current_trade)
                in_trade = False
                current_trade = None

        # ── 포지션 중이면 새 진입 스킵 ──
        if in_trade:
            continue

        # NaN 체크
        if pd.isna(row.get("ema20")) or pd.isna(row.get("vol_ma20")) or pd.isna(row.get("atr_1h")):
            pullback_ready = False
            continue

        # ── Step 1: 1h Meta Filter ──
        meta = get_1h_meta(row)
        if meta == "LONG":
            meta_long_count += 1
        elif meta == "SHORT":
            meta_short_count += 1

        if meta == "HOLD":
            pullback_ready = False
            continue

        # ── Step 2: 눌림(Pullback) 감지 ──
        # 이전 봉이 눌림 조건을 충족했는지 확인
        if pullback_ready and pullback_side == meta:
            # ── Step 4: 추세 재개 확인 (현재 봉 close 기준) ──
            if pullback_side == "LONG" and row["close"] > row["ema20"]:
                # 진입: 이 봉의 open (추세 재개 확인된 봉)
                # 실제로는 close 시점에 확인하므로 다음 봉 open에 진입해야 look-ahead 방지
                # 하지만 사양서에 "직후 캔들의 종가가 EMA20 상향 돌파한 첫 번째 캔들의 시가"라고 되어 있으므로
                # → 이 봉(close > EMA20)의 open에서 진입은 look-ahead bias
                # → 정확히는: prev가 pullback, 현재 봉 close > EMA20 확인 → 다음 봉 open 진입
                # 여기서는 다음 봉 open으로 처리
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    entry_price = next_row["open"]
                    atr_1h = row["atr_1h"]

                    sl = entry_price - atr_1h * ATR_SL_MULT
                    tp = entry_price + atr_1h * ATR_TP_MULT

                    current_trade = Trade(
                        entry_time=df.index[i + 1],
                        entry_price=entry_price,
                        side="LONG",
                        sl=sl, tp=tp,
                    )
                    in_trade = True
                    pullback_ready = False
                    entry_triggered += 1
                    continue

            elif pullback_side == "SHORT" and row["close"] < row["ema20"]:
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    entry_price = next_row["open"]
                    atr_1h = row["atr_1h"]

                    sl = entry_price + atr_1h * ATR_SL_MULT
                    tp = entry_price - atr_1h * ATR_TP_MULT

                    current_trade = Trade(
                        entry_time=df.index[i + 1],
                        entry_price=entry_price,
                        side="SHORT",
                        sl=sl, tp=tp,
                    )
                    in_trade = True
                    pullback_ready = False
                    entry_triggered += 1
                    continue

        # ── Step 2+3: 눌림 + 거래량 고갈 감지 (다음 봉에서 재개 확인) ──
        vol_dry = row["volume"] < row["vol_ma20"] * VOL_DRY_RATIO

        if meta == "LONG" and row["close"] < row["ema20"] and vol_dry:
            pullback_ready = True
            pullback_side = "LONG"
            pullback_detected += 1
        elif meta == "SHORT" and row["close"] > row["ema20"] and vol_dry:
            pullback_ready = True
            pullback_side = "SHORT"
            pullback_detected += 1
        else:
            # 조건 불충족 시 pullback 상태 리셋
            # 단, 연속 pullback 허용 (여러 봉 동안 눌림 지속 가능)
            if not (meta == pullback_side):
                pullback_ready = False

    # ── 결과 출력 ──
    m = calc_metrics(trades)
    long_trades = [t for t in trades if t.side == "LONG"]
    short_trades = [t for t in trades if t.side == "SHORT"]
    lm = calc_metrics(long_trades)
    sm = calc_metrics(short_trades)

    print(f"\n─── 신호 파이프라인 ───")
    print(f"1h Meta LONG:  {meta_long_count} bars | SHORT: {meta_short_count} bars")
    print(f"Pullback 감지: {pullback_detected}건")
    print(f"진입 트리거:   {entry_triggered}건")
    print(f"실제 거래:     {m['trades']}건 (L:{m['long_trades']} / S:{m['short_trades']})")

    print(f"\n{'=' * 70}")
    print(f"  결과")
    print(f"{'=' * 70}")

    header = f"{'구분':<10} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'PnL(bps)':>10} {'MaxDD(bps)':>11} {'AvgWin':>8} {'AvgLoss':>8}"
    print(header)
    print("-" * len(header))
    print(f"{'전체':<10} {m['trades']:>7} {m['win_rate']:>7.1f}% {m['pf']:>6.2f} {m['pnl_bps']:>10.1f} {m['max_dd_bps']:>11.1f} {m['avg_win_bps']:>8.1f} {m['avg_loss_bps']:>8.1f}")
    print(f"{'LONG':<10} {lm['trades']:>7} {lm['win_rate']:>7.1f}% {lm['pf']:>6.2f} {lm['pnl_bps']:>10.1f} {lm['max_dd_bps']:>11.1f} {lm['avg_win_bps']:>8.1f} {lm['avg_loss_bps']:>8.1f}")
    print(f"{'SHORT':<10} {sm['trades']:>7} {sm['win_rate']:>7.1f}% {sm['pf']:>6.2f} {sm['pnl_bps']:>10.1f} {sm['max_dd_bps']:>11.1f} {sm['avg_win_bps']:>8.1f} {sm['avg_loss_bps']:>8.1f}")

    # 개별 거래 목록
    if trades:
        print(f"\n─── 개별 거래 ───")
        print(f"{'#':>3} {'Side':<6} {'Entry Time':<20} {'Entry':>10} {'Exit':>10} {'PnL(bps)':>10} {'Result':>8}")
        print("-" * 75)
        for idx, t in enumerate(trades, 1):
            result = "WIN" if t.pnl_pct > 0 else "LOSS"
            pnl_bps = t.pnl_pct * 10000
            print(f"{idx:>3} {t.side:<6} {str(t.entry_time):<20} {t.entry_price:>10.4f} {t.exit_price:>10.4f} {pnl_bps:>+10.1f} {result:>8}")


if __name__ == "__main__":
    main()
