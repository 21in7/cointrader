"""
Trade History + L/S Ratio 종합 분석
- 봇 대시보드 API에서 거래 기록 로드
- Binance API에서 L/S ratio (30일) 로드 + 로컬 parquet 병합
- 진입/청산 시점 L/S ratio 매칭
- 수익/손실 거래별 L/S 분포 분석
- L/S 임계값 필터링 시뮬레이션

Usage: python scripts/trade_ls_analysis.py [--api URL]
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json

BASE = "https://fapi.binance.com"
DASHBOARD_API = "http://10.1.10.24:8080/api/trades"
DATA_DIR = Path("data")
SYMBOLS_FOR_LS = ["XRPUSDT", "BTCUSDT", "ETHUSDT"]


async def fetch_json(session, url, params=None):
    async with session.get(url, params=params) as resp:
        return await resp.json()


async def fetch_ls_ratios_from_api(session, symbol, start_ms, end_ms):
    """Binance API에서 L/S ratio 전체 기간 가져오기 (페이징)"""
    all_top_acct = []
    all_global = []

    for endpoint, target in [
        (f"{BASE}/futures/data/topLongShortAccountRatio", all_top_acct),
        (f"{BASE}/futures/data/globalLongShortAccountRatio", all_global),
    ]:
        current = start_ms
        while current < end_ms:
            params = {
                "symbol": symbol,
                "period": "15m",
                "startTime": current,
                "endTime": end_ms,
                "limit": 500,
            }
            data = await fetch_json(session, endpoint, params)
            if not data or not isinstance(data, list):
                break
            target.extend(data)
            last_ts = int(data[-1]["timestamp"])
            if last_ts <= current:
                break
            current = last_ts + 1

    def to_df(data, col_name):
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df[col_name] = df["longShortRatio"].astype(float)
        return df[["timestamp", col_name]].drop_duplicates("timestamp")

    df_top = to_df(all_top_acct, "top_acct_ls_ratio")
    df_global = to_df(all_global, "global_ls_ratio")

    if df_top.empty and df_global.empty:
        return pd.DataFrame()

    if df_top.empty:
        return df_global
    if df_global.empty:
        return df_top

    return df_top.merge(df_global, on="timestamp", how="outer").sort_values("timestamp")


def load_local_ls_ratio(symbol):
    """로컬 parquet에서 L/S ratio 로드"""
    path = DATA_DIR / symbol.lower() / "ls_ratio_15m.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def find_nearest_ls(ls_df, target_time, max_gap_minutes=30):
    """타겟 시간에 가장 가까운 L/S ratio 찾기"""
    if ls_df.empty:
        return None, None, None

    target = pd.Timestamp(target_time, tz="UTC")
    diffs = (ls_df["timestamp"] - target).abs()
    idx = diffs.idxmin()
    gap = diffs[idx]

    if gap > pd.Timedelta(minutes=max_gap_minutes):
        return None, None, gap

    row = ls_df.loc[idx]
    return row.get("top_acct_ls_ratio"), row.get("global_ls_ratio"), gap


def classify_signal(trade):
    """진입 신호 분류"""
    rsi = trade.get("rsi", 0)
    macd = trade.get("macd_hist", 0)
    direction = trade["direction"]

    signals = []
    if direction == "LONG":
        if rsi and rsi > 65:
            signals.append("RSI과매수진입")
        elif rsi and rsi < 35:
            signals.append("RSI역방향")
        if macd and macd > 0:
            signals.append("MACD+")
        elif macd and macd < 0:
            signals.append("MACD역방향")
    else:  # SHORT
        if rsi and rsi < 35:
            signals.append("RSI과매도진입")
        elif rsi and rsi > 65:
            signals.append("RSI역방향")
        if macd and macd < 0:
            signals.append("MACD-")
        elif macd and macd > 0:
            signals.append("MACD역방향")

    return ", ".join(signals) if signals else "복합신호"


def classify_close_reason(trade):
    """청산 이유 분류"""
    reason = trade["close_reason"]
    if reason == "TP":
        return "TP(익절)"
    elif reason == "SYNC":
        return "SL(손절)"
    elif reason == "MANUAL":
        # MANUAL인데 SL가격과 exit가격이 같으면 SL
        sl = trade.get("sl")
        exit_p = trade.get("exit_price")
        if sl and exit_p and abs(float(sl) - float(exit_p)) < 0.0001:
            return "SL(손절)"
        # 역방향 시그널로 청산
        extra = trade.get("extra", "{}")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        if extra.get("recovery"):
            return "신호반전"
        return "SL(손절)"  # 대부분 MANUAL은 SL 히트
    return reason


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=DASHBOARD_API)
    args = parser.parse_args()

    print("=" * 80)
    print("  Trade History + L/S Ratio 종합 분석")
    print("=" * 80)

    # 1. 거래 데이터 로드
    async with aiohttp.ClientSession() as session:
        trade_data = await fetch_json(session, args.api)
        trades = trade_data["trades"]
        print(f"\n📊 거래 데이터: {len(trades)}건 로드")

        # 2. L/S ratio 데이터 로드 (API + local)
        # 가장 오래된 거래 기준으로 시작 시간 설정
        earliest = min(t["entry_time"] for t in trades)
        start_dt = pd.Timestamp(earliest, tz="UTC") - timedelta(hours=1)
        end_dt = datetime.now(timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        print(f"📡 Binance API에서 L/S ratio 로딩 ({start_dt.date()} ~ {end_dt.date()})...")

        ls_data = {}
        for sym in SYMBOLS_FOR_LS:
            api_df = await fetch_ls_ratios_from_api(session, sym, start_ms, end_ms)
            local_df = load_local_ls_ratio(sym)

            if not api_df.empty and not local_df.empty:
                combined = pd.concat([api_df, local_df]).drop_duplicates("timestamp").sort_values("timestamp")
            elif not api_df.empty:
                combined = api_df
            elif not local_df.empty:
                combined = local_df
            else:
                combined = pd.DataFrame()

            ls_data[sym] = combined.reset_index(drop=True)
            print(f"  {sym}: {len(ls_data[sym])} rows ({ls_data[sym]['timestamp'].min()} ~ {ls_data[sym]['timestamp'].max()})" if not combined.empty else f"  {sym}: no data")

    # 3. 거래별 L/S ratio 매칭
    print("\n" + "=" * 80)
    print("  1. 거래 기록 + L/S Ratio 매칭")
    print("=" * 80)

    enriched = []
    for t in trades:
        sym = t["symbol"]
        # XRP 거래에는 XRP L/S, 다른 심볼도 XRP L/S 참조 (크로스 분석)
        ls_sym = ls_data.get(sym, pd.DataFrame())
        ls_xrp = ls_data.get("XRPUSDT", pd.DataFrame())
        ls_btc = ls_data.get("BTCUSDT", pd.DataFrame())

        entry_top, entry_global, _ = find_nearest_ls(ls_sym if not ls_sym.empty else ls_xrp, t["entry_time"])
        exit_top, exit_global, _ = find_nearest_ls(ls_sym if not ls_sym.empty else ls_xrp, t["exit_time"])

        # BTC L/S for cross-reference
        btc_entry_top, btc_entry_global, _ = find_nearest_ls(ls_btc, t["entry_time"])

        enriched.append({
            "id": t["id"],
            "symbol": sym,
            "direction": t["direction"],
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "signal": classify_signal(t),
            "close_reason": classify_close_reason(t),
            "rsi": t.get("rsi"),
            "macd_hist": t.get("macd_hist"),
            "entry_top_acct_ls": entry_top,
            "entry_global_ls": entry_global,
            "exit_top_acct_ls": exit_top,
            "exit_global_ls": exit_global,
            "ls_change_top": (exit_top - entry_top) if entry_top and exit_top else None,
            "ls_change_global": (exit_global - entry_global) if entry_global and exit_global else None,
            "btc_entry_top_ls": btc_entry_top,
            "btc_entry_global_ls": btc_entry_global,
            "net_pnl": t["net_pnl"],
            "is_win": t["net_pnl"] > 0,
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
        })

    df = pd.DataFrame(enriched)

    # 거래 기록 테이블 출력
    print(f"\n{'ID':>3} {'심볼':<10} {'방향':<5} {'진입시간':<20} {'진입신호':<16} "
          f"{'진입L/S':>7} {'청산L/S':>7} {'ΔL/S':>7} {'청산이유':<10} {'PnL':>8}")
    print("-" * 120)
    for _, r in df.iterrows():
        entry_ls = f"{r['entry_top_acct_ls']:.3f}" if pd.notna(r['entry_top_acct_ls']) else "N/A"
        exit_ls = f"{r['exit_top_acct_ls']:.3f}" if pd.notna(r['exit_top_acct_ls']) else "N/A"
        delta_ls = f"{r['ls_change_top']:+.3f}" if pd.notna(r['ls_change_top']) else "N/A"
        pnl_str = f"{r['net_pnl']:+.4f}"
        print(f"{r['id']:>3} {r['symbol']:<10} {r['direction']:<5} {r['entry_time']:<20} {r['signal']:<16} "
              f"{entry_ls:>7} {exit_ls:>7} {delta_ls:>7} {r['close_reason']:<10} {pnl_str:>8}")

    # 4. 수익 거래 vs 손실 거래 L/S 비교
    print("\n" + "=" * 80)
    print("  2. 수익 거래 vs 손실 거래: L/S Ratio 비교")
    print("=" * 80)

    has_ls = df.dropna(subset=["entry_top_acct_ls"])
    if len(has_ls) > 0:
        wins = has_ls[has_ls["is_win"]]
        losses = has_ls[~has_ls["is_win"]]

        print(f"\n  L/S ratio 매칭된 거래: {len(has_ls)}건 (수익: {len(wins)}, 손실: {len(losses)})")
        print(f"\n  {'지표':<30} {'수익 거래':>12} {'손실 거래':>12} {'차이':>10}")
        print("  " + "-" * 70)

        for col, label in [
            ("entry_top_acct_ls", "진입 시 top_acct L/S"),
            ("entry_global_ls", "진입 시 global L/S"),
            ("exit_top_acct_ls", "청산 시 top_acct L/S"),
            ("exit_global_ls", "청산 시 global L/S"),
            ("ls_change_top", "진입→청산 ΔL/S (top)"),
            ("ls_change_global", "진입→청산 ΔL/S (global)"),
            ("btc_entry_top_ls", "BTC 진입 시 top_acct L/S"),
        ]:
            w_vals = wins[col].dropna()
            l_vals = losses[col].dropna()
            if len(w_vals) > 0 and len(l_vals) > 0:
                w_mean = w_vals.mean()
                l_mean = l_vals.mean()
                diff = w_mean - l_mean
                print(f"  {label:<30} {w_mean:>12.4f} {l_mean:>12.4f} {diff:>+10.4f}")
            else:
                w_str = f"{w_vals.mean():.4f}" if len(w_vals) > 0 else "N/A"
                l_str = f"{l_vals.mean():.4f}" if len(l_vals) > 0 else "N/A"
                print(f"  {label:<30} {w_str:>12} {l_str:>12} {'N/A':>10}")
    else:
        print("\n  ⚠️ L/S ratio 매칭 가능한 거래가 없습니다")

    # 5. 진입 시점 L/S와 거래 결과의 상관계수
    print("\n" + "=" * 80)
    print("  3. L/S Ratio ↔ PnL 상관계수")
    print("=" * 80)

    for col, label in [
        ("entry_top_acct_ls", "진입 top_acct L/S"),
        ("entry_global_ls", "진입 global L/S"),
        ("btc_entry_top_ls", "BTC 진입 top_acct L/S"),
        ("ls_change_top", "ΔL/S (top)"),
    ]:
        valid = df.dropna(subset=[col, "net_pnl"])
        if len(valid) >= 3:
            corr = valid[col].corr(valid["net_pnl"])
            print(f"  {label:<30} r = {corr:>+.4f}  (n={len(valid)})")
        else:
            print(f"  {label:<30} 데이터 부족 (n={len(valid)})")

    # 6. 방향별 분석 (LONG 진입 시 L/S 높으면? SHORT 진입 시 낮으면?)
    print("\n" + "=" * 80)
    print("  4. 방향별 L/S Ratio 분석")
    print("=" * 80)

    for direction in ["LONG", "SHORT"]:
        subset = has_ls[has_ls["direction"] == direction]
        if len(subset) == 0:
            continue
        print(f"\n  [{direction}] ({len(subset)}건)")
        wins_d = subset[subset["is_win"]]
        losses_d = subset[~subset["is_win"]]
        print(f"    수익: {len(wins_d)}건, 손실: {len(losses_d)}건")
        if len(subset) > 0:
            for col in ["entry_top_acct_ls", "entry_global_ls"]:
                vals = subset[col].dropna()
                if len(vals) > 0:
                    w = wins_d[col].dropna()
                    l = losses_d[col].dropna()
                    w_str = f"{w.mean():.4f}" if len(w) > 0 else "N/A"
                    l_str = f"{l.mean():.4f}" if len(l) > 0 else "N/A"
                    print(f"    {col}: 수익평균={w_str}, 손실평균={l_str}")

    # 7. 청산 이유별 L/S ratio 분포
    print("\n" + "=" * 80)
    print("  5. 청산 이유별 L/S Ratio 분포")
    print("=" * 80)

    for reason in df["close_reason"].unique():
        subset = has_ls[has_ls["close_reason"] == reason]
        if len(subset) == 0:
            continue
        print(f"\n  [{reason}] ({len(subset)}건)")
        for col, label in [("entry_top_acct_ls", "진입 L/S"), ("exit_top_acct_ls", "청산 L/S"), ("ls_change_top", "ΔL/S")]:
            vals = subset[col].dropna()
            if len(vals) > 0:
                print(f"    {label}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    # 8. L/S 임계값 필터링 시뮬레이션
    print("\n" + "=" * 80)
    print("  6. L/S 임계값 필터링 시뮬레이션")
    print("  '만약 L/S 조건으로 진입을 필터링했다면?'")
    print("=" * 80)

    if len(has_ls) > 0:
        # 시뮬레이션 1: top_acct L/S ratio 기준 필터
        print("\n  [A] top_acct_ls_ratio 임계값별 (LONG 진입 시 ratio > threshold)")
        print(f"  {'Threshold':>10} {'통과':>5} {'차단':>5} {'통과 PnL':>10} {'차단 PnL':>10} {'통과 승률':>10} {'원본 승률':>10}")
        print("  " + "-" * 70)

        longs = has_ls[has_ls["direction"] == "LONG"]
        shorts = has_ls[has_ls["direction"] == "SHORT"]
        all_wr = has_ls["is_win"].mean() * 100 if len(has_ls) > 0 else 0

        if len(longs) > 0:
            ls_vals = longs["entry_top_acct_ls"].dropna()
            if len(ls_vals) > 0:
                for pct in [0.25, 0.50, 0.75]:
                    threshold = ls_vals.quantile(pct)
                    passed = longs[longs["entry_top_acct_ls"] >= threshold]
                    blocked = longs[longs["entry_top_acct_ls"] < threshold]
                    p_pnl = passed["net_pnl"].sum()
                    b_pnl = blocked["net_pnl"].sum()
                    p_wr = passed["is_win"].mean() * 100 if len(passed) > 0 else 0
                    print(f"  {threshold:>10.4f} {len(passed):>5} {len(blocked):>5} "
                          f"{p_pnl:>+10.4f} {b_pnl:>+10.4f} {p_wr:>9.1f}% {all_wr:>9.1f}%")

        # 시뮬레이션 2: SHORT 진입 시 ratio < threshold
        print(f"\n  [B] top_acct_ls_ratio 임계값별 (SHORT 진입 시 ratio < threshold)")
        print(f"  {'Threshold':>10} {'통과':>5} {'차단':>5} {'통과 PnL':>10} {'차단 PnL':>10} {'통과 승률':>10}")
        print("  " + "-" * 70)

        if len(shorts) > 0:
            ls_vals = shorts["entry_top_acct_ls"].dropna()
            if len(ls_vals) > 0:
                for pct in [0.75, 0.50, 0.25]:
                    threshold = ls_vals.quantile(pct)
                    passed = shorts[shorts["entry_top_acct_ls"] <= threshold]
                    blocked = shorts[shorts["entry_top_acct_ls"] > threshold]
                    p_pnl = passed["net_pnl"].sum()
                    b_pnl = blocked["net_pnl"].sum()
                    p_wr = passed["is_win"].mean() * 100 if len(passed) > 0 else 0
                    print(f"  {threshold:>10.4f} {len(passed):>5} {len(blocked):>5} "
                          f"{p_pnl:>+10.4f} {b_pnl:>+10.4f} {p_wr:>9.1f}%")

        # 시뮬레이션 3: Momentum 전략 - L/S 방향과 같은 방향만 진입
        print(f"\n  [C] Momentum 필터: L/S ratio > 중앙값이면 LONG만, < 중앙값이면 SHORT만")
        if len(has_ls) > 0:
            median_ls = has_ls["entry_top_acct_ls"].median()
            momentum_filter = has_ls.apply(
                lambda r: (r["direction"] == "LONG" and r["entry_top_acct_ls"] >= median_ls) or
                          (r["direction"] == "SHORT" and r["entry_top_acct_ls"] < median_ls),
                axis=1
            )
            passed = has_ls[momentum_filter]
            blocked = has_ls[~momentum_filter]
            print(f"    중앙값: {median_ls:.4f}")
            print(f"    통과: {len(passed)}건, PnL합계: {passed['net_pnl'].sum():+.4f}, "
                  f"승률: {passed['is_win'].mean()*100:.1f}%")
            print(f"    차단: {len(blocked)}건, PnL합계: {blocked['net_pnl'].sum():+.4f}, "
                  f"승률: {blocked['is_win'].mean()*100:.1f}%")

        # 시뮬레이션 4: Contrarian 전략 - L/S 반대 방향만 진입
        print(f"\n  [D] Contrarian 필터: L/S ratio > 중앙값이면 SHORT만, < 중앙값이면 LONG만")
        if len(has_ls) > 0:
            contrarian_filter = has_ls.apply(
                lambda r: (r["direction"] == "SHORT" and r["entry_top_acct_ls"] >= median_ls) or
                          (r["direction"] == "LONG" and r["entry_top_acct_ls"] < median_ls),
                axis=1
            )
            passed = has_ls[contrarian_filter]
            blocked = has_ls[~contrarian_filter]
            print(f"    통과: {len(passed)}건, PnL합계: {passed['net_pnl'].sum():+.4f}, "
                  f"승률: {passed['is_win'].mean()*100:.1f}%")
            print(f"    차단: {len(blocked)}건, PnL합계: {blocked['net_pnl'].sum():+.4f}, "
                  f"승률: {blocked['is_win'].mean()*100:.1f}%")

    # 9. 전체 L/S ratio 시계열 + 거래 오버레이 요약
    print("\n" + "=" * 80)
    print("  7. 전체 요약")
    print("=" * 80)

    total_trades = len(df)
    ls_matched = len(has_ls)
    total_pnl = df["net_pnl"].sum()
    win_rate = df["is_win"].mean() * 100
    print(f"\n  전체 거래: {total_trades}건 (L/S 매칭: {ls_matched}건)")
    print(f"  총 PnL: {total_pnl:+.4f} USDT")
    print(f"  승률: {win_rate:.1f}% ({df['is_win'].sum()}/{total_trades})")

    if len(has_ls) > 0:
        print(f"\n  L/S 매칭 거래 통계:")
        print(f"    진입 top_acct L/S 범위: {has_ls['entry_top_acct_ls'].min():.4f} ~ {has_ls['entry_top_acct_ls'].max():.4f}")
        print(f"    진입 global L/S 범위: {has_ls['entry_global_ls'].min():.4f} ~ {has_ls['entry_global_ls'].max():.4f}")

    print(f"\n  ⚠️  주의: 거래 {total_trades}건은 통계적 유의성이 부족합니다.")
    print(f"  현재 결과는 탐색적 분석이며, 최소 50건 이상의 거래가 필요합니다.")
    print(f"  L/S ratio 데이터는 계속 축적 중이므로 4월 말 재분석을 권장합니다.")


if __name__ == "__main__":
    asyncio.run(main())
