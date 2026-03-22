"""
Taker Buy/Sell Ratio vs Next-Candle Price Change Correlation Analysis
- Taker Buy Ratio (from klines + Trading Data API)
- Long/Short Ratio (global)
- Top Trader Long/Short Ratio (accounts & positions)

Usage: python scripts/taker_ratio_analysis.py [SYMBOL1] [SYMBOL2] ...
Default: XRPUSDT BTCUSDT ETHUSDT
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys

BASE = "https://fapi.binance.com"
SYMBOLS = sys.argv[1:] if len(sys.argv) > 1 else ["XRPUSDT", "BTCUSDT", "ETHUSDT"]
INTERVAL = "15m"
DAYS = 30

async def fetch_json(session, url, params):
    async with session.get(url, params=params) as resp:
        return await resp.json()

async def fetch_klines(session, symbol, start_ms, end_ms):
    all_klines = []
    current = start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": current, "endTime": end_ms, "limit": 1500}
        data = await fetch_json(session, f"{BASE}/fapi/v1/klines", params)
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 1
    return all_klines

async def fetch_ratio(session, url, symbol):
    params = {"symbol": symbol, "period": INTERVAL, "limit": 500}
    data = await fetch_json(session, url, params)
    return data if isinstance(data, list) else []

async def analyze_symbol(session, symbol, start_ms, end_ms):
    """Fetch and analyze a single symbol"""
    klines, ls_ratio, top_acct, top_pos, taker = await asyncio.gather(
        fetch_klines(session, symbol, start_ms, end_ms),
        fetch_ratio(session, f"{BASE}/futures/data/globalLongShortAccountRatio", symbol),
        fetch_ratio(session, f"{BASE}/futures/data/topLongShortAccountRatio", symbol),
        fetch_ratio(session, f"{BASE}/futures/data/topLongShortPositionRatio", symbol),
        fetch_ratio(session, f"{BASE}/futures/data/takerlongshortRatio", symbol),
    )

    print(f"\n  {symbol}: Klines={len(klines)}, L/S={len(ls_ratio)}, TopAcct={len(top_acct)}, TopPos={len(top_pos)}, Taker={len(taker)}")

    # Build DataFrame
    df_k = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","ignore"
    ])
    df_k["open_time"] = pd.to_datetime(df_k["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","taker_buy_vol","taker_buy_quote_vol","quote_vol"]:
        df_k[c] = df_k[c].astype(float)

    df_k["kline_taker_buy_ratio"] = (df_k["taker_buy_vol"] / df_k["volume"]).replace([np.inf, -np.inf], np.nan)
    df_k["next_return"] = df_k["close"].shift(-1) / df_k["close"] - 1
    df_k["next_4_return"] = df_k["close"].shift(-4) / df_k["close"] - 1
    df_k = df_k.set_index("open_time")

    def join_ratio(data, col_name):
        if not data:
            return
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        if "buySellRatio" in df.columns:
            df["buySellRatio"] = df["buySellRatio"].astype(float)
            df["buyVol"] = df["buyVol"].astype(float)
            df["sellVol"] = df["sellVol"].astype(float)
            df = df.set_index("timestamp")
            df_k.update(df_k.join(df[["buySellRatio","buyVol","sellVol"]], how="left"))
            for c in ["buySellRatio","buyVol","sellVol"]:
                if c not in df_k.columns:
                    df_k[c] = np.nan
            joined = df_k.join(df[["buySellRatio","buyVol","sellVol"]], how="left", rsuffix="_new")
            for c in ["buySellRatio","buyVol","sellVol"]:
                if f"{c}_new" in joined.columns:
                    df_k[c] = joined[f"{c}_new"]
        else:
            df["longShortRatio"] = df["longShortRatio"].astype(float)
            df = df.set_index("timestamp").rename(columns={"longShortRatio": col_name})
            df_k[col_name] = df_k.join(df[[col_name]], how="left")[col_name]

    join_ratio(taker, "buySellRatio")
    join_ratio(ls_ratio, "global_ls_ratio")
    join_ratio(top_acct, "top_acct_ls_ratio")
    join_ratio(top_pos, "top_pos_ls_ratio")

    return df_k

def print_analysis(symbol, df_k):
    """Print analysis results for a symbol"""
    print("\n" + "="*70)
    print(f"{symbol} {INTERVAL} Taker/Ratio → Price Correlation Analysis ({DAYS} days klines, ~5 days ratios)")
    print("="*70)

    features = ["kline_taker_buy_ratio", "buySellRatio", "global_ls_ratio",
                 "top_acct_ls_ratio", "top_pos_ls_ratio"]
    available = [f for f in features if f in df_k.columns and df_k[f].notna().sum() > 20]

    # 1. Correlation
    print("\n[1] Pearson Correlation with Next-Candle Returns")
    print("-"*55)
    print(f"{'Feature':<25} {'next_15m':>12} {'next_1h':>12}")
    print("-"*55)
    for feat in available:
        c1 = df_k[feat].corr(df_k["next_return"])
        c4 = df_k[feat].corr(df_k["next_4_return"])
        print(f"{feat:<25} {c1:>12.4f} {c4:>12.4f}")

    # 2. Quintile - Taker
    print("\n[2] Taker Buy Ratio Quintile → Next Returns")
    print("-"*60)
    for ratio_col in ["kline_taker_buy_ratio", "buySellRatio"]:
        if ratio_col not in available:
            continue
        valid = df_k[[ratio_col, "next_return", "next_4_return"]].dropna()
        try:
            valid["quintile"] = pd.qcut(valid[ratio_col], 5, labels=["Q1(sell)","Q2","Q3","Q4","Q5(buy)"])
        except ValueError:
            continue
        print(f"\n  {ratio_col}:")
        print(f"  {'Quintile':<12} {'mean_ratio':>12} {'next_15m_bps':>14} {'next_1h_bps':>13} {'count':>7} {'win_rate':>10}")
        for q in ["Q1(sell)","Q2","Q3","Q4","Q5(buy)"]:
            grp = valid[valid["quintile"] == q]
            if len(grp) == 0:
                continue
            mr = grp[ratio_col].mean()
            r1 = grp["next_return"].mean() * 10000
            r4 = grp["next_4_return"].mean() * 10000
            wr = (grp["next_return"] > 0).mean() * 100
            print(f"  {q:<12} {mr:>12.4f} {r1:>14.2f} {r4:>13.2f} {len(grp):>7} {wr:>9.1f}%")

    # 3. Extreme analysis
    print("\n[3] Extreme Taker Buy Ratio Analysis (top/bottom 10%)")
    print("-"*60)
    for ratio_col in ["kline_taker_buy_ratio", "buySellRatio"]:
        if ratio_col not in available:
            continue
        valid = df_k[[ratio_col, "next_return", "next_4_return"]].dropna()
        p10 = valid[ratio_col].quantile(0.10)
        p90 = valid[ratio_col].quantile(0.90)
        bottom = valid[valid[ratio_col] <= p10]
        top = valid[valid[ratio_col] >= p90]
        mid = valid[(valid[ratio_col] > p10) & (valid[ratio_col] < p90)]

        print(f"\n  {ratio_col}:")
        print(f"  {'Group':<18} {'mean_ratio':>12} {'next_15m_bps':>14} {'next_1h_bps':>13} {'win_rate':>10} {'count':>7}")
        for name, grp in [("Bottom 10% (sell)", bottom), ("Middle 80%", mid), ("Top 10% (buy)", top)]:
            if len(grp) == 0:
                continue
            mr = grp[ratio_col].mean()
            r1 = grp["next_return"].mean() * 10000
            r4 = grp["next_4_return"].mean() * 10000
            wr = (grp["next_return"] > 0).mean() * 100
            print(f"  {name:<18} {mr:>12.4f} {r1:>14.2f} {r4:>13.2f} {wr:>9.1f}% {len(grp):>7}")

    # 4. L/S ratio quintile
    print("\n[4] Long/Short Ratio Quintile → Next Returns")
    print("-"*60)
    for ratio_col in ["global_ls_ratio", "top_acct_ls_ratio", "top_pos_ls_ratio"]:
        if ratio_col not in available:
            continue
        valid = df_k[[ratio_col, "next_return", "next_4_return"]].dropna()
        if len(valid) < 20:
            continue
        try:
            valid["quintile"] = pd.qcut(valid[ratio_col], 5, labels=["Q1(short)","Q2","Q3","Q4","Q5(long)"], duplicates="drop")
        except ValueError:
            continue
        print(f"\n  {ratio_col}:")
        print(f"  {'Quintile':<12} {'mean_ratio':>12} {'next_15m_bps':>14} {'next_1h_bps':>13} {'win_rate':>10} {'count':>7}")
        for q in valid["quintile"].cat.categories:
            grp = valid[valid["quintile"] == q]
            if len(grp) == 0:
                continue
            mr = grp[ratio_col].mean()
            r1 = grp["next_return"].mean() * 10000
            r4 = grp["next_4_return"].mean() * 10000
            wr = (grp["next_return"] > 0).mean() * 100
            print(f"  {q:<12} {mr:>12.4f} {r1:>14.2f} {r4:>13.2f} {wr:>9.1f}% {len(grp):>7}")

    # 5. Contrarian vs Momentum
    print("\n[5] Contrarian vs Momentum Signal Test")
    print("-"*60)
    for ratio_col, label in [("kline_taker_buy_ratio", "Taker Buy Ratio"),
                              ("global_ls_ratio", "Global L/S Ratio"),
                              ("top_acct_ls_ratio", "Top Trader Acct Ratio"),
                              ("top_pos_ls_ratio", "Top Trader Pos Ratio")]:
        if ratio_col not in available:
            continue
        valid = df_k[[ratio_col, "next_return", "next_4_return"]].dropna()
        median = valid[ratio_col].median()
        high = valid[valid[ratio_col] > median]
        low = valid[valid[ratio_col] <= median]
        h_wr = (high["next_return"] > 0).mean() * 100
        l_wr = (low["next_return"] > 0).mean() * 100
        h_r = high["next_return"].mean() * 10000
        l_r = low["next_return"].mean() * 10000
        signal = "Momentum" if h_r > l_r else "Contrarian"
        print(f"\n  {label}:")
        print(f"    Above median → next 15m: {h_r:+.2f} bps (win {h_wr:.1f}%)")
        print(f"    Below median → next 15m: {l_r:+.2f} bps (win {l_wr:.1f}%)")
        print(f"    → Signal type: {signal}")

    # 6. Stats
    print("\n[6] Feature Statistics Summary")
    print("-"*60)
    for feat in available:
        s = df_k[feat].dropna()
        print(f"  {feat}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}, n={len(s)}")
    print(f"\n  Total klines: {len(df_k)}")
    print(f"  Period: {df_k.index[0]} ~ {df_k.index[-1]}")

async def main():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching {DAYS} days of {INTERVAL} data for {', '.join(SYMBOLS)}...")

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[analyze_symbol(session, sym, start_ms, end_ms) for sym in SYMBOLS]
        )

    for sym, df in zip(SYMBOLS, results):
        print_analysis(sym, df)

    # Cross-symbol comparison
    if len(SYMBOLS) > 1:
        print("\n" + "="*70)
        print("CROSS-SYMBOL COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Symbol':<12} {'taker_buy→15m':>14} {'taker_buy→1h':>13} {'global_ls→1h':>13} {'top_acct→1h':>13} {'top_pos→1h':>12}")
        print("-"*78)
        for sym, df in zip(SYMBOLS, results):
            tb = df["kline_taker_buy_ratio"].corr(df["next_return"]) if "kline_taker_buy_ratio" in df.columns else float('nan')
            tb4 = df["kline_taker_buy_ratio"].corr(df["next_4_return"]) if "kline_taker_buy_ratio" in df.columns else float('nan')
            gl = df["global_ls_ratio"].corr(df["next_4_return"]) if "global_ls_ratio" in df.columns and df["global_ls_ratio"].notna().sum() > 20 else float('nan')
            ta = df["top_acct_ls_ratio"].corr(df["next_4_return"]) if "top_acct_ls_ratio" in df.columns and df["top_acct_ls_ratio"].notna().sum() > 20 else float('nan')
            tp = df["top_pos_ls_ratio"].corr(df["next_4_return"]) if "top_pos_ls_ratio" in df.columns and df["top_pos_ls_ratio"].notna().sum() > 20 else float('nan')
            print(f"{sym:<12} {tb:>14.4f} {tb4:>13.4f} {gl:>13.4f} {ta:>13.4f} {tp:>12.4f}")

if __name__ == "__main__":
    asyncio.run(main())
