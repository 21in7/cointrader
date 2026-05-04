"""
FR × OI 변화율 백테스트 — Phase 1: 12개 조합

신호: FR × OI변화율(1h) = funding_rate × oi_pct_change_4
- SHORT: 피처 >= threshold (롱 스퀴즈 전조)
- LONG: 피처 <= threshold (숏 스퀴즈 전조)
- 보유: 1h(4캔들) / 4h(16캔들)

Usage: python scripts/fr_oi_backtest.py
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://fapi.binance.com"
SYMBOL = "XRPUSDT"
DATA_DIR = Path("data/xrpusdt")
FEE_RATE = 0.0004


async def fetch_oi_history(session, symbol, start_ms, end_ms):
    all_data = []
    current = start_ms
    calls = 0
    while current < end_ms:
        params = {"symbol": symbol, "period": "15m", "startTime": current, "endTime": end_ms, "limit": 500}
        async with session.get(f"{BASE}/futures/data/openInterestHist", params=params) as resp:
            data = await resp.json()
        if not data or not isinstance(data, list):
            break
        all_data.extend(data)
        last_ts = int(data[-1]["timestamp"])
        if last_ts <= current:
            break
        current = last_ts + 1
        calls += 1
        if calls % 50 == 0:
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(0.1)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi_value"] = df["sumOpenInterestValue"].astype(float)
    return df[["timestamp", "oi_value"]].drop_duplicates("timestamp").sort_values("timestamp")


async def fetch_funding_rate(session, symbol, start_ms, end_ms):
    all_data = []
    current = start_ms
    while current < end_ms:
        params = {"symbol": symbol, "startTime": current, "endTime": end_ms, "limit": 1000}
        async with session.get(f"{BASE}/fapi/v1/fundingRate", params=params) as resp:
            data = await resp.json()
        if not data or not isinstance(data, list):
            break
        all_data.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= current:
            break
        current = last_ts + 1
        await asyncio.sleep(0.1)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp")


def run_backtest(df, feature_col, percentile, direction, hold_bars):
    threshold = df[feature_col].quantile(percentile / 100)
    trades = []
    i = 0
    while i < len(df) - hold_bars - 1:
        val = df.iloc[i][feature_col]
        if pd.isna(val):
            i += 1
            continue

        trigger = False
        if direction == "SHORT" and val >= threshold:
            trigger = True
        elif direction == "LONG" and val <= threshold:
            trigger = True

        if trigger:
            entry_idx = i + 1
            exit_idx = i + 1 + hold_bars - 1
            if exit_idx >= len(df):
                break
            entry_price = df.iloc[entry_idx]["open"]
            exit_price = df.iloc[exit_idx]["close"]

            if direction == "LONG":
                gross_return = (exit_price / entry_price) - 1
            else:
                gross_return = (entry_price / exit_price) - 1

            fee = FEE_RATE * 2
            net_return = gross_return - fee

            trades.append({
                "entry_time": df.iloc[entry_idx]["timestamp"],
                "exit_time": df.iloc[exit_idx]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "feature_val": val,
                "gross_return_bps": gross_return * 10000,
                "net_return_bps": net_return * 10000,
            })
            i = exit_idx + 1  # 포지션 종료 후 다음
        else:
            i += 1

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net_return_bps"] > 0]["net_return_bps"]
    losses = tdf[tdf["net_return_bps"] <= 0]["net_return_bps"]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    cum_pnl = tdf["net_return_bps"].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    return {
        "trades": len(tdf),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(tdf) * 100,
        "pf": pf,
        "total_pnl_bps": tdf["net_return_bps"].sum(),
        "avg_pnl_bps": tdf["net_return_bps"].mean(),
        "max_dd_bps": max_dd,
        "threshold": threshold,
        "df_trades": tdf,
    }


def confidence(n):
    if n < 20:
        return "🔴", "폐기"
    elif n < 50:
        return "🟡", "참고"
    else:
        return "🟢", "검토"


async def main():
    print("=" * 80)
    print("  FR × OI 변화율 백테스트 — Phase 1: 12개 조합")
    print("=" * 80)

    # 데이터 수집
    print("\n[1] 데이터 수집")
    df_kline = pd.read_parquet(DATA_DIR / "combined_15m.parquet")

    end_dt = datetime.now(timezone.utc)
    oi_start_dt = end_dt - timedelta(days=29)
    oi_start_ms = int(oi_start_dt.replace(microsecond=0, second=0).timestamp()) * 1000
    fr_start_ms = oi_start_ms
    end_ms = int(end_dt.replace(microsecond=0, second=0).timestamp()) * 1000

    async with aiohttp.ClientSession() as session:
        print("  OI 수집...")
        oi_df = await fetch_oi_history(session, SYMBOL, oi_start_ms, end_ms)
        print(f"  OI: {len(oi_df)} rows")
        print("  FR 수집...")
        fr_df = await fetch_funding_rate(session, SYMBOL, fr_start_ms, end_ms)
        print(f"  FR: {len(fr_df)} rows")

    # 병합
    print("\n[2] 데이터 병합")
    df = df_kline.loc[oi_start_dt:].copy().reset_index()
    print(f"  Kline (29일): {len(df)} rows")

    # OI 병합
    df = pd.merge_asof(df.sort_values("timestamp"), oi_df.sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=20))
    df["oi_pct_change_4"] = df["oi_value"].pct_change(4)

    # FR 병합 (forward fill)
    df = pd.merge_asof(df.sort_values("timestamp"), fr_df.rename(columns={"funding_rate": "fr_api"}).sort_values("timestamp"),
                       on="timestamp", direction="backward")

    # 핵심 피처: FR × OI변화율(1h)
    df["fr_x_oi_1h"] = df["fr_api"] * df["oi_pct_change_4"]

    valid = df.dropna(subset=["fr_x_oi_1h"])
    print(f"  유효 데이터: {len(valid)} rows")
    print(f"  fr_x_oi_1h: mean={valid['fr_x_oi_1h'].mean():.8f}, std={valid['fr_x_oi_1h'].std():.8f}")

    for p in [25, 50, 75]:
        v = valid["fr_x_oi_1h"].quantile(p / 100)
        print(f"  P{p}: {v:.8f}")

    # 12개 조합 백테스트
    print("\n[3] 12개 조합 백테스트")
    print("=" * 80)

    combos = []
    for hold_label, hold_bars in [("1h", 4), ("4h", 16)]:
        for direction in ["SHORT", "LONG"]:
            for pct in [75, 50, 25]:
                desc_dir = "롱스퀴즈" if direction == "SHORT" else "숏스퀴즈"
                combos.append({
                    "hold_label": hold_label,
                    "hold_bars": hold_bars,
                    "direction": direction,
                    "percentile": pct,
                    "desc": f"{direction} {hold_label} P{pct} ({desc_dir})",
                })

    results = []
    for c in combos:
        r = run_backtest(valid.reset_index(drop=True), "fr_x_oi_1h",
                         c["percentile"], c["direction"], c["hold_bars"])
        if r:
            r.update(c)
        else:
            r = {**c, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                 "pf": 0, "total_pnl_bps": 0, "avg_pnl_bps": 0, "max_dd_bps": 0, "threshold": 0}
        results.append(r)

    # 결과 테이블
    print(f"\n{'ID':>3} {'조합':<28} {'거래수':>6} {'승률':>7} {'PF':>7} {'PnL(bps)':>10} {'MaxDD':>10} {'신뢰도'}")
    print("-" * 90)

    for i, r in enumerate(results, 1):
        emoji, label = confidence(r["trades"])
        pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "INF"
        print(f"{i:>3} {r['desc']:<28} {r['trades']:>6} {r['win_rate']:>6.1f}% {pf_str:>7} "
              f"{r['total_pnl_bps']:>+10.1f} {r['max_dd_bps']:>10.1f} {emoji} {label}")

    # 대칭성 검증
    print("\n" + "=" * 80)
    print("  [대칭성 검증]")
    print("=" * 80)

    for hold_label in ["1h", "4h"]:
        shorts = [r for r in results if r["hold_label"] == hold_label and r["direction"] == "SHORT" and r["trades"] > 0]
        longs = [r for r in results if r["hold_label"] == hold_label and r["direction"] == "LONG" and r["trades"] > 0]

        best_short = max(shorts, key=lambda x: x["pf"]) if shorts else None
        best_long = max(longs, key=lambda x: x["pf"]) if longs else None

        print(f"\n  [{hold_label} 보유]")
        if best_short:
            print(f"    Best SHORT: {best_short['desc']} — PF={best_short['pf']:.2f}, {best_short['trades']}건")
        if best_long:
            print(f"    Best LONG:  {best_long['desc']} — PF={best_long['pf']:.2f}, {best_long['trades']}건")

        if best_short and best_long:
            s_pf = best_short["pf"]
            l_pf = best_long["pf"]
            if s_pf > 1.5 and l_pf > 1.5:
                print(f"    → Case 1: 양방향 생존 ✓ Phase 2 후보")
            elif (s_pf > 1.5 and l_pf < 0.5) or (l_pf > 1.5 and s_pf < 0.5):
                print(f"    → Case 2: 한쪽만 성공 ✗ 시장 베타/우연")
            elif s_pf > 1.5 or l_pf > 1.5:
                print(f"    → Case 3: 부분적 edge ~ 낮은 신뢰도")
            elif s_pf > 1.0 and l_pf > 1.0:
                print(f"    → 양쪽 PF > 1.0이나 < 1.5 — 약한 edge")
            else:
                print(f"    → 양쪽 모두 약함")

    # 보유시간 비교
    print("\n" + "=" * 80)
    print("  [보유시간 비교]")
    print("=" * 80)

    for direction in ["SHORT", "LONG"]:
        r_1h = [r for r in results if r["hold_label"] == "1h" and r["direction"] == direction and r["trades"] > 0]
        r_4h = [r for r in results if r["hold_label"] == "4h" and r["direction"] == direction and r["trades"] > 0]
        best_1h = max(r_1h, key=lambda x: x["pf"]) if r_1h else None
        best_4h = max(r_4h, key=lambda x: x["pf"]) if r_4h else None

        print(f"\n  [{direction}]")
        if best_1h:
            print(f"    1h Best: PF={best_1h['pf']:.2f} ({best_1h['desc']}, {best_1h['trades']}건)")
        if best_4h:
            print(f"    4h Best: PF={best_4h['pf']:.2f} ({best_4h['desc']}, {best_4h['trades']}건)")
        if best_1h and best_4h:
            if best_4h["pf"] > best_1h["pf"]:
                print(f"    → 4h가 더 강함 (상관분석 r=-0.1734과 일치)")
            else:
                print(f"    → 1h가 더 강함 (주의: 상관분석은 4h 기준)")

    # 최종 판정
    print("\n" + "=" * 80)
    print("  [최종 판정]")
    print("=" * 80)

    # Phase 2 후보 찾기
    phase2 = []
    for hold_label in ["4h", "1h"]:
        shorts = [r for r in results if r["hold_label"] == hold_label and r["direction"] == "SHORT" and r["trades"] >= 20]
        longs = [r for r in results if r["hold_label"] == hold_label and r["direction"] == "LONG" and r["trades"] >= 20]

        best_s = max(shorts, key=lambda x: x["pf"]) if shorts else None
        best_l = max(longs, key=lambda x: x["pf"]) if longs else None

        if best_s and best_l:
            if best_s["pf"] > 1.5 and best_l["pf"] > 1.5:
                phase2.append(("Case1", hold_label, best_s, best_l))
            elif best_s["pf"] > 1.5 or best_l["pf"] > 1.5:
                phase2.append(("Case3", hold_label, best_s, best_l))

    if phase2:
        print(f"\n  🟢 Phase 2 후보 발견!")
        for case, hl, bs, bl in phase2:
            print(f"    [{case}] {hl}: SHORT PF={bs['pf']:.2f}({bs['trades']}건), "
                  f"LONG PF={bl['pf']:.2f}({bl['trades']}건)")
        print(f"\n  → Phase 2 (Bot Simulation) 진행 권장")
        print(f"  → 단, 29일 OI 데이터 + 448행 제한 감안")
    else:
        all_pf = [(r["desc"], r["pf"], r["trades"]) for r in results if r["trades"] > 0]
        all_pf.sort(key=lambda x: x[1], reverse=True)
        best = all_pf[0] if all_pf else ("N/A", 0, 0)

        above_1 = [r for r in results if r["pf"] > 1.0 and r["trades"] >= 20]
        if above_1:
            print(f"\n  🟡 PF > 1.0 조합 존재 ({len(above_1)}개), 단 < 1.5")
            for r in sorted(above_1, key=lambda x: x["pf"], reverse=True):
                emoji, _ = confidence(r["trades"])
                print(f"    {r['desc']}: PF={r['pf']:.2f}, {r['trades']}건 {emoji}")
            print(f"\n  → 약한 edge. 4월 데이터 축적 후 재검증 권장.")
        else:
            print(f"\n  🔴 PF > 1.0 조합 없음 (20건 이상)")
            print(f"    Best: {best[0]} (PF={best[1]:.2f}, {best[2]}건)")
            print(f"\n  → FR × OI 시그널도 비용 후 edge 없음")

    # Best 조합 상세
    valid_results = [r for r in results if r["trades"] > 10 and "df_trades" in r]
    if valid_results:
        best_r = max(valid_results, key=lambda x: x["pf"])
        print(f"\n[참고] Best 조합 상세: {best_r['desc']}")
        print("-" * 60)
        tdf = best_r["df_trades"]
        print(f"  기간: {tdf['entry_time'].min()} ~ {tdf['exit_time'].max()}")
        print(f"  평균 피처값: {tdf['feature_val'].mean():.8f}")
        w = tdf[tdf["net_return_bps"] > 0]
        l = tdf[tdf["net_return_bps"] <= 0]
        if len(w) > 0:
            print(f"  수익 거래 평균: {w['net_return_bps'].mean():.1f} bps ({len(w)}건)")
        if len(l) > 0:
            print(f"  손실 거래 평균: {l['net_return_bps'].mean():.1f} bps ({len(l)}건)")

    print("\n" + "=" * 80)
    print("  분석 완료.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
