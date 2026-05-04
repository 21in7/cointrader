"""
L/S Ratio 단독 백테스트 — Phase 1: Pure Edge Test

6개 조합 (3 임계값 × 2 방향) 스윕, 3단계 필터 판정.
데이터: 프로덕션 수집 L/S ratio + Binance kline (같은 기간).

Usage: python scripts/ls_ratio_backtest.py
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import timezone
from pathlib import Path

BASE = "https://fapi.binance.com"
DATA_DIR = Path("data")
SYMBOL = "XRPUSDT"
FEE_RATE = 0.0004  # 0.04% per side
HOLD_BARS = 4  # 4 candles = 1 hour


async def fetch_klines(session, symbol, start_ms, end_ms):
    """Binance kline 데이터 가져오기"""
    all_klines = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol, "interval": "15m",
            "startTime": current, "endTime": end_ms, "limit": 1500,
        }
        async with session.get(f"{BASE}/fapi/v1/klines", params=params) as resp:
            data = await resp.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 1
    return all_klines


def load_ls_ratio(symbol):
    """프로덕션 수집 L/S ratio 로드"""
    path = DATA_DIR / symbol.lower() / "ls_ratio_15m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Sync from production first.")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_dataset(klines_raw, ls_df):
    """Kline + L/S ratio 조인"""
    df = pd.DataFrame(klines_raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # L/S ratio 조인 (가장 가까운 타임스탬프)
    df = df.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        df, ls_df, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(minutes=20),
    )
    return merged


def run_backtest(df, percentile, direction, hold_bars=HOLD_BARS):
    """
    단일 조합 백테스트 실행.

    - percentile: L/S ratio 임계값 (0~100)
    - direction: "LONG" or "SHORT"
    - hold_bars: 보유 캔들 수

    LONG 진입: ratio >= threshold (Momentum)
    SHORT 진입: ratio <= threshold (Momentum)
    """
    threshold = df["top_acct_ls_ratio"].quantile(percentile / 100)

    trades = []
    i = 0
    while i < len(df) - hold_bars:
        ratio = df.iloc[i]["top_acct_ls_ratio"]
        if pd.isna(ratio):
            i += 1
            continue

        # 시그널 체크
        if direction == "LONG" and ratio >= threshold:
            entry_price = df.iloc[i + 1]["open"]  # 다음 캔들 시가 진입
            exit_price = df.iloc[i + 1 + hold_bars - 1]["close"]  # hold_bars 후 종가
            gross_return = (exit_price / entry_price) - 1
            fee = FEE_RATE * 2  # 진입 + 청산
            net_return = gross_return - fee
            trades.append({
                "entry_time": df.iloc[i + 1]["timestamp"],
                "exit_time": df.iloc[i + 1 + hold_bars - 1]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_ls_ratio": ratio,
                "gross_return_bps": gross_return * 10000,
                "net_return_bps": net_return * 10000,
                "fee_bps": fee * 10000,
            })
            i += 1 + hold_bars  # 포지션 종료 후 다음 캔들부터
        elif direction == "SHORT" and ratio <= threshold:
            entry_price = df.iloc[i + 1]["open"]
            exit_price = df.iloc[i + 1 + hold_bars - 1]["close"]
            gross_return = (entry_price / exit_price) - 1  # SHORT: 반대
            fee = FEE_RATE * 2
            net_return = gross_return - fee
            trades.append({
                "entry_time": df.iloc[i + 1]["timestamp"],
                "exit_time": df.iloc[i + 1 + hold_bars - 1]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_ls_ratio": ratio,
                "gross_return_bps": gross_return * 10000,
                "net_return_bps": net_return * 10000,
                "fee_bps": fee * 10000,
            })
            i += 1 + hold_bars
        else:
            i += 1

    if not trades:
        return None

    df_trades = pd.DataFrame(trades)

    # PF 계산: Σ(net profit) / Σ(|net loss|)
    wins = df_trades[df_trades["net_return_bps"] > 0]["net_return_bps"]
    losses = df_trades[df_trades["net_return_bps"] <= 0]["net_return_bps"]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Max Drawdown (cumulative bps)
    cum_pnl = df_trades["net_return_bps"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    return {
        "trades": len(df_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(df_trades) * 100,
        "pf": pf,
        "total_pnl_bps": df_trades["net_return_bps"].sum(),
        "avg_pnl_bps": df_trades["net_return_bps"].mean(),
        "max_dd_bps": max_dd,
        "threshold": threshold,
        "df_trades": df_trades,
    }


def confidence_emoji(n_trades):
    if n_trades < 20:
        return "🔴"
    elif n_trades < 50:
        return "🟡"
    elif n_trades < 100:
        return "🟢"
    else:
        return "🟢"


def confidence_label(n_trades):
    if n_trades < 20:
        return "폐기(과적합)"
    elif n_trades < 50:
        return "낮음(참고만)"
    elif n_trades < 100:
        return "보통(검토)"
    else:
        return "높음(우선)"


async def main():
    print("=" * 80)
    print("  L/S Ratio 단독 백테스트 — Phase 1: Pure Edge Test")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    ls_df = load_ls_ratio(SYMBOL)
    print(f"  L/S ratio: {len(ls_df)} rows ({ls_df['timestamp'].min()} ~ {ls_df['timestamp'].max()})")

    start_ms = int(ls_df["timestamp"].min().timestamp() * 1000)
    end_ms = int(ls_df["timestamp"].max().timestamp() * 1000)

    async with aiohttp.ClientSession() as session:
        klines = await fetch_klines(session, SYMBOL, start_ms, end_ms)
    print(f"  Klines: {len(klines)} rows")

    df = build_dataset(klines, ls_df)
    valid = df.dropna(subset=["top_acct_ls_ratio"])
    print(f"  조인 결과: {len(df)} rows (L/S 매칭: {len(valid)})")
    print(f"  top_acct_ls_ratio: mean={valid['top_acct_ls_ratio'].mean():.4f}, "
          f"std={valid['top_acct_ls_ratio'].std():.4f}")

    # 백분위수 표시
    for p in [25, 50, 75]:
        v = valid["top_acct_ls_ratio"].quantile(p / 100)
        print(f"  P{p}: {v:.4f}")

    # 2. 6개 조합 백테스트
    print("\n[2] 6개 조합 백테스트 실행")
    print("-" * 80)

    combinations = [
        (75, "LONG", "모멘텀 강함: ratio ≥ P75 → LONG"),
        (75, "SHORT", "역모멘텀: ratio ≥ P75 → SHORT"),
        (50, "LONG", "모멘텀 중간: ratio ≥ P50 → LONG"),
        (50, "SHORT", "역모멘텀 중간: ratio ≤ P50 → SHORT"),
        (25, "LONG", "역모멘텀 약: ratio ≤ P25 → LONG"),
        (25, "SHORT", "모멘텀 강함: ratio ≤ P25 → SHORT"),
    ]

    results = []
    for pct, direction, desc in combinations:
        # LONG: ratio >= threshold, SHORT: ratio <= threshold
        # 25th percentile LONG = ratio가 낮을 때 LONG (Contrarian)
        # 실제 로직:
        #   75 LONG = ratio >= P75 (상위 25% 롱비율 높을 때 롱) = Momentum
        #   75 SHORT = ratio >= P75 (상위 25% 롱비율 높을 때 숏) = Contrarian
        #   25 SHORT = ratio <= P25 (하위 25% 롱비율 낮을 때 숏) = Momentum
        #   25 LONG = ratio <= P25 (하위 25% 롱비율 낮을 때 롱) = Contrarian

        # 방향 보정: 25th에서 LONG은 "ratio <= P25일 때 LONG" (Contrarian)
        if pct == 25 and direction == "LONG":
            # 특수 케이스: 낮은 ratio에서 LONG (Contrarian)
            result = run_backtest_contrarian(df, pct, "LONG")
        elif pct == 25 and direction == "SHORT":
            # ratio <= P25일 때 SHORT (Momentum)
            result = run_backtest(df, pct, "SHORT")
        elif pct == 75 and direction == "SHORT":
            # ratio >= P75일 때 SHORT (Contrarian)
            result = run_backtest_contrarian(df, pct, "SHORT")
        else:
            result = run_backtest(df, pct, direction)

        if result:
            result["percentile"] = pct
            result["direction"] = direction
            result["description"] = desc
            results.append(result)
        else:
            results.append({
                "percentile": pct, "direction": direction,
                "description": desc, "trades": 0, "pf": 0,
                "win_rate": 0, "total_pnl_bps": 0, "max_dd_bps": 0,
                "threshold": 0, "wins": 0, "losses": 0, "avg_pnl_bps": 0,
            })

    # 3. 결과 테이블
    print("\n[3] 결과 테이블")
    print("=" * 80)
    print(f"{'조합':<35} {'거래수':>6} {'승률':>7} {'PF':>7} {'PnL(bps)':>10} {'MaxDD':>10} {'신뢰도':<15}")
    print("-" * 80)

    for r in results:
        emoji = confidence_emoji(r["trades"])
        label = confidence_label(r["trades"])
        pf_str = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "INF"
        print(f"{r['description']:<35} {r['trades']:>6} {r['win_rate']:>6.1f}% {pf_str:>7} "
              f"{r['total_pnl_bps']:>+10.1f} {r['max_dd_bps']:>10.1f} {emoji} {label}")

    # 4. 필터 1: PF 판정
    print("\n[4] 필터 1: PF 판정")
    print("-" * 80)

    strong = [r for r in results if r["pf"] > 1.5 and r["trades"] > 0]
    weak = [r for r in results if 0.5 <= r["pf"] <= 1.5 and r["trades"] > 0]
    failed = [r for r in results if r["pf"] < 0.5 and r["trades"] > 0]

    print(f"  PF > 1.5 (명확한 edge):  {len(strong)}개 조합")
    for r in strong:
        print(f"    → {r['description']} (PF={r['pf']:.2f}, trades={r['trades']})")
    print(f"  0.5 ≤ PF ≤ 1.5 (보류): {len(weak)}개 조합")
    for r in weak:
        print(f"    ~ {r['description']} (PF={r['pf']:.2f}, trades={r['trades']})")
    print(f"  PF < 0.5 (실패):        {len(failed)}개 조합")
    for r in failed:
        print(f"    ✗ {r['description']} (PF={r['pf']:.2f}, trades={r['trades']})")

    # 5. 필터 2: 거래수 신뢰도
    print("\n[5] 필터 2: 거래수 신뢰도 (필터 1 통과 조합)")
    print("-" * 80)

    filter2_passed = [r for r in strong if r["trades"] >= 20]
    filter2_ref = [r for r in strong if r["trades"] < 20]

    if filter2_passed:
        for r in filter2_passed:
            print(f"  ✓ {r['description']} — {r['trades']}건 ({confidence_label(r['trades'])})")
    else:
        print("  ⚠️ PF > 1.5 조합 중 거래수 20건 이상인 것 없음")
    if filter2_ref:
        for r in filter2_ref:
            print(f"  🔴 {r['description']} — {r['trades']}건 (폐기: 과적합)")

    # 6. 필터 3: 대칭성 판정
    print("\n[6] 필터 3: 대칭성 판정")
    print("-" * 80)

    # 같은 percentile에서 LONG/SHORT 양쪽 확인
    for pct in [75, 50, 25]:
        long_r = next((r for r in results if r["percentile"] == pct and r["direction"] == "LONG"), None)
        short_r = next((r for r in results if r["percentile"] == pct and r["direction"] == "SHORT"), None)
        if not long_r or not short_r:
            continue

        l_pf = long_r["pf"]
        s_pf = short_r["pf"]

        if l_pf > 1.5 and s_pf > 1.5:
            verdict = "Case 1: 양방향 생존 → ✓ Phase 2 후보"
        elif (l_pf > 1.5 and s_pf < 0.5) or (s_pf > 1.5 and l_pf < 0.5):
            verdict = "Case 2: 한쪽만 성공 → ✗ 시장 베타/우연 (폐기)"
        elif l_pf > 1.5 or s_pf > 1.5:
            verdict = "Case 3: 부분적 edge → ~ 낮은 신뢰도"
        else:
            verdict = "양쪽 모두 약함 → 해당 없음"

        print(f"  P{pct}: LONG PF={l_pf:.2f}, SHORT PF={s_pf:.2f}")
        print(f"       → {verdict}")

    # 7. 최종 판정
    print("\n" + "=" * 80)
    print("  [최종 판정]")
    print("=" * 80)

    # Phase 2 후보 찾기
    phase2_candidates = []
    for pct in [75, 50, 25]:
        long_r = next((r for r in results if r["percentile"] == pct and r["direction"] == "LONG"), None)
        short_r = next((r for r in results if r["percentile"] == pct and r["direction"] == "SHORT"), None)
        if not long_r or not short_r:
            continue

        # Case 1: 양방향 PF > 1.5
        if long_r["pf"] > 1.5 and short_r["pf"] > 1.5:
            if long_r["trades"] >= 20 and short_r["trades"] >= 20:
                phase2_candidates.append(("Case1", pct, long_r, short_r))
        # Case 3: 한쪽만 PF > 1.5
        elif long_r["pf"] > 1.5 and long_r["trades"] >= 20:
            phase2_candidates.append(("Case3-LONG", pct, long_r, short_r))
        elif short_r["pf"] > 1.5 and short_r["trades"] >= 20:
            phase2_candidates.append(("Case3-SHORT", pct, long_r, short_r))

    if phase2_candidates:
        print("\n  🟢 Phase 2 진행 후보 발견!")
        for case, pct, lr, sr in phase2_candidates:
            print(f"    [{case}] P{pct}: LONG PF={lr['pf']:.2f}({lr['trades']}건), "
                  f"SHORT PF={sr['pf']:.2f}({sr['trades']}건)")
        print("\n  → Phase 2 (Bot Simulation) 진행 권장")
        print("  → 단, 8일 데이터이므로 4월 15일 재검증 필수")
    else:
        # 모든 조합 중 최고 PF
        best = max(results, key=lambda r: r["pf"] if r["trades"] > 0 else 0)
        if best["pf"] > 1.0:
            print(f"\n  🟡 필터 미통과이나 PF > 1.0 조합 존재")
            print(f"    Best: {best['description']} (PF={best['pf']:.2f}, {best['trades']}건)")
            print(f"\n  → 데이터 부족. 4월 15일까지 수집 후 재검증")
        else:
            print(f"\n  🔴 PF > 1.0 조합 없음")
            print(f"    Best: {best['description']} (PF={best['pf']:.2f}, {best['trades']}건)")
            print(f"\n  → L/S ratio 단독 시그널로는 edge 없음")
            print(f"  → 다른 데이터 소스 탐색 권장")

    # 8. 추가: 전 구간 상세 (best 조합)
    best = max(results, key=lambda r: r["pf"] if r["trades"] > 10 else 0)
    if "df_trades" in best and best["trades"] > 0:
        print(f"\n[참고] Best 조합 상세: {best['description']}")
        print("-" * 60)
        tdf = best["df_trades"]
        print(f"  거래 기간: {tdf['entry_time'].min()} ~ {tdf['exit_time'].max()}")
        print(f"  평균 진입 L/S ratio: {tdf['entry_ls_ratio'].mean():.4f}")
        print(f"  수익 거래 평균: {tdf[tdf['net_return_bps']>0]['net_return_bps'].mean():.1f} bps")
        if len(tdf[tdf['net_return_bps'] <= 0]) > 0:
            print(f"  손실 거래 평균: {tdf[tdf['net_return_bps']<=0]['net_return_bps'].mean():.1f} bps")
        print(f"  최대 연승: ", end="")
        streaks = []
        streak = 0
        for _, row in tdf.iterrows():
            if row["net_return_bps"] > 0:
                streak += 1
            else:
                if streak > 0:
                    streaks.append(streak)
                streak = 0
        if streak > 0:
            streaks.append(streak)
        print(f"{max(streaks) if streaks else 0}연승")

    print("\n" + "=" * 80)
    print("  분석 완료. 결과를 바탕으로 의사결정하세요.")
    print("=" * 80)


def run_backtest_contrarian(df, percentile, direction, hold_bars=HOLD_BARS):
    """
    Contrarian 방향 백테스트.
    - P25 + LONG: ratio <= P25일 때 LONG (낮은 ratio에서 롱)
    - P75 + SHORT: ratio >= P75일 때 SHORT (높은 ratio에서 숏)
    """
    threshold = df["top_acct_ls_ratio"].quantile(percentile / 100)

    trades = []
    i = 0
    while i < len(df) - hold_bars:
        ratio = df.iloc[i]["top_acct_ls_ratio"]
        if pd.isna(ratio):
            i += 1
            continue

        trigger = False
        if direction == "LONG" and ratio <= threshold:
            trigger = True
        elif direction == "SHORT" and ratio >= threshold:
            trigger = True

        if trigger:
            entry_price = df.iloc[i + 1]["open"]
            exit_price = df.iloc[i + 1 + hold_bars - 1]["close"]
            if direction == "LONG":
                gross_return = (exit_price / entry_price) - 1
            else:
                gross_return = (entry_price / exit_price) - 1
            fee = FEE_RATE * 2
            net_return = gross_return - fee
            trades.append({
                "entry_time": df.iloc[i + 1]["timestamp"],
                "exit_time": df.iloc[i + 1 + hold_bars - 1]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_ls_ratio": ratio,
                "gross_return_bps": gross_return * 10000,
                "net_return_bps": net_return * 10000,
                "fee_bps": fee * 10000,
            })
            i += 1 + hold_bars
        else:
            i += 1

    if not trades:
        return None

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["net_return_bps"] > 0]["net_return_bps"]
    losses = df_trades[df_trades["net_return_bps"] <= 0]["net_return_bps"]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    cum_pnl = df_trades["net_return_bps"].cumsum()
    running_max = cum_pnl.cummax()
    max_dd = (cum_pnl - running_max).min()

    return {
        "trades": len(df_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(df_trades) * 100,
        "pf": pf,
        "total_pnl_bps": df_trades["net_return_bps"].sum(),
        "avg_pnl_bps": df_trades["net_return_bps"].mean(),
        "max_dd_bps": max_dd,
        "threshold": threshold,
        "df_trades": df_trades,
    }


if __name__ == "__main__":
    asyncio.run(main())
