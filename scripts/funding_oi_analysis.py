"""
Funding Rate + OI 변화율 상관분석

기존 combined_15m.parquet에 funding_rate 2년치 있음.
OI는 Binance API에서 2개월치 수집 후 병합.
상관분석 → r 값으로 edge 판정.

Usage: python scripts/funding_oi_analysis.py
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

BASE = "https://fapi.binance.com"
SYMBOL = "XRPUSDT"
DATA_DIR = Path("data/xrpusdt")
FEE_RATE = 0.0004  # 0.04% per side


async def fetch_oi_history(session, symbol, start_ms, end_ms):
    """Binance Open Interest Statistics (15m) 수집"""
    all_data = []
    current = start_ms
    calls = 0

    while current < end_ms:
        params = {
            "symbol": symbol,
            "period": "15m",
            "startTime": current,
            "endTime": end_ms,
            "limit": 500,
        }
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

        # Rate limit: ~10 weight per call, 1200/min limit
        if calls % 50 == 0:
            print(f"    ... {len(all_data)} rows fetched, sleeping 5s for rate limit")
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].drop_duplicates("timestamp").sort_values("timestamp")


async def fetch_funding_rate_history(session, symbol, start_ms, end_ms):
    """Binance Funding Rate History 수집 (8시간 간격)"""
    all_data = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
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
    df["funding_rate_api"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate_api"]].drop_duplicates("timestamp").sort_values("timestamp")


async def main():
    print("=" * 80)
    print("  Funding Rate + OI 변화율 상관분석")
    print("=" * 80)

    # Step 1: 데이터 수집
    print("\n[Step 1] 데이터 수집")

    # 기존 kline 로드
    kline_path = DATA_DIR / "combined_15m.parquet"
    df = pd.read_parquet(kline_path)
    print(f"  기존 kline: {len(df)} rows ({df.index.min()} ~ {df.index.max()})")

    # 기간 설정: OI는 30일 제한, FR은 무제한
    end_dt = datetime.now(timezone.utc)
    oi_start_dt = end_dt - timedelta(days=29)  # OI: 30일 제한
    fr_start_dt = end_dt - timedelta(days=60)  # FR: 60일
    kline_start_dt = fr_start_dt  # kline도 60일

    # Clean timestamps (no microseconds)
    oi_start_ms = int(oi_start_dt.replace(microsecond=0, second=0).timestamp()) * 1000
    fr_start_ms = int(fr_start_dt.replace(microsecond=0, second=0).timestamp()) * 1000
    end_ms = int(end_dt.replace(microsecond=0, second=0).timestamp()) * 1000

    print(f"  OI 수집 기간: {oi_start_dt.date()} ~ {end_dt.date()} (29일)")
    print(f"  FR 수집 기간: {fr_start_dt.date()} ~ {end_dt.date()} (60일)")

    async with aiohttp.ClientSession() as session:
        print("  OI 수집 중...")
        oi_df = await fetch_oi_history(session, SYMBOL, oi_start_ms, end_ms)
        print(f"  OI: {len(oi_df)} rows")

        print("  Funding Rate 수집 중...")
        fr_df = await fetch_funding_rate_history(session, SYMBOL, fr_start_ms, end_ms)
        print(f"  Funding Rate: {len(fr_df)} rows")

    # Step 2: 병합
    print("\n[Step 2] 데이터 병합")

    # 2개월 kline 슬라이스
    df_2m = df.loc[kline_start_dt:].copy()
    print(f"  2개월 kline: {len(df_2m)} rows")

    # OI 병합 (merge_asof)
    df_2m = df_2m.reset_index()
    if not oi_df.empty:
        df_2m = pd.merge_asof(
            df_2m.sort_values("timestamp"),
            oi_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=20),
        )
        # OI 변화율 계산
        df_2m["oi"] = df_2m["sumOpenInterestValue"]
        df_2m["oi_pct_change"] = df_2m["oi"].pct_change()
        df_2m["oi_pct_change_4"] = df_2m["oi"].pct_change(4)  # 1시간 변화율
        print(f"  OI 매칭: {df_2m['oi'].notna().sum()} rows")

    # Funding Rate 병합 (8h → 15m forward fill)
    if not fr_df.empty:
        df_2m = pd.merge_asof(
            df_2m.sort_values("timestamp"),
            fr_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",  # 가장 최근 funding rate 사용
        )
        # Funding rate 변화율
        df_2m["fr"] = df_2m["funding_rate_api"]
        df_2m["fr_change"] = df_2m["fr"].diff()
        print(f"  Funding Rate 매칭: {df_2m['fr'].notna().sum()} rows")

    # 기존 funding_rate 컬럼도 활용
    df_2m["fr_existing"] = df_2m["funding_rate"]
    df_2m["fr_existing_change"] = df_2m["fr_existing"].diff()

    # 미래 수익률 계산
    df_2m["next_1h_return"] = df_2m["close"].shift(-4) / df_2m["close"] - 1
    df_2m["next_4h_return"] = df_2m["close"].shift(-16) / df_2m["close"] - 1
    df_2m["next_15m_return"] = df_2m["close"].shift(-1) / df_2m["close"] - 1

    # 복합 피처
    if "oi_pct_change" in df_2m.columns and "fr" in df_2m.columns:
        df_2m["fr_x_oi"] = df_2m["fr"] * df_2m["oi_pct_change"]  # 펀딩비 × OI변화율
        df_2m["fr_x_oi_4"] = df_2m["fr"] * df_2m["oi_pct_change_4"]

    df_2m = df_2m.set_index("timestamp")

    # OI velocity (변화율의 변화율)
    if "oi_pct_change" in df_2m.columns:
        df_2m["oi_velocity"] = df_2m["oi_pct_change"].diff()
        df_2m["oi_acceleration"] = df_2m["oi_velocity"].diff()

    print(f"\n  최종 데이터셋: {len(df_2m)} rows, {len(df_2m.columns)} columns")

    # Step 3: 상관분석
    print("\n[Step 3] 상관분석")
    print("=" * 80)

    features = [
        ("fr_existing", "Funding Rate (기존)"),
        ("fr_existing_change", "ΔFunding Rate"),
        ("fr", "Funding Rate (API)"),
        ("fr_change", "ΔFunding Rate (API)"),
        ("oi_pct_change", "OI 변화율 (15m)"),
        ("oi_pct_change_4", "OI 변화율 (1h)"),
        ("oi_velocity", "OI Velocity"),
        ("oi_acceleration", "OI Acceleration"),
        ("fr_x_oi", "FR × OI변화율"),
        ("fr_x_oi_4", "FR × OI변화율(1h)"),
    ]

    targets = [
        ("next_15m_return", "다음 15m"),
        ("next_1h_return", "다음 1h"),
        ("next_4h_return", "다음 4h"),
    ]

    print(f"\n{'피처':<25} {'→15m':>8} {'→1h':>8} {'→4h':>8} {'N':>7}")
    print("-" * 60)

    strong_signals = []
    for feat_col, feat_name in features:
        if feat_col not in df_2m.columns:
            continue
        corrs = []
        n = 0
        for tgt_col, _ in targets:
            valid = df_2m[[feat_col, tgt_col]].dropna()
            n = len(valid)
            if n > 50:
                r = valid[feat_col].corr(valid[tgt_col])
                corrs.append(r)
            else:
                corrs.append(float("nan"))

        r_strs = [f"{r:>+8.4f}" if not np.isnan(r) else f"{'N/A':>8}" for r in corrs]
        print(f"{feat_name:<25} {''.join(r_strs)} {n:>7}")

        # 강한 시그널 체크 (|r| > 0.05)
        for r, (tgt_col, tgt_name) in zip(corrs, targets):
            if not np.isnan(r) and abs(r) > 0.05:
                strong_signals.append((feat_name, tgt_name, r, n))

    # Quintile 분석 (강한 시그널에 대해)
    print("\n" + "=" * 80)
    print("  [Quintile 분석] |r| > 0.05 피처")
    print("=" * 80)

    for feat_col, feat_name in features:
        if feat_col not in df_2m.columns:
            continue

        for tgt_col, tgt_name in targets:
            valid = df_2m[[feat_col, tgt_col]].dropna()
            if len(valid) < 100:
                continue
            r = valid[feat_col].corr(valid[tgt_col])
            if abs(r) < 0.05:
                continue

            print(f"\n  {feat_name} → {tgt_name} (r={r:+.4f}, n={len(valid)})")
            print(f"  {'Quintile':<12} {'mean_feat':>12} {'return_bps':>12} {'win_rate':>10} {'count':>7}")
            print("  " + "-" * 55)

            try:
                valid["q"] = pd.qcut(valid[feat_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            except ValueError:
                continue

            for q in valid["q"].cat.categories:
                grp = valid[valid["q"] == q]
                if len(grp) == 0:
                    continue
                mr = grp[feat_col].mean()
                ret = grp[tgt_col].mean() * 10000
                wr = (grp[tgt_col] > 0).mean() * 100
                print(f"  {q:<12} {mr:>12.6f} {ret:>+12.2f} {wr:>9.1f}% {len(grp):>7}")

    # 판정
    print("\n" + "=" * 80)
    print("  [최종 판정]")
    print("=" * 80)

    if strong_signals:
        print(f"\n  |r| > 0.05 시그널: {len(strong_signals)}개")
        for feat, tgt, r, n in sorted(strong_signals, key=lambda x: abs(x[2]), reverse=True):
            marker = "🟢" if abs(r) > 0.15 else "🟡" if abs(r) > 0.10 else "⚪"
            print(f"  {marker} {feat} → {tgt}: r={r:+.4f} (n={n})")

        best_r = max(abs(r) for _, _, r, _ in strong_signals)
        if best_r > 0.15:
            print(f"\n  ✅ r > 0.15 시그널 발견! 백테스트 진행 가치 있음")
        elif best_r > 0.10:
            print(f"\n  🟡 r = 0.10~0.15. L/S ratio(0.1158)과 비슷한 수준.")
            print(f"     단, 2개월 데이터(8일 대비 7.5배)이므로 신뢰도 높음.")
            print(f"     백테스트로 비용 후 PF 확인 필요.")
        else:
            print(f"\n  ⚠️ 최대 |r| = {best_r:.4f}. 약한 시그널.")
            print(f"     비용(0.08%) 커버 가능성 낮음.")
    else:
        print("\n  🔴 |r| > 0.05 시그널 없음. Edge 없음.")

    print("\n" + "=" * 80)
    print("  분석 완료.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
