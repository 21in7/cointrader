"""
일봉 spot klines 전체 히스토리 취득 + sanity (momentum precheck용).

모멘텀은 느린 신호 → 표본 길수록 좋다 → 상장 이후 전체(2017~). 일봉 directional
수익률은 spot≈perp이고 벤치마크(buy&hold)도 spot이라 내부 일관 → spot 사용.
scripts/fetch_history.py 컨벤션(get_klines, UTC, 페이지네이션)을 따른다.

실행:  python -m src.momentum.data       # 8심볼 일봉 취득 + sanity 후 STOP
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from binance import AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTERVAL = "1d"
START_MS = 1483228800000  # 2017-01-01 UTC (상장 이후 전체 포착)
REQUEST_DELAY = 0.25
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT",
           "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "TRXUSDT"]


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def _daily(client: AsyncClient, symbol: str) -> pd.DataFrame:
    start_ts, rows = START_MS, []
    while True:
        for attempt in range(3):
            try:
                k = await client.get_klines(symbol=symbol, interval=INTERVAL,
                                            startTime=start_ts, limit=1000)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"{symbol} 일봉 실패: {e}")
        if not k:
            break
        rows.extend(k)
        last = k[-1][0]
        if last >= _now_ms() or len(k) < 1000:
            break
        start_ts = last + 1
        await asyncio.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbb", "tbq", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="first")]


def acquire(symbols=SYMBOLS, save=True) -> dict:
    async def _run():
        client = await AsyncClient.create()
        try:
            return {s: await _daily(client, s) for s in symbols}
        finally:
            await client.close_connection()

    print("=" * 80)
    print("  일봉 spot 전체 히스토리 취득 (2017~) — momentum precheck 데이터 sanity")
    print("=" * 80)
    data = asyncio.run(_run())

    sanity, closes = {}, {}
    one_day = pd.Timedelta(days=1)
    for s, df in data.items():
        diffs = pd.Series(df.index).diff().dropna()
        gaps = int((diffs != one_day).sum())
        nan = int(df["close"].isna().sum())
        sanity[s] = {"rows": int(len(df)), "start": str(df.index[0].date()),
                     "end": str(df.index[-1].date()), "gaps": gaps, "nan_close": nan,
                     "years": round((df.index[-1] - df.index[0]).days / 365.25, 2)}
        closes[s] = df["close"].rename(s)
        print(f"  {s:9s} rows={len(df):>5d}  {sanity[s]['start']} ~ {sanity[s]['end']} "
              f"({sanity[s]['years']:.1f}yr)  1d外갭={gaps}  NaN={nan}")
        if save:
            outdir = ROOT / "data" / s.lower()
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(outdir / "daily_spot.parquet")

    panel = pd.concat(closes.values(), axis=1, join="outer")
    inter = pd.concat(closes.values(), axis=1, join="inner")
    print(f"\n  패널(8심볼): union {panel.index[0].date()}~{panel.index[-1].date()} "
          f"({len(panel)}행), 공통구간(전8종 존재) {inter.index[0].date()}~{inter.index[-1].date()} "
          f"({len(inter)}행, {round((inter.index[-1]-inter.index[0]).days/365.25,1)}yr)")
    print("  → TSMOM은 코인별 전체 사용, XSMOM 단면 랭킹은 공통구간 필요.")
    if save:
        print(f"  저장: data/{{sym}}/daily_spot.parquet ({len(symbols)}개)")

    print("\n" + "=" * 80)
    print("  [체크포인트] 일봉 취득 + sanity 완료. 게이트는 다음 단계(승인 후).")
    print("=" * 80)
    return sanity


if __name__ == "__main__":
    acquire()
