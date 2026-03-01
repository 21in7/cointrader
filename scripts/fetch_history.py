"""
바이낸스 선물 REST API로 과거 캔들 데이터를 수집해 parquet으로 저장한다.
사용법: python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90
"""
import asyncio
import argparse
from datetime import datetime, timedelta
import pandas as pd
from binance import AsyncClient
from dotenv import load_dotenv
import os

load_dotenv()


async def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    try:
        start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_klines = []
        while True:
            klines = await client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            last_ts = klines[-1][0]
            if last_ts >= int(datetime.utcnow().timestamp() * 1000):
                break
            start_ts = last_ts + 1
            print(f"수집 중... {len(all_klines)}개")
    finally:
        await client.close_connection()

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="XRPUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days",     type=int, default=90)
    parser.add_argument("--output",   default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()

    df = asyncio.run(fetch_klines(args.symbol, args.interval, args.days))
    df.to_parquet(args.output)
    print(f"저장 완료: {args.output} ({len(df)}행)")


if __name__ == "__main__":
    main()
