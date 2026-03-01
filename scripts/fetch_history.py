"""
바이낸스 선물 REST API로 과거 캔들 데이터를 수집해 parquet으로 저장한다.
사용법: python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90
       python scripts/fetch_history.py --symbols XRPUSDT BTCUSDT ETHUSDT --days 90
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
from binance import AsyncClient
from dotenv import load_dotenv
import os

load_dotenv()

# 요청 사이 딜레이 (초). 바이낸스 선물 기본 한도: 2400 req/min = 40 req/s
# 1500개씩 가져오므로 90일 1m 데이터 = ~65회 요청/심볼
# 심볼 간 딜레이 없이 연속 요청하면 레이트 리밋(-1003) 발생
_REQUEST_DELAY = 0.3  # 초당 ~3.3 req → 안전 마진 충분


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def _fetch_klines_with_client(
    client: AsyncClient,
    symbol: str,
    interval: str,
    days: int,
) -> pd.DataFrame:
    """기존 클라이언트를 재사용해 단일 심볼 캔들을 수집한다."""
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
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
        if last_ts >= _now_ms():
            break
        start_ts = last_ts + 1
        print(f"  [{symbol}] 수집 중... {len(all_klines):,}개")
        await asyncio.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


async def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """단일 심볼 수집 (하위 호환용)."""
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    try:
        return await _fetch_klines_with_client(client, symbol, interval, days)
    finally:
        await client.close_connection()


async def fetch_klines_all(
    symbols: list[str],
    interval: str,
    days: int,
) -> dict[str, pd.DataFrame]:
    """
    단일 클라이언트로 여러 심볼을 순차 수집한다.
    asyncio.run()을 심볼마다 반복하면 연결 오버헤드와 레이트 리밋 위험이 있으므로
    하나의 연결 안에서 심볼 간 딜레이를 두고 순차 처리한다.
    """
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )
    dfs = {}
    try:
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] {symbol} 수집 시작...")
            dfs[symbol] = await _fetch_klines_with_client(client, symbol, interval, days)
            print(f"  [{symbol}] 완료: {len(dfs[symbol]):,}행")
            # 심볼 간 추가 대기: 레이트 리밋 카운터가 리셋될 시간 확보
            if i < len(symbols) - 1:
                print(f"  다음 심볼 수집 전 5초 대기...")
                await asyncio.sleep(5)
    finally:
        await client.close_connection()
    return dfs


def main():
    parser = argparse.ArgumentParser(
        description="바이낸스 선물 과거 캔들 수집. 단일 심볼 또는 멀티 심볼 병합 저장."
    )
    parser.add_argument("--symbols", nargs="+", default=["XRPUSDT"])
    parser.add_argument("--symbol",   default=None, help="단일 심볼 (--symbols 미사용 시)")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days",     type=int, default=90)
    parser.add_argument("--output",   default="data/xrpusdt_1m.parquet")
    args = parser.parse_args()

    # 하위 호환: --symbol 단독 사용 시 symbols로 통합
    if args.symbol and args.symbols == ["XRPUSDT"]:
        args.symbols = [args.symbol]

    if len(args.symbols) == 1:
        df = asyncio.run(fetch_klines(args.symbols[0], args.interval, args.days))
        df.to_parquet(args.output)
        print(f"저장 완료: {args.output} ({len(df):,}행)")
    else:
        # 멀티 심볼: 단일 클라이언트로 순차 수집 후 타임스탬프 기준 inner join 병합
        dfs = asyncio.run(fetch_klines_all(args.symbols, args.interval, args.days))

        primary = args.symbols[0]
        merged = dfs[primary].copy()
        for symbol in args.symbols[1:]:
            suffix = "_" + symbol.lower().replace("usdt", "")
            merged = merged.join(
                dfs[symbol].add_suffix(suffix),
                how="inner",
            )

        output = args.output.replace("xrpusdt", "combined")
        merged.to_parquet(output)
        print(f"\n병합 저장 완료: {output} ({len(merged):,}행, {len(merged.columns)}컬럼)")


if __name__ == "__main__":
    main()
