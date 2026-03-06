"""
바이낸스 선물 REST API로 과거 캔들 데이터를 수집해 parquet으로 저장한다.
사용법: python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90
       python scripts/fetch_history.py --symbols XRPUSDT BTCUSDT ETHUSDT --days 90

OI/펀딩비 수집 제약:
  - OI 히스토리: 바이낸스 API 제한으로 최근 30일치만 제공 (period=15m, limit=500/req)
  - 펀딩비: 8시간 주기 → 15분봉에 forward-fill 병합
  - 30일 이전 구간은 oi_change=0, funding_rate=0으로 채움
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
import aiohttp
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
_FAPI_BASE = "https://fapi.binance.com"


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
        for attempt in range(3):
            try:
                klines = await client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_ts,
                    limit=1500,
                )
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  [{symbol}] API 오류 ({e}), {wait}초 후 재시도 ({attempt+1}/3)")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [{symbol}] API 3회 실패, 수집 중단: {e}")
                    raise
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


async def _fetch_oi_hist(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str = "15m",
) -> pd.DataFrame:
    """
    바이낸스 /futures/data/openInterestHist 엔드포인트로 OI 히스토리를 수집한다.
    API 제한: 최근 30일치만 제공, 1회 최대 500개.
    """
    url = f"{_FAPI_BASE}/futures/data/openInterestHist"
    all_rows = []
    # 30일 전부터 현재까지 수집
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"  [{symbol}] OI 히스토리 수집 중 (최근 30일)...")
    while start_ts < now_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": 500,
            "startTime": start_ts,
        }
        async with session.get(url, params=params) as resp:
            data = await resp.json()

        if not data or not isinstance(data, list):
            break

        all_rows.extend(data)
        last_ts = int(data[-1]["timestamp"])
        if last_ts >= now_ms or len(data) < 500:
            break
        start_ts = last_ts + 1
        await asyncio.sleep(_REQUEST_DELAY)

    if not all_rows:
        print(f"  [{symbol}] OI 데이터 없음 — 빈 DataFrame 반환")
        return pd.DataFrame(columns=["oi", "oi_value"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["sumOpenInterest", "sumOpenInterestValue"]].copy()
    df.columns = ["oi", "oi_value"]
    df["oi"] = df["oi"].astype(float)
    df["oi_value"] = df["oi_value"].astype(float)
    # OI 변화율 (1캔들 전 대비)
    df["oi_change"] = df["oi"].pct_change(1).fillna(0)
    print(f"  [{symbol}] OI 수집 완료: {len(df):,}행")
    return df[["oi_change"]]


async def _fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str,
    days: int,
) -> pd.DataFrame:
    """
    바이낸스 /fapi/v1/fundingRate 엔드포인트로 펀딩비 히스토리를 수집한다.
    8시간 주기 데이터 → 15분봉 인덱스에 forward-fill로 병합 예정.
    """
    url = f"{_FAPI_BASE}/fapi/v1/fundingRate"
    all_rows = []
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"  [{symbol}] 펀딩비 히스토리 수집 중 ({days}일)...")
    while start_ts < now_ms:
        params = {
            "symbol": symbol,
            "startTime": start_ts,
            "limit": 1000,
        }
        async with session.get(url, params=params) as resp:
            data = await resp.json()

        if not data or not isinstance(data, list):
            break

        all_rows.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts >= now_ms or len(data) < 1000:
            break
        start_ts = last_ts + 1
        await asyncio.sleep(_REQUEST_DELAY)

    if not all_rows:
        print(f"  [{symbol}] 펀딩비 데이터 없음 — 빈 DataFrame 반환")
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["funding_rate"] = df["fundingRate"].astype(float)
    print(f"  [{symbol}] 펀딩비 수집 완료: {len(df):,}행")
    return df[["funding_rate"]]


def _merge_oi_funding(
    candles: pd.DataFrame,
    oi_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    캔들 DataFrame에 OI 변화율과 펀딩비를 병합한다.
    - oi_change: 15분봉 인덱스에 nearest merge (없는 구간은 0)
    - funding_rate: 8시간 주기 → forward-fill 후 병합 (없는 구간은 0)
    """
    result = candles.copy()

    # OI 병합: 타임스탬프 기준 reindex + nearest fill
    if not oi_df.empty:
        oi_reindexed = oi_df.reindex(result.index, method="nearest", tolerance=pd.Timedelta("8min"))
        result["oi_change"] = oi_reindexed["oi_change"].fillna(0).astype(float)
    else:
        result["oi_change"] = 0.0

    # 펀딩비 병합: forward-fill (8시간 주기이므로 다음 펀딩 시점까지 이전 값 유지)
    if not funding_df.empty:
        funding_reindexed = funding_df.reindex(
            result.index.union(funding_df.index)
        ).sort_index()
        funding_reindexed = funding_reindexed["funding_rate"].ffill()
        result["funding_rate"] = funding_reindexed.reindex(result.index).fillna(0).astype(float)
    else:
        result["funding_rate"] = 0.0

    return result


async def _fetch_oi_and_funding(
    symbol: str,
    days: int,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """단일 심볼의 OI + 펀딩비를 수집해 캔들에 병합한다."""
    async with aiohttp.ClientSession() as session:
        oi_df = await _fetch_oi_hist(session, symbol)
        await asyncio.sleep(1)
        funding_df = await _fetch_funding_rate(session, symbol, days)

    return _merge_oi_funding(candles, oi_df, funding_df)


def upsert_parquet(path: "Path | str", new_df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 parquet 파일에 신규 데이터를 Upsert(병합)한다.

    규칙:
    - 기존 행의 oi_change / funding_rate가 0.0이면 신규 값으로 덮어씀
    - 기존 행의 oi_change / funding_rate가 이미 0이 아니면 유지
    - 신규 타임스탬프 행은 그냥 추가
    - 결과는 timestamp 기준 오름차순 정렬, 중복 제거

    Args:
        path: 기존 parquet 경로 (없으면 new_df 그대로 반환)
        new_df: 새로 수집한 DataFrame (timestamp index)

    Returns:
        병합된 DataFrame
    """
    path = Path(path)
    if not path.exists():
        return new_df.sort_index()

    existing = pd.read_parquet(path)

    # timestamp index 통일 (tz-aware UTC)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")

    # 기존 데이터에서 oi_change / funding_rate가 0.0인 행만 신규 값으로 업데이트
    UPSERT_COLS = ["oi_change", "funding_rate"]
    overlap_idx = existing.index.intersection(new_df.index)

    for col in UPSERT_COLS:
        if col not in existing.columns or col not in new_df.columns:
            continue
        # 겹치는 행 중 기존 값이 0.0인 경우에만 신규 값으로 교체
        zero_mask = existing.loc[overlap_idx, col] == 0.0
        update_idx = overlap_idx[zero_mask]
        if len(update_idx) > 0:
            existing.loc[update_idx, col] = new_df.loc[update_idx, col]

    # 신규 타임스탬프 행 추가 (기존에 없는 것만)
    new_only_idx = new_df.index.difference(existing.index)
    if len(new_only_idx) > 0:
        existing = pd.concat([existing, new_df.loc[new_only_idx]])

    # 컬럼 불일치(기존 parquet에 oi_change/funding_rate 없음)로 생긴 NaN을 0으로 채움
    for col in UPSERT_COLS:
        if col in existing.columns:
            existing[col] = existing[col].fillna(0.0)

    existing = existing[~existing.index.duplicated(keep='last')]
    return existing.sort_index()


def main():
    parser = argparse.ArgumentParser(
        description="바이낸스 선물 과거 캔들 수집. 단일 심볼 또는 멀티 심볼 병합 저장."
    )
    parser.add_argument("--symbols", nargs="+", default=["XRPUSDT"])
    parser.add_argument("--symbol",   default=None, help="단일 심볼 (--symbols 미사용 시)")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days",     type=int, default=365)
    parser.add_argument("--output",   default="data/combined_15m.parquet")
    parser.add_argument(
        "--no-oi", action="store_true",
        help="OI/펀딩비 수집을 건너뜀 (캔들 데이터만 저장)",
    )
    parser.add_argument(
        "--no-upsert", action="store_true",
        help="기존 parquet을 Upsert하지 않고 새로 덮어씀 (기본: Upsert 활성화)",
    )
    parser.add_argument(
        "--corr-cache-dir", default=None,
        help="상관 심볼(BTC/ETH) 캐시 디렉토리. 첫 수집 시 저장, 이후 재사용",
    )
    args = parser.parse_args()

    # --symbol 모드: 단일 거래 심볼 + 상관관계 심볼 자동 추가, 출력 경로 자동 결정
    if args.symbol:
        from src.config import Config
        try:
            cfg = Config()
            corr_symbols = cfg.correlation_symbols
        except Exception:
            corr_symbols = ["BTCUSDT", "ETHUSDT"]
        args.symbols = [args.symbol] + corr_symbols
        if args.output == "data/combined_15m.parquet":
            sym_lower = args.symbol.lower()
            os.makedirs(f"data/{sym_lower}", exist_ok=True)
            args.output = f"data/{sym_lower}/combined_15m.parquet"
    # 하위 호환: 단일 심볼만 지정된 경우
    elif args.symbols == ["XRPUSDT"] and not args.symbol:
        pass  # 기본값 유지

    if len(args.symbols) == 1:
        df = asyncio.run(fetch_klines(args.symbols[0], args.interval, args.days))
        if not args.no_oi:
            print(f"\n[OI/펀딩비] {args.symbols[0]} 수집 중...")
            df = asyncio.run(_fetch_oi_and_funding(args.symbols[0], args.days, df))
        if not args.no_upsert:
            df = upsert_parquet(args.output, df)
        df.to_parquet(args.output)
        print(f"{'Upsert' if not args.no_upsert else '저장'} 완료: {args.output} ({len(df):,}행, {len(df.columns)}컬럼)")
    else:
        # 멀티 심볼: 상관 심볼 캐시 활용
        corr_cache_dir = args.corr_cache_dir
        cached_symbols = {}
        symbols_to_fetch = list(args.symbols)

        if corr_cache_dir:
            os.makedirs(corr_cache_dir, exist_ok=True)
            remaining = []
            for sym in args.symbols:
                cache_file = os.path.join(corr_cache_dir, f"{sym.lower()}_{args.interval}.parquet")
                if os.path.exists(cache_file):
                    print(f"  [{sym}] 캐시 사용: {cache_file}")
                    cached_symbols[sym] = pd.read_parquet(cache_file)
                else:
                    remaining.append(sym)
            symbols_to_fetch = remaining

        if symbols_to_fetch:
            dfs = asyncio.run(fetch_klines_all(symbols_to_fetch, args.interval, args.days))
        else:
            dfs = {}

        # 캐시에 저장 (상관 심볼만)
        if corr_cache_dir:
            from src.config import Config
            try:
                corr_list = Config().correlation_symbols
            except Exception:
                corr_list = ["BTCUSDT", "ETHUSDT"]
            for sym, df in dfs.items():
                if sym in corr_list:
                    cache_file = os.path.join(corr_cache_dir, f"{sym.lower()}_{args.interval}.parquet")
                    df.to_parquet(cache_file)
                    print(f"  [{sym}] 캐시 저장: {cache_file}")

        # 캐시 + 새로 수집한 데이터 합치기
        dfs.update(cached_symbols)

        primary = args.symbols[0]
        merged = dfs[primary].copy()
        for symbol in args.symbols[1:]:
            suffix = "_" + symbol.lower().replace("usdt", "")
            merged = merged.join(
                dfs[symbol].add_suffix(suffix),
                how="inner",
            )

        # 주 심볼(XRP)에 대해서만 OI/펀딩비 수집 후 병합
        if not args.no_oi:
            print(f"\n[OI/펀딩비] {primary} 수집 중...")
            merged = asyncio.run(_fetch_oi_and_funding(primary, args.days, merged))

        output = args.output
        if not args.no_upsert:
            merged = upsert_parquet(output, merged)
        merged.to_parquet(output)
        print(f"\n{'Upsert' if not args.no_upsert else '병합 저장'} 완료: {output} ({len(merged):,}행, {len(merged.columns)}컬럼)")


if __name__ == "__main__":
    main()
