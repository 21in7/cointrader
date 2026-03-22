"""
Long/Short Ratio 장기 수집 스크립트.
15분마다 cron 실행하여 Binance Trading Data API에서
top_acct_ls_ratio, global_ls_ratio를 data/{symbol}/ls_ratio_15m.parquet에 누적한다.

수집 대상:
  - topLongShortAccountRatio × 3심볼 (XRPUSDT, BTCUSDT, ETHUSDT)
  - globalLongShortAccountRatio × 3심볼 (XRPUSDT, BTCUSDT, ETHUSDT)
  → 총 API 호출 6회/15분 (rate limit 무관)

사용법:
  python scripts/collect_ls_ratio.py
  python scripts/collect_ls_ratio.py --symbols XRPUSDT BTCUSDT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
from datetime import datetime, timezone

import aiohttp
import pandas as pd

BASE_URL = "https://fapi.binance.com"
DEFAULT_SYMBOLS = ["XRPUSDT", "BTCUSDT", "ETHUSDT"]

ENDPOINTS = {
    "top_acct_ls_ratio": "/futures/data/topLongShortAccountRatio",
    "global_ls_ratio": "/futures/data/globalLongShortAccountRatio",
}


async def fetch_latest(session: aiohttp.ClientSession, symbol: str) -> dict | None:
    """심볼 하나에 대해 두 ratio의 최신 1건씩 가져온다."""
    row = {"timestamp": None, "symbol": symbol}

    for col_name, endpoint in ENDPOINTS.items():
        url = f"{BASE_URL}{endpoint}"
        params = {"symbol": symbol, "period": "15m", "limit": 1}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if isinstance(data, list) and data:
                    row[col_name] = float(data[0]["longShortRatio"])
                    # 타임스탬프는 첫 번째 응답에서 설정
                    if row["timestamp"] is None:
                        row["timestamp"] = pd.Timestamp(
                            int(data[0]["timestamp"]), unit="ms", tz="UTC"
                        )
                else:
                    print(f"[WARN] {symbol} {col_name}: unexpected response: {data}")
                    return None
        except Exception as e:
            print(f"[ERROR] {symbol} {col_name}: {e}")
            return None

    return row


async def collect(symbols: list[str]):
    """모든 심볼 데이터를 수집하고 parquet에 추가한다."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_latest(session, sym) for sym in symbols]
        results = await asyncio.gather(*tasks)

    now = datetime.now(timezone.utc)
    collected = 0

    for row in results:
        if row is None:
            continue

        symbol = row["symbol"]
        out_path = Path(f"data/{symbol.lower()}/ls_ratio_15m.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([{
            "timestamp": row["timestamp"],
            "top_acct_ls_ratio": row["top_acct_ls_ratio"],
            "global_ls_ratio": row["global_ls_ratio"],
        }])

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            # 중복 방지: 동일 timestamp가 이미 있으면 스킵
            if row["timestamp"] in existing["timestamp"].values:
                print(f"[SKIP] {symbol} ts={row['timestamp']} already exists")
                continue
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(out_path, index=False)
        collected += 1
        print(
            f"[{now.isoformat()}] {symbol}: "
            f"top_acct={row['top_acct_ls_ratio']:.4f}, "
            f"global={row['global_ls_ratio']:.4f} "
            f"→ {out_path} ({len(combined)} rows)"
        )

    if collected == 0:
        print(f"[{now.isoformat()}] No new data collected")


def main():
    parser = argparse.ArgumentParser(description="L/S Ratio 장기 수집")
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="수집 대상 심볼 (기본: XRPUSDT BTCUSDT ETHUSDT)",
    )
    args = parser.parse_args()
    asyncio.run(collect(args.symbols))


if __name__ == "__main__":
    main()
