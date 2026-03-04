"""
OI 장기 수집 스크립트.
15분마다 cron 실행하여 Binance OI를 data/oi_history.parquet에 누적한다.

사용법:
  python scripts/collect_oi.py
  python scripts/collect_oi.py --symbol XRPUSDT

crontab 예시:
  */15 * * * * cd /path/to/cointrader && .venv/bin/python scripts/collect_oi.py >> logs/collect_oi.log 2>&1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os

load_dotenv()

OI_PATH = Path("data/oi_history.parquet")


def collect(symbol: str = "XRPUSDT"):
    client = Client(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
    )

    result = client.futures_open_interest(symbol=symbol)
    oi_value = float(result["openInterest"])
    ts = datetime.now(timezone.utc)

    new_row = pd.DataFrame([{
        "timestamp": ts,
        "symbol": symbol,
        "open_interest": oi_value,
    }])

    if OI_PATH.exists():
        existing = pd.read_parquet(OI_PATH)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        OI_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined = new_row

    combined.to_parquet(OI_PATH, index=False)
    print(f"[{ts.isoformat()}] OI={oi_value:.2f} → {OI_PATH}")


def main():
    parser = argparse.ArgumentParser(description="OI 장기 수집")
    parser.add_argument("--symbol", default="XRPUSDT")
    args = parser.parse_args()
    collect(symbol=args.symbol)


if __name__ == "__main__":
    main()
