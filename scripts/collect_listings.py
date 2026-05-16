"""신규 상장 이벤트 데이터 수집 (이벤트 스터디용, 데이터 전용).

설계: docs/plans/2026-05-17-new-listing-microstructure-design.md

Binance fapi exchangeInfo 의 onboardDate 로 최근 N개월 신규 USDT PERP 를
식별하고, 각 상장 시점부터 1m klines 를 수집해 캐시한다.
캐시 존재 시 재요청 안 함(idempotent). 봇/src 무변경.

사용법:
  python scripts/collect_listings.py                 # 최근 24개월
  python scripts/collect_listings.py --months 12 --hours 24
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

_FAPI = "https://fapi.binance.com"
_OUT = Path("data/listings")


def _parse_args():
    p = argparse.ArgumentParser(description="신규 상장 1m 이벤트 데이터 수집")
    p.add_argument("--months", type=int, default=24, help="최근 N개월 상장 (기본 24)")
    p.add_argument("--hours", type=int, default=25, help="상장 후 수집 시간 (기본 25h)")
    p.add_argument("--delay", type=float, default=0.25, help="요청 간 딜레이(초)")
    return p.parse_args()


def _get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == tries - 1:
                print(f"  [err] {e}")
                return None
            time.sleep(2 ** i)
    return None


def main():
    args = _parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    ex = _get(f"{_FAPI}/fapi/v1/exchangeInfo", {})
    if not ex:
        print("[ERR] exchangeInfo 실패", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.months * 30)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    perp = [
        s for s in ex["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("onboardDate")
        and int(s["onboardDate"]) >= cutoff_ms
    ]
    perp.sort(key=lambda s: int(s["onboardDate"]))
    print(f"대상 신규 상장: {len(perp)}개 (최근 {args.months}개월)")

    bars_needed = args.hours * 60  # 1m
    ok = skip = fail = 0
    for i, s in enumerate(perp):
        sym = s["symbol"]
        ob = int(s["onboardDate"])
        out = _OUT / f"{sym}.parquet"
        if out.exists():
            skip += 1
            continue
        rows = []
        start = ob
        while len(rows) < bars_needed:
            k = _get(f"{_FAPI}/fapi/v1/klines", {
                "symbol": sym, "interval": "1m",
                "startTime": start, "limit": 1500,
            })
            if not k or isinstance(k, dict) or len(k) == 0:
                break
            rows.extend(k)
            start = k[-1][0] + 60_000
            time.sleep(args.delay)
            if len(k) < 1500:
                break
        if not rows:
            fail += 1
            print(f"  [{i+1}/{len(perp)}] {sym}: klines 없음")
            continue
        rows = rows[:bars_needed]
        df = pd.DataFrame(rows, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "ct", "qv", "nt", "tb", "tq", "ig"])
        df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df.attrs["onboard_ms"] = ob
        df.to_parquet(out)
        ok += 1
        if (i + 1) % 25 == 0 or i == len(perp) - 1:
            print(f"  [{i+1}/{len(perp)}] {sym} 수집 (ok={ok} skip={skip} fail={fail})")
        time.sleep(args.delay)

    print(f"\n완료: 신규수집={ok} 캐시스킵={skip} 실패={fail} → {_OUT}")


if __name__ == "__main__":
    main()
