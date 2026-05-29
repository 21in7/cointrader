"""
BTC/ETH spot + perp + funding 데이터 취득 + 15m 정렬 + sanity (carry precheck용).

검증된 데이터 실태(2026-05-29):
  - combined_15m.parquet 은 PERP(futures_klines, fapi). spot 아님.
  - data/btcusdt·ethusdt 엔 ls_ratio 뿐 → BTC/ETH klines/funding 신규 취득 필요.
  - 네트워크 도달 OK (api.binance.com spot, fapi.binance.com funding, 무인증).

scripts/fetch_history.py 컨벤션을 따른다:
  perp = client.futures_klines, spot = client.get_klines,
  funding = /fapi/v1/fundingRate, UTC, 페이지네이션, _REQUEST_DELAY=0.3.

실행:  python -m src.carry.data            # BTC,ETH 취득 + sanity 후 STOP
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from binance import AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTERVAL = "15m"
FAPI_BASE = "https://fapi.binance.com"
REQUEST_DELAY = 0.3
DEFAULT_DAYS = 810  # ~2024-03 ~ 2026-05 (xrp combined 범위 + 여유)
UNIVERSE = ["BTCUSDT", "ETHUSDT"]


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _start_ms(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)


async def _klines(client: AsyncClient, symbol: str, days: int, market: str) -> pd.DataFrame:
    """market: 'perp'(futures_klines) | 'spot'(get_klines). OHLCV @15m, UTC index."""
    fn = client.futures_klines if market == "perp" else client.get_klines
    limit = 1500 if market == "perp" else 1000
    start_ts = _start_ms(days)
    rows = []
    while True:
        for attempt in range(3):
            try:
                k = await fn(symbol=symbol, interval=INTERVAL, startTime=start_ts, limit=limit)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"{symbol} {market} klines 실패: {e}")
        if not k:
            break
        rows.extend(k)
        last = k[-1][0]
        if last >= _now_ms() or len(k) < limit:
            break
        start_ts = last + 1
        await asyncio.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbb", "tbq", "ignore"])
    df = df[["timestamp", "close"]].copy()
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.rename(columns={"close": f"{market}_close"})


async def _funding(session: aiohttp.ClientSession, symbol: str, days: int) -> pd.DataFrame:
    """/fapi/v1/fundingRate — 정산점(8h/4h) 단위 funding. index=fundingTime."""
    url = f"{FAPI_BASE}/fapi/v1/fundingRate"
    start_ts, now = _start_ms(days), _now_ms()
    rows = []
    while start_ts < now:
        params = {"symbol": symbol, "startTime": start_ts, "limit": 1000}
        for attempt in range(3):
            try:
                async with session.get(url, params=params) as r:
                    if r.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = await r.json()
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"{symbol} funding 실패: {e}")
        if not data or not isinstance(data, list):
            break
        rows.extend(data)
        last = int(data[-1]["fundingTime"])
        if last >= now or len(data) < 1000:
            break
        start_ts = last + 1
        await asyncio.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.set_index("timestamp").sort_index()
    return df[["funding_rate"]]


async def _acquire_symbol(client, session, symbol, days):
    perp = await _klines(client, symbol, days, "perp")
    spot = await _klines(client, symbol, days, "spot")
    fund = await _funding(session, symbol, days)
    return symbol, perp, spot, fund


def _align(perp: pd.DataFrame, spot: pd.DataFrame, fund: pd.DataFrame):
    """perp·spot inner join(15m) + funding ffill(merge_asof backward)."""
    j = perp.join(spot, how="inner").dropna()
    j["basis_rel"] = (j["perp_close"] - j["spot_close"]) / j["spot_close"]
    f = fund.reset_index().rename(columns={"timestamp": "ft"})
    j2 = pd.merge_asof(
        j.reset_index().rename(columns={"timestamp": "t"}),
        f, left_on="t", right_on="ft", direction="backward")
    j2 = j2.set_index("t")
    out = j2[["spot_close", "perp_close", "basis_rel", "funding_rate"]]
    out.index.name = "timestamp"
    return out, fund


def _sanity(symbol: str, df: pd.DataFrame, fund: pd.DataFrame) -> dict:
    bar = pd.Series(df.index).diff().dropna().mode().iloc[0]
    bar_min = bar.total_seconds() / 60
    gaps = int((pd.Series(df.index).diff().dropna() != bar).sum())
    # funding cadence 자동 감지
    fdiff = pd.Series(fund.index).diff().dropna()
    cadence_h = fdiff.median().total_seconds() / 3600 if len(fdiff) else float("nan")
    settles_per_yr = (365 * 24 / cadence_h) if cadence_h else float("nan")
    fr = fund["funding_rate"]
    ann_funding = fr.mean() * settles_per_yr * 100  # 연율 % (단순 합산)
    br = df["basis_rel"] * 1e4  # bps

    print(f"\n── {symbol} ──")
    print(f"  perp∩spot rows : {len(df):,}  ({df.index[0]} ~ {df.index[-1]})")
    print(f"  bar interval   : {bar_min:.0f}분  (15m 외 간격 {gaps}개)")
    print(f"  NaN            : spot {int(df['spot_close'].isna().sum())} / "
          f"perp {int(df['perp_close'].isna().sum())} / "
          f"funding {int(df['funding_rate'].isna().sum())}")
    print(f"  funding 정산점  : {len(fund):,}개, 주기 ≈ {cadence_h:.1f}h "
          f"(={settles_per_yr:.0f}/yr), {fr.index[0]} ~ {fr.index[-1]}")
    print(f"  funding 연율(단순)= {ann_funding:+.2f}%/yr  (평균 {fr.mean()*100:+.5f}%/정산, "
          f"양(+)비율 {(fr > 0).mean()*100:.1f}%)")
    print(f"  basis_rel(perp−spot) bps: 평균 {br.mean():+.2f}  중앙 {br.median():+.2f}  "
          f"std {br.std():.2f}  [min {br.min():+.1f}, max {br.max():+.1f}]  "
          f"양(+contango)비율 {(br > 0).mean()*100:.1f}%")
    return {
        "symbol": symbol, "rows": int(len(df)),
        "range": [str(df.index[0]), str(df.index[-1])],
        "bar_minutes": bar_min, "gaps": gaps,
        "funding_settlements": int(len(fund)), "cadence_hours": float(cadence_h),
        "settles_per_year": float(settles_per_yr),
        "ann_funding_pct_simple": float(ann_funding),
        "funding_pos_ratio": float((fr > 0).mean()),
        "basis_bps_mean": float(br.mean()), "basis_bps_std": float(br.std()),
        "basis_pos_ratio": float((br > 0).mean()),
    }


def acquire(symbols=UNIVERSE, days=DEFAULT_DAYS, save=True) -> dict:
    async def _run():
        client = await AsyncClient.create()
        async with aiohttp.ClientSession() as session:
            try:
                return await asyncio.gather(
                    *[_acquire_symbol(client, session, s, days) for s in symbols])
            finally:
                await client.close_connection()

    print("=" * 78)
    print(f"  BTC/ETH spot+perp+funding 취득 ({days}일)  — carry precheck 데이터 sanity")
    print("=" * 78)
    results = asyncio.run(_run())

    sanity = {}
    for symbol, perp, spot, fund in results:
        aligned, fund2 = _align(perp, spot, fund)
        sanity[symbol] = _sanity(symbol, aligned, fund2)
        if save:
            outdir = ROOT / "data" / symbol.lower()
            outdir.mkdir(parents=True, exist_ok=True)
            aligned.to_parquet(outdir / "carry_15m.parquet")
            print(f"  저장: data/{symbol.lower()}/carry_15m.parquet")

    print("\n" + "=" * 78)
    print("  [체크포인트] 데이터 취득 + 정렬 sanity 완료. 게이트는 다음 단계(승인 후).")
    print("=" * 78)
    return sanity


if __name__ == "__main__":
    acquire()
