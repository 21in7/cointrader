"""온체인 일봉 메트릭 취득 (Coin Metrics community API) — netflow precheck용.

Coin Metrics Community Metrics 는 Network Data Pro 의 무료 부분집합. API 키 불필요,
rate 10req/6s, CC 라이선스. 거래소 in/outflow(FlowInExNtv/FlowOutExNtv)가 BTC는
2012-12-30부터 무료 제공됨 (실 호출로 검증).

거래소 플로우는 가격·OI·펀딩 등 거래소 *내부* 신호와 정보 차원이 다른 유일한 미탐색
데이터 → netflow directional precheck 의 입력.

scripts/fetch_history.py · src/momentum/data.py 컨벤션(UTC, 페이지네이션, sanity 후
STOP)을 따른다. 게이트는 다음 단계(src/onchain/precheck.py, 승인 후).

실행:  python -m src.onchain.data       # BTC/ETH 온체인 일봉 취득 + sanity 후 STOP
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["FlowInExNtv", "FlowOutExNtv", "PriceUSD", "CapMVRVCur",
           "AdrActCnt", "TxCnt"]   # 바스켓(접근2): 밸류에이션·네트워크활동 추가
ASSETS = ["btc", "eth"]
START_TIME = "2010-01-01"          # 상장 이후 전체 (실제 가용 시점부터 채워짐)
PAGE_SIZE = 10000
REQUEST_DELAY = 0.7                # community 10req/6s 안전 마진
VALUE_COLS = METRICS               # 숫자 변환 대상


def _fetch_page(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                raise RuntimeError(f"CM fetch 실패: {e}\nurl={url}") from e
    return {}


def _fetch_asset(asset: str) -> pd.DataFrame:
    """단일 자산 전체 히스토리 페이지네이션 취득 → date-indexed DataFrame."""
    params = {
        "assets": asset,
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": START_TIME,
        "page_size": str(PAGE_SIZE),
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    rows: list[dict] = []
    while url:
        payload = _fetch_page(url)
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
        if url:
            time.sleep(REQUEST_DELAY)

    if not rows:
        raise RuntimeError(f"{asset}: 데이터 0행 (메트릭/티어 확인)")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    keep = ["timestamp"] + [c for c in VALUE_COLS if c in df.columns]
    df = df[keep].copy()
    for c in VALUE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="first")]


def acquire(assets=ASSETS, save=True) -> dict:
    print("=" * 80)
    print("  온체인 일봉 취득 (Coin Metrics community) — netflow precheck 데이터 sanity")
    print("=" * 80)

    sanity = {}
    one_day = pd.Timedelta(days=1)
    for asset in assets:
        df = _fetch_asset(asset)
        # netflow 가용구간 = FlowIn·FlowOut·PriceUSD 모두 존재하는 첫 시점부터
        core = df.dropna(subset=["FlowInExNtv", "FlowOutExNtv", "PriceUSD"])
        netflow = (core["FlowInExNtv"] - core["FlowOutExNtv"])
        diffs = pd.Series(core.index).diff().dropna()
        gaps = int((diffs != one_day).sum())
        sanity[asset] = {
            "rows_all": int(len(df)),
            "rows_core": int(len(core)),
            "start_core": str(core.index[0].date()) if len(core) else None,
            "end_core": str(core.index[-1].date()) if len(core) else None,
            "gaps": gaps,
            "nan_flowin": int(df["FlowInExNtv"].isna().sum()),
            "nan_price": int(df["PriceUSD"].isna().sum()),
            "years": round((core.index[-1] - core.index[0]).days / 365.25, 2) if len(core) else 0,
            "netflow_mean": float(netflow.mean()) if len(netflow) else None,
            "netflow_std": float(netflow.std()) if len(netflow) else None,
        }
        s = sanity[asset]
        print(f"  {asset.upper():4s} 전체={s['rows_all']:>5d}행  netflow가용={s['rows_core']:>5d}행  "
              f"{s['start_core']} ~ {s['end_core']} ({s['years']:.1f}yr)")
        print(f"       1일外갭={gaps}  NaN(flowin)={s['nan_flowin']}  NaN(price)={s['nan_price']}  "
              f"netflow μ={s['netflow_mean']:.0f} σ={s['netflow_std']:.0f} coins")
        if save:
            outdir = ROOT / "data" / f"{asset}usdt"
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(outdir / "onchain_daily.parquet")

    if save:
        print(f"\n  저장: data/{{btc,eth}}usdt/onchain_daily.parquet")

    print("\n" + "=" * 80)
    print("  [체크포인트] 온체인 취득 + sanity 완료. 게이트는 다음 단계(precheck.py, 승인 후).")
    print("=" * 80)
    return sanity


if __name__ == "__main__":
    acquire()
