"""Alpha Vantage 과거 크립토 뉴스 → 로컬 MLX 일괄 추론 → 15m 정렬 센티먼트 데이터셋.

게이트 A 산출물. 설계: docs/plans/2026-05-16-tradingagents-sentiment-fusion-design.md

흐름:
  1. data/{symbol}/combined_15m.parquet 의 15m UTC 그리드 로드 ([start,end])
  2. Alpha Vantage NEWS_SENTIMENT(크립토, 과거 시계열)를 시간창 단위로 수집,
     원본 feed를 data/sentiment_cache/ 에 캐시 (재실행 시 재요청 없음)
  3. cadence(기본 4h) 시각마다, 그 시점까지 공개된 기사만(look-ahead 가드:
     time_published <= cadence, 직전 lookback 시간 윈도우) 모아 뉴스 블록 구성
  4. src.sentiment_provider.analyze_sentiment 로 단일 판정 (응답 캐시 → 결정론)
  5. cadence 스코어를 15m 그리드에 merge_asof(backward) 로 forward-fill
  6. data/{symbol}/sentiment_15m.parquet 저장 (timestamp UTC 인덱스, combined와 정렬)

사용법:
  export ALPHAVANTAGE_API_KEY=...        # 없으면 --dry-run 만 가능
  python scripts/build_sentiment_dataset.py --symbol XRPUSDT --asset XRP \
      --start 2024-06-01 --end 2026-05-01
  python scripts/build_sentiment_dataset.py --symbol XRPUSDT --asset XRP --dry-run

주의:
  - Alpha Vantage 무료 티어는 25 req/day. --max-av-calls 로 상한을 두고,
    원본 feed 캐시로 중단/재개를 지원한다(idempotent).
  - AV time_published 에는 타임존이 없다. AV의 time_from/time_to 와 동일
    스케일이므로 동일 스케일로 비교하면 look-ahead 가드는 안전하다. 캔들
    그리드와의 정렬 보정이 필요하면 --publish-tz-offset-hours 로 조정.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
import os

from src.sentiment_provider import analyze_sentiment

load_dotenv()

_AV_URL = "https://www.alphavantage.co/query"
_CACHE_DIR = Path("data/sentiment_cache")
_AV_TS_FMT = "%Y%m%dT%H%M%S"  # AV time_published
_AV_PARAM_FMT = "%Y%m%dT%H%M"  # AV time_from / time_to


def _parse_args():
    p = argparse.ArgumentParser(description="센티먼트 데이터셋 빌더 (Alpha Vantage + 로컬 MLX)")
    p.add_argument("--symbol", required=True, help="예: XRPUSDT")
    p.add_argument("--asset", required=True, help="자산 표시명, 예: XRP")
    p.add_argument("--av-ticker", default=None, help="AV 티커. 기본 CRYPTO:{base}")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--cadence-hours", type=float, default=4.0, help="센티먼트 산출 주기 (기본 4h)")
    p.add_argument("--lookback-hours", type=float, default=24.0, help="각 cadence의 기사 윈도우 (기본 24h)")
    p.add_argument("--window-days", type=int, default=7, help="AV 수집 페이지 창 (기본 7일)")
    p.add_argument("--max-articles", type=int, default=40, help="cadence당 프롬프트 기사 수 상한")
    p.add_argument("--max-av-calls", type=int, default=20, help="AV 호출 상한 (무료 티어 보호)")
    p.add_argument("--publish-tz-offset-hours", type=float, default=0.0, help="AV time_published 보정")
    p.add_argument("--out", default=None, help="출력 parquet (기본 data/{symbol}/sentiment_15m.parquet)")
    p.add_argument("--dry-run", action="store_true", help="AV/LLM 네트워크 미사용 (캐시만)")
    return p.parse_args()


def _utc(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _load_grid(symbol: str, start: datetime, end: datetime) -> pd.DatetimeIndex:
    path = Path(f"data/{symbol.lower()}/combined_15m.parquet")
    if not path.exists():
        print(f"[ERR] 그리드 없음: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path, columns=[])
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx[(idx >= start) & (idx <= end)]
    return idx.sort_values()


def _av_cache_path(ticker: str, w_from: datetime, w_to: datetime) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"av_{ticker.replace(':','-')}_{w_from.strftime('%Y%m%d')}_{w_to.strftime('%Y%m%d')}.json"
    return _CACHE_DIR / tag


def _fetch_av_window(
    ticker: str, w_from: datetime, w_to: datetime, api_key: str, dry_run: bool
) -> list[dict]:
    """한 시간창의 AV NEWS_SENTIMENT feed. 캐시 우선, idempotent."""
    cache = _av_cache_path(ticker, w_from, w_to)
    if cache.exists():
        try:
            return json.loads(cache.read_text()).get("feed", [])
        except (json.JSONDecodeError, OSError):
            pass
    if dry_run:
        return []
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": w_from.strftime(_AV_PARAM_FMT),
        "time_to": w_to.strftime(_AV_PARAM_FMT),
        "limit": "1000",
        "sort": "EARLIEST",
        "apikey": api_key,
    }
    for attempt in range(3):
        try:
            r = requests.get(_AV_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  [AV] 요청 실패 ({attempt+1}/3): {e}")
            time.sleep(2 ** (attempt + 1))
            continue
        # 레이트리밋/안내 메시지 감지
        if "Note" in data or "Information" in data:
            print(f"  [AV] 한도/안내: {data.get('Note') or data.get('Information')}")
            return []
        cache.write_text(json.dumps(data, ensure_ascii=False))
        return data.get("feed", [])
    return []


def _collect_articles(
    ticker: str, start: datetime, end: datetime, window_days: int,
    api_key: str, dry_run: bool, max_calls: int,
) -> list[dict]:
    """[start,end] 전체를 시간창으로 페이지네이션해 기사 누적."""
    out: list[dict] = []
    cur = start
    calls = 0
    while cur < end:
        w_to = min(cur + timedelta(days=window_days), end)
        cache = _av_cache_path(ticker, cur, w_to)
        cached = cache.exists()
        if not cached and not dry_run:
            if calls >= max_calls:
                print(f"  [AV] 호출 상한({max_calls}) 도달 — 나머지 구간 미수집 (재실행으로 재개)")
                break
            calls += 1
        feed = _fetch_av_window(ticker, cur, w_to, api_key, dry_run)
        out.extend(feed)
        print(f"  [AV] {cur:%Y-%m-%d}~{w_to:%Y-%m-%d}: {len(feed)}건"
              f"{' (cache)' if cached else ''}")
        cur = w_to
        if not cached and not dry_run:
            time.sleep(13)  # 무료 5 req/min 보호
    return out


def _parse_pub(ts: str, offset_h: float) -> datetime | None:
    try:
        dt = datetime.strptime(ts, _AV_TS_FMT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    return dt + timedelta(hours=offset_h)


def select_window(
    articles: list[dict], cadence: datetime, lookback: timedelta
) -> list[dict]:
    """look-ahead 가드의 핵심: cadence 시점까지 공개된 기사만 선택한다.

    조건: cadence - lookback < pub <= cadence  (strict <= cadence).
    미래 기사(pub > cadence)는 절대 포함하지 않는다. articles 는 pub
    오름차순 정렬돼 있다고 가정한다(반환도 동일 순서 유지).
    """
    lo = cadence - lookback
    return [a for a in articles if lo < a["pub"] <= cadence]


def main():
    args = _parse_args()
    base = args.symbol.upper().replace("USDT", "").replace("USDC", "")
    av_ticker = args.av_ticker or f"CRYPTO:{base}"
    start, end = _utc(args.start), _utc(args.end)
    out_path = Path(args.out) if args.out else Path(
        f"data/{args.symbol.lower()}/sentiment_15m.parquet"
    )
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not api_key and not args.dry_run:
        print("[ERR] ALPHAVANTAGE_API_KEY 미설정. --dry-run 으로만 실행 가능", file=sys.stderr)
        sys.exit(1)

    print(f"=== 센티먼트 데이터셋: {args.symbol} ({av_ticker}) {args.start}~{args.end} ===")
    print(f"  cadence={args.cadence_hours}h lookback={args.lookback_hours}h "
          f"dry_run={args.dry_run}")

    grid = _load_grid(args.symbol, start, end)
    if len(grid) == 0:
        print("[ERR] 그리드가 비었습니다 (기간 확인)", file=sys.stderr)
        sys.exit(1)
    print(f"  15m 그리드: {len(grid):,}개 ({grid[0]} ~ {grid[-1]})")

    # 1) 기사 수집
    feed = _collect_articles(
        av_ticker, start - timedelta(hours=args.lookback_hours), end,
        args.window_days, api_key, args.dry_run, args.max_av_calls,
    )
    articles = []
    for a in feed:
        pub = _parse_pub(a.get("time_published", ""), args.publish_tz_offset_hours)
        if pub is None:
            continue
        articles.append({
            "pub": pub,
            "title": (a.get("title") or "").strip(),
            "summary": (a.get("summary") or "").strip(),
            "av_score": a.get("overall_sentiment_score"),
        })
    articles.sort(key=lambda x: x["pub"])
    print(f"  파싱된 기사: {len(articles):,}건")

    # 2) cadence 루프 (look-ahead 가드)
    cadence = timedelta(hours=args.cadence_hours)
    lookback = timedelta(hours=args.lookback_hours)
    rows = []
    c = start
    n_scored = n_skip = 0
    while c <= end:
        win = select_window(articles, c, lookback)  # look-ahead 가드
        if not win:
            c += cadence
            n_skip += 1
            continue
        recent = win[-args.max_articles:]
        block = "\n".join(
            f"[{x['pub']:%Y-%m-%d %H:%M}] {x['title']} — {x['summary'][:200]}"
            for x in recent
        )
        av_vals = [float(x["av_score"]) for x in win
                   if isinstance(x["av_score"], (int, float, str))
                   and str(x["av_score"]).replace("-", "").replace(".", "").isdigit()]
        sc = analyze_sentiment(args.asset, news_block=block, asof=c.isoformat())
        if sc is None:
            n_skip += 1
            c += cadence
            continue
        rows.append({
            "timestamp": c,
            "sentiment_score": sc.score,
            "sentiment_label": sc.label,
            "sentiment_conf": sc.confidence,
            "sentiment_asof": c.isoformat(),
            "av_score": sum(av_vals) / len(av_vals) if av_vals else float("nan"),
        })
        n_scored += 1
        c += cadence
    print(f"  cadence 판정: {n_scored}건, 스킵(데이터없음/실패): {n_skip}건")

    if not rows:
        print("[ERR] 판정 0건 — AV 키/캐시/기간 확인 (dry-run은 캐시 필요)", file=sys.stderr)
        sys.exit(1)

    cad_df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    cad_df.index = pd.to_datetime(cad_df.index, utc=True)

    # 3) 15m 그리드에 backward merge_asof (forward-fill, look-ahead 없음)
    grid_df = pd.DataFrame(index=grid)
    grid_df.index.name = "timestamp"
    left = grid_df.reset_index()
    right = cad_df.reset_index()
    # merge_asof는 키 dtype/해상도가 정확히 일치해야 한다. parquet은 ms,
    # 구성한 cadence는 us/ns → 양쪽을 ns,UTC로 통일하고 정렬 보장.
    left["timestamp"] = left["timestamp"].astype("datetime64[ns, UTC]")
    right["timestamp"] = right["timestamp"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    cov = merged["sentiment_score"].notna().mean() * 100
    dist = merged["sentiment_label"].value_counts(dropna=True).to_dict()
    print(f"  그리드 커버리지: {cov:.1f}%  라벨분포: {dist}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(f"=== 저장 완료: {out_path} ({len(merged):,} rows) ===")


if __name__ == "__main__":
    main()
