"""센티먼트 프로바이더 — TradingAgents Sentiment Analyst 추출본.

TradingAgents(TauricResearch)의 Sentiment Analyst에서 가치 있는 부분만
벤더링·트림한 모듈:
  - 3소스(뉴스/StockTwits/Reddit) 사전수집 → 프롬프트 주입 패턴
  - 군중심리 해석 best-practice 프롬프트 (StockTwits 비율, 교차소스 괴리,
    engagement 가중, 과열 시 contrarian 경고)
  - graceful-degradation fetcher

LangChain 플러밍은 제거했다. 프로젝트 .venv에 langchain/openai가 없고,
LLM-free 봇에 거대 의존성을 더하는 것은 과도하기 때문이다. 로컬 MLX 서버가
OpenAI 호환이므로 이미 설치된 ``httpx``로 ``/v1/chat/completions``를 직접
호출한다. 분석기 출력은 prose 리포트 대신 **엄격 JSON**으로 강제하여 정규식
파싱 단계를 없애고 결정론을 강화했다.

설계 문서: docs/plans/2026-05-16-tradingagents-sentiment-fusion-design.md

본 모듈은 동기(sync) API다. ml_filter.py와 같은 스타일이며, 향후 비동기
사이드카는 ``asyncio.to_thread(analyze_sentiment, ...)``로 감싸 호출한다.
모든 외부 호출(LLM/HTTP)은 실패 시 None 또는 placeholder를 반환하며 예외를
표면화하지 않는다(graceful degradation).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import httpx
from loguru import logger

# ── 5등급 센티먼트 어휘 → 스코어 매핑 ───────────────────────────────────
# post-mortem 정합: 군중/뉴스 센티먼트의 *극단*이 신호다. 5등급을 [-1,+1]
# 연속값으로 환산해 게이트(veto/contrarian)에서 임계 비교에 사용한다.
SENTIMENT_SCALE: dict[str, float] = {
    "VeryBearish": -1.0,
    "Bearish": -0.5,
    "Neutral": 0.0,
    "Bullish": 0.5,
    "VeryBullish": 1.0,
}
_LABELS = tuple(SENTIMENT_SCALE.keys())

# ── 환경설정 (기본값 = 실측 확인된 로컬 MLX) ────────────────────────────
_DEFAULT_BASE_URL = "http://localhost:8080/v1"
_DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"


def _env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def _base_url() -> str:
    return _env("SENTIMENT_LLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return _env("SENTIMENT_LLM_MODEL", _DEFAULT_MODEL)


def _timeout() -> float:
    try:
        return float(_env("SENTIMENT_LLM_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def _cache_dir() -> Path:
    d = Path(_env("SENTIMENT_CACHE_DIR", "data/sentiment_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 결과 모델 ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SentimentScore:
    """단일 센티먼트 판정 결과.

    score: [-1.0, +1.0] (VeryBearish ~ VeryBullish)
    confidence: [0.0, 1.0] 데이터 품질/표본 기반 모델 자기보고
    asof: 이 판정이 적용되는 기준 시각(ISO8601, UTC). 백테스트 정렬용.
    """

    label: str
    score: float
    confidence: float
    rationale: str
    asof: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "asof": self.asof,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SentimentScore":
        return cls(
            label=str(d["label"]),
            score=float(d["score"]),
            confidence=float(d.get("confidence", 0.0)),
            rationale=str(d.get("rationale", "")),
            asof=d.get("asof"),
        )


# ── 프롬프트 (TradingAgents sentiment_analyst 트림 + 크립토/JSON 적응) ──
_SYSTEM_PROMPT = """You are a cryptocurrency market sentiment analyst. Produce a single sentiment read for the asset {asset} covering roughly the past 24 hours, drawing only on the pre-collected data blocks below. Do not invent data not present here.

### News headlines & summaries
Institutional/journalistic framing. Fact-driven, slower-moving.
<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages (retail traders, cashtag-indexed)
Fast-moving retail signal. Each message may carry a user Bullish/Bearish tag.
<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts (crypto communities)
Community discussion; weight by engagement (upvotes/comments).
<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to judge
1. Read StockTwits Bullish/Bearish ratio as a leading retail signal. A ~70/30 split is moderately bullish; **>=90/10 indicates over-extension / contrarian risk**, not strength. 50/50 is uncertainty. Weight by message count, not percentage alone.
2. Cross-source divergence is itself signal (e.g. bearish news vs euphoric retail).
3. Weight Reddit by engagement; ignore low-score noise.
4. Crowd euphoria and panic are often contrarian at extremes — reflect that in the label, not just the surface mood.
5. Be honest about data limits: if a source is "<unavailable>" or sparse, lower your confidence.

## Output (STRICT)
Respond with ONE JSON object and nothing else. No markdown, no prose, no code fences. Schema:
{{"label": one of ["VeryBearish","Bearish","Neutral","Bullish","VeryBullish"], "confidence": float 0.0-1.0, "rationale": short string <= 240 chars}}
"""


def _build_messages(
    *, asset: str, news_block: str, stocktwits_block: str, reddit_block: str
) -> list[dict]:
    system = _SYSTEM_PROMPT.format(
        asset=asset,
        news_block=news_block.strip() or "<unavailable>",
        stocktwits_block=stocktwits_block.strip() or "<unavailable>",
        reddit_block=reddit_block.strip() or "<unavailable>",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Produce the JSON sentiment object for {asset} now."},
    ]


# ── JSON 견고 추출 ──────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    """LLM 응답에서 첫 JSON 객체를 견고하게 추출한다.

    엄격 출력을 지시하지만 모델이 코드펜스/잡설을 덧붙일 수 있으므로,
    전체 파싱 실패 시 중괄호 균형 스캔으로 첫 객체만 떼어낸다.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalize_label(raw: str) -> str | None:
    """모델 라벨을 정규 5등급으로 정규화 (대소문자/공백/하이픈 허용)."""
    if not raw:
        return None
    key = re.sub(r"[\s_\-]+", "", str(raw)).lower()
    for lbl in _LABELS:
        if key == lbl.lower():
            return lbl
    return None


def _parse_score(obj: dict, asof: str | None) -> SentimentScore | None:
    label = _normalize_label(obj.get("label", ""))
    if label is None:
        logger.warning(f"[sentiment] 알 수 없는 label: {obj.get('label')!r}")
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    rationale = str(obj.get("rationale", ""))[:240]
    return SentimentScore(
        label=label,
        score=SENTIMENT_SCALE[label],
        confidence=conf,
        rationale=rationale,
        asof=asof,
    )


# ── 캐시 (결정론 재현 + 비용 0) ─────────────────────────────────────────
def _cache_key(model: str, messages: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def _cache_get(key: str) -> SentimentScore | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return SentimentScore.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        logger.warning(f"[sentiment] 캐시 손상 무시: {p.name}: {e}")
        return None


def _cache_put(key: str, score: SentimentScore) -> None:
    p = _cache_path(key)
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(score.to_dict(), ensure_ascii=False))
        tmp.replace(p)  # atomic
    except OSError as e:
        logger.warning(f"[sentiment] 캐시 저장 실패 (무시): {e}")


# ── 핵심 API ────────────────────────────────────────────────────────────
def analyze_sentiment(
    asset: str,
    news_block: str = "",
    stocktwits_block: str = "",
    reddit_block: str = "",
    *,
    asof: str | None = None,
    use_cache: bool = True,
) -> SentimentScore | None:
    """3소스 텍스트로 단일 센티먼트 판정을 반환한다.

    결정론: ``temperature=0`` + ``seed=0`` + 응답 캐시. 동일 입력은 캐시
    히트로 비트단위 동일 결과를 보장한다(validator 백테스트 재현성 핵심).

    실패(네트워크/파싱/알수없는 라벨) 시 None을 반환한다. 호출측은 None을
    '게이트 통과(pass-through)'로 취급한다(ml_filter 미로드와 동일 철학).
    """
    model = _model()
    messages = _build_messages(
        asset=asset,
        news_block=news_block,
        stocktwits_block=stocktwits_block,
        reddit_block=reddit_block,
    )
    key = _cache_key(model, messages)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            # asof는 호출 컨텍스트마다 다르므로 캐시값에 현재 asof를 입힌다.
            return SentimentScore(
                label=cached.label,
                score=cached.score,
                confidence=cached.confidence,
                rationale=cached.rationale,
                asof=asof,
            )

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 256,
        "stream": False,
    }
    url = f"{_base_url()}/chat/completions"
    try:
        resp = httpx.post(url, json=body, timeout=_timeout())
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
        logger.warning(f"[sentiment] LLM 호출 실패 (graceful None): {e}")
        return None

    obj = _extract_json(content)
    if obj is None:
        logger.warning(f"[sentiment] JSON 추출 실패: {content[:200]!r}")
        return None

    score = _parse_score(obj, asof)
    if score is None:
        return None

    if use_cache:
        _cache_put(key, score)
    return score


# ── 라이브 fetcher (크립토 적응, stdlib·graceful) ───────────────────────
# 백테스트는 Alpha Vantage 과거 뉴스 텍스트를 쓰므로 아래 fetcher는
# 라이브 사이드카(게이트 C) 전용이다. 설계 §8 step1 완전성 위해 포함.
_UA = "cointrader-sentiment/0.1"
_STOCKTWITS = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_REDDIT = "https://www.reddit.com/r/{sub}/search.json?{qs}"
CRYPTO_SUBREDDITS = ("CryptoCurrency", "CryptoMarkets")


def fetch_stocktwits_messages(ticker: str, limit: int = 30, timeout: float = 10.0) -> str:
    """StockTwits 심볼 스트림 (크립토는 보통 ``BTC.X`` 형태 캐시태그)."""
    url = _STOCKTWITS.format(ticker=ticker.upper())
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError, ValueError) as e:
        logger.warning(f"[sentiment] StockTwits 실패 {ticker}: {e}")
        return f"<stocktwits unavailable: {type(e).__name__}>"
    msgs = data.get("messages", []) if isinstance(data, dict) else []
    if not msgs:
        return f"<no StockTwits messages for ${ticker.upper()}>"
    lines, bull, bear, none_ = [], 0, 0, 0
    for m in msgs[:limit]:
        ent = (m.get("entities") or {}).get("sentiment") or {}
        s = ent.get("basic") if isinstance(ent, dict) else None
        body = (m.get("body") or "").replace("\n", " ").strip()[:280]
        if s == "Bullish":
            bull += 1
            tag = "Bullish"
        elif s == "Bearish":
            bear += 1
            tag = "Bearish"
        else:
            none_ += 1
            tag = "no-label"
        lines.append(f"[{m.get('created_at','')} · {tag}] {body}")
    total = bull + bear + none_
    head = (
        f"Bullish:{bull}({round(100*bull/total) if total else 0}%) "
        f"Bearish:{bear}({round(100*bear/total) if total else 0}%) "
        f"Unlabeled:{none_} Total:{total}"
    )
    return head + "\n\n" + "\n".join(lines)


def fetch_reddit_posts(
    query: str,
    subreddits: Iterable[str] = CRYPTO_SUBREDDITS,
    limit_per_sub: int = 8,
    timeout: float = 10.0,
) -> str:
    """크립토 subreddit에서 ``query`` 관련 최근 7일 게시물."""
    import time

    blocks: list[str] = []
    total = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(0.4)  # reddit 공개 한도 ~10 req/min
        qs = urlencode(
            {"q": query, "restrict_sr": "on", "sort": "new", "t": "week", "limit": limit_per_sub}
        )
        url = _REDDIT.format(sub=sub, qs=qs)
        req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        try:
            with urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
        except (HTTPError, URLError, json.JSONDecodeError, TimeoutError, ValueError) as e:
            logger.warning(f"[sentiment] Reddit 실패 r/{sub}: {e}")
            blocks.append(f"r/{sub}: <unavailable>")
            continue
        children = (payload.get("data") or {}).get("children") or []
        posts = [c.get("data", {}) for c in children if isinstance(c, dict)]
        total += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts for {query} past 7d>")
            continue
        ls = [f"r/{sub} — {len(posts)} posts:"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            ls.append(f"  [{p.get('score',0)}↑ {p.get('num_comments',0)}c] {title}")
        blocks.append("\n".join(ls))
    if total == 0:
        return f"<no Reddit posts for {query}>"
    return "\n\n".join(blocks)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
