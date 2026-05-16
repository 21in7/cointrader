"""sentiment_provider 결정론·graceful 회귀 테스트 + look-ahead 가드 테스트.

게이트 A 검증. 외부 LLM(MLX) 호출은 전부 모킹한다(실네트워크 금지).
핵심 보증: 동일 입력 → 캐시 → 비트단위 동일 결과(validator 백테스트 재현성).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.sentiment_provider import (
    SENTIMENT_SCALE,
    analyze_sentiment,
    _extract_json,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """모든 테스트를 격리된 임시 캐시 디렉토리에서 실행."""
    monkeypatch.setenv("SENTIMENT_CACHE_DIR", str(tmp_path / "scache"))
    monkeypatch.setenv("SENTIMENT_LLM_MODEL", "test-model")
    monkeypatch.setenv("SENTIMENT_LLM_BASE_URL", "http://localhost:9999/v1")


def _mock_resp(content: str) -> MagicMock:
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def test_parses_and_maps_score():
    content = '{"label":"VeryBullish","confidence":0.9,"rationale":"euphoria everywhere"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)):
        sc = analyze_sentiment("XRP", news_block="big rally", asof="2026-05-01T00:00:00+00:00")
    assert sc is not None
    assert sc.label == "VeryBullish"
    assert sc.score == 1.0
    assert sc.confidence == 0.9
    assert sc.asof == "2026-05-01T00:00:00+00:00"


def test_deterministic_cache_hit():
    """동일 입력 2회 호출 → 네트워크 1회만, 결과 동일."""
    content = '{"label":"Bearish","confidence":0.7,"rationale":"fear"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)) as mp:
        a = analyze_sentiment("XRP", news_block="dump", asof="2026-01-01T00:00:00+00:00")
        b = analyze_sentiment("XRP", news_block="dump", asof="2026-02-02T00:00:00+00:00")
    assert mp.call_count == 1  # 두 번째는 캐시 히트
    assert a.label == b.label == "Bearish"
    assert a.score == b.score == -0.5
    # asof는 호출 컨텍스트별로 갱신된다
    assert a.asof == "2026-01-01T00:00:00+00:00"
    assert b.asof == "2026-02-02T00:00:00+00:00"


def test_use_cache_false_always_calls():
    content = '{"label":"Neutral","confidence":0.5,"rationale":"flat"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)) as mp:
        analyze_sentiment("XRP", news_block="x", use_cache=False)
        analyze_sentiment("XRP", news_block="x", use_cache=False)
    assert mp.call_count == 2


def test_graceful_network_error():
    with patch("src.sentiment_provider.httpx.post", side_effect=httpx.HTTPError("boom")):
        sc = analyze_sentiment("XRP", news_block="x")
    assert sc is None


def test_graceful_bad_json():
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp("not json at all")):
        sc = analyze_sentiment("XRP", news_block="x")
    assert sc is None


def test_graceful_unknown_label():
    content = '{"label":"SuperMegaBull","confidence":0.9,"rationale":"?"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)):
        sc = analyze_sentiment("XRP", news_block="x")
    assert sc is None


def test_extract_json_fenced():
    obj = _extract_json('```json\n{"label":"Bullish","confidence":0.6}\n```')
    assert obj == {"label": "Bullish", "confidence": 0.6}


def test_extract_json_prose_wrapped():
    obj = _extract_json('Sure! Here it is: {"label":"Bearish","confidence":0.3} hope that helps')
    assert obj["label"] == "Bearish"


@pytest.mark.parametrize("label,expected", list(SENTIMENT_SCALE.items()))
def test_full_scale_mapping(label, expected):
    content = f'{{"label":"{label}","confidence":1.0,"rationale":"r"}}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)):
        sc = analyze_sentiment("XRP", news_block="x", use_cache=False)
    assert sc.score == expected


def test_label_normalization_tolerant():
    """대소문자/공백/하이픈 변형 라벨도 정규화된다."""
    content = '{"label":"very bullish","confidence":0.8,"rationale":"r"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)):
        sc = analyze_sentiment("XRP", news_block="x", use_cache=False)
    assert sc.label == "VeryBullish"


def test_confidence_clamped():
    content = '{"label":"Bullish","confidence":9.9,"rationale":"r"}'
    with patch("src.sentiment_provider.httpx.post", return_value=_mock_resp(content)):
        sc = analyze_sentiment("XRP", news_block="x", use_cache=False)
    assert sc.confidence == 1.0


# ── look-ahead 가드 (build_sentiment_dataset.select_window) ──────────────
def test_select_window_excludes_future_articles():
    from scripts.build_sentiment_dataset import select_window

    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    arts = [
        {"pub": base - timedelta(hours=30), "title": "too old"},
        {"pub": base - timedelta(hours=10), "title": "in window"},
        {"pub": base, "title": "exactly at cadence"},        # 포함 (<=)
        {"pub": base + timedelta(minutes=1), "title": "FUTURE"},  # 제외
        {"pub": base + timedelta(hours=5), "title": "FUTURE2"},
    ]
    win = select_window(arts, base, timedelta(hours=24))
    titles = [a["title"] for a in win]
    assert "in window" in titles
    assert "exactly at cadence" in titles  # 상한 포함
    assert "too old" not in titles         # 하한 배타 (24h 초과)
    assert "FUTURE" not in titles          # look-ahead 금지
    assert "FUTURE2" not in titles


def test_select_window_lower_bound_exclusive():
    from scripts.build_sentiment_dataset import select_window

    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    arts = [{"pub": base - timedelta(hours=24), "title": "exactly lookback"}]
    # cadence - lookback < pub  → 정확히 경계는 배타
    assert select_window(arts, base, timedelta(hours=24)) == []
