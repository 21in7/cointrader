"""backtester 센티먼트 게이트 순수 로직 + 스코어 lookup 테스트 (게이트 B).

게이트 결정은 진입 차단 여부를 좌우하는 안전 핵심 로직이므로 모드별
경계 동작을 명시적으로 검증한다.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtester import _sentiment_gate, _get_sentiment_score


# ── off / 결측: 항상 통과 (베이스라인 동일) ─────────────────────────────
@pytest.mark.parametrize("mode", ["off", "veto", "contrarian", "confirm"])
def test_none_score_always_allows(mode):
    assert _sentiment_gate(mode, "LONG", None, 0.5, 1.0) is True
    assert _sentiment_gate(mode, "SHORT", None, 0.5, 1.0) is True


def test_off_mode_always_allows():
    assert _sentiment_gate("off", "LONG", -1.0, 0.5, 1.0) is True
    assert _sentiment_gate("off", "SHORT", 1.0, 0.5, 1.0) is True


# ── veto: 신호와 군중이 정면 충돌하면 차단 ──────────────────────────────
def test_veto_blocks_conflict():
    assert _sentiment_gate("veto", "LONG", -0.6, 0.5, 1.0) is False  # 롱인데 약세
    assert _sentiment_gate("veto", "SHORT", 0.6, 0.5, 1.0) is False  # 숏인데 강세


def test_veto_allows_non_conflict():
    assert _sentiment_gate("veto", "LONG", -0.4, 0.5, 1.0) is True   # 임계 미만
    assert _sentiment_gate("veto", "LONG", 0.9, 0.5, 1.0) is True    # 동의 → 허용
    assert _sentiment_gate("veto", "SHORT", -0.9, 0.5, 1.0) is True


def test_veto_boundary_exclusive():
    # score == -threshold 는 <= 이므로 차단
    assert _sentiment_gate("veto", "LONG", -0.5, 0.5, 1.0) is False
    assert _sentiment_gate("veto", "LONG", -0.49, 0.5, 1.0) is True


# ── contrarian: 군중 극단에 *순응하는* 신호만 차단 (페이드) ──────────────
def test_contrarian_blocks_agreeing_extreme():
    assert _sentiment_gate("contrarian", "LONG", 1.0, 0.5, 1.0) is False   # 극단강세에 롱 금지
    assert _sentiment_gate("contrarian", "SHORT", -1.0, 0.5, 1.0) is False  # 극단약세에 숏 금지


def test_contrarian_allows_fade_and_non_extreme():
    assert _sentiment_gate("contrarian", "SHORT", 1.0, 0.5, 1.0) is True   # 극단강세 페이드 숏 허용
    assert _sentiment_gate("contrarian", "LONG", -1.0, 0.5, 1.0) is True   # 극단약세 페이드 롱 허용
    assert _sentiment_gate("contrarian", "LONG", 0.9, 0.5, 1.0) is True    # 비극단 통과
    assert _sentiment_gate("contrarian", "SHORT", -0.9, 0.5, 1.0) is True


# ── confirm: 군중이 적극 동의할 때만 허용 (엄격, 중립도 차단) ────────────
def test_confirm_requires_active_agreement():
    assert _sentiment_gate("confirm", "LONG", 0.6, 0.5, 1.0) is True
    assert _sentiment_gate("confirm", "LONG", 0.4, 0.5, 1.0) is False  # 미달
    assert _sentiment_gate("confirm", "LONG", 0.0, 0.5, 1.0) is False  # 중립도 차단
    assert _sentiment_gate("confirm", "SHORT", -0.6, 0.5, 1.0) is True
    assert _sentiment_gate("confirm", "SHORT", -0.4, 0.5, 1.0) is False


def test_unknown_mode_safe_passthrough():
    assert _sentiment_gate("garbage", "LONG", -1.0, 0.5, 1.0) is True


# ── _get_sentiment_score: lookup / NaN / 결측 / tz ──────────────────────
def _sent_df():
    idx = pd.date_range("2026-05-01", periods=4, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"sentiment_score": [0.5, np.nan, -1.0, 0.0]}, index=idx
    )


def test_get_score_hit():
    df = _sent_df()
    ts = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
    assert _get_sentiment_score(df, ts) == 0.5


def test_get_score_nan_returns_none():
    df = _sent_df()
    ts = pd.Timestamp("2026-05-01 00:15:00", tz="UTC")
    assert _get_sentiment_score(df, ts) is None


def test_get_score_missing_key_returns_none():
    df = _sent_df()
    ts = pd.Timestamp("2030-01-01 00:00:00", tz="UTC")
    assert _get_sentiment_score(df, ts) is None


def test_get_score_none_df_returns_none():
    ts = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
    assert _get_sentiment_score(None, ts) is None


def test_get_score_naive_ts_localized():
    df = _sent_df()
    naive = pd.Timestamp("2026-05-01 00:30:00")  # tz 없음 → UTC로 가정
    assert _get_sentiment_score(df, naive) == -1.0
