"""fetch_history.py의 upsert_parquet() 함수 테스트."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def _make_parquet(tmp_path: Path, rows: dict) -> Path:
    """테스트용 parquet 파일 생성 헬퍼."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path


def test_upsert_fills_zero_oi_with_real_value(tmp_path):
    """기존 행의 oi_change=0.0이 신규 데이터의 실제 값으로 덮어써진다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00", "2026-01-01 00:15"],
        "close": [1.0, 1.1],
        "oi_change": [0.0, 0.0],
        "funding_rate": [0.0, 0.0],
    })

    new_df = pd.DataFrame({
        "close": [1.0, 1.1],
        "oi_change": [0.05, 0.03],
        "funding_rate": [0.0001, 0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert result.loc["2026-01-01 00:00+00:00", "oi_change"] == pytest.approx(0.05)
    assert result.loc["2026-01-01 00:15+00:00", "oi_change"] == pytest.approx(0.03)


def test_upsert_appends_new_rows(tmp_path):
    """신규 타임스탬프 행이 기존 데이터 아래에 추가된다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00"],
        "close": [1.0],
        "oi_change": [0.05],
        "funding_rate": [0.0001],
    })

    new_df = pd.DataFrame({
        "close": [1.1],
        "oi_change": [0.03],
        "funding_rate": [0.0002],
    }, index=pd.to_datetime(["2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert len(result) == 2
    assert pd.Timestamp("2026-01-01 00:15", tz="UTC") in result.index


def test_upsert_keeps_nonzero_existing_oi(tmp_path):
    """기존 행의 oi_change가 이미 0이 아니면 덮어쓰지 않는다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00"],
        "close": [1.0],
        "oi_change": [0.07],   # 이미 실제 값 존재
        "funding_rate": [0.0003],
    })

    new_df = pd.DataFrame({
        "close": [1.0],
        "oi_change": [0.05],   # 다른 값으로 덮어쓰려 해도
        "funding_rate": [0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    # 기존 값(0.07)이 유지되어야 한다
    assert result.iloc[0]["oi_change"] == pytest.approx(0.07)


def test_upsert_no_existing_file_returns_new_df(tmp_path):
    """기존 parquet 파일이 없으면 신규 데이터를 그대로 반환한다."""
    from scripts.fetch_history import upsert_parquet

    nonexistent_path = tmp_path / "nonexistent.parquet"
    new_df = pd.DataFrame({
        "close": [1.0, 1.1],
        "oi_change": [0.05, 0.03],
        "funding_rate": [0.0001, 0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(nonexistent_path, new_df)

    assert len(result) == 2
    assert result.iloc[0]["oi_change"] == pytest.approx(0.05)


def test_upsert_result_is_sorted_by_timestamp(tmp_path):
    """결과 DataFrame이 timestamp 기준 오름차순 정렬되어 있다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:15"],
        "close": [1.1],
        "oi_change": [0.0],
        "funding_rate": [0.0],
    })

    new_df = pd.DataFrame({
        "close": [1.0, 1.1, 1.2],
        "oi_change": [0.05, 0.03, 0.02],
        "funding_rate": [0.0001, 0.0001, 0.0002],
    }, index=pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"], utc=True
    ))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert result.index.is_monotonic_increasing
    assert len(result) == 3
