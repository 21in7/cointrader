"""
evaluate_oos.py 비용 모델 단위 테스트
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_oos import (
    apply_cost_model,
    calc_metrics,
    calc_trade_cost,
    count_funding_events,
)
from src.config import COST_MODEL, COST_SCENARIOS


# ── count_funding_events 테스트 ──────────────────────────────────


def test_count_funding_events_no_crossing():
    """진입 01:00 UTC -> 청산 05:00 UTC, 펀딩 경계(00/08/16) 미포함 -> count == 0."""
    entry = pd.Timestamp("2026-04-10 01:00:00+00:00")
    exit_ = pd.Timestamp("2026-04-10 05:00:00+00:00")
    assert count_funding_events(entry, exit_) == 0


def test_count_funding_events_single_crossing():
    """진입 06:00 UTC -> 청산 10:00 UTC, 08:00 포함 -> count == 1."""
    entry = pd.Timestamp("2026-04-10 06:00:00+00:00")
    exit_ = pd.Timestamp("2026-04-10 10:00:00+00:00")
    assert count_funding_events(entry, exit_) == 1


def test_count_funding_events_multiple_crossings():
    """12시간 보유: 02:00 -> 14:00, 08:00 포함 -> count == 1."""
    entry = pd.Timestamp("2026-04-10 02:00:00+00:00")
    exit_ = pd.Timestamp("2026-04-10 14:00:00+00:00")
    assert count_funding_events(entry, exit_) == 1

    # 22:00 -> 10:00 (다음날), 00:00 + 08:00 포함 -> count == 2
    entry2 = pd.Timestamp("2026-04-10 22:00:00+00:00")
    exit2 = pd.Timestamp("2026-04-11 10:00:00+00:00")
    assert count_funding_events(entry2, exit2) == 2


def test_count_funding_events_short_trade_no_overcounting():
    """75분 거래, 경계 미포함 -> count == 0."""
    # 18:15 -> 19:30, 펀딩 경계 없음
    entry = pd.Timestamp("2026-04-10 18:15:00+00:00")
    exit_ = pd.Timestamp("2026-04-10 19:30:00+00:00")
    assert count_funding_events(entry, exit_) == 0


def test_count_funding_events_exact_boundary():
    """정확히 경계에서 진입/청산하는 경우."""
    # entry=08:00, exit=16:00 -> ceil(08:00)=08:00, floor(16:00)=16:00
    # hours: 08, 09, ..., 16 -> 08:00(yes), 16:00(yes) -> count == 2
    entry = pd.Timestamp("2026-04-10 08:00:00+00:00")
    exit_ = pd.Timestamp("2026-04-10 16:00:00+00:00")
    assert count_funding_events(entry, exit_) == 2


# ── 비용 계산 테스트 ─────────────────────────────────────────────


def test_cost_calculation_taker_roundtrip():
    """진입 taker + SL taker, slippage 0, funding 0 -> 8 bps."""
    row = pd.Series({
        "entry_ts": pd.Timestamp("2026-04-10 01:00:00+00:00"),
        "exit_ts": pd.Timestamp("2026-04-10 02:00:00+00:00"),
        "pnl_bps": -50.0,
        "reason": "SL 히트 (1.3012)",
        "side": "SHORT",
    })
    scenario = COST_SCENARIOS["fees_only"]
    cost = calc_trade_cost(row, scenario)
    assert cost == 8.0  # taker(4) + taker(4) + 0 + 0


def test_cost_calculation_tp_exit():
    """TP 히트 시에도 현재 설정에서는 taker -> 8 bps."""
    row = pd.Series({
        "entry_ts": pd.Timestamp("2026-04-10 01:00:00+00:00"),
        "exit_ts": pd.Timestamp("2026-04-10 02:00:00+00:00"),
        "pnl_bps": 80.0,
        "reason": "TP 히트 (1.3826)",
        "side": "LONG",
    })
    scenario = COST_SCENARIOS["fees_only"]
    cost = calc_trade_cost(row, scenario)
    assert cost == 8.0


def test_cost_with_slippage_and_funding():
    """realistic 시나리오: fee 8 + slippage 2 + funding 1 = 11 bps."""
    # 진입 15:45, 청산 17:00 -> funding event at 16:00 -> count=1
    row = pd.Series({
        "entry_ts": pd.Timestamp("2026-04-02 15:45:00+00:00"),
        "exit_ts": pd.Timestamp("2026-04-02 17:00:00+00:00"),
        "pnl_bps": -68.0,
        "reason": "SL 히트 (1.3012)",
        "side": "SHORT",
    })
    scenario = COST_SCENARIOS["realistic"]
    cost = calc_trade_cost(row, scenario)
    # fee=8, slippage=1*2=2, funding=1*1=1 -> total=11
    assert cost == 11.0


def test_adjusted_pnl_matches_manual():
    """첫 번째 거래(Trade #0)에 대해 수작업 계산값과 일치 확인."""
    # Trade #0: SHORT, entry 15:45 UTC, exit 17:00 UTC, pnl_bps=-68.0, SL 히트
    # fees_only: cost=8 (fee only, funding event at 16:00 but funding_bps=0) -> adjusted=-76.0
    # realistic: cost=8+2+1=11 -> adjusted=-79.0
    # pessimistic: cost=8+6+2=16 -> adjusted=-84.0
    row = pd.Series({
        "entry_ts": pd.Timestamp("2026-04-02 15:45:02.285284+00:00"),
        "exit_ts": pd.Timestamp("2026-04-02 17:00:00.791551+00:00"),
        "pnl_bps": -68.0,
        "reason": "SL 히트 (1.3012)",
        "side": "SHORT",
    })

    for scenario_name, expected_adj in [
        ("fees_only", -76.0),
        ("realistic", -79.0),
        ("pessimistic", -84.0),
    ]:
        scenario = COST_SCENARIOS[scenario_name]
        cost = calc_trade_cost(row, scenario)
        adjusted = row["pnl_bps"] - cost
        assert adjusted == expected_adj, f"{scenario_name}: {adjusted} != {expected_adj}"


# ── 회귀 테스트 ──────────────────────────────────────────────────


def test_regression_fees_only_cum_pnl():
    """18건 전체를 fees_only로 돌렸을 때 CumPnL == -173.9 bps (+-0.5 bps 허용)."""
    jsonl_path = Path("data/trade_history/mtf_xrpusdtusdt.jsonl")
    if not jsonl_path.exists():
        pytest.skip("로컬 jsonl 파일 없음")

    df = pd.read_json(jsonl_path, lines=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["duration_min"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60

    result = apply_cost_model(df, "fees_only")
    metrics = calc_metrics(result, pnl_col="adjusted_pnl_bps")

    assert metrics["trades"] == 18
    assert abs(metrics["cum_pnl"] - (-173.9)) <= 0.5, f"CumPnL={metrics['cum_pnl']}, expected -173.9"


# ── calc_metrics 테스트 ──────────────────────────────────────────


def test_calc_metrics_empty():
    """빈 DataFrame -> 안전한 기본값."""
    df = pd.DataFrame(columns=["pnl_bps", "duration_min"])
    m = calc_metrics(df)
    assert m["trades"] == 0
    assert m["pf"] == 0.0


def test_calc_metrics_with_avg_pnl():
    """avg_pnl 필드가 정확히 계산되는지 확인."""
    df = pd.DataFrame({
        "pnl_bps": [10.0, -5.0, 20.0],
        "duration_min": [60.0, 30.0, 90.0],
    })
    m = calc_metrics(df)
    assert m["trades"] == 3
    assert m["avg_pnl"] == pytest.approx(25.0 / 3, abs=0.01)
