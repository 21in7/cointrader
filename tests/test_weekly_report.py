import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_fetch_latest_data_calls_subprocess():
    """fetch_latest_data가 심볼별로 fetch_history.py를 호출하는지 확인."""
    from scripts.weekly_report import fetch_latest_data

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        fetch_latest_data(["XRPUSDT", "TRXUSDT"], days=35)

    assert mock_run.call_count == 2
    args_0 = mock_run.call_args_list[0][0][0]
    assert "--symbol" in args_0
    assert "XRPUSDT" in args_0
    assert "--days" in args_0
    assert "35" in args_0


def test_run_backtest_returns_summary():
    """run_backtest가 WF 백테스트를 실행하고 결과를 반환하는지 확인."""
    from scripts.weekly_report import run_backtest

    mock_result = {
        "summary": {
            "total_trades": 27, "total_pnl": 217.0, "return_pct": 21.7,
            "win_rate": 66.7, "profit_factor": 1.57, "max_drawdown_pct": 12.0,
            "sharpe_ratio": 33.3, "avg_win": 20.0, "avg_loss": -10.0,
            "total_fees": 5.0, "close_reasons": {},
        },
        "folds": [], "trades": [],
    }

    with patch("scripts.weekly_report.WalkForwardBacktester") as MockWF:
        MockWF.return_value.run.return_value = mock_result
        result = run_backtest(
            symbols=["XRPUSDT"], train_months=3, test_months=1,
            params={"atr_sl_mult": 2.0, "atr_tp_mult": 2.0,
                    "signal_threshold": 3, "adx_threshold": 25,
                    "volume_multiplier": 2.5},
        )

    assert result["summary"]["profit_factor"] == 1.57
    assert result["summary"]["total_trades"] == 27


def test_parse_live_trades_extracts_entries(tmp_path):
    """봇 로그에서 진입/청산 패턴을 파싱하여 트레이드 리스트를 반환."""
    from scripts.weekly_report import parse_live_trades

    log_content = """2026-03-01 10:00:00.000 | INFO     | src.bot:process_candle:42 - [XRPUSDT] LONG 진입: 가격=2.5000, 수량=100.0, SL=2.4000, TP=2.7000
2026-03-01 10:15:00.000 | INFO     | src.bot:process_candle:42 - [XRPUSDT] 신호: HOLD | 현재가: 2.5500 USDT
2026-03-01 12:00:00.000 | INFO     | src.user_data_stream:_handle_order:80 - [XRPUSDT] 청산 감지(TAKE_PROFIT): exit=2.7000, rp=20.0000, commission=0.2160, net_pnl=19.5680
"""
    log_file = tmp_path / "bot.log"
    log_file.write_text(log_content)

    trades = parse_live_trades(str(log_file), days=7)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "XRPUSDT"
    assert trades[0]["side"] == "LONG"
    assert trades[0]["net_pnl"] == pytest.approx(19.568)
    assert trades[0]["close_reason"] == "TAKE_PROFIT"


def test_parse_live_trades_empty_log(tmp_path):
    """로그 파일이 없으면 빈 리스트 반환."""
    from scripts.weekly_report import parse_live_trades

    trades = parse_live_trades(str(tmp_path / "nonexistent.log"), days=7)
    assert trades == []


import json
from datetime import date, timedelta


def test_load_trend_reads_previous_reports(tmp_path):
    """이전 주간 리포트를 읽어 PF/승률/MDD 추이를 반환."""
    from scripts.weekly_report import load_trend

    for i, (pf, wr, mdd) in enumerate([
        (1.31, 48.0, 9.0), (1.24, 45.0, 11.0),
        (1.20, 44.0, 12.0), (1.18, 43.0, 14.0),
    ]):
        d = date(2026, 3, 7) - timedelta(weeks=3 - i)
        report = {
            "date": d.isoformat(),
            "backtest": {"summary": {
                "profit_factor": pf, "win_rate": wr, "max_drawdown_pct": mdd,
                "total_trades": 20,
            }},
        }
        (tmp_path / f"report_{d.isoformat()}.json").write_text(json.dumps(report))

    trend = load_trend(str(tmp_path), weeks=4)
    assert len(trend["pf"]) == 4
    assert trend["pf"] == [1.31, 1.24, 1.20, 1.18]
    assert trend["pf_declining_3w"] is True


def test_load_trend_empty_dir(tmp_path):
    """리포트가 없으면 빈 추이 반환."""
    from scripts.weekly_report import load_trend

    trend = load_trend(str(tmp_path), weeks=4)
    assert trend["pf"] == []
    assert trend["pf_declining_3w"] is False


def test_check_ml_trigger_all_met():
    """3개 조건 모두 충족 시 recommend=True."""
    from scripts.weekly_report import check_ml_trigger

    result = check_ml_trigger(
        cumulative_trades=200, current_pf=0.85, pf_declining_3w=True,
    )
    assert result["recommend"] is True
    assert result["met_count"] == 3


def test_check_ml_trigger_none_met():
    """조건 미충족 시 recommend=False."""
    from scripts.weekly_report import check_ml_trigger

    result = check_ml_trigger(
        cumulative_trades=50, current_pf=1.5, pf_declining_3w=False,
    )
    assert result["recommend"] is False
    assert result["met_count"] == 0


def test_run_degradation_sweep_returns_top_n():
    """스윕을 실행하고 PF 상위 N개 대안을 반환."""
    from scripts.weekly_report import run_degradation_sweep
    from unittest.mock import patch

    fake_summaries = [
        {"profit_factor": 1.15, "total_trades": 30, "total_pnl": 50, "return_pct": 5,
         "win_rate": 55, "avg_win": 10, "avg_loss": -8, "max_drawdown_pct": 10,
         "sharpe_ratio": 2.0, "total_fees": 1, "close_reasons": {}},
        {"profit_factor": 1.08, "total_trades": 25, "total_pnl": 30, "return_pct": 3,
         "win_rate": 50, "avg_win": 8, "avg_loss": -7, "max_drawdown_pct": 12,
         "sharpe_ratio": 1.5, "total_fees": 1, "close_reasons": {}},
        {"profit_factor": 0.95, "total_trades": 20, "total_pnl": -10, "return_pct": -1,
         "win_rate": 40, "avg_win": 6, "avg_loss": -9, "max_drawdown_pct": 15,
         "sharpe_ratio": 0.5, "total_fees": 1, "close_reasons": {}},
    ]
    fake_combos = [
        {"atr_sl_mult": 1.5}, {"atr_sl_mult": 1.0}, {"atr_sl_mult": 2.0},
    ]

    with patch("scripts.weekly_report.run_single_backtest") as mock_bt:
        mock_bt.side_effect = fake_summaries
        with patch("scripts.weekly_report.generate_combinations", return_value=fake_combos):
            alternatives = run_degradation_sweep(
                symbols=["XRPUSDT"], train_months=3, test_months=1, top_n=3,
            )

    assert len(alternatives) <= 3
    assert alternatives[0]["summary"]["profit_factor"] >= alternatives[1]["summary"]["profit_factor"]
