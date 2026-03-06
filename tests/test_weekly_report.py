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
