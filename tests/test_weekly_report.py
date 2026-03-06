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
