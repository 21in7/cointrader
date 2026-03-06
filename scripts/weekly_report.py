#!/usr/bin/env python3
"""
주간 전략 리포트: 데이터 수집 → WF 백테스트 → 실전 로그 → 추이 → Discord 알림.

사용법:
  python scripts/weekly_report.py
  python scripts/weekly_report.py --skip-fetch
  python scripts/weekly_report.py --date 2026-03-07
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
from datetime import date, timedelta

from loguru import logger

from src.backtester import WalkForwardBacktester, WalkForwardConfig


# ── 프로덕션 파라미터 ──────────────────────────────────────────────
SYMBOLS = ["XRPUSDT", "TRXUSDT", "DOGEUSDT"]
PROD_PARAMS = {
    "atr_sl_mult": 2.0,
    "atr_tp_mult": 2.0,
    "signal_threshold": 3,
    "adx_threshold": 25,
    "volume_multiplier": 2.5,
}
TRAIN_MONTHS = 3
TEST_MONTHS = 1
FETCH_DAYS = 35


def fetch_latest_data(symbols: list[str], days: int = FETCH_DAYS) -> None:
    """심볼별로 fetch_history.py를 subprocess로 호출하여 최신 데이터를 수집한다."""
    script = str(Path(__file__).parent / "fetch_history.py")
    for sym in symbols:
        cmd = [
            sys.executable, script,
            "--symbol", sym,
            "--interval", "15m",
            "--days", str(days),
        ]
        logger.info(f"데이터 수집: {sym} (최근 {days}일)")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  {sym} 수집 실패: {result.stderr[:200]}")
        else:
            logger.info(f"  {sym} 수집 완료")


def run_backtest(
    symbols: list[str],
    train_months: int,
    test_months: int,
    params: dict,
) -> dict:
    """현재 파라미터로 Walk-Forward 백테스트를 실행하고 결과를 반환한다."""
    cfg = WalkForwardConfig(
        symbols=symbols,
        use_ml=False,
        train_months=train_months,
        test_months=test_months,
        **params,
    )
    wf = WalkForwardBacktester(cfg)
    return wf.run()
