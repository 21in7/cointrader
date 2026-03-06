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

import json
import re
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


# ── 로그 파싱 패턴 ────────────────────────────────────────────────
_RE_ENTRY = re.compile(
    r"\[(\w+)\]\s+(LONG|SHORT)\s+진입:\s+가격=([\d.]+),\s+수량=([\d.]+),\s+SL=([\d.]+),\s+TP=([\d.]+)"
)
_RE_CLOSE = re.compile(
    r"\[(\w+)\]\s+청산 감지\((\w+)\):\s+exit=([\d.]+),\s+rp=([\d.-]+),\s+commission=([\d.]+),\s+net_pnl=([\d.-]+)"
)
_RE_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})\s")


def parse_live_trades(log_path: str, days: int = 7) -> list[dict]:
    """봇 로그에서 최근 N일간의 진입/청산 기록을 파싱한다."""
    path = Path(log_path)
    if not path.exists():
        return []

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    open_trades: dict[str, dict] = {}
    closed_trades: list[dict] = []

    for line in path.read_text().splitlines():
        m_ts = _RE_TIMESTAMP.match(line)
        if m_ts and m_ts.group(1) < cutoff:
            continue

        m = _RE_ENTRY.search(line)
        if m:
            sym, side, price, qty, sl, tp = m.groups()
            open_trades[sym] = {
                "symbol": sym, "side": side,
                "entry_price": float(price), "quantity": float(qty),
                "sl": float(sl), "tp": float(tp),
                "entry_time": m_ts.group(1) if m_ts else "",
            }
            continue

        m = _RE_CLOSE.search(line)
        if m:
            sym, reason, exit_price, rp, commission, net_pnl = m.groups()
            trade = open_trades.pop(sym, {"symbol": sym, "side": "UNKNOWN"})
            trade.update({
                "close_reason": reason, "exit_price": float(exit_price),
                "expected_pnl": float(rp), "commission": float(commission),
                "net_pnl": float(net_pnl),
                "exit_time": m_ts.group(1) if m_ts else "",
            })
            closed_trades.append(trade)

    return closed_trades


# ── 추이 추적 ────────────────────────────────────────────────────
WEEKLY_DIR = Path("results/weekly")


def load_trend(report_dir: str, weeks: int = 4) -> dict:
    """이전 주간 리포트에서 PF/승률/MDD 추이를 로드한다."""
    rdir = Path(report_dir)
    if not rdir.exists():
        return {"pf": [], "win_rate": [], "mdd": [], "pf_declining_3w": False}

    reports = sorted(rdir.glob("report_*.json"))
    recent = reports[-weeks:] if len(reports) >= weeks else reports

    pf_list, wr_list, mdd_list = [], [], []
    for rpath in recent:
        try:
            data = json.loads(rpath.read_text())
            s = data["backtest"]["summary"]
            pf_list.append(s["profit_factor"])
            wr_list.append(s["win_rate"])
            mdd_list.append(s["max_drawdown_pct"])
        except (json.JSONDecodeError, KeyError):
            continue

    declining = False
    if len(pf_list) >= 3:
        last3 = pf_list[-3:]
        declining = last3[0] > last3[1] > last3[2]

    return {
        "pf": pf_list,
        "win_rate": wr_list,
        "mdd": mdd_list,
        "pf_declining_3w": declining,
    }
