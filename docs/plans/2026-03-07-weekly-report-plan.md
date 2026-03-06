# 주간 전략 리포트 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 매주 전략 성능을 자동 측정하고, 추이를 추적하며, 성능 저하를 감지하고, Discord 리포트를 전송한다.

**Architecture:** 단일 스크립트 `scripts/weekly_report.py`가 데이터 수집(subprocess), Walk-Forward 백테스트(import), 로그 파싱(`dashboard/api/log_parser.py` 재사용), 추이 분석(기존 `results/weekly/*.json` 읽기), 선택적 파라미터 스윕(import), Discord 알림(`src/notifier.py` import)을 오케스트레이션한다. 프로덕션 봇 코드 변경 없음.

**Tech Stack:** Python 3.12, 기존 backtester/sweep/notifier/log_parser 모듈, `fetch_history.py` subprocess 호출, Discord용 httpx.

---

### Task 1: 주간 리포트 코어 — 데이터 수집 + 백테스트

**Files:**
- Create: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: `fetch_latest_data()` 실패 테스트 작성**

```python
# tests/test_weekly_report.py
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
    # 첫 번째 호출이 XRPUSDT인지 확인
    args_0 = mock_run.call_args_list[0][0][0]
    assert "--symbol" in args_0
    assert "XRPUSDT" in args_0
    assert "--days" in args_0
    assert "35" in args_0
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py::test_fetch_latest_data_calls_subprocess -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.weekly_report'`

**Step 3: `run_backtest()` 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
def test_run_backtest_returns_summary():
    """run_backtest가 심볼별 WF 백테스트를 실행하고 결과를 반환하는지 확인."""
    from scripts.weekly_report import run_backtest

    mock_result = {
        "summary": {
            "total_trades": 27,
            "total_pnl": 217.0,
            "return_pct": 21.7,
            "win_rate": 66.7,
            "profit_factor": 1.57,
            "max_drawdown_pct": 12.0,
            "sharpe_ratio": 33.3,
            "avg_win": 20.0,
            "avg_loss": -10.0,
            "total_fees": 5.0,
            "close_reasons": {},
        },
        "folds": [],
        "trades": [],
    }

    with patch("scripts.weekly_report.WalkForwardBacktester") as MockWF:
        MockWF.return_value.run.return_value = mock_result
        result = run_backtest(
            symbols=["XRPUSDT"],
            train_months=3,
            test_months=1,
            params={"atr_sl_mult": 2.0, "atr_tp_mult": 2.0,
                    "signal_threshold": 3, "adx_threshold": 25,
                    "volume_multiplier": 2.5},
        )

    assert result["summary"]["profit_factor"] == 1.57
    assert result["summary"]["total_trades"] == 27
```

**Step 4: 최소 구현 작성**

```python
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

import argparse
import json
import subprocess
from datetime import datetime, date, timedelta

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
FETCH_DAYS = 35  # 최근 35일 upsert


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
```

**Step 5: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 2 PASS

**Step 6: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): 데이터 수집 및 WF 백테스트 코어 추가"
```

---

### Task 2: 실전 트레이드 로그 파싱 추가

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
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
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py::test_parse_live_trades_extracts_entries -v`
Expected: FAIL — `ImportError: cannot import name 'parse_live_trades'`

**Step 3: 구현 작성**

`scripts/weekly_report.py`에 추가:

```python
import re

# ── 로그 파싱 패턴 (dashboard/api/log_parser.py와 동일) ──────────
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
    open_trades: dict[str, dict] = {}  # symbol -> pending trade
    closed_trades: list[dict] = []

    for line in path.read_text().splitlines():
        # 날짜 필터
        m_ts = _RE_TIMESTAMP.match(line)
        if m_ts and m_ts.group(1) < cutoff:
            continue

        # 진입
        m = _RE_ENTRY.search(line)
        if m:
            sym, side, price, qty, sl, tp = m.groups()
            open_trades[sym] = {
                "symbol": sym,
                "side": side,
                "entry_price": float(price),
                "quantity": float(qty),
                "sl": float(sl),
                "tp": float(tp),
                "entry_time": m_ts.group(1) if m_ts else "",
            }
            continue

        # 청산
        m = _RE_CLOSE.search(line)
        if m:
            sym, reason, exit_price, rp, commission, net_pnl = m.groups()
            trade = open_trades.pop(sym, {"symbol": sym, "side": "UNKNOWN"})
            trade.update({
                "close_reason": reason,
                "exit_price": float(exit_price),
                "expected_pnl": float(rp),
                "commission": float(commission),
                "net_pnl": float(net_pnl),
                "exit_time": m_ts.group(1) if m_ts else "",
            })
            closed_trades.append(trade)

    return closed_trades
```

**Step 4: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 4 PASS

**Step 5: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): 실전 트레이드 로그 파서 추가"
```

---

### Task 3: 추이 추적 (이전 리포트 읽기) 추가

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
def test_load_trend_reads_previous_reports(tmp_path):
    """이전 주간 리포트를 읽어 PF/승률/MDD 추이를 반환."""
    from scripts.weekly_report import load_trend

    # 4주치 리포트 생성
    for i, (pf, wr, mdd) in enumerate([
        (1.31, 48.0, 9.0),
        (1.24, 45.0, 11.0),
        (1.20, 44.0, 12.0),
        (1.18, 43.0, 14.0),
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
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py::test_load_trend_reads_previous_reports -v`
Expected: FAIL

**Step 3: 구현 작성**

`scripts/weekly_report.py`에 추가:

```python
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

    # PF 3주 연속 하락 체크
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
```

**Step 4: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 6 PASS

**Step 5: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): 이전 리포트 추이 추적 추가"
```

---

### Task 4: ML 재트리거 체크 + 성능 저하 스윕 추가

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
def test_check_ml_trigger_all_met():
    """3개 조건 모두 충족 시 recommend=True."""
    from scripts.weekly_report import check_ml_trigger

    result = check_ml_trigger(
        cumulative_trades=200,
        current_pf=0.85,
        pf_declining_3w=True,
    )
    assert result["recommend"] is True
    assert result["met_count"] == 3


def test_check_ml_trigger_none_met():
    """조건 미충족 시 recommend=False."""
    from scripts.weekly_report import check_ml_trigger

    result = check_ml_trigger(
        cumulative_trades=50,
        current_pf=1.5,
        pf_declining_3w=False,
    )
    assert result["recommend"] is False
    assert result["met_count"] == 0


def test_run_degradation_sweep_called_when_pf_low():
    """PF < 1.0이면 스윕을 실행하고 상위 3개 대안을 반환."""
    from scripts.weekly_report import run_degradation_sweep

    fake_results = [
        {"params": {"atr_sl_mult": 1.5}, "summary": {"profit_factor": 1.15, "total_trades": 30}},
        {"params": {"atr_sl_mult": 1.0}, "summary": {"profit_factor": 1.08, "total_trades": 25}},
        {"params": {"atr_sl_mult": 2.0}, "summary": {"profit_factor": 0.95, "total_trades": 20}},
    ]

    with patch("scripts.weekly_report.run_single_backtest") as mock_bt:
        mock_bt.side_effect = [r["summary"] for r in fake_results]
        with patch("scripts.weekly_report.generate_combinations", return_value=[
            r["params"] for r in fake_results
        ]):
            alternatives = run_degradation_sweep(
                symbols=["XRPUSDT"],
                train_months=3,
                test_months=1,
                top_n=3,
            )

    assert len(alternatives) <= 3
    # PF 내림차순 정렬
    assert alternatives[0]["summary"]["profit_factor"] >= alternatives[1]["summary"]["profit_factor"]
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -k "ml_trigger or degradation" -v`
Expected: FAIL

**Step 3: 구현 작성**

`scripts/weekly_report.py`에 추가:

```python
from scripts.strategy_sweep import (
    run_single_backtest,
    generate_combinations,
    PARAM_GRID,
)

ML_TRADE_THRESHOLD = 150


def check_ml_trigger(
    cumulative_trades: int,
    current_pf: float,
    pf_declining_3w: bool,
) -> dict:
    """ML 재학습 조건 체크. 3개 중 2개 이상 충족 시 권장."""
    conditions = {
        "cumulative_trades_enough": cumulative_trades >= ML_TRADE_THRESHOLD,
        "pf_below_1": current_pf < 1.0,
        "pf_declining_3w": pf_declining_3w,
    }
    met = sum(conditions.values())
    return {
        "conditions": conditions,
        "met_count": met,
        "recommend": met >= 2,
        "cumulative_trades": cumulative_trades,
        "threshold": ML_TRADE_THRESHOLD,
    }


def run_degradation_sweep(
    symbols: list[str],
    train_months: int,
    test_months: int,
    top_n: int = 3,
) -> list[dict]:
    """전체 파라미터 스윕을 실행하고 PF 상위 N개 대안을 반환한다."""
    combos = generate_combinations(PARAM_GRID)
    results = []

    for params in combos:
        try:
            summary = run_single_backtest(symbols, params, train_months, test_months)
            results.append({"params": params, "summary": summary})
        except Exception as e:
            logger.warning(f"스윕 실패: {e}")

    results.sort(
        key=lambda r: r["summary"]["profit_factor"]
        if r["summary"]["profit_factor"] != float("inf") else 999,
        reverse=True,
    )
    return results[:top_n]
```

**Step 4: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 9 PASS

**Step 5: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): ML 트리거 체크 및 성능 저하 스윕 추가"
```

---

### Task 5: Discord 리포트 포맷팅 + 전송 추가

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
def test_format_report_normal():
    """정상 상태(PF >= 1.0) 리포트 포맷."""
    from scripts.weekly_report import format_report

    report_data = {
        "date": "2026-03-07",
        "backtest": {
            "summary": {
                "profit_factor": 1.24, "win_rate": 45.0,
                "max_drawdown_pct": 12.0, "total_trades": 88,
            },
            "per_symbol": {
                "XRPUSDT": {"profit_factor": 1.57, "total_trades": 27, "win_rate": 66.7},
                "TRXUSDT": {"profit_factor": 1.29, "total_trades": 25, "win_rate": 52.0},
                "DOGEUSDT": {"profit_factor": 1.09, "total_trades": 36, "win_rate": 44.4},
            },
        },
        "live_trades": {"count": 8, "net_pnl": 12.34, "win_rate": 62.5},
        "trend": {"pf": [1.31, 1.24], "win_rate": [48.0, 45.0], "mdd": [9.0, 12.0], "pf_declining_3w": False},
        "ml_trigger": {"recommend": False, "met_count": 0, "conditions": {
            "cumulative_trades_enough": False, "pf_below_1": False, "pf_declining_3w": False,
        }, "cumulative_trades": 47, "threshold": 150},
        "sweep": None,
    }

    text = format_report(report_data)
    assert "주간 전략 리포트" in text
    assert "1.24" in text
    assert "XRPUSDT" in text
    assert "스윕 불필요" in text or "파라미터 스윕" not in text


def test_format_report_degraded():
    """PF < 1.0일 때 스윕 결과가 포함되는지 확인."""
    from scripts.weekly_report import format_report

    report_data = {
        "date": "2026-06-07",
        "backtest": {
            "summary": {
                "profit_factor": 0.87, "win_rate": 38.0,
                "max_drawdown_pct": 22.0, "total_trades": 90,
            },
            "per_symbol": {},
        },
        "live_trades": {"count": 0, "net_pnl": 0, "win_rate": 0},
        "trend": {"pf": [1.1, 1.0, 0.87], "win_rate": [], "mdd": [], "pf_declining_3w": True},
        "ml_trigger": {"recommend": True, "met_count": 3, "conditions": {
            "cumulative_trades_enough": True, "pf_below_1": True, "pf_declining_3w": True,
        }, "cumulative_trades": 182, "threshold": 150},
        "sweep": [
            {"params": {"atr_sl_mult": 2.0, "atr_tp_mult": 2.5, "adx_threshold": 30, "volume_multiplier": 2.5, "signal_threshold": 3},
             "summary": {"profit_factor": 1.15, "total_trades": 30}},
        ],
    }

    text = format_report(report_data)
    assert "0.87" in text
    assert "ML" in text
    assert "1.15" in text  # 스윕 대안


def test_send_report_uses_notifier():
    """Discord 웹훅으로 리포트를 전송."""
    from scripts.weekly_report import send_report

    with patch("scripts.weekly_report.DiscordNotifier") as MockNotifier:
        instance = MockNotifier.return_value
        send_report("test report content", webhook_url="https://example.com/webhook")
        instance._send.assert_called_once_with("test report content")
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -k "format_report or send_report" -v`
Expected: FAIL

**Step 3: 구현 작성**

`scripts/weekly_report.py`에 추가:

```python
import os
from src.notifier import DiscordNotifier


def format_report(data: dict) -> str:
    """리포트 데이터를 Discord 메시지 텍스트로 포맷한다."""
    d = data["date"]
    bt = data["backtest"]["summary"]
    pf = bt["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"

    status = ""
    if pf < 1.0:
        status = "  🚨 손실 구간"

    lines = [
        f"📊 주간 전략 리포트 ({d})",
        "",
        f"[현재 성능 — Walk-Forward 백테스트]",
        f"  합산 PF: {pf_str} | 승률: {bt['win_rate']:.0f}% | MDD: {bt['max_drawdown_pct']:.0f}%{status}",
    ]

    # 심볼별 성능
    per_sym = data["backtest"].get("per_symbol", {})
    if per_sym:
        sym_parts = []
        for sym, s in per_sym.items():
            short = sym.replace("USDT", "")
            spf = f"{s['profit_factor']:.2f}" if s["profit_factor"] != float("inf") else "INF"
            sym_parts.append(f"{short}: PF {spf} ({s['total_trades']}건)")
        lines.append(f"  {' | '.join(sym_parts)}")

    # 실전 트레이드
    lt = data["live_trades"]
    if lt["count"] > 0:
        lines += [
            "",
            f"[실전 트레이드 (이번 주)]",
            f"  거래: {lt['count']}건 | 순수익: {lt['net_pnl']:+.2f} USDT | 승률: {lt['win_rate']:.1f}%",
        ]

    # 추이
    trend = data["trend"]
    if trend["pf"]:
        pf_trend = " → ".join(f"{v:.2f}" for v in trend["pf"])
        warn = "  ⚠ 하락 추세" if trend["pf_declining_3w"] else ""
        lines += ["", f"[추이 (최근 {len(trend['pf'])}주)]", f"  PF: {pf_trend}{warn}"]
        if trend["win_rate"]:
            wr_trend = " → ".join(f"{v:.0f}%" for v in trend["win_rate"])
            lines.append(f"  승률: {wr_trend}")
        if trend["mdd"]:
            mdd_trend = " → ".join(f"{v:.0f}%" for v in trend["mdd"])
            lines.append(f"  MDD: {mdd_trend}")

    # ML 재도전 체크리스트
    ml = data["ml_trigger"]
    cond = ml["conditions"]
    lines += [
        "",
        f"[ML 재도전 체크리스트]",
        f"  {'✅' if cond['cumulative_trades_enough'] else '☐'} 누적 트레이드 ≥ {ml['threshold']}건: {ml['cumulative_trades']}/{ml['threshold']}",
        f"  {'✅' if cond['pf_below_1'] else '☐'} PF < 1.0: {'예' if cond['pf_below_1'] else '아니오'} (현재 {pf_str})",
        f"  {'✅' if cond['pf_declining_3w'] else '☐'} PF 3주 연속 하락: {'예 ⚠' if cond['pf_declining_3w'] else '아니오'}",
    ]
    if ml["recommend"]:
        lines.append(f"  → 🔔 ML 재학습 권장! ({ml['met_count']}/3 충족)")
    else:
        lines.append(f"  → ML 재도전 시점: 아직 아님 ({ml['met_count']}/3 충족)")

    # 파라미터 스윕
    sweep = data.get("sweep")
    if sweep:
        lines += ["", "[파라미터 스윕 결과]"]
        current_pf_str = pf_str
        lines.append(f"  현재: {_param_str(PROD_PARAMS)} → PF {current_pf_str}")
        for i, alt in enumerate(sweep):
            apf = alt["summary"]["profit_factor"]
            apf_str = f"{apf:.2f}" if apf != float("inf") else "INF"
            diff = apf - pf
            lines.append(f"  대안 {i+1}: {_param_str(alt['params'])} → PF {apf_str} ({diff:+.2f})")
        lines.append("")
        lines.append("  ⚠ 자동 적용되지 않음. 검토 후 승인 필요.")
    elif pf >= 1.0:
        lines += ["", "[파라미터 스윕]", "  현재 파라미터가 최적 — 스윕 불필요"]

    return "\n".join(lines)


def _param_str(p: dict) -> str:
    return (f"SL={p.get('atr_sl_mult', '?')}, TP={p.get('atr_tp_mult', '?')}, "
            f"ADX={p.get('adx_threshold', '?')}, Vol={p.get('volume_multiplier', '?')}")


def send_report(content: str, webhook_url: str | None = None) -> None:
    """Discord 웹훅으로 리포트를 전송한다."""
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL이 설정되지 않아 전송 스킵")
        return
    notifier = DiscordNotifier(url)
    notifier._send(content)
    logger.info("Discord 리포트 전송 완료")
```

**Step 4: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 12 PASS

**Step 5: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): Discord 리포트 포맷팅 및 전송 추가"
```

---

### Task 6: 메인 오케스트레이션 + CLI + JSON 저장 추가

**Files:**
- Modify: `scripts/weekly_report.py`
- Test: `tests/test_weekly_report.py`

**Step 1: 실패 테스트 작성**

```python
# tests/test_weekly_report.py (추가)
def test_generate_report_orchestration(tmp_path):
    """generate_report가 모든 단계를 조합하여 리포트 dict를 반환."""
    from scripts.weekly_report import generate_report

    mock_bt_result = {
        "summary": {
            "profit_factor": 1.24, "win_rate": 45.0,
            "max_drawdown_pct": 12.0, "total_trades": 88,
            "total_pnl": 379.0, "return_pct": 37.9,
            "avg_win": 20.0, "avg_loss": -10.0,
            "sharpe_ratio": 33.0, "total_fees": 5.0,
            "close_reasons": {},
        },
        "folds": [],
        "trades": [],
    }

    with patch("scripts.weekly_report.run_backtest", return_value=mock_bt_result):
        with patch("scripts.weekly_report.parse_live_trades", return_value=[]):
            with patch("scripts.weekly_report.load_trend", return_value={
                "pf": [1.31], "win_rate": [48.0], "mdd": [9.0], "pf_declining_3w": False,
            }):
                report = generate_report(
                    symbols=["XRPUSDT"],
                    report_dir=str(tmp_path),
                    log_path=str(tmp_path / "bot.log"),
                    report_date=date(2026, 3, 7),
                )

    assert report["date"] == "2026-03-07"
    assert report["backtest"]["summary"]["profit_factor"] == 1.24
    assert report["sweep"] is None  # PF >= 1.0이면 스윕 안 함


def test_save_report_creates_json(tmp_path):
    """리포트를 JSON으로 저장."""
    from scripts.weekly_report import save_report

    report = {"date": "2026-03-07", "test": True}
    save_report(report, str(tmp_path))

    saved = list(tmp_path.glob("report_*.json"))
    assert len(saved) == 1
    loaded = json.loads(saved[0].read_text())
    assert loaded["date"] == "2026-03-07"
```

**Step 2: 테스트 실행하여 실패 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -k "generate_report or save_report" -v`
Expected: FAIL

**Step 3: 구현 작성**

`scripts/weekly_report.py`에 추가:

```python
import numpy as np


def _sanitize(obj):
    """JSON 직렬화를 위해 numpy/inf 값을 변환."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def save_report(report: dict, report_dir: str) -> Path:
    """리포트를 JSON으로 저장하고 경로를 반환한다."""
    rdir = Path(report_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"report_{report['date']}.json"
    with open(path, "w") as f:
        json.dump(_sanitize(report), f, indent=2, ensure_ascii=False)
    logger.info(f"리포트 저장: {path}")
    return path


def generate_report(
    symbols: list[str],
    report_dir: str = str(WEEKLY_DIR),
    log_path: str = "logs/bot.log",
    report_date: date | None = None,
) -> dict:
    """전체 주간 리포트를 생성한다."""
    today = report_date or date.today()

    # 1) Walk-Forward 백테스트
    logger.info("백테스트 실행 중...")
    bt_results = {}
    combined_trades = 0
    combined_pnl = 0.0
    combined_gp = 0.0
    combined_gl = 0.0

    for sym in symbols:
        result = run_backtest([sym], TRAIN_MONTHS, TEST_MONTHS, PROD_PARAMS)
        bt_results[sym] = result["summary"]
        s = result["summary"]
        n = s["total_trades"]
        combined_trades += n
        combined_pnl += s["total_pnl"]
        if n > 0:
            wr = s["win_rate"] / 100.0
            n_wins = round(wr * n)
            n_losses = n - n_wins
            combined_gp += s["avg_win"] * n_wins if n_wins > 0 else 0
            combined_gl += abs(s["avg_loss"]) * n_losses if n_losses > 0 else 0

    combined_pf = combined_gp / combined_gl if combined_gl > 0 else float("inf")
    combined_wr = (
        sum(s["win_rate"] * s["total_trades"] for s in bt_results.values())
        / combined_trades if combined_trades > 0 else 0
    )
    combined_mdd = max((s["max_drawdown_pct"] for s in bt_results.values()), default=0)

    backtest_summary = {
        "profit_factor": round(combined_pf, 2),
        "win_rate": round(combined_wr, 1),
        "max_drawdown_pct": round(combined_mdd, 1),
        "total_trades": combined_trades,
        "total_pnl": round(combined_pnl, 2),
    }

    # 2) 실전 트레이드 파싱
    logger.info("실전 로그 파싱 중...")
    live_trades = parse_live_trades(log_path, days=7)
    live_wins = sum(1 for t in live_trades if t.get("net_pnl", 0) > 0)
    live_pnl = sum(t.get("net_pnl", 0) for t in live_trades)
    live_summary = {
        "count": len(live_trades),
        "net_pnl": round(live_pnl, 2),
        "win_rate": round(live_wins / len(live_trades) * 100, 1) if live_trades else 0,
    }

    # 3) 추이 로드
    trend = load_trend(report_dir)

    # 4) 누적 트레이드 수 계산
    cumulative = combined_trades
    for rpath in sorted(Path(report_dir).glob("report_*.json")) if Path(report_dir).exists() else []:
        try:
            prev = json.loads(rpath.read_text())
            cumulative += prev.get("live_trades", {}).get("count", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    cumulative += len(live_trades)

    # 5) ML 트리거 체크
    ml_trigger = check_ml_trigger(
        cumulative_trades=cumulative,
        current_pf=combined_pf,
        pf_declining_3w=trend["pf_declining_3w"],
    )

    # 6) PF < 1.0이면 스윕 실행
    sweep = None
    if combined_pf < 1.0:
        logger.info("PF < 1.0 — 파라미터 스윕 실행 중...")
        sweep = run_degradation_sweep(symbols, TRAIN_MONTHS, TEST_MONTHS)

    return {
        "date": today.isoformat(),
        "backtest": {"summary": backtest_summary, "per_symbol": bt_results},
        "live_trades": live_summary,
        "trend": trend,
        "ml_trigger": ml_trigger,
        "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description="주간 전략 리포트")
    parser.add_argument("--skip-fetch", action="store_true", help="데이터 수집 스킵")
    parser.add_argument("--date", type=str, help="리포트 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()

    # 1) 데이터 수집
    if not args.skip_fetch:
        fetch_latest_data(SYMBOLS)

    # 2) 리포트 생성
    report = generate_report(
        symbols=SYMBOLS,
        report_date=report_date,
    )

    # 3) 저장
    save_report(report, str(WEEKLY_DIR))

    # 4) Discord 전송
    text = format_report(report)
    print(text)
    send_report(text)


if __name__ == "__main__":
    main()
```

**Step 4: 테스트 실행하여 통과 확인**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 14 PASS

**Step 5: 기존 테스트 스위트 실행하여 회귀 없음 확인**

Run: `source .venv/bin/activate && bash scripts/run_tests.sh`
Expected: 121+ 기존 통과 + 14 신규 = 135+ 통과

**Step 6: 커밋**

```bash
git add scripts/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(weekly-report): 메인 오케스트레이션, CLI, JSON 저장 추가"
```

---

### Task 7: 수동 스모크 테스트 + 크론탭 가이드

**Files:**
- 신규 파일 없음

**Step 1: 드라이 런 (데이터 수집 스킵, Discord 스킵)**

Run:
```bash
source .venv/bin/activate && python scripts/weekly_report.py --skip-fetch --date 2026-03-07
```

Expected: 리포트가 터미널에 출력되고 `results/weekly/report_2026-03-07.json` 저장됨.

**Step 2: 저장된 JSON 확인**

Run: `cat results/weekly/report_2026-03-07.json | python -m json.tool | head -30`
Expected: date, backtest, live_trades, trend, ml_trigger 키가 포함된 유효한 JSON

**Step 3: 최종 상태 커밋**

```bash
git add results/weekly/.gitkeep
git commit -m "chore: results/weekly 디렉토리 추가"
```

**Step 4: 크론탭 설정 문서화**

프로덕션 서버에서:
```bash
# 매주 일요일 새벽 3시 (KST = UTC+9 → UTC 18:00 토요일)
crontab -e
# 추가:
0 18 * * 6 cd /app && python scripts/weekly_report.py >> logs/cron.log 2>&1
```

---

### Task 8: CLAUDE.md 플랜 히스토리 업데이트

**Files:**
- Modify: `CLAUDE.md`

**Step 1: 히스토리 테이블에 플랜 항목 추가**

플랜 히스토리 테이블에 추가:
```
| 2026-03-07 | `weekly-report` (plan) | Completed |
```

**Step 2: Common Commands 섹션에 주간 리포트 명령어 추가**

```bash
# 주간 전략 리포트 (수동)
python scripts/weekly_report.py --skip-fetch

# 주간 리포트 (데이터 새로고침 포함)
python scripts/weekly_report.py
```

**Step 3: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: 주간 리포트를 플랜 히스토리 및 명령어에 추가"
```
