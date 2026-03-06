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
import os
import re
import subprocess
from datetime import date, timedelta

from loguru import logger

from src.backtester import WalkForwardBacktester, WalkForwardConfig
from src.notifier import DiscordNotifier


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


# ── ML 재학습 트리거 & 성능 저하 스윕 ─────────────────────────────
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


# ── Discord 리포트 포맷 & 전송 ─────────────────────────────────────

_EMOJI_CHART = "\U0001F4CA"
_EMOJI_ALERT = "\U0001F6A8"
_EMOJI_BELL = "\U0001F514"
_CHECK = "\u2705"
_UNCHECK = "\u2610"
_WARN = "\u26A0"
_ARROW = "\u2192"


def format_report(data: dict) -> str:
    """리포트 데이터를 Discord 메시지 텍스트로 포맷한다."""
    d = data["date"]
    bt = data["backtest"]["summary"]
    pf = bt["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"

    status = ""
    if pf < 1.0:
        status = f"  {_EMOJI_ALERT} 손실 구간"

    lines = [
        f"{_EMOJI_CHART} 주간 전략 리포트 ({d})",
        "",
        "[현재 성능 — Walk-Forward 백테스트]",
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
            "[실전 트레이드 (이번 주)]",
            f"  거래: {lt['count']}건 | 순수익: {lt['net_pnl']:+.2f} USDT | 승률: {lt['win_rate']:.1f}%",
        ]

    # 추이
    trend = data["trend"]
    if trend["pf"]:
        pf_trend = f" {_ARROW} ".join(f"{v:.2f}" for v in trend["pf"])
        warn = f"  {_WARN} 하락 추세" if trend["pf_declining_3w"] else ""
        pf_len = len(trend["pf"])
        lines += ["", f"[추이 (최근 {pf_len}주)]", f"  PF: {pf_trend}{warn}"]
        if trend["win_rate"]:
            wr_trend = f" {_ARROW} ".join(f"{v:.0f}%" for v in trend["win_rate"])
            lines.append(f"  승률: {wr_trend}")
        if trend["mdd"]:
            mdd_trend = f" {_ARROW} ".join(f"{v:.0f}%" for v in trend["mdd"])
            lines.append(f"  MDD: {mdd_trend}")

    # ML 재도전 체크리스트
    ml = data["ml_trigger"]
    cond = ml["conditions"]
    threshold = ml["threshold"]
    cum_trades = ml["cumulative_trades"]

    c1 = _CHECK if cond["cumulative_trades_enough"] else _UNCHECK
    c2 = _CHECK if cond["pf_below_1"] else _UNCHECK
    c3 = _CHECK if cond["pf_declining_3w"] else _UNCHECK
    pf_below_label = "예" if cond["pf_below_1"] else "아니오"
    pf_dec_label = f"예 {_WARN}" if cond["pf_declining_3w"] else "아니오"

    lines += [
        "",
        "[ML 재도전 체크리스트]",
        f"  {c1} 누적 트레이드 \u2265 {threshold}건: {cum_trades}/{threshold}",
        f"  {c2} PF < 1.0: {pf_below_label} (현재 {pf_str})",
        f"  {c3} PF 3주 연속 하락: {pf_dec_label}",
    ]
    met_count = ml["met_count"]
    if ml["recommend"]:
        lines.append(f"  {_ARROW} {_EMOJI_BELL} ML 재학습 권장! ({met_count}/3 충족)")
    else:
        lines.append(f"  {_ARROW} ML 재도전 시점: 아직 아님 ({met_count}/3 충족)")

    # 파라미터 스윕
    sweep = data.get("sweep")
    if sweep:
        lines += ["", "[파라미터 스윕 결과]"]
        lines.append(f"  현재: {_param_str(PROD_PARAMS)} {_ARROW} PF {pf_str}")
        for i, alt in enumerate(sweep):
            apf = alt["summary"]["profit_factor"]
            apf_str = f"{apf:.2f}" if apf != float("inf") else "INF"
            diff = apf - pf
            idx = i + 1
            lines.append(f"  대안 {idx}: {_param_str(alt['params'])} {_ARROW} PF {apf_str} ({diff:+.2f})")
        lines.append("")
        lines.append(f"  {_WARN} 자동 적용되지 않음. 검토 후 승인 필요.")
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
