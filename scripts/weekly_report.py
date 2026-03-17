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
import os
import subprocess
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

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


# ── 대시보드 API에서 실전 트레이드 가져오기 ──────────────────────────
DASHBOARD_API_URL = os.getenv("DASHBOARD_API_URL", "http://10.1.10.24:8000")


def fetch_live_trades(api_url: str = DASHBOARD_API_URL, limit: int = 500) -> list[dict]:
    """운영 LXC 대시보드 API에서 청산된 트레이드 내역을 가져온다."""
    try:
        resp = httpx.get(f"{api_url}/api/trades", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("trades", [])
    except Exception as e:
        logger.warning(f"대시보드 API 트레이드 조회 실패: {e}")
        return []


def fetch_live_stats(api_url: str = DASHBOARD_API_URL) -> dict:
    """운영 LXC 대시보드 API에서 전체 통계를 가져온다."""
    try:
        resp = httpx.get(f"{api_url}/api/stats", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"대시보드 API 통계 조회 실패: {e}")
        return {}


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


def _sanitize(obj):
    """JSON 직렬화를 위해 numpy/inf 값을 변환."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
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


def generate_quantstats_report(
    trades: list[dict],
    output_path: str,
    title: str = "CoinTrader 주간 전략 리포트",
    initial_balance: float = 1000.0,
) -> str | None:
    """백테스트 트레이드 결과로 quantstats HTML 리포트를 생성한다."""
    if not trades:
        logger.warning("트레이드가 없어 quantstats 리포트를 생성할 수 없습니다.")
        return None

    try:
        import quantstats as qs

        # 트레이드 PnL을 일별 수익률 시계열로 변환
        records = []
        for t in trades:
            exit_time = pd.Timestamp(t["exit_time"])
            records.append({"date": exit_time.date(), "pnl": t["net_pnl"]})

        df = pd.DataFrame(records)
        daily_pnl = df.groupby("date")["pnl"].sum()
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
        daily_pnl = daily_pnl.sort_index()

        # PnL → 수익률로 변환 (equity 기반)
        equity = initial_balance + daily_pnl.cumsum()
        returns = equity.pct_change().fillna(daily_pnl.iloc[0] / initial_balance)

        qs.reports.html(returns, output=output_path, title=title, download_filename=output_path)
        logger.info(f"quantstats HTML 리포트 저장: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"quantstats 리포트 생성 실패: {e}")
        return None


def save_report(report: dict, report_dir: str) -> Path:
    """리포트를 JSON으로 저장하고 경로를 반환한다."""
    rdir = Path(report_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"report_{report['date']}.json"
    with open(path, "w") as f:
        json.dump(_sanitize(report), f, indent=2, ensure_ascii=False)
    logger.info(f"리포트 저장: {path}")
    return path


def _calc_combined_summary(trades: list[dict], initial_balance: float = 1000.0) -> dict:
    """개별 트레이드 리스트에서 합산 지표를 직접 계산한다."""
    if not trades:
        return {
            "profit_factor": 0.0, "win_rate": 0.0, "max_drawdown_pct": 0.0,
            "total_trades": 0, "total_pnl": 0.0,
        }

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    # 시간순 정렬 후 포트폴리오 equity curve 기반 MDD
    sorted_trades = sorted(trades, key=lambda t: t["exit_time"])
    sorted_pnls = [t["net_pnl"] for t in sorted_trades]
    cumulative = np.cumsum(sorted_pnls)
    equity = initial_balance + cumulative
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    mdd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0.0

    return {
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "max_drawdown_pct": round(mdd, 1),
        "total_trades": len(trades),
        "total_pnl": round(sum(pnls), 2),
    }


def generate_report(
    symbols: list[str],
    report_dir: str = str(WEEKLY_DIR),
    report_date: date | None = None,
    api_url: str | None = None,
) -> dict:
    """전체 주간 리포트를 생성한다."""
    today = report_date or date.today()
    dashboard_url = api_url or DASHBOARD_API_URL

    # 1) Walk-Forward 백테스트 (심볼별)
    logger.info("백테스트 실행 중...")
    bt_results = {}
    all_bt_trades = []

    for sym in symbols:
        result = run_backtest([sym], TRAIN_MONTHS, TEST_MONTHS, PROD_PARAMS)
        bt_results[sym] = result["summary"]
        all_bt_trades.extend(result.get("trades", []))

    # 합산 지표를 개별 트레이드에서 직접 계산 (간접 역산 제거)
    backtest_summary = _calc_combined_summary(all_bt_trades)

    # 2) 운영 대시보드 API에서 실전 트레이드 조회
    logger.info(f"대시보드 API에서 실전 트레이드 조회 중... ({dashboard_url})")
    live_stats = fetch_live_stats(dashboard_url)
    live_trades_list = fetch_live_trades(dashboard_url)

    live_count = live_stats.get("total_trades", len(live_trades_list))
    live_wins = live_stats.get("wins", 0)
    live_pnl = live_stats.get("total_pnl", 0)
    live_summary = {
        "count": live_count,
        "net_pnl": round(float(live_pnl), 2),
        "win_rate": round(live_wins / live_count * 100, 1) if live_count > 0 else 0,
    }

    # 3) 추이 로드
    trend = load_trend(report_dir)

    # 4) 누적 트레이드 수 (실전 + 이전 리포트)
    cumulative = live_count
    rdir = Path(report_dir)
    if rdir.exists():
        for rpath in sorted(rdir.glob("report_*.json")):
            try:
                prev = json.loads(rpath.read_text())
                cumulative += prev.get("live_trades", {}).get("count", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    # 5) ML 트리거 체크
    current_pf = backtest_summary["profit_factor"]
    ml_trigger = check_ml_trigger(
        cumulative_trades=cumulative,
        current_pf=current_pf,
        pf_declining_3w=trend["pf_declining_3w"],
    )

    # 6) PF < 1.0이면 스윕 실행
    sweep = None
    if current_pf < 1.0:
        logger.info("PF < 1.0 — 파라미터 스윕 실행 중...")
        sweep = run_degradation_sweep(symbols, TRAIN_MONTHS, TEST_MONTHS)

    return {
        "date": today.isoformat(),
        "backtest": {"summary": backtest_summary, "per_symbol": bt_results, "trades": all_bt_trades},
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
    report = generate_report(symbols=SYMBOLS, report_date=report_date)

    # 3) 저장
    save_report(report, str(WEEKLY_DIR))

    # 4) quantstats HTML 리포트
    bt_trades = report["backtest"].get("trades", [])
    if bt_trades:
        html_path = str(WEEKLY_DIR / f"report_{report['date']}.html")
        generate_quantstats_report(bt_trades, html_path, title=f"CoinTrader 주간 리포트 ({report['date']})")

    # 5) Discord 전송
    text = format_report(report)
    print(text)
    send_report(text)


if __name__ == "__main__":
    main()
