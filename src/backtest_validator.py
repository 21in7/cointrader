"""
백테스트 결과 Sanity Check 검증.
논리적 불변 조건(FAIL) + 통계적 이상 감지(WARNING)를 수행한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    level: str   # "FAIL" | "WARNING"
    message: str


def validate(trades: list[dict], summary: dict, cfg) -> dict:
    """
    모든 검증을 실행하고 결과를 dict로 반환한다.
    CLI에도 PASS/WARNING/FAIL을 출력한다.
    """
    results: list[CheckResult] = []

    # 검증 1: 논리적 불변 조건
    results.extend(_check_invariants(trades))

    # 검증 2: 통계적 이상 감지
    results.extend(_check_statistics(trades, summary))

    # 결과 출력
    _print_results(results)

    return {
        "overall": "PASS" if all(r.passed for r in results) else "FAIL",
        "checks": [
            {"name": r.name, "passed": r.passed, "level": r.level, "message": r.message}
            for r in results
        ],
    }


def _check_invariants(trades: list[dict]) -> list[CheckResult]:
    """논리적 불변 조건. 하나라도 위반 시 FAIL."""
    results = []

    if not trades:
        results.append(CheckResult(
            "trade_count", True, "FAIL", "트레이드 없음 (검증 스킵)"
        ))
        return results

    # 1. 청산 시각 >= 진입 시각 (END_OF_DATA는 동일 캔들 가능)
    bad_times = []
    for i, t in enumerate(trades):
        if pd.Timestamp(t["exit_time"]) < pd.Timestamp(t["entry_time"]):
            bad_times.append(i)
    passed = len(bad_times) == 0
    results.append(CheckResult(
        "exit_after_entry",
        passed,
        "FAIL",
        f"모든 트레이드에서 청산 > 진입" if passed else f"위반 트레이드 인덱스: {bad_times}",
    ))

    # 2. SL/TP 방향 정합성
    bad_sltp = []
    for i, t in enumerate(trades):
        if t["side"] == "LONG":
            if not (t["sl"] < t["entry_price"] < t["tp"]):
                bad_sltp.append(i)
        else:
            if not (t["tp"] < t["entry_price"] < t["sl"]):
                bad_sltp.append(i)
    passed = len(bad_sltp) == 0
    results.append(CheckResult(
        "sl_tp_direction",
        passed,
        "FAIL",
        "SL/TP 방향 정합" if passed else f"위반 트레이드 인덱스: {bad_sltp}",
    ))

    # 3. 포지션 비중첩 (같은 심볼에서 직전 청산 ≤ 다음 진입)
    by_symbol: dict[str, list[dict]] = {}
    for t in trades:
        by_symbol.setdefault(t["symbol"], []).append(t)

    overlap_symbols = []
    for sym, sym_trades in by_symbol.items():
        sorted_trades = sorted(sym_trades, key=lambda x: pd.Timestamp(x["entry_time"]))
        for j in range(1, len(sorted_trades)):
            prev_exit = pd.Timestamp(sorted_trades[j - 1]["exit_time"])
            curr_entry = pd.Timestamp(sorted_trades[j]["entry_time"])
            if prev_exit > curr_entry:
                overlap_symbols.append(sym)
                break
    passed = len(overlap_symbols) == 0
    results.append(CheckResult(
        "no_overlap",
        passed,
        "FAIL",
        "포지션 비중첩 확인" if passed else f"중첩 심볼: {overlap_symbols}",
    ))

    # 4. 수수료 항상 양수
    bad_fees = [i for i, t in enumerate(trades) if t["entry_fee"] <= 0 or t["exit_fee"] <= 0]
    passed = len(bad_fees) == 0
    results.append(CheckResult(
        "positive_fees",
        passed,
        "FAIL",
        "수수료 양수 확인" if passed else f"위반 트레이드 인덱스: {bad_fees}",
    ))

    # 5. 잔고가 음수가 된 적 없음
    balance = 1000.0  # cfg.initial_balance를 몰라도 trades에서 추적 가능
    min_balance = balance
    for t in trades:
        balance += t["net_pnl"]
        min_balance = min(min_balance, balance)
    passed = min_balance >= 0
    results.append(CheckResult(
        "no_negative_balance",
        passed,
        "FAIL",
        "잔고 양수 유지" if passed else f"최저 잔고: {min_balance:.4f}",
    ))

    return results


def _check_statistics(trades: list[dict], summary: dict) -> list[CheckResult]:
    """통계적 이상 감지. WARNING 수준."""
    results = []

    if not trades:
        return results

    win_rate = summary.get("win_rate", 0)
    mdd = summary.get("max_drawdown_pct", 0)
    pf = summary.get("profit_factor", 0)

    # 승률 > 80%
    passed = win_rate <= 80
    results.append(CheckResult(
        "win_rate_high",
        passed,
        "WARNING",
        f"승률 정상 ({win_rate:.1f}%)" if passed else f"승률 {win_rate:.1f}% > 80% — look-ahead bias 의심",
    ))

    # 승률 < 20%
    passed = win_rate >= 20
    results.append(CheckResult(
        "win_rate_low",
        passed,
        "WARNING",
        f"승률 정상 ({win_rate:.1f}%)" if passed else f"승률 {win_rate:.1f}% < 20% — 신호 로직 반전 의심",
    ))

    # MDD 0%
    passed = mdd > 0
    results.append(CheckResult(
        "mdd_nonzero",
        passed,
        "WARNING",
        f"MDD 정상 ({mdd:.1f}%)" if passed else "MDD 0% — SL 미작동 의심",
    ))

    # 월 평균 거래 < 5건
    if len(trades) >= 2:
        first = pd.Timestamp(trades[0]["entry_time"])
        last = pd.Timestamp(trades[-1]["entry_time"])
        months = max(1, (last - first).days / 30)
        trades_per_month = len(trades) / months
        passed = trades_per_month >= 5
        results.append(CheckResult(
            "trade_frequency",
            passed,
            "WARNING",
            f"월 평균 {trades_per_month:.1f}건" if passed else f"월 평균 {trades_per_month:.1f}건 < 5건 — 신호 생성 부족",
        ))

    # Profit Factor > 5.0
    if pf != float("inf"):
        passed = pf <= 5.0
        results.append(CheckResult(
            "profit_factor_high",
            passed,
            "WARNING",
            f"PF 정상 ({pf:.2f})" if passed else f"PF {pf:.2f} > 5.0 — 비현실적 수익",
        ))

    return results


def _print_results(results: list[CheckResult]):
    print("\n" + "=" * 60)
    print("  BACKTEST SANITY CHECK")
    print("=" * 60)

    has_fail = any(not r.passed and r.level == "FAIL" for r in results)
    has_warn = any(not r.passed and r.level == "WARNING" for r in results)

    for r in results:
        if r.passed:
            status = f"{GREEN}PASS{RESET}"
        elif r.level == "FAIL":
            status = f"{RED}FAIL{RESET}"
        else:
            status = f"{YELLOW}WARNING{RESET}"
        print(f"  [{status}] {r.name}: {r.message}")

    print("-" * 60)
    if has_fail:
        print(f"  {RED}RESULT: FAIL — 논리적 불변 조건 위반{RESET}")
    elif has_warn:
        print(f"  {YELLOW}RESULT: WARNING — 수동 확인 필요{RESET}")
    else:
        print(f"  {GREEN}RESULT: ALL PASS{RESET}")
    print("=" * 60 + "\n")
