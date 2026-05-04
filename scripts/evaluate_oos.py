"""
MTF Pullback Bot — OOS Dry-run 평가 스크립트
─────────────────────────────────────────────
프로덕션 서버에서 JSONL 거래 기록을 가져와
승률·PF·누적PnL·평균보유시간을 계산하고 LIVE 배포 판정을 출력한다.

비용 모델(수수료·슬리피지·펀딩)을 사후보정으로 적용하여
fees_only / realistic / pessimistic 3개 시나리오 결과를 출력한다.

Usage:
    python scripts/evaluate_oos.py
    python scripts/evaluate_oos.py --symbol xrpusdt
    python scripts/evaluate_oos.py --local  # 로컬 파일만 사용 (서버 fetch 스킵)
    python scripts/evaluate_oos.py --local --scenario all
    python scripts/evaluate_oos.py --local --scenario fees_only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── 비용 모델 import ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import COST_MODEL, COST_SCENARIOS  # noqa: E402

# ── 설정 ──────────────────────────────────────────────────────────
PROD_HOST = "root@10.1.10.24"
REMOTE_DIR = "/root/cointrader/data/trade_history"
LOCAL_DIR = Path("data/trade_history")

# ── 판정 기준 ─────────────────────────────────────────────────────
MIN_TRADES = 5
MIN_PF = 1.0


def fetch_from_prod(filename: str) -> Path:
    """프로덕션 서버에서 JSONL 파일을 scp로 가져온다."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    remote_path = f"{PROD_HOST}:{REMOTE_DIR}/{filename}"
    local_path = LOCAL_DIR / filename

    print(f"[Fetch] {remote_path} → {local_path}")
    result = subprocess.run(
        ["scp", remote_path, str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[Fetch] scp 실패: {result.stderr.strip()}")
        if local_path.exists():
            print(f"[Fetch] 로컬 캐시 사용: {local_path}")
        else:
            print("[Fetch] 로컬 캐시도 없음. 종료.")
            sys.exit(1)
    else:
        print(f"[Fetch] 완료 ({local_path.stat().st_size:,} bytes)")

    return local_path


def load_trades(path: Path) -> pd.DataFrame:
    """JSONL 파일을 DataFrame으로 로드."""
    df = pd.read_json(path, lines=True)

    if df.empty:
        print("[Load] 거래 기록이 비어있습니다.")
        sys.exit(1)

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["duration_min"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60

    print(f"[Load] {len(df)}건 로드 완료 ({df['entry_ts'].min():%Y-%m-%d} ~ {df['exit_ts'].max():%Y-%m-%d})")
    return df


def count_funding_events(entry_ts, exit_ts) -> int:
    """
    Binance USDⓈ-M Futures 펀딩 스냅샷 시각(00/08/16 UTC)이
    [entry_ts, exit_ts] 구간에 몇 번 포함되는지 카운트.
    """
    start = entry_ts.ceil("h")
    end = exit_ts.floor("h")
    if start > end:
        return 0
    hours = pd.date_range(start, end, freq="1h", inclusive="both")
    return sum(1 for h in hours if h.hour % 8 == 0)


def _get_fee_bps(order_type: str) -> float:
    """주문 타입에 따른 수수료 bps 반환."""
    if order_type == "taker":
        return COST_MODEL["taker_fee_bps"]
    return COST_MODEL["maker_fee_bps"]


def calc_trade_cost(row, scenario: dict) -> float:
    """개별 거래의 총 비용(bps)을 계산."""
    # 1) Fee: entry + exit
    entry_fee = _get_fee_bps(COST_MODEL["entry_order_type"])

    # exit order type: SL 히트면 sl_order_type, TP 히트면 tp_order_type
    reason = row.get("reason", "")
    if "SL" in reason:
        exit_fee = _get_fee_bps(COST_MODEL["sl_order_type"])
    else:
        exit_fee = _get_fee_bps(COST_MODEL["tp_order_type"])

    fee = entry_fee + exit_fee

    # 2) Slippage: 왕복
    slippage = scenario["slippage_bps_per_side"] * 2

    # 3) Funding: 경계 교차 카운트
    funding_count = count_funding_events(row["entry_ts"], row["exit_ts"])
    funding = funding_count * scenario["funding_bps_per_8h"]

    return fee + slippage + funding


def apply_cost_model(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """DataFrame에 비용을 적용하여 adjusted_pnl_bps 컬럼 추가."""
    scenario = COST_SCENARIOS[scenario_name]
    result = df.copy()
    result["cost_bps"] = result.apply(lambda row: calc_trade_cost(row, scenario), axis=1)
    result["adjusted_pnl_bps"] = result["pnl_bps"] - result["cost_bps"]
    return result


def calc_metrics(df: pd.DataFrame, pnl_col: str = "pnl_bps") -> dict:
    """핵심 지표 계산. 빈 DataFrame이면 안전한 기본값 반환."""
    n = len(df)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "cum_pnl": 0.0, "avg_pnl": 0.0, "avg_dur": 0.0}

    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] < 0]

    win_rate = len(wins) / n * 100
    gross_profit = wins[pnl_col].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    cum_pnl = df[pnl_col].sum()
    avg_pnl = cum_pnl / n
    avg_dur = df["duration_min"].mean()

    return {
        "trades": n,
        "win_rate": round(win_rate, 1),
        "pf": round(pf, 2),
        "cum_pnl": round(cum_pnl, 1),
        "avg_pnl": round(avg_pnl, 2),
        "avg_dur": round(avg_dur, 1),
    }


def print_report(df: pd.DataFrame):
    """성적표 출력 (raw, 비용 미반영)."""
    total = calc_metrics(df)
    longs = calc_metrics(df[df["side"] == "LONG"])
    shorts = calc_metrics(df[df["side"] == "SHORT"])

    header = f"{'':>10} {'Trades':>8} {'WinRate':>9} {'PF':>8} {'CumPnL':>10} {'AvgDur':>10}"
    sep = "\u2500" * 60

    print()
    print(sep)
    print("  MTF Pullback Bot \u2014 OOS Dry-run \uc131\uc801\ud45c")
    print(sep)
    print(header)
    print(sep)

    for label, m in [("Total", total), ("LONG", longs), ("SHORT", shorts)]:
        pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "\u221e"
        dur_str = f"{m['avg_dur']:.0f}m" if m["trades"] > 0 else "-"
        print(
            f"{label:>10} {m['trades']:>8d} {m['win_rate']:>8.1f}% {pf_str:>8} "
            f"{m['cum_pnl']:>+10.1f} {dur_str:>10}"
        )

    print(sep)

    # ── 개별 거래 내역 ──
    print()
    print("  거래 내역")
    print(sep)
    print(f"{'#':>3} {'Side':>6} {'Entry':>10} {'Exit':>10} {'PnL(bps)':>10} {'Dur':>8} {'Reason'}")
    print(sep)
    for i, row in df.iterrows():
        dur = f"{row['duration_min']:.0f}m"
        reason = row.get("reason", "")
        if len(reason) > 25:
            reason = reason[:25] + "\u2026"
        print(
            f"{i+1:>3} {row['side']:>6} {row['entry_price']:>10.4f} {row['exit_price']:>10.4f} "
            f"{row['pnl_bps']:>+10.1f} {dur:>8} {reason}"
        )
    print(sep)

    # ── 최종 판정 (비용 반영 기준) ──
    # Raw PF는 비현실적 — fees_only 기준으로 판정
    cost_df = apply_cost_model(df, "fees_only")
    cost_total = calc_metrics(cost_df, pnl_col="adjusted_pnl_bps")
    cost_long = calc_metrics(cost_df[cost_df["side"] == "LONG"], pnl_col="adjusted_pnl_bps")
    cost_short = calc_metrics(cost_df[cost_df["side"] == "SHORT"], pnl_col="adjusted_pnl_bps")

    # 대칭성 체크: LONG/SHORT 양쪽 모두 PF >= 0.8 이상이어야 함
    symmetry_ok = True
    if cost_long["trades"] >= 5 and cost_short["trades"] >= 5:
        symmetry_ok = cost_long["pf"] >= 0.8 and cost_short["pf"] >= 0.8

    print()
    if cost_total["trades"] >= MIN_TRADES and cost_total["pf"] >= MIN_PF and symmetry_ok:
        print(f"  [\ud310\uc815: \ud1b5\uacfc] \uc5e3\uc9c0\uac00 \uc99d\uba85\ub418\uc5c8\uc2b5\ub2c8\ub2e4. LIVE \ubc30\ud3ec(\uc790\uae08 \ud22c\uc785)\ub97c \uad8c\uc7a5\ud569\ub2c8\ub2e4.")
        print(f"  (\uac70\ub798\uc218 {cost_total['trades']} >= {MIN_TRADES}, fees_only PF {cost_total['pf']:.2f} >= {MIN_PF:.1f})")
    else:
        reasons = []
        if cost_total["trades"] < MIN_TRADES:
            reasons.append(f"\uac70\ub798\uc218 {cost_total['trades']} < {MIN_TRADES}")
        if cost_total["pf"] < MIN_PF:
            reasons.append(f"fees_only PF {cost_total['pf']:.2f} < {MIN_PF:.1f}")
        if not symmetry_ok:
            reasons.append(f"LONG/SHORT \ube44\ub300\uce6d (L:{cost_long['pf']:.2f} / S:{cost_short['pf']:.2f})")
        print(f"  [\ud310\uc815: \uc2e4\ud328] OOS \uac80\uc99d \uc2e4\ud328. \uc2e4\uc804 \ud22c\uc785 \ubd88\uac00.")
        print(f"  ({', '.join(reasons)})")
    print()


def print_cost_report(df: pd.DataFrame, scenario_names: list[str]):
    """비용 보정 시나리오별 성적표 출력."""
    sep = "\u2500" * 61

    # 시나리오별 데이터 준비
    scenario_dfs = {}
    for name in scenario_names:
        scenario_dfs[name] = apply_cost_model(df, name)

    print()
    print(sep)
    print("  MTF Pullback Bot \u2014 OOS Cost-Adjusted Results")
    print(sep)

    # 헤더
    header = f"{'Scenario:':>16}"
    for name in scenario_names:
        header += f" {name:>14}"
    print(header)
    print(sep)

    # Total / LONG / SHORT 각각
    for section_label, filter_fn in [
        ("Total", lambda d: d),
        ("LONG", lambda d: d[d["side"] == "LONG"]),
        ("SHORT", lambda d: d[d["side"] == "SHORT"]),
    ]:
        print(section_label)

        # 각 시나리오에 대해 metrics 계산
        metrics_list = []
        for name in scenario_names:
            sdf = filter_fn(scenario_dfs[name])
            m = calc_metrics(sdf, pnl_col="adjusted_pnl_bps")
            metrics_list.append(m)

        # Trades
        line = f"{'Trades:':>16}"
        for m in metrics_list:
            line += f" {m['trades']:>14d}"
        print(line)

        # WinRate
        line = f"{'WinRate:':>16}"
        for m in metrics_list:
            line += f" {m['win_rate']:>13.1f}%"
        print(line)

        # PF
        line = f"{'PF:':>16}"
        for m in metrics_list:
            pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "\u221e"
            line += f" {pf_str:>14}"
        print(line)

        # CumPnL(bps)
        line = f"{'CumPnL(bps):':>16}"
        for m in metrics_list:
            line += f" {m['cum_pnl']:>+14.1f}"
        print(line)

        # AvgPnL(bps)
        line = f"{'AvgPnL(bps):':>16}"
        for m in metrics_list:
            line += f" {m['avg_pnl']:>+14.2f}"
        print(line)

        # AvgDur
        line = f"{'AvgDur:':>16}"
        for m in metrics_list:
            dur_str = f"{m['avg_dur']:.0f}m" if m["trades"] > 0 else "-"
            line += f" {dur_str:>14}"
        print(line)

    print(sep)

    # Raw 참고
    raw_total = calc_metrics(df)
    print(f"Raw (\ube44\uc6a9 \ubbf8\ubc18\uc601, \ucc38\uace0\uc6a9):")
    pf_str = f"{raw_total['pf']:.2f}" if raw_total["pf"] != float("inf") else "\u221e"
    print(f"  Total PF: {pf_str}, CumPnL: {raw_total['cum_pnl']:+.1f} bps")
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(description="MTF OOS Dry-run \ud3c9\uac00")
    parser.add_argument("--symbol", default="xrpusdt", help="\uc2ec\ubcfc (\ud30c\uc77c\uba85 \uc18c\ubb38\uc790, \uae30\ubcf8: xrpusdt)")
    parser.add_argument("--local", action="store_true", help="\ub85c\uceec \ud30c\uc77c\ub9cc \uc0ac\uc6a9 (\uc11c\ubc84 fetch \uc2a4\ud0b5)")
    parser.add_argument(
        "--scenario",
        choices=["fees_only", "realistic", "pessimistic", "all"],
        default="all",
        help="\ube44\uc6a9 \ubcf4\uc815 \uc2dc\ub098\ub9ac\uc624 (\uae30\ubcf8: all)",
    )
    args = parser.parse_args()

    # MTF bot은 ccxt 심볼(XRP/USDT:USDT)에서 /,:를 제거하여 파일명 생성
    # → mtf_xrpusdtusdt.jsonl  (심볼 인자 xrpusdt → xrpusdtusdt 변환)
    raw = args.symbol.lower()
    if not raw.endswith("usdt"):
        raw = raw + "usdt"
    # xrpusdt → xrpusdtusdt (ccxt 포맷 XRP/USDT:USDT 의 슬래시·콜론 제거 결과)
    if raw.endswith("usdt") and not raw.endswith("usdtusdt"):
        raw = raw + "usdt"
    filename = f"mtf_{raw}.jsonl"

    if args.local:
        local_path = LOCAL_DIR / filename
        if not local_path.exists():
            print(f"[Error] \ub85c\uceec \ud30c\uc77c \uc5c6\uc74c: {local_path}")
            sys.exit(1)
    else:
        local_path = fetch_from_prod(filename)

    df = load_trades(local_path)

    # 비용 보정 리포트 출력
    if args.scenario == "all":
        scenario_names = ["fees_only", "realistic", "pessimistic"]
    else:
        scenario_names = [args.scenario]

    print_cost_report(df, scenario_names)

    # raw 리포트도 하단에 유지
    print_report(df)


if __name__ == "__main__":
    main()
