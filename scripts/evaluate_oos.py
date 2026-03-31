"""
MTF Pullback Bot — OOS Dry-run 평가 스크립트
─────────────────────────────────────────────
프로덕션 서버에서 JSONL 거래 기록을 가져와
승률·PF·누적PnL·평균보유시간을 계산하고 LIVE 배포 판정을 출력한다.

Usage:
    python scripts/evaluate_oos.py
    python scripts/evaluate_oos.py --symbol xrpusdt
    python scripts/evaluate_oos.py --local  # 로컬 파일만 사용 (서버 fetch 스킵)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

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


def calc_metrics(df: pd.DataFrame) -> dict:
    """핵심 지표 계산. 빈 DataFrame이면 안전한 기본값 반환."""
    n = len(df)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "cum_pnl": 0.0, "avg_dur": 0.0}

    wins = df[df["pnl_bps"] > 0]
    losses = df[df["pnl_bps"] < 0]

    win_rate = len(wins) / n * 100
    gross_profit = wins["pnl_bps"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl_bps"].sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    cum_pnl = df["pnl_bps"].sum()
    avg_dur = df["duration_min"].mean()

    return {
        "trades": n,
        "win_rate": round(win_rate, 1),
        "pf": round(pf, 2),
        "cum_pnl": round(cum_pnl, 1),
        "avg_dur": round(avg_dur, 1),
    }


def print_report(df: pd.DataFrame):
    """성적표 출력."""
    total = calc_metrics(df)
    longs = calc_metrics(df[df["side"] == "LONG"])
    shorts = calc_metrics(df[df["side"] == "SHORT"])

    header = f"{'':>10} {'Trades':>8} {'WinRate':>9} {'PF':>8} {'CumPnL':>10} {'AvgDur':>10}"
    sep = "─" * 60

    print()
    print(sep)
    print("  MTF Pullback Bot — OOS Dry-run 성적표")
    print(sep)
    print(header)
    print(sep)

    for label, m in [("Total", total), ("LONG", longs), ("SHORT", shorts)]:
        pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "∞"
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
            reason = reason[:25] + "…"
        print(
            f"{i+1:>3} {row['side']:>6} {row['entry_price']:>10.4f} {row['exit_price']:>10.4f} "
            f"{row['pnl_bps']:>+10.1f} {dur:>8} {reason}"
        )
    print(sep)

    # ── 최종 판정 ──
    print()
    if total["trades"] >= MIN_TRADES and total["pf"] >= MIN_PF:
        print(f"  [판정: 통과] 엣지가 증명되었습니다. LIVE 배포(자금 투입)를 권장합니다.")
        print(f"  (거래수 {total['trades']} >= {MIN_TRADES}, PF {total['pf']:.2f} >= {MIN_PF:.1f})")
    else:
        reasons = []
        if total["trades"] < MIN_TRADES:
            reasons.append(f"거래수 {total['trades']} < {MIN_TRADES}")
        if total["pf"] < MIN_PF:
            reasons.append(f"PF {total['pf']:.2f} < {MIN_PF:.1f}")
        print(f"  [판정: 보류] 기준 미달. OOS 검증 실패로 실전 투입을 보류합니다.")
        print(f"  ({', '.join(reasons)})")
    print()


def main():
    parser = argparse.ArgumentParser(description="MTF OOS Dry-run 평가")
    parser.add_argument("--symbol", default="xrpusdt", help="심볼 (파일명 소문자, 기본: xrpusdt)")
    parser.add_argument("--local", action="store_true", help="로컬 파일만 사용 (서버 fetch 스킵)")
    args = parser.parse_args()

    filename = f"mtf_{args.symbol}.jsonl"

    if args.local:
        local_path = LOCAL_DIR / filename
        if not local_path.exists():
            print(f"[Error] 로컬 파일 없음: {local_path}")
            sys.exit(1)
    else:
        local_path = fetch_from_prod(filename)

    df = load_trades(local_path)
    print_report(df)


if __name__ == "__main__":
    main()
