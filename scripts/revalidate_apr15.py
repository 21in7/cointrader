"""
4월 15일 재검증 스크립트 — L/S ratio + FR×OI 동시 재실행

crontab: 0 10 15 4 * cd /root/cointrader && /root/cointrader/.venv/bin/python scripts/revalidate_apr15.py

재검증 대상:
1. L/S ratio (top_acct_ls_ratio) — 24일 데이터로 6개 조합
2. FR × OI변화율(1h) — 29일 데이터로 12개 조합
3. 대칭성 재판정

Usage: python scripts/revalidate_apr15.py
"""

import subprocess
import sys
from datetime import datetime, timezone

def main():
    now = datetime.now(timezone.utc)
    print("=" * 80)
    print(f"  4월 재검증 실행 — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 80)

    print("\n[1/2] L/S ratio 백테스트 재실행")
    print("-" * 40)
    r1 = subprocess.run(
        [sys.executable, "scripts/ls_ratio_backtest.py"],
        capture_output=False,
    )

    print("\n\n[2/2] FR × OI 백테스트 재실행")
    print("-" * 40)
    r2 = subprocess.run(
        [sys.executable, "scripts/fr_oi_backtest.py"],
        capture_output=False,
    )

    print("\n" + "=" * 80)
    print("  재검증 완료")
    print("=" * 80)
    print(f"\n  L/S ratio: {'성공' if r1.returncode == 0 else '실패'}")
    print(f"  FR × OI:   {'성공' if r2.returncode == 0 else '실패'}")
    print(f"\n  판정 기준:")
    print(f"    - L/S ratio: PF > 1.0인 조합 있으면 재검토")
    print(f"    - FR × OI: SHORT+LONG 모두 PF > 1.0이면 대칭성 통과")
    print(f"    - 둘 다 실패 시 확정 폐기")

if __name__ == "__main__":
    main()
