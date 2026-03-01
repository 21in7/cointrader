#!/usr/bin/env bash
# 반대 시그널 재진입 기능 테스트 스크립트
# 사용법: bash scripts/test_reverse_reenter.sh [task]
#
# 예시:
#   bash scripts/test_reverse_reenter.sh          # 전체 태스크 순서대로 실행
#   bash scripts/test_reverse_reenter.sh 1        # Task 1: 신규 테스트만 (실패 확인)
#   bash scripts/test_reverse_reenter.sh 2        # Task 2: _close_and_reenter 메서드 테스트
#   bash scripts/test_reverse_reenter.sh 3        # Task 3: process_candle 분기 테스트
#   bash scripts/test_reverse_reenter.sh bot      # test_bot.py 전체
#   bash scripts/test_reverse_reenter.sh all      # tests/ 전체

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
else
    echo "경고: 가상환경을 찾을 수 없습니다 ($VENV_PATH). 시스템 Python을 사용합니다." >&2
fi

cd "$PROJECT_ROOT"

TASK="${1:-all}"

# ── 태스크별 테스트 이름 ──────────────────────────────────────────────────────
TASK1_TESTS=(
    "tests/test_bot.py::test_close_and_reenter_calls_open_when_ml_passes"
    "tests/test_bot.py::test_close_and_reenter_skips_open_when_ml_blocks"
    "tests/test_bot.py::test_close_and_reenter_skips_open_when_max_positions_reached"
)

TASK2_TESTS=(
    "tests/test_bot.py::test_close_and_reenter_calls_open_when_ml_passes"
    "tests/test_bot.py::test_close_and_reenter_skips_open_when_ml_blocks"
    "tests/test_bot.py::test_close_and_reenter_skips_open_when_max_positions_reached"
)

TASK3_TESTS=(
    "tests/test_bot.py::test_process_candle_calls_close_and_reenter_on_reverse_signal"
)

run_pytest() {
    echo ""
    echo "▶ pytest $*"
    echo "────────────────────────────────────────"
    python -m pytest "$@" -v
}

case "$TASK" in
    1)
        echo "=== Task 1: 신규 테스트 실행 (구현 전 → FAIL 예상) ==="
        run_pytest "${TASK1_TESTS[@]}"
        ;;
    2)
        echo "=== Task 2: _close_and_reenter 메서드 테스트 (구현 후 → PASS 예상) ==="
        run_pytest "${TASK2_TESTS[@]}"
        ;;
    3)
        echo "=== Task 3: process_candle 분기 테스트 (수정 후 → PASS 예상) ==="
        run_pytest "${TASK3_TESTS[@]}"
        ;;
    bot)
        echo "=== test_bot.py 전체 ==="
        run_pytest tests/test_bot.py
        ;;
    all)
        echo "=== 전체 테스트 스위트 ==="
        run_pytest tests/
        ;;
    *)
        echo "알 수 없는 태스크: $TASK"
        echo "사용법: bash scripts/test_reverse_reenter.sh [1|2|3|bot|all]"
        exit 1
        ;;
esac

echo ""
echo "=== 완료 ==="
