#!/usr/bin/env bash
# 전체 테스트 실행 스크립트
#
# 사용법:
#   bash scripts/run_tests.sh           # 전체 실행
#   bash scripts/run_tests.sh -k bot    # 특정 키워드 필터

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

python -m pytest tests/ \
    -v \
    "$@"
