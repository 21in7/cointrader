#!/usr/bin/env bash
# Optuna로 LightGBM 하이퍼파라미터를 탐색하고 결과를 출력한다.
# 사람이 결과를 확인·승인한 후 train_model.py에 수동으로 반영하는 방식.
#
# 사용법:
#   bash scripts/run_optuna.sh              # 기본 (50 trials, 5폴드)
#   bash scripts/run_optuna.sh 100          # 100 trials
#   bash scripts/run_optuna.sh 100 3        # 100 trials, 3폴드
#   bash scripts/run_optuna.sh 10 3 --no-baseline  # 빠른 테스트
#
# 결과 확인 후 승인하면:
#   python scripts/train_model.py --tuned-params models/tune_results_YYYYMMDD_HHMMSS.json

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

TRIALS="${1:-50}"
FOLDS="${2:-5}"
EXTRA_ARGS="${3:-}"

cd "$PROJECT_ROOT"

echo "=== Optuna 하이퍼파라미터 탐색 ==="
echo "  trials=${TRIALS}, folds=${FOLDS}"
echo ""

python scripts/tune_hyperparams.py \
    --trials "$TRIALS" \
    --folds  "$FOLDS" \
    $EXTRA_ARGS

echo ""
echo "=== 탐색 완료 ==="
echo ""
echo "결과 JSON을 확인하고 승인하면 아래 명령으로 재학습하세요:"
echo "  python scripts/train_model.py --tuned-params models/tune_results_<timestamp>.json"
echo ""
echo "Walk-Forward 검증과 함께 재학습:"
echo "  python scripts/train_model.py --tuned-params models/tune_results_<timestamp>.json --wf"
