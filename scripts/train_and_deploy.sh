#!/usr/bin/env bash
# 맥미니에서 전체 학습 파이프라인을 실행하고 LXC로 배포한다.
# 사용법: bash scripts/train_and_deploy.sh [mlx|lgbm]
#
# 예시:
#   bash scripts/train_and_deploy.sh        # LightGBM (기본값)
#   bash scripts/train_and_deploy.sh mlx    # MLX GPU 학습

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

BACKEND="${1:-lgbm}"

cd "$PROJECT_ROOT"

echo "=== [1/3] 데이터 수집 (XRP + BTC + ETH 3심볼, 1년치) ==="
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/combined_15m.parquet

echo ""
echo "=== [2/3] 모델 학습 (21개 피처: XRP 13 + BTC/ETH 상관관계 8) ==="
DECAY="${TIME_WEIGHT_DECAY:-2.0}"
if [ "$BACKEND" = "mlx" ]; then
    echo "  백엔드: MLX (Apple Silicon GPU), decay=${DECAY}"
    python scripts/train_mlx_model.py --data data/combined_15m.parquet --decay "$DECAY"
else
    echo "  백엔드: LightGBM (CPU), decay=${DECAY}"
    python scripts/train_model.py --data data/combined_15m.parquet --decay "$DECAY"
fi

echo ""
echo "=== [3/3] LXC 배포 ==="
bash scripts/deploy_model.sh "$BACKEND"

echo ""
echo "=== 전체 파이프라인 완료 ==="
echo ""
echo "봇 재시작이 필요하면:"
echo "  ssh root@10.1.10.24 'cd /root/cointrader && docker compose restart cointrader'"
