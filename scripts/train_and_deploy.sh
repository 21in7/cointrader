#!/usr/bin/env bash
# 맥미니에서 전체 학습 파이프라인을 실행하고 LXC로 배포한다.
# 사용법: bash scripts/train_and_deploy.sh [LXC_HOST] [LXC_MODELS_PATH]
#
# 예시:
#   bash scripts/train_and_deploy.sh
#   bash scripts/train_and_deploy.sh root@10.1.10.24 /root/cointrader/models

set -euo pipefail

LXC_HOST="${1:-root@10.1.10.24}"
LXC_MODELS_PATH="${2:-/root/cointrader/models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== [1/3] 데이터 수집 ==="
python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90 --output data/xrpusdt_1m.parquet

echo ""
echo "=== [2/3] 모델 학습 ==="
# TRAIN_BACKEND=mlx 로 설정하면 Apple Silicon GPU(Metal)를 사용한다 (기본: lgbm)
BACKEND="${TRAIN_BACKEND:-lgbm}"
if [ "$BACKEND" = "mlx" ]; then
    echo "  백엔드: MLX (Apple Silicon GPU)"
    python scripts/train_mlx_model.py --data data/xrpusdt_1m.parquet
else
    echo "  백엔드: LightGBM (CPU)"
    python scripts/train_model.py --data data/xrpusdt_1m.parquet
fi

echo ""
echo "=== [3/3] LXC 배포 ==="
bash scripts/deploy_model.sh "$LXC_HOST" "$LXC_MODELS_PATH"

echo ""
echo "=== 전체 파이프라인 완료 ==="
echo ""
echo "봇 재시작이 필요하면:"
echo "  ssh ${LXC_HOST} 'cd /root/cointrader && docker compose restart cointrader'"
