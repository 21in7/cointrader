#!/usr/bin/env bash
# 맥미니에서 전체 학습 파이프라인을 실행하고 LXC로 배포한다.
# 사용법: bash scripts/train_and_deploy.sh [mlx|lgbm] [wf-splits]
#
# 예시:
#   bash scripts/train_and_deploy.sh             # LightGBM + Walk-Forward 5폴드 (기본값)
#   bash scripts/train_and_deploy.sh mlx         # MLX GPU 학습 + Walk-Forward 5폴드
#   bash scripts/train_and_deploy.sh lgbm 3      # LightGBM + Walk-Forward 3폴드
#   bash scripts/train_and_deploy.sh mlx 0       # MLX 학습만 (Walk-Forward 건너뜀)
#   bash scripts/train_and_deploy.sh lgbm 0      # LightGBM 학습만 (Walk-Forward 건너뜀)

set -euo pipefail

# cron 환경에서 sysctl 경로 누락 방지
export PATH="/usr/sbin:/usr/bin:/bin:/opt/homebrew/bin:$PATH"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}"

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
WF_SPLITS="${2:-5}"   # 두 번째 인자: Walk-Forward 폴드 수 (0이면 건너뜀)

cd "$PROJECT_ROOT"

mkdir -p data

PARQUET_FILE="data/combined_15m.parquet"

echo ""
echo "========================================"
echo "  학습 파이프라인 시작: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
echo ""

echo "=== [1/3] 데이터 수집 (XRP + BTC + ETH 3심볼 + OI/펀딩비) ==="
if [ ! -f "$PARQUET_FILE" ]; then
    echo "  [최초 실행] 기존 데이터 없음 → 1년치(365일) 전체 수집 (--no-upsert)"
    FETCH_DAYS=365
    UPSERT_FLAG="--no-upsert"
else
    echo "  [일반 실행] 기존 데이터 존재 → 35일치 Upsert (OI/펀딩비 0.0 구간 보충)"
    FETCH_DAYS=35
    UPSERT_FLAG=""
fi

python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days "$FETCH_DAYS" \
    $UPSERT_FLAG \
    --output "$PARQUET_FILE"

DECAY="${TIME_WEIGHT_DECAY:-2.0}"

echo ""
echo "=== [1.5/3] OI 파생 피처 A/B 비교 ==="
python scripts/train_model.py --compare --data "$PARQUET_FILE" --decay "$DECAY" || true

echo ""
echo "=== [2/3] 모델 학습 (26개 피처: XRP 13 + BTC/ETH 8 + OI/펀딩비 2 + OI파생 2 + ADX) ==="
if [ "$BACKEND" = "mlx" ]; then
    echo "  백엔드: MLX (Apple Silicon GPU), decay=${DECAY}"
    python scripts/train_mlx_model.py --data data/combined_15m.parquet --decay "$DECAY"
else
    echo "  백엔드: LightGBM (CPU), decay=${DECAY}"
    python scripts/train_model.py --data data/combined_15m.parquet --decay "$DECAY"
fi

# Walk-Forward 검증 (WF_SPLITS > 0 인 경우)
if [ "$WF_SPLITS" -gt 0 ] 2>/dev/null; then
    echo ""
    echo "=== [2.5/3] Walk-Forward 검증 (${WF_SPLITS}폴드) ==="
    if [ "$BACKEND" = "mlx" ]; then
        python scripts/train_mlx_model.py \
            --data data/combined_15m.parquet \
            --decay "$DECAY" \
            --wf \
            --wf-splits "$WF_SPLITS"
    else
        python scripts/train_model.py \
            --data data/combined_15m.parquet \
            --decay "$DECAY" \
            --wf \
            --wf-splits "$WF_SPLITS"
    fi
fi

echo ""
echo "=== [3/3] LXC 배포 ==="
bash scripts/deploy_model.sh "$BACKEND"

echo ""
echo "=== 전체 파이프라인 완료: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo ""
echo "봇 재시작이 필요하면:"
echo "  ssh root@10.1.10.24 'cd /root/cointrader && docker compose restart cointrader'"
