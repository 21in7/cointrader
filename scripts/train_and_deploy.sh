#!/usr/bin/env bash
# 맥미니에서 전체 학습 파이프라인을 실행하고 LXC로 배포한다.
# 사용법: bash scripts/train_and_deploy.sh [mlx|lgbm] [--symbol TRXUSDT] [--all] [wf-splits]
#
# 예시:
#   bash scripts/train_and_deploy.sh                        # 전체 심볼 (SYMBOLS 환경변수) + LightGBM
#   bash scripts/train_and_deploy.sh --symbol TRXUSDT       # TRXUSDT만 학습+배포
#   bash scripts/train_and_deploy.sh mlx --symbol TRXUSDT   # MLX + TRXUSDT만
#   bash scripts/train_and_deploy.sh --all                  # 전체 심볼 순차 처리
#   bash scripts/train_and_deploy.sh lgbm 3                 # 전체 심볼 + Walk-Forward 3폴드
#   bash scripts/train_and_deploy.sh mlx 0                  # 전체 심볼 + MLX 학습만 (WF 건너뜀)

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

cd "$PROJECT_ROOT"

# ── 인자 파싱 ───────────────────────────────────────────────────────────────
BACKEND="lgbm"
WF_SPLITS="5"
SYMBOL_ARG=""
ALL_FLAG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --symbol)
            SYMBOL_ARG="$2"
            shift 2
            ;;
        --all)
            ALL_FLAG=true
            shift
            ;;
        mlx|lgbm)
            BACKEND="$1"
            shift
            ;;
        *)
            # 숫자면 WF_SPLITS로 처리
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                WF_SPLITS="$1"
            fi
            shift
            ;;
    esac
done

# ── 대상 심볼 결정 ──────────────────────────────────────────────────────────
if [ -n "$SYMBOL_ARG" ]; then
    TARGETS=("$SYMBOL_ARG")
else
    # .env에서 SYMBOLS 로드 (없으면 XRPUSDT 기본값)
    TARGETS=($(python -c "from dotenv import load_dotenv; load_dotenv(); from src.config import Config; c=Config(); print(' '.join(c.symbols))"))
fi

DECAY="${TIME_WEIGHT_DECAY:-2.0}"

echo ""
echo "========================================"
echo "  학습 파이프라인 시작: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  대상 심볼: ${TARGETS[*]}"
echo "  백엔드: ${BACKEND}, WF 폴드: ${WF_SPLITS}"
echo "========================================"
echo ""

# ── 심볼별 파이프라인 ───────────────────────────────────────────────────────
for SYM in "${TARGETS[@]}"; do
    SYM_LOWER=$(echo "$SYM" | tr '[:upper:]' '[:lower:]')
    mkdir -p "data/$SYM_LOWER" "models/$SYM_LOWER"

    PARQUET_FILE="data/$SYM_LOWER/combined_15m.parquet"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$SYM] 파이프라인 시작"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # === [1/3] 데이터 수집 ===
    echo ""
    echo "=== [$SYM] [1/3] 데이터 수집 (+ BTC/ETH 상관관계 + OI/펀딩비) ==="
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
        --symbol "$SYM" \
        --interval 15m \
        --days "$FETCH_DAYS" \
        $UPSERT_FLAG

    # === [1.5/3] OI 파생 피처 A/B 비교 ===
    echo ""
    echo "=== [$SYM] [1.5/3] OI 파생 피처 A/B 비교 ==="
    python scripts/train_model.py --compare --symbol "$SYM" --decay "$DECAY" || true

    # === [2/3] 모델 학습 ===
    echo ""
    echo "=== [$SYM] [2/3] 모델 학습 ==="
    if [ "$BACKEND" = "mlx" ]; then
        echo "  백엔드: MLX (Apple Silicon GPU), decay=${DECAY}"
        python scripts/train_mlx_model.py --data "$PARQUET_FILE" --decay "$DECAY"
    else
        echo "  백엔드: LightGBM (CPU), decay=${DECAY}"
        python scripts/train_model.py --symbol "$SYM" --decay "$DECAY"
    fi

    # Walk-Forward 검증 (WF_SPLITS > 0 인 경우)
    if [ "$WF_SPLITS" -gt 0 ] 2>/dev/null; then
        echo ""
        echo "=== [$SYM] [2.5/3] Walk-Forward 검증 (${WF_SPLITS}폴드) ==="
        if [ "$BACKEND" = "mlx" ]; then
            python scripts/train_mlx_model.py \
                --data "$PARQUET_FILE" \
                --decay "$DECAY" \
                --wf \
                --wf-splits "$WF_SPLITS"
        else
            python scripts/train_model.py \
                --symbol "$SYM" \
                --decay "$DECAY" \
                --wf \
                --wf-splits "$WF_SPLITS"
        fi
    fi

    # === [3/3] 배포 ===
    echo ""
    echo "=== [$SYM] [3/3] LXC 배포 ==="
    bash scripts/deploy_model.sh "$BACKEND" --symbol "$SYM"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$SYM] 파이프라인 완료"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
done

echo ""
echo "=== 전체 파이프라인 완료: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo ""
echo "봇 재시작이 필요하면:"
echo "  ssh root@10.1.10.24 'cd /root/cointrader && docker compose restart cointrader'"
