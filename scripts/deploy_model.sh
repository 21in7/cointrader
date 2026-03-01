#!/usr/bin/env bash
# 맥미니에서 학습한 모델을 LXC 컨테이너 볼륨 경로로 전송한다.
# 사용법: bash scripts/deploy_model.sh [lgbm|mlx]
#
# 예시:
#   bash scripts/deploy_model.sh        # LightGBM (기본값)
#   bash scripts/deploy_model.sh mlx    # MLX 신경망

set -euo pipefail

BACKEND="${1:-lgbm}"
LXC_HOST="root@10.1.10.24"
LXC_MODELS_PATH="/root/cointrader/models"
LOCAL_LOG="models/training_log.json"

# ── 백엔드별 파일 목록 설정 ──────────────────────────────────────────────────
# mlx: ONNX 파일만 전송 (Linux 서버는 onnxruntime으로 추론)
# lgbm: pkl 파일 전송
if [ "$BACKEND" = "mlx" ]; then
  LOCAL_FILES=("models/mlx_filter.weights.onnx")
else
  LOCAL_FILES=("models/lgbm_filter.pkl")
fi

# ── 파일 존재 확인 ────────────────────────────────────────────────────────────
for f in "${LOCAL_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[오류] 모델 파일 없음: $f"
    exit 1
  fi
done

echo "=== 모델 전송 시작 (백엔드: ${BACKEND}) ==="
echo "  대상: ${LXC_HOST}:${LXC_MODELS_PATH}"

# ── 원격 디렉터리 생성 + lgbm 기존 모델 백업 ─────────────────────────────────
ssh "${LXC_HOST}" "
  mkdir -p '${LXC_MODELS_PATH}'
  if [ '$BACKEND' = 'lgbm' ] && [ -f '${LXC_MODELS_PATH}/lgbm_filter.pkl' ]; then
    cp '${LXC_MODELS_PATH}/lgbm_filter.pkl' '${LXC_MODELS_PATH}/lgbm_filter_prev.pkl'
    echo '  기존 lgbm 모델 백업 완료'
  fi
"

# ── 파일 전송 헬퍼 (rsync 우선, scp 폴백) ────────────────────────────────────
_send() {
  local src="$1" dst="$2"
  echo "  전송: $src → ${LXC_HOST}:$dst"
  if command -v rsync &>/dev/null && ssh "${LXC_HOST}" "command -v rsync" &>/dev/null; then
    rsync -avz --progress "$src" "${LXC_HOST}:$dst"
  else
    scp "$src" "${LXC_HOST}:$dst"
  fi
}

# ── 모델 파일 전송 ────────────────────────────────────────────────────────────
for f in "${LOCAL_FILES[@]}"; do
  _send "$f" "${LXC_MODELS_PATH}/$(basename "$f")"
done

# ── 학습 로그 전송 ────────────────────────────────────────────────────────────
if [[ -f "$LOCAL_LOG" ]]; then
  _send "$LOCAL_LOG" "${LXC_MODELS_PATH}/training_log.json"
  echo "  학습 로그 전송 완료"
fi

echo "=== 전송 완료 ==="
echo ""

# ── 핫리로드 안내 ────────────────────────────────────────────────────────────
# 봇이 캔들마다 모델 파일 mtime을 감지해 자동 리로드한다.
# 컨테이너가 실행 중이면 다음 캔들(최대 1분) 안에 자동 적용된다.
echo "=== 모델 전송 완료 — 봇이 다음 캔들에서 자동 리로드합니다 ==="
if ssh "${LXC_HOST}" "docker inspect -f '{{.State.Running}}' cointrader 2>/dev/null | grep -q true"; then
  echo "  컨테이너 실행 중: 다음 캔들 마감 시 자동 핫리로드 예정"
else
  echo "  cointrader 컨테이너가 실행 중이 아닙니다."
fi
