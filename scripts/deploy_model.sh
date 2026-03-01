#!/usr/bin/env bash
# 맥미니에서 학습한 모델을 LXC 컨테이너 볼륨 경로로 전송한다.
# 사용법: bash scripts/deploy_model.sh [LXC_HOST] [LXC_MODELS_PATH]
#
# 예시:
#   bash scripts/deploy_model.sh 10.1.10.28 /path/to/cointrader/models
#   bash scripts/deploy_model.sh root@10.1.10.28 /root/cointrader/models

set -euo pipefail

LXC_HOST="${1:-root@10.1.10.24}"
LXC_MODELS_PATH="${2:-/root/cointrader/models}"
LOCAL_MODEL="models/lgbm_filter.pkl"
LOCAL_LOG="models/training_log.json"

if [[ ! -f "$LOCAL_MODEL" ]]; then
  echo "[오류] 모델 파일 없음: $LOCAL_MODEL"
  echo "먼저 python scripts/train_model.py 를 실행하세요."
  exit 1
fi

echo "=== 모델 전송 시작 ==="
echo "  대상: ${LXC_HOST}:${LXC_MODELS_PATH}"
echo "  파일: $LOCAL_MODEL"

# 기존 모델을 prev로 백업 (원격)
ssh "${LXC_HOST}" "
  if [ -f '${LXC_MODELS_PATH}/lgbm_filter.pkl' ]; then
    cp '${LXC_MODELS_PATH}/lgbm_filter.pkl' '${LXC_MODELS_PATH}/lgbm_filter_prev.pkl'
    echo '  기존 모델 백업 완료'
  fi
  mkdir -p '${LXC_MODELS_PATH}'
"

# 모델 파일 전송 (rsync 우선, 없으면 scp 폴백)
if command -v rsync &>/dev/null && ssh "${LXC_HOST}" "command -v rsync" &>/dev/null; then
  rsync -avz --progress \
    "$LOCAL_MODEL" \
    "${LXC_HOST}:${LXC_MODELS_PATH}/lgbm_filter.pkl"
else
  echo "  rsync 없음 → scp 사용"
  scp "$LOCAL_MODEL" "${LXC_HOST}:${LXC_MODELS_PATH}/lgbm_filter.pkl"
fi

# 학습 로그도 함께 전송 (있을 경우)
if [[ -f "$LOCAL_LOG" ]]; then
  if command -v rsync &>/dev/null && ssh "${LXC_HOST}" "command -v rsync" &>/dev/null; then
    rsync -avz "$LOCAL_LOG" "${LXC_HOST}:${LXC_MODELS_PATH}/training_log.json"
  else
    scp "$LOCAL_LOG" "${LXC_HOST}:${LXC_MODELS_PATH}/training_log.json"
  fi
  echo "  학습 로그 전송 완료"
fi

echo "=== 전송 완료 ==="
echo ""

# 봇 컨테이너가 실행 중이면 모델 핫리로드, 아니면 건너뜀
echo "=== 핫리로드 시도 ==="
if ssh "${LXC_HOST}" "docker inspect -f '{{.State.Running}}' cointrader 2>/dev/null | grep -q true"; then
  ssh "${LXC_HOST}" "docker exec cointrader python -c \
    \"from src.ml_filter import MLFilter; f=MLFilter(); f.reload_model(); print('리로드 완료')\""
  echo "=== 핫리로드 완료 ==="
else
  echo "  cointrader 컨테이너가 실행 중이 아닙니다. 건너뜁니다."
fi
