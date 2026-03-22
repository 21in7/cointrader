#!/bin/sh
# 15분 경계에 맞춰 collect_ls_ratio.py를 반복 실행한다.
# Docker 컨테이너 entrypoint용.

set -e

echo "[collect_ls_ratio] Starting loop (interval: 15m)"

while true; do
    # 현재 분/초를 기준으로 다음 15분 경계(00/15/30/45)까지 대기
    now_min=$(date -u +%M | sed 's/^0//')
    now_sec=$(date -u +%S | sed 's/^0//')
    # 다음 15분 경계까지 남은 분
    remainder=$((now_min % 15))
    wait_min=$((15 - remainder))
    # 초 단위로 변환 (경계 직후 10초 여유)
    wait_sec=$(( wait_min * 60 - now_sec + 10 ))
    if [ "$wait_sec" -le 10 ]; then
        wait_sec=$((wait_sec + 900))
    fi

    echo "[collect_ls_ratio] Next run in ${wait_sec}s ($(date -u))"
    sleep "$wait_sec"

    echo "[collect_ls_ratio] Running collection... ($(date -u))"
    python scripts/collect_ls_ratio.py || echo "[collect_ls_ratio] ERROR: collection failed"
done
