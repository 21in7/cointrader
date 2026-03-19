#!/bin/bash
set -e

echo "=== Trading Dashboard ==="
echo "LOG_DIR=${LOG_DIR:-/app/logs}"
echo "DB_PATH=${DB_PATH:-/app/data/dashboard.db}"

# 로그 파서를 백그라운드로 실행
python -u log_parser.py &
PARSER_PID=$!
echo "Log parser started (PID: $PARSER_PID)"

# 파서가 기존 로그를 처리할 시간 부여
sleep 3

# SIGTERM/SIGINT → 파서에도 전달 후 대기
cleanup() {
    echo "Shutting down..."
    kill -TERM "$PARSER_PID" 2>/dev/null
    wait "$PARSER_PID" 2>/dev/null
    kill -TERM "$UVICORN_PID" 2>/dev/null
    wait "$UVICORN_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

# FastAPI 서버를 백그라운드로 실행 (exec 대신 — 셸이 PID 1을 유지해야 signal forwarding 가능)
echo "Starting API server on :8080"
uvicorn dashboard_api:app --host 0.0.0.0 --port 8080 --log-level info &
UVICORN_PID=$!

# 자식 프로세스 중 하나라도 종료되면 전체 종료
wait -n "$PARSER_PID" "$UVICORN_PID" 2>/dev/null
cleanup
