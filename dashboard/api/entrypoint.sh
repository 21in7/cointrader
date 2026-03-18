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

# FastAPI 서버 실행
echo "Starting API server on :8080"
exec uvicorn dashboard_api:app --host 0.0.0.0 --port 8080 --log-level info
