#!/bin/bash
# Cached Token Server 启动脚本
# 用法: bash start_cache_server.sh [port] [max_records]

PORT=${1:-18731}
MAX_RECORDS=${2:-10000}
HOST="127.0.0.1"
TOKEN_MODEL="cl100k_base"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/cache_server_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Cached Token Server..."
echo "  Host:        ${HOST}"
echo "  Port:        ${PORT}"
echo "  Max Records: ${MAX_RECORDS}"
echo "  Token Model: ${TOKEN_MODEL}"
echo "  Log:         ${LOG_FILE}"
echo ""

cd "${SCRIPT_DIR}"

nohup python -m src.module.cached_token_server_http \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-records "${MAX_RECORDS}" \
    --token-model "${TOKEN_MODEL}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!

sleep 2

if kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "Server started successfully (PID: ${SERVER_PID})"
    echo "${SERVER_PID}" > "${SCRIPT_DIR}/.cache_server_pid"

    HEALTH=$(curl -s "http://${HOST}:${PORT}/health" 2>/dev/null)
    if echo "${HEALTH}" | grep -q "healthy"; then
        echo "Health check passed: ${HEALTH}"
    else
        echo "Warning: health check did not return expected response"
        echo "  Response: ${HEALTH}"
        echo "  Check log: ${LOG_FILE}"
    fi

    echo ""
    echo "Usage:"
    echo "  Stop:    kill ${SERVER_PID}  or  bash start_cache_server.sh stop"
    echo "  Status:  curl http://${HOST}:${PORT}/status"
    echo "  Health:  curl http://${HOST}:${PORT}/health"
else
    echo "Failed to start server, check log: ${LOG_FILE}"
    exit 1
fi


#! # 默认启动（端口 18731，最大 10000 条记录）
# bash start_cache_server.sh
#! # 自定义端口和最大记录数
# bash start_cache_server.sh 18731 10000
#! # 查看状态
# curl http://127.0.0.1:18731/status
#! # 停止服务
# kill $(cat .cache_server_pid)
