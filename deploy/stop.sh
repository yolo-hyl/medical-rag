#!/usr/bin/env bash
# =============================================================================
# 停止基础服务
#
# 用法：
#   ./stop.sh            停止容器（保留容器与数据）
#   ./stop.sh --down     停止并删除容器（保留数据目录）
#   ./stop.sh --purge    停止并删除容器，同时删除全部数据（需确认）
# =============================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
DATA_DIR="${DATA_DIR:-./volumes}"
case "$DATA_DIR" in /*) DATA_PATH="$DATA_DIR" ;; *) DATA_PATH="$DEPLOY_DIR/${DATA_DIR#./}" ;; esac

# 容器已删除且没有 .env 时，compose 仍需解析必填变量，这里给占位值
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-unused}" REDIS_PASSWORD="${REDIS_PASSWORD:-unused}" NEO4J_PASSWORD="${NEO4J_PASSWORD:-unused}"

case "${1:-}" in
    "")
        docker compose stop
        echo "[deploy] 已停止（数据保留）" ;;
    --down)
        docker compose down
        echo "[deploy] 已删除容器（数据保留在 $DATA_PATH）" ;;
    --purge)
        read -r -p "将删除所有容器以及 $DATA_PATH 下的全部数据，确认请输入 y: " ok
        if [ "$ok" = "y" ] || [ "$ok" = "Y" ]; then
            docker compose down
            # 数据文件由容器内的不同用户创建，借助容器以 root 身份删除，避免宿主机权限不足
            docker run --rm -v "$DATA_PATH:/target" \
                "${REGISTRY:-docker.io}/library/redis:${REDIS_VERSION:-7.4-alpine}" \
                sh -c 'rm -rf /target/* /target/.[!.]* 2>/dev/null || true'
            echo "[deploy] 已删除容器和数据"
        else
            echo "[deploy] 已取消"
        fi ;;
    -h|--help)
        sed -n '2,8p' "$0" ;;
    *)
        echo "未知参数: $1（可选：--down / --purge）" >&2; exit 2 ;;
esac
