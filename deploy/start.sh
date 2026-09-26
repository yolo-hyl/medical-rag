#!/usr/bin/env bash
# =============================================================================
# 启动项目依赖的基础服务：Milvus / PostgreSQL / Redis / Neo4j
#
# 用法：
#   ./start.sh                 拉取镜像并启动全部服务，等待健康检查通过
#   ./start.sh --no-pull       跳过拉取（离线或镜像已存在时使用）
#   ./start.sh milvus redis    只启动指定服务
#
# 运行项目代码之前请先执行本脚本，脚本以 0 退出即表示所有服务可用。
# =============================================================================
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPLOY_DIR"

ALL_SERVICES=(milvus postgres redis neo4j)
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"   # 等待健康检查的最长秒数

PULL=1
SERVICES=()
for arg in "$@"; do
    case "$arg" in
        --no-pull) PULL=0 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        -*) echo "未知参数: $arg" >&2; exit 2 ;;
        *) SERVICES+=("$arg") ;;
    esac
done
[ ${#SERVICES[@]} -eq 0 ] && SERVICES=("${ALL_SERVICES[@]}")

info()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[deploy]\033[0m %s\n' "$*" >&2; }
fail()  { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. 检查 docker 环境
# -----------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "未找到 docker，请先安装 Docker Engine"
docker compose version >/dev/null 2>&1 || fail "未找到 docker compose（v2），请安装 docker-compose-plugin"
docker info >/dev/null 2>&1 || fail "无法连接 Docker daemon，请确认 docker 服务已启动且当前用户有权限"

# -----------------------------------------------------------------------------
# 2. 准备 .env（首次运行时生成随机密码）
# -----------------------------------------------------------------------------
gen_secret() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 16
    else
        head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}

if [ ! -f .env ]; then
    info "未找到 .env，从 .env.example 生成（含随机密码）"
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" == *__GENERATE__* ]] && line="${line//__GENERATE__/$(gen_secret)}"
        printf '%s\n' "$line"
    done < .env.example > .env
    chmod 600 .env
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a
DATA_DIR="${DATA_DIR:-./volumes}"
case "$DATA_DIR" in /*) DATA_PATH="$DATA_DIR" ;; *) DATA_PATH="$DEPLOY_DIR/${DATA_DIR#./}" ;; esac

# -----------------------------------------------------------------------------
# 3. 旧部署方式的兼容提示
# -----------------------------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx 'milvus-standalone'; then
    fail "检测到旧的 milvus-standalone 容器正在运行（会占用 19530 端口），请先执行: docker stop milvus-standalone"
fi
if [ -d ../Milvus/volumes/milvus ] && [ ! -e "$DATA_DIR/milvus" ]; then
    warn "检测到旧目录 Milvus/volumes/milvus 中的数据。如需沿用，请在首次启动前执行："
    warn "  mkdir -p $DATA_DIR && mv ../Milvus/volumes/milvus $DATA_DIR/milvus"
    fail "为避免在新目录下建出空库、误以为数据丢失，本次启动已中止；若不需要旧数据，删除 ../Milvus/volumes 后重试"
fi

# -----------------------------------------------------------------------------
# 4. 创建数据目录
# -----------------------------------------------------------------------------
mkdir -p "$DATA_DIR"/{milvus,postgres,redis} "$DATA_DIR"/neo4j/{data,logs,import,plugins}

# -----------------------------------------------------------------------------
# 5. 拉取镜像并启动
# -----------------------------------------------------------------------------
if [ "$PULL" -eq 1 ]; then
    info "拉取镜像（仓库: ${REGISTRY:-docker.io}）: ${SERVICES[*]}"
    pull_args=(); [ -t 1 ] || pull_args=(--quiet)   # 非交互终端下不刷进度条
    if ! docker compose pull "${pull_args[@]+"${pull_args[@]}"}" "${SERVICES[@]}"; then
        fail "镜像拉取失败。若 Docker Hub 限流(429)或网络不通，可在 deploy/.env 中设置 REGISTRY=mirror.gcr.io 或 REGISTRY=docker.m.daocloud.io 后重试"
    fi
fi

info "启动容器: ${SERVICES[*]}"
docker compose up -d "${SERVICES[@]}"

# -----------------------------------------------------------------------------
# 6. 等待健康检查通过
# -----------------------------------------------------------------------------
health_of() {
    local cid
    cid="$(docker compose ps -q "$1")"
    [ -z "$cid" ] && { echo "missing"; return; }
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid"
}

info "等待服务就绪（最长 ${WAIT_TIMEOUT}s）..."
deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
pending=("${SERVICES[@]}")
while [ ${#pending[@]} -gt 0 ]; do
    still=()
    for svc in "${pending[@]}"; do
        status="$(health_of "$svc")"
        case "$status" in
            healthy) info "  ✔ $svc 已就绪" ;;
            unhealthy|exited|dead|missing)
                docker compose logs --tail 50 "$svc" >&2 || true
                fail "$svc 启动失败（状态: $status），日志见上方" ;;
            *) still+=("$svc") ;;
        esac
    done
    pending=("${still[@]+"${still[@]}"}")
    [ ${#pending[@]} -eq 0 ] && break
    if [ "$(date +%s)" -ge "$deadline" ]; then
        for svc in "${pending[@]}"; do docker compose logs --tail 50 "$svc" >&2 || true; done
        fail "等待超时，仍未就绪: ${pending[*]}"
    fi
    sleep 3
done

# -----------------------------------------------------------------------------
# 7. 汇总
# -----------------------------------------------------------------------------
host="${BIND_ADDR:-127.0.0.1}"; [ "$host" = "0.0.0.0" ] && host="localhost"
info "全部服务已就绪："
for svc in "${SERVICES[@]}"; do
    case "$svc" in
        milvus)   echo "  Milvus      http://$host:${MILVUS_PORT:-19530}" ;;
        postgres) echo "  PostgreSQL  postgresql://${POSTGRES_USER:-medrag}@$host:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-medrag}" ;;
        redis)    echo "  Redis       redis://$host:${REDIS_PORT:-6379}" ;;
        neo4j)    echo "  Neo4j       bolt://$host:${NEO4J_BOLT_PORT:-7687}  (浏览器: http://$host:${NEO4J_HTTP_PORT:-7474})" ;;
    esac
done
echo "  密码见 deploy/.env；数据目录: $DATA_PATH"
