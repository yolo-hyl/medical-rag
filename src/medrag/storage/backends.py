"""服务地址没有配置时，降级为嵌入式实现

| 服务 | 配置了地址 | 没有配置地址 |
| --- | --- | --- |
| Milvus（milvus.uri） | Milvus 服务 | Milvus Lite：本地文件，由 pymilvus 在本进程里拉起 |
| Redis（redis.host） | Redis 服务 | fakeredis：进程内存，支持 Lua（会话锁要用） |
| PostgreSQL（postgres.host） | PostgreSQL | SQLite：本地文件 |
| Neo4j（neo4j.uri） | Neo4j | FalkorDBLite：本地文件，兼容 Cypher |

降级实现的边界：
- 只适合单进程的开发、测试和小规模使用：fakeredis 的数据不跨进程、进程退出即丢失；
  Milvus Lite / SQLite / FalkorDBLite 的数据文件同一时间只应被一个进程打开。
- 功能上搬家时用到的 Milvus 操作（upsert / flush / HNSW / 托管 BM25 / 混合检索 / 删除）在
  milvus-lite 3.x 上都验证过；Milvus Lite 不支持 Windows。
- 第一次降级时打一条 warning，避免生产环境因为漏配地址而悄悄跑在嵌入式实现上。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config.models import MilvusConfig, Neo4jConfig, PostgresConfig, RedisConfig

logger = logging.getLogger(__name__)

_warned: set[str] = set()
_fake_server = None   # 进程内唯一的 fakeredis 服务端，懒创建


def _warn_once(service: str, message: str) -> None:
    if service not in _warned:
        _warned.add(service)
        logger.warning(message)


def _local_file(path: str) -> str:
    """确保数据文件所在目录存在，返回绝对路径（避免进程切换工作目录后指向别处）"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------
class _LiteAllocTimestampFilter(logging.Filter):
    """去掉 Milvus Lite 对 AllocTimestamp 的报错日志。

    pymilvus 2.6 的客户端会先调用 AllocTimestamp，Milvus Lite 3.x 没有实现这个接口，它在本进程内运行的
    gRPC 服务会记一条带完整调用栈的 ERROR；客户端随后自动退回其他方式，入库与检索都不受影响（实测）。
    只过滤这一种报错，Milvus Lite 的其他错误照常输出。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, NotImplementedError) and record.exc_info[2] is not None:
            import traceback
            return not any(f.name == "AllocTimestamp" for f in traceback.extract_tb(record.exc_info[2]))
        return True


def milvus_connection(cfg: MilvusConfig) -> dict:
    """给 MilvusClient / AsyncMilvusClient 的连接参数"""
    if not cfg.is_lite:
        return {"uri": cfg.uri, "token": cfg.token or ""}
    if "milvus" not in _warned:
        logging.getLogger("grpc._server").addFilter(_LiteAllocTimestampFilter())
    path = _local_file(cfg.lite_path)
    _warn_once("milvus", f"未配置 milvus.uri，降级为 Milvus Lite（{path}），仅适合单进程开发与测试")
    return {"uri": path}


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
def create_redis(cfg: RedisConfig) -> Any:
    """返回 redis.asyncio.Redis 或接口相同的 fakeredis 客户端（decode_responses=True）"""
    if not cfg.is_fake:
        import redis.asyncio as aioredis
        # 连接池是懒创建的，第一次使用时绑定到当前事件循环
        return aioredis.Redis.from_url(cfg.url(), decode_responses=True)
    import fakeredis

    global _fake_server
    if _fake_server is None:
        _fake_server = fakeredis.FakeServer()
    _warn_once("redis", "未配置 redis.host，降级为 fakeredis（进程内存，不跨进程、进程退出即丢失）")
    # fakeredis 的实例默认各自一份数据；共用一个 FakeServer，同一进程里的多个客户端才像连着同一个 Redis
    return fakeredis.FakeAsyncRedis(server=_fake_server, decode_responses=True)


# ---------------------------------------------------------------------------
# PostgreSQL / SQLite
# ---------------------------------------------------------------------------
def sqlite_path(cfg: PostgresConfig) -> str:
    path = _local_file(cfg.sqlite_path)
    _warn_once("postgres", f"未配置 postgres.host，降级为 SQLite（{path}），仅适合单进程开发与测试")
    return path


# ---------------------------------------------------------------------------
# Neo4j / FalkorDBLite
# ---------------------------------------------------------------------------
def falkordb_path(cfg: Neo4jConfig) -> str:
    path = _local_file(cfg.falkordb_path)
    _warn_once("neo4j", f"未配置 neo4j.uri，降级为 FalkorDBLite（{path}），仅适合单进程开发与测试")
    return path
