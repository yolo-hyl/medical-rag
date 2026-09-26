"""
基础服务连通性检查：Milvus / PostgreSQL / Redis / Neo4j

每项检查都会做一次真实的读写或查询，而不仅是端口探测。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, List

from ..config.models import AppConfig

START_HINT = "请先执行 deploy/start.sh 启动基础服务"


@dataclass
class ServiceStatus:
    name: str
    ok: bool
    detail: str
    latency_ms: float


def check_milvus(config: AppConfig) -> str:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.milvus.uri, token=config.milvus.token or "", timeout=5)
    try:
        version = client.get_server_version()
        collections = client.list_collections()
        return f"server {version}，{len(collections)} 个 collection"
    finally:
        client.close()


def check_postgres(config: AppConfig) -> str:
    import psycopg

    with psycopg.connect(config.postgres.dsn(), connect_timeout=5) as conn:
        version = conn.execute("SHOW server_version").fetchone()[0]
    return f"server {version}"


def check_redis(config: AppConfig) -> str:
    import redis

    client = redis.Redis.from_url(config.redis.url(), socket_timeout=5, socket_connect_timeout=5)
    try:
        key = f"medrag:healthcheck:{uuid.uuid4().hex}"
        client.set(key, "1", ex=10)
        if client.get(key) != b"1":
            raise RuntimeError("写入后读取结果不一致")
        client.delete(key)
        return f"server {client.info('server')['redis_version']}"
    finally:
        client.close()


def check_neo4j(config: AppConfig) -> str:
    from neo4j import GraphDatabase

    auth = (config.neo4j.user, config.neo4j.password())
    with GraphDatabase.driver(config.neo4j.uri, auth=auth, connection_timeout=5) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "CALL dbms.components() YIELD name, versions, edition RETURN versions[0] AS v, edition AS e"
        )
        return f"server {records[0]['v']} ({records[0]['e']})"


CHECKS: dict[str, Callable[[AppConfig], str]] = {
    "milvus": check_milvus,
    "postgres": check_postgres,
    "redis": check_redis,
    "neo4j": check_neo4j,
}


def check_services(config: AppConfig, names: List[str] | None = None) -> List[ServiceStatus]:
    """依次检查各服务，返回每项的结果（不抛异常）"""
    results = []
    for name in names or list(CHECKS):
        start = time.perf_counter()
        try:
            detail, ok = CHECKS[name](config), True
        except Exception as e:  # 汇总所有失败原因，交给调用方决定如何处理
            detail, ok = f"{type(e).__name__}: {e}", False
        results.append(ServiceStatus(name, ok, detail, (time.perf_counter() - start) * 1000))
    return results
