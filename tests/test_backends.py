"""未配置服务地址时的降级：选择逻辑，以及 SQL / 图数据库在两种后端上的行为一致"""
import asyncio
import logging

import pytest

from medrag.config import AppConfig, ConfigLoader
from medrag.storage import backends
from medrag.storage.graph import GraphDatabase
from medrag.storage.sql import SqlDatabase


def test_package_defaults_degrade_everything():
    cfg = ConfigLoader().config
    assert cfg.milvus.is_lite and cfg.redis.is_fake and cfg.postgres.is_sqlite and cfg.neo4j.is_lite


def test_deploy_override_uses_services(service_config):
    cfg = service_config
    assert cfg.milvus.uri == "http://localhost:19530"
    assert not (cfg.milvus.is_lite or cfg.redis.is_fake or cfg.postgres.is_sqlite or cfg.neo4j.is_lite)


def test_env_overrides(monkeypatch):
    from conftest import REPO
    monkeypatch.setenv("MEDRAG_CONFIG_OVERRIDES", "deploy/storage.yaml")
    monkeypatch.chdir(REPO / "tests")       # 相对路径在当前目录找不到时按仓库根目录解析
    assert ConfigLoader().config.redis.host == "127.0.0.1"


def test_milvus_connection_and_warning(lite_config, caplog, monkeypatch):
    monkeypatch.setattr(backends, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="medrag.storage.backends"):
        conn = backends.milvus_connection(lite_config.milvus)
        backends.milvus_connection(lite_config.milvus)
    assert conn == {"uri": lite_config.milvus.lite_path}
    assert sum("Milvus Lite" in r.message for r in caplog.records) == 1   # 只提醒一次
    assert backends.milvus_connection(lite_config.milvus.model_copy(update={"uri": "http://m:19530"}))["uri"] == "http://m:19530"


def test_fakeredis_clients_share_one_server():
    async def run():
        cfg = AppConfig().redis
        a, b = backends.create_redis(cfg), backends.create_redis(cfg)
        await a.set("medrag:test:shared", "1")
        try:
            return await b.get("medrag:test:shared")
        finally:
            await a.delete("medrag:test:shared")

    assert asyncio.run(run()) == "1"


# ---------------------------------------------------------------------------
# SQL：SQLite 与 PostgreSQL 跑同一组用例
# ---------------------------------------------------------------------------
async def _sql_roundtrip(db: SqlDatabase):
    await db.execute("DROP TABLE IF EXISTS medrag_test_doc")
    await db.execute("CREATE TABLE medrag_test_doc (doc_id TEXT PRIMARY KEY, state TEXT, n INTEGER)")
    await db.executemany("INSERT INTO medrag_test_doc VALUES (?, ?, ?)", [("d1", "pending", 1), ("d2", "done", 2)])
    await db.execute(
        "INSERT INTO medrag_test_doc VALUES (?, ?, ?) "
        "ON CONFLICT (doc_id) DO UPDATE SET state = excluded.state, n = excluded.n",
        ("d1", "done", 3),
    )
    rows = await db.fetchall("SELECT doc_id, state, n FROM medrag_test_doc WHERE state = ? ORDER BY doc_id", ("done",))
    affected = await db.execute("DELETE FROM medrag_test_doc WHERE doc_id = ?", ("d2",))
    await db.execute("DROP TABLE medrag_test_doc")
    await db.close()
    return rows, affected


EXPECTED_ROWS = [{"doc_id": "d1", "state": "done", "n": 3}, {"doc_id": "d2", "state": "done", "n": 2}]


def test_sqlite_fallback(lite_config):
    db = SqlDatabase(lite_config.postgres)
    assert db.backend == "sqlite"
    assert asyncio.run(_sql_roundtrip(db)) == (EXPECTED_ROWS, 1)


@pytest.mark.services("postgres")
def test_postgres(service_config):
    db = SqlDatabase(service_config.postgres)
    assert db.backend == "postgres"
    assert asyncio.run(_sql_roundtrip(db)) == (EXPECTED_ROWS, 1)


# ---------------------------------------------------------------------------
# 图：FalkorDBLite 与 Neo4j 跑同一组用例
# ---------------------------------------------------------------------------
async def _graph_roundtrip(graph: GraphDatabase):
    await graph.query("MATCH (n:MedragTest) DETACH DELETE n")
    await graph.query(
        "MERGE (d:MedragTest {name: $d}) MERGE (s:MedragTest {name: $s}) MERGE (d)-[:HAS_SYMPTOM]->(s)",
        {"d": "高血压", "s": "头痛"},
    )
    rows = await graph.query(
        "MATCH (d:MedragTest)-[r:HAS_SYMPTOM]->(s) RETURN d.name AS disease, type(r) AS rel, s AS symptom"
    )
    await graph.query("MATCH (n:MedragTest) DETACH DELETE n")
    await graph.close()
    return rows


EXPECTED_GRAPH = [{"disease": "高血压", "rel": "HAS_SYMPTOM", "symptom": {"name": "头痛"}}]


def test_falkordb_fallback(lite_config):
    graph = GraphDatabase(lite_config.neo4j)
    assert graph.backend == "falkordb"
    assert asyncio.run(_graph_roundtrip(graph)) == EXPECTED_GRAPH


@pytest.mark.services("neo4j")
def test_neo4j(service_config):
    graph = GraphDatabase(service_config.neo4j)
    assert graph.backend == "neo4j"
    assert asyncio.run(_graph_roundtrip(graph)) == EXPECTED_GRAPH


# ---------------------------------------------------------------------------
# 降级实现关闭后重新打开，数据还在（redislite 的异步 close 默认不落盘，GraphDatabase.close 里补了 SAVE）
# ---------------------------------------------------------------------------
def test_lite_backends_persist_after_close(lite_config):
    async def write():
        graph = GraphDatabase(lite_config.neo4j)
        await graph.query("CREATE (:MedragPersist {name: 'x'})")
        await graph.close()
        db = SqlDatabase(lite_config.postgres)
        await db.execute("CREATE TABLE persist_test (k TEXT)")
        await db.execute("INSERT INTO persist_test VALUES (?)", ("v",))
        await db.close()

    async def read():
        graph = GraphDatabase(lite_config.neo4j)
        nodes = await graph.query("MATCH (n:MedragPersist) RETURN count(n) AS c")
        await graph.close()
        db = SqlDatabase(lite_config.postgres)
        rows = await db.fetchall("SELECT k FROM persist_test")
        await db.close()
        return nodes, rows

    asyncio.run(write())
    assert asyncio.run(read()) == ([{"c": 1}], [{"k": "v"}])

    from pymilvus import MilvusClient
    client = MilvusClient(**backends.milvus_connection(lite_config.milvus))
    client.create_collection("persist_test", dimension=4)
    client.insert("persist_test", [{"id": i, "vector": [float(i)] * 4} for i in range(3)])
    client.flush("persist_test")
    client.close()
    client = MilvusClient(**backends.milvus_connection(lite_config.milvus))
    assert client.query("persist_test", filter="", output_fields=["count(*)"])[0]["count(*)"] == 3
    client.close()
