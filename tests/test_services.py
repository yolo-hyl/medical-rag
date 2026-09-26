"""
基础服务集成测试：需要先执行 deploy/start.sh
    pytest tests/test_services.py -v
"""
import pytest

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.config.models import PostgresConfig, RedisConfig, read_secret


@pytest.fixture(scope="module")
def config():
    return ConfigLoader().config


def test_deploy_env_loaded(config):
    # ConfigLoader 应自动加载 deploy/.env 中的密码
    for env_name in (config.postgres.password_env, config.redis.password_env, config.neo4j.password_env):
        assert read_secret(env_name)


def test_milvus_roundtrip(config):
    # 建临时 collection → 插入 → 检索 → 删除，确认向量库可读写
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.milvus.uri, token=config.milvus.token or "")
    name = "medrag_healthcheck_tmp"
    try:
        if client.has_collection(name):
            client.drop_collection(name)
        client.create_collection(name, dimension=4, metric_type="COSINE", consistency_level="Strong")
        client.insert(name, [{"id": i, "vector": [float(i == j) for j in range(4)]} for i in range(4)])
        hits = client.search(name, data=[[0.0, 0.0, 1.0, 0.0]], limit=1)
        assert hits[0][0]["id"] == 2
    finally:
        if client.has_collection(name):
            client.drop_collection(name)
        client.close()


def test_missing_secret_gives_hint(monkeypatch):
    monkeypatch.delenv("MEDRAG_TEST_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="deploy/start.sh"):
        PostgresConfig(password_env="MEDRAG_TEST_MISSING").dsn()


def test_special_chars_in_password_are_escaped(monkeypatch):
    monkeypatch.setenv("MEDRAG_TEST_PW", "p@ss:w/rd")
    url = RedisConfig(password_env="MEDRAG_TEST_PW").url()
    assert url == "redis://:p%40ss%3Aw%2Frd@127.0.0.1:6379/0"
