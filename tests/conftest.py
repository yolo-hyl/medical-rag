"""测试约定

- 不带标记的测试不依赖任何外部服务：模型请求走本地的假 OpenAI 服务（medrag.testing.fake_openai），
  存储走未配置地址时的降级实现（Milvus Lite / fakeredis / SQLite / FalkorDBLite），数据文件放在 pytest 的临时目录。
- @pytest.mark.services(...) 的测试连接 deploy/start.sh 启动的真实服务，参数是需要的服务名
  （milvus / redis / postgres / neo4j，不写时默认 milvus + redis）；服务不可用时自动跳过，
  设置 MEDRAG_REQUIRE_SERVICES=1 则改为失败（验收时用）。
"""
import os
import socket
from pathlib import Path

import pytest

from medrag.config import AppConfig, ConfigLoader
from medrag.testing.fake_openai import FakeOpenAIServer

REPO = Path(__file__).resolve().parents[1]
SERVICES = {
    "milvus": ("127.0.0.1", 19530),
    "redis": ("127.0.0.1", 6379),
    "postgres": ("127.0.0.1", 5432),
    "neo4j": ("127.0.0.1", 7687),
}
DEFAULT_SERVICES = ("milvus", "redis")


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "services(*names): 需要 deploy/start.sh 启动的真实服务")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MEDRAG_REQUIRE_SERVICES"):
        return
    status = {}
    for item in items:
        marker = item.get_closest_marker("services")
        if marker is None:
            continue
        needed = marker.args or DEFAULT_SERVICES
        down = [n for n in needed if not status.setdefault(n, _reachable(*SERVICES[n]))]
        if down:
            item.add_marker(pytest.mark.skip(reason=f"基础服务不可用（{', '.join(down)}），先执行 deploy/start.sh"))


@pytest.fixture(scope="session")
def fake_openai():
    with FakeOpenAIServer() as server:
        yield server


@pytest.fixture(scope="session")
def service_config() -> AppConfig:
    """连接 deploy/start.sh 启动的服务：包内默认配置 + deploy/storage.yaml"""
    return ConfigLoader(overrides=[str(REPO / "deploy" / "storage.yaml")]).config


@pytest.fixture
def lite_config(tmp_path) -> AppConfig:
    """不配置任何服务地址，四个服务都走降级实现；数据文件放在本用例的临时目录"""
    return AppConfig.model_validate({
        "milvus": {"lite_path": str(tmp_path / "milvus_lite.db"), "drop_old": True, "auto_id": False},
        "postgres": {"sqlite_path": str(tmp_path / "medrag.sqlite3")},
        "neo4j": {"falkordb_path": str(tmp_path / "falkordb.db")},
    })
