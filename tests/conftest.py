"""测试约定

- 不带标记的测试不依赖任何外部服务：模型请求走本地的假 OpenAI 服务（medrag.testing.fake_openai）。
- @pytest.mark.services 的测试需要 deploy/start.sh 启动的 Milvus / Redis；服务不可用时自动跳过，
  设置 MEDRAG_REQUIRE_SERVICES=1 则改为失败（CI 或里程碑验收时用）。
"""
import os
import socket

import pytest

from medrag.testing.fake_openai import FakeOpenAIServer

SERVICES = {"milvus": ("127.0.0.1", 19530), "redis": ("127.0.0.1", 6379)}


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MEDRAG_REQUIRE_SERVICES"):
        return
    down = [name for name, addr in SERVICES.items() if not _reachable(*addr)]
    if not down:
        return
    skip = pytest.mark.skip(reason=f"基础服务不可用（{', '.join(down)}），先执行 deploy/start.sh")
    for item in items:
        if "services" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fake_openai():
    with FakeOpenAIServer() as server:
        yield server
