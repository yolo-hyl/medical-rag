"""
Redis 会话存储测试：需要先执行 deploy/start.sh
    pytest tests/test_session_store.py -v

没有引入 pytest-asyncio，统一用 asyncio.run 跑协程。
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.session_store import RedisSessionStore

SESSION_ID = "pytest-session"
HISTORY_KEY = f"medrag:chat:{SESSION_ID}:history"
JSON_KEY = f"medrag:chat:{SESSION_ID}:tokens"


def run_with_store(coro_fn):
    """新建 store → 清干净 → 执行 → 清理并关闭（连接池绑定在本次的事件循环上）"""
    async def main():
        store = RedisSessionStore(ConfigLoader().config.redis)
        try:
            await store.r.delete(HISTORY_KEY, JSON_KEY)
            return await coro_fn(store)
        finally:
            await store.r.delete(HISTORY_KEY, JSON_KEY)
            await store.close()

    return asyncio.run(main())


def test_history_roundtrip():
    async def case(store):
        hist = store.history(HISTORY_KEY)
        await hist.aadd_messages([HumanMessage(content="你好"), AIMessage(content="您好")])
        return await hist.aget_messages()

    messages = run_with_store(case)
    assert [m.content for m in messages] == ["你好", "您好"]
    assert isinstance(messages[0], HumanMessage) and isinstance(messages[1], AIMessage)


def test_history_trim_keeps_tail():
    async def case(store):
        hist = store.history(HISTORY_KEY)
        await hist.aadd_messages([HumanMessage(content=f"第{i}条") for i in range(4)])
        await hist.atrim(2)
        return [m.content for m in await hist.aget_messages()]

    assert run_with_store(case) == ["第2条", "第3条"]


def test_keys_have_ttl():
    async def case(store):
        await store.history(HISTORY_KEY).aadd_messages([HumanMessage(content="你好")])
        await store.set_json(JSON_KEY, {"msg_len": [10]})
        return await store.r.ttl(HISTORY_KEY), await store.r.ttl(JSON_KEY), store.ttl

    history_ttl, json_ttl, ttl = run_with_store(case)
    assert 0 < history_ttl <= ttl
    assert 0 < json_ttl <= ttl


def test_json_default_when_missing():
    async def case(store):
        missing = await store.get_json("medrag:chat:not-exist:tokens")
        default = await store.get_json("medrag:chat:not-exist:summary", "")
        await store.set_json(JSON_KEY, {"msg_len": [1, 2]})
        return missing, default, await store.get_json(JSON_KEY)

    missing, default, saved = run_with_store(case)
    assert missing is None
    assert default == ""
    assert saved == {"msg_len": [1, 2]}


def test_lock_serializes_same_session():
    async def case(store):
        order = []

        async def worker(name: str):
            async with store.lock(SESSION_ID):
                order.append(f"{name}-进入")
                await asyncio.sleep(0.05)
                order.append(f"{name}-离开")

        await asyncio.gather(worker("A"), worker("B"))
        return order

    order = run_with_store(case)
    # 两个协程不会交错：一个完整结束后另一个才进入
    assert order[0].endswith("进入") and order[1].endswith("离开")
    assert order[0][0] == order[1][0]
