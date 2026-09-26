"""Redis 会话存储验证：需要 Redis（deploy/start.sh），不需要模型。

验证点：
1. 对话历史的读写与裁剪（LTRIM）；
2. key 带 TTL；
3. JSON 读写与缺省值；
4. 同一会话加锁串行，不同会话互不阻塞。
"""
import asyncio
import time

from common import clear_session, load_config

from langchain_core.messages import AIMessage, HumanMessage
from MedicalRag.core.session_store import RedisSessionStore

SID = "debug-store"
HISTORY = f"medrag:chat:{SID}:history"
TOKENS = f"medrag:chat:{SID}:tokens"


async def main():
    store = RedisSessionStore(load_config().redis)
    await clear_session(store, SID, "debug-store-2")

    try:
        # 1) 历史读写
        hist = store.history(HISTORY)
        await hist.aadd_messages([HumanMessage(content="你好"), AIMessage(content="您好")])
        await hist.aadd_messages([HumanMessage(content="第二轮"), AIMessage(content="好的")])
        msgs = await hist.aget_messages()
        print(f"1) 历史 {len(msgs)} 条: {[(type(m).__name__, m.content) for m in msgs]}")

        # 2) 裁剪（摘要压缩时丢弃旧消息）
        await hist.atrim(2)
        print(f"2) atrim(2) 后保留: {[m.content for m in await hist.aget_messages()]}")

        # 3) TTL 与 JSON
        await store.set_json(TOKENS, {"msg_len": [10], "msg_token_len": [4]})
        print(f"3) TTL: history={await store.r.ttl(HISTORY)}s tokens={await store.r.ttl(TOKENS)}s"
              f"（配置 {store.ttl}s）")
        print(f"   JSON 读回: {await store.get_json(TOKENS)}")
        print(f"   缺省值: {await store.get_json('medrag:chat:not-exist:summary', '(空)')!r}")

        # 4) 同一会话串行
        order = []

        async def worker(name, sid):
            async with store.lock(sid):
                order.append(f"{name}进")
                await asyncio.sleep(0.2)
                order.append(f"{name}出")

        t0 = time.time()
        await asyncio.gather(worker("A", SID), worker("B", SID))
        same = time.time() - t0
        print(f"4) 同一会话串行: {order}（耗时 {same:.2f}s，两个各 0.2s）")

        order.clear()
        t0 = time.time()
        await asyncio.gather(worker("A", SID), worker("B", "debug-store-2"))
        diff = time.time() - t0
        print(f"   不同会话并行: {order}（耗时 {diff:.2f}s）")
        print(f"   锁 timeout={store.lock_timeout}s 等待上限={store.lock_wait}s")
    finally:
        await clear_session(store, SID, "debug-store-2")
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
