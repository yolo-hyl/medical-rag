"""
Redis 会话存储

多轮对话历史、摘要、token 统计、Agent 跨轮状态都存在 Redis 里，组件本身保持无状态：
同一会话用分布式锁串行执行，不同会话完全并行，多个进程 / worker 之间也能共享会话。

key 规划（统一前缀 medrag:，除锁以外都带 TTL）：
    medrag:chat:{sid}:history   List          多轮 RAG 的对话历史
    medrag:chat:{sid}:summary   String(JSON)  多轮 RAG 的 running summary
    medrag:chat:{sid}:tokens    String(JSON)  多轮 RAG 的 token 统计
    medrag:agent:{sid}:state    String(JSON)  Agent 的跨轮状态
    medrag:lock:{sid}           —             会话锁
"""
import json
import logging
from typing import Any, List, Optional

import redis.asyncio as aioredis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from ..config.models import RedisConfig

logger = logging.getLogger(__name__)


class RedisChatHistory(BaseChatMessageHistory):
    """基于 Redis List 的对话历史，每个元素是一条消息的 JSON。

    只实现异步方法，供 RunnableWithMessageHistory 在 ainvoke / astream_events 下使用。
    """

    def __init__(self, client: aioredis.Redis, key: str, ttl: int):
        self.client = client
        self.key = key
        self.ttl = ttl

    async def aget_messages(self) -> List[BaseMessage]:
        raw = await self.client.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(item) for item in raw])

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        if not messages:
            return
        payload = [json.dumps(d, ensure_ascii=False) for d in messages_to_dict(messages)]
        await self.client.rpush(self.key, *payload)
        await self.client.expire(self.key, self.ttl)

    async def atrim(self, start: int) -> None:
        """丢弃前 start 条消息，保留其后的部分（摘要压缩时使用）"""
        await self.client.ltrim(self.key, start, -1)
        await self.client.expire(self.key, self.ttl)

    async def aclear(self) -> None:
        await self.client.delete(self.key)

    # BaseChatMessageHistory 要求的同步接口：本项目只走异步路径
    @property
    def messages(self) -> List[BaseMessage]:
        raise NotImplementedError("请使用 await aget_messages()")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError("请使用 await aadd_messages()")

    def clear(self) -> None:
        raise NotImplementedError("请使用 await aclear()")


class RedisSessionStore:
    """会话存储：对话历史、任意 JSON 会话数据、会话锁"""

    def __init__(self, redis_config: RedisConfig):
        # redis.asyncio 的连接池是懒创建的，第一次使用时绑定到当前事件循环
        self.r = aioredis.Redis.from_url(redis_config.url(), decode_responses=True)
        self.ttl = redis_config.session_ttl
        self.lock_timeout = redis_config.lock_timeout
        self.lock_wait = redis_config.lock_wait

    def history(self, key: str) -> RedisChatHistory:
        return RedisChatHistory(self.r, key, self.ttl)

    async def get_json(self, key: str, default: Any = None) -> Any:
        raw = await self.r.get(key)
        return default if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any) -> None:
        await self.r.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl)

    def lock(self, session_id: str):
        """同一会话串行化。timeout 保证进程崩溃后锁能自动过期，blocking_timeout 防止无限等待。

        timeout 必须大于一轮对话的最长耗时，否则锁会在处理过程中过期，
        release 时抛 LockNotOwnedError，同一会话也可能被另一个请求插进来。
        """
        return self.r.lock(
            f"medrag:lock:{session_id}",
            timeout=self.lock_timeout,
            blocking_timeout=self.lock_wait,
        )

    async def close(self) -> None:
        await self.r.aclose()
