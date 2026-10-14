"""Redis 会话存储

多轮对话历史、摘要、token 统计、Agent 跨轮状态都存在 Redis 里，组件本身保持无状态：
同一会话用分布式锁串行执行，不同会话完全并行，多个进程 / worker 之间也能共享会话。

key 规划（统一前缀 medrag:，除锁以外都带 TTL）：
    medrag:chat:{sid}:history   List          多轮 RAG 的对话历史（每个元素是一条消息的 JSON）
    medrag:chat:{sid}:summary   String(JSON)  多轮 RAG 的 running summary
    medrag:chat:{sid}:tokens    String(JSON)  多轮 RAG 的 token 统计
    medrag:agent:{sid}:state    String(JSON)  Agent 的跨轮状态
    medrag:lock:{sid}           —             会话锁

本模块只做存储，不依赖 LangChain；把 List 包装成 LangChain 对话历史的适配器在 apps/chat_history.py。
"""
import json
import logging
from typing import Any, List

import redis.asyncio as aioredis

from ..config.models import RedisConfig

logger = logging.getLogger(__name__)


class RedisSessionStore:
    """会话存储：任意 JSON 会话数据、JSON 列表、会话锁"""

    def __init__(self, redis_config: RedisConfig):
        # redis.asyncio 的连接池是懒创建的，第一次使用时绑定到当前事件循环
        self.r = aioredis.Redis.from_url(redis_config.url(), decode_responses=True)
        self.ttl = redis_config.session_ttl
        self.lock_timeout = redis_config.lock_timeout
        self.lock_wait = redis_config.lock_wait

    # ---------- 单个 JSON 值 ----------
    async def get_json(self, key: str, default: Any = None) -> Any:
        raw = await self.r.get(key)
        return default if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any) -> None:
        await self.r.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl)

    # ---------- JSON 列表（对话历史） ----------
    async def list_get(self, key: str) -> List[Any]:
        return [json.loads(item) for item in await self.r.lrange(key, 0, -1)]

    async def list_append(self, key: str, values: List[Any]) -> None:
        if not values:
            return
        await self.r.rpush(key, *(json.dumps(v, ensure_ascii=False) for v in values))
        await self.r.expire(key, self.ttl)

    async def list_trim_head(self, key: str, start: int) -> None:
        """丢弃前 start 个元素，保留其后的部分（摘要压缩时使用）"""
        await self.r.ltrim(key, start, -1)
        await self.r.expire(key, self.ttl)

    async def delete(self, *keys: str) -> None:
        if keys:
            await self.r.delete(*keys)

    # ---------- 会话锁 ----------
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
