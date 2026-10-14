"""把 RedisSessionStore 的 JSON 列表包装成 LangChain 的对话历史

只实现异步方法，供 RunnableWithMessageHistory 在 ainvoke / astream_events 下使用。
存储格式与 v2 一致（每个元素是 messages_to_dict 的一项），已有会话可以直接读。
"""
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from ..storage.redis_session import RedisSessionStore


class RedisChatHistory(BaseChatMessageHistory):
    def __init__(self, store: RedisSessionStore, key: str):
        self.store = store
        self.key = key

    async def aget_messages(self) -> List[BaseMessage]:
        return messages_from_dict(await self.store.list_get(self.key))

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        await self.store.list_append(self.key, messages_to_dict(messages))

    async def atrim(self, start: int) -> None:
        """丢弃前 start 条消息，保留其后的部分（摘要压缩时使用）"""
        await self.store.list_trim_head(self.key, start)

    async def aclear(self) -> None:
        await self.store.delete(self.key)

    # BaseChatMessageHistory 要求的同步接口：本项目只走异步路径
    @property
    def messages(self) -> List[BaseMessage]:
        raise NotImplementedError("请使用 await aget_messages()")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError("请使用 await aadd_messages()")

    def clear(self) -> None:
        raise NotImplementedError("请使用 await aclear()")
