"""数据模型

M0 只有一个过渡类型 LegacyDocument：v2 用 LangChain 的 Document 在入库、检索、应用层之间传递数据，
核心层去掉 LangChain 之后由它顶替，字段与 LangChain Document 一致（page_content / metadata），
apps/ 里按这两个字段读取的代码不用改。
M1 起由 Document / Chunk / ScoredChunk（见 docs/architecture.md 第 4 节）取代。
"""
from typing import Any, Dict

from pydantic import BaseModel, Field


class LegacyDocument(BaseModel):
    page_content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
