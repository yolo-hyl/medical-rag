from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union

from langchain_core.prompts import ChatPromptTemplate

from ..config.models import FusionSpec, LLMConfig, SearchRequest, SingleSearchRequest
from ..llm.langchain_adapter import create_chat_model
from ..prompts import PromptRegistry
from ..storage.milvus import MedicalHybridKnowledgeBase
from ..types import LegacyDocument

logger = logging.getLogger(__name__)


def default_search_config(kb: MedicalHybridKnowledgeBase) -> SearchRequest:
    """默认检索：问题向量 + BM25 稀疏向量，加权融合（0.8 / 0.2），返回 10 条。

    稀疏向量由 Milvus 托管时，度量必须是 BM25、检索参数里不能带 drop_ratio_search；
    v2 的默认配置固定写 IP，在托管模式下会直接报错，这里按知识库的模式选择。
    """
    if kb.manage_sparse_self:
        sparse = SingleSearchRequest(anns_field="text_sparse", metric_type="IP",
                                     search_params={"drop_ratio_search": 0.0}, limit=10, expr="")
    else:
        sparse = SingleSearchRequest(anns_field="text_sparse", metric_type="BM25",
                                     search_params={}, limit=10, expr="")
    return SearchRequest(
        query="",
        collection_name=kb.collection_name,
        requests=[
            SingleSearchRequest(anns_field="summary_dense", metric_type="COSINE",
                                search_params={"ef": 64}, limit=10, expr=""),
            sparse,
        ],
        output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
        fuse=FusionSpec(method="weighted", weights=[0.8, 0.2]),
        limit=10,
    )


class BasicRAG(ABC):
    """基础医疗RAG系统 - 使用LangChain标准组件构建"""

    def __init__(
        self,
        llm: LLMConfig,
        kb: MedicalHybridKnowledgeBase,
        search_config: SearchRequest = None,
        prompts: PromptRegistry | None = None,
    ):
        # 知识库、提示词注册表由调用方创建后传入，多个 RAG / Agent 可以共用同一个实例
        self.knowledge_base = kb
        self.prompts = prompts or PromptRegistry.builtin()
        self.llm = create_chat_model(llm)

        self.search_config = search_config or default_search_config(kb)
        logger.info("完成检索配置初始化")

    @abstractmethod
    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        pass

    @abstractmethod
    def _setup_chain(self):
        """构建RAG检索链"""
        pass

    @abstractmethod
    async def answer(
        self, query: str, return_document: bool = False
    ) -> Dict[str, Union[str, List[LegacyDocument]]]:
        """
        return:
            Dict(
                {
                    "answer": "...",
                    "documents": [Document(..), Document(..)..]
                }
            )
        """
        pass

    async def batch_answer(
        self,
        queries: List[str],
        return_document: bool = False
    ) -> List[Dict[str, Union[str, List[LegacyDocument]]]]:
        """批量回答问题，多个问题并发执行"""
        return await asyncio.gather(
            *(self.answer(query, return_document=return_document) for query in queries)
        )

    @abstractmethod
    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建链"""
        pass
