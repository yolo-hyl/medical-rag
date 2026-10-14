import logging
import time

from langchain_core.retrievers import BaseRetriever

from ..config.models import SearchRequest
from ..storage.milvus import MedicalHybridKnowledgeBase

logger = logging.getLogger(__name__)


class MedicalHybridRetriever(BaseRetriever):
    """把知识库检索包装成 LangChain Retriever，只有 apps/ 里的链在用"""

    knowledge_base: MedicalHybridKnowledgeBase
    search_config: SearchRequest

    def __init__(self, knowledge_base: MedicalHybridKnowledgeBase, search_config: SearchRequest):
        super().__init__(knowledge_base=knowledge_base, search_config=search_config)

    def _get_relevant_documents(self, inputs: dict, **kwargs) -> dict:
        """BaseRetriever 要求实现的同步接口：知识库只有异步实现，走 ainvoke。"""
        raise NotImplementedError("MedicalHybridRetriever 只支持异步调用，请使用 ainvoke")

    async def _aget_relevant_documents(self, inputs: dict, **kwargs) -> dict:
        """查询写进请求的副本，避免并发请求互相覆盖共享配置。"""
        req = self.search_config.model_copy(update={"query": inputs.get("input", "")})
        start_time = time.time()
        documents = await self.knowledge_base.search(req)
        return {"documents": documents, "search_time": time.time() - start_time}
