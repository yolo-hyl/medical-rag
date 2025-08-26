from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from ..config.models import *
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
import traceback
import re
from abc import ABC, abstractmethod
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BasicRAG(ABC):
    """基础医疗RAG系统 - 使用LangChain标准组件构建"""

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        self.config = config
        
        if not search_config:
            # 如果没有传入检索配置，则初始化默认配置
            ssr1 = SingleSearchRequest(
                anns_field="summary_dense",  # 检索的字段
                metric_type="COSINE",
                search_params={"ef": 64},  # 参数
                limit=10,  # 查询数量
                expr=""  # 过滤参数
            )
            ssr2 = SingleSearchRequest(
                anns_field="text_sparse",
                metric_type="IP", 
                search_params={ "drop_ratio_search": 0.0 },
                limit=10,
                expr=""
            )
            fuse = FusionSpec(
                method="weighted",
                weights=[0.8, 0.2]
            )
            self.search_config = SearchRequest(
                query="",
                collection_name=config.milvus.collection_name,
                requests=[ssr1, ssr2],
                output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
                fuse=fuse,
                limit=10
            )
        else:
            self.search_config = search_config

        logger.info("完成检索配置初始化")

    @abstractmethod
    def _setup_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        pass

    @abstractmethod
    def _setup_chain(self):
        """构建RAG检索链"""
        pass
    
    @abstractmethod
    def answer(
        self, query: str, return_document: bool = False
    ) -> Dict[str, Union[str, List[Document]]]:
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

    def batch_answer(
        self, 
        queries: List[str],
        return_document: bool = False
    ) -> List[Dict[str, Union[str, List[Document]]]]:
        """批量回答问题"""
        results = []
        
        for query in queries:
            result = self.answer(query, return_document=return_document)
            results.append(result)
            
        return results

    @abstractmethod
    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建链"""
        pass