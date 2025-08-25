from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel

from ..config.models import *
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template


logger = logging.getLogger(__name__)


class BasicRAG:
    """基础医疗RAG系统 - 使用LangChain标准组件构建"""

    def __init__(self, config: AppConfig):
        self.config = config
        
        # 初始化知识库
        ssr1 = SingleSearchRequest(
            anns_field="summary_dense",  # 检索的字段
            metric_type="COSINE",  # 标尺
            search_params={"ef": 64},  # 参数
            limit=10,  # 查询数量
            expr=""  # 过滤参数
        )
        ssr2 = SingleSearchRequest(
            anns_field="text_sparse",  # 检索的字段
            metric_type="IP",  # 标尺
            search_params={ "drop_ratio_search": 0.0 },  # 参数
            limit=10,  # 查询数量
            expr=""  # 过滤参数
        )
        fuse = FusionSpec(
            method="weighted",
            weights=[0.8, 0.2]
        )
        sr = SearchRequest(
            query="",
            collection_name=config.milvus.collection_name,
            requests=[ssr1, ssr2],
            output_fields=["summary", "document", "source", "source_name", "lt_doc_id", "chunk_id", "text"],
            fuse=fuse,
            limit=10
        )
    
        self.knowledge_base = MedicalHybridKnowledgeBase(config)
        self.self_retriever = MedicalHybridRetriever(self.knowledge_base, sr)
        
        # 初始化LLM
        self.llm = create_llm_client(config.llm)
        
        # 设置prompt模板
        self.prompt = self._setup_prompt()
        
        # 构建RAG链
        self._setup_chain()
        
        logger.info("BasicRAG 系统初始化完成")

    def _setup_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        template = get_prompt_template("basic_rag")
        
        if isinstance(template, dict):
            prompt = ChatPromptTemplate.from_messages([
                ("system", template["system"]),
                ("human", template["user"]),
            ])
        else:
            # 简单字符串模板
            prompt = ChatPromptTemplate.from_template(template)
        
        return prompt

    def _setup_chain(self):
        """构建RAG检索链"""
        
        # 创建文档处理链
        document_chain = create_stuff_documents_chain(
            self.llm,
            self.prompt
        )
        
        # 创建完整的RAG链
        self.rag_chain = create_retrieval_chain(self.self_retriever, document_chain)
        
        logger.info("RAG链构建完成")

    def answer(
        self, 
        query: str, 
        return_context: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """回答问题
        
        Args:
            query: 用户问题
            return_context: 是否返回检索到的上下文
            search_config: 自定义检索配置
            
        Returns:
            str 或包含答案和上下文的dict
        """
        logger.info(f"处理问题: {query}")
        
        try:
            result = self.rag_chain.invoke({"input": query})
            
            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")
            
            if return_context:
                context_docs = result.get("context", [])
                formatted_context = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "unknown"),
                        "distance": doc.metadata.get("distance", 0.0)
                    }
                    for doc in context_docs
                ]
                
                return {
                    "answer": answer,
                    "query": query,
                    "context": formatted_context,
                    "context_count": len(formatted_context)
                }
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            
            if return_context:
                return {
                    "answer": error_msg,
                    "query": query,
                    "context": [],
                    "context_count": 0
                }
            
            return error_msg

    def batch_answer(
        self, 
        queries: List[str],
        return_context: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """批量回答问题"""
        results = []
        
        for query in queries:
            result = self.answer(query, return_context=return_context)
            results.append(result)
            
        return results

    def update_search_config(self, search_config: Dict[str, Any]):
        """更新检索配置并重建链"""
        retriever = self.knowledge_base.as_retriever(search_config)
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(retriever, document_chain)
        logger.info(f"搜索配置已更新: {search_config}")

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        kb_info = self.knowledge_base.get_collection_info()
        
        return {
            "knowledge_base": kb_info,
            "llm_model": self.config.llm.model,
            "llm_provider": self.config.llm.provider,
            "embedding_model": self.config.embedding.summary_dense.model
        }