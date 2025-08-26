from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from ..config.models import *
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
import traceback
import re
from .RagBase import BasicRAG

logger = logging.getLogger(__name__)


class SimpleRAG(BasicRAG):
    """基础医疗RAG系统 - 使用LangChain标准组件构建"""

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        super().__init__(config, search_config)
        # 初始化向量知识库和文档检索器
        self.knowledge_base = MedicalHybridKnowledgeBase(config)
        self.self_retriever: BaseRetriever = MedicalHybridRetriever(self.knowledge_base, self.search_config)
        
        # 初始化LLM
        self.llm = create_llm_client(config.llm)
        # 设置prompt模板
        self.prompt = self._setup_prompt()
        # 构建RAG链
        self._setup_chain()
        
        logger.info("BasicRAG 系统初始化完成")

    def _setup_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        template = get_prompt_template("basic_rag")  # 获取基础RAG的提示词模板
        
        if isinstance(template, dict):
            prompt = ChatPromptTemplate.from_messages([
                ("system", template["system"]),
                ("human", template["user"]),
            ])
        else:
            # 简单的字符串模板
            prompt = ChatPromptTemplate.from_template(template)
        
        return prompt

    def _setup_chain(self):
        """构建RAG检索链"""
        
        def format_document(inputs: dict) -> str:
            """组合文本"""
            formatted_docs = []
            for index, item in enumerate(inputs["documents"]):
                formatted_docs.append(f"## 文档{index+1}：\n{item.page_content}\n")
            return "".join(formatted_docs)
        
        def strip_think(msg):
            """去掉 <think>...</think> 标签内的内容，返回纯文本"""
            if hasattr(msg, "content"):   # AIMessage
                text = msg.content
            else:                         # 已经是 str
                text = str(msg)

            # 删除所有 <think>...</think> 的部分
            cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            return cleaned.strip()
        
        # 1) 检索 -> 放入 inputs["documents"]
        retrieve = (
            RunnablePassthrough.assign(
                documents=self.self_retriever
            ).with_config(run_name="retrieve_documents")
        )
        
        #  2) 把文档格式化成一个大字符串，供 prompt 使用
        format_docs = (
            RunnablePassthrough.assign(
                all_document_str=format_document
            ).with_config(run_name="format_doc")
        )
        
        # 3) 填充prompt，大模型输出回答
        out_answer = (
            self.prompt.with_config(run_name="apply_prompt")
            | self.llm.with_config(run_name="generate")
        )
        
        
        # 4) 并联：左边生成答案，右边把命中的原始文档直接返回
        self.rag_chain = (
            retrieve
            | format_docs
            | RunnableParallel(
                answer=out_answer | RunnableLambda(strip_think),
                documents=lambda x: x["documents"],  # 保留documents字段，用于展示文档
            )
        ).with_config(run_name="rag")

        logger.info("RAG链构建完成")

    def answer(
        self, 
        query: str, 
        return_document: bool = False
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        logger.info(f"处理问题: {query}")
        
        try:
            result = self.rag_chain.invoke({"input": query})
            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")
            if return_document:
                context_docs = result.get("documents", [])
                return {
                    "answer": answer,
                    "documents": context_docs
                }
            return {"answer": answer}
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            print(traceback(e))
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            if return_document:
                return {
                    "answer": error_msg,
                    "documents": []
                }
            return {"answer": error_msg}

    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建链"""
        self.self_retriever = MedicalHybridRetriever(self.knowledge_base, search_config)  # 重新设置search配置
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")