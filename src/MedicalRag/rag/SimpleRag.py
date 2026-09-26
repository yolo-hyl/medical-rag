from __future__ import annotations
import logging
import re
import traceback
from typing import Dict, List, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ..config.models import LLMConfig, SearchRequest
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..prompts.templates import get_prompt_template
from .RagBase import BasicRAG
from .utils import StageTimer

logger = logging.getLogger(__name__)

# 链中 LLM 调用的 run_name，同时作为 StageTimer 的计时键
GENERATE_LLM = "generate"


class SimpleRAG(BasicRAG):
    """基础医疗RAG系统 - 使用LangChain标准组件构建"""

    def __init__(
        self,
        llm: LLMConfig,
        kb: MedicalHybridKnowledgeBase,
        search_config: SearchRequest = None
    ):
        super().__init__(llm, kb, search_config)
        # 文档检索器
        self.milvus_retriever: BaseRetriever = MedicalHybridRetriever(self.knowledge_base, self.search_config)
        # 设置prompt模板
        self.prompt = self._setup_dialogue_rag_prompt()
        # 构建RAG链
        self._setup_chain()
        
        logger.info("BasicRAG 系统初始化完成")

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
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
        def format_document_str(inputs: dict) -> str:
            documents: List[Document] = inputs["milvus_result"]["documents"]
            parts = []
            for i, d in enumerate(documents):
                parts.append(f"## 文档{i+1}：\n{d.page_content}\n")
            return "".join(parts)

        def strip_think_and_time(msg: AIMessage):
            text = msg.content
            cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            # total_duration 是 Ollama 专有字段；OpenAI 兼容接口的耗时由 StageTimer 在链外统计
            dur = msg.response_metadata.get("total_duration", 0) / 1e9
            return {"answer": cleaned.strip(), "generate_time": dur}


        # 1) 检索：从 inputs["input"] 取查询，再喂给 retriever，结果放入 inputs["documents"]
        retrieve = RunnablePassthrough.assign(
            milvus_result=self.milvus_retriever
        ).with_config(run_name="retrieve_documents")

        # 2) 格式化：把 documents 合成一个字符串，放入 inputs["all_document_str"]
        format_docs = RunnablePassthrough.assign(
            all_document_str=RunnableLambda(format_document_str)
        ).with_config(run_name="format_doc")

        # 3) 生成：prompt -> llm -> 清洗
        generate = (
            self.prompt.with_config(run_name="apply_prompt")
            | self.llm.with_config(run_name=GENERATE_LLM)
            | RunnableLambda(strip_think_and_time)
        )

        # 4) 把答案挂回到上下文中，保留 documents 等键
        self.rag_chain = (
            retrieve
            | format_docs
            | RunnablePassthrough.assign(llm=generate)
        ).with_config(run_name="rag")

        logger.info("RAG链构建完成")

    async def answer(
        self,
        query: str,
        return_document: bool = False
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        logger.info(f"处理问题: {query}")

        try:
            timer = StageTimer()   # 每个请求一个，互不干扰
            result = await self.rag_chain.ainvoke({"input": query}, config={"callbacks": [timer]})
            times = {
                "search_time": result["milvus_result"]["search_time"],
                "generation_time": timer.duration(GENERATE_LLM, result["llm"]["generate_time"]),
            }
            answer = result["llm"]["answer"]
            if return_document:
                return {"answer": answer, "documents": result["milvus_result"]["documents"], **times}
            return {"answer": answer, **times}
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            print(traceback.format_exc())
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            if return_document:
                return {
                    "answer": error_msg,
                    "documents": [],
                    "search_time": 0,
                    "generation_time": 0
                }
            return {"answer": error_msg, "search_time": 0, "generation_time": 0}

    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建链"""
        self.milvus_retriever = MedicalHybridRetriever(self.knowledge_base, search_config)  # 重新设置search配置
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")