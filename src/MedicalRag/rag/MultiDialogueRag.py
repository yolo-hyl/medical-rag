from __future__ import annotations
import logging, re, traceback
from typing import List, Dict, Any, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableMap
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..config.models import *
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
from .RagBase import BasicRAG

# import os
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "rag-dev"

logger = logging.getLogger(__name__)

MAX_TOKEN = 2048
DEL_DIALOGUE_COUNT = 2

class MultiDialogueRag(BasicRAG):
    """多轮对话医疗RAG系统"""

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        super().__init__(config, search_config)
        self.knowledge_base = MedicalHybridKnowledgeBase(config)
        self.self_retriever: BaseRetriever = MedicalHybridRetriever(self.knowledge_base, self.search_config)
        self.llm = create_llm_client(config.llm)

        # ---- 会话存储 & 摘要存储 ----
        self._histories: Dict[str, ChatMessageHistory] = {}  # session_id -> 一个history
        self._token_meta_store = {}      # {"session_id": {"msg_len": List[int], "msg_token_len": List[int]}
        self._running_summaries: Dict[str, str] = {}  # session_id -> str
        self.max_history_tokens = 1200  # 预算阈值（需要时可暴露配置）
        # 若需要更精确的 token 估算，可引入 tiktoken 估算消息长度

        self.dialogue_rag_prompt = self._setup_dialogue_rag_prompt()
        self._setup_chain()

        logger.info("对论对话 RAG 初始化完成")

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        base = get_prompt_template("dialogue_rag")

        # 我们包装成统一消息结构：system + running_summary + history + context(doc) + human
        system_msg = base["system"]
        user_msg = base["user"]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),  # 可能包含长期摘要
                # 短期记忆（最近若干轮原文消息）
                MessagesPlaceholder(variable_name="history"),
                # 当前用户意图
                ("human", user_msg),
            ]
        )
        return prompt
    
    def _estimate_over_max_token(self, session_id: str, exist_chars: int):
        meta = self._token_meta_store.get(session_id)
        if not meta: 
            return False
        # 1) 计算平均一个字符花费多少token
        msg_len = meta["msg_len"]
        msg_token_len = meta["msg_token_len"]
        avg_tokens_per_char =  sum(msg_token_len) / sum(msg_len)
        # 2) 预测这次回答可能会生成多少token
        avg_char_len = sum(msg_len) / max(1, len(msg_len))
        predict_token = int(avg_tokens_per_char * avg_char_len)
        # 3) 判断是否可能超出最长token数量
        curr_all_token = int(predict_token + exist_chars * avg_tokens_per_char)
        if curr_all_token > MAX_TOKEN:
            # 需要删除历史信息
            return True
        else:
            return False

    # ---------- 上下文压缩（旧消息→摘要） ----------
    def _maybe_compress_history(self, session_id: str):
        """当历史过长时，把旧轮次归纳进 running_summary，并裁剪历史。"""
        hist = self._histories.get(session_id)
        
        if not hist:
            return
        
        # 估算是否超过token
        total_chars = sum(len(m.content) for m in hist.messages if hasattr(m, "content"))
        if not self._estimate_over_max_token(session_id=session_id, exist_chars=total_chars):  
            return

        # 把旧消息（配置50%）压缩
        self._get_summary(session_id)
        
    def _get_summary(self, session_id: str):
        """ 生成摘要 """
        hist = self._histories.get(session_id)
        logger.warning(f"[{session_id}] 对话过长，需要生成摘要...")
        cutoff = max(2, len(hist.messages) // DEL_DIALOGUE_COUNT)
        old_msgs = hist.messages[:cutoff]
        # 生成摘要
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("summary")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("summary")["user"])
        ])
        summary_result: AIMessage = (summarize_prompt | self.llm).invoke({"history": old_msgs})
        summary = re.sub(r"<think>.*?</think>\s*", "", summary_result.content, flags=re.DOTALL).strip()
        dur = summary_result.response_metadata.get("total_duration", 0) / 1e9
        tokens = summary_result.usage_metadata["total_tokens"]
        logger.warning(f"[{session_id}] 摘要生成完毕，耗时：{dur} s，使用tokens：{tokens}...")
        
        # 如果有摘要，那就回车换行继续加
        prev = self._running_summaries.get(session_id, "")
        merged = (prev + "\n" + summary).strip() if prev else summary
        self._running_summaries[session_id] = merged

        # 丢弃已压缩的旧消息，保留后半段
        hist.messages = hist.messages[cutoff:]

    # ---------- 历史获取器 ----------
    def _get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = ChatMessageHistory()
        return self._histories[session_id]

    @staticmethod
    def _strip_think_get_tokens(msg: AIMessage):
        text = msg.content
        # 用于衡量大概每一个字消耗多少token
        msg_len = len(msg.content)   
        msg_token_len = msg.usage_metadata["output_tokens"]
        dur = msg.response_metadata.get("total_duration", 0) / 1e9
        return {
            "msg" : re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
            "msg_len": msg_len,
            "msg_token_len": msg_token_len,
            "generate_time": dur
        }
    
    # ---------- 构建多轮 RAG 链 ----------
    def _setup_chain(self):
        
        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("rewriter")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("rewriter")["user"])
        ])
        # 填充模板 -> llm生成 -> 处理think
        rewritten_query_chain = rewrite_template | self.llm | RunnableLambda(self._strip_think_get_tokens)

        def do_retrieve(inputs: dict):
            logger.info(f"改写后的问题: {inputs['llm_rewritten_query']['msg']}")
            return self.self_retriever.invoke({"input": inputs["llm_rewritten_query"]["msg"]})

        def do_format(inputs: dict) -> str:
            documents: List[Document] = inputs["milvus_result"]["documents"]
            formatted = []
            for idx, d in enumerate(documents):
                formatted.append(f"## 文档{idx+1}：\n{d.page_content}\n")
                logger.info(f"检索到的内容: {d.page_content[0:100]}...")
            all_document_str = "".join(formatted)
            return {**inputs, "all_document_str": all_document_str, "llm_rewritten_content": inputs["llm_rewritten_query"]["msg"]}
        

        out_answer = (
            RunnableLambda(do_format) 
            | self.dialogue_rag_prompt 
            | self.llm 
            | RunnableLambda(self._strip_think_get_tokens)
        )

        core_chain = (
            RunnablePassthrough.assign(llm_rewritten_query=rewritten_query_chain).with_config(run_name="rewritten_query")
            | RunnablePassthrough.assign(milvus_result=RunnableLambda(do_retrieve)).with_config(run_name="search_documents")
            | RunnablePassthrough.assign(llm_out_result=out_answer).with_config(run_name="generate")
            | RunnableLambda(lambda x: {**x, "answer": x["llm_out_result"]["msg"]})  # for history
        )

        def _get_history_wrapper(session_id: str):
            return self._get_history(session_id)

        self.rag_chain = RunnableWithMessageHistory(
            core_chain,
            _get_history_wrapper,
            input_messages_key="original_input",
            history_messages_key="history",
            output_messages_key="answer",
        ).with_config(run_name="rag")
        
        
    def _update_tokens_metadata(
        self, 
        answer_result: dict, 
        session_id: str
    ):
        """ 更新token信息，以便估算下一次对话是否需要摘要 """
        # {"session_id": {"msg_len": List[int], "msg_token_len": List[int]}}
        if self._token_meta_store.get(session_id) is not None:
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_rewritten_query"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(answer_result["llm_rewritten_query"]["msg_token_len"])
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_out_result"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(answer_result["llm_out_result"]["msg_token_len"])
        else:
            msg_len = [
                answer_result["llm_rewritten_query"]["msg_len"],
                answer_result["llm_out_result"]["msg_len"]
            ]
            msg_token_len = [
                answer_result["llm_rewritten_query"]["msg_token_len"],
                answer_result["llm_out_result"]["msg_token_len"]
            ]
            self._token_meta_store[session_id] = {
                "msg_len": msg_len, 
                "msg_token_len": msg_token_len
            }
        

    # ---------- 对外 API：增加 session_id & 多轮 ----------
    def answer(
        self,
        query: str,
        return_document: bool = False,
        session_id: str = "default"
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        logger.info(f"[{session_id}] 问题: {query}")
        try:
            # 1) 可能压缩旧历史
            self._maybe_compress_history(session_id)

            # 2) 运行链（注意 config 中要传 session_id）
            result = self.rag_chain.invoke(
                {
                    "original_input": query,
                    "running_summary": self._running_summaries.get(session_id, ""),
                },
                config={"configurable": {"session_id": session_id}}
            )

            # 3) 更新token计数，为估计更准确的token数以生成摘要
            self._update_tokens_metadata(answer_result=result, session_id=session_id)
            
            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")
            if return_document:
                context_docs = result.get("documents", [])
                return {"answer": answer, "documents": context_docs}
            return {"answer": answer}

        except Exception as e:
            logger.exception(f"[{session_id}] RAG处理失败: {e}")
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            if return_document:
                return {"answer": error_msg, "documents": []}
            return {"answer": error_msg}

    # ---------- 更新检索配置 ----------
    def update_search_config(self, search_config: SearchRequest):
        self.self_retriever = MedicalHybridRetriever(self.knowledge_base, search_config)
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")
