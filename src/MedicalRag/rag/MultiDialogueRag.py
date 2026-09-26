from __future__ import annotations
import logging
import os
import re
import time
from typing import AsyncIterator, Dict, List, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from ..config.models import LLMConfig, MultiDialogueRagConfig, SearchRequest
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.session_store import RedisChatHistory, RedisSessionStore
from ..prompts.templates import get_prompt_template
from .RagBase import BasicRAG
from .utils import ESTIMATE_FUNCTION_REGISTRY, StageTimer

logger = logging.getLogger(__name__)

# 没有历史统计时的初始值：给一个很小的值，避免第一轮就误判为超长
INIT_TOKENS_PER_CHAR = 1e-5

# 链中两次 LLM 调用的 run_name，同时作为 StageTimer 的计时键
REWRITE_LLM = "rewrite_llm"
ANSWER_LLM = "answer_llm"


class MultiDialogueRag(BasicRAG):
    """多轮对话医疗RAG系统

    本身无状态：对话历史、running summary、token 统计都存在 Redis 里，
    同一会话由 store 的分布式锁串行执行，一个实例可以服务所有会话。
    """

    def __init__(
        self,
        llm: LLMConfig,
        dialogue: MultiDialogueRagConfig,
        kb: MedicalHybridKnowledgeBase,
        store: RedisSessionStore,
        search_config: SearchRequest = None
    ):
        super().__init__(llm, kb, search_config)
        self.dialogue_config = dialogue
        self.store = store

        if dialogue.smith_debug:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "rag-dev"

        self.self_retriever: BaseRetriever = MedicalHybridRetriever(self.knowledge_base, self.search_config)

        self.dialogue_rag_prompt = self._setup_dialogue_rag_prompt()
        self._setup_chain()

        logger.info("多轮对话 RAG 初始化完成")

    # ---------- 会话数据 ----------
    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"medrag:chat:{session_id}:history"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"medrag:chat:{session_id}:summary"

    @staticmethod
    def _tokens_key(session_id: str) -> str:
        return f"medrag:chat:{session_id}:tokens"

    def _get_history(self, session_id: str) -> RedisChatHistory:
        return self.store.history(self._history_key(session_id))

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        base = get_prompt_template("dialogue_rag")

        # 我们包装成统一消息结构：system + running_summary + history + context(doc) + human
        return ChatPromptTemplate.from_messages(
            [
                ("system", base["system"]),  # 可能包含长期摘要
                # 短期记忆（最近若干轮原文消息）
                MessagesPlaceholder(variable_name="history"),
                # 当前用户意图
                ("human", base["user"]),
            ]
        )

    # ---------- token 估算 ----------
    @staticmethod
    def _avg_tokens_per_char(token_meta: dict | None) -> float:
        """按本会话的历史统计现算：平均一个字符消耗多少 token"""
        if not token_meta or not sum(token_meta["msg_len"]):
            return INIT_TOKENS_PER_CHAR
        return sum(token_meta["msg_token_len"]) / sum(token_meta["msg_len"])

    def _estimate_tokens(self, text: str) -> int:
        """按配置的估计函数计算文本的 token 数；配置为 avg 时由调用方另行处理"""
        return ESTIMATE_FUNCTION_REGISTRY[self.dialogue_config.estimate_token_fun](text)

    def _avg_over_max_token(self, token_meta: dict | None, history_text: str) -> bool:
        """用平均 token 占比预测本轮总量是否会超出模型上限"""
        if not token_meta:
            return False
        avg = self._avg_tokens_per_char(token_meta)
        msg_len = token_meta["msg_len"]
        # 用历史消息的平均长度，预测这次回答可能会生成多少 token
        predict_token = avg * sum(msg_len) / max(1, len(msg_len))
        curr_all_token = predict_token + len(history_text) * avg
        cfg = self.dialogue_config
        return curr_all_token > cfg.llm_max_token * cfg.max_token_threshold

    def _need_compress(self, token_meta: dict | None, history_text: str) -> bool:
        cfg = self.dialogue_config
        if cfg.estimate_token_fun != "avg":
            try:
                return self._estimate_tokens(history_text) >= cfg.llm_max_token * cfg.max_token_threshold
            except Exception:
                logger.exception("注册的估计函数错误，回退到默认avg实现...")
        return self._avg_over_max_token(token_meta, history_text)

    # ---------- 上下文压缩（旧消息→摘要） ----------
    async def _maybe_compress_history(self, session_id: str, token_meta: dict | None) -> None:
        """当历史过长时，把旧轮次归纳进 running_summary，并裁剪历史。"""
        hist = self._get_history(session_id)
        messages = await hist.aget_messages()
        if not messages:
            return

        history_text = "\n".join(m.content for m in messages if hasattr(m, "content"))
        if not self._need_compress(token_meta, history_text):
            return

        # 把旧消息（配置50%）压缩
        await self._get_summary(session_id, hist, messages)

    async def _get_summary(
        self,
        session_id: str,
        hist: RedisChatHistory,
        messages: List[BaseMessage]
    ) -> None:
        """ 生成摘要 """
        if self.dialogue_config.console_debug:
            logger.warning(f"[{session_id}] 对话过长，需要生成摘要...")

        cutoff = max(2, len(messages) // self.dialogue_config.cut_dialogue_scale)
        old_msgs = messages[:cutoff]
        # 生成摘要
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("summary")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("summary")["user"])
        ])
        summary_result: AIMessage = await (summarize_prompt | self.llm).ainvoke({"history": old_msgs})
        summary = re.sub(r"<think>.*?</think>\s*", "", summary_result.content, flags=re.DOTALL).strip()

        if self.dialogue_config.console_debug:
            dur = summary_result.response_metadata.get("total_duration", 0) / 1e9
            tokens = summary_result.usage_metadata["total_tokens"]
            logger.warning(f"[{session_id}] 摘要生成完毕，耗时：{dur} s，使用tokens：{tokens}\n摘要文本：\n{summary}")

        # 如果有摘要，那就回车换行继续加
        prev = await self.store.get_json(self._summary_key(session_id), "")
        merged = (prev + "\n" + summary).strip() if prev else summary
        await self.store.set_json(self._summary_key(session_id), merged)

        # 丢弃已压缩的旧消息，保留后半段
        await hist.atrim(cutoff)

    @staticmethod
    def _strip_think_get_tokens(msg: AIMessage):
        text = msg.content
        # 用于衡量大概每一个字消耗多少token
        msg_len = len(text)
        msg_token_len = (msg.usage_metadata or {}).get("output_tokens", 0)
        # total_duration 是 Ollama 专有字段；OpenAI 兼容接口的耗时由 StageTimer 在链外统计
        dur = msg.response_metadata.get("total_duration", 0) / 1e9
        return {
            "msg" : re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
            "msg_len": msg_len,
            "msg_token_len": msg_token_len,
            "generate_time": dur
        }

    def _build_document_context(
        self,
        documents: List[Document],
        rewritten_query: str,
        running_summary: str,
        avg_tokens_per_char: float,
        history_msgs: List[BaseMessage]
    ) -> str:
        """按剩余 token 预算，尽可能多地把检索文档拼进上下文（纯计算，不涉及 IO）"""
        cfg = self.dialogue_config
        remain_token = cfg.llm_max_token
        his_text = "\n".join(
            getattr(m, "content", "") for m in history_msgs if hasattr(m, "content")
        )

        user_text = get_prompt_template("dialogue_rag")["user"].format(
            llm_rewritten_content=rewritten_query,
            all_document_str=""  # 先占位，后面再计算文档长度
        )
        system_text = get_prompt_template("dialogue_rag")["system"].format(running_summary=running_summary)
        all_chars = his_text + system_text + user_text

        use_estimate_fun = cfg.estimate_token_fun != "avg"
        if use_estimate_fun:
            all_token = self._estimate_tokens(all_chars)
        else:
            all_token = len(all_chars)
        remain_token -= all_token - cfg.llm_max_token * 0.01

        parts = []
        used = 0
        for idx, d in enumerate(documents):
            header = f"## 文档{idx+1}：\n"
            body = d.page_content or ""
            if use_estimate_fun:
                header_tokens = self._estimate_tokens(header)
                body_tokens = self._estimate_tokens(body)
            else:
                header_tokens = avg_tokens_per_char * len(header)
                body_tokens = avg_tokens_per_char * len(body)

            if used + header_tokens + body_tokens <= remain_token:
                parts.append(header + body + "\n")
                used += header_tokens + body_tokens
            else:
                # 放不下整篇
                if cfg.console_debug:
                    logger.warning(f"根据给定的token估计方法，预估无法完成全部文档编码，文档{idx+1}被截断，后续文档将无法被放入上下文...")
                remain = remain_token - used - header_tokens
                if remain > 0:
                    # 依据平均token估算可保留字符数
                    keep_chars = max(0, int(remain / max(0.1, avg_tokens_per_char)))
                    if keep_chars > 0:
                        parts.append(header + body[:keep_chars] + "\n...[内容已截断]\n")
                break  # 无论是否部分放入，预算已到

        return "".join(parts)


    # ---------- 构建多轮 RAG 链 ----------
    def _setup_chain(self):
        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("rewriter")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("rewriter")["user"])
        ])
        # 填充模板 -> llm生成 -> 处理think
        rewritten_query_chain = (
            rewrite_template
            | self.llm.with_config(run_name=REWRITE_LLM)
            | RunnableLambda(self._strip_think_get_tokens)
        )

        async def do_retrieve(inputs: dict):
            logger.info(f"改写后的问题: {inputs['llm_rewritten_query']['msg']}")
            return await self.self_retriever.ainvoke({"input": inputs["llm_rewritten_query"]["msg"]})

        def do_format(inputs: dict) -> dict:
            all_document_str = self._build_document_context(
                documents=inputs["milvus_result"]["documents"],
                rewritten_query=inputs["llm_rewritten_query"]["msg"],
                running_summary=inputs.get("running_summary", ""),
                avg_tokens_per_char=inputs.get("avg_tokens_per_char", INIT_TOKENS_PER_CHAR),
                history_msgs=inputs.get("history", [])
            )
            return {**inputs, "all_document_str": all_document_str, "llm_rewritten_content": inputs["llm_rewritten_query"]["msg"]}


        out_answer = (
            RunnableLambda(do_format)
            | self.dialogue_rag_prompt
            | self.llm.with_config(run_name=ANSWER_LLM)
            | RunnableLambda(self._strip_think_get_tokens)
        )

        core_chain = (
            RunnablePassthrough.assign(llm_rewritten_query=rewritten_query_chain).with_config(run_name="rewritten_query")
            | RunnablePassthrough.assign(milvus_result=RunnableLambda(do_retrieve)).with_config(run_name="search_documents")
            | RunnablePassthrough.assign(llm_out_result=out_answer).with_config(run_name="generate")
            | RunnableLambda(lambda x: {**x, "answer": x["llm_out_result"]["msg"]})  # for history
        )

        self.rag_chain = RunnableWithMessageHistory(
            core_chain,
            self._get_history,
            input_messages_key="original_input",
            history_messages_key="history",
            output_messages_key="answer",
        ).with_config(run_name="rag")

    # ---------- 一轮对话的准备与收尾（阻塞版与流式版共用） ----------
    async def _prepare(self, query: str, session_id: str) -> dict:
        """压缩历史，读取摘要和 token 统计，拼好链的输入"""
        token_meta = await self.store.get_json(self._tokens_key(session_id))
        await self._maybe_compress_history(session_id, token_meta)
        return {
            "original_input": query,
            "running_summary": await self.store.get_json(self._summary_key(session_id), ""),
            "avg_tokens_per_char": self._avg_tokens_per_char(token_meta),
            "session_id": session_id,
        }

    async def _finish(self, result: dict, session_id: str) -> None:
        """ 更新token统计，以便估算下一次对话是否需要摘要 """
        token_meta = await self.store.get_json(
            self._tokens_key(session_id), {"msg_len": [], "msg_token_len": []}
        )
        for key in ("llm_rewritten_query", "llm_out_result"):
            token_meta["msg_len"].append(result[key]["msg_len"])
            token_meta["msg_token_len"].append(result[key]["msg_token_len"])
        await self.store.set_json(self._tokens_key(session_id), token_meta)

    # ---------- 对外 API：增加 session_id & 多轮 ----------
    async def answer(
        self,
        query: str,
        return_document: bool = False,
        session_id: str = "default"
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        logger.info(f"[{session_id}] 问题: {query}")
        try:
            timer = StageTimer()   # 每个请求一个，互不干扰
            async with self.store.lock(session_id):
                inputs = await self._prepare(query, session_id)
                result = await self.rag_chain.ainvoke(
                    inputs,
                    config={"configurable": {"session_id": session_id}, "callbacks": [timer]},
                )
                await self._finish(result, session_id)

            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")
            times = {
                "search_time": result["milvus_result"]["search_time"],
                "rewriten_generate_time": timer.duration(
                    REWRITE_LLM, result["llm_rewritten_query"]["generate_time"]),
                "out_generate_time": timer.duration(
                    ANSWER_LLM, result["llm_out_result"]["generate_time"]),
            }
            if return_document:
                return {"answer": answer, "documents": result["milvus_result"]["documents"], **times}
            return {"answer": answer, **times}

        except Exception as e:
            logger.exception(f"[{session_id}] RAG处理失败: {e}")
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"
            if return_document:
                return {
                    "answer": error_msg,
                    "documents": [],
                    "search_time": -1,
                    "rewriten_generate_time": -1,
                    "out_generate_time": -1
                }
            return {
                "answer": error_msg,
                "search_time": -1,
                "rewriten_generate_time": -1,
                "out_generate_time": -1
            }

    async def stream(self, query: str, session_id: str = "default") -> AsyncIterator[dict]:
        """原样透传 LangChain astream_events(v2) 事件，由外部转换成自己的协议。

        链上的 run_name 就是识别阶段的依据：
            rewritten_query   改写查询
            search_documents  检索文档
            generate          生成回答
            rag               整条链（on_chain_end 时 data.output 是完整结果）
        两次 LLM 调用分别叫 rewrite_llm 和 answer_llm，逐 token 的输出来自它们的
        on_chat_model_stream 事件（event["data"]["chunk"].content）。
        """
        logger.info(f"[{session_id}] 流式问题: {query}")
        async with self.store.lock(session_id):
            inputs = await self._prepare(query, session_id)
            result = None
            async for event in self.rag_chain.astream_events(
                inputs, config={"configurable": {"session_id": session_id}}, version="v2"
            ):
                if event["event"] == "on_chain_end" and event["name"] == "rag":
                    result = event["data"]["output"]
                yield event
            if result:
                await self._finish(result, session_id)

    # ---------- 更新检索配置 ----------
    def update_search_config(self, search_config: SearchRequest):
        self.self_retriever = MedicalHybridRetriever(self.knowledge_base, search_config)
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")
