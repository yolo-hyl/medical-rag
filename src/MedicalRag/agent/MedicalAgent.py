from __future__ import annotations

import logging
import re
from functools import partial
from operator import add
from typing import Annotated, Any, AsyncIterator, List

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
    messages_from_dict, messages_to_dict,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..config.models import AgentConfig, LLMConfig
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.session_store import RedisSessionStore
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
from .SearchGraph import SearchGraph, SearchMessagesState
from .utils import strip_think_get_tokens

logger = logging.getLogger(__name__)


# ===================== Pydantic 输出模型 =====================

class AskDecision(BaseModel):
    ask_signal: bool = Field(default=False, description="根据已有信息及用户需要问询的内容，是否需要主动询问")
    questions: List[str] = Field(default_factory=list, description="问题列表")

class AskState(BaseModel):
    # 全局维护
    asked_messages: List[List[BaseMessage]] = Field(default_factory=list)  # 二维：每轮对话的追问消息列表

    # 当前轮维护
    ask_decision: AskDecision = Field(default_factory=AskDecision, description="当前轮的追问决策")
    curr_ask_num: int = Field(default=0, description="当前追问次数")

class RewrittenQueries(BaseModel):
    rewritten_queries: List[str] = Field(default_factory=list, description="改写后的查询列表")  # min=1,max=3，最多三个子查询

class RetrievalState(BaseModel):
    original_input: str = Field(default="", description="原始用户输入内容")
    rewritten_queries: RewrittenQueries = Field(default_factory=RewrittenQueries, description="改写后的查询列表")
    # RetrievalState 同时是检索子图的 state：多个并行 search_one 的结果经 operator.add 汇合到这里
    retrieval_outputs: Annotated[List[SearchMessagesState], add] = Field(default_factory=list, description="检索输出列表")


class SearchTask(TypedDict):
    """search_one 节点的输入：由 Send 分发的单个子查询"""
    query: str

# ===================== 顶层图状态 =====================

class MedicalAgentState(TypedDict, total=False):
    # 全局维护
    dialogue_messages: List[BaseMessage]  # 多轮全局对话消息列表，system + 原始用户输入（最近一轮检索到的文档） + 原始模型回答（不包含追问内容、历史文档等，以防撑爆上下文）
    background_info: str  # 通过追问后获取的背景信息
    summary: List[str]     # 跨轮摘要，压缩后的历史摘要（超过8条时触发压缩）
    ask_messages: AskState  # 当前需要追问的内容，包含是否需要追问及问题列表
    curr_input: str  # 当前用户输入的内容

    # 规划与检索
    retrieval_states: List[RetrievalState]  # 每轮检索的状态列表,如果没有检索行为，则存入空[]

    # 供 UI 消化的输出
    final_answer: str
    performance: List[Any]


# ===================== 节点函数 =====================

async def ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """判断是否需要向用户追问，并输出追问问题。"""
    parser = PydanticOutputParser(pydantic_object=AskDecision)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("ask_user")["system"].format(
            format_instructions=parser.get_format_instructions()
                .replace("{", "{{").replace("}", "}}")
        )),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("ask_user")["user"]),
    ])

    ask = state["ask_messages"]
    curr_ask_mess = [] if ask.curr_ask_num == 0 else ask.asked_messages[-1]
    ai = await (prompt | llm | RunnableLambda(strip_think_get_tokens)).ainvoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "asking_history": curr_ask_mess,
    })

    if ask.curr_ask_num == 0:
        ask.asked_messages.append([HumanMessage(content=state["curr_input"])])
    else:
        ask.asked_messages[-1].append(HumanMessage(content=state["curr_input"]))

    decision: AskDecision = await fixing.aparse(ai["msg"])
    ask.ask_decision = decision

    if decision.ask_signal:
        ask.asked_messages[-1].append(AIMessage(content="\n".join(decision.questions)))
    else:
        ask.asked_messages[-1].append(AIMessage(content="不需要询问任何其他信息"))

    ask.curr_ask_num += 1
    state["performance"].append(("ask", ai))
    return state


def route_ask_again(state: MedicalAgentState, max_ask_num: int) -> str:
    """
    路由判断：
    - "ask"  → 结束本轮图执行，把追问消息返回给用户，等待下一次输入
    - "pass" → 信息已充分（或达到最大追问次数），继续后续处理
    """
    ask = state["ask_messages"]
    if ask.ask_decision.ask_signal and ask.curr_ask_num < max_ask_num:
        return "ask"
    return "pass"


async def extract_background_info(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """抽取追问轮次中的关键用户背景信息。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("extract_user_info")["system"]),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("extract_user_info")["user"]),
    ])
    asked = state["ask_messages"].asked_messages
    asking_hist = asked[-1] if asked else []
    ai = await (prompt | llm | RunnableLambda(strip_think_get_tokens)).ainvoke({
        "question": asking_hist[0].content if asking_hist else state["curr_input"],
        "asking_history": asking_hist,
    })
    state["performance"].append(("extract", ai))
    state["background_info"] = ai["msg"]
    return state


async def check_update_background(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """第二轮及以后：检查用户输入是否在纠正或补充背景信息，如是则更新。"""
    tmpl = get_prompt_template("update_background")
    result = await llm.ainvoke([
        SystemMessage(content=tmpl["system"]),
        HumanMessage(content=tmpl["user"].format(
            background_info=state.get("background_info", ""),
            question=state["curr_input"],
        )),
    ])
    updated = re.sub(r"<think>.*?</think>\s*", "", result.content, flags=re.DOTALL).strip()
    state["background_info"] = updated
    return state


def route_entry(state: MedicalAgentState) -> str:
    """
    START 路由：
    - 有 background_info → 跳过追问，直接更新背景并检索
    - 无 background_info → 进入追问流程
    """
    return "check_update_background" if state.get("background_info") else "ask"


def resolve_round_question(state: MedicalAgentState) -> str:
    """本轮要回答的问题：
    - 走了追问路径（curr_ask_num > 0，answer 结束时会归零）→ 触发本次追问的那句话，
      追问中补充的信息已由 extract_background_info 写进 background_info
    - 没走追问 → 当前输入
    """
    ask = state["ask_messages"]
    if ask.curr_ask_num > 0 and ask.asked_messages:
        return ask.asked_messages[-1][0].content
    return state["curr_input"]


async def judge_split_query(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """把本轮问题改写为 1~3 个检索查询，开启本轮的 RetrievalState。"""
    question = resolve_round_question(state)
    parser = PydanticOutputParser(pydantic_object=RewrittenQueries)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("handle_query")["system"].format(
            format_instructions=parser.get_format_instructions()
                .replace("{", "{{").replace("}", "}}"),
            summary="\n".join(state["summary"]).replace("{", "{{").replace("}", "}}"),
        )),
        MessagesPlaceholder(variable_name="dialogue_messages"),
        ("user", get_prompt_template("handle_query")["user"]),
    ])
    ai = await (prompt | llm | RunnableLambda(strip_think_get_tokens)).ainvoke({
        "background_info": state["background_info"],
        "question": question,
        "dialogue_messages": state["dialogue_messages"],
    })
    patch: RewrittenQueries = await fixing.aparse(ai["msg"])
    queries = [q.strip() for q in patch.rewritten_queries if q.strip()][:3]
    patch.rewritten_queries = queries or [question]

    state["retrieval_states"].append(RetrievalState(
        original_input=question,
        rewritten_queries=patch,
    ))
    state["performance"].append(("split_query", ai))
    return state


async def retrieve(
    state: MedicalAgentState, config: RunnableConfig, retrieval_app: CompiledStateGraph
) -> MedicalAgentState:
    """以本轮的 RetrievalState 运行检索子图，用填好 retrieval_outputs 的结果替换它。

    config 需要显式传给子图：Python 3.10 下异步上下文不会自动传播，
    不传的话子图事件无法出现在父图的 astream(subgraphs=True) 里。
    """
    out = await retrieval_app.ainvoke(state["retrieval_states"][-1], config=config)
    state["retrieval_states"][-1] = RetrievalState.model_validate(out)
    return state


# ---------- 检索子图（state = RetrievalState） ----------

def fan_out_queries(state: RetrievalState) -> List[Send]:
    """每个改写后的查询分发一个 search_one，LangGraph 并发执行它们"""
    queries = state.rewritten_queries.rewritten_queries
    logger.info(f"[route] 拆分为 {len(queries)} 个子查询: {queries}")
    return [Send("search_one", {"query": q}) for q in queries]


async def search_one(task: SearchTask, search_graph: SearchGraph) -> dict:
    """单个子查询的执行节点，结果经 retrieval_outputs 的 add reducer 汇合"""
    result = await search_graph.run(search_graph.init_state(task["query"]))
    return {"retrieval_outputs": [result]}


async def gather_answer(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    汇总各子查询答案：
    - 单子查询：直接使用检索结果
    - 多子查询：用 LLM 将多份子答案综合为统一的最终回复
    更新 final_answer、dialogue_messages、summary，并结束本次追问。
    """
    question = state["retrieval_states"][-1].original_input  # 本轮问题，见 resolve_round_question
    sub_results: List[SearchMessagesState] = state["retrieval_states"][-1].retrieval_outputs

    if not sub_results:
        final_answer = "抱歉，检索未能获取到相关资料，请稍后再试。"

    elif len(sub_results) == 1:
        final_answer = (
            sub_results[0].get("final") or sub_results[0].get("summary") or ""
        ).strip()
        if not final_answer:
            final_answer = "抱歉，根据提供的资料无法回答您的问题。"

    else:
        sub_answers = []
        for i, res in enumerate(sub_results):
            ans = (res.get("final") or res.get("summary") or "").strip()
            if ans:
                sub_answers.append(f"### 子问题 {i + 1} 分析：\n{ans}")

        if not sub_answers:
            final_answer = "抱歉，根据提供的资料无法回答您的问题。"
        else:
            background = state["background_info"]
            history = "\n".join(state["summary"])
            context_prefix = ""
            if background:
                context_prefix += f"用户背景：{background}\n"
            if history:
                context_prefix += f"历史摘要：{history}\n"

            combined_context = "\n\n".join(sub_answers)
            if context_prefix:
                combined_context = context_prefix + "\n" + combined_context

            sys_tmpl = get_prompt_template("basic_rag")["system"]
            user_tmpl = get_prompt_template("basic_rag")["user"]
            synthesis_ai = await llm.ainvoke([
                SystemMessage(content=sys_tmpl),
                HumanMessage(content=user_tmpl.format(
                    all_document_str=combined_context,
                    input=question,
                )),
            ])
            final_answer = re.sub(
                r"<think>.*?</think>\s*", "", synthesis_ai.content, flags=re.DOTALL
            ).strip()

    state["final_answer"] = final_answer
    state["dialogue_messages"].append(HumanMessage(content=question))
    state["dialogue_messages"].append(AIMessage(content=final_answer))

    # 更新多轮摘要
    state["summary"].append(f"问：{question}\n答：{final_answer[:300]}")

    # 压缩：达到 8 条时把最旧的 4 条压成 1 条放回开头（开头那条可能是上次压缩的结果，会被一并滚动压缩）
    if len(state["summary"]) >= 8:
        old_entries = "\n".join(state["summary"][:4])
        compressed = await llm.ainvoke([
            SystemMessage(content=get_prompt_template("summary")["system"]),
            HumanMessage(content=old_entries + "\n" + get_prompt_template("summary")["user"]),
        ])
        compressed_text = re.sub(
            r"<think>.*?</think>\s*", "", compressed.content, flags=re.DOTALL
        ).strip()
        state["summary"] = [compressed_text] + state["summary"][4:]

    # 本次追问结束；background_info 刻意保留，供下一轮 check_update_background 使用
    state["ask_messages"].curr_ask_num = 0
    state["ask_messages"].ask_decision = AskDecision()

    return state


# ===================== 跨轮状态的 JSON 序列化 =====================
# BaseMessage / Document 不能直接 json.dumps，逐字段转换

_SEARCH_MESSAGE_FIELDS = ("main_messages", "other_messages")


def _ask_state_to_dict(ask: AskState) -> dict:
    return {
        "asked_messages": [messages_to_dict(round_) for round_ in ask.asked_messages],
        "ask_decision": ask.ask_decision.model_dump(),
        "curr_ask_num": ask.curr_ask_num,
    }


def _ask_state_from_dict(data: dict) -> AskState:
    return AskState(
        asked_messages=[messages_from_dict(round_) for round_ in data["asked_messages"]],
        ask_decision=AskDecision.model_validate(data["ask_decision"]),
        curr_ask_num=data["curr_ask_num"],
    )


def _search_output_to_dict(out: SearchMessagesState) -> dict:
    data = dict(out)
    for key in _SEARCH_MESSAGE_FIELDS:
        if key in data:
            data[key] = messages_to_dict(data[key])
    if "docs" in data:
        data["docs"] = [doc.model_dump() for doc in data["docs"]]
    return data


def _search_output_from_dict(data: dict) -> SearchMessagesState:
    out = dict(data)
    for key in _SEARCH_MESSAGE_FIELDS:
        if key in out:
            out[key] = messages_from_dict(out[key])
    if "docs" in out:
        out["docs"] = [Document(**doc) for doc in out["docs"]]
    return out


def _retrieval_state_to_dict(rs: RetrievalState) -> dict:
    return {
        "original_input": rs.original_input,
        "rewritten_queries": rs.rewritten_queries.model_dump(),
        "retrieval_outputs": [_search_output_to_dict(o) for o in rs.retrieval_outputs],
    }


def _retrieval_state_from_dict(data: dict) -> RetrievalState:
    return RetrievalState(
        original_input=data["original_input"],
        rewritten_queries=RewrittenQueries.model_validate(data["rewritten_queries"]),
        retrieval_outputs=[_search_output_from_dict(o) for o in data["retrieval_outputs"]],
    )


# ===================== MedicalAgent 主类 =====================

class MedicalAgent:
    """医疗 Agent

    本身无状态：跨轮状态（含每轮的检索全文）存在 Redis 里，一个实例服务所有会话。
    curr_input / final_answer / performance 每次执行重新开始，不持久化。
    """

    def __init__(
        self,
        llm: LLMConfig,
        agent: AgentConfig,
        kb: MedicalHybridKnowledgeBase,
        store: RedisSessionStore,
        power_model: BaseChatModel,
    ) -> None:
        self.agent_config = agent
        self.store = store
        self.power_model = power_model
        self.normal_llm = create_llm_client(llm)
        self.search_graph = SearchGraph(llm, agent, kb, power_model)
        self.build_graph()

    def build_retrieval_graph(self) -> CompiledStateGraph:
        """检索子图：START → 并行分发多个 search_one → END，结果汇合到 RetrievalState.retrieval_outputs"""
        g = StateGraph(RetrievalState)
        g.add_node("search_one", partial(search_one, search_graph=self.search_graph), input_schema=SearchTask)
        g.add_conditional_edges(START, fan_out_queries, ["search_one"])
        g.add_edge("search_one", END)
        return g.compile()

    def build_graph(self):
        g = StateGraph(MedicalAgentState)

        g.add_node("ask",                    partial(ask_judge,               llm=self.normal_llm))
        g.add_node("extract_ask_and_reply",  partial(extract_background_info, llm=self.normal_llm))
        g.add_node("check_update_background", partial(check_update_background, llm=self.normal_llm))
        g.add_node("split_query",            partial(judge_split_query,       llm=self.power_model))
        g.add_node("retrieve",               partial(retrieve,                retrieval_app=self.build_retrieval_graph()))
        g.add_node("answer",                 partial(gather_answer,           llm=self.normal_llm))

        # START → 条件路由：有背景则跳过追问
        g.add_conditional_edges(START, route_entry, {
            "ask": "ask",
            "check_update_background": "check_update_background",
        })

        g.add_conditional_edges(
            "ask",
            partial(route_ask_again, max_ask_num=self.agent_config.max_ask_num),
            {
                "ask": END,                       # 需要追问 → 结束本轮，等待用户下一次输入
                "pass": "extract_ask_and_reply",  # 信息已充分 → 继续后续处理
            },
        )
        g.add_edge("extract_ask_and_reply",   "split_query")
        g.add_edge("check_update_background", "split_query")

        # split_query 开启本轮 RetrievalState → retrieve 子图并行检索 → answer
        g.add_edge("split_query", "retrieve")
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", END)

        self.app = g.compile()

    # ---------- 会话状态：读写 Redis ----------
    @staticmethod
    def _state_key(session_id: str) -> str:
        return f"medrag:agent:{session_id}:state"

    async def _load_state(self, session_id: str) -> MedicalAgentState:
        """读取跨轮状态，没有就返回初始状态；单次执行的字段每次都重置"""
        state: MedicalAgentState = {
            # 跨轮字段
            "dialogue_messages":  [],
            "background_info":    "",
            "summary":            [],
            "ask_messages":       AskState(),
            "retrieval_states":   [],
            # 单次执行字段：每次重置
            "curr_input":         "",
            "final_answer":       "",
            "performance":        [],
        }

        saved = await self.store.get_json(self._state_key(session_id))
        if saved:
            state.update({
                "dialogue_messages": messages_from_dict(saved["dialogue_messages"]),
                "background_info": saved["background_info"],
                "summary": saved["summary"],
                "ask_messages": _ask_state_from_dict(saved["ask_messages"]),
                "retrieval_states": [_retrieval_state_from_dict(r) for r in saved["retrieval_states"]],
            })
        return state

    async def _save_state(self, session_id: str, state: MedicalAgentState) -> None:
        """只写跨轮字段"""
        await self.store.set_json(self._state_key(session_id), {
            "dialogue_messages": messages_to_dict(state["dialogue_messages"]),
            "background_info": state["background_info"],
            "summary": state["summary"],
            "ask_messages": _ask_state_to_dict(state["ask_messages"]),
            "retrieval_states": [_retrieval_state_to_dict(r) for r in state["retrieval_states"]],
        })

    @staticmethod
    def _merge_updates(state: MedicalAgentState, updates: dict) -> None:
        """合并主图节点的更新：主图字段都没有 reducer，直接覆盖"""
        state.update(updates)

    # ---------- 对外 API ----------
    async def answer(self, user_input: str, session_id: str = "default") -> MedicalAgentState:
        async with self.store.lock(session_id):
            state = await self._load_state(session_id)
            state["curr_input"] = user_input
            state = await self.app.ainvoke(state)
            await self._save_state(session_id, state)
        return state

    async def stream(self, user_input: str, session_id: str = "default") -> AsyncIterator[dict]:
        """透传 LangGraph astream(stream_mode="updates", subgraphs=True) 的 {节点名: 更新}。

        外部按节点名识别阶段：
            ask / extract_ask_and_reply / check_update_background /
            split_query / search_one（检索子图内，每个子查询完成时一条）/ retrieve / answer
        子图的 namespace 不对外暴露；只有主图的更新合并进 state。
        """
        async with self.store.lock(session_id):
            state = await self._load_state(session_id)
            state["curr_input"] = user_input
            async for namespace, chunk in self.app.astream(state, stream_mode="updates", subgraphs=True):
                if not namespace:
                    for updates in chunk.values():
                        self._merge_updates(state, updates)
                yield chunk
            await self._save_state(session_id, state)
