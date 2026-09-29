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
from ..core.memory import BaseMemory, SummaryBufferMemory, TokenBudget, TokenStats, messages_text
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
    dialogue_messages: List[BaseMessage]  # 短期记忆：每轮的本轮问题 + 最终回答（不含追问内容、文档），超出 token 预算时由记忆策略裁掉最旧的部分
    background_info: str  # 通过追问后获取的背景信息
    summary: List[str]     # 长期记忆：每次从 dialogue_messages 移出的旧消息压成的一段摘要
    token_stats: TokenStats  # 会话内 LLM 输出的字符数 / token 数，用于估算 token 预算
    ask_messages: AskState  # 当前需要追问的内容，包含是否需要追问及问题列表
    curr_input: str  # 当前用户输入的内容

    # 规划与检索
    retrieval_states: List[RetrievalState]  # 每轮检索的状态列表,如果没有检索行为，则存入空[]

    # 供 UI 消化的输出
    final_answer: str
    performance: List[Any]


# ===================== 节点函数 =====================

async def compress_memory(state: MedicalAgentState, memory: BaseMemory) -> MedicalAgentState:
    """每轮开始时按记忆策略收缩短期记忆：丢弃最旧的消息，新摘要追加进长期记忆。"""
    result = await memory.compress(state["dialogue_messages"], state["token_stats"])
    if result is not None:
        state["dialogue_messages"] = state["dialogue_messages"][result.drop:]
        if result.summary:
            state["summary"].append(result.summary)
    return state


async def ask_judge(state: MedicalAgentState, llm: BaseChatModel, max_ask_num: int) -> MedicalAgentState:
    """判断是否需要向用户追问，并输出追问问题。

    已追问满 max_ask_num 次时，本次输入就是对最后一次追问的回答：只记下回答，不再调 LLM。
    curr_ask_num 照常加 1，变成 max_ask_num + 1，外部据此区分"达到上限"与"信息已充分"。
    """
    ask = state["ask_messages"]
    if ask.curr_ask_num >= max_ask_num:
        ask.asked_messages[-1].append(HumanMessage(content=state["curr_input"]))
        ask.asked_messages[-1].append(AIMessage(content="已达最大追问次数，不再询问"))
        ask.ask_decision = AskDecision()
        ask.curr_ask_num += 1
        return state

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
    # ask_judge 已先把 curr_ask_num 加 1，用 <= 才能真正追问 max_ask_num 次
    if ask.ask_decision.ask_signal and ask.curr_ask_num <= max_ask_num:
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


async def _update_background(state: MedicalAgentState, llm: BaseChatModel, user_text: str) -> None:
    """让 LLM 判断 user_text 是否在纠正或补充背景信息，并写回 background_info"""
    tmpl = get_prompt_template("update_background")
    result = await llm.ainvoke([
        SystemMessage(content=tmpl["system"]),
        HumanMessage(content=tmpl["user"].format(
            background_info=state.get("background_info", ""),
            question=user_text,
        )),
    ])
    state["background_info"] = re.sub(r"<think>.*?</think>\s*", "", result.content, flags=re.DOTALL).strip()


async def check_update_background(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """第二轮及以后：检查用户输入是否在纠正或补充背景信息，如是则更新。"""
    await _update_background(state, llm, state["curr_input"])
    return state


async def follow_ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """第二轮及以后：结合对话历史、背景与本轮输入，谨慎判断是否要补充背景信息，最多追问一次。

    追问内容只记进 AskState.asked_messages（新开一轮），不进 dialogue_messages，
    免得占用全局上下文；需要追问时 curr_ask_num 置 1，由 route_entry 把下一次输入交给 follow_reply。
    """
    parser = PydanticOutputParser(pydantic_object=AskDecision)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)
    tmpl = get_prompt_template("follow_ask")

    # 直接构造消息而非 ChatPromptTemplate：摘要、历史里可能含有大括号
    ai = strip_think_get_tokens(await llm.ainvoke([
        SystemMessage(content=tmpl["system"].format(
            format_instructions=parser.get_format_instructions(),
            summary="\n".join(state["summary"]) or "（无）",
        )),
        *state["dialogue_messages"],
        HumanMessage(content=tmpl["user"].format(
            background_info=state["background_info"],
            question=state["curr_input"],
        )),
    ]))
    decision: AskDecision = await fixing.aparse(ai["msg"])

    ask = state["ask_messages"]
    if decision.ask_signal and decision.questions:
        ask.asked_messages.append([
            HumanMessage(content=state["curr_input"]),
            AIMessage(content="\n".join(decision.questions)),
        ])
        ask.ask_decision = decision
        ask.curr_ask_num = 1
    else:
        ask.ask_decision = AskDecision()

    state["token_stats"].add(ai["msg_len"], ai["msg_token_len"])
    state["performance"].append(("follow_ask", ai))
    return state


def route_follow_ask(state: MedicalAgentState) -> str:
    """follow_ask 之后：要追问 → 结束本轮等待用户回答；否则照常更新背景并检索"""
    return "ask" if state["ask_messages"].curr_ask_num > 0 else "pass"


async def merge_follow_reply(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """用户回答了 follow_ask 的追问：回答记进 AskState，再据这一轮问答补充背景信息。

    本轮要回答的仍是触发追问的那句话（见 resolve_round_question），它此前没有经过
    check_update_background，所以连同追问和回答一起交给 LLM 更新背景。
    """
    round_ = state["ask_messages"].asked_messages[-1]
    round_.append(HumanMessage(content=state["curr_input"]))
    question, asked, reply = round_[0].content, round_[1].content, round_[2].content
    await _update_background(state, llm, f"用户提问：{question}\n补充询问：{asked}\n用户回答：{reply}")
    return state


def route_entry(state: MedicalAgentState) -> str:
    """
    START 路由：
    - 无 background_info → 首轮追问流程（ask）
    - 有 background_info 且上一次 follow_ask 追问过（curr_ask_num > 0）→ 本次输入是对追问的回答（follow_reply）
    - 有 background_info → 谨慎判断是否需要补充背景（follow_ask）
    """
    if not state.get("background_info"):
        return "ask"
    if state["ask_messages"].curr_ask_num > 0:
        return "follow_reply"
    return "follow_ask"


def resolve_round_question(state: MedicalAgentState) -> str:
    """本轮要回答的问题：
    - 走了追问路径（ask 或 follow_ask，curr_ask_num > 0，answer 结束时会归零）→ 触发本次追问的那句话，
      追问中补充的信息已由 extract_background_info / merge_follow_reply 写进 background_info
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
    state["token_stats"].add(ai["msg_len"], ai["msg_token_len"])
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


async def search_one(task: SearchTask, config: RunnableConfig, search_graph: SearchGraph) -> dict:
    """单个子查询的执行节点，结果经 retrieval_outputs 的 add reducer 汇合。

    config 同 retrieve 一样要显式传下去，检索图内部的 db_search / rag / judge 等节点
    才会出现在 astream(subgraphs=True) 里。
    """
    result = await search_graph.run(search_graph.init_state(task["query"]), config=config)
    return {"retrieval_outputs": [result]}


async def gather_answer(state: MedicalAgentState, llm: BaseChatModel, budget: TokenBudget) -> MedicalAgentState:
    """
    结合多轮上下文生成本轮最终回答（不论几个子查询都统一生成一次）：
    - system：长期记忆 summary + 用户背景
    - history：短期记忆 dialogue_messages
    - 参考资料：各子查询的检索分析，按扣除上述内容后的剩余 token 预算装入
    更新 final_answer、dialogue_messages、token_stats，并结束本次追问。
    """
    rs = state["retrieval_states"][-1]
    question = rs.original_input  # 本轮问题，见 resolve_round_question
    sub_answers = []
    for res in rs.retrieval_outputs:
        ans = (res.get("final") or res.get("summary") or "").strip()
        if ans:
            sub_answers.append(f"子查询：{res.get('query', '')}\n{ans}")

    tmpl = get_prompt_template("agent_answer")
    system_text = tmpl["system"].format(
        summary="\n".join(state["summary"]) or "（无）",
        background_info=state["background_info"] or "（无）",
    )
    history = state["dialogue_messages"]
    all_document_str = budget.fit_documents(
        sub_answers,
        fixed_text=system_text + messages_text(history) + tmpl["user"].format(all_document_str="", question=question),
        avg_tokens_per_char=state["token_stats"].avg_tokens_per_char(),
        title="子问题",
    )

    # 直接构造消息而非 ChatPromptTemplate：摘要、资料里可能含有大括号
    ai = strip_think_get_tokens(await llm.ainvoke([
        SystemMessage(content=system_text),
        *history,
        HumanMessage(content=tmpl["user"].format(all_document_str=all_document_str or "（无）", question=question)),
    ]))
    final_answer = ai["msg"] or "抱歉，根据提供的资料无法回答您的问题。"

    state["final_answer"] = final_answer
    state["dialogue_messages"].append(HumanMessage(content=question))
    state["dialogue_messages"].append(AIMessage(content=final_answer))
    state["token_stats"].add(ai["msg_len"], ai["msg_token_len"])
    state["performance"].append(("answer", ai))

    # 本次追问结束；background_info 刻意保留，供下一轮 check_update_background 使用
    state["ask_messages"].curr_ask_num = 0
    state["ask_messages"].ask_decision = AskDecision()

    return state


# ===================== 跨轮状态的 JSON 序列化 =====================
# BaseMessage / Document 不能直接 json.dumps，逐字段转换

_SEARCH_MESSAGE_FIELDS = ("main_messages", "other_messages")
# _save_state 写入的跨轮字段
_SAVED_KEYS = {"dialogue_messages", "background_info", "summary", "token_stats",
               "ask_messages", "retrieval_states"}


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
    多轮记忆与 MultiDialogueRag 共用 core/memory.py：记忆策略可通过 memory 参数替换，
    默认超出 agent.memory 的 token 预算时，把最旧的对话压成摘要。
    """

    def __init__(
        self,
        llm: LLMConfig,
        agent: AgentConfig,
        kb: MedicalHybridKnowledgeBase,
        store: RedisSessionStore,
        power_model: BaseChatModel,
        memory: BaseMemory | None = None,
    ) -> None:
        self.agent_config = agent
        self.store = store
        self.power_model = power_model
        self.normal_llm = create_llm_client(llm)
        self.search_graph = SearchGraph(llm, agent, kb, power_model)
        self.budget = TokenBudget(agent.memory)
        self.memory = memory or SummaryBufferMemory(agent.memory, self.normal_llm)
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

        g.add_node("compress_memory",        partial(compress_memory,         memory=self.memory))
        g.add_node("ask",                    partial(ask_judge,               llm=self.normal_llm,
                                                     max_ask_num=self.agent_config.max_ask_num))
        g.add_node("extract_ask_and_reply",  partial(extract_background_info, llm=self.normal_llm))
        g.add_node("check_update_background", partial(check_update_background, llm=self.normal_llm))
        g.add_node("follow_ask",             partial(follow_ask_judge,        llm=self.normal_llm))
        g.add_node("follow_reply",           partial(merge_follow_reply,      llm=self.normal_llm))
        g.add_node("split_query",            partial(judge_split_query,       llm=self.power_model))
        g.add_node("retrieve",               partial(retrieve,                retrieval_app=self.build_retrieval_graph()))
        g.add_node("answer",                 partial(gather_answer,           llm=self.normal_llm, budget=self.budget))

        # START → 每轮先收缩短期记忆 → 条件路由：无背景走首轮追问，有背景走谨慎追问
        g.add_edge(START, "compress_memory")
        g.add_conditional_edges("compress_memory", route_entry, {
            "ask": "ask",
            "follow_ask": "follow_ask",
            "follow_reply": "follow_reply",
        })

        # 第二轮及以后：最多追问一次；追问 → 结束本轮等回答，下一次输入进 follow_reply
        g.add_conditional_edges("follow_ask", route_follow_ask, {
            "ask": END,
            "pass": "check_update_background",
        })
        g.add_edge("follow_reply", "split_query")

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
            "token_stats":        TokenStats(),
            "ask_messages":       AskState(),
            "retrieval_states":   [],
            # 单次执行字段：每次重置
            "curr_input":         "",
            "final_answer":       "",
            "performance":        [],
        }

        saved = await self.store.get_json(self._state_key(session_id))
        if saved and not _SAVED_KEYS <= saved.keys():
            # 旧版本存下的状态字段对不上，读了只会 KeyError，丢掉从头开始
            logger.warning(f"会话 {session_id} 的已存状态格式不兼容，已忽略并从头开始")
            saved = None
        if saved:
            state.update({
                "dialogue_messages": messages_from_dict(saved["dialogue_messages"]),
                "background_info": saved["background_info"],
                "summary": saved["summary"],
                "token_stats": TokenStats.model_validate(saved["token_stats"]),
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
            "token_stats": state["token_stats"].model_dump(),
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
            compress_memory / ask / extract_ask_and_reply / follow_ask / follow_reply / check_update_background /
            split_query / search_one（检索子图内，每个子查询完成时一条）/ retrieve / answer
        以及每个子查询内部检索图的 db_search / web_search / rag / judge / finish_success / finish_fail，
        它们的更新是完整的 SearchMessagesState，可用其中的 query 区分属于哪个子查询。
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
