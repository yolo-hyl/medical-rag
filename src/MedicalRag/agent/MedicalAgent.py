from __future__ import annotations

import logging
import re
from functools import partial
from operator import add
from typing import Annotated, Any, List, TypedDict

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from ..config.models import AppConfig
from ..core.utils import create_llm_client
from .SearchGraph import SearchGraph, SearchMessagesState
from .utils import strip_think_get_tokens
from MedicalRag.prompts.templates import get_prompt_template

logger = logging.getLogger(__name__)


# ===================== Pydantic 输出模型 =====================

class AskMess(BaseModel):
    need_ask: bool = Field(default=False, description="根据已有信息，是否需要主动询问")
    questions: List[str] = Field(default_factory=list, description="问题列表")


class SplitQuery(BaseModel):
    need_split: bool = Field(default=False, description="是否需要拆分子查询进行检索回答")
    sub_query: List[str] = Field(
        default_factory=list,
        description="子查询列表，每个子查询互不影响，最多生成3个独立子查询"
    )
    rewrite_query: str = Field(default="", description="如果不需要拆分，改写查询为便于检索的句子")


# ===================== 顶层图状态 =====================

class MedicalAgentState(TypedDict, total=False):
    # 会话与用户画像
    dialogue_messages: List[BaseMessage]
    asking_messages: List[List[BaseMessage]]  # 二维：每轮对话的追问消息列表
    background_info: str
    ask_obj: AskMess
    multi_summary: List[str]     # 跨轮摘要，供下轮 judge_split_query 参考
    running_summary: str         # 压缩后的历史摘要（超过8条时触发压缩）
    curr_input: str

    # 规划与检索
    sub_query: SplitQuery
    rewritten_query: str         # 最终发送给检索的查询（供前端展示）
    # Annotated[..., add]：使用 operator.add 作为 reducer，
    # 让多个并行 search_one 节点的结果自动聚合到同一个列表。
    sub_query_results: Annotated[List[SearchMessagesState], add]

    # 中间产物
    max_ask_num: int
    curr_ask_num: int

    # 供 UI 消化的输出
    final_answer: str
    performance: List[Any]


# ===================== 节点函数 =====================

def ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """判断是否需要向用户追问，并输出追问问题。"""
    parser = PydanticOutputParser(pydantic_object=AskMess)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("ask_user")["system"].format(
            format_instructions=parser.get_format_instructions()
                .replace("{", "{{").replace("}", "}}")
        )),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("ask_user")["user"]),
    ])

    curr_ask_mess = [] if state["curr_ask_num"] == 0 else state["asking_messages"][-1]
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "asking_history": curr_ask_mess,
    })

    if state["curr_ask_num"] == 0:
        state["asking_messages"].append([HumanMessage(content=state["curr_input"])])
    else:
        state["asking_messages"][-1].append(HumanMessage(content=state["curr_input"]))

    patch: AskMess = fixing.parse(ai["msg"])
    state["ask_obj"] = patch

    if patch.need_ask:
        state["asking_messages"][-1].append(
            AIMessage(content="\n".join(patch.questions))
        )
    else:
        state["asking_messages"][-1].append(AIMessage(content="不需要询问任何其他信息"))

    state["performance"].append(("ask", ai))
    state["curr_ask_num"] += 1
    return state


def route_ask_again(state: MedicalAgentState) -> str:
    """
    路由判断：
    - "ask"  → 结束本轮图执行，把追问消息返回给用户，等待下一次输入
    - "pass" → 信息已充分（或达到最大追问次数），继续后续处理
    """
    ask_obj = state.get("ask_obj")
    if ask_obj is None:
        return "pass"
    if ask_obj.need_ask and state["curr_ask_num"] < state["max_ask_num"]:
        return "ask"
    return "pass"


def extract_background_info(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """抽取追问轮次中的关键用户背景信息。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("extract_user_info")["system"]),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("extract_user_info")["user"]),
    ])
    asking_hist = state["asking_messages"][-1] if state["asking_messages"] else []
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "question": asking_hist[0].content if asking_hist else state["curr_input"],
        "asking_history": asking_hist,
    })
    state["performance"].append(("extract", ai))
    state["background_info"] = ai["msg"]
    return state


def check_update_background(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """第二轮及以后：检查用户输入是否在纠正或补充背景信息，如是则更新。"""
    tmpl = get_prompt_template("update_background")
    result = llm.invoke([
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


def judge_split_query(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """判断是否需要拆分成多个子查询，并给出改写后的查询。"""
    parser = PydanticOutputParser(pydantic_object=SplitQuery)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # 合并压缩摘要与近期摘要
    running = state.get("running_summary", "")
    recent = "\n".join(state.get("multi_summary", []))
    summary_context = (running + "\n" + recent).strip() if running else recent

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("handle_query")["system"].format(
            format_instructions=parser.get_format_instructions()
                .replace("{", "{{").replace("}", "}}"),
            summary=summary_context,
        )),
        MessagesPlaceholder(variable_name="dialogue_messages"),
        ("user", get_prompt_template("handle_query")["user"]),
    ])
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "dialogue_messages": state["dialogue_messages"],
    })
    patch: SplitQuery = fixing.parse(ai["msg"])
    if len(patch.sub_query) > 3:
        patch.sub_query = patch.sub_query[:3]

    state["sub_query"] = patch
    state["performance"].append(("split_query", ai))
    return state


def route_to_subgraphs(state: MedicalAgentState) -> List[Send]:
    """
    使用 LangGraph Send API 进行并行分发。
    返回 List[Send]，LangGraph 会将每个 Send 作为独立任务并行执行 search_one 节点，
    各节点返回的 sub_query_results 通过 add reducer 自动聚合，无需手动线程管理。
    """
    sq: SplitQuery = state.get("sub_query")
    queries: List[str] = []

    if sq and sq.need_split and sq.sub_query:
        queries = sq.sub_query[:3]
    else:
        base_q = (sq.rewrite_query if sq and sq.rewrite_query else "").strip()
        queries = [base_q or state["curr_input"]]

    logger.info(f"[route] 拆分为 {len(queries)} 个子查询: {queries}")
    return [Send("search_one", {"query": q}) for q in queries]


def search_one(task_input: dict, search_graph: SearchGraph) -> dict:
    """
    单个子查询的执行节点，由 Send 调度，可并行运行多个实例。
    返回值通过 sub_query_results 的 add reducer 自动合并到主状态。
    """
    query = task_input["query"]
    init_state: SearchMessagesState = {
        "query": query,
        "main_messages": [HumanMessage(content=query)],
        "other_messages": [],
        "docs": [],
        "summary": "",
        "retry": search_graph.config.agent.max_attempts,
        "final": "",
    }
    result = search_graph.run(init_state)
    return {"sub_query_results": [result]}


def gather_answer(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    汇总各子查询答案：
    - 单子查询：直接使用检索结果
    - 多子查询：用 LLM 将多份子答案综合为统一的最终回复
    更新 final_answer、dialogue_messages、multi_summary。
    """
    state["curr_ask_num"] = 0
    sub_results: List[SearchMessagesState] = state.get("sub_query_results", [])

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
            background = state.get("background_info", "")
            running = state.get("running_summary", "")
            context_prefix = ""
            if background:
                context_prefix += f"用户背景：{background}\n"
            if running:
                context_prefix += f"历史摘要：{running}\n"

            combined_context = "\n\n".join(sub_answers)
            if context_prefix:
                combined_context = context_prefix + "\n" + combined_context

            sys_tmpl = get_prompt_template("basic_rag")["system"]
            user_tmpl = get_prompt_template("basic_rag")["user"]
            synthesis_ai = llm.invoke([
                SystemMessage(content=sys_tmpl),
                HumanMessage(content=user_tmpl.format(
                    all_document_str=combined_context,
                    input=state["curr_input"],
                )),
            ])
            final_answer = re.sub(
                r"<think>.*?</think>\s*", "", synthesis_ai.content, flags=re.DOTALL
            ).strip()

    state["final_answer"] = final_answer
    state["dialogue_messages"].append(HumanMessage(content=state["curr_input"]))
    state["dialogue_messages"].append(AIMessage(content=final_answer))

    # 更新 rewritten_query 供前端展示
    sq = state.get("sub_query")
    if sq:
        if sq.need_split and sq.sub_query:
            state["rewritten_query"] = sq.sub_query[0]
        elif sq.rewrite_query:
            state["rewritten_query"] = sq.rewrite_query
        else:
            state["rewritten_query"] = state["curr_input"]

    # 更新多轮摘要
    short_summary = f"问：{state['curr_input']}\n答：{final_answer[:300]}"
    state["multi_summary"].append(short_summary)

    # 压缩：当 multi_summary >= 8 条时，将最旧的 4 条压缩进 running_summary
    if len(state["multi_summary"]) >= 8:
        old_entries = state["multi_summary"][:4]
        summary_text = "\n".join(old_entries)
        compressed = llm.invoke([
            SystemMessage(content=get_prompt_template("summary")["system"]),
            HumanMessage(content=summary_text + "\n" + get_prompt_template("summary")["user"]),
        ])
        compressed_text = re.sub(
            r"<think>.*?</think>\s*", "", compressed.content, flags=re.DOTALL
        ).strip()
        prev = state.get("running_summary", "")
        state["running_summary"] = (prev + "\n" + compressed_text).strip() if prev else compressed_text
        state["multi_summary"] = state["multi_summary"][4:]

    # 保留最近 10 条（压缩后通常不超过此数）
    if len(state["multi_summary"]) > 10:
        state["multi_summary"] = state["multi_summary"][-10:]

    # 注意：background_info 刻意保留，不清空，供下一轮 check_update_background 使用
    state["ask_obj"] = None

    return state


# ===================== MedicalAgent 主类 =====================

class MedicalAgent:
    def __init__(self, config: AppConfig, power_model: BaseChatModel) -> None:
        self.config = config
        self.power_model = power_model
        self.normal_llm = create_llm_client(self.config.llm)
        self.search_graph = SearchGraph(self.config, power_model)
        self.build_graph()

    def build_graph(self):
        g = StateGraph(MedicalAgentState)

        g.add_node("ask",                    partial(ask_judge,               llm=self.normal_llm))
        g.add_node("extract_ask_and_reply",  partial(extract_background_info, llm=self.normal_llm))
        g.add_node("check_update_background", partial(check_update_background, llm=self.normal_llm))
        g.add_node("split_query",            partial(judge_split_query,       llm=self.power_model))
        g.add_node("search_one",             partial(search_one,              search_graph=self.search_graph))
        g.add_node("answer",                 partial(gather_answer,           llm=self.normal_llm))

        # START → 条件路由：有背景则跳过追问
        g.add_conditional_edges(START, route_entry, {
            "ask": "ask",
            "check_update_background": "check_update_background",
        })

        g.add_conditional_edges(
            "ask",
            route_ask_again,
            {
                "ask": END,                       # 需要追问 → 结束本轮，等待用户下一次输入
                "pass": "extract_ask_and_reply",  # 信息已充分 → 继续后续处理
            },
        )
        g.add_edge("extract_ask_and_reply",   "split_query")
        g.add_edge("check_update_background", "split_query")

        # Send API：split_query → 并行分发多个 search_one → answer
        g.add_conditional_edges("split_query", route_to_subgraphs, ["search_one"])
        g.add_edge("search_one", "answer")
        g.add_edge("answer", END)

        self.app = g.compile()
        self._reset_state()

    def _reset_state(self):
        self.state: MedicalAgentState = {
            "dialogue_messages":  [],
            "asking_messages":    [],
            "background_info":    "",
            "ask_obj":            None,
            "multi_summary":      [],
            "running_summary":    "",
            "rewritten_query":    "",
            "curr_input":         "",
            "sub_query":          None,
            "sub_query_results":  [],   # 每轮 invoke 前重置，避免 add reducer 跨轮累积
            "max_ask_num":        5,
            "curr_ask_num":       0,
            "final_answer":       "",
            "performance":        [],
        }

    def answer(self, user_input: str) -> MedicalAgentState:
        self.state["curr_input"] = user_input
        # 每轮 invoke 前重置子查询结果，使 add reducer 从空列表开始积累
        self.state["sub_query_results"] = []
        self.state = self.app.invoke(self.state)
        return self.state
