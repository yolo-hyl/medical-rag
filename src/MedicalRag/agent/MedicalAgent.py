from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START
from langgraph.constants import END
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
import logging
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import re
from langchain_core.documents import Document
from ..config.models import SearchRequest, AppConfig
from ..core.utils import create_llm_client
from operator import add
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from .SearchGraph import SearchMessagesState, SearchGraph
from MedicalRag.agent.tools import tencent_cloud_search
from typing import TypedDict, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from MedicalRag.prompts.templates import get_prompt_template
from langgraph.types import Command, interrupt
from .utils import strip_think_get_tokens, get_last_human
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Tuple
from functools import partial

logger = logging.getLogger(__name__)

class AskMess(BaseModel):
    need_ask: bool = Field(default=False, description="根据已有信息，是否需要主动询问")
    questions: List[str] = Field(default=False, description="问题列表")
    
class SplitQuery(BaseModel):
    need_split: bool = Field(default=False, description="根据已有信息，是否需要拆分子查询进行检索回答")
    sub_query: List[str] = Field(default=False, description="子查询列表，每个子查询互不影响，最多生成3个独立子查询")
    rewrite_query: str = Field(default="", description="如果不需要拆分子查询，则改写查询为便于检索的句子")
    
# 顶层图的状态
class MedicalAgentState(TypedDict, total=False):
    # 会话与用户画像
    dialogue_messages: List[BaseMessage]
    asking_messages: List[List[BaseMessage]]  # 每一轮对话都可能会需要主动询问，所以是一个二维数组
    background_info: str = ""
    ask_obj: AskMess = None      # 若需要向用户追问，将问题写到这里并结束本轮
    multi_summary: List[str]  # 多轮对话的摘要信息
    curr_input: str = ""
    
    # 规划与检索
    sub_query: SplitQuery = None
    sub_query_results: List[SearchMessagesState]  # 每个结果内含 query, out_state(子图输出), tag等
    
    # 中间产物
    max_ask_num: int = 0
    curr_ask_num: int = 0

    # 供UI消化的输出
    final_answer: str
    performance: List[Any]
    


def ask_judge(state: MedicalAgentState, llm: BaseChatModel):
    """ 判断是否需要询问其他信息，并输出问题 """
    parser = PydanticOutputParser(pydantic_object=AskMess)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("ask_user")["system"].format(
            format_instructions=parser.get_format_instructions().replace("{","{{").replace("}","}}")
        )),
        MessagesPlaceholder(variable_name="asking_history"),  # 带上历史
        ("human", get_prompt_template("ask_user")["user"])
    ])
    curr_ask_mess = [] if state["curr_ask_num"] == 0 else state["asking_messages"][-1]
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "asking_history": curr_ask_mess
    })
    
    if state["curr_ask_num"] == 0:
        state["asking_messages"].append([HumanMessage(content=state["curr_input"])])
    else:
        state["asking_messages"][-1].append(HumanMessage(content=state["curr_input"]))
        
    patch: AskMess = fixing.parse(ai["msg"])
    state["ask_obj"] = patch
    if patch.need_ask:
        state["asking_messages"][-1].append(AIMessage(content="\n".join([item for item in patch.questions])))
    else:
        state["asking_messages"][-1].append(AIMessage(content="不需要询问任何其他信息"))
    state["performance"].append(("ask", ai))
    state["curr_ask_num"] += 1
    return state

def route_ask_again(state: MedicalAgentState):
    return "ask" if state["ask_obj"].need_ask and state["curr_ask_num"] < state["max_ask_num"] else "pass"

def extract_background_info(state: MedicalAgentState, llm: BaseChatModel):
    """ 抽取连续追问中的关键背景信息 """
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("extract_user_info")["system"]),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("extract_user_info")["user"])
    ])
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "question": state["asking_messages"][-1][0].content,
        "asking_history":  [] if len(state["asking_messages"]) == 0 else state["asking_messages"][-1]
    })
    state["performance"].append(("extract", ai))
    state["background_info"] = ai["msg"]
    return state
    
    
def judge_split_query(state: MedicalAgentState, llm):
    """ 判断是否需要拆分子查询 """
    parser = PydanticOutputParser(pydantic_object=SplitQuery)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("handle_query")["system"].format(
            format_instructions=parser.get_format_instructions().replace("{","{{").replace("}","}}"),
            summary=state["multi_summary"]
        )),
        MessagesPlaceholder(variable_name="dialogue_messages"),
        ("user", get_prompt_template("handle_query")["user"])
    ])
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["asking_messages"][-1][0].content,
        "dialogue_messages": state["dialogue_messages"]
    })
    patch: SplitQuery = fixing.parse(ai["msg"])
    if len(patch.sub_query) > 3:  # 如果有三个以上的子查询，则截断
        patch.sub_query = patch.sub_query[:3]
        
    state["sub_query"] = patch
    state["performance"].append(("split_query", ai))
    return state



def run_parallel_subgraphs(state: SearchMessagesState, search_graph: SearchGraph) -> SearchMessagesState:
    """
    并行跑多个子查询的子图（SearchGraph），将检索结果合并回主 state。
    - 每个线程使用 deepcopy 后的 SearchGraph 副本，避免共享连接/客户端/已编译图带来的线程安全问题
    - 单个子任务失败不会中断其它任务；错误会被记录在 state["other_messages"] 里
    """
    # 组装并行任务列表
    sq: SplitQuery = state.get("sub_query")  # 如果你原状态里这个键名不同，请调整
    jobs: List[str] = []
    if sq and getattr(sq, "need_split", False):
        # 假设 sq.sub_query 是 List[str]
        jobs.extend(sq.sub_query)
    else:
        # 假设 sq.rewrite_query 是 str；没有 sq 时就拿主 query
        base_q = sq.rewrite_query
        jobs.append(base_q)

    # 工作函数：在线程内跑一个“独立副本”的 SearchGraph
    def _run_one(query: str) -> Tuple[str, SearchMessagesState]:

        # 3) 组装线程内初始状态（不要把外部 state 的可变列表直接传入，避免交叉写）
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": search_graph.config.agent.max_attempts,
            "final": ""
        }

        # 4) 执行子图（你也可以用 sg.answer(query) 返回字符串；这里我们要拿到 docs 等完整状态，所以用 run）
        out_state: SearchMessagesState = search_graph.run(init_state)

        return out_state

    # 并发执行
    max_workers = min(8, max(1, len(jobs)))
    results: List[SearchMessagesState] = []
    errors: List[Tuple[str, Exception]] = []

    # 控制台调试输出
    if getattr(search_graph.config.multi_dialogue_rag, "console_debug", False):
        logger.info(f"[parallel] 启动并行检索，任务数={len(jobs)}，max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_run_one, q): q for q in jobs}
        for fut in as_completed(future_map):
            q = future_map[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append((q, e))
                if getattr(search_graph.config.multi_dialogue_rag, "console_debug", False):
                    logger.exception(f"[parallel] 子任务失败: q={q}")
    return state

def gather_answer(state: MedicalAgentState, llm: BaseChatModel):
    
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", get_prompt_template("dialogue_rag")["system"]),
    #     MessagesPlaceholder(variable_name=state["dialogue_messages"]),
    #     ("user", get_prompt_template("dialogue_rag")["user"])
    # ])
    
    # ai = (prompt | llm).invoke({
        
    # })
    state["curr_ask_num"] = 0
    state["dialogue_messages"].append(HumanMessage(
        content=state["curr_input"]
    ))
    state["dialogue_messages"].append(AIMessage(
        content="\n".join([item["summary"] for item in state["sub_query_results"]])
    ))
    
    return state


class MedicalAgent:
    def __init__(self, config: AppConfig, power_model: BaseChatModel) -> None:
        self.config = config
        self.power_model = power_model
        self.normal_llm = create_llm_client(self.config.llm)
        self.search_graph = SearchGraph(self.config, power_model)
        self.build_graph()
        
    def build_graph(self):
        g = StateGraph(MedicalAgentState)
        ask_node = partial(
            ask_judge,
            llm=self.normal_llm
        )
        extract_node = partial(
            extract_background_info,
            llm=self.normal_llm
        )
        split_query_node = partial(
            judge_split_query,
            llm=self.power_model
        )
        run_query_node = partial(
            run_parallel_subgraphs,
            search_graph=self.search_graph
        )
        gather_answer_node = partial(
            gather_answer,
            llm=self.normal_llm
        )
        g.add_node("ask", ask_node)
        g.add_node("extract_ask_and_reply", extract_node)
        g.add_node("split_query", split_query_node)
        g.add_node("run_query", run_query_node)
        g.add_node("answer", gather_answer_node)
        
        g.set_entry_point("ask")
        g.add_conditional_edges(
            "ask",
            route_ask_again,
            {
                "ask": END,
                "pass": "extract_ask_and_reply"
            }
        )  # 如果需要完善信息则结束这场对话
        g.add_edge("extract_ask_and_reply", "split_query")
        g.add_edge("split_query", "run_query")
        g.add_edge("run_query", "answer")
        g.add_edge("answer", END)
        self.app = g.compile()
        self.state: MedicalAgentState = {
            "dialogue_messages": [],
            "asking_messages": [],
            "background_info": "",
            "ask_obj": None,      # 若需要向用户追问，将问题写到这里并结束本轮
            "multi_summary": [],  # 多轮对话的摘要信息
            "curr_input": "",
            
            # 规划与检索
            "sub_query": None,
            "sub_query_results": [],  # 每个结果内含 query, out_state(子图输出), tag等
            
            # 中间产物
            "max_ask_num": 5,
            "curr_ask_num": 0,

            # 供UI消化的输出
            "final_answer": "",
            "performance": []
        }
        
    def answer(self, user_input):
        self.state["curr_input"] = user_input
        self.state = self.app.invoke(self.state)
        return self.state
        
